from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import uuid
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .processes import start_new_process_group_kwargs, terminate_process_tree


@dataclass(frozen=True)
class TerminalCommandResult:
    command: list[str]
    cwd: str | None
    ok: bool
    status: str
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    stdout_length: int
    stderr_length: int
    stdout_sha256: str
    stderr_sha256: str
    elapsed_ms: float
    recovery: dict[str, str] = field(default_factory=dict)
    job_id: str | None = None
    process_id: int | None = None


@dataclass
class _TerminalJob:
    job_id: str
    command: list[str]
    cwd: str | None
    process: subprocess.Popen[str]
    stdout_parts: _BoundedTextBuffer
    stderr_parts: _BoundedTextBuffer
    readers: list[threading.Thread]
    state: dict[str, float]
    state_lock: threading.Lock
    started: float
    timeout_sec: float
    idle_timeout_sec: float


class TerminalCommandRunner:
    HARD_TIMEOUT_CAP_SEC = 1800.0

    def __init__(
        self,
        *,
        default_timeout_sec: float = 1800,
        default_idle_timeout_sec: float = 120,
        default_background_after_sec: float = 120,
        poll_interval_sec: float = 0.25,
        max_tail_chars: int = 4000,
    ) -> None:
        self.default_timeout_sec = _bounded_timeout(default_timeout_sec, self.HARD_TIMEOUT_CAP_SEC)
        self.default_idle_timeout_sec = _positive_float(default_idle_timeout_sec, 120)
        self.default_background_after_sec = _positive_float(default_background_after_sec, 120)
        self.poll_interval_sec = _positive_float(poll_interval_sec, 0.25)
        self.max_tail_chars = max(200, int(max_tail_chars))
        self._jobs: dict[str, _TerminalJob] = {}
        self._jobs_lock = threading.Lock()

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout_sec: float | None = None,
        idle_timeout_sec: float | None = None,
        background_after_sec: float | None = None,
    ) -> TerminalCommandResult:
        command_list = _normalize_terminal_command([str(part) for part in command])
        if not command_list:
            raise ValueError("command must include at least one argument")
        timeout = _bounded_timeout(timeout_sec, self.default_timeout_sec)
        idle_timeout = _positive_float(idle_timeout_sec, self.default_idle_timeout_sec)
        background_after = _positive_float(background_after_sec, self.default_background_after_sec)
        background_after = min(background_after, timeout)
        cwd_text = str(cwd) if cwd is not None else None
        started = time.perf_counter()
        stdout_parts = _BoundedTextBuffer(self.max_tail_chars)
        stderr_parts = _BoundedTextBuffer(self.max_tail_chars)
        state = {"last_output_at": started}
        state_lock = threading.Lock()

        process = subprocess.Popen(
            command_list,
            cwd=cwd_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **start_new_process_group_kwargs(),
        )
        readers = [
            threading.Thread(
                target=_read_stream,
                args=(process.stdout, stdout_parts, state, state_lock),
                daemon=True,
            ),
            threading.Thread(
                target=_read_stream,
                args=(process.stderr, stderr_parts, state, state_lock),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        status = "completed"
        recovery: dict[str, str] = {}
        exit_code: int | None = None
        job_id: str | None = None
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                status = "completed" if exit_code == 0 else "failed"
                break
            now = time.perf_counter()
            if now - started >= timeout:
                status = "timeout"
                recovery = _recovery("hard_timeout", timeout)
                _terminate_process(process)
                exit_code = None
                break
            with state_lock:
                idle_for = now - state["last_output_at"]
            if idle_for >= idle_timeout:
                status = "stalled"
                recovery = _recovery("idle_timeout", idle_timeout)
                job_id = self._store_job(
                    command_list,
                    cwd_text,
                    process,
                    stdout_parts,
                    stderr_parts,
                    readers,
                    state,
                    state_lock,
                    started,
                    timeout,
                    idle_timeout,
                )
                exit_code = None
                break
            if now - started >= background_after:
                status = "running"
                recovery = _recovery("backgrounded", background_after)
                job_id = self._store_job(
                    command_list,
                    cwd_text,
                    process,
                    stdout_parts,
                    stderr_parts,
                    readers,
                    state,
                    state_lock,
                    started,
                    timeout,
                    idle_timeout,
                )
                exit_code = None
                break
            time.sleep(self.poll_interval_sec)

        if job_id is None:
            for reader in readers:
                reader.join(timeout=0.5)

        stdout, stdout_length, stdout_sha256 = stdout_parts.snapshot()
        stderr, stderr_length, stderr_sha256 = stderr_parts.snapshot()
        return TerminalCommandResult(
            command=command_list,
            cwd=cwd_text,
            ok=status in {"completed", "running"},
            status=status,
            exit_code=exit_code,
            stdout_tail=_tail(stdout, self.max_tail_chars),
            stderr_tail=_tail(stderr, self.max_tail_chars),
            stdout_length=stdout_length,
            stderr_length=stderr_length,
            stdout_sha256=stdout_sha256,
            stderr_sha256=stderr_sha256,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            recovery=recovery,
            job_id=job_id,
            process_id=process.pid,
        )

    def job_status(self, job_id: str) -> TerminalCommandResult:
        job = self._get_job(job_id)
        exit_code = job.process.poll()
        now = time.perf_counter()
        status = "running"
        recovery: dict[str, str] = {}
        remove = False
        if exit_code is not None:
            status = "completed" if exit_code == 0 else "failed"
            remove = True
        elif now - job.started >= job.timeout_sec:
            status = "timeout"
            recovery = _recovery("hard_timeout", job.timeout_sec)
            _terminate_process(job.process)
            exit_code = None
            remove = True
        else:
            with job.state_lock:
                idle_for = now - job.state["last_output_at"]
            if idle_for >= job.idle_timeout_sec:
                status = "stalled"
                recovery = _recovery("idle_timeout", job.idle_timeout_sec)

        if remove:
            self._remove_job(job_id)
            for reader in job.readers:
                reader.join(timeout=0.5)
        return self._result_from_job(job, status=status, exit_code=exit_code, recovery=recovery)

    def kill_job(self, job_id: str | None, *, reason: str = "agent requested termination") -> TerminalCommandResult:
        if not job_id:
            raise ValueError("job_id is required")
        job = self._get_job(job_id)
        _terminate_process(job.process)
        self._remove_job(job_id)
        for reader in job.readers:
            reader.join(timeout=0.5)
        return self._result_from_job(
            job,
            status="killed",
            exit_code=None,
            recovery=_recovery(reason or "agent requested termination", 0),
        )

    def _store_job(
        self,
        command: list[str],
        cwd: str | None,
        process: subprocess.Popen[str],
        stdout_parts: _BoundedTextBuffer,
        stderr_parts: _BoundedTextBuffer,
        readers: list[threading.Thread],
        state: dict[str, float],
        state_lock: threading.Lock,
        started: float,
        timeout_sec: float,
        idle_timeout_sec: float,
    ) -> str:
        job_id = uuid.uuid4().hex
        job = _TerminalJob(
            job_id=job_id,
            command=command,
            cwd=cwd,
            process=process,
            stdout_parts=stdout_parts,
            stderr_parts=stderr_parts,
            readers=readers,
            state=state,
            state_lock=state_lock,
            started=started,
            timeout_sec=timeout_sec,
            idle_timeout_sec=idle_timeout_sec,
        )
        with self._jobs_lock:
            self._jobs[job_id] = job
        self._start_job_watchdog(job)
        return job_id

    def _start_job_watchdog(self, job: _TerminalJob) -> None:
        threading.Thread(
            target=self._watch_job_timeout,
            args=(job,),
            daemon=True,
        ).start()

    def _watch_job_timeout(self, job: _TerminalJob) -> None:
        deadline = job.started + job.timeout_sec
        while True:
            with self._jobs_lock:
                if self._jobs.get(job.job_id) is not job:
                    return
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 1.0))
        with self._jobs_lock:
            if self._jobs.get(job.job_id) is not job:
                return
        if job.process.poll() is None:
            _terminate_process(job.process)
        self._remove_job(job.job_id)
        for reader in job.readers:
            reader.join(timeout=0.5)

    def _get_job(self, job_id: str) -> _TerminalJob:
        with self._jobs_lock:
            job = self._jobs.get(str(job_id))
        if job is None:
            raise ValueError(f"unknown terminal job {job_id!r}")
        return job

    def _remove_job(self, job_id: str) -> None:
        with self._jobs_lock:
            self._jobs.pop(str(job_id), None)

    def _result_from_job(
        self,
        job: _TerminalJob,
        *,
        status: str,
        exit_code: int | None,
        recovery: dict[str, str],
    ) -> TerminalCommandResult:
        stdout, stdout_length, stdout_sha256 = job.stdout_parts.snapshot()
        stderr, stderr_length, stderr_sha256 = job.stderr_parts.snapshot()
        live_job_id = job.job_id if status in {"running", "stalled"} else None
        return TerminalCommandResult(
            command=job.command,
            cwd=job.cwd,
            ok=status in {"completed", "running"},
            status=status,
            exit_code=exit_code,
            stdout_tail=_tail(stdout, self.max_tail_chars),
            stderr_tail=_tail(stderr, self.max_tail_chars),
            stdout_length=stdout_length,
            stderr_length=stderr_length,
            stdout_sha256=stdout_sha256,
            stderr_sha256=stderr_sha256,
            elapsed_ms=round((time.perf_counter() - job.started) * 1000, 3),
            recovery=recovery,
            job_id=live_job_id,
            process_id=job.process.pid,
        )


class _BoundedTextBuffer:
    def __init__(self, max_chars: int) -> None:
        self.max_chars = max(1, int(max_chars))
        self._chunks: deque[str] = deque()
        self._retained_length = 0
        self._length = 0
        self._sha256 = hashlib.sha256()
        self._lock = threading.Lock()

    @property
    def retained_length(self) -> int:
        with self._lock:
            return self._retained_length

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        with self._lock:
            self._length += len(chunk)
            self._sha256.update(chunk.encode("utf-8"))
            if len(chunk) >= self.max_chars:
                self._chunks.clear()
                tail = chunk[-self.max_chars :]
                self._chunks.append(tail)
                self._retained_length = len(tail)
                return
            self._chunks.append(chunk)
            self._retained_length += len(chunk)
            while self._retained_length > self.max_chars and self._chunks:
                excess = self._retained_length - self.max_chars
                first = self._chunks[0]
                if len(first) <= excess:
                    self._chunks.popleft()
                    self._retained_length -= len(first)
                    continue
                self._chunks[0] = first[excess:]
                self._retained_length -= excess

    def snapshot(self) -> tuple[str, int, str]:
        with self._lock:
            return "".join(self._chunks), self._length, self._sha256.hexdigest()


def _read_stream(stream, sink: _BoundedTextBuffer, state: dict[str, float], state_lock: threading.Lock) -> None:
    if stream is None:
        return
    try:
        for chunk in iter(lambda: stream.read(1), ""):
            sink.append(chunk)
            with state_lock:
                state["last_output_at"] = time.perf_counter()
    finally:
        stream.close()


def _normalize_terminal_command(command: list[str]) -> list[str]:
    command = _normalize_npx_ctx7_args(command)
    return _resolve_windows_npx_command(command)


def _normalize_npx_ctx7_args(command: list[str]) -> list[str]:
    if len(command) < 4 or _command_name(command[0]) != "npx":
        return command
    for index in range(1, len(command) - 1):
        if not _is_ctx7_npx_package_token(command[index]):
            continue
        if index > 1 and command[index - 1].lower() in {"--package", "-p"}:
            continue
        if _command_name(command[index + 1]) == "ctx7":
            return [*command[: index + 1], *command[index + 2 :]]
    return command


def _resolve_windows_npx_command(command: list[str]) -> list[str]:
    if os.name != "nt" or not command or _command_name(command[0]) != "npx":
        return command
    executable = shutil.which("npx.cmd") or shutil.which("npx.exe")
    if executable is None:
        return command
    return [executable, *command[1:]]


def _command_name(value: str) -> str:
    name = os.path.basename(value).lower()
    for suffix in (".cmd", ".exe", ".bat"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _is_ctx7_npx_package_token(value: str) -> bool:
    package = value.lower()
    return package == "ctx7" or package.startswith("ctx7@")


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    terminate_process_tree(process.pid, wait_process=process.wait, timeout=1)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def _recovery(reason: str, seconds: float) -> dict[str, str]:
    return {
        "reason": reason,
        "threshold_sec": f"{seconds:.3f}",
        "next_action": (
            "Stop waiting blindly; inspect partial output and current state. Do not repeat "
            "the same command or only shrink a timed-out range. Set an explicit "
            "timeout_sec/timeout_ms for expected-long commands, and for subnet probes use "
            "bounded parallel or vectorized checks with per-probe timeouts. Check active "
            "install or package-manager processes before retrying, and use Context7 or "
            "official docs before guessing a different command."
        ),
    }


def _positive_float(value: float | None, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0 else float(default)


def _bounded_timeout(value: float | None, default: float) -> float:
    return min(_positive_float(value, default), TerminalCommandRunner.HARD_TIMEOUT_CAP_SEC)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]
