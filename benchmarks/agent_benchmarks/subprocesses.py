from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemma_agent.processes import start_new_process_group_kwargs, terminate_process_tree


OUTPUT_TAIL_CHARS = 4000


@dataclass(frozen=True)
class BoundedCompletedProcess:
    return_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


def run_bounded_subprocess(
    command: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    timeout: float,
    text: bool = True,
) -> BoundedCompletedProcess:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
        **start_new_process_group_kwargs(),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return BoundedCompletedProcess(
            return_code=process.returncode,
            stdout=_coerce_output(stdout),
            stderr=_coerce_output(stderr),
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        terminate_process_tree(process.pid, wait_process=process.wait, timeout=1)
        try:
            stdout, stderr = process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            terminate_process_tree(process.pid, wait_process=process.wait, timeout=1)
            stdout, stderr = "", ""
        return BoundedCompletedProcess(
            return_code=124,
            stdout=_coerce_output(stdout),
            stderr=_coerce_output(stderr),
            timed_out=True,
        )


def output_metadata(stdout: str, stderr: str, *, tail_chars: int = OUTPUT_TAIL_CHARS) -> dict[str, Any]:
    return {
        "stdout_tail": redacted_tail(stdout, max_chars=tail_chars),
        "stdout_length": len(stdout),
        "stdout_sha256": _sha256(stdout),
        "stderr_tail": redacted_tail(stderr, max_chars=tail_chars),
        "stderr_length": len(stderr),
        "stderr_sha256": _sha256(stderr),
    }


def write_redacted_output(path: str | Path, text: str, *, tail_chars: int = OUTPUT_TAIL_CHARS) -> None:
    Path(path).write_text(redacted_tail(text, max_chars=tail_chars), encoding="utf-8")


def redacted_tail(text: str, *, max_chars: int = OUTPUT_TAIL_CHARS) -> str:
    tail = text[-max(0, max_chars) :] if len(text) > max_chars else text
    return redact_sensitive_text(tail)


def redact_sensitive_text(text: str) -> str:
    redacted = str(text or "")
    redacted = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        redacted,
        flags=re.DOTALL,
    )
    redacted = re.sub(
        r"(?i)\b(password|passwd|token|api[_-]?key|bearer|openai_api_key)\s*[:=]\s*\S+",
        lambda match: f"{match.group(1)}=[REDACTED]",
        redacted,
    )
    redacted = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "sk-[REDACTED]", redacted)
    return redacted


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()


def _coerce_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
