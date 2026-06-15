import json
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from gemma_response_proxy import EMPTY_VISIBLE_RESPONSE_MESSAGE
from gemma_agent.processes import start_new_process_group_kwargs, terminate_process_tree

STARTUP_TIMEOUT_SECONDS = 180
CODEX_EXEC_TIMEOUT_ENV = "GEMMA_CODEX_EXEC_TIMEOUT"
CODEX_EXEC_TIMEOUT_RETURN_CODE = 124
CODEX_BACKGROUND_TERMINAL_MAX_TIMEOUT_MS = 3_600_000
GEMMA_AGENT_MCP_TOOL_TIMEOUT_SECONDS = 1800
CODEX_NPM_VERSION = "0.139.0"
CODEX_DRY_RUN_ENV = "GEMMA_CODEX_DRY_RUN"
NULL_FINAL_RECOVERY_PROMPT = (
    "The previous turn completed without a visible assistant message after tool execution. "
    "Do not run more tools. Provide the final answer now from the existing tool results, "
    "in concise software-engineering terms. Never return empty."
)
YOLO_CODEX_ARGS = [
    "--ask-for-approval",
    "never",
    "--sandbox",
    "danger-full-access",
]
NO_CONFIRM_YOLO_CODEX_ARGS = ["--dangerously-bypass-approvals-and-sandbox"]
SONION_NO_CONFIRM_YOLO_ENV = "GEMMA_CODEX_SONION_NO_CONFIRM_YOLO"


@dataclass
class LauncherArgs:
    reasoning: bool
    codex_args: list[str]
    yolo: bool = False


def parse_launcher_args(argv, yolo_args=None):
    reasoning = False
    yolo = False
    codex_args = []
    yolo_args = YOLO_CODEX_ARGS if yolo_args is None else yolo_args
    for arg in argv:
        if arg in {"--reasoning", "--reasoing"}:
            reasoning = True
        elif arg == "--yolo":
            yolo = True
            codex_args.extend(yolo_args)
        else:
            codex_args.append(arg)
    return LauncherArgs(reasoning=reasoning, codex_args=codex_args, yolo=yolo)


def build_codex_environment(root, reasoning, base_env=None):
    env = dict(os.environ if base_env is None else base_env)
    local_home_name = ".codex-local-reasoning" if reasoning else ".codex-local"
    root_path = Path(root)
    env["CODEX_HOME"] = str(root_path / local_home_name)
    path_key = next((key for key in env if key.upper() == "PATH"), "PATH")
    tool_bin = str(root_path / "tools" / "bin")
    node_bin = str(root_path / "node_modules" / ".bin")
    current_entries = [entry for entry in env.get(path_key, "").split(os.pathsep) if entry]
    managed_entries = [tool_bin, node_bin]
    managed_lookup = {entry.lower() for entry in managed_entries}
    env[path_key] = os.pathsep.join(
        [*managed_entries, *[entry for entry in current_entries if entry.lower() not in managed_lookup]]
    )
    return env


def build_codex_command(root, target_dir, codex_args):
    local_codex = selected_codex_binary(root)
    return [
        str(local_codex),
        "--enable",
        "unified_exec",
        "-c",
        f"background_terminal_max_timeout={CODEX_BACKGROUND_TERMINAL_MAX_TIMEOUT_MS}",
        "--cd",
        str(Path(target_dir)),
        *codex_args,
    ]


def selected_codex_binary(root):
    root_path = Path(root)
    patched_codex = root_path / "tools" / "codex-local" / "bin" / "codex.exe"
    packaged_codex = (
        root_path
        / "node_modules"
        / "@openai"
        / "codex-win32-x64"
        / "vendor"
        / "x86_64-pc-windows-msvc"
        / "bin"
        / "codex.exe"
    )
    return patched_codex if patched_codex.exists() else packaged_codex


def codex_binary_provenance(root, *, version_runner=None):
    selected = selected_codex_binary(root)
    runner = version_runner or _codex_version_runner
    version = str(runner([str(selected), "--version"])).strip()
    sha256 = ""
    if selected.is_file():
        sha256 = hashlib.sha256(selected.read_bytes()).hexdigest()
    return {
        "selected_path": str(selected),
        "version": version,
        "sha256": sha256,
        "npm_version": CODEX_NPM_VERSION,
    }


def _codex_version_runner(command):
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )
    output = (completed.stdout or completed.stderr or "").strip()
    if completed.returncode != 0:
        return output or f"codex exited with code {completed.returncode}"
    return output


def repair_codex_timeout_configs(root):
    root_path = Path(root)
    for local_home_name in (".codex-local", ".codex-local-reasoning"):
        _repair_codex_timeout_config(root_path / local_home_name / "config.toml")


def _repair_codex_timeout_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        return
    original_text = config_path.read_text(encoding="utf-8")
    lines = original_text.splitlines()
    repaired = []
    section = ""
    saw_background_timeout = False
    gemma_agent_section_open = False
    saw_gemma_agent_tool_timeout = False

    for line in lines:
        section_name = _toml_section_name(line)
        if section_name is not None:
            if gemma_agent_section_open and not saw_gemma_agent_tool_timeout:
                repaired.append(f"tool_timeout_sec = {GEMMA_AGENT_MCP_TOOL_TIMEOUT_SECONDS}")
            section = section_name
            gemma_agent_section_open = section == "mcp_servers.gemma_agent"
            if gemma_agent_section_open:
                saw_gemma_agent_tool_timeout = False

        if _is_toml_key(line, "background_terminal_max_timeout"):
            repaired.append(f"background_terminal_max_timeout = {CODEX_BACKGROUND_TERMINAL_MAX_TIMEOUT_MS}")
            saw_background_timeout = True
            continue
        if gemma_agent_section_open and _is_toml_key(line, "tool_timeout_sec"):
            repaired.append(f"tool_timeout_sec = {GEMMA_AGENT_MCP_TOOL_TIMEOUT_SECONDS}")
            saw_gemma_agent_tool_timeout = True
            continue
        repaired.append(line)

    if gemma_agent_section_open and not saw_gemma_agent_tool_timeout:
        repaired.append(f"tool_timeout_sec = {GEMMA_AGENT_MCP_TOOL_TIMEOUT_SECONDS}")
    if not saw_background_timeout:
        insert_at = _top_level_timeout_insert_index(repaired)
        repaired.insert(insert_at, f"background_terminal_max_timeout = {CODEX_BACKGROUND_TERMINAL_MAX_TIMEOUT_MS}")

    repaired_text = "\n".join(repaired) + ("\n" if original_text.endswith("\n") or repaired else "")
    if repaired_text != original_text:
        config_path.write_text(repaired_text, encoding="utf-8")


def _toml_section_name(line):
    stripped = line.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return None
    return stripped.strip("[]").strip().strip('"').strip("'")


def _is_toml_key(line, key):
    stripped = line.lstrip()
    return stripped.startswith(f"{key} ") or stripped.startswith(f"{key}=")


def _top_level_timeout_insert_index(lines):
    for index, line in enumerate(lines):
        if line.strip().startswith("["):
            return index
    return len(lines)


def toml_basic_string(value):
    return json.dumps(str(Path(value).resolve()), ensure_ascii=True)


def trust_codex_target_dir(root, target_dir, reasoning):
    local_home_name = ".codex-local-reasoning" if reasoning else ".codex-local"
    config_path = Path(root) / local_home_name / "config.toml"
    _ensure_trusted_project(config_path, target_dir)


def _ensure_trusted_project(config_path, target_dir):
    config_path = Path(config_path)
    if not config_path.exists():
        return
    original_text = config_path.read_text(encoding="utf-8")
    project_header = f"[projects.{toml_basic_string(target_dir)}]"
    if project_header in original_text:
        return
    separator = "" if not original_text or original_text.endswith("\n") else "\n"
    addition = f"{separator}\n{project_header}\ntrust_level = \"trusted\"\n"
    config_path.write_text(original_text + addition, encoding="utf-8")


def codex_exec_timeout_seconds(env=None):
    env = os.environ if env is None else env
    raw_value = env.get(CODEX_EXEC_TIMEOUT_ENV)
    if raw_value in {None, ""}:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


CODEX_OPTIONS_WITH_VALUES = {
    "-a",
    "-C",
    "-c",
    "-i",
    "-m",
    "-s",
    "--add-dir",
    "--ask-for-approval",
    "--cd",
    "--color",
    "--config",
    "--config-profile",
    "--enable",
    "--model",
    "--output-last-message",
    "--profile",
    "--sandbox",
}


def codex_first_positional_index(command):
    args = [str(part) for part in command[1:]]
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            return None
        if arg in CODEX_OPTIONS_WITH_VALUES:
            index += 2
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return index + 1
    return None


def codex_exec_command_index(command):
    index = codex_first_positional_index(command)
    if index is None or index >= len(command):
        return None
    return index if str(command[index]) == "exec" else None


def codex_command_uses_exec_timeout(command):
    return codex_exec_command_index(command) is not None


def is_json_exec_command(command):
    exec_index = codex_exec_command_index(command)
    if exec_index is None:
        return False
    return "--json" in [str(part) for part in command[exec_index + 1:]]


def run_codex_json_exec_with_recovery(command, *, cwd, env, timeout):
    first = run_captured_codex_command(command, cwd=cwd, env=env, timeout=timeout)
    if first.returncode != 0:
        write_completed_process_output(first)
        return first.returncode

    summary = summarize_json_exec_output(first.stdout)
    if not summary["needs_recovery"] or not summary["thread_id"]:
        write_completed_process_output(first)
        return first.returncode

    write_completed_process_output(first, stdout=filter_empty_visible_fallback_events(first.stdout))
    recovery_command = build_null_final_recovery_command(command, summary["thread_id"])
    second = run_captured_codex_command(recovery_command, cwd=cwd, env=env, timeout=timeout)
    write_completed_process_output(second)
    return second.returncode


def run_captured_codex_command(command, *, cwd, env, timeout):
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            sys.stdout.write(exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", "replace"))
        if exc.stderr:
            sys.stderr.write(exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", "replace"))
        return subprocess.CompletedProcess(command, CODEX_EXEC_TIMEOUT_RETURN_CODE)


def write_completed_process_output(completed, stdout=None):
    output = completed.stdout if stdout is None else stdout
    if output:
        sys.stdout.write(output)
    if completed.stderr:
        sys.stderr.write(completed.stderr)


def filter_empty_visible_fallback_events(stdout):
    lines = []
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            lines.append(line)
            continue
        if is_empty_visible_fallback_event(event):
            continue
        lines.append(line)
    return "".join(f"{line}\n" for line in lines)


def is_empty_visible_fallback_event(event):
    text = json_event_agent_message_text(event)
    if text is not None and is_empty_visible_fallback_message(text):
        return True
    if event.get("type") == "turn.completed":
        last_agent_message = event.get("last_agent_message")
        return isinstance(last_agent_message, str) and is_empty_visible_fallback_message(last_agent_message)
    return False


def summarize_json_exec_output(stdout):
    summary = {
        "thread_id": None,
        "turn_completed": False,
        "visible_agent_message": False,
        "real_visible_agent_message": False,
        "needs_recovery": False,
    }
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started" and event.get("thread_id"):
            summary["thread_id"] = event["thread_id"]
        agent_text = json_event_agent_message_text(event)
        if agent_text is not None and agent_text.strip():
            summary["visible_agent_message"] = True
            if not is_empty_visible_fallback_message(agent_text):
                summary["real_visible_agent_message"] = True
        if event.get("type") == "turn.completed":
            summary["turn_completed"] = True
            last_agent_message = event.get("last_agent_message")
            if isinstance(last_agent_message, str) and last_agent_message.strip():
                summary["visible_agent_message"] = True
                if not is_empty_visible_fallback_message(last_agent_message):
                    summary["real_visible_agent_message"] = True
    summary["needs_recovery"] = summary["turn_completed"] and not summary["real_visible_agent_message"]
    return summary


def json_event_has_visible_agent_message(event):
    text = json_event_agent_message_text(event)
    return text is not None and bool(text.strip())


def json_event_agent_message_text(event):
    if not isinstance(event, dict):
        return None
    item = event.get("item")
    if isinstance(item, dict):
        if item.get("type") == "agent_message":
            return str(item.get("text", ""))
        if item.get("type") == "message" and item.get("role") == "assistant":
            if "text" in item:
                return str(item.get("text", ""))
            content = item.get("content")
            if isinstance(content, list):
                return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            return ""
    if event.get("type") in {"agent_message", "assistant_message"}:
        return str(event.get("message", event.get("text", "")))
    return None


def is_empty_visible_fallback_message(text):
    return str(text).strip() == EMPTY_VISIBLE_RESPONSE_MESSAGE


def build_null_final_recovery_command(command, thread_id):
    exec_index = codex_exec_command_index(command)
    if exec_index is None:
        return command
    return [
        *command[: exec_index + 1],
        "resume",
        "--json",
        thread_id,
        NULL_FINAL_RECOVERY_PROMPT,
    ]


def run_codex_command(command, *, cwd, env):
    timeout = codex_exec_timeout_seconds(env) if codex_command_uses_exec_timeout(command) else None
    if is_json_exec_command(command):
        return run_codex_json_exec_with_recovery(command, cwd=cwd, env=env, timeout=timeout)
    if timeout is None:
        return subprocess.call(command, cwd=cwd, env=env)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        **start_new_process_group_kwargs(),
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process_tree(process.pid, wait_process=process.wait, timeout=1)
        return CODEX_EXEC_TIMEOUT_RETURN_CODE


def run(argv=None, root=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parent if root is None else Path(root)
    sonion_no_confirm_yolo = os.environ.get(SONION_NO_CONFIRM_YOLO_ENV) == "1"
    yolo_args = (
        NO_CONFIRM_YOLO_CODEX_ARGS
        if sonion_no_confirm_yolo
        else YOLO_CODEX_ARGS
    )
    parsed = parse_launcher_args(argv, yolo_args=yolo_args)
    target_dir = Path(os.environ.get("GEMMA_CODEX_TARGET_DIR") or os.getcwd())

    if os.environ.get(CODEX_DRY_RUN_ENV) == "1":
        command = build_codex_command(root, target_dir, parsed.codex_args)
        env = build_codex_environment(root, parsed.reasoning)
        codex_home_name = ".codex-local-reasoning" if parsed.reasoning else ".codex-local"
        print(
            json.dumps(
                {
                    "root": str(root),
                    "target_dir": str(target_dir),
                    "reasoning": parsed.reasoning,
                    "yolo": parsed.yolo,
                    "sonion_no_confirm_yolo": sonion_no_confirm_yolo,
                    "codex_home_name": codex_home_name,
                    "codex_home": env["CODEX_HOME"],
                    "codex_args": parsed.codex_args,
                    "selected_command": command,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / "start-gemma-runtime.ps1"),
        ],
        check=True,
        cwd=root,
        timeout=STARTUP_TIMEOUT_SECONDS,
    )

    repair_codex_timeout_configs(root)
    if sonion_no_confirm_yolo and parsed.yolo:
        trust_codex_target_dir(root, target_dir, parsed.reasoning)
    command = build_codex_command(root, target_dir, parsed.codex_args)
    env = build_codex_environment(root, parsed.reasoning)
    return run_codex_command(command, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(run())
