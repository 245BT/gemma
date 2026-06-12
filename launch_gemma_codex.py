import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

STARTUP_TIMEOUT_SECONDS = 180
YOLO_CODEX_ARGS = [
    "--ask-for-approval",
    "never",
    "--sandbox",
    "danger-full-access",
]


@dataclass
class LauncherArgs:
    reasoning: bool
    codex_args: list[str]


def parse_launcher_args(argv):
    reasoning = False
    codex_args = []
    for arg in argv:
        if arg == "--reasoning":
            reasoning = True
        elif arg == "--yolo":
            codex_args.extend(YOLO_CODEX_ARGS)
        else:
            codex_args.append(arg)
    return LauncherArgs(reasoning=reasoning, codex_args=codex_args)


def build_codex_environment(root, reasoning, base_env=None):
    env = dict(os.environ if base_env is None else base_env)
    local_home_name = ".codex-local-reasoning" if reasoning else ".codex-local"
    env["CODEX_HOME"] = str(Path(root) / local_home_name)
    return env


def build_codex_command(root, target_dir, codex_args):
    local_codex = (
        Path(root)
        / "node_modules"
        / "@openai"
        / "codex-win32-x64"
        / "vendor"
        / "x86_64-pc-windows-msvc"
        / "bin"
        / "codex.exe"
    )
    return [str(local_codex), "--cd", str(Path(target_dir)), *codex_args]


def run(argv=None, root=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parent if root is None else Path(root)
    parsed = parse_launcher_args(argv)
    target_dir = Path(os.environ.get("GEMMA_CODEX_TARGET_DIR") or os.getcwd())

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

    command = build_codex_command(root, target_dir, parsed.codex_args)
    env = build_codex_environment(root, parsed.reasoning)
    return subprocess.call(command, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(run())
