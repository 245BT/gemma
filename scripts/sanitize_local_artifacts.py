from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


LOCAL_CODEX_HOMES = (".codex-local", ".codex-local-reasoning")
SENSITIVE_FILE_PATTERNS = (
    "history.jsonl",
    "sessions/**/*.jsonl",
    ".sandbox/**/*",
    ".sandbox-secrets/**/*",
    "logs_*.sqlite*",
    "state_*.sqlite*",
    "memories_*.sqlite*",
    "goals_*.sqlite*",
    "sandbox.*.log",
)


def find_artifact_paths(root: str | Path) -> list[Path]:
    root_path = Path(root).resolve()
    paths: dict[Path, None] = {}
    for home_name in LOCAL_CODEX_HOMES:
        home = root_path / home_name
        if not home.exists():
            continue
        for pattern in SENSITIVE_FILE_PATTERNS:
            for path in home.glob(pattern):
                resolved = path.resolve()
                if path.is_file() and _is_relative_to(resolved, home):
                    paths[resolved] = None
    return sorted(paths)


def sanitize(root: str | Path, *, apply: bool = False) -> dict[str, object]:
    root_path = Path(root).resolve()
    paths = find_artifact_paths(root_path)
    removed: list[str] = []
    failed: list[dict[str, str]] = []
    if apply:
        for path in paths:
            try:
                path.unlink()
                removed.append(str(path))
            except OSError as exc:
                failed.append({"path": str(path), "error": str(exc)})
        _prune_empty_session_dirs(root_path)

    return {
        "root": str(root_path),
        "apply": apply,
        "candidate_count": len(paths),
        "removed_count": len(removed),
        "failed_count": len(failed),
        "candidates": [str(path) for path in paths],
        "removed": removed,
        "failed": failed,
    }


def _prune_empty_session_dirs(root: Path) -> None:
    for home_name in LOCAL_CODEX_HOMES:
        home = root / home_name
        for directory_name in ("sessions", ".sandbox", ".sandbox-secrets"):
            directory = home / directory_name
            if directory.exists():
                _remove_empty_dirs_bottom_up(directory)


def _remove_empty_dirs_bottom_up(root: Path) -> None:
    dirs: Iterable[Path] = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in dirs:
        try:
            path.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Remove sensitive generated Codex history/session/log artifacts from local homes."
    )
    parser.add_argument("--root", default=".", help="Project root containing .codex-local homes.")
    parser.add_argument("--apply", action="store_true", help="Delete matched artifacts. Default is dry-run.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = sanitize(args.root, apply=args.apply)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
