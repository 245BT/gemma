from __future__ import annotations

import os
import signal
import subprocess


def start_new_process_group_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def prepare_current_process_group() -> None:
    if os.name == "nt":
        return
    try:
        os.setsid()
    except OSError:
        pass


def terminate_process_tree(pid: int, *, wait_process=None, timeout: float = 1) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(0.1, timeout),
                check=False,
            )
        except Exception:
            pass
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    if wait_process is not None:
        try:
            wait_process(timeout=timeout)
        except Exception:
            pass
