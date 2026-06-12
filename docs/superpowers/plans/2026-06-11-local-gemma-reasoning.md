# Local Gemma Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in LangGraph + DSPy planner/verifier reasoning proxy for local Gemma Codex, enabled only by `operator --reasoning`, `son --reasoning`, or `sonion --reasoning`.

**Architecture:** Keep the default local Codex path on `.codex-local` and `8081`. Add a second project-local Codex home `.codex-local-reasoning` that points to a reasoning proxy on `8082`. The reasoning proxy wraps `/v1/responses` with a LangGraph state graph and DSPy planner/verifier modules, then calls the existing direct proxy on `8081`.

**Tech Stack:** Python 3.14, `unittest`, local `.venv`, LangGraph, DSPy, llama.cpp OpenAI-compatible Responses API, project-local Codex 0.139.0.

---

## File Structure

- `launch_gemma_codex.py`: parses local launcher flags, strips `--reasoning`, chooses direct or reasoning `CODEX_HOME`, and invokes the vendored Codex binary.
- `gemma-codex.cmd`: thin batch wrapper that calls `launch_gemma_codex.py`.
- `setup_local_codex.py`: generates both `.codex-local` and `.codex-local-reasoning` configs and updated launchers.
- `start-gemma-runtime.ps1`: starts llama.cpp on `8080`, direct proxy on `8081`, and reasoning proxy on `8082` when needed.
- `gemma_reasoning/`: focused package for request extraction, upstream calls, DSPy programs, and LangGraph graph assembly.
- `gemma_reasoning_proxy.py`: HTTP proxy serving the reasoning Responses API on `8082`.
- `tests/test_launch_gemma_codex.py`: launcher argument and local-only tests.
- `tests/test_setup_local_codex.py`: generated config/launcher tests.
- `tests/test_gemma_reasoning.py`: graph and response shaping tests with fake model calls.
- `tests/test_gemma_reasoning_proxy.py`: reasoning proxy JSON/SSE behavior tests.

### Task 1: Launcher Mode Selection

**Files:**
- Create: `tests/test_launch_gemma_codex.py`
- Create: `launch_gemma_codex.py`
- Modify: `gemma-codex.cmd`
- Modify: `setup_local_codex.py`

- [ ] **Step 1: Write failing launcher tests**

```python
import os
import unittest
from pathlib import Path

import launch_gemma_codex


class LaunchGemmaCodexTests(unittest.TestCase):
    def test_parse_launcher_args_strips_reasoning(self):
        parsed = launch_gemma_codex.parse_launcher_args(["--reasoning", "--foo", "bar"])
        self.assertTrue(parsed.reasoning)
        self.assertEqual(parsed.codex_args, ["--foo", "bar"])

    def test_parse_launcher_args_keeps_direct_mode_by_default(self):
        parsed = launch_gemma_codex.parse_launcher_args(["--foo"])
        self.assertFalse(parsed.reasoning)
        self.assertEqual(parsed.codex_args, ["--foo"])

    def test_build_codex_environment_selects_local_home(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        env = launch_gemma_codex.build_codex_environment(root, reasoning=False, base_env={})
        self.assertEqual(env["CODEX_HOME"], str(root / ".codex-local"))

    def test_build_codex_environment_selects_reasoning_home(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        env = launch_gemma_codex.build_codex_environment(root, reasoning=True, base_env={})
        self.assertEqual(env["CODEX_HOME"], str(root / ".codex-local-reasoning"))

    def test_build_codex_command_preserves_target_dir(self):
        root = Path("C:/Users/Agent-1/Desktop/gemma")
        command = launch_gemma_codex.build_codex_command(root, Path("C:/work"), ["--foo"])
        self.assertEqual(command[-3:], ["--cd", "C:\\work", "--foo"])
        self.assertIn("codex.exe", command[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_launch_gemma_codex -v`

Expected: fails because `launch_gemma_codex.py` does not exist.

- [ ] **Step 3: Implement launcher helpers**

Create `launch_gemma_codex.py` with:

```python
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


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
        else:
            codex_args.append(arg)
    return LauncherArgs(reasoning=reasoning, codex_args=codex_args)


def build_codex_environment(root, reasoning, base_env=None):
    env = dict(os.environ if base_env is None else base_env)
    env["CODEX_HOME"] = str(Path(root) / (".codex-local-reasoning" if reasoning else ".codex-local"))
    return env


def build_codex_command(root, target_dir, codex_args):
    root = Path(root)
    local_codex = root / "node_modules" / "@openai" / "codex-win32-x64" / "vendor" / "x86_64-pc-windows-msvc" / "bin" / "codex.exe"
    return [str(local_codex), "--cd", str(Path(target_dir)), *codex_args]


def run(argv=None, root=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parent if root is None else Path(root)
    parsed = parse_launcher_args(argv)
    target_dir = Path(os.environ.get("GEMMA_CODEX_TARGET_DIR") or os.getcwd())
    runtime = root / "start-gemma-runtime.ps1"
    subprocess.run([
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(runtime),
    ], check=True, cwd=root)
    command = build_codex_command(root, target_dir, parsed.codex_args)
    env = build_codex_environment(root, parsed.reasoning)
    return subprocess.call(command, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(run())
```

- [ ] **Step 4: Update batch wrapper generation**

Make `gemma-codex.cmd` call `launch_gemma_codex.py %*` through `.venv\Scripts\python.exe`. Update `setup_local_codex.write_launchers()` to generate the same wrapper.

- [ ] **Step 5: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_launch_gemma_codex tests.test_setup_local_codex -v`

Expected: launcher tests pass and setup tests are updated to expect the Python launcher.

### Task 2: Local Reasoning Config Generation

**Files:**
- Modify: `setup_local_codex.py`
- Modify: `tests/test_setup_local_codex.py`

- [ ] **Step 1: Write failing config tests**

Add tests asserting `write_local_codex_config()` creates:

```python
direct_config = root / ".codex-local" / "config.toml"
reasoning_config = root / ".codex-local-reasoning" / "config.toml"
self.assertIn('base_url = "http://127.0.0.1:8081/v1"', direct_config.read_text(encoding="utf-8"))
self.assertIn('base_url = "http://127.0.0.1:8082/v1"', reasoning_config.read_text(encoding="utf-8"))
self.assertNotIn(str(Path.home() / ".codex"), reasoning_config.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v`

Expected: fails because `.codex-local-reasoning` is not generated.

- [ ] **Step 3: Implement config generation**

Add `REASONING_PROXY_PORT = 8082`, allow `build_config_text(root, proxy_port=PROXY_PORT)`, and write both local homes from `write_local_codex_config(root)`.

- [ ] **Step 4: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v`

Expected: all setup tests pass.

### Task 3: Reasoning Graph

**Files:**
- Create: `gemma_reasoning/__init__.py`
- Create: `gemma_reasoning/dspy_programs.py`
- Create: `gemma_reasoning/graph.py`
- Create: `tests/test_gemma_reasoning.py`

- [ ] **Step 1: Write failing graph tests**

```python
import unittest

from gemma_reasoning.graph import ReasoningConfig, run_reasoning_graph


class FakePrograms:
    def plan(self, task, constraints):
        return "Plan: answer directly."

    def verify(self, task, plan, draft):
        return {"approved": True, "notes": "Draft answers the task."}


class FakeClient:
    def __init__(self):
        self.calls = []

    def create_response(self, payload):
        self.calls.append(payload)
        return {"output": [{"content": [{"type": "output_text", "text": "final answer"}]}]}


class ReasoningGraphTests(unittest.TestCase):
    def test_run_reasoning_graph_returns_approved_draft(self):
        client = FakeClient()
        result = run_reasoning_graph(
            {"input": [{"role": "user", "content": "say hi"}]},
            client=client,
            programs=FakePrograms(),
            config=ReasoningConfig(),
        )
        self.assertEqual(result["final_text"], "final answer")
        self.assertEqual(result["plan"], "Plan: answer directly.")
        self.assertEqual(len(client.calls), 1)

    def test_run_reasoning_graph_revises_when_verifier_rejects(self):
        class RejectOnce(FakePrograms):
            def verify(self, task, plan, draft):
                return {"approved": False, "notes": "Missing direct answer."}

        client = FakeClient()
        result = run_reasoning_graph(
            {"input": [{"role": "user", "content": "say hi"}]},
            client=client,
            programs=RejectOnce(),
            config=ReasoningConfig(max_revisions=1),
        )
        self.assertEqual(result["final_text"], "final answer")
        self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_reasoning -v`

Expected: fails because `gemma_reasoning` does not exist.

- [ ] **Step 3: Implement graph and DSPy wrappers**

Implement a small LangGraph `StateGraph` with `plan`, `draft`, `verify`, `revise`, and `finalize` nodes. Implement DSPy wrappers that import `dspy` lazily and can be replaced by fake programs in tests.

- [ ] **Step 4: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_reasoning -v`

Expected: graph tests pass.

### Task 4: Reasoning Proxy

**Files:**
- Create: `gemma_reasoning/upstream.py`
- Create: `gemma_reasoning_proxy.py`
- Create: `tests/test_gemma_reasoning_proxy.py`

- [ ] **Step 1: Write failing proxy tests**

```python
import json
import unittest

import gemma_reasoning_proxy


class ReasoningProxyTests(unittest.TestCase):
    def test_build_response_payload_wraps_final_text(self):
        payload = gemma_reasoning_proxy.build_response_payload("hello")
        text = payload["output"][0]["content"][0]["text"]
        self.assertEqual(text, "hello")

    def test_build_sse_payload_emits_done(self):
        body = gemma_reasoning_proxy.build_sse_payload("hello").decode("utf-8")
        self.assertIn('"delta": "hello"', body)
        self.assertIn("data: [DONE]", body)

    def test_reasoning_handler_uses_graph_runner(self):
        request_payload = {"input": [{"role": "user", "content": "say hi"}]}
        result = gemma_reasoning_proxy.run_reasoning_request(
            request_payload,
            graph_runner=lambda payload: {"final_text": "hello"},
        )
        self.assertEqual(result["output"][0]["content"][0]["text"], "hello")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_reasoning_proxy -v`

Expected: fails because `gemma_reasoning_proxy.py` does not exist.

- [ ] **Step 3: Implement proxy**

Implement an HTTP proxy for `/v1/models` passthrough and `/v1/responses` reasoning. JSON responses return a Responses-like object; streaming requests return a buffered SSE final delta and `[DONE]`.

- [ ] **Step 4: Verify green**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_gemma_reasoning_proxy tests.test_gemma_response_proxy -v`

Expected: reasoning proxy and existing direct proxy tests pass.

### Task 5: Runtime Startup and Dependencies

**Files:**
- Modify: `start-gemma-runtime.ps1`
- Modify: `setup_local_codex.py`
- Modify: `verify-local-setup.ps1`

- [ ] **Step 1: Write failing setup/runtime tests**

Update setup tests to assert `start-gemma-runtime.ps1` contains `gemma_reasoning_proxy.py`, port `8082`, and still starts the direct proxy on `8081`.

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v`

Expected: fails because runtime script does not start the reasoning proxy.

- [ ] **Step 3: Update runtime script**

Start `gemma_reasoning_proxy.py --host 127.0.0.1 --port 8082 --upstream http://127.0.0.1:8081` after the direct proxy is ready.

- [ ] **Step 4: Install local dependencies**

Run: `.\.venv\Scripts\python.exe -m pip install -U langgraph dspy openai`

Expected: installs into the project venv only.

- [ ] **Step 5: Verify unit suite**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_launch_gemma_codex tests.test_setup_local_codex tests.test_gemma_response_proxy tests.test_gemma_reasoning tests.test_gemma_reasoning_proxy -v`

Expected: all listed tests pass.

### Task 6: End-to-End Smoke Checks

**Files:**
- Run: `setup_local_codex.py`
- Run: `verify-local-setup.ps1`

- [ ] **Step 1: Regenerate local setup files**

Run: `.\.venv\Scripts\python.exe setup_local_codex.py`

Expected: `.codex-local` and `.codex-local-reasoning` exist.

- [ ] **Step 2: Verify local setup files**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\verify-local-setup.ps1`

Expected: local Codex binary exists, both local config homes exist, and no global config is modified.

- [ ] **Step 3: Start runtime**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\start-gemma-runtime.ps1`

Expected: `/v1/models` responds on `8080`, `8081`, and `8082`.

- [ ] **Step 4: Smoke the reasoning proxy**

Run a small `/v1/responses` POST to `http://127.0.0.1:8082/v1/responses` with a short prompt.

Expected: JSON response includes output text from Gemma after planner/verifier orchestration.

## Self-Review

- Spec coverage: local-only flag behavior, two Codex homes, reasoning proxy, LangGraph/DSPy graph, error handling, tests, and non-global constraints are each represented by tasks.
- Placeholder scan: no `TBD`, `TODO`, or unspecified "add tests later" steps remain.
- Type consistency: launcher functions, reasoning graph entrypoint, and proxy helper names are defined before later tasks reference them.
- Repository constraint: commit and worktree steps are intentionally omitted because this folder is not a git repository.
