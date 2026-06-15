# Gemma Public Release Redesign Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tidy the workspace, introduce a clearer runtime V2 spine, and harden the local Gemma agent for public-release-quality claims without adding package distribution or hosted multi-tenant support.

**Architecture:** Keep the current local Codex/Gemma launch path compatible while adding explicit runtime policy boundaries, safer artifact handling, reproducible Codex 0.139.0 provenance, and launch verification gates. Apply hardening over clear interfaces instead of scattered patches.

**Tech Stack:** Python 3.14 stdlib, unittest, PowerShell generation tests, local Codex CLI 0.139.0, existing Gemma runtime modules.

---

## File Structure

- Modify: `.gitignore`
- Modify: `plan.md`
- Create/modify: `.superpowers/brainstorm/manual-20260614120000/**`
- Create: `docs/superpowers/specs/2026-06-14-gemma-public-release-redesign-hardening-design.md`
- Create: `docs/superpowers/plans/2026-06-14-gemma-public-release-redesign-hardening.md`
- Modify: `gemma_agent/config.py`
- Modify: `gemma_agent/tools.py`
- Modify: `gemma_agent/supervisor.py`
- Modify: `gemma_agent/safety.py` if redaction helpers belong there after inspection
- Modify: `gemma_agent_mcp.py`
- Create: `benchmarks/agent_benchmarks/subprocesses.py`
- Modify: `benchmarks/agent_benchmarks/runner.py`
- Modify: `benchmarks/agent_benchmarks/local_repo_fix.py`
- Modify: `benchmarks/agent_benchmarks/capability_card.py`
- Modify: `scripts/sanitize_local_artifacts.py`
- Modify: `setup_local_codex.py`
- Modify: `launch_gemma_codex.py`
- Modify: `verify-local-setup.ps1` only through `setup_local_codex.py`
- Modify: `README.md`
- Modify: `tests/test_gemma_agent_runtime.py`
- Modify: `tests/test_gemma_agent_mcp.py`
- Modify: `tests/test_agent_benchmarks.py`
- Modify: `tests/test_capability_card.py`
- Modify: `tests/test_sanitize_local_artifacts.py`
- Modify: `tests/test_setup_local_codex.py`
- Modify: `tests/test_launch_gemma_codex.py`

## Current Baseline

- Full suite baseline: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` ran 471 tests with 0 failures before this plan.
- `npm test` ran the same 471 tests with 0 failures before this plan.
- `.\verify-local-setup.ps1` passed and reported `codex-cli 0.139.0`.
- `npm exec --yes --package @openai/codex@0.139.0 -- codex --version` reported `codex-cli 0.139.0`.
- Process scan found one `llama-server.exe`.

---

### Task 1: Workspace Tidiness Artifacts

**Files:**
- Modify: `.gitignore`
- Create/modify: `.superpowers/brainstorm/manual-20260614120000/**`
- Create/modify: `plan.md`
- Create: `docs/superpowers/specs/2026-06-14-gemma-public-release-redesign-hardening-design.md`
- Create: `docs/superpowers/plans/2026-06-14-gemma-public-release-redesign-hardening.md`

- [ ] **Step 1: Verify tidy structure exists**

Run:

```powershell
Get-ChildItem -Force .superpowers\brainstorm\manual-20260614120000 -Recurse | Select-Object FullName,Length
Test-Path plan.md
Test-Path docs\superpowers\specs\2026-06-14-gemma-public-release-redesign-hardening-design.md
```

Expected: `.superpowers` content/state files exist, `plan.md` exists, design spec exists.

- [ ] **Step 2: Verify ignored local evidence**

Run:

```powershell
git check-ignore .superpowers .worktrees
```

Expected: both paths are ignored.

- [ ] **Step 3: Scan planning files for placeholders**

Run:

```powershell
rg -n "TBD|TODO|implement later|fill in|placeholder|\?\?\?" plan.md docs\superpowers\specs docs\superpowers\plans\2026-06-14-gemma-public-release-redesign-hardening.md
```

Expected: no matches.

### Task 2: Runtime V2 Policy Spine

**Files:**
- Modify: `gemma_agent/config.py`
- Modify: `gemma_agent/tools.py`
- Modify: `gemma_agent_mcp.py`
- Modify: `tests/test_gemma_agent_runtime.py`
- Modify: `tests/test_gemma_agent_mcp.py`

- [ ] **Step 1: Add failing tests for runtime policy**

Add tests that make the expected boundary explicit:

```python
def test_runtime_config_defaults_to_local_trusted_profile(self):
    config = RuntimeConfig()
    self.assertEqual(config.runtime_profile, "local_trusted")
    self.assertTrue(config.terminal_tools_enabled)

def test_public_profile_does_not_register_terminal_command(self):
    registry = build_default_tool_registry(runtime_config=RuntimeConfig(runtime_profile="public_local"))
    self.assertIsNone(registry.get("terminal_command"))
    self.assertIsNotNone(registry.get("context7_search"))
    self.assertIsNotNone(registry.get("duckduckgo_search"))

def test_gemma_agent_mcp_public_profile_disables_terminal_for_subagents(self):
    response = gemma_run_subagents_response(
        ["Inspect docs only."],
        model_client=FakeModelClient([json.dumps({"action": "final", "content": "ok"})]),
        runtime_config=RuntimeConfig(runtime_profile="public_local"),
    )
    self.assertEqual(response["status"], "ok")
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.GemmaAgentRuntimeTests.test_runtime_config_defaults_to_local_trusted_profile tests.test_gemma_agent_runtime.GemmaAgentRuntimeTests.test_public_profile_does_not_register_terminal_command tests.test_gemma_agent_mcp.GemmaAgentMCPTests.test_gemma_agent_mcp_public_profile_disables_terminal_for_subagents -v
```

Expected: fail because `RuntimeConfig` has no profile fields and `build_default_tool_registry` has no runtime-config argument.

- [ ] **Step 3: Implement runtime policy fields**

In `gemma_agent/config.py`, extend `RuntimeConfig`:

```python
@dataclass(frozen=True)
class RuntimeConfig:
    base_url: str = "http://127.0.0.1:8081/v1"
    model: str | None = None
    request_timeout_sec: float = 1800
    default_tool_timeout_sec: float = 30
    max_iterations: int = 8
    max_subagents: int = 8
    workspace_roots: tuple[Path, ...] = field(default_factory=lambda: (Path.cwd(),))
    runtime_profile: str = "local_trusted"
    terminal_tools_enabled: bool = True

    def public_local(self) -> "RuntimeConfig":
        return replace(self, runtime_profile="public_local", terminal_tools_enabled=False)
```

Also import `replace` from `dataclasses`.

- [ ] **Step 4: Wire policy into tool registry**

Update `build_default_tool_registry(...)` to accept `runtime_config: RuntimeConfig | None = None`. Use `runtime_config.workspace_roots` when explicit `workspace_roots` is not passed. Register `terminal_command` only when `runtime_config.terminal_tools_enabled` is true.

- [ ] **Step 5: Wire policy into MCP subagents**

Update `gemma_run_subagents_response(...)` to accept `runtime_config: RuntimeConfig | None = None` and pass it to `build_default_tool_registry(...)`.

- [ ] **Step 6: Run targeted tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime tests.test_gemma_agent_mcp -v
```

Expected: pass.

### Task 3: Benchmark Subprocess Cleanup And Redacted Output

**Files:**
- Create: `benchmarks/agent_benchmarks/subprocesses.py`
- Modify: `benchmarks/agent_benchmarks/runner.py`
- Modify: `benchmarks/agent_benchmarks/local_repo_fix.py`
- Modify: `tests/test_agent_benchmarks.py`

- [ ] **Step 1: Add failing child-process timeout test**

Add a test that starts a command which spawns a child process that writes a marker after the parent timeout. Assert the marker is not created.

```python
def test_agent_benchmark_timeout_kills_child_process_tree(self):
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "child-survived.txt"
        command = [
            sys.executable,
            "-c",
            (
                "import subprocess, sys, time; "
                f"subprocess.Popen([sys.executable, '-c', \"import pathlib,time; time.sleep(1); pathlib.Path({str(marker)!r}).write_text('alive')\"]); "
                "time.sleep(5)"
            ),
        ]
        result = run_bounded_subprocess(command, cwd=Path(tmp), timeout=0.2)
        time.sleep(1.2)
        self.assertEqual(result.return_code, 124)
        self.assertFalse(marker.exists())
```

- [ ] **Step 2: Add failing redaction test**

Add a test that fake stdout/stderr contain `OPENAI_API_KEY=sk-test-secret`, `password=hunter2`, and a private-key marker. Assert written benchmark logs and summary do not contain those strings and do contain hashes/lengths.

- [ ] **Step 3: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agent_benchmarks -v
```

Expected: new tests fail before the bounded subprocess helper and redacted output writer exist.

- [ ] **Step 4: Implement subprocess helper**

Create `benchmarks/agent_benchmarks/subprocesses.py` with:

```python
@dataclass(frozen=True)
class BoundedCompletedProcess:
    return_code: int | None
    stdout: str
    stderr: str
    timed_out: bool

def run_bounded_subprocess(command, *, cwd, env=None, timeout, text=True):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        encoding="utf-8",
        errors="replace",
        **start_new_process_group_kwargs(),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return BoundedCompletedProcess(process.returncode, stdout or "", stderr or "", False)
    except subprocess.TimeoutExpired:
        terminate_process_tree(process.pid, wait_process=process.wait, timeout=1)
        stdout, stderr = process.communicate(timeout=1)
        return BoundedCompletedProcess(124, stdout or "", stderr or "", True)
```

- [ ] **Step 5: Implement redacted output metadata**

Add helpers that write bounded redacted tails, not raw streams:

```python
def redact_sensitive_text(text: str) -> str:
    patterns = [
        (r"sk-[A-Za-z0-9_-]{8,}", "sk-[REDACTED]"),
        (r"(?i)(password|passwd|token|api[_-]?key|bearer)\s*[:=]\s*\S+", r"\1=[REDACTED]"),
        (r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]"),
    ]
    ...
```

Write `stdout_tail`, `stdout_length`, `stdout_sha256`, `stderr_tail`, `stderr_length`, and `stderr_sha256` into summary metadata. Keep full raw stdout/stderr out of shareable logs by default.

- [ ] **Step 6: Replace direct `subprocess.run` in benchmark runner paths**

Use `run_bounded_subprocess(...)` in `benchmarks/agent_benchmarks/runner.py`, `_run_gemma_solver(...)`, and `_run_tests(...)` in `local_repo_fix.py`.

- [ ] **Step 7: Run targeted tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agent_benchmarks -v
```

Expected: pass.

### Task 4: Capability Card Path Safety

**Files:**
- Modify: `benchmarks/agent_benchmarks/capability_card.py`
- Modify: `tests/test_capability_card.py`

- [ ] **Step 1: Add failing traversal tests**

Add tests for all writers:

```python
def test_write_capability_card_sanitizes_run_id_under_output_dir(self):
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "runs"
        outside = Path(tmp) / "outside"
        paths = write_capability_card([], output_dir=out_dir, run_id=str(outside), model_label="Gemma")
        for path in paths.values():
            self.assertTrue(path.resolve().is_relative_to(out_dir.resolve()))
        self.assertFalse(Path(str(outside) + ".capability-card.json").exists())

def test_write_comparison_card_sanitizes_run_id_under_output_dir(self):
    ...

def test_run_choice_benchmark_sanitizes_run_id_under_output_dir(self):
    ...
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_capability_card -v
```

Expected: traversal tests fail.

- [ ] **Step 3: Implement safe label helper**

Add:

```python
def _safe_run_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value))
    return cleaned.strip("._-") or "capability-card"
```

Use the safe value for filenames while preserving the original `run_id` in JSON payloads.

- [ ] **Step 4: Run capability-card tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_capability_card -v
```

Expected: pass.

### Task 5: Public Bundle Sanitizer Coverage

**Files:**
- Modify: `scripts/sanitize_local_artifacts.py`
- Modify: `tests/test_sanitize_local_artifacts.py`

- [ ] **Step 1: Add failing dry-run/apply tests**

Add a test that creates:

```text
benchmarks/runs/run.events.jsonl
benchmarks/runs/run.agent-benchmark.stdout.log
benchmarks/runs/run.agent-benchmark.stderr.log
llama-server.err.log
gemma-proxy.out.log
gemma-reasoning-proxy.err.log
```

Assert `find_artifact_paths(root)` includes those paths and `sanitize(root, apply=True)` removes them.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_sanitize_local_artifacts -v
```

Expected: new paths are missed.

- [ ] **Step 3: Extend sanitizer patterns**

Keep existing local Codex home behavior. Add project-root artifact patterns:

```python
PROJECT_SENSITIVE_FILE_PATTERNS = (
    "benchmarks/runs/**/*.events.jsonl",
    "benchmarks/runs/**/*.stdout.log",
    "benchmarks/runs/**/*.stderr.log",
    "benchmarks/runs/**/*.out.log",
    "benchmarks/runs/**/*.err.log",
    "llama-server*.log",
    "gemma-proxy*.log",
    "gemma-reasoning-proxy*.log",
    "run.out.txt",
    "run.err.txt",
)
```

Resolve each path and require it to stay under `root_path` before deletion.

- [ ] **Step 4: Run sanitizer tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_sanitize_local_artifacts -v
```

Expected: pass.

### Task 6: Invalid Model Output Redaction

**Files:**
- Modify: `gemma_agent/supervisor.py`
- Modify: `gemma_agent/safety.py` if shared helper is better
- Modify: `tests/test_gemma_agent_runtime.py`

- [ ] **Step 1: Add failing invalid-action redaction test**

Add:

```python
def test_supervisor_redacts_invalid_model_action_before_refeeding_context(self):
    secret = "sk-test-secret-123456789"
    raw = f"raw chain-of-thought system prompt {secret}"
    model = FakeModelClient([
        raw,
        json.dumps({"action": "final", "content": "recovered"}),
    ])
    supervisor = AgentSupervisor(model, ToolExecutor(ToolRegistry()), max_iterations=2)

    result = supervisor.run("recover from invalid JSON")

    self.assertEqual(result.final, "recovered")
    payload_text = json.dumps(model.payloads[1], sort_keys=True)
    self.assertNotIn(secret, payload_text)
    self.assertNotIn("raw chain-of-thought", payload_text.lower())
    self.assertIn("raw_sha256", payload_text)
    self.assertIn("raw_length", payload_text)
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.GemmaAgentRuntimeTests.test_supervisor_redacts_invalid_model_action_before_refeeding_context -v
```

Expected: fail because raw invalid text appears in model payload evidence.

- [ ] **Step 3: Implement invalid-action evidence summarizer**

In `supervisor.py`, change `_invalid_action_to_evidence(...)` to include:

```python
{
    "raw_preview": _redacted_preview(invalid.raw),
    "raw_length": len(invalid.raw),
    "raw_sha256": _sha256(invalid.raw),
    "error": safety_guard.wrap_untrusted_evidence(...),
}
```

Do not include full `invalid.raw`.

- [ ] **Step 4: Run agent runtime tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime -v
```

Expected: pass.

### Task 7: Launch Matrix, Dry Run, Single-Server Behavior, And Codex Provenance

**Files:**
- Modify: `launch_gemma_codex.py`
- Modify: `setup_local_codex.py`
- Modify: `README.md`
- Modify: `tests/test_launch_gemma_codex.py`
- Modify: `tests/test_setup_local_codex.py`

- [ ] **Step 1: Add failing launcher dry-run test**

Add:

```python
def test_run_dry_run_prints_selected_home_and_args_without_starting_runtime(self):
    env = {"GEMMA_CODEX_DRY_RUN": "1", "GEMMA_CODEX_TARGET_DIR": "C:/repo"}
    with patch.dict(os.environ, env, clear=False), patch.object(launch_gemma_codex.subprocess, "run") as start:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = launch_gemma_codex.run(["--reasoning", "--yolo", "exec", "hi"], root=Path("C:/Users/Agent-1/Desktop/gemma"))
    self.assertEqual(code, 0)
    self.assertFalse(start.called)
    payload = json.loads(output.getvalue())
    self.assertEqual(payload["codex_home_name"], ".codex-local-reasoning")
    self.assertIn("exec", payload["codex_args"])
```

- [ ] **Step 2: Add generated shim dry-run verification test**

Use the generated shim text from `setup_local_codex.build_global_shim_text(...)` and assert it preserves `%*`, `GEMMA_CODEX_TARGET_DIR`, and the `sonion` no-confirm marker.

- [ ] **Step 3: Add functional single-llama cleanup script test**

Add a generated PowerShell harness test that mocks process rows and asserts the generated `Stop-ExtraLlamaServers` logic preserves listener PID/ancestors and stops stale same-repo duplicates.

- [ ] **Step 4: Add Codex provenance manifest test**

Add a source function that reports:

```python
{
    "selected_path": "...tools/codex-local/bin/codex.exe",
    "version": "codex-cli 0.139.0",
    "sha256": "...",
    "npm_version": "0.139.0",
}
```

Test that `verify-local-setup.ps1` prints the selected binary and version and that `README.md` documents temporary comparison with:

```powershell
npm exec --yes --package @openai/codex@0.139.0 -- codex --version
```

- [ ] **Step 5: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_launch_gemma_codex tests.test_setup_local_codex -v
```

Expected: new dry-run/provenance/functional tests fail before implementation.

- [ ] **Step 6: Implement dry-run path**

In `launch_gemma_codex.run(...)`, before starting runtime when `GEMMA_CODEX_DRY_RUN=1`, print JSON with root, target dir, reasoning, yolo, codex home name, selected command, and codex args. Return 0.

- [ ] **Step 7: Implement README command matrix**

Document `son`, `sonion`, `operator`, `--reasoning`, typo-compatible `--reasoing`, and `--yolo`. State plainly that this pass targets local trusted operation, not hosted untrusted users.

- [ ] **Step 8: Run launch/setup tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_launch_gemma_codex tests.test_setup_local_codex -v
```

Expected: pass.

### Task 8: Final Verification And Report

**Files:**
- Create: `docs/superpowers/reports/2026-06-14-gemma-public-release-redesign-hardening-report.md`
- Modify: `benchmarks/BENCHMARKS.md` only if benchmarks are rerun

- [ ] **Step 1: Run focused suites**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime tests.test_gemma_agent_mcp tests.test_agent_benchmarks tests.test_capability_card tests.test_sanitize_local_artifacts tests.test_launch_gemma_codex tests.test_setup_local_codex -v
```

Expected: pass.

- [ ] **Step 2: Run full suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: pass.

- [ ] **Step 3: Run package script**

Run:

```powershell
npm test
```

Expected: pass.

- [ ] **Step 4: Run setup verifier**

Run:

```powershell
.\verify-local-setup.ps1
```

Expected: pass and report Codex 0.139.0.

- [ ] **Step 5: Verify exactly one llama server**

Run:

```powershell
Get-Process llama-server -ErrorAction SilentlyContinue | Select-Object Id,Path
```

Expected: one process for this workspace when runtime is running, or zero if runtime is intentionally stopped.

- [ ] **Step 6: Write final report**

Report sections:

1. Executive summary
2. Research/docs used
3. Baseline measurements
4. Bugs and leaks found
5. Design changes
6. Speed/context measurements
7. Gemma sub-agent implementation status
8. DuckDuckGo verification
9. Tests run
10. Before/after benchmark table
11. Remaining risks
12. Reproduction commands
13. Final verdict

## Self-Review

- Spec coverage: workspace tidiness, runtime policy spine, subprocess cleanup, artifact redaction, path safety, sanitizer coverage, invalid-output handling, launch matrix, Codex comparison, and final report all map to tasks.
- Placeholder scan: no placeholder sections are intentionally left open.
- Type consistency: new names are `RuntimeConfig.runtime_profile`, `terminal_tools_enabled`, `run_bounded_subprocess`, `BoundedCompletedProcess`, and `_safe_run_id`.
- Scope check: package distribution and hosted public access are explicitly out of scope for this pass.
