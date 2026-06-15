# Gemma Parallel Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local Gemma Codex CLI runtime prefer and enforce safe bounded parallel work for independent subagent/tool-capable tasks.

**Architecture:** Keep the existing Gemma-owned agent runtime. Update durable Codex metadata/instructions in `setup_local_codex.py`, make `AgentSupervisor` default multi-subagent fanout to concurrent execution, and lock the behavior with focused unit tests plus README documentation.

**Tech Stack:** Python `unittest`, local Codex model catalog generation, Gemma `AgentSupervisor`, MCP `gemma_run_subagents`.

---

### Task 1: Codex Catalog And Base Instructions

**Files:**
- Modify: `setup_local_codex.py`
- Modify: `tests/test_setup_local_codex.py`

- [ ] **Step 1: Write failing tests for parallel instructions and catalog capability**

Update `tests/test_setup_local_codex.py`:

```python
    def test_base_instructions_require_safe_parallel_work(self):
        instructions = setup_local_codex.build_base_instructions()

        self.assertContainsAll(
            instructions,
            [
                "parallel batches",
                "independent reads, searches, checks, and subagent tasks",
                "gemma_run_subagents",
                "Do not parallelize dependent commands, overlapping file edits, git add/commit, installs, migrations, or test-after-edit loops",
                "bounded fanout",
            ],
        )

    def test_catalog_advertises_parallel_tool_call_capability(self):
        model = setup_local_codex.build_model_catalog()["models"][0]

        self.assertIs(model["supports_parallel_tool_calls"], True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex.LocalCodexSetupTests.test_base_instructions_require_safe_parallel_work tests.test_setup_local_codex.LocalCodexSetupTests.test_catalog_advertises_parallel_tool_call_capability -v
```

Expected: both tests fail because the instruction fragments are absent and catalog parallel support is false.

- [ ] **Step 3: Implement base instruction and catalog changes**

Update `build_base_instructions()` to include the exact fragments from Step 1 while staying under `BASE_INSTRUCTIONS_CL100K_LIMIT`. Update `build_model_catalog()` so `"supports_parallel_tool_calls": True`.

- [ ] **Step 4: Run focused setup tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex.LocalCodexSetupTests -v
```

Expected: all `LocalCodexSetupTests` pass.

### Task 2: Supervisor Concurrent Fanout Default

**Files:**
- Modify: `gemma_agent/supervisor.py`
- Modify: `tests/test_gemma_agent_runtime.py`

- [ ] **Step 1: Write a failing test for default concurrent multi-subagent fanout**

Add a test near existing subagent concurrency tests in `tests/test_gemma_agent_runtime.py`:

```python
    def test_spawn_subagents_defaults_multi_task_fanout_to_concurrent_execution(self):
        started = threading.Event()
        release = threading.Event()
        entered = []

        class BlockingSubAgent:
            def __init__(self, spec):
                self.spec = spec

            def run(self):
                entered.append(self.spec["id"])
                if len(entered) == 2:
                    started.set()
                release.wait(1)
                return SubAgentResult(agent_id=self.spec["id"], task=self.spec["task"], final="done", ok=True)

        model = FakeModelClient(
            [
                json.dumps(
                    {
                        "action": "spawn_subagents",
                        "tasks": [
                            {"id": "a", "task": "inspect a"},
                            {"id": "b", "task": "inspect b"},
                        ],
                    }
                ),
                json.dumps({"action": "final", "content": "done"}),
            ]
        )

        def factory(spec, context):
            return BlockingSubAgent(spec)

        result_holder = {}
        thread = threading.Thread(
            target=lambda: result_holder.setdefault(
                "result",
                AgentSupervisor(
                    model,
                    ToolExecutor(ToolRegistry()),
                    max_iterations=2,
                    subagent_factory=factory,
                ).run("split independent inspections"),
            )
        )
        thread.start()
        self.assertTrue(started.wait(1), "second subagent did not start before first was released")
        release.set()
        thread.join(2)

        self.assertEqual(result_holder["result"].final, "done")
```

Ensure `threading` is imported at the top of the test file.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.GemmaAgentRuntimeTests.test_spawn_subagents_defaults_multi_task_fanout_to_concurrent_execution -v
```

Expected: fail because omitted `concurrent` currently runs sequentially.

- [ ] **Step 3: Implement the supervisor default**

In `AgentSupervisor._handle_spawn_subagents`, compute:

```python
concurrent = action.get("concurrent")
if concurrent is None:
    concurrent = len(agents) > 1
```

Pass `concurrent=bool(concurrent)` to `run_subagents`.

- [ ] **Step 4: Run focused runtime tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.GemmaAgentRuntimeTests.test_spawn_subagents_defaults_multi_task_fanout_to_concurrent_execution tests.test_gemma_agent_runtime.GemmaAgentRuntimeTests.test_supervisor_spawns_subagents_with_isolated_contexts tests.test_gemma_agent_runtime.GemmaAgentRuntimeTests.test_run_subagents_can_use_thread_pool_for_independent_tasks -v
```

Expected: all listed tests pass.

### Task 3: MCP And README Documentation

**Files:**
- Modify: `tests/test_gemma_agent_mcp.py`
- Modify: `README.md`

- [ ] **Step 1: Add MCP default-concurrency regression test**

Add a test in `tests/test_gemma_agent_mcp.py` that records overlapping subagent starts through a fake model client or subagent runner and asserts `gemma_run_subagents_response(["alpha", "beta"])` uses concurrent execution by default.

- [ ] **Step 2: Run the MCP test to verify current behavior**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_mcp.GemmaAgentMCPTests -v
```

Expected: existing tests pass; new default-concurrency test should pass if current MCP default is preserved.

- [ ] **Step 3: Update README**

Add a short subsection under "Agent Action Or Tool-Calling Behavior" explaining:

```markdown
### Safe Parallel Work

Gemma should use parallel batches for independent reads, searches, checks, and subagent tasks. Use `gemma_run_subagents` for independent investigations or review tracks, with bounded `max_workers`. Keep dependent commands, overlapping file edits, `git add`/`git commit`, installs, migrations, and test-after-edit loops serial.
```

- [ ] **Step 4: Run documentation-related tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_mcp.GemmaAgentMCPTests tests.test_setup_local_codex.LocalCodexSetupTests -v
```

Expected: all listed tests pass.

### Task 4: Regenerate And Verify

**Files:**
- Generated/verify: `.codex-local/model-catalog.json`
- Generated/verify: `.codex-local-reasoning/model-catalog.json`

- [ ] **Step 1: Regenerate local Codex config**

Run:

```powershell
.\.venv\Scripts\python.exe setup_local_codex.py
```

Expected: regenerated direct and reasoning Codex homes.

- [ ] **Step 2: Run setup verifier**

Run:

```powershell
.\verify-local-setup.ps1
```

Expected: verifier exits 0.

- [ ] **Step 3: Run full Python unit test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 4: Check diff and self-review**

Run:

```powershell
git diff -- setup_local_codex.py gemma_agent/supervisor.py tests/test_setup_local_codex.py tests/test_gemma_agent_runtime.py tests/test_gemma_agent_mcp.py README.md docs/superpowers/specs/2026-06-14-gemma-parallel-agent-design.md docs/superpowers/plans/2026-06-14-gemma-parallel-agent.md
```

Expected: diff is limited to the parallel policy, tests, docs, and generated catalog source behavior.

## Self-Review

- Spec coverage: prompt/catalog policy, supervisor concurrency default, MCP behavior, docs, and verification are covered.
- Placeholder scan: no placeholder markers.
- Type consistency: all referenced function and test names match existing modules or are defined in task steps.
