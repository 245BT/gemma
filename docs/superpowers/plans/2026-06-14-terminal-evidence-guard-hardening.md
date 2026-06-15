# Terminal Evidence Guard Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix reviewed terminal-runner lifecycle/output risks and supervisor final-claim evidence gaps.

**Architecture:** Keep the terminal runner API stable while adding a per-background-job watchdog and replacing unbounded stream buffers with bounded rolling buffers that preserve total lengths and hashes. Keep supervisor evidence checks regex-based, but expand positive edit/command detection and strip common negative caveats before matching.

**Tech Stack:** Python `unittest`, `subprocess`, `threading`, existing `gemma_agent.terminal` and `gemma_agent.supervisor` modules.

---

### Task 1: Terminal Background Hard Timeout

**Files:**
- Modify: `tests/test_gemma_agent_runtime.py`
- Modify: `gemma_agent/terminal.py`

- [ ] **Step 1: Write the failing test**

Add a test that starts a Python process with `background_after_sec` lower than `timeout_sec`, waits past the hard timeout without polling `job_status`, and asserts the process has been killed.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.GemmaAgentRuntimeTests.test_terminal_runner_background_job_enforces_hard_timeout_without_polling -v`

Expected before implementation: FAIL because the process remains alive until polled or manually killed.

- [ ] **Step 3: Implement minimal watchdog**

Start a daemon watchdog thread when storing a background/stalled job. It sleeps until the absolute deadline, terminates the process if still running, removes the job, and joins readers briefly.

- [ ] **Step 4: Run focused terminal tests**

Run the RED test plus existing background/stall tests.

### Task 2: Bounded Terminal Output Buffering

**Files:**
- Modify: `tests/test_gemma_agent_runtime.py`
- Modify: `gemma_agent/terminal.py`

- [ ] **Step 1: Write the failing test**

Add a test that emits more than `max_tail_chars` output and asserts the internal stored stdout buffer is bounded while `stdout_length`, `stdout_tail`, and `stdout_sha256` still reflect the full stream.

- [ ] **Step 2: Run focused test and verify RED**

Run: `.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime.GemmaAgentRuntimeTests.test_terminal_runner_bounds_background_stdout_buffer_while_tracking_full_length_and_hash -v`

Expected before implementation: FAIL because `stdout_parts` stores the full output.

- [ ] **Step 3: Implement bounded stream buffer**

Replace list sinks with a small buffer object that updates SHA-256 and total length for every chunk while retaining only the tail window.

- [ ] **Step 4: Run focused terminal tests**

Run terminal runner tests covering completion, stall, background, kill, timeout, and output.

### Task 3: Supervisor Positive Claim Coverage

**Files:**
- Modify: `tests/test_gemma_agent_runtime.py`
- Modify: `gemma_agent/supervisor.py`

- [ ] **Step 1: Write failing tests for edit verbs**

Add cases for `deleted`, `removed`, `added`, `made changes to`, `moved`, `copied`, and `replaced` file claims without tool evidence. Expected: first final is rejected and fallback final is accepted.

- [ ] **Step 2: Write failing tests for bare command names**

Add cases for `dir`, `pwd`, `grep`, `pip`, and `make`. Expected: unsupported final claim is rejected without command evidence.

- [ ] **Step 3: Run focused supervisor tests and verify RED**

Run the new supervisor tests individually.

- [ ] **Step 4: Expand regex and command allowlist carefully**

Add common file edit verbs and common shell/build commands while preserving ordinary prose exclusions.

### Task 4: Supervisor Negative Caveat Coverage

**Files:**
- Modify: `tests/test_gemma_agent_runtime.py`
- Modify: `gemma_agent/supervisor.py`

- [ ] **Step 1: Write failing tests for test caveats**

Add cases for `I didn't run tests.`, `I have not run tests.`, and contractions around `didn't`/`haven't`.

- [ ] **Step 2: Write failing tests for no-change file caveats**

Add cases for `Files weren't changed.`, `I haven't edited files.`, and `No files have been removed.`

- [ ] **Step 3: Run focused tests and verify RED**

Run the new caveat tests individually.

- [ ] **Step 4: Strip negative caveats before positive matching**

Extend negation patterns for auxiliary verbs, contractions, and no-change passive forms.

### Task 5: Verification And Review

**Files:**
- Inspect: `gemma_agent/terminal.py`
- Inspect: `gemma_agent/supervisor.py`
- Run: focused tests and full suite

- [ ] **Step 1: Run focused regression suite**

Run: `.venv\Scripts\python.exe -m unittest tests.test_gemma_agent_runtime -v`

- [ ] **Step 2: Run full project suite**

Run: `npm test`

- [ ] **Step 3: Self-review diff**

Check for process leaks, unbounded buffers, overly broad claim regexes, and compatibility with existing ToolResult matching.

- [ ] **Step 4: Report evidence**

Report exact commands, pass/fail counts, remaining caveats, and files changed.
