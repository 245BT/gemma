# Gemma DuckDuckGo MCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gemma-local DuckDuckGo MCP search tool for `son`, `sonion`, and `operator`.

**Architecture:** Create `duckduckgo_mcp.py` as a local Python MCP stdio server using the official `mcp==1.27.2` SDK. Keep search fetching/parsing dependency-light with stdlib HTTP and HTML parsing, and generate MCP config only inside `.codex-local` and `.codex-local-reasoning`.

**Tech Stack:** Python 3.14, `unittest`, official MCP Python SDK `mcp==1.27.2`, Codex `config.toml` MCP server tables.

---

## File Structure

- Create `duckduckgo_mcp.py`: local MCP server, DuckDuckGo request helper, HTML parser, formatting, and tool registration.
- Create `tests/test_duckduckgo_mcp.py`: parser, formatting, fallback, and tool behavior tests.
- Create `requirements-gemma.txt`: pins `mcp==1.27.2`.
- Modify `setup_local_codex.py`: generate local MCP config, add model instructions for search behavior, and include MCP dependency checks in verification script.
- Modify `tests/test_setup_local_codex.py`: assert generated local config and verification script behavior.

## Task 1: Add DuckDuckGo Parser Tests

**Files:**
- Create: `tests/test_duckduckgo_mcp.py`
- Create: `duckduckgo_mcp.py`

- [ ] **Step 1: Write failing parser and fallback tests**

Add tests that import `duckduckgo_mcp` and assert:

```python
SAMPLE_HTML = """
<html>
  <body>
    <div class="result">
      <a class="result__a" href="https://example.com/alpha">Alpha Result</a>
      <a class="result__snippet">First useful snippet.</a>
    </div>
    <div class="result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.org%2Fbeta">Beta Result</a>
      <div class="result__snippet">Second useful snippet.</div>
    </div>
  </body>
</html>
"""
```

Expected behaviors:

- `parse_duckduckgo_html(SAMPLE_HTML, max_results=10)` returns two `SearchResult` items.
- Redirect URLs with `uddg` are decoded.
- `format_results("alpha beta", results)` includes numbered titles, URLs, and snippets.
- `cap_max_results(0)` returns 1 and `cap_max_results(999)` returns 10.
- `search_duckduckgo("x", fetcher=failing_fetcher)` returns the internet-unavailable fallback message.

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_duckduckgo_mcp -v`

Expected: fail because `duckduckgo_mcp` does not exist.

## Task 2: Implement DuckDuckGo Search Module

**Files:**
- Create: `duckduckgo_mcp.py`

- [ ] **Step 1: Implement minimal parser and search helpers**

Implement:

- `UNAVAILABLE_MESSAGE`
- `SearchResult`
- `DuckDuckGoHTMLParser`
- `cap_max_results`
- `decode_duckduckgo_url`
- `parse_duckduckgo_html`
- `fetch_duckduckgo_html`
- `format_results`
- `search_duckduckgo`

Use `urllib.request`, `urllib.parse`, `html.parser`, and `dataclasses`.

- [ ] **Step 2: Run parser tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_duckduckgo_mcp -v`

Expected: pass for parser and fallback helpers.

## Task 3: Add MCP Server Wiring

**Files:**
- Modify: `duckduckgo_mcp.py`
- Create: `requirements-gemma.txt`
- Modify: `tests/test_duckduckgo_mcp.py`

- [ ] **Step 1: Write failing MCP wiring tests**

Add tests that assert:

- `build_server()` returns an object when the `mcp` package is importable.
- `duckduckgo_search("python", max_results=3, fetcher=fake_fetcher)` returns parsed fake results.

- [ ] **Step 2: Run tests to verify expected failure before dependency/code**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_duckduckgo_mcp -v`

Expected: fail until MCP wiring and local dependency are added.

- [ ] **Step 3: Add dependency and MCP wiring**

Create `requirements-gemma.txt`:

```text
mcp==1.27.2
```

In `duckduckgo_mcp.py`, add:

```python
def build_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("duckduckgo", json_response=True)

    @server.tool()
    def duckduckgo_search(query: str, max_results: int = 5) -> str:
        """Search DuckDuckGo and return compact source-bearing web results."""
        return search_duckduckgo(query, max_results=max_results)

    return server
```

Add `main()` that runs `build_server().run(transport="stdio")`.

- [ ] **Step 4: Install MCP dependency locally**

Run: `.\.venv\Scripts\python.exe -m pip install -r requirements-gemma.txt`

Expected: installs `mcp==1.27.2`.

- [ ] **Step 5: Run MCP tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_duckduckgo_mcp -v`

Expected: pass.

## Task 4: Generate Gemma-Local MCP Config

**Files:**
- Modify: `setup_local_codex.py`
- Modify: `tests/test_setup_local_codex.py`

- [ ] **Step 1: Write failing setup tests**

Add assertions that `build_config_text(Path("C:/work/gemma"))` contains:

- `[mcp_servers.duckduckgo]`
- `enabled = true`
- `command = "C:\\\\work\\\\gemma\\\\.venv\\\\Scripts\\\\python.exe"`
- `args = ["C:\\\\work\\\\gemma\\\\duckduckgo_mcp.py"]`
- `enabled_tools = ["duckduckgo_search"]`
- `startup_timeout_sec = 20`
- `tool_timeout_sec = 30`

Add generated-file assertions for both `.codex-local/config.toml` and `.codex-local-reasoning/config.toml`.

- [ ] **Step 2: Run setup tests to verify failure**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v`

Expected: fail because config generation does not include the DuckDuckGo MCP server.

- [ ] **Step 3: Implement config generation**

Add a helper that formats Windows paths for TOML and appends `[mcp_servers.duckduckgo]` to generated local configs.

- [ ] **Step 4: Run setup tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v`

Expected: pass.

## Task 5: Improve Gemma Instructions And Verification Script

**Files:**
- Modify: `setup_local_codex.py`
- Modify: `tests/test_setup_local_codex.py`

- [ ] **Step 1: Write failing instruction and verify-script tests**

Add tests that assert base instructions include:

- use DuckDuckGo MCP when current internet information is needed
- if internet search is unavailable, say it cannot search the internet right now

Add tests that generated `verify-local-setup.ps1` checks:

```powershell
& $python -c "import mcp"
```

and mentions:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gemma.txt
```

- [ ] **Step 2: Run setup tests to verify failure**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v`

Expected: fail because instructions and verify script do not include these requirements yet.

- [ ] **Step 3: Implement instruction and verify-script updates**

Update `build_base_instructions()` and `write_launchers()` verification script text.

- [ ] **Step 4: Run setup tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v`

Expected: pass.

## Task 6: Regenerate Local Config And Verify Everything

**Files:**
- Generated: `.codex-local/config.toml`
- Generated: `.codex-local/model-catalog.json`
- Generated: `.codex-local-reasoning/config.toml`
- Generated: `.codex-local-reasoning/model-catalog.json`
- Generated: `verify-local-setup.ps1`

- [ ] **Step 1: Run setup helper**

Run: `.\.venv\Scripts\python.exe setup_local_codex.py`

Expected: local configs and launcher files are regenerated.

- [ ] **Step 2: Run full unit suite**

Run: `.\.venv\Scripts\python.exe -m unittest discover -v`

Expected: all tests pass.

- [ ] **Step 3: Run local setup verification**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\verify-local-setup.ps1`

Expected: prints local Codex version and confirms setup files are present.

## Self-Review Checklist

- Spec coverage: local-only scope, official MCP SDK, DuckDuckGo fallback, config generation, dependency management, and tests are all represented.
- Placeholder scan: no TODO/TBD placeholders are present.
- Type consistency: helper and test names match the planned module API.
