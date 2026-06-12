# Gemma DuckDuckGo MCP Design

## Goal

Give the local Gemma Codex setup a DuckDuckGo internet search capability through a local Python MCP server. The capability must be available only when Gemma is launched through `son`, `sonion`, or `operator`, and it must not modify or depend on the normal global Codex configuration.

## Scope

The feature applies to the two project-local Codex homes generated in this folder:

- `.codex-local`
- `.codex-local-reasoning`

The feature does not change `%USERPROFILE%\.codex\config.toml`, global `codex`, global MCP servers, or global model settings.

## Architecture

Add a local Python stdio MCP server named `duckduckgo_mcp.py`. The server uses the official MCP Python SDK through `mcp.server.fastmcp.FastMCP`, pinned to `mcp==1.27.2`, which was verified as the latest available MCP package for this environment on June 11, 2026. The MCP server process is started by Codex from the Gemma-local `config.toml` files.

DuckDuckGo search itself stays dependency-light. The server uses Python standard library HTTP and HTML parsing against DuckDuckGo's non-JavaScript HTML search page. This keeps the only new dependency focused on the MCP protocol layer instead of search scraping logic.

## Tool Interface

The MCP server exposes one tool:

- `duckduckgo_search(query: str, max_results: int = 5) -> str`

`query` is required. `max_results` defaults to 5 and is capped at 10. The returned text is compact and includes numbered results with title, URL, and snippet when available.

## Data Flow

1. Gemma determines that current internet information is needed.
2. Codex calls the local `duckduckgo_search` MCP tool.
3. `duckduckgo_mcp.py` requests DuckDuckGo HTML search results.
4. The server parses titles, result URLs, and snippets.
5. The server returns compact source-bearing results to Codex.
6. Gemma uses those results in its answer.

## Internet-Unavailable Behavior

If DNS, HTTP, TLS, timeout, parsing, or connectivity fails, the tool returns this brief message:

`Internet search is unavailable right now, so I cannot verify current web information from DuckDuckGo.`

If DuckDuckGo is reachable but produces no parsed results, the tool returns a brief no-results message instead of claiming the internet is unavailable.

The Gemma model instructions will also tell the model to use the DuckDuckGo MCP tool for current web facts when available, and to say it cannot search the internet right now when the tool is unavailable.

## Configuration

`setup_local_codex.py` generates `[mcp_servers.duckduckgo]` into both Gemma-local `config.toml` files. The server command points to this folder's `.venv\Scripts\python.exe`, and the first argument points to `duckduckgo_mcp.py`. The MCP server is enabled, scoped to the `duckduckgo_search` tool, and uses reasonable startup and tool timeouts.

The generated config is local because `son`, `sonion`, and `operator` launch through `gemma-codex.cmd`, which sets `CODEX_HOME` to one of the Gemma-local homes.

## Dependency Management

Add a local requirements file for the MCP dependency:

- `requirements-gemma.txt` with `mcp==1.27.2`

The current environment should install it into `.venv` during implementation. The verification script should check that `mcp` is importable and tell the user how to install local requirements if it is missing.

## Testing

Tests cover:

- DuckDuckGo HTML parsing of title, URL, and snippet
- result formatting and `max_results` capping
- internet failure fallback message
- generated direct and reasoning configs include only the local DuckDuckGo MCP server
- generated verification script checks the MCP dependency
- existing launcher, setup, and proxy behavior still passes

## Constraints

This folder is not currently a git repository. The spec and implementation plan can be written locally, but they cannot be committed unless a git repository is initialized or selected later.

## Sources

- OpenAI Codex config reference: https://developers.openai.com/codex/config-reference
- Official MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- PyPI `mcp` package: https://pypi.org/project/mcp/
- DuckDuckGo search syntax/help: https://duckduckgo.com/duckduckgo-help-pages/results/syntax/
