from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    default_dataset: str
    executable: str
    official_url: str
    description: str


_SUITES = (
    SuiteSpec(
        name="swe-bench",
        default_dataset="SWE-bench/SWE-bench_Verified",
        executable="python",
        official_url="https://github.com/swe-bench/SWE-bench",
        description="Official SWE-bench harness for real GitHub issue patch evaluation.",
    ),
    SuiteSpec(
        name="terminal-bench",
        default_dataset="terminal-bench-core@0.1.1",
        executable="tb",
        official_url="https://github.com/harbor-framework/terminal-bench",
        description="Terminal-Bench CLI harness for terminal-environment tasks.",
    ),
    SuiteSpec(
        name="terminal-bench-2",
        default_dataset="terminal-bench@2.0",
        executable="harbor",
        official_url="https://github.com/harbor-framework/harbor",
        description="Harbor official harness for Terminal-Bench 2.0.",
    ),
    SuiteSpec(
        name="github-bugs",
        default_dataset="local-manifest",
        executable="gemma-codex.cmd",
        official_url="https://www.swebench.com/",
        description="Local manifest-driven benchmark for fixing GitHub bugs.",
    ),
    SuiteSpec(
        name="local-agent-behavior",
        default_dataset="deterministic-local",
        executable="python",
        official_url="docs/research/composer2-user-provided.pdf",
        description=(
            "Deterministic local behavior checks for stall recovery, terminal discipline, "
            "large-context handling, research citation readiness, and tool reliability."
        ),
    ),
    SuiteSpec(
        name="local-repo-fix",
        default_dataset="generated-local",
        executable="python",
        official_url="docs/research/composer2/Composer2.extracted.txt",
        description=(
            "Generated CPU-only repo-fix benchmark with fail-to-pass, pass-to-pass, "
            "changed-line, regression, and latency metrics."
        ),
    ),
)


def list_suites() -> list[SuiteSpec]:
    return list(_SUITES)


def get_suite(name: str) -> SuiteSpec:
    for suite in _SUITES:
        if suite.name == name:
            return suite
    known = ", ".join(suite.name for suite in _SUITES)
    raise ValueError(f"Unknown benchmark suite {name!r}. Known suites: {known}")
