---
name: code-simplifier
description: Simplify recently changed code for clarity, consistency, and maintainability without changing behavior. Use when Codex is asked to refactor messy code, reduce nesting, improve naming, remove redundancy, or make implementation details easier to follow while preserving functionality.
---

# Code Simplifier

Improve clarity without changing what the code does.

## Workflow

1. Start with recently modified code or the files the user points at.
2. Preserve public behavior, interfaces, and side effects unless the user explicitly asks for broader refactoring.
3. Make the code easier to read in fewer jumps and with fewer hidden assumptions.
4. Verify with tests or targeted checks when possible.

## Simplification Moves

- reduce unnecessary nesting
- extract helpers when the new boundary is obvious and useful
- remove dead branches, duplication, and redundant state
- prefer explicit control flow over dense ternaries or clever one-liners
- rename variables and functions to match their real role
- align with the repo's existing style instead of imposing a new one

## Guardrails

- Do not refactor unrelated areas just because they can be improved.
- Do not trade readability for abstraction.
- Do not change behavior in the name of cleanup.
- Keep comments only when they explain non-obvious intent.

## Output

When asked to make changes, implement the simplification directly. When asked to review first, describe the highest-value simplifications and the risks of touching them.
