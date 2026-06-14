# Gemma Release-Readiness Work Plan

> Controller file for the current public-readiness pass. Detailed specs and task plans live under `docs/superpowers/`.

## Goal

Tidy the workspace, replace MVP-shaped runtime structure with a stronger Gemma-owned agent architecture, then harden the resulting product surface for a credible public-release trajectory.

## Current Direction

1. Controlled big-bang rewrite of the agent runtime core.
2. Launch, proxy, and Codex command layer rebuild over the new runtime.
3. Workspace discipline, `.superpowers` evidence structure, release tidiness, and public-claim gates.

This is not a blind rewrite. The old system remains a behavior oracle where it is proven correct, and the new system replaces internals behind compatibility adapters, tests, benchmarks, Codex 0.139 public-behavior comparison, and leak/security gates.

## Non-Goals For This Pass

- No npm package publishing.
- No public git installer.
- No hosted multi-tenant service claim.
- No model-weight or uncensored-behavior change.

## Verification Gate

Every behavior change gets a failing test first, a focused passing test after implementation, and a final full-suite pass before any completion claim.

## Active Spec And Plan

- Design: `docs/superpowers/specs/2026-06-14-gemma-controlled-big-bang-rebuild-design.md`
- Implementation plan: pending user review of the design spec
