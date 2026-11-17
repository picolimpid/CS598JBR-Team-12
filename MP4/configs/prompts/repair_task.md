# Repair iteration {{iteration}} of at most {{max_iterations}} for `{{project_name}}`

{{project_description}}

You are continuing an ongoing C-to-Rust translation. After the previous iteration the controller ran the full verification (cargo build, cargo test, Clippy with the four graded groups, the unsafe-metrics tool, interface conformance and an unimplemented!/FFI scan). Its report is below. Repair the problems in the order of the PRIMARY FOCUS, then continue down the list. Do not regress what already works: re-run `cargo_test` after changes.

## Progress so far
{{history}}

## Verification report of the current code
{{feedback}}

## Modules
{{module_table}}

## Current Rust sources
{{current_rust}}

## Required Rust interface (specification)
{{interfaces}}

## Benchmark tests (protected)
{{tests}}

## Reminders
- The original C code is under `c_src/` (read-only): re-read the relevant C function whenever a test fails, to match its exact behaviour (formulas, loop bounds, edge cases, return codes).
- Keep every interface signature, all functions implemented, no `unsafe`, no raw pointers, no hard-coded test answers, no `#[allow(clippy::...)]` unless the lint is forced by the mandated interface.
- Typical fixes for Clippy: `needless_range_loop` -> iterate with `iter()/enumerate()/zip()`; `new_without_default` -> add `impl Default`; `needless_return` -> tail expression; `redundant_clone`/`clone_on_copy` -> borrow or copy; `manual_memcpy` -> `copy_from_slice`; `len_zero` -> `is_empty()`; `collapsible_if`/`collapsible_else_if` -> merge conditions; `legacy_numeric_constants` -> `f64::MAX`; unused imports -> delete them.
- Finish with `finish` (summary of what changed and what still fails).
