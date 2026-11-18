# project_2: jccc - pipeline summary

- status: **success** (stop reason: success: all tests pass, interface conformant, zero unsafe, Clippy target met)
- model: `moonshotai/kimi-k3.1-pro` via https://integrate.api.nvidia.com/v1
- agent iterations: 1 (best = iteration 1); LLM calls: 30; tool calls: 69; tokens: 1461338 prompt / 42148 completion
- toolchain: rustc 1.89.0-nightly (8405332bd 2025-05-12) / clippy 0.1.88 (8405332bdf 2025-05-12)

## Final result

| Project | Compiles | Tests Passed | Total Tests | Pass Rate | Agent Iterations |
|---|---|---|---|---|---|
| jccc | Yes | 4 | 4 | 100.0% | 1 |

| Unsafe Lines | Unsafe Calls | Unsafe Casts | Pointer Arithmetic | Raw ptr derefs | Raw ptr decls | tool |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | newmetrics |

| Clippy (own files) | correctness | style | complexity | perf | other |
|---|---|---|---|---|---|
| count | 0 | 0 | 0 | 0 | 0 |
| (in protected test files) | 0 | 4 | 0 | 0 | 0 |

## Iterations

| iter | build | tests | interface problems | unimplemented | unsafe (primary) | clippy (own) | agent's summary |
|---|---|---|---|---|---|---|---|
| 0 | OK | 0/4 | 0 | 56 | 0 | 0 |  |
| 1 | OK | 4/4 | 0 | 0 | 0 | 0 |  |
