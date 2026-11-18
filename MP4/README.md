# MP4: Agentic C-to-Rust Translation (Team 12)

| | Project | CRUST-Bench name | C source | Rust output |
|---|---|---|---|---|
| project_1 | satc | `satc`: Separating Axis Theorem 2D collision detection (single header) | `subjects/project_1/c/` | `translations/project_1/` |
| project_2 | jccc | `jccc`: JabaCat C Compiler (lexer, list, hashmap, parser, x86 codegen) | `subjects/project_2/c/` | `translations/project_2/` |

We wrote our own agent controller in Python (`src/`). An LLM, reached through any OpenAI-compatible chat API, drives a tool-calling loop. The tools let it read the C project, write and edit Rust, and run `cargo check/build/test/clippy` and the course's unsafe-metrics tool. After each agent iteration, the controller runs a full verification and returns a prioritized report to the agent. This repeats until the stopping condition is met.

## Environment

| Item | Value |
|---|---|
| Agent framework + version | Custom controller `c2rust-agent` v1.0 (this repo, `MP4/src/`): a ReAct-style tool-calling loop over the OpenAI `/chat/completions` protocol. No third-party agent framework. |
| Model(s) | Several free options; only one model per run (single agent). **(A) Self-hosted open-weight model on the Colab GPU:** `qwen3:8b` (Qwen3-8B, Q4_K_M) served by **Ollama** through its OpenAI-compatible API, 32k context, `configs/config.ollama.json` (`qwen3:14b` on L4/A100). **(B) Free hosted open-weight model on NVIDIA's API catalog** at `https://integrate.api.nvidia.com/v1` with a free developer key (`configs/config.nvidia.json`, used by `MP4_colab.ipynb`). The free catalog changes (Qwen3-Coder-480B-A35B-Instruct was retired on 2026-06-11 and returned HTTP 410 in our first attempt), so the config sets `model: "auto"`: `--check-llm` lists the endpoint's models, walks `model_preferences` (Qwen3-Coder, Kimi-K2, DeepSeek-V3, GLM, MiniMax, Qwen3, gpt-oss-120b, ...) and keeps the first model that answers a tool-call probe; the notebook then pins it with `--model` for both projects. **(C) Other free hosted endpoints:** `qwen/qwen3-coder:free` on **OpenRouter** (`configs/config.json`), or `gemini-2.5-flash` free tier (`configs/config.gemini.json`). A vLLM variant is in `configs/config.local_vllm.json`. **Model used for the submitted runs:** recorded in `trajectories/project_*/summary.json` (`model`, `endpoint`) and in every `llm_response` event of `trajectory.jsonl`. |
| Inference settings | temperature 0.2, top_p 0.95, max_tokens 16384 per response (8192 and 30 min timeout for local Ollama), seed 598. For the local Qwen3 model, thinking mode is off (`/no_think` appended to the system prompt, `system_prompt_suffix`): on a T4 it generates about 10 tokens/s and a thinking step took up to 15 minutes in our first run. Other settings: `tool_choice=auto`, native function calling (falls back automatically to a text tool protocol if the endpoint rejects `tools`). Request timeout 600 s, up to 8 retries with exponential backoff on 429/5xx, ≥4 s between requests. |
| OS / runtime | Google Colab (Ubuntu 22.04, x86_64; T4 GPU runtime for the local Ollama model, CPU runtime is enough for a hosted model), Python 3.10+, `requests`. Rust toolchain **nightly-2025-05-13** (rustup) with clippy, rustfmt, rustc-dev, llvm-tools-preview. |
| Tools available to the agent | `list_files`, `read_file`, `search` (regex) over the C project (`c_src/`, read-only), the benchmark spec (`rust_spec/`, read-only) and the crate; `write_file`, `append_file`, `edit_file` (only `src/**/*.rs`, not tests/manifest); `cargo_check` (`cargo +nightly-2025-05-13 check --all-targets`), `cargo_build`, `cargo_test`, `cargo_clippy` (the exact grading command), `unsafe_metrics` (course `newmetrics` tool + lexical scan), `check_interface` (signature/struct/enum/const conformance against the CRUST-Bench interface), `finish`. Schemas: `configs/tools.json`. |
| API / usage limits | Local Ollama: no API limits; bounded by the Colab GPU session length (runs are resumable with `--resume`) and T4 speed. NVIDIA API catalog free tier: about 40 requests/min per account, no daily cap; the config keeps ≥2 s between requests, 500 LLM calls and 240 min per project. OpenRouter free models: about 20 requests/min, and 50 requests/day (1000/day once the account has ≥$10 credit). Gemini free tier: per-model RPM/RPD limits. The pipeline throttles requests, retries on 429, and on quota exhaustion saves its state so the run continues with `--resume`. Budgets per project are set in `configs/config.json` (300 LLM calls, 15M tokens, 240 min). *Limits actually hit during our runs:* see `trajectories/project_*/execution.log` (lines `LLM request retry ...` / `STOP: LLM quota ...`). |

## Models, frameworks and tools used

- **Model:** one LLM per run, the agent and repair model, selected in `configs/config*.json`: an open-weight model on NVIDIA's API catalog picked by `--check-llm` (exact id in `trajectories/project_*/summary.json`), `qwen3:8b` via Ollama on the Colab GPU, or `qwen/qwen3-coder:free` on OpenRouter. No second model, no multi-agent setup.
- **Local model runtime:** Ollama (`scripts/setup_ollama.sh`), OpenAI-compatible endpoint `http://localhost:11434/v1`, with flash attention and a q8_0 KV cache to fit 32k context on a T4.
- **Agent framework:** our own Python controller (`src/pipeline.py`, `src/agent.py`, `src/tools.py`, `src/rust_checks.py`, `src/workspace.py`, `src/llm.py`, `src/trajectory.py`), using the OpenAI-compatible chat-completions + function-calling protocol over HTTP (`requests`).
- **External tools the agent invokes:** `cargo +nightly-2025-05-13 check | build | test | clippy`, `rustfmt` (final formatting only), the course's unsafe-metrics tool `cargo newmetrics` (`MP4/newmetrics/`, aggregated like `aggregate.py`), file read/search/write/edit tools, and our interface-conformance checker.
- **Benchmark:** CRUST-Bench (`anirudhkhatry/CRUST-bench`, commit `4b47025`), using `CBench/{satc,jccc}` and `RBench/{satc,jccc}`, copied unmodified into `subjects/`.

## Repository layout

```
MP4/
├── README.md
├── MP4_colab.ipynb          # notebook that runs everything on Colab
├── src/                     # pipeline
│   ├── pipeline.py          # controller: setup, iterations, verification, stopping, best-version restore, summary
│   ├── agent.py             # LLM tool-calling loop (native function calling or text protocol), context compaction
│   ├── tools.py             # tool implementations (sandboxed file access + cargo/metrics wrappers)
│   ├── rust_checks.py       # cargo check/build/test/clippy(4 groups)/newmetrics runners + parsers, interface checker, scoring, feedback
│   ├── workspace.py         # creates the crate from the CRUST-Bench spec, protected files, snapshots
│   ├── llm.py               # OpenAI-compatible client (retries, throttling, token accounting)
│   ├── trajectory.py        # trajectory logging + secret redaction
│   └── evaluate.py          # independent final evaluation -> report tables
├── configs/
│   ├── config.json          # model/endpoint, inference settings, budgets & stopping condition, policies, feedback limits
│   ├── config.gemini.json   # variant: Gemini (OpenAI-compatible endpoint)
│   ├── config.ollama.json   # variant: free open-weight model on the Colab GPU (Ollama)
│   ├── config.nvidia.json   # variant: free hosted open-weight model on NVIDIA's API catalog (auto-selected)
│   ├── config.local_vllm.json  # variant: self-hosted open-weight model (vLLM)
│   ├── projects.json        # project_1 = satc, project_2 = jccc (paths)
│   ├── tools.json           # tool (function) schemas given to the model
│   └── prompts/             # system.md, initial_task.md, repair_task.md, text_tool_protocol.md, nudge.md, truncated_nudge.md
├── scripts/                 # setup_colab.sh, setup_ollama.sh, run_pipeline.sh, run_all.sh
├── subjects/project_{1,2}/  # c/ = original C project, rust_spec/ = CRUST-Bench interface + tests
├── translations/project_{1,2}/  # final Rust crates (cargo build/test here)
├── trajectories/project_{1,2}/  # execution logs (see "Outputs")
└── newmetrics/              # unsafe-metrics tool provided by the course
```

## Setup

On Colab (or any Ubuntu machine):

```bash
git clone https://github.com/picolimpid/CS598JBR-Team-12.git && cd CS598JBR-Team-12
bash MP4/scripts/setup_colab.sh        # rustup + nightly-2025-05-13 (+clippy, rustfmt, rustc-dev, llvm-tools) + newmetrics + requests
export PATH="$HOME/.cargo/bin:$PATH"
# model, option A (free, local GPU, no key):
bash MP4/scripts/setup_ollama.sh qwen3:8b     # then add --config MP4/configs/config.ollama.json to the commands below
# model, option B (free hosted, NVIDIA API catalog): export NVIDIA_API_KEY=nvapi-...   (never commit keys)
# model, option C (free hosted, OpenRouter): export OPENROUTER_API_KEY=...
```

On Colab, open `MP4/MP4_colab.ipynb` (option B, CPU runtime is enough), add the key as a Colab *Secret* named `NVIDIA_API_KEY` and run the cells. The notebook backs up `translations/` and `trajectories/` to Google Drive every 5 minutes, so a disconnected run can continue with `--resume`.

## Run (reproduce)

```bash
python3 MP4/src/pipeline.py --project 1                  # satc  -> translations/project_1, trajectories/project_1
python3 MP4/src/pipeline.py --project 2                  # jccc  -> translations/project_2, trajectories/project_2
python3 MP4/src/pipeline.py --project 2 --resume         # continue after a quota/rate-limit stop
python3 MP4/src/pipeline.py --project 1 --dry-run        # only build the crate skeleton + baseline checks (no LLM)
python3 MP4/src/pipeline.py --project 1 --config MP4/configs/config.nvidia.json --check-llm   # pick a model on NVIDIA's API catalog
python3 MP4/src/pipeline.py --project 1 --config MP4/configs/config.nvidia.json --model <picked id>
python3 MP4/src/pipeline.py --project 1 --config MP4/configs/config.ollama.json   # local Ollama model on the GPU
python3 MP4/src/pipeline.py --project 1 --config MP4/configs/config.gemini.json   # other provider
python3 MP4/src/pipeline.py --project 1 --model <id> --max-iterations 6           # quick overrides
bash MP4/scripts/run_all.sh                              # both projects + final evaluation
python3 MP4/src/evaluate.py                              # final tables (build/test/clippy-by-group/newmetrics)
```

A fresh run (without `--resume`) deletes and recreates `translations/project_N/` and `trajectories/project_N/`.

## Pipeline

```
            ┌──────────────────────── controller (pipeline.py) ────────────────────────┐
 subjects/  │ 1. setup: copy Cargo.toml, tests (protected), seed src/<mod>.rs with the  │
 c/ + spec ─┼──▶   CRUST-Bench interface, add `pub mod` + #![forbid(unsafe_code)]        │
            │ 2. baseline verification (iteration 0)                                    │
            │ 3. for iteration = 1..N:                                                  │
            │      prompt = initial task (it. 1) | repair task + verification report    │
            │      ┌──────────── agent loop (agent.py, ≤ 30 LLM steps) ───────────┐     │
            │      │ LLM ─▶ tool call ─▶ tools.py ─▶ result ─▶ LLM ─▶ ...         │     │
            │      │ read C / spec, write/edit Rust, cargo check/test/clippy,      │     │
            │      │ unsafe_metrics, check_interface ... until `finish`            │     │
            │      └───────────────────────────────────────────────────────────────┘     │
            │      full verification: check+build ─▶ test ─▶ clippy ×5 (4 groups)       │
            │        ─▶ newmetrics ─▶ interface conformance ─▶ unimplemented!/FFI scan  │
            │      score, snapshot, keep best, stop?                                    │
            │ 4. restore best version, rustfmt, final verification, summary            │
            └───────────────────────────────────────────────────────────────────────────┘
```

* **Inspect:** the first prompt gives the C file tree, the module → interface → C-file mapping, the full Rust interface and the benchmark tests. The agent reads the C sources itself with `read_file`/`search`. This shows up as tool calls in the trajectory.
* **Generate/modify:** `write_file` / `append_file` (large files in chunks) / `edit_file`. Every write is checked against the interface right away, and the result reports any signature drift.
* **Compile/test/lint/safety:** the agent calls the cargo tools itself inside an iteration. Compiler errors come back as rustc's rendered diagnostics (code, span, help). Test results come back per benchmark test, with panic or assertion messages. Clippy comes back as rendered lint messages in the agent's files; warnings in the protected test files are only counted. The `newmetrics` counters are returned together with the file:line of every `unsafe`, raw-pointer cast, `transmute`, `ptr::` or FFI use.
* **Controller feedback:** after each iteration the controller runs everything again and builds a report ordered by priority: compilation errors → interface → unimplemented → failing tests → safety → Clippy → rustc warnings. It names a PRIMARY FOCUS, includes the current Rust sources and a per-iteration progress table, and starts the next iteration with a fresh context.
* **Scoring and best version:** the score is (compiles, tests passed, −unimplemented, −interface problems, −unsafe, −Clippy, −rustc warnings), compared lexicographically. The best snapshot is kept, and restored at the end if a later iteration regressed.
* **Policies (configs/config.json → `policy`):** `#![forbid(unsafe_code)]` is injected into the crate roots, so any unsafe becomes a compile error the agent must repair. Tests, `Cargo.toml` and `Cargo.lock` are write-protected and hash-checked. Writes are restricted to `src/**/*.rs`. `rustfmt` runs on the agent's files after the loop; this only changes formatting.
* **Reusability:** nothing is project-specific in code. Projects are described only by paths in `configs/projects.json`; module names, the files to seed, the protected tests and the C-file hints are all derived from the CRUST-Bench layout.

### Stopping condition

Each project run stops at the first of:
1. **success**: it compiles, all benchmark tests pass, no interface problems, no `unimplemented!`/`todo!`, all primary unsafe metrics are 0, no FFI, and 0 Clippy warnings in the generated files (`success_criteria`);
2. `max_iterations` = 10 agent iterations (each ≤ 30 LLM steps);
3. budget exhausted: 300 LLM calls, 15M tokens, or 240 minutes of wall clock (local Ollama config: 600 calls, 40M tokens, 180 minutes);
4. no progress: 3 consecutive iterations without a better score (`patience`);
5. repeated request failures: 3 iterations in a row ending in an LLM request error (`max_consecutive_llm_errors`). A single failed request ends only the current iteration; the next one starts from a fresh conversation built from the latest verification report.

On stop, the best-scoring version is what remains in `translations/project_N/`.

## Outputs

`trajectories/project_N/` contains:
- `execution.log`: human-readable log of every LLM step, tool call (with result head), file write (+/− lines), command and verification.
- `trajectory.jsonl`: one JSON event per line: `llm_response` (content, tool calls, reasoning if provided, token usage, latency), `tool_call` (arguments + result), `file_write` (unified diff), `command` (cargo command, exit code, output), `tool_feedback`, `verification`, `iteration_start` (full prompt).
- `llm_messages/iter_XX.json`: the complete conversation for each iteration (system prompt, task/repair prompt, assistant tool calls, tool results).
- `verification/iter_XX.json`, `final.json`: structured build/test/Clippy/unsafe/interface results per iteration.
- `snapshots/iter_XX/`: the Rust sources at the end of each iteration; `best/`: the best version.
- `summary.md` / `summary.json`: final result tables and per-iteration progression; `config_used.json`: the exact configuration; `final_evaluation.json`, `newmetrics_summary.json` from `evaluate.py`.

API keys are never stored. All log text passes through a redactor (`trajectory.py`) as well.

## Evaluation

Graders' commands, run inside `translations/project_N/`:

```bash
cargo +nightly-2025-05-13 build
cargo +nightly-2025-05-13 test
cargo +nightly-2025-05-13 clippy --all-targets --all-features -- -W clippy::correctness -W clippy::style -W clippy::complexity -W clippy::perf
RUSTUP_TOOLCHAIN=nightly-2025-05-13 cargo newmetrics 2>&1 | python3 ../../newmetrics/aggregate.py --out summary.json
```

`python3 MP4/src/evaluate.py` runs all of these for both projects. It counts each Clippy group by enabling one group at a time (`-A clippy::all -W clippy::<group>`) and writes the report tables to `trajectories/evaluation_tables.md`.

## Manual interventions

No program logic was written or edited by hand. Humans only wrote the pipeline, configs and prompts, copied the benchmark files into `subjects/`, and launched the runs. Any infrastructure-level intervention (for example resuming after a quota stop) is visible in `execution.log` as a new `pipeline start ... resume=True` block.
