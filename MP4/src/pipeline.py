#!/usr/bin/env python3
"""Agentic, iterative C-to-Rust translation pipeline (CS598JBR MP4).

    python3 MP4/src/pipeline.py --project 1                 # satc
    python3 MP4/src/pipeline.py --project 2                 # jccc
    python3 MP4/src/pipeline.py --project 2 --resume        # continue after a quota stop
    python3 MP4/src/pipeline.py --project 1 --dry-run       # set up crate + baseline checks, no LLM

Workflow (same code for every project; project-specific data comes from configs/projects.json):

    inspect C project + Rust interface/tests
      -> agent writes/edits Rust (tool calls)
      -> agent runs cargo check / test / clippy / unsafe-metrics itself and reacts
      -> controller runs the full verification (build, tests, 4 Clippy groups, newmetrics,
         interface conformance, unimplemented!/FFI scan) and feeds a prioritised report back
      -> repeat until the stopping condition (see README / configs/config.json):
         success, max iterations, LLM call/token/time budget, or `patience` iterations without improvement.
    The best-scoring version is restored at the end, so a late regression never overwrites a better translation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
MP4_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

# rustup installs cargo/rustup/cargo-newmetrics here; make them visible even if the shell did not source ~/.cargo/env
_CARGO_BIN = Path.home() / ".cargo" / "bin"
if _CARGO_BIN.is_dir() and str(_CARGO_BIN) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = str(_CARGO_BIN) + os.pathsep + os.environ.get("PATH", "")

from agent import Agent, BudgetExceeded  # noqa: E402
from llm import LLMClient, LLMError, LLMFatalError, LLMQuotaError  # noqa: E402
from rust_checks import (CLIPPY_GROUPS, RustChecker, diag_to_dict, format_feedback, headline,  # noqa: E402
                         is_success, score_tuple)
from tools import ToolBox  # noqa: E402
from trajectory import Trajectory  # noqa: E402
from workspace import Workspace  # noqa: E402


# ====================================================================== config
def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if "extends" in cfg:
        base = load_config((path.parent / cfg.pop("extends")).resolve())
        cfg = deep_merge(base, cfg)
    return cfg


def resolve_project(projects: dict, key: str) -> tuple[str, dict]:
    k = key.strip()
    for cand in (k, f"project_{k}"):
        if cand in projects:
            return cand, projects[cand]
    for pk, pv in projects.items():
        if pv["name"] == k:
            return pk, pv
    raise SystemExit(f"unknown project {key!r}; choose from {list(projects)} or names {[p['name'] for p in projects.values()]}")


def load_prompts(cfg: dict) -> dict:
    pdir = (MP4_ROOT / cfg["prompts"]["dir"]).resolve()
    return {k: (pdir / v).read_text(encoding="utf-8") for k, v in cfg["prompts"]["files"].items()}


def fill(template: str, **values) -> str:
    for k, v in values.items():
        template = template.replace("{{" + k + "}}", str(v))
    return template


# ====================================================================== verification
def verify(ws: Workspace, checker: RustChecker, cfg: dict, traj: Trajectory, label: str) -> dict:
    t0 = time.time()
    restored = ws.restore_protected()
    if restored:
        traj.log(f"WARNING: protected files were modified and have been restored: {restored}")
    own = ws.own_files()
    chk = checker.check()
    build_ok = chk["ok"]
    build_out = ""
    if build_ok:
        b = checker.build()
        build_ok, build_out = b["ok"], b["output"]
    if build_ok:
        tests = checker.test()
        clippy = checker.clippy(categorize=True)
    else:
        bench = checker.benchmark_tests()
        tests = {"compiled": False, "timed_out": False, "total": sum(len(v) for v in bench.values()), "passed": 0,
                 "benchmark_status": {f"{f}::{n}": "not compiled" for f, ns in bench.items() for n in ns},
                 "all_results": [], "failed": [], "failure_details": {}, "raw_tail": ""}
        clippy = {"ok": False, "errors": [], "by_group": {}, "own": [], "protected": [], "rustc_own": []}
    safety = checker.unsafe_metrics(own, run_tool=build_ok)
    v = {
        "label": label,
        "build_ok": build_ok,
        "check": {"errors": [diag_to_dict(d) for d in chk["errors"]],
                  "warnings_own": [diag_to_dict(d) for d in chk["warnings"] if not checker.is_protected(d.file)]},
        "build_output_tail": build_out[-3000:],
        "tests": tests,
        "clippy": {k: v for k, v in clippy.items() if k != "errors"},
        "safety": safety,
        "interface": ws.interface_report(),
        "static": safety["lexical"],
        "protected_restored": restored,
        "own_files": own,
    }
    v["score"] = list(score_tuple(v))
    v["success"] = is_success(v, cfg["success_criteria"])
    v["headline"] = headline(v)
    v["duration_s"] = round(time.time() - t0, 1)
    traj.event("verification", label=label, headline=v["headline"], score=v["score"], success=v["success"])
    traj.save_json(f"verification/{label}.json", v)
    traj.log(f"VERIFY [{label}] {v['headline']}  (score={tuple(v['score'])}, success={v['success']}, {v['duration_s']}s)")
    return v


# ====================================================================== prompts
def module_table(ws: Workspace) -> str:
    rows = ["| module | interface (spec) | file you implement | likely related C files |", "|---|---|---|---|"]
    for m in ws.module_map():
        rows.append(f"| {m['module']} | {m['interface']} | {m['target']} | {', '.join(m['c_files_hint']) or '(search c_src/)'} |")
    return "\n".join(rows)


def code_block(path: str, text: str) -> str:
    return f"### {path}\n```rust\n{text.rstrip()}\n```"


def interfaces_text(ws: Workspace) -> str:
    return "\n\n".join(code_block(m["interface"], (ws.spec_dir / "src" / "interfaces" / f"{m['module']}.rs").read_text(encoding="utf-8"))
                       for m in ws.module_map())


def tests_text(ws: Workspace) -> str:
    return "\n\n".join(code_block(t + "  (PROTECTED - read-only)", (ws.crate / t).read_text(encoding="utf-8"))
                       for t in ws.test_files())


def c_sources_text(ws: Workspace) -> str:
    parts = []
    for p in sorted(ws.c_dir.rglob("*")):
        if p.is_file() and p.suffix in (".c", ".h"):
            rel = p.relative_to(ws.c_dir).as_posix()
            parts.append(f"### c_src/{rel}\n```c\n{p.read_text(encoding='utf-8', errors='replace').rstrip()}\n```")
    return "\n\n".join(parts)


def current_rust_text(ws: Workspace, limit: int) -> str:
    files = ws.own_files()
    texts = {f: (ws.crate / f).read_text(encoding="utf-8") for f in files}
    if sum(len(t) for t in texts.values()) > limit:
        return ("(Current Rust files are too large to inline; read them with read_file.)\n" +
                "\n".join(f"- {f} ({len(t.splitlines())} lines)" for f, t in texts.items()))
    return "\n\n".join(code_block(f + "  (current content)", t) for f, t in texts.items())


def history_table(history: list[dict]) -> str:
    rows = ["| iter | build | tests | interface problems | unimplemented | unsafe (primary) | clippy (own) | agent's summary |",
            "|---|---|---|---|---|---|---|---|"]
    for h in history:
        summ = (h.get("agent_summary") or "").replace("\n", " ").replace("|", "/")[:220]
        rows.append(f"| {h['iteration']} | {'OK' if h['build_ok'] else 'FAIL'} | {h['tests_passed']}/{h['tests_total']} | "
                    f"{h['interface_problems']} | {h['unimplemented']} | {h['unsafe_primary']} | {h['clippy_own']} | {summ} |")
    return "\n".join(rows)


def build_messages(ws, cfg, prompts, iteration, last_v, history) -> list[dict]:
    ctx = cfg["context"]
    system = fill(prompts["system"], toolchain=cfg["toolchain"], project_name=ws.name)
    if cfg["llm"].get("tool_mode") == "text":
        system += "\n\n" + prompts["text_protocol"]
    common = dict(
        project_name=ws.name, project_description=ws.cfg.get("description", ""),
        toolchain=cfg["toolchain"], iteration=iteration,
        max_iterations=cfg["budget"]["max_iterations"], module_table=module_table(ws),
        c_tree=ws.file_tree(ws.c_dir, "c_src/"), crate_tree=ws.file_tree(ws.crate, ""),
        interfaces=interfaces_text(ws), tests=tests_text(ws),
        c_sources=(c_sources_text(ws) if ctx.get("inline_c_sources") else
                   "(not inlined - read the C files you need with read_file / search)"),
    )
    if iteration == 1 and not history[1:]:
        user = fill(prompts["initial_task"], **common)
    else:
        user = fill(prompts["repair_task"], **common, history=history_table(history),
                    feedback=format_feedback(last_v, cfg["feedback"]),
                    current_rust=current_rust_text(ws, ctx.get("inline_rust_max_chars", 120000)))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ====================================================================== state & summary
def history_entry(it: int, v: dict, run: dict | None) -> dict:
    return {
        "iteration": it, "build_ok": v["build_ok"], "tests_passed": v["tests"]["passed"], "tests_total": v["tests"]["total"],
        "interface_problems": len(v["interface"]), "unimplemented": v["static"]["unimplemented"],
        "unsafe_primary": v["safety"]["primary_total"],
        "unsafe_metrics": v["safety"].get("metrics"),
        "clippy_own": len(v["clippy"].get("own", [])) if v["build_ok"] else None,
        "clippy_by_group": {g: c["own"] for g, c in v["clippy"].get("by_group", {}).items()},
        "rustc_warnings_own": len(v["clippy"].get("rustc_own", [])) if v["build_ok"] else None,
        "score": v["score"], "success": v["success"], "headline": v["headline"],
        "agent_steps": (run or {}).get("steps"), "agent_stop_reason": (run or {}).get("stop_reason"),
        "agent_summary": (run or {}).get("summary", ""), "files_written": (run or {}).get("files_written", []),
    }


def write_summary(traj: Trajectory, ws: Workspace, cfg: dict, state: dict, final_v: dict, toolinfo: dict) -> None:
    t = final_v["tests"]
    m = final_v["safety"].get("metrics") or {}
    cg = final_v["clippy"].get("by_group", {})
    agent_iters = sum(1 for h in state["history"] if h["iteration"] > 0)
    summary = {
        "project": ws.key, "name": ws.name, "status": state["status"], "stop_reason": state.get("stop_reason"),
        "agent_iterations": agent_iters, "best_iteration": state.get("best_iteration"),
        "llm_calls": state["llm_calls"], "prompt_tokens": state["prompt_tokens"],
        "completion_tokens": state["completion_tokens"], "tool_calls": state["tool_calls"],
        "model": cfg["llm"]["model"], "endpoint": cfg["llm"]["base_url"], "toolchain": toolinfo,
        "final": {
            "compiles": final_v["build_ok"], "tests_passed": t["passed"], "tests_total": t["total"],
            "pass_rate": round(t["passed"] / t["total"], 4) if t["total"] else 0.0,
            "benchmark_status": t["benchmark_status"], "interface_problems": final_v["interface"],
            "unimplemented": final_v["static"]["unimplemented"], "unsafe_tool": final_v["safety"]["tool"],
            "unsafe_metrics": m, "unsafe_primary_total": final_v["safety"]["primary_total"],
            "clippy_own_by_group": {g: c["own"] for g, c in cg.items()},
            "clippy_protected_by_group": {g: c["protected"] for g, c in cg.items()},
            "rustc_warnings_own": len(final_v["clippy"].get("rustc_own", [])),
            "success": final_v["success"],
        },
        "history": state["history"],
    }
    traj.save_json("summary.json", summary)
    md = [f"# {ws.key}: {ws.name} - pipeline summary", "",
          f"- status: **{state['status']}** (stop reason: {state.get('stop_reason')})",
          f"- model: `{cfg['llm']['model']}` via {cfg['llm']['base_url']}",
          f"- agent iterations: {agent_iters} (best = iteration {state.get('best_iteration')}); "
          f"LLM calls: {state['llm_calls']}; tool calls: {state['tool_calls']}; "
          f"tokens: {state['prompt_tokens']} prompt / {state['completion_tokens']} completion",
          f"- toolchain: {toolinfo.get('rustc')} / {toolinfo.get('clippy')}", "",
          "## Final result", "",
          "| Project | Compiles | Tests Passed | Total Tests | Pass Rate | Agent Iterations |", "|---|---|---|---|---|---|",
          f"| {ws.name} | {'Yes' if final_v['build_ok'] else 'No'} | {t['passed']} | {t['total']} | "
          f"{(100 * t['passed'] / t['total']) if t['total'] else 0:.1f}% | {agent_iters} |", "",
          "| Unsafe Lines | Unsafe Calls | Unsafe Casts | Pointer Arithmetic | Raw ptr derefs | Raw ptr decls | tool |",
          "|---|---|---|---|---|---|---|",
          f"| {m.get('unsafe_lines', '-')} | {m.get('unsafe_calls', '-')} | {m.get('unsafe_casts', '-')} | "
          f"{m.get('pointer_arithmetic', '-')} | {m.get('raw_pointer_dereferences', '-')} | "
          f"{m.get('raw_pointer_declarations', '-')} | {final_v['safety']['tool']} |", "",
          "| Clippy (own files) | " + " | ".join(CLIPPY_GROUPS + ["other"]) + " |",
          "|---|" + "---|" * (len(CLIPPY_GROUPS) + 1),
          "| count | " + " | ".join(str(cg.get(g, {}).get("own", "-")) for g in CLIPPY_GROUPS + ["other"]) + " |",
          "| (in protected test files) | " + " | ".join(str(cg.get(g, {}).get("protected", "-")) for g in CLIPPY_GROUPS + ["other"]) + " |",
          "", "## Iterations", "", history_table(state["history"]), ""]
    traj.save_text("summary.md", "\n".join(md))


# ====================================================================== main
def check_llm(llm_cfg: dict) -> int:
    """--check-llm: resolve the model and send one tool-call probe; prints the result, writes nothing."""
    try:
        llm = LLMClient(llm_cfg)
        if llm.model == "auto":
            print(f"model is \"auto\": probing models listed at {llm.base_url}/models ...")
            llm.resolve_model()
            print(f"selected model: {llm.model}")
            return 0
        ok, why = llm.probe(llm.model)
    except LLMError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"{llm.model}: {why}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="1, 2, project_1, project_2, or the CRUST-Bench name")
    ap.add_argument("--config", default=str(MP4_ROOT / "configs" / "config.json"))
    ap.add_argument("--projects", default=str(MP4_ROOT / "configs" / "projects.json"))
    ap.add_argument("--model", help="override llm.model")
    ap.add_argument("--base-url", help="override llm.base_url")
    ap.add_argument("--api-key-env", help="override llm.api_key_env (name of the env var holding the key)")
    ap.add_argument("--tool-mode", choices=["native", "text"], help="override llm.tool_mode")
    ap.add_argument("--max-iterations", type=int, help="override budget.max_iterations")
    ap.add_argument("--resume", action="store_true", help="continue from the existing translation + state")
    ap.add_argument("--dry-run", action="store_true", help="set up the crate and run the baseline checks only")
    ap.add_argument("--check-llm", action="store_true",
                    help="only check the endpoint: resolve the model (for model \"auto\") and send one tool-call probe")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.model:
        cfg["llm"]["model"] = args.model
    if args.base_url:
        cfg["llm"]["base_url"] = args.base_url
    if args.api_key_env:
        cfg["llm"]["api_key_env"] = args.api_key_env
    if args.tool_mode:
        cfg["llm"]["tool_mode"] = args.tool_mode
    if args.max_iterations:
        cfg["budget"]["max_iterations"] = args.max_iterations
    if args.check_llm:
        return check_llm(cfg["llm"])
    projects = json.loads(Path(args.projects).read_text(encoding="utf-8"))
    pkey, pcfg = resolve_project(projects, args.project)

    ws = Workspace(MP4_ROOT, pkey, pcfg, cfg["policy"])
    state_path = ws.traj_dir / "state.json"
    resume = args.resume and state_path.exists()
    if not resume and not args.dry_run and ws.traj_dir.exists():
        # fresh run: archive nothing, start a clean trajectory directory
        shutil.rmtree(ws.traj_dir)
    traj = Trajectory(ws.traj_dir)
    traj.log(f"===== pipeline start: {pkey} ({ws.name}) resume={resume} dry_run={args.dry_run} =====")
    traj.event("run_start", project=pkey, name=ws.name, resume=resume, dry_run=args.dry_run,
               config={k: v for k, v in cfg.items() if k != "prompts"}, argv=sys.argv[1:])
    if not args.dry_run:
        traj.save_json("config_used.json", cfg)

    for act in ws.init(fresh=not resume):
        traj.log(f"setup: {act}")
    checker = RustChecker(ws.crate, cfg["toolchain"], cfg["tools"], ws.protected_files(), traj)
    toolinfo = checker.toolchain_info()
    traj.log(f"toolchain: {json.dumps(toolinfo)}")
    traj.event("toolchain", **toolinfo)
    if "UNAVAILABLE" in toolinfo["cargo"]:
        traj.log("ERROR: cargo +%s not available. Run MP4/scripts/setup_colab.sh first." % cfg["toolchain"])
        return 1

    if resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        traj.iteration = state["next_iteration"] - 1
        last_v = verify(ws, checker, cfg, traj, f"resume_{state['next_iteration']:02d}")
    else:
        state = {"next_iteration": 1, "history": [], "best_score": None, "best_iteration": None, "no_improve": 0,
                 "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "tool_calls": 0,
                 "status": "running", "stop_reason": None, "elapsed_s": 0.0}
        last_v = verify(ws, checker, cfg, traj, "iter_00_baseline")
        state["history"].append(history_entry(0, last_v, None))
        ws.snapshot(ws.traj_dir / "snapshots" / "iter_00")
        ws.snapshot(ws.traj_dir / "best")
        state["best_score"], state["best_iteration"] = last_v["score"], 0
    if args.dry_run:
        traj.log("dry run finished (no LLM calls).")
        return 0

    prompts = load_prompts(cfg)
    tool_schemas = json.loads((MP4_ROOT / cfg["tools"]["schema_file"]).read_text(encoding="utf-8"))
    try:
        llm = LLMClient(cfg["llm"], traj)
        if llm.model == "auto":
            # keep the model chosen at the start of this run when resuming
            if state.get("model"):
                llm.model = state["model"]
            else:
                llm.resolve_model()
    except LLMError as exc:
        traj.log(f"ERROR: {exc}")
        return 1
    cfg["llm"]["model"] = state["model"] = llm.model
    traj.log(f"model: {llm.model} via {llm.base_url}")
    traj.save_json("config_used.json", cfg)
    toolbox = ToolBox(ws, checker, cfg["tools"], traj)
    agent = Agent(llm, toolbox, tool_schemas, cfg, traj, prompts)
    budget = cfg["budget"]
    t_start = time.time() - state.get("elapsed_s", 0.0)
    calls0, ptok0, ctok0, tool0 = state["llm_calls"], state["prompt_tokens"], state["completion_tokens"], state["tool_calls"]

    def sync_counters():
        state["llm_calls"] = calls0 + llm.calls
        state["prompt_tokens"] = ptok0 + llm.prompt_tokens
        state["completion_tokens"] = ctok0 + llm.completion_tokens
        state["tool_calls"] = tool0 + toolbox.calls
        state["elapsed_s"] = round(time.time() - t_start, 1)

    # budgets are global across resumes
    budget_local = dict(budget)
    budget_local["max_llm_calls"] = max(0, budget.get("max_llm_calls", 10 ** 9) - calls0)
    budget_local["max_total_tokens"] = max(0, budget.get("max_total_tokens", 10 ** 12) - ptok0 - ctok0)
    agent.budget = budget_local

    stop_reason = None
    if last_v["success"]:
        stop_reason = "success (already passing)"
    it = state["next_iteration"]
    while stop_reason is None:
        if it > budget["max_iterations"]:
            stop_reason = f"max_iterations ({budget['max_iterations']}) reached"
            break
        if (time.time() - t_start) / 60 > budget.get("wall_clock_limit_min", 10 ** 6):
            stop_reason = f"wall-clock limit ({budget['wall_clock_limit_min']} min) reached"
            break
        traj.iteration = it
        traj.log(f"----- iteration {it}/{budget['max_iterations']} -----")
        messages = build_messages(ws, cfg, prompts, it, last_v, state["history"])
        traj.event("iteration_start", prompt_chars=sum(len(m["content"]) for m in messages),
                   user_prompt=messages[1]["content"])
        run, interrupted, soft_error = None, None, None
        try:
            run = agent.run_iteration(messages, budget["max_steps_per_iteration"])
            state["llm_errors_in_a_row"] = 0
        except BudgetExceeded as exc:
            interrupted = f"budget exhausted: {exc}"
        except LLMQuotaError as exc:
            interrupted = f"LLM quota/rate limit: {exc}"
        except LLMFatalError as exc:
            interrupted = f"LLM error (not retryable): {exc}"
        except LLMError as exc:
            # a single bad request ends this iteration only; the next one starts from a fresh conversation
            state["llm_errors_in_a_row"] = state.get("llm_errors_in_a_row", 0) + 1
            soft_error = f"LLM error: {exc}"
            if state["llm_errors_in_a_row"] >= budget.get("max_consecutive_llm_errors", 3):
                interrupted = f"{soft_error} ({state['llm_errors_in_a_row']} iterations in a row)"
        except KeyboardInterrupt:
            interrupted = "interrupted by user"
        traj.save_messages(it, messages)
        if run is None:
            run = {"steps": None, "stop_reason": interrupted or soft_error, "summary": toolbox.finish_summary,
                   "files_written": sorted(toolbox.files_written)}
        toolbox.files_written = set()
        traj.log(f"agent iteration ended: {run['stop_reason']} after {run['steps']} steps; summary: {run['summary'][:300]}")
        v = verify(ws, checker, cfg, traj, f"iter_{it:02d}")
        ws.snapshot(ws.traj_dir / "snapshots" / f"iter_{it:02d}")
        state["history"].append(history_entry(it, v, run))
        if tuple(v["score"]) > tuple(state["best_score"]):
            state["best_score"], state["best_iteration"], state["no_improve"] = v["score"], it, 0
            ws.snapshot(ws.traj_dir / "best")
            traj.log(f"new best version (iteration {it})")
        else:
            state["no_improve"] += 1
            traj.log(f"no improvement over best (iteration {state['best_iteration']}) for {state['no_improve']} iteration(s)")
        last_v = v
        it += 1
        state["next_iteration"] = it
        sync_counters()
        traj.save_json("state.json", state)
        if v["success"]:
            stop_reason = "success: all tests pass, interface conformant, zero unsafe, Clippy target met"
        elif interrupted:
            stop_reason = interrupted
        elif state["no_improve"] >= budget.get("patience", 3):
            stop_reason = f"no progress for {state['no_improve']} iterations (patience)"

    traj.iteration = it - 1
    traj.log(f"STOP: {stop_reason}")
    # ---------------------------------------------------------------- finalisation
    if tuple(last_v["score"]) < tuple(state["best_score"]):
        traj.log(f"restoring best version from iteration {state['best_iteration']}")
        ws.restore_snapshot(ws.traj_dir / "best")
        last_v = verify(ws, checker, cfg, traj, "final_restored_best")
    if cfg["policy"].get("format_final", True) and last_v["build_ok"]:
        pre = ws.traj_dir / "pre_format"
        ws.snapshot(pre)
        edition = "2021"
        for line in (ws.crate / "Cargo.toml").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("edition"):
                edition = line.split("=")[1].strip().strip('"')
        roots = [r for r in ("src/lib.rs", "src/main.rs") if (ws.crate / r).exists()]
        r = checker.run(["rustfmt", f"+{cfg['toolchain']}", "--edition", edition, *roots], 120)
        if r.returncode == 0:
            v2 = verify(ws, checker, cfg, traj, "final_formatted")
            if tuple(v2["score"]) >= tuple(last_v["score"]):
                last_v = v2
                traj.log("applied rustfmt to the final translation (formatting only)")
            else:
                ws.restore_snapshot(pre)
                last_v = verify(ws, checker, cfg, traj, "final")
        shutil.rmtree(pre, ignore_errors=True)
    sync_counters()
    state["status"] = "success" if last_v["success"] else ("stopped" if "budget" in (stop_reason or "") or "quota" in (stop_reason or "") else "finished")
    state["stop_reason"] = stop_reason
    traj.save_json("state.json", state)
    traj.save_json("verification/final.json", last_v)
    write_summary(traj, ws, cfg, state, last_v, toolinfo)
    traj.event("run_end", status=state["status"], stop_reason=stop_reason, headline=last_v["headline"])
    traj.log(f"FINAL: {last_v['headline']}")
    traj.log(f"summary written to {ws.traj_dir.relative_to(MP4_ROOT)}/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
