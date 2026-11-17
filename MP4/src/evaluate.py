#!/usr/bin/env python3
"""Independent final evaluation of the submitted translations (for the Task 4 report).

    python3 MP4/src/evaluate.py            # both projects
    python3 MP4/src/evaluate.py --project 2

For each project in configs/projects.json it runs, on MP4/translations/project_<N>/ with
nightly-2025-05-13: cargo build, cargo test, the grading Clippy command split into the four
groups, and the course's newmetrics tool (`cargo newmetrics | aggregate.py`). Results go to
MP4/trajectories/project_<N>/final_evaluation.json and the Markdown tables are printed and saved
to MP4/trajectories/evaluation_tables.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
MP4_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from pipeline import load_config, resolve_project  # noqa: E402
from rust_checks import CLIPPY_GROUPS, RustChecker  # noqa: E402
from workspace import Workspace  # noqa: E402


def official_newmetrics(crate: Path, toolchain: str, out_json: Path) -> dict | None:
    """Exactly the documented command: cargo newmetrics 2>&1 | python3 aggregate.py --out summary.json"""
    if not RustChecker.newmetrics_path():
        return None
    agg = MP4_ROOT / "newmetrics" / "aggregate.py"
    cmd = f'cargo newmetrics 2>&1 | "{sys.executable}" "{agg}" --out "{out_json}"'
    env = dict(os.environ, RUSTUP_TOOLCHAIN=toolchain,
               CARGO_TARGET_DIR=str(crate / "target" / "newmetrics"))
    r = subprocess.run(cmd, shell=True, cwd=crate, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not out_json.exists():
        print(r.stdout[-2000:], r.stderr[-2000:])
        return None
    return json.loads(out_json.read_text(encoding="utf-8"))


def evaluate(pkey: str, pcfg: dict, cfg: dict) -> dict:
    ws = Workspace(MP4_ROOT, pkey, pcfg, cfg["policy"])
    restored = ws.restore_protected()
    checker = RustChecker(ws.crate, cfg["toolchain"], cfg["tools"], ws.protected_files())
    build = checker.build()
    tests = checker.test() if build["ok"] else {"passed": 0, "total": sum(len(v) for v in checker.benchmark_tests().values()),
                                                 "benchmark_status": {}, "all_results": []}
    clippy = checker.clippy(categorize=True) if build["ok"] else {"by_group": {}, "own": [], "protected": []}
    nm_path = ws.traj_dir / "newmetrics_summary.json"
    metrics = official_newmetrics(ws.crate, cfg["toolchain"], nm_path) if build["ok"] else None
    lexical = checker.unsafe_metrics(ws.own_files(), run_tool=False)["lexical"]
    summary_path = ws.traj_dir / "summary.json"
    iters = json.loads(summary_path.read_text(encoding="utf-8")).get("agent_iterations") if summary_path.exists() else None
    all_tests = tests.get("all_results", [])
    res = {
        "project": pkey, "name": ws.name, "compiles": build["ok"],
        "tests_passed": tests["passed"], "tests_total": tests["total"],
        "pass_rate": round(tests["passed"] / tests["total"], 4) if tests["total"] else 0.0,
        "all_cargo_tests": {"passed": sum(t["status"] == "ok" for t in all_tests), "total": len(all_tests)},
        "benchmark_status": tests.get("benchmark_status", {}),
        "agent_iterations": iters,
        "clippy_all_targets": {g: clippy.get("by_group", {}).get(g, {}).get("own", 0) +
                               clippy.get("by_group", {}).get(g, {}).get("protected", 0) for g in CLIPPY_GROUPS + ["other"]},
        "clippy_own_files": {g: clippy.get("by_group", {}).get(g, {}).get("own", 0) for g in CLIPPY_GROUPS + ["other"]},
        "clippy_warnings": [{k: x[k] for k in ("group", "lint", "file", "line", "message")}
                            for x in clippy.get("own", []) + clippy.get("protected", [])],
        "newmetrics": metrics,
        "lexical_scan": {k: v for k, v in lexical.items() if k != "locations"},
        "interface_problems": ws.interface_report(),
        "protected_files_restored": restored,
    }
    (ws.traj_dir / "final_evaluation.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


def tables(results: list[dict]) -> str:
    out = ["### Overall results", "",
           "| Project | Compiles | Tests Passed | Total Tests | Pass Rate | Agent Iterations |", "|---|---|---|---|---|---|"]
    for r in results:
        out.append(f"| {r['name']} | {'Yes' if r['compiles'] else 'No'} | {r['tests_passed']} | {r['tests_total']} | "
                   f"{100 * r['pass_rate']:.1f}% | {r['agent_iterations'] if r['agent_iterations'] is not None else '-'} |")
    out += ["", "### Safety (newmetrics)", "",
            "| Project | Unsafe Lines | Unsafe Calls | Unsafe Casts | Pointer Arithmetic | Raw Ptr Derefs | Raw Ptr Decls | Total Spans | Unsafe Ratio |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["newmetrics"] or {}
        out.append(f"| {r['name']} | {m.get('unsafe_lines', 'n/a')} | {m.get('unsafe_calls', 'n/a')} | {m.get('unsafe_casts', 'n/a')} | "
                   f"{m.get('pointer_arithmetic', 'n/a')} | {m.get('raw_pointer_dereferences', 'n/a')} | "
                   f"{m.get('raw_pointer_declarations', 'n/a')} | {m.get('total_spans', 'n/a')} | {m.get('unsafe_ratio', 'n/a')} |")
    out += ["", "### Idiomaticity (Clippy, all targets incl. benchmark tests)", "",
            "| Project | Correctness | Style | Complexity | Performance | Other (e.g. suspicious) |", "|---|---|---|---|---|---|"]
    for r in results:
        c = r["clippy_all_targets"]
        out.append(f"| {r['name']} | {c['correctness']} | {c['style']} | {c['complexity']} | {c['perf']} | {c['other']} |")
    out += ["", "Clippy restricted to the generated (non-test) files:", "",
            "| Project | Correctness | Style | Complexity | Performance | Other |", "|---|---|---|---|---|---|"]
    for r in results:
        c = r["clippy_own_files"]
        out.append(f"| {r['name']} | {c['correctness']} | {c['style']} | {c['complexity']} | {c['perf']} | {c['other']} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="1/2/name; default: all projects")
    ap.add_argument("--config", default=str(MP4_ROOT / "configs" / "config.json"))
    ap.add_argument("--projects", default=str(MP4_ROOT / "configs" / "projects.json"))
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    projects = json.loads(Path(args.projects).read_text(encoding="utf-8"))
    keys = [resolve_project(projects, args.project)[0]] if args.project else list(projects)
    results = [evaluate(k, projects[k], cfg) for k in keys]
    md = tables(results)
    print(md)
    if not args.project:
        (MP4_ROOT / "trajectories" / "evaluation_tables.md").write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
