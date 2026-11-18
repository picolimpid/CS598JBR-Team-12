"""Tools exposed to the LLM agent (schemas live in MP4/configs/tools.json).

Path conventions seen by the agent:
  c_src/<path>      original C project (read-only)
  rust_spec/<path>  CRUST-Bench interface + tests as shipped (read-only)
  <path>            the Rust crate being written (e.g. src/lib.rs, src/satc.rs)

Writes are only allowed to `.rs` files under src/ of the crate that are not
protected benchmark files (tests in src/bin/, Cargo.toml, Cargo.lock).
"""

from __future__ import annotations

import difflib
import inspect
import json
import re
from pathlib import Path, PurePosixPath

from rust_checks import (RustChecker, diag_to_dict, extract_fns, interface_problems, lexical_safety_scan,
                         truncate)


class ToolError(Exception):
    pass


class ToolBox:
    def __init__(self, ws, checker: RustChecker, cfg: dict, traj):
        self.ws = ws
        self.checker = checker
        self.cfg = cfg
        self.traj = traj
        self.max_read = cfg.get("max_read_chars", 60000)
        self.max_out = cfg.get("max_tool_output_chars", 16000)
        self.files_written: set[str] = set()
        self.calls = 0
        self.finished = False
        self.finish_summary = ""

    # ------------------------------------------------------------------ path handling
    def _resolve(self, path: str, for_write: bool = False) -> tuple[Path, str, str]:
        """Return (absolute path, root label, relative posix path)."""
        if not isinstance(path, str) or not path.strip():
            raise ToolError("path must be a non-empty string")
        p = path.strip().replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        if p in ("", "."):
            p = "."
        if p.startswith("/") or re.match(r"^[A-Za-z]:", p):
            raise ToolError("use relative paths (crate-relative, or prefixed with c_src/ or rust_spec/)")
        if p == "c_src" or p.startswith("c_src/"):
            base, label, rel = self.ws.c_dir, "c_src", p[len("c_src"):].lstrip("/") or "."
        elif p == "rust_spec" or p.startswith("rust_spec/"):
            base, label, rel = self.ws.spec_dir, "rust_spec", p[len("rust_spec"):].lstrip("/") or "."
        else:
            base, label, rel = self.ws.crate, "crate", p
        if ".." in PurePosixPath(rel).parts:
            raise ToolError("'..' is not allowed in paths")
        full = (base / rel).resolve()
        if base.resolve() not in full.parents and full != base.resolve():
            raise ToolError("path escapes the workspace")
        if for_write:
            if label != "crate":
                raise ToolError(f"{label}/ is read-only (it is the original C project / benchmark spec)")
            if rel in self.ws.protected_files():
                raise ToolError(f"{rel} is a protected benchmark file (tests/manifest) and must not be modified")
            if rel.startswith("src/bin/"):
                raise ToolError("do not add files under src/bin/ (reserved for the benchmark tests)")
            if not rel.startswith("src/") or not rel.endswith(".rs"):
                raise ToolError("you may only write Rust source files under src/ (e.g. src/lib.rs, src/<module>.rs)")
        if "target" in PurePosixPath(rel).parts[:1]:
            raise ToolError("the target/ directory is build output; use the cargo_* tools instead")
        return full, label, rel

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, name: str, args: dict) -> str:
        self.calls += 1
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            valid = sorted(n[5:] for n in dir(self) if n.startswith("tool_"))
            return f"ERROR: unknown tool `{name}`. Valid tools: {', '.join(valid)}"
        if not isinstance(args, dict):
            return "ERROR: tool arguments must be a JSON object"
        accepted = set(inspect.signature(fn).parameters)
        ignored = sorted(k for k in args if k not in accepted)
        args = {k: v for k, v in args.items() if k in accepted}
        try:
            out = fn(**args)
        except ToolError as exc:
            out = f"ERROR: {exc}"
        except TypeError as exc:
            out = f"ERROR: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface any tool failure to the agent
            out = f"ERROR: {type(exc).__name__}: {exc}"
        if ignored and not out.startswith("ERROR"):
            out += f"\n(note: ignored unknown arguments {ignored})"
        return truncate(out, self.max_out) if name != "read_file" else out

    # ------------------------------------------------------------------ file tools
    def tool_list_files(self, path: str = ".") -> str:
        full, label, rel = self._resolve(path)
        if not full.exists():
            raise ToolError(f"{path} does not exist")
        if full.is_file():
            return f"{path} is a file"
        prefix = "" if label == "crate" else f"{label}/"
        prefix += "" if rel == "." else rel.rstrip("/") + "/"
        tree = self.ws.file_tree(full, prefix)
        if label == "crate" and rel == ".":
            tree += "\n\n(read-only roots also available: c_src/ = original C project, rust_spec/ = benchmark spec)"
        return tree or "(empty)"

    def tool_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        full, _, _ = self._resolve(path)
        if not full.is_file():
            raise ToolError(f"{path} is not a file")
        lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
        s = max(1, int(start_line or 1))
        e = min(len(lines), int(end_line or len(lines)))
        out, size = [], 0
        for i in range(s, e + 1):
            row = f"{i:>5}| {lines[i - 1]}"
            size += len(row) + 1
            if size > self.max_read:
                out.append(f"... [truncated at line {i - 1} of {len(lines)}; call read_file with start_line={i}]")
                break
            out.append(row)
        header = f"{path} (lines {s}-{min(e, s + len(out) - 1)} of {len(lines)}; the `NNNNN| ` prefix is NOT part of the file)"
        return header + "\n" + "\n".join(out)

    def tool_search(self, pattern: str, path: str = ".") -> str:
        full, label, rel = self._resolve(path)
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regex: {exc}")
        files = [full] if full.is_file() else [p for p in sorted(full.rglob("*")) if p.is_file()]
        base = {"crate": self.ws.crate, "c_src": self.ws.c_dir, "rust_spec": self.ws.spec_dir}[label]
        prefix = "" if label == "crate" else f"{label}/"
        hits = []
        for f in files:
            relf = f.relative_to(base).as_posix()
            if relf.startswith("target/") or f.suffix not in (".c", ".h", ".rs", ".md", ".toml", ".txt", ""):
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{prefix}{relf}:{i}: {line.strip()[:200]}")
                    if len(hits) >= 150:
                        return "\n".join(hits) + "\n... (more matches truncated)"
        return "\n".join(hits) or "(no matches)"

    def _record_write(self, rel: str, before: str, after: str, op: str) -> str:
        self.files_written.add(rel)
        diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                            fromfile=f"a/{rel}", tofile=f"b/{rel}", n=2))
        added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
        self.traj.event("file_write", op=op, path=rel, lines_added=added, lines_removed=removed,
                        size_after=len(after), diff=truncate(diff, self.cfg.get("log_diff_chars", 40000)))
        self.traj.log(f"  {op} {rel}: +{added} -{removed} lines ({len(after.splitlines())} lines now)")
        warn = ""
        if self.cfg.get("check_interface_on_write", True):
            iface = self._interface_for(rel)
            if iface is not None:
                probs = interface_problems(iface.read_text(encoding="utf-8"), after)
                if probs:
                    warn = "\nWARNING - interface problems in this file:\n" + "\n".join(f"- {p}" for p in probs[:15])
        return f"OK: {op} {rel} ({len(after.splitlines())} lines, +{added}/-{removed}).{warn}"

    def _interface_for(self, rel: str) -> Path | None:
        for item in self.ws.module_map():
            if item["target"] == rel:
                return self.ws.spec_dir / "src" / "interfaces" / f"{item['module']}.rs"
        return None

    def tool_write_file(self, path: str, content: str) -> str:
        full, _, rel = self._resolve(path, for_write=True)
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        before = full.read_text(encoding="utf-8") if full.exists() else ""
        full.parent.mkdir(parents=True, exist_ok=True)
        if content and not content.endswith("\n"):
            content += "\n"
        full.write_text(content, encoding="utf-8")
        return self._record_write(rel, before, content, "write_file")

    def tool_append_file(self, path: str, content: str) -> str:
        full, _, rel = self._resolve(path, for_write=True)
        before = full.read_text(encoding="utf-8") if full.exists() else ""
        after = before + content + ("" if content.endswith("\n") else "\n")
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(after, encoding="utf-8")
        return self._record_write(rel, before, after, "append_file")

    def tool_edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        full, _, rel = self._resolve(path, for_write=True)
        if not full.is_file():
            raise ToolError(f"{path} does not exist (use write_file to create it)")
        before = full.read_text(encoding="utf-8")
        n = before.count(old_string) if old_string else 0
        if n == 0:
            hint = ""
            first = old_string.strip().splitlines()[0].strip() if old_string.strip() else ""
            if first and first in before:
                # show what the file really contains there, so the model can copy it instead of guessing again
                lines = before.splitlines()
                start = next(i for i, ln in enumerate(lines) if first in ln)
                span = max(1, len(old_string.strip("\n").splitlines()))
                actual = "\n".join(lines[start:start + span + 2])
                hint = (" (the first line exists but the following lines differ). The file currently has, from line "
                        f"{start + 1}:\n```\n{actual}\n```\nUse this exact text as old_string, or rewrite the whole "
                        "file with write_file")
            if re.match(r"\s*\d+\|", old_string):
                hint = " (do not include the `NNNNN| ` line-number prefix from read_file)"
            raise ToolError(f"old_string not found in {rel}{hint}. Re-read the file and copy the text exactly.")
        if n > 1 and not replace_all:
            raise ToolError(f"old_string occurs {n} times in {rel}; add surrounding context or set replace_all=true")
        after = before.replace(old_string, new_string) if replace_all else before.replace(old_string, new_string, 1)
        full.write_text(after, encoding="utf-8")
        return self._record_write(rel, before, after, "edit_file")

    # ------------------------------------------------------------------ cargo tools
    def _fmt_diags(self, diags, limit: int) -> str:
        ds = sorted(diags, key=lambda d: (d.file.startswith("src/bin"), d.file, d.line))
        return "\n\n".join(d.rendered for d in ds[:limit])

    def tool_cargo_check(self, reason: str | None = None) -> str:
        r = self.checker.check()
        self.traj.event("tool_feedback", tool="cargo_check", errors=[diag_to_dict(d) for d in r["errors"]],
                        warnings=len(r["warnings"]))
        if r["ok"]:
            own_w = [d for d in r["warnings"] if not self.checker.is_protected(d.file)]
            msg = f"cargo check --all-targets: OK (0 errors, {len(own_w)} warnings in your files)"
            if own_w:
                msg += "\n\n" + self._fmt_diags(own_w, 10)
            return msg
        return (f"cargo check --all-targets: FAILED with {len(r['errors'])} errors"
                f"{' (TIMEOUT)' if r['timed_out'] else ''}\n\n" + self._fmt_diags(r["errors"], 15))

    def tool_cargo_build(self, reason: str | None = None) -> str:
        r = self.checker.build()
        return f"cargo build: {'OK' if r['ok'] else 'FAILED'}\n" + truncate(r["output"], 8000)

    def tool_cargo_test(self, test_name: str | None = None, reason: str | None = None) -> str:
        r = self.checker.test(test_name)
        self.traj.event("tool_feedback", tool="cargo_test", passed=r["passed"], total=r["total"],
                        status=r["benchmark_status"], failed=r["failed"])
        if not r["compiled"]:
            return "cargo test: test harness did NOT compile.\n" + r["raw_tail"]
        lines = [f"cargo test{(' ' + test_name) if test_name else ''}: benchmark tests passed {r['passed']}/{r['total']}"]
        for k, st in r["benchmark_status"].items():
            lines.append(f"  {k}: {st}")
        if r["timed_out"]:
            lines.append("TIMEOUT: the test run did not finish (infinite loop?).\n" + r["raw_tail"])
        for name, body in list(r["failure_details"].items())[:8]:
            lines.append(f"\n---- {name} ----\n{body}")
        return "\n".join(lines)

    def tool_cargo_clippy(self, reason: str | None = None) -> str:
        r = self.checker.clippy(categorize=False)
        if not r["ok"]:
            return "cargo clippy: compilation failed\n\n" + self._fmt_diags(r["errors"], 10)
        own = r["own"]
        self.traj.event("tool_feedback", tool="cargo_clippy", own=len(own), protected=len(r["protected"]),
                        lints=[x["lint"] for x in own])
        lines = [f"cargo clippy --all-targets --all-features -W clippy::{{correctness,style,complexity,perf}}: "
                 f"{len(own)} Clippy warnings in your files, {len(r['protected'])} in protected test files (ignore those), "
                 f"{len(r['rustc_own'])} rustc warnings in your files"]
        lines += [x["rendered"] for x in own[:15]]
        lines += [x["rendered"] for x in r["rustc_own"][:8]]
        return "\n\n".join(lines)

    def tool_unsafe_metrics(self, reason: str | None = None) -> str:
        r = self.checker.unsafe_metrics(self.ws.own_files())
        self.traj.event("tool_feedback", tool="unsafe_metrics", result={k: v for k, v in r.items() if k != "lexical"},
                        lexical={k: v for k, v in r["lexical"].items() if k != "locations"})
        lex = r["lexical"]
        out = [f"unsafe metrics ({r['tool']}): primary total = {r['primary_total']}"]
        if r.get("metrics"):
            out.append(json.dumps(r["metrics"], indent=1))
        if r.get("error"):
            out.append("tool error:\n" + r["error"][:2000])
        for k in ("unsafe_keyword", "raw_pointer_casts", "transmute", "pointer_ops", "ffi"):
            if lex[k]:
                out.append(f"{k}: {lex[k]}\n" + "\n".join("  " + x for x in lex["locations"][k]))
        return "\n".join(out)

    def tool_check_interface(self, reason: str | None = None) -> str:
        probs = self.ws.interface_report()
        scan = lexical_safety_scan(self.ws.crate, self.ws.own_files())
        out = [f"interface problems: {len(probs)}"] + [f"- {p}" for p in probs]
        out.append(f"remaining unimplemented!/todo!: {scan['unimplemented']}")
        out += [f"  {x}" for x in scan["locations"]["unimplemented"]]
        n_fns = sum(len(extract_fns((self.ws.spec_dir / 'src' / 'interfaces' / f"{i['module']}.rs").read_text(encoding='utf-8')))
                    for i in self.ws.module_map())
        out.append(f"(the interface declares {n_fns} public functions in total)")
        return "\n".join(out)

    def tool_finish(self, summary: str = "") -> str:
        self.finished = True
        self.finish_summary = summary or ""
        return "Iteration finished. The controller will now run the full verification (build, tests, Clippy, unsafe metrics)."
