"""Compiler / test / Clippy / unsafe-metrics / interface feedback for the agent.

Every check runs the pinned toolchain (`cargo +nightly-2025-05-13 ...`) inside the
translation crate and returns structured results plus a compact text rendering
that is shown to the LLM.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

CLIPPY_GROUPS = ["correctness", "style", "complexity", "perf"]

# Same regexes as newmetrics/aggregate.py (the instructional team's tool).
NEWMETRICS_LABELS = {
    "pointer_arithmetic": "Pointer arithmetic",
    "unsafe_lines": "Unsafe lines",
    "unsafe_calls": "Unsafe calls",
    "unsafe_casts": "Unsafe casts",
    "total_spans": "Total spans",
    "raw_pointer_dereferences": "Raw pointer dereferences",
    "raw_pointer_declarations": "Raw pointer declarations",
}
PRIMARY_UNSAFE = ["unsafe_lines", "unsafe_calls", "unsafe_casts", "pointer_arithmetic"]
# raw pointers are legal in safe Rust (only dereferencing needs `unsafe`), but newmetrics reports them and they
# usually mean C-style ownership (e.g. a leaked Box::into_raw tail pointer), so they are minimised too
RAW_POINTER_METRICS = ["raw_pointer_declarations", "raw_pointer_dereferences"]


@dataclass
class CmdResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def output(self) -> str:
        return (self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr).strip()


@dataclass
class Diagnostic:
    level: str
    code: str
    message: str
    file: str
    line: int
    col: int
    rendered: str

    def key(self):
        return (self.level, self.code, self.file, self.line, self.col, self.message)


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.8)]
    tail = text[-int(max_chars * 0.15):]
    return f"{head}\n... [{len(text) - len(head) - len(tail)} chars truncated] ...\n{tail}"


def strip_comments(src: str) -> str:
    """Remove // and /* */ comments while preserving line numbers; leaves string literals alone."""
    out = []
    i, n = 0, len(src)
    in_str = False
    while i < n:
        c = src[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(src[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            # skip char literals like '"'
            if i > 0 and src[i - 1] == "'" and i + 1 < n and src[i + 1] == "'":
                out.append(c)
                i += 1
                continue
            in_str = True
            out.append(c)
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j == -1 else j
            i = j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append("\n" * src.count("\n", i, j))
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


# ====================================================================== commands
class RustChecker:
    def __init__(self, crate_dir: Path, toolchain: str, cfg: dict, protected: list[str], traj=None):
        self.crate = Path(crate_dir)
        self.toolchain = toolchain
        self.cfg = cfg
        self.protected = set(protected)
        self.traj = traj
        self.timeouts = cfg.get("timeouts_s", {})

    # ---------------------------------------------------------------- process
    def run(self, args: list[str], timeout: int, env_extra: dict | None = None) -> CmdResult:
        env = dict(os.environ)
        env.setdefault("CARGO_TERM_COLOR", "never")
        env["CARGO_TERM_COLOR"] = "never"
        env.pop("RUSTFLAGS", None)
        if env_extra:
            env.update(env_extra)
        t0 = time.time()
        popen_kw = {}
        if os.name == "posix":
            popen_kw["start_new_session"] = True
        try:
            proc = subprocess.Popen(args, cwd=self.crate, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                    errors="replace", **popen_kw)
        except FileNotFoundError as exc:
            return CmdResult(args, 127, "", f"command not found: {exc}", 0.0)
        try:
            out, err = proc.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            out, err = proc.communicate()
            timed_out = True
            err = (err or "") + f"\n[TIMEOUT after {timeout}s - possible infinite loop or deadlock]"
        res = CmdResult(args, proc.returncode if not timed_out else -9, out or "", err or "",
                        round(time.time() - t0, 2), timed_out)
        if self.traj is not None:
            self.traj.event("command", cmd=" ".join(args), returncode=res.returncode,
                            duration_s=res.duration_s, timed_out=timed_out,
                            output=truncate(res.output, self.cfg.get("log_command_output_chars", 20000)))
        return res

    def cargo(self, *args: str, timeout_key: str = "build", env_extra: dict | None = None) -> CmdResult:
        return self.run(["cargo", f"+{self.toolchain}", *args], self.timeouts.get(timeout_key, 300), env_extra)

    def toolchain_info(self) -> dict:
        info = {}
        for name, args in {"cargo": ["cargo", f"+{self.toolchain}", "--version"],
                           "rustc": ["rustc", f"+{self.toolchain}", "--version"],
                           "clippy": ["cargo", f"+{self.toolchain}", "clippy", "--version"]}.items():
            r = self.run(args, 60)
            info[name] = r.stdout.strip() if r.returncode == 0 else f"UNAVAILABLE ({r.output[:200]})"
        info["newmetrics"] = self.newmetrics_path() or "UNAVAILABLE (lexical fallback will be used)"
        return info

    # ---------------------------------------------------------------- diagnostics
    def _parse_json_diags(self, stdout: str) -> list[Diagnostic]:
        diags: dict = {}
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("reason") != "compiler-message":
                continue
            m = rec.get("message") or {}
            level = m.get("level", "")
            if level not in ("error", "warning"):
                continue
            spans = m.get("spans") or []
            prim = next((s for s in spans if s.get("is_primary")), spans[0] if spans else None)
            if prim is None:
                # summary lines like "aborting due to N previous errors"
                if level == "error" and not re.match(r"(aborting due to|could not compile)", m.get("message", "")):
                    d = Diagnostic(level, "", m.get("message", ""), "", 0, 0, m.get("rendered") or m.get("message", ""))
                    diags[d.key()] = d
                continue
            code = (m.get("code") or {}).get("code") or ""
            d = Diagnostic(level, code, m.get("message", ""), prim.get("file_name", ""),
                           int(prim.get("line_start") or 0), int(prim.get("column_start") or 0),
                           (m.get("rendered") or m.get("message", "")).rstrip())
            diags[d.key()] = d
        return list(diags.values())

    def is_protected(self, rel: str) -> bool:
        return rel.replace("\\", "/") in self.protected

    def check(self) -> dict:
        r = self.cargo("check", "--all-targets", "--message-format=json", timeout_key="build")
        diags = self._parse_json_diags(r.stdout)
        errors = [d for d in diags if d.level == "error"]
        if r.returncode != 0 and not errors:
            errors = [Diagnostic("error", "", "cargo check failed", "", 0, 0, truncate(r.stderr, 4000))]
        warnings = [d for d in diags if d.level == "warning"]
        return {"ok": r.returncode == 0, "errors": errors, "warnings": warnings,
                "timed_out": r.timed_out, "duration_s": r.duration_s}

    def build(self) -> dict:
        r = self.cargo("build", timeout_key="build")
        return {"ok": r.returncode == 0, "output": r.output, "duration_s": r.duration_s}

    # ---------------------------------------------------------------- tests
    def benchmark_tests(self) -> dict[str, list[str]]:
        """Static list of #[test] functions in the protected benchmark test files."""
        out = {}
        for rel in sorted(self.protected):
            if not rel.endswith(".rs"):
                continue
            p = self.crate / rel
            if not p.exists():
                continue
            src = strip_comments(p.read_text(encoding="utf-8", errors="replace"))
            names = re.findall(r"#\[test\](?:\s*#\[[^\]]*\])*\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)", src)
            if names:
                out[rel] = names
        return out

    def test(self, test_filter: str | None = None) -> dict:
        args = ["test", "--no-fail-fast"]
        if test_filter:
            args += ["--", test_filter]
        # Windows Smart App Control sometimes refuses to start a freshly built test binary (os error 4551);
        # that says nothing about the code, so retry instead of reporting the tests as "not run".
        retries = int(self.cfg.get("os_block_retries", 4))
        for attempt in range(retries + 1):
            r = self.cargo(*args, timeout_key="test")
            if "os error 4551" not in r.output or attempt == retries:
                break
            if self.traj is not None:
                self.traj.log(f"test binary blocked by the OS application-control policy; retry {attempt + 1}/{retries}")
            time.sleep(10)
        os_blocked = "os error 4551" in r.output
        text = r.output
        compiled = "could not compile" not in text and ("test result:" in text or r.returncode == 0 or r.timed_out)
        results = []  # (section_src, name, status)
        section = ""
        for line in text.splitlines():
            m = re.match(r"\s*Running (?:unittests )?(\S+)", line)
            if m:
                section = m.group(1)
                continue
            if re.match(r"\s*Doc-tests ", line):
                section = "doctests"
                continue
            m = re.match(r"test (.+?) \.\.\. (ok|FAILED|ignored)", line)
            if m:
                results.append((section, m.group(1).strip(), m.group(2)))
        bench = self.benchmark_tests()
        total = sum(len(v) for v in bench.values())
        passed = 0
        bench_status = {}
        for rel, names in bench.items():
            for nm in names:
                cands = [(sec, status) for sec, name, status in results if name.split("::")[-1] == nm]
                exact = [st for sec, st in cands if sec.replace("\\", "/").endswith(rel)]
                st = exact[0] if exact else (cands[0][1] if cands else "not run")
                bench_status[f"{rel}::{nm}"] = st
                passed += st == "ok"
        failures = re.findall(r"---- (.+?) stdout ----\n(.*?)(?=\n---- |\nfailures:\n|\Z)", text, re.S)
        failed_other = [f"{sec}::{n}" for sec, n, s in results if s == "FAILED"]
        return {
            "compiled": compiled, "timed_out": r.timed_out, "os_blocked": os_blocked, "total": total, "passed": passed,
            "benchmark_status": bench_status, "all_results": [{"target": s, "name": n, "status": st} for s, n, st in results],
            "failed": failed_other,
            "failure_details": {name: truncate(body.strip(), 3000) for name, body in failures},
            "raw_tail": truncate(text, 6000) if (not compiled or r.timed_out) else "",
            "duration_s": r.duration_s,
        }

    # ---------------------------------------------------------------- clippy
    def _touch_sources(self) -> None:
        now = time.time()
        for p in (self.crate / "src").rglob("*.rs"):
            try:
                os.utime(p, (now, now))
            except OSError:
                pass

    def clippy_once(self, group: str | None = None) -> dict:
        """group=None runs the exact grading command; otherwise only that clippy group is enabled."""
        self._touch_sources()
        lint_args = ["-W", "clippy::correctness", "-W", "clippy::style", "-W", "clippy::complexity", "-W", "clippy::perf"]
        if group:
            lint_args = ["-A", "clippy::all", "-W", f"clippy::{group}"]
        r = self.cargo("clippy", "--all-targets", "--all-features", "--message-format=json", "--", *lint_args,
                       timeout_key="clippy")
        diags = self._parse_json_diags(r.stdout)
        errors = [d for d in diags if d.level == "error"]
        return {"ok": r.returncode == 0, "diags": diags, "errors": errors}

    def clippy(self, categorize: bool = True) -> dict:
        full = self.clippy_once(None)
        if not full["ok"] and full["errors"]:
            return {"ok": False, "errors": full["errors"], "by_group": {}, "own": [], "protected": [], "rustc_own": []}
        clippy_all = [d for d in full["diags"] if d.code.startswith("clippy::")]
        rustc_warn = [d for d in full["diags"] if d.level == "warning" and not d.code.startswith("clippy::")]
        group_of: dict = {}
        by_group = {}
        if categorize:
            for g in CLIPPY_GROUPS:
                res = self.clippy_once(g)
                ds = [d for d in res["diags"] if d.code.startswith("clippy::")]
                for d in ds:
                    group_of.setdefault((d.code, d.file, d.line, d.col), g)
        own, prot = [], []
        for d in clippy_all:
            g = group_of.get((d.code, d.file, d.line, d.col), "other")
            item = {"group": g, "lint": d.code, "file": d.file, "line": d.line, "message": d.message, "rendered": d.rendered}
            (prot if self.is_protected(d.file) else own).append(item)
        for g in CLIPPY_GROUPS + ["other"]:
            by_group[g] = {"own": sum(1 for x in own if x["group"] == g), "protected": sum(1 for x in prot if x["group"] == g)}
        rustc_own = [{"lint": d.code, "file": d.file, "line": d.line, "message": d.message, "rendered": d.rendered}
                     for d in rustc_warn if not self.is_protected(d.file)]
        return {"ok": True, "errors": [], "by_group": by_group, "own": own, "protected": prot, "rustc_own": rustc_own,
                "total_all_files": len(clippy_all)}

    # ---------------------------------------------------------------- unsafe metrics
    @staticmethod
    def newmetrics_path() -> str | None:
        p = shutil.which("cargo-newmetrics")
        if p:
            return p
        cand = Path.home() / ".cargo" / "bin" / ("cargo-newmetrics.exe" if os.name == "nt" else "cargo-newmetrics")
        return str(cand) if cand.exists() else None

    def unsafe_metrics(self, own_files: list[str], run_tool: bool = True) -> dict:
        lexical = lexical_safety_scan(self.crate, own_files)
        res = {"tool": "lexical-fallback", "ok": True, "metrics": None, "lexical": lexical}
        if run_tool and self.cfg.get("use_newmetrics", True) and self.newmetrics_path():
            env = {"RUSTUP_TOOLCHAIN": self.toolchain,
                   "CARGO_TARGET_DIR": str(self.crate / "target" / "newmetrics")}
            r = self.run(["cargo", "newmetrics"], self.timeouts.get("metrics", 600), env)
            totals = {k: 0 for k in NEWMETRICS_LABELS}
            found = False
            for key, label in NEWMETRICS_LABELS.items():
                for m in re.finditer(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", r.output, re.M):
                    totals[key] += int(m.group(1))
                    found = True
            totals["unsafe_ratio"] = totals["unsafe_lines"] / totals["total_spans"] if totals["total_spans"] else 0.0
            res["tool"] = "newmetrics"
            res["ok"] = r.returncode == 0 and found
            res["metrics"] = totals
            if not res["ok"]:
                res["error"] = truncate(r.output, 3000)
        if res["metrics"] and res["ok"]:
            res["primary_total"] = sum(res["metrics"][k] for k in PRIMARY_UNSAFE)
            res["raw_pointer_total"] = sum(res["metrics"][k] for k in RAW_POINTER_METRICS)
        else:
            res["primary_total"] = lexical["unsafe_keyword"] + lexical["raw_pointer_casts"] + lexical["transmute"]
            res["raw_pointer_total"] = lexical["raw_pointer_ops"]
        return res


# ====================================================================== static scans
_SCAN_PATTERNS = {
    "unimplemented": r"\b(?:unimplemented|todo)!\s*\(",
    "unsafe_keyword": r"\bunsafe\b",
    "raw_pointer_casts": r"\bas\s+\*\s*(?:const|mut)\b",
    "transmute": r"\btransmute\b",
    "ffi": r"extern\s+\"C\"|#\[\s*link\b|#\[\s*no_mangle\s*\]|\blibc::",
    "pointer_ops": r"\.(?:offset|byte_add|byte_sub|byte_offset|offset_from)\s*\(|\bptr::(?:read|write|copy|null)",
    "raw_pointer_ops": r"\bBox::(?:into|from)_raw\b|\.as_(?:mut_)?ptr\s*\(|\bptr::(?:NonNull|null_mut|addr_of)",
    "clippy_allow": r"#!?\[\s*allow\s*\([^\]]*clippy::",
    "unwrap_expect_panic": r"\.unwrap\(\)|\.expect\(|\bpanic!\s*\(",
}


def lexical_safety_scan(crate: Path, own_files: list[str]) -> dict:
    counts = {k: 0 for k in _SCAN_PATTERNS}
    locations = {k: [] for k in _SCAN_PATTERNS}
    for rel in own_files:
        p = crate / rel
        if not p.exists():
            continue
        raw_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        code_lines = strip_comments("\n".join(raw_lines)).splitlines()
        for i, line in enumerate(code_lines, 1):
            scrubbed = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
            for k, pat in _SCAN_PATTERNS.items():
                hits = len(re.findall(pat, scrubbed))
                if hits:
                    counts[k] += hits
                    if len(locations[k]) < 25:
                        locations[k].append(f"{rel}:{i}: {raw_lines[i - 1].strip()[:160]}")
    counts["locations"] = locations
    return counts


# ====================================================================== interface conformance
def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _match_bracket(s: str, i: int, open_c: str, close_c: str) -> int:
    """s[i] == open_c; return index just after the matching close_c ("->" is not a closer)."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "-" and j + 1 < len(s) and s[j + 1] == ">":
            j += 2
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(s)


def _find_sig_end(s: str, i: int) -> int:
    """Index of the `{` (body) or `;` (declaration) that ends a signature, ignoring ones inside [..], <..>, (..)."""
    depth = 0
    while i < len(s):
        c = s[i]
        if c == "-" and i + 1 < len(s) and s[i + 1] == ">":
            i += 2
            continue
        if c in "([<":
            depth += 1
        elif c in ")]>":
            depth -= 1
        elif c in "{;" and depth <= 0:
            return i
        i += 1
    return len(s)


def _split_top(s: str, sep: str = ",") -> list[str]:
    parts, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c == "-" and i + 1 < len(s) and s[i + 1] == ">":
            cur.append("->")
            i += 2
            continue
        if c in "([<{":
            depth += 1
        elif c in ")]>}":
            depth -= 1
        if c == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def _norm(t: str) -> str:
    t = re.sub(r"\s+", "", t)
    t = re.sub(r",([)\]>}])", r"\1", t)
    return t


def extract_fns(src: str) -> list[tuple]:
    """Normalised public function signatures: (name, generics, param types, return type)."""
    return [sig for sig, _ in extract_fns_with_text(src)]


def extract_fns_with_text(src: str) -> list[tuple[tuple, str]]:
    src = strip_comments(src)
    out = []
    for m in re.finditer(r"\bpub\s+fn\s+([A-Za-z_]\w*)", src):
        name = m.group(1)
        j = _skip_ws(src, m.end())
        generics = ""
        if j < len(src) and src[j] == "<":
            end = _match_bracket(src, j, "<", ">")
            generics = src[j:end]
            j = _skip_ws(src, end)
        if j >= len(src) or src[j] != "(":
            continue
        end = _match_bracket(src, j, "(", ")")
        params = _split_top(src[j + 1:end - 1])
        stop = _find_sig_end(src, end)
        rest = re.split(r"\bwhere\b", src[end:stop])[0].strip()
        ret = rest[2:].strip() if rest.startswith("->") else "()"
        types = []
        for p in params:
            p = p.strip()
            if not p:
                continue
            if re.fullmatch(r"(&\s*('\w+\s+)?)?(mut\s+)?self", p):
                types.append(_norm(p))
                continue
            colon = _split_top(p, ":")
            types.append(_norm(":".join(colon[1:])) if len(colon) > 1 else _norm(p))
        display = re.sub(r"\s+", " ", src[m.start():stop]).strip()
        display = re.sub(r"\(\s+", "(", re.sub(r",\s*\)", ")", display))
        out.append(((name, _norm(generics), tuple(types), _norm(ret)), display))
    return out


def _items(src: str, kind: str) -> dict:
    """Return {name: (attrs, body)} for `pub struct|enum Name { body }` items."""
    src = strip_comments(src)
    out = {}
    for m in re.finditer(rf"((?:#\[[^\]]*\]\s*)*)pub\s+{kind}\s+(\w+)\s*(<[^{{;(]*>)?\s*", src):
        name = m.group(2)
        j = m.end()
        body = ""
        if j < len(src) and src[j] == "{":
            end = _match_bracket(src, j, "{", "}")
            body = src[j + 1:end - 1]
        elif j < len(src) and src[j] == "(":
            end = _match_bracket(src, j, "(", ")")
            body = src[j + 1:end - 1]
        out[name] = (m.group(1), body)
    return out


def _derives(attrs: str) -> set:
    ds = set()
    for d in re.findall(r"#\[\s*derive\s*\(([^)]*)\)\s*\]", attrs):
        ds |= {x.strip().split("::")[-1] for x in d.split(",") if x.strip()}
    return ds


def _fields(body: str) -> dict:
    fields = {}
    for part in _split_top(body):
        part = re.sub(r"#\[[^\]]*\]", "", part).strip()
        m = re.match(r"pub(?:\([^)]*\))?\s+(\w+)\s*:\s*(.+)$", part, re.S)
        if m:
            fields[m.group(1)] = _norm(m.group(2))
    return fields


def _variants(body: str) -> list[str]:
    vs = []
    for part in _split_top(body):
        part = re.sub(r"#\[[^\]]*\]", "", part).strip()
        m = re.match(r"(\w+)", part)
        if m:
            vs.append(m.group(1))
    return vs


def _consts(src: str) -> dict:
    src = strip_comments(src)
    return {m.group(1): (_norm(m.group(2)), _norm(m.group(3)))
            for m in re.finditer(r"\bpub\s+const\s+(\w+)\s*:\s*([^=]+)=\s*([^;]+);", src)}


def interface_problems(interface_src: str, impl_src: str) -> list[str]:
    probs = []
    impl_fns = dict(extract_fns_with_text(impl_src))
    for sig, text in extract_fns_with_text(interface_src):
        if sig not in impl_fns:
            same_name = [t for s, t in impl_fns.items() if s[0] == sig[0]]
            hint = f" (found: {' | '.join(f'`{t}`' for t in same_name)})" if same_name else " (missing)"
            probs.append(f"function signature mismatch: required `{text}`{hint}")
    for kind in ("struct", "enum"):
        want, have = _items(interface_src, kind), _items(impl_src, kind)
        for name, (attrs, body) in want.items():
            if name not in have:
                probs.append(f"missing `pub {kind} {name}`")
                continue
            h_attrs, h_body = have[name]
            missing_derives = _derives(attrs) - _derives(h_attrs)
            if missing_derives:
                probs.append(f"`{kind} {name}` must derive {sorted(missing_derives)} (required by the interface)")
            if kind == "struct":
                hf = _fields(h_body)
                for f, t in _fields(body).items():
                    if f not in hf:
                        probs.append(f"`struct {name}` is missing public field `{f}: {t}`")
                    elif hf[f] != t:
                        probs.append(f"`struct {name}` field `{f}` has type `{hf[f]}` but the interface requires `{t}`")
            else:
                hv = set(_variants(h_body))
                miss = [v for v in _variants(body) if v not in hv]
                if miss:
                    probs.append(f"`enum {name}` is missing variants {miss}")
    hc = _consts(impl_src)
    for name, (t, v) in _consts(interface_src).items():
        if name not in hc:
            probs.append(f"missing `pub const {name}: {t}`")
        elif hc[name][0] != t:
            probs.append(f"`const {name}` has type `{hc[name][0]}` but the interface requires `{t}`")
        elif hc[name][1] != v:
            probs.append(f"`const {name}` has value `{hc[name][1]}` but the interface requires `{v}`")
    return probs


# ====================================================================== scoring & feedback
def score_tuple(v: dict) -> tuple:
    """Lexicographic quality score used to pick the best snapshot and detect progress."""
    return (
        int(v["build_ok"]),
        v["tests"]["passed"],
        -v["static"]["unimplemented"],
        -len(v["interface"]),
        -(v["safety"]["primary_total"] + v["safety"].get("raw_pointer_total", 0)),
        -(len(v["clippy"].get("own", [])) if v["build_ok"] else 999),
        -(len(v["clippy"].get("rustc_own", [])) if v["build_ok"] else 999),
    )


def is_success(v: dict, criteria: dict) -> bool:
    if not v["build_ok"] or v["tests"]["total"] == 0:
        return False
    if v["tests"]["passed"] < v["tests"]["total"]:
        return False
    if criteria.get("require_interface_ok", True) and v["interface"]:
        return False
    if criteria.get("require_no_unimplemented", True) and v["static"]["unimplemented"]:
        return False
    if v["safety"]["primary_total"] > criteria.get("max_unsafe_primary", 0):
        return False
    if v["safety"].get("raw_pointer_total", 0) > criteria.get("max_raw_pointers", 0):
        return False
    if v["static"]["ffi"]:
        return False
    if len(v["clippy"].get("own", [])) > criteria.get("max_clippy_warnings", 0):
        return False
    mrw = criteria.get("max_rustc_warnings")
    if mrw is not None and len(v["clippy"].get("rustc_own", [])) > mrw:
        return False
    return True


def headline(v: dict) -> str:
    c = v["clippy"].get("by_group", {})
    cl = " ".join(f"{g}={c[g]['own']}" for g in CLIPPY_GROUPS + ["other"] if g in c) or "n/a"
    m = v["safety"].get("metrics") or {}
    um = (f"lines={m.get('unsafe_lines', '?')} calls={m.get('unsafe_calls', '?')} casts={m.get('unsafe_casts', '?')} "
          f"ptr_arith={m.get('pointer_arithmetic', '?')} raw_ptr={v['safety'].get('raw_pointer_total', '?')}") \
        if m else f"lexical={v['safety']['primary_total']} raw_ptr={v['safety'].get('raw_pointer_total', '?')}"
    return (f"build={'OK' if v['build_ok'] else 'FAIL'} | tests {v['tests']['passed']}/{v['tests']['total']} | "
            f"interface problems={len(v['interface'])} | unimplemented!={v['static']['unimplemented']} | "
            f"unsafe[{v['safety'].get('tool')}] {um} | clippy(own) {cl} | rustc warnings(own)={len(v['clippy'].get('rustc_own', []))}")


def format_feedback(v: dict, fcfg: dict) -> str:
    lim = fcfg.get("max_items", 12)
    sec_chars = fcfg.get("max_section_chars", 12000)
    parts = [f"OVERALL: {headline(v)}", ""]
    focus = None

    if not v["build_ok"]:
        errs = v["check"]["errors"]
        own_first = sorted(errs, key=lambda d: (d["file"].startswith("src/bin"), d["file"], d["line"]))
        body = "\n\n".join(d["rendered"] for d in own_first[:lim])
        parts.append(f"## Compilation errors ({len(errs)} total, first {min(lim, len(errs))} shown)\n" + truncate(body, sec_chars))
        focus = focus or "compilation errors"
    if v["interface"]:
        parts.append("## Interface conformance problems (the benchmark interface is part of the specification)\n" +
                     "\n".join(f"- {p}" for p in v["interface"][:40]))
        focus = focus or "interface conformance"
    if v["static"]["unimplemented"]:
        locs = v["static"]["locations"]["unimplemented"]
        parts.append(f"## Unimplemented functions ({v['static']['unimplemented']} `unimplemented!`/`todo!` left)\n" +
                     "\n".join(f"- {x}" for x in locs))
        focus = focus or "implementing the remaining functions"
    t = v["tests"]
    if v["build_ok"]:
        failing = [k for k, s in t["benchmark_status"].items() if s != "ok"]
        if failing or t.get("timed_out") or t.get("failed"):
            lines = [f"## Failing tests ({t['passed']}/{t['total']} benchmark tests pass)"]
            for k in failing:
                lines.append(f"- {k}: {t['benchmark_status'][k]}")
            extra = [x for x in t.get("failed", []) if not any(x.endswith(k.split('::')[-1]) for k in failing)]
            for x in extra:
                lines.append(f"- (non-benchmark) {x}: FAILED")
            details = "\n\n".join(f"---- {n} ----\n{b}" for n, b in list(t["failure_details"].items())[:lim])
            if t.get("os_blocked"):
                details += ("\n[the operating system refused to start some test binaries (application-control policy, "
                            "os error 4551). Tests marked 'not run' were NOT executed; this is not a failure of your code. "
                            "Do not change working code because of it; run cargo_test again later.]")
            if t.get("timed_out"):
                details += "\n[the test run TIMED OUT: look for infinite loops]\n" + t.get("raw_tail", "")
            elif not t.get("compiled", True):
                details += "\n" + t.get("raw_tail", "")
            parts.append("\n".join(lines) + "\n\n" + truncate(details, sec_chars))
            focus = focus or "failing tests"
    s = v["safety"]
    lex = s["lexical"]
    raw_excess = s.get("raw_pointer_total", 0) > fcfg.get("max_raw_pointers", 0)
    if (s["primary_total"] or raw_excess or lex["unsafe_keyword"] or lex["ffi"] or lex["raw_pointer_casts"]
            or lex["transmute"] or lex["pointer_ops"]):
        lines = ["## Safety (target: every unsafe metric == 0, including raw pointer declarations/dereferences; "
                 "no unsafe, raw pointers, transmute, FFI)"]
        if s.get("metrics"):
            lines.append("newmetrics: " + json.dumps({k: s["metrics"][k] for k in s["metrics"]}))
        if s.get("error"):
            lines.append("newmetrics did not run cleanly:\n" + s["error"][:1500])
        if raw_excess:
            lines.append(
                f"Raw pointers: newmetrics counts {s.get('raw_pointer_total')} raw pointer declarations/dereferences "
                "(e.g. `let p = Box::into_raw(..)`, locals or parameters of type `*mut T` / `*const T`). Replace them "
                "with safe ownership: `Option<Box<T>>`, indices into a `Vec`, or walking the list to its last element. "
                "If the interface itself declares a raw-pointer field, keep the field with that exact type but do not "
                "create raw pointers for it (leave it `None` and find the element safely instead).")
        for k in ("unsafe_keyword", "raw_pointer_casts", "transmute", "pointer_ops", "raw_pointer_ops", "ffi"):
            if lex.get(k):
                lines.append(f"{k}: {lex[k]}\n" + "\n".join(f"  - {x}" for x in lex["locations"][k]))
        parts.append("\n".join(lines))
        if v["build_ok"]:
            focus = focus or "removing unsafe code"
    cl = v["clippy"]
    if v["build_ok"] and cl.get("own"):
        groups = ", ".join(f"{g}: {cl['by_group'][g]['own']}" for g in cl.get("by_group", {}))
        body = "\n\n".join(f"[{x['group']}] {x['rendered']}" for x in cl["own"][:lim])
        parts.append(f"## Clippy warnings in your files ({len(cl['own'])}; {groups})\n" + truncate(body, sec_chars))
        focus = focus or "Clippy warnings"
    if v["build_ok"] and cl.get("rustc_own"):
        body = "\n\n".join(x["rendered"] for x in cl["rustc_own"][:lim])
        parts.append(f"## rustc warnings in your files ({len(cl['rustc_own'])})\n" + truncate(body, sec_chars // 2))
        focus = focus or "compiler warnings"
    if v["build_ok"] and cl.get("protected"):
        parts.append(f"(Note: {len(cl['protected'])} Clippy warnings are inside the protected benchmark test files; "
                     "you cannot change those files - ignore them.)")
    if lex.get("clippy_allow"):
        parts.append("## Lint suppressions found (`#[allow(clippy::...)]`) - fix the code instead unless the lint is "
                     "caused by a signature that the interface mandates:\n" +
                     "\n".join(f"- {x}" for x in lex["locations"]["clippy_allow"]))
    if lex.get("unwrap_expect_panic"):
        parts.append(f"(Idiomaticity note: {lex['unwrap_expect_panic']} uses of unwrap()/expect()/panic!; prefer "
                     "Option/Result handling, pattern matching and `?` where the failure is recoverable.)")
    if focus is None:
        focus = "nothing - all checks pass"
    parts.insert(1, f"PRIMARY FOCUS FOR THIS ITERATION: {focus}")
    return "\n\n".join(parts).strip()


def diag_to_dict(d: Diagnostic) -> dict:
    return {"level": d.level, "code": d.code, "message": d.message, "file": d.file, "line": d.line, "col": d.col,
            "rendered": d.rendered}
