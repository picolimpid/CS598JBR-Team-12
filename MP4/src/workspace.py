"""Translation workspace management.

Layout produced for a project (e.g. project_1 = satc):

  subjects/project_1/c/          original C project            (read-only for the agent, path prefix `c_src/`)
  subjects/project_1/rust_spec/  CRUST-Bench Rust interface+tests (read-only, path prefix `rust_spec/`)
  translations/project_1/        the Rust crate the agent edits  (paths relative to the crate root)

Initialisation is fully generic (works for any CRUST-Bench project):
  * Cargo.toml, Cargo.lock, src/lib.rs and src/bin/*.rs (benchmark tests) are copied from rust_spec/;
  * every interface file src/interfaces/<m>.rs becomes the starting point of src/<m>.rs
    (src/main.rs for `main`, which cargo builds as the binary target);
  * lib.rs gets a `pub mod <m>;` line for every interface module that it does not declare yet;
  * the benchmark tests and Cargo manifest are recorded as protected (hash-checked, never writable).
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Workspace:
    def __init__(self, mp4_root: Path, project_key: str, project_cfg: dict, policy: dict):
        self.root = Path(mp4_root)
        self.key = project_key
        self.cfg = project_cfg
        self.policy = policy
        self.name = project_cfg["name"]
        self.c_dir = self.root / project_cfg["c_dir"]
        self.spec_dir = self.root / project_cfg["spec_dir"]
        self.crate = self.root / project_cfg["translation_dir"]
        self.traj_dir = self.root / project_cfg["trajectory_dir"]
        if not self.c_dir.is_dir() or not self.spec_dir.is_dir():
            raise FileNotFoundError(f"subject directories missing for {project_key}: {self.c_dir} / {self.spec_dir}")

    # ------------------------------------------------------------------ discovery
    def interface_files(self) -> list[Path]:
        return sorted((self.spec_dir / "src" / "interfaces").glob("*.rs"))

    def module_map(self) -> list[dict]:
        out = []
        for f in self.interface_files():
            m = f.stem
            target = "src/main.rs" if m == "main" else f"src/{m}.rs"
            related = [p.relative_to(self.c_dir).as_posix() for p in sorted(self.c_dir.rglob("*"))
                       if p.is_file() and p.suffix in (".c", ".h") and m.lower() in p.stem.lower()]
            out.append({"module": m, "interface": f"rust_spec/src/interfaces/{f.name}", "target": target,
                        "c_files_hint": [f"c_src/{r}" for r in related]})
        return out

    def protected_files(self) -> list[str]:
        prot = ["Cargo.toml", "Cargo.lock"]
        bin_dir = self.spec_dir / "src" / "bin"
        if bin_dir.is_dir():
            prot += [f"src/bin/{p.relative_to(bin_dir).as_posix()}" for p in sorted(bin_dir.rglob("*")) if p.is_file()]
        tests_dir = self.spec_dir / "tests"
        if tests_dir.is_dir():
            prot += [f"tests/{p.relative_to(tests_dir).as_posix()}" for p in sorted(tests_dir.rglob("*")) if p.is_file()]
        return [p for p in prot if (self.spec_dir / p).exists()]

    def test_files(self) -> list[str]:
        return [p for p in self.protected_files() if p.endswith(".rs")]

    def own_files(self) -> list[str]:
        prot = set(self.protected_files())
        files = []
        for p in sorted((self.crate / "src").rglob("*.rs")):
            rel = p.relative_to(self.crate).as_posix()
            if rel not in prot and not rel.startswith("src/interfaces/"):
                files.append(rel)
        return files

    # ------------------------------------------------------------------ init
    def init(self, fresh: bool) -> list[str]:
        """Create the crate from the spec. Returns a list of human-readable setup actions."""
        actions = []
        if fresh and self.crate.exists():
            shutil.rmtree(self.crate)
            actions.append(f"removed previous {self.crate.relative_to(self.root)}")
        if (self.crate / "Cargo.toml").exists() and not fresh:
            actions.append("reusing existing crate (resume)")
            self.restore_protected(actions)
            return actions
        (self.crate / "src").mkdir(parents=True, exist_ok=True)
        for rel in self.protected_files():
            dst = self.crate / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.spec_dir / rel, dst)
            actions.append(f"copied protected {rel}")
        lib_src = self.spec_dir / "src" / "lib.rs"
        lib_text = lib_src.read_text(encoding="utf-8") if lib_src.exists() else ""
        for item in self.module_map():
            src = self.spec_dir / "src" / "interfaces" / f"{item['module']}.rs"
            dst = self.crate / item["target"]
            body = src.read_text(encoding="utf-8")
            if item["module"] != "main" and not re.search(rf"\bmod\s+{re.escape(item['module'])}\b", lib_text):
                lib_text = lib_text.rstrip("\n") + ("\n" if lib_text.strip() else "") + f"pub mod {item['module']};\n"
                actions.append(f"declared `pub mod {item['module']};` in src/lib.rs")
            if item["module"] == "main":
                body = self._main_preamble(body)
            dst.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")
            actions.append(f"seeded {item['target']} from {item['interface']}")
        if self.policy.get("inject_forbid_unsafe", True) and "forbid(unsafe_code)" not in lib_text:
            lib_text = "#![forbid(unsafe_code)]\n" + lib_text
            actions.append("added #![forbid(unsafe_code)] to src/lib.rs (pipeline safety policy)")
        (self.crate / "src" / "lib.rs").write_text(lib_text, encoding="utf-8")
        (self.crate / ".gitignore").write_text("/target\n", encoding="utf-8")
        return actions

    def _main_preamble(self, body: str) -> str:
        """A binary root cannot see the library modules unless it imports them via the crate name."""
        crate = re.search(r'^\s*name\s*=\s*"([^"]+)"', (self.spec_dir / "Cargo.toml").read_text(encoding="utf-8"), re.M)
        cname = crate.group(1).replace("-", "_") if crate else self.name
        pre = ""
        if self.policy.get("inject_forbid_unsafe", True):
            pre += "#![forbid(unsafe_code)]\n"
        pre += f"#[allow(unused_imports)]\nuse {cname}::*;\n\n"
        return pre + body

    def restore_protected(self, actions: list[str] | None = None) -> list[str]:
        """Re-copy any protected file whose hash differs from the spec. Returns restored paths."""
        restored = []
        for rel in self.protected_files():
            src, dst = self.spec_dir / rel, self.crate / rel
            if not dst.exists() or sha256(src) != sha256(dst):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                restored.append(rel)
        if actions is not None and restored:
            actions.append(f"restored protected files: {restored}")
        return restored

    # ------------------------------------------------------------------ interface conformance
    def interface_report(self) -> list[str]:
        """All interface-conformance problems of the current crate (empty list == conformant)."""
        from rust_checks import interface_problems  # local import keeps workspace.py dependency-free

        probs = []
        lib = self.crate / "src" / "lib.rs"
        lib_text = lib.read_text(encoding="utf-8") if lib.exists() else ""
        for item in self.module_map():
            target = self.crate / item["target"]
            iface = (self.spec_dir / "src" / "interfaces" / f"{item['module']}.rs").read_text(encoding="utf-8")
            if not target.exists():
                probs.append(f"{item['target']}: file missing")
                continue
            probs += [f"{item['target']}: {p}" for p in interface_problems(iface, target.read_text(encoding="utf-8"))]
            if item["module"] != "main" and not re.search(rf"\bpub\s+mod\s+{re.escape(item['module'])}\b", lib_text):
                probs.append(f"src/lib.rs must declare `pub mod {item['module']};`")
        if self.policy.get("inject_forbid_unsafe", True) and not re.search(r"#!\[\s*forbid\s*\(\s*unsafe_code\s*\)\s*\]", lib_text):
            probs.append("src/lib.rs must keep `#![forbid(unsafe_code)]` (pipeline safety policy)")
        return probs

    # ------------------------------------------------------------------ snapshots
    def snapshot(self, dest: Path) -> None:
        if dest.exists():
            shutil.rmtree(dest)
        for rel in self.own_files():
            d = dest / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.crate / rel, d)

    def restore_snapshot(self, src: Path) -> None:
        for rel in self.own_files():
            (self.crate / rel).unlink()
        for p in src.rglob("*.rs"):
            rel = p.relative_to(src)
            d = self.crate / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d)

    # ------------------------------------------------------------------ prompt helpers
    def file_tree(self, base: Path, prefix: str, skip_dirs=("target", ".git")) -> str:
        lines = []
        for p in sorted(base.rglob("*")):
            if not p.is_file() or any(part in skip_dirs for part in p.relative_to(base).parts):
                continue
            rel = p.relative_to(base).as_posix()
            try:
                n = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                n = 0
            lines.append(f"{prefix}{rel}  ({n} lines)")
        return "\n".join(lines)

    def read_rel(self, base: Path, rel: str) -> str:
        return (base / rel).read_text(encoding="utf-8", errors="replace")
