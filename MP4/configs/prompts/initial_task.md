# Task: translate the C project `{{project_name}}` to safe, idiomatic Rust (iteration {{iteration}} of at most {{max_iterations}})

{{project_description}}

## Modules to implement
{{module_table}}

The C file hints are only name-based guesses; use `search` on `c_src` to find everything a module needs (e.g. test helpers defined in the C test files, macros in headers).

## Original C project (read-only, prefix `c_src/`)
```
{{c_tree}}
```

{{c_sources}}

## Current Rust crate
```
{{crate_tree}}
```

## Required Rust interface (specification; already copied into the module files with `unimplemented!()` bodies)
{{interfaces}}

## Benchmark tests (protected; they must compile and pass unchanged)
{{tests}}

## Plan
1. Read the C sources that correspond to each module (start with the headers, then the .c files and the C tests). Note data structures, invariants, constants and error codes.
2. Decide the safe Rust representation for each C data structure (e.g. pointer + length -> `Vec`/slice, linked nodes -> `Option<Box<T>>` or `Vec` with indices, `char*` -> `String`/`&str`, `FILE*` -> `std::fs::File`/`BufReader`, int flags -> `bool`/enums as fixed by the interface).
3. Write the complete implementation of every module (`write_file`), keeping all interface signatures.
4. `cargo_check` -> fix; `cargo_test` -> fix; `cargo_clippy` -> fix; `unsafe_metrics`; `check_interface`.
5. Call `finish` with a summary.

Start now by reading the C sources.
