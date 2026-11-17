# Tool protocol (text mode)
Native function calling is not available, so call tools by writing blocks in your reply. The controller executes every block in order and replies with all results.

To call any tool, write a fenced block tagged `tool` containing one JSON object:
```tool
{"name": "read_file", "arguments": {"path": "c_src/foo.h", "start_line": 1, "end_line": 200}}
```

To create/overwrite or append to a source file, do NOT put the code in JSON; use these blocks instead (raw file content between the tags):
<write_file path="src/example.rs">
pub fn example() -> i32 {
    42
}
</write_file>

<append_file path="src/example.rs">
// more code
</append_file>

Available tools: list_files(path), read_file(path, start_line?, end_line?), search(pattern, path?), write_file(path, content), append_file(path, content), edit_file(path, old_string, new_string, replace_all?), cargo_check(), cargo_build(), cargo_test(test_name?), cargo_clippy(), unsafe_metrics(), check_interface(), finish(summary).

Use a few tool blocks per reply, then wait for the results. End the iteration with the `finish` tool.
