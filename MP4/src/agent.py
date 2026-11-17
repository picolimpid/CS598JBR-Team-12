"""The LLM agent: a ReAct-style tool-calling loop for one repair iteration.

Within one iteration the model repeatedly chooses a tool (read C/Rust files, write or
edit Rust, run cargo check/test/clippy, run the unsafe-metrics tool, ...), sees the
tool output, and decides the next action, until it calls `finish` or runs out of
steps. The outer controller (pipeline.py) then runs a full verification and starts
the next iteration with that feedback.

Two tool protocols are supported:
  * native  - OpenAI function calling (`tools=[...]`), the default;
  * text    - for models/endpoints without function calling: the model writes
              ```tool {"name": ..., "arguments": {...}}``` blocks, and whole files as
              <write_file path="...">...</write_file> blocks.
If the endpoint rejects `tools`, the agent switches to text mode automatically.
"""

from __future__ import annotations

import json
import re

from llm import LLMClient, ToolsUnsupportedError
from rust_checks import truncate


class BudgetExceeded(Exception):
    pass


class Agent:
    def __init__(self, llm: LLMClient, toolbox, tool_schemas: list[dict], cfg: dict, traj, prompts: dict):
        self.llm = llm
        self.tools = toolbox
        self.schemas = tool_schemas
        self.cfg = cfg
        self.traj = traj
        self.prompts = prompts
        self.mode = cfg["llm"].get("tool_mode", "native")
        self.budget = cfg["budget"]
        self.ctx = cfg["context"]

    # ------------------------------------------------------------------ budget
    def _check_budget(self) -> None:
        if self.llm.calls >= self.budget.get("max_llm_calls", 10 ** 9):
            raise BudgetExceeded(f"max_llm_calls={self.budget['max_llm_calls']} reached")
        if self.llm.total_tokens >= self.budget.get("max_total_tokens", 10 ** 12):
            raise BudgetExceeded(f"max_total_tokens={self.budget['max_total_tokens']} reached")

    # ------------------------------------------------------------------ context compaction
    def _size(self, messages: list[dict]) -> int:
        n = 0
        for m in messages:
            n += len(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                n += len(tc["function"]["arguments"])
        return n

    def compact(self, messages: list[dict]) -> None:
        budget = self.ctx.get("max_context_chars", 400000)
        keep = self.ctx.get("keep_recent_messages", 8)
        if self._size(messages) <= budget:
            return
        elided = 0
        for m in messages[2:max(2, len(messages) - keep)]:
            if self._size(messages) <= budget:
                break
            c = m.get("content") or ""
            if m["role"] in ("tool", "user") and len(c) > 600 and not c.startswith("[elided"):
                m["content"] = f"[elided older tool output ({len(c)} chars) to save context]\n" + c[:300]
                elided += 1
            for tc in m.get("tool_calls") or []:
                args = tc["function"]["arguments"]
                if len(args) > 600:
                    try:
                        a = json.loads(args)
                    except json.JSONDecodeError:
                        a = {}
                    for k in ("content", "new_string", "old_string"):
                        if isinstance(a.get(k), str) and len(a[k]) > 200:
                            a[k] = f"[elided {len(a[k])} chars; read the file for its current content]"
                    tc["function"]["arguments"] = json.dumps(a)
                    elided += 1
            if m["role"] == "assistant" and len(c) > 2000:
                m["content"] = c[:1500] + "\n[... elided ...]"
                elided += 1
        if elided:
            self.traj.log(f"context compaction: elided {elided} old messages/arguments (now {self._size(messages)} chars)")

    # ------------------------------------------------------------------ main loop
    def run_iteration(self, messages: list[dict], max_steps: int) -> dict:
        tb = self.tools
        tb.finished = False
        tb.finish_summary = ""
        nudges = 0
        steps = 0
        stop_reason = "max_steps"
        while steps < max_steps:
            self._check_budget()
            self.compact(messages)
            steps += 1
            try:
                resp = self.llm.chat(messages, tools=self.schemas if self.mode == "native" else None)
            except ToolsUnsupportedError as exc:
                self.traj.log(f"endpoint rejected native tool calling ({exc}); switching to text tool protocol")
                self.mode = "text"
                messages[0]["content"] += "\n\n" + self.prompts["text_protocol"]
                steps -= 1
                continue
            self.traj.event("llm_response", step=steps, mode=self.mode, model=resp.model,
                            finish_reason=resp.finish_reason, usage=resp.usage, latency_s=resp.latency_s,
                            content=resp.content, reasoning=resp.reasoning,
                            tool_calls=[{"name": tc["function"]["name"],
                                         "arguments": truncate(tc["function"]["arguments"], 4000)}
                                        for tc in resp.tool_calls])
            short = (resp.content or "").strip().replace("\n", " ")
            self.traj.log(f"step {steps}: LLM ({resp.usage.get('prompt_tokens', '?')} in / "
                          f"{resp.usage.get('completion_tokens', '?')} out, {resp.latency_s}s) "
                          f"tools={[tc['function']['name'] for tc in resp.tool_calls]} "
                          f"{('text: ' + short[:200]) if short else ''}")

            if self.mode == "native":
                done = self._handle_native(messages, resp)
            else:
                done = self._handle_text(messages, resp)
            if done == "finish":
                stop_reason = "agent_called_finish"
                break
            if done == "no_action":
                nudges += 1
                if nudges > self.cfg["budget"].get("max_nudges", 3):
                    stop_reason = "agent_stopped_without_finish"
                    break
                if resp.finish_reason == "length":
                    messages.append({"role": "user", "content": self.prompts["truncated_nudge"]})
                else:
                    messages.append({"role": "user", "content": self.prompts["nudge"]})
            else:
                nudges = 0
        return {"steps": steps, "stop_reason": stop_reason, "summary": tb.finish_summary,
                "files_written": sorted(tb.files_written)}

    # ------------------------------------------------------------------ native protocol
    def _handle_native(self, messages: list[dict], resp) -> str:
        msg = {"role": "assistant", "content": resp.content or None}
        if resp.tool_calls:
            msg["tool_calls"] = resp.tool_calls
        messages.append(msg)
        if not resp.tool_calls:
            return "no_action"
        finished = False
        for tc in resp.tool_calls:
            name = tc["function"]["name"]
            raw = tc["function"]["arguments"] or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError as exc:
                result = (f"ERROR: the arguments of `{name}` are not valid JSON ({exc}). If you were writing a large "
                          "file, the output was probably cut off by the token limit: write the file in smaller parts "
                          "(write_file with the first part, then append_file for the rest) or use edit_file.")
                args = None
            if args is not None:
                result = self.tools.dispatch(name, args)
            self._log_tool(name, args, result)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            finished = finished or (name == "finish" and args is not None)
        return "finish" if finished else "acted"

    # ------------------------------------------------------------------ text protocol
    _TOOL_BLOCK = re.compile(r"```(?:tool|json)?\s*\n(\{.*?\})\s*\n```", re.S)
    _WRITE_BLOCK = re.compile(r"<(write_file|append_file)\s+path=\"([^\"]+)\"\s*>\n?(.*?)</\1>", re.S)

    def _handle_text(self, messages: list[dict], resp) -> str:
        text = resp.content or ""
        messages.append({"role": "assistant", "content": text})
        actions = []
        for m in self._WRITE_BLOCK.finditer(text):
            actions.append((m.start(), m.group(1), {"path": m.group(2), "content": m.group(3)}))
        masked = self._WRITE_BLOCK.sub(lambda m: " " * len(m.group(0)), text)
        for m in self._TOOL_BLOCK.finditer(masked):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                actions.append((m.start(), "__bad_json__", {"raw": m.group(1)[:300]}))
                continue
            if isinstance(obj, dict) and "name" in obj:
                actions.append((m.start(), obj["name"], obj.get("arguments") or {}))
        if not actions:
            return "no_action"
        actions.sort(key=lambda a: a[0])
        results, finished = [], False
        for i, (_, name, args) in enumerate(actions, 1):
            if name == "__bad_json__":
                result = "ERROR: could not parse this tool block as JSON: " + args["raw"]
            else:
                result = self.tools.dispatch(name, args)
            self._log_tool(name, args, result)
            shown = {k: (v if not isinstance(v, str) or len(v) < 200 else f"<{len(v)} chars>") for k, v in args.items()}
            results.append(f"### Result {i}: {name}({json.dumps(shown)})\n{result}")
            finished = finished or name == "finish"
        messages.append({"role": "user", "content": "Tool results:\n\n" + "\n\n".join(results)})
        return "finish" if finished else "acted"

    # ------------------------------------------------------------------ logging
    def _log_tool(self, name: str, args, result: str) -> None:
        shown = args
        if isinstance(args, dict):
            shown = {k: (v if not isinstance(v, str) or len(v) < 300 else f"<{len(v)} chars>") for k, v in args.items()}
        self.traj.event("tool_call", tool=name, arguments=shown,
                        result=truncate(result, self.cfg["tools"].get("log_tool_result_chars", 8000)))
        first = result.strip().splitlines()[0] if result.strip() else ""
        self.traj.log(f"  tool {name}({json.dumps(shown)[:160] if shown is not None else 'INVALID'}) -> {first[:160]}")
