"""Minimal OpenAI-compatible chat-completions client.

Works with any endpoint that speaks the OpenAI `/chat/completions` protocol:
OpenRouter (default), Google Gemini's OpenAI endpoint, DeepSeek, Together, Groq,
or a local vLLM / Ollama / llama.cpp server. The model and endpoint are chosen in
MP4/configs/config*.json; the API key is read from an environment variable whose
*name* is in the config (the key itself is never written to disk).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests


class LLMError(RuntimeError):
    pass


class LLMQuotaError(LLMError):
    """Raised when retries are exhausted on rate-limit/quota errors (resume later)."""


class ToolsUnsupportedError(LLMError):
    """Raised when the endpoint rejects the `tools` parameter (switch to text mode)."""


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict]
    finish_reason: str | None
    reasoning: str | None
    usage: dict
    model: str
    latency_s: float
    raw: dict = field(default_factory=dict)


class LLMClient:
    def __init__(self, cfg: dict, trajectory=None):
        self.cfg = cfg
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.traj = trajectory
        key_env = cfg.get("api_key_env") or ""
        self.api_key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not self.api_key and not cfg.get("allow_missing_key", False):
            raise LLMError(
                f"Environment variable {key_env} is not set. Export your API key, e.g. "
                f"`export {key_env}=...` (never commit it)."
            )
        if trajectory is not None:
            trajectory.add_secret(self.api_key)
        self.timeout = cfg.get("request_timeout_s", 600)
        self.max_retries = cfg.get("max_retries", 8)
        self.min_interval = cfg.get("min_seconds_between_requests", 0)
        self._last_request = 0.0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # ------------------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.cfg.get("extra_headers", {}) or {})
        return h

    def _throttle(self) -> None:
        wait = self.min_interval - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()

    def _prepare(self, messages: list[dict]) -> list[dict]:
        """Copy of the conversation as sent on the wire: no null contents, optional system suffix."""
        out = []
        suffix = self.cfg.get("system_prompt_suffix") or ""
        for i, m in enumerate(messages):
            m = dict(m)
            if m.get("content") is None:
                m["content"] = ""
            if i == 0 and suffix and m.get("role") == "system":
                m["content"] = m["content"] + "\n\n" + suffix
            out.append(m)
        return out

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._prepare(messages),
            "temperature": self.cfg.get("temperature", 0.2),
            "max_tokens": self.cfg.get("max_tokens", 16384),
        }
        if self.cfg.get("top_p") is not None:
            body["top_p"] = self.cfg["top_p"]
        if self.cfg.get("seed") is not None:
            body["seed"] = self.cfg["seed"]
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
            if self.cfg.get("parallel_tool_calls") is not None:
                body["parallel_tool_calls"] = self.cfg["parallel_tool_calls"]
        body.update(self.cfg.get("extra_body", {}) or {})

        url = f"{self.base_url}/chat/completions"
        last_err = ""
        quota_like = False
        for attempt in range(self.max_retries + 1):
            self._throttle()
            t0 = time.time()
            try:
                resp = requests.post(url, headers=self._headers(), json=body, timeout=self.timeout)
            except requests.RequestException as exc:
                last_err = f"network error: {exc}"
                self._backoff(attempt, last_err)
                continue
            latency = time.time() - t0
            text = resp.text
            if resp.status_code == 429 or resp.status_code in (408, 500, 502, 503, 504, 520, 522, 524, 529):
                quota_like = resp.status_code == 429
                last_err = f"HTTP {resp.status_code}: {text[:500]}"
                self._backoff(attempt, last_err, resp.headers.get("Retry-After"))
                continue
            if resp.status_code in (400, 404, 422) and tools and _mentions_tools(text):
                raise ToolsUnsupportedError(f"HTTP {resp.status_code}: {text[:800]}")
            if resp.status_code >= 400:
                raise LLMError(f"HTTP {resp.status_code}: {text[:1500]}")
            try:
                data = resp.json()
            except ValueError:
                last_err = f"invalid JSON body: {text[:300]}"
                self._backoff(attempt, last_err)
                continue
            if data.get("error"):
                err = data["error"]
                code = err.get("code") if isinstance(err, dict) else None
                last_err = f"provider error: {json.dumps(err)[:800]}"
                quota_like = code == 429 or "rate" in last_err.lower() or "quota" in last_err.lower()
                if tools and _mentions_tools(last_err) and code in (400, 404, 422):
                    raise ToolsUnsupportedError(last_err)
                self._backoff(attempt, last_err)
                continue
            choices = data.get("choices") or []
            if not choices:
                last_err = f"empty choices: {text[:300]}"
                self._backoff(attempt, last_err)
                continue
            msg = choices[0].get("message") or {}
            usage = data.get("usage") or {}
            self.calls += 1
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            tool_calls = []
            for i, tc in enumerate(msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args if args is not None else {})
                tool_calls.append({
                    "id": tc.get("id") or f"call_{self.calls}_{i}",
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args},
                })
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            content = msg.get("content") or ""
            # local reasoning models (e.g. Qwen3 on Ollama) may inline their thinking; keep it out of the context
            thinks = re.findall(r"<think>(.*?)</think>", content, re.S)
            if thinks:
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S)
                reasoning = (reasoning + "\n" if isinstance(reasoning, str) else "") + "\n".join(t.strip() for t in thinks)
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=choices[0].get("finish_reason"),
                reasoning=reasoning if isinstance(reasoning, str) else None,
                usage=usage,
                model=data.get("model", self.model),
                latency_s=round(latency, 2),
                raw={k: v for k, v in data.items() if k not in ("choices",)},
            )
        if quota_like:
            raise LLMQuotaError(f"rate limit / quota exhausted after {self.max_retries} retries: {last_err}")
        raise LLMError(f"request failed after {self.max_retries} retries: {last_err}")

    def _backoff(self, attempt: int, reason: str, retry_after: str | None = None) -> None:
        if attempt >= self.max_retries:
            return
        delay = min(self.cfg.get("max_backoff_s", 120), (2 ** attempt) * 2 + random.uniform(0, 2))
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        if self.traj is not None:
            self.traj.log(f"LLM request retry {attempt + 1}/{self.max_retries} in {delay:.1f}s ({reason[:200]})")
        time.sleep(delay)


def _mentions_tools(text: str) -> bool:
    t = text.lower()
    return "tool" in t and ("support" in t or "not allowed" in t or "invalid" in t or "unknown" in t)
