"""后端 LLM 客户端（与网关解耦，仅配置同源）：非流式 + SSE 流式。

两个入口都支持 llm_log：传入列表时自动追加一条完整调用留痕
{purpose, model, messages, output, latency_ms[, error]}，供轨迹面板展示。
"""
from __future__ import annotations

import json
import time

import requests

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT

_HEADERS = {"Authorization": f"Bearer {LLM_API_KEY}"}


def _log_call(llm_log, purpose, messages, output, t0, error=None):
    if llm_log is None:
        return
    llm_log.append({
        "purpose": purpose, "model": LLM_MODEL,
        "messages": [{"role": m.get("role"), "content": m.get("content")}
                     for m in messages],
        "output": output[:6000], "latency_ms": int((time.perf_counter() - t0) * 1000),
        **({"error": str(error)[:300]} if error else {}),
    })


def chat_completion(
    messages: list[dict],
    *,
    temperature: float = 0.5,
    max_tokens: int = 1200,
    llm_log: list | None = None,
    purpose: str = "LLM调用",
) -> str:
    """非流式调用，返回完整回复文本。失败抛异常，由调用方决定降级策略。"""
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers=_HEADERS,
            json={"model": LLM_MODEL, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        _log_call(llm_log, purpose, messages, content, t0)
        return content
    except Exception as e:
        _log_call(llm_log, purpose, messages, "", t0, error=e)
        raise


def chat_completion_stream(
    messages: list[dict],
    *,
    temperature: float = 0.5,
    max_tokens: int = 1200,
    llm_log: list | None = None,
    purpose: str = "LLM流式调用",
):
    """流式调用，逐块 yield 文本增量（打字机效果）。失败抛异常由调用方降级。"""
    t0 = time.perf_counter()
    parts: list[str] = []
    try:
        resp = requests.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers=_HEADERS,
            json={"model": LLM_MODEL, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens,
                  "stream": True},
            stream=True,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if delta:
                parts.append(delta)
                yield delta
        _log_call(llm_log, purpose, messages, "".join(parts), t0)
    except Exception as e:
        _log_call(llm_log, purpose, messages, "".join(parts), t0, error=e)
        raise
