"""后端 LLM 客户端（与网关解耦，仅配置同源）：非流式 + SSE 流式。"""
from __future__ import annotations

import json

import requests

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT

_HEADERS = {"Authorization": f"Bearer {LLM_API_KEY}"}


def chat_completion(
    messages: list[dict],
    *,
    temperature: float = 0.5,
    max_tokens: int = 1200,
) -> str:
    """非流式调用，返回完整回复文本。失败抛异常由调用方降级。"""
    resp = requests.post(
        f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers=_HEADERS,
        json={"model": LLM_MODEL, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def chat_completion_stream(
    messages: list[dict],
    *,
    temperature: float = 0.5,
    max_tokens: int = 1200,
):
    """流式调用，逐块 yield 文本增量（打字机效果）。"""
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
            yield delta
