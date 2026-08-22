"""共享 LLM 客户端：OpenAI 兼容接口，供 L3 精判与业务 Agent 生成共用。

chat_completion        非流式（L3 JSON 精判等一次性场景）
chat_completion_stream 流式 SSE，逐块 yield 文本增量（Agent 打字机输出）
"""
from __future__ import annotations

import json

import requests

from .config import LLM_API_KEY, LLM_BASE_URL, GEN_MODEL, GEN_TIMEOUT

_HEADERS = {"Authorization": f"Bearer {LLM_API_KEY}"}


def _payload(messages, model, temperature, max_tokens, stream: bool) -> dict:
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def chat_completion(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.5,
    max_tokens: int = 1000,
    timeout: int | None = None,
) -> str:
    """非流式调用，返回完整回复文本。失败抛异常，由调用方决定降级策略。"""
    resp = requests.post(
        f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers=_HEADERS,
        json=_payload(messages, model or GEN_MODEL, temperature, max_tokens, False),
        timeout=timeout or GEN_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def chat_completion_stream(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.5,
    max_tokens: int = 1000,
    timeout: int | None = None,
):
    """流式调用，逐块 yield 文本增量（打字机效果）。失败抛异常由调用方降级。"""
    resp = requests.post(
        f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers=_HEADERS,
        json=_payload(messages, model or GEN_MODEL, temperature, max_tokens, True),
        stream=True,
        timeout=timeout or GEN_TIMEOUT,
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
            continue  # 容忍心跳/注释行
        choices = chunk.get("choices") or [{}]
        delta = (choices[0].get("delta") or {}).get("content")
        if delta:
            yield delta
