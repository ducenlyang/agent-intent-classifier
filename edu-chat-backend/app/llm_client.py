"""后端自己的 LLM 客户端（与网关解耦，仅配置同源）。"""
from __future__ import annotations

import requests

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT


def chat_completion(
    messages: list[dict],
    *,
    temperature: float = 0.5,
    max_tokens: int = 1200,
) -> str:
    """非流式调用，返回完整回复文本。失败抛异常由调用方降级。"""
    resp = requests.post(
        f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
