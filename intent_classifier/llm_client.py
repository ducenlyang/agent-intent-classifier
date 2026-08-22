"""共享 LLM 客户端：OpenAI 兼容 chat/completions，供 L3 精判与业务 Agent 生成共用。"""
from __future__ import annotations

import requests

from .config import LLM_API_KEY, LLM_BASE_URL, GEN_MODEL, GEN_TIMEOUT


def chat_completion(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.5,
    max_tokens: int = 1000,
    timeout: int | None = None,
) -> str:
    """调用生成模型，返回首条回复文本。失败抛异常，由调用方决定降级策略。"""
    resp = requests.post(
        f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": model or GEN_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout or GEN_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
