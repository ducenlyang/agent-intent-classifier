"""意图网关 HTTP 客户端：POST {GATEWAY_URL}/classify，无状态单句识别。

网关职责(只做这些)：解析当前一句 → 意图/槽位抽取/单轮missing_slots/风险。
多轮缓存合并、反问、Agent 编排全部在本后端。
数据模型 IntentResult/Slots 直接复用网关仓库定义(单一数据源)。
"""
from __future__ import annotations

import sys

import requests

from .config import GATEWAY_PATH, GATEWAY_TIMEOUT, GATEWAY_URL

if str(GATEWAY_PATH) not in sys.path:
    sys.path.insert(0, str(GATEWAY_PATH))

from intent_classifier.schemas import IntentResult, Slots  # noqa: E402

_healthy = False


def classify(query: str) -> IntentResult:
    """调用网关，返回 IntentResult 模型实例。"""
    global _healthy
    resp = requests.post(
        f"{GATEWAY_URL.rstrip('/')}/classify",
        json={"query": query},
        timeout=GATEWAY_TIMEOUT,
    )
    resp.raise_for_status()
    _healthy = True
    return IntentResult.model_validate(resp.json())


def healthy() -> bool:
    return _healthy
