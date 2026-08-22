"""edu-chat-backend 全局配置。"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 意图网关（agent-intent-classifier，无状态 HTTP 服务）
#   GATEWAY_URL : 服务地址（uvicorn intent_classifier.api:app --port 8601）
#   GATEWAY_PATH: 网关仓库路径(本解决方案内即上级目录)，仅用于 import 其数据模型
# ---------------------------------------------------------------------------
GATEWAY_URL = os.getenv("INTENT_GATEWAY_URL", "http://127.0.0.1:8601")
GATEWAY_PATH = Path(
    os.getenv("INTENT_GATEWAY_PATH", "") or str(ROOT_DIR.parent)
)
GATEWAY_TIMEOUT = int(os.getenv("INTENT_GATEWAY_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# LLM（OpenAI 兼容；与网关仓库 config.local.json 同格式，本仓库自己的副本）
# ---------------------------------------------------------------------------
_LOCAL = ROOT_DIR / "config.local.json"


def _load() -> dict:
    if not _LOCAL.exists():
        return {}
    try:
        with open(_LOCAL, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[config] config.local.json 解析失败(忽略): {e}")
        return {}


_cfg = _load().get("llm", {})

LLM_API_KEY = os.getenv("INTENT_LLM_API_KEY", "") or str(_cfg.get("api_key") or "")
LLM_BASE_URL = (
    os.getenv("INTENT_LLM_BASE_URL", "")
    or str(_cfg.get("base_url") or "")
    or "https://chatapi.weixin.qq.com/openai/v1"
)
LLM_MODEL = (
    os.getenv("INTENT_LLM_MODEL", "")
    or str(_cfg.get("model") or "")
    or "Deepseek-v4-flash"
)
_timeout_src = os.getenv("INTENT_LLM_TIMEOUT") or _cfg.get("timeout")
LLM_TIMEOUT = int(_timeout_src) if _timeout_src else 60

HISTORY_WINDOW = 6  # Agent 生成时携带的最近对话条数
