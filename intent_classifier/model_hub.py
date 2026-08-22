"""HuggingFace 模型加载辅助。

必须在进程内其它模块 import transformers 之前导入本模块：
若 huggingface.co 不可达（国内网络），自动切换 hf-mirror.com 镜像。
用户可用环境变量 INTENT_HF_ENDPOINT / HF_ENDPOINT 强制指定。
"""
from __future__ import annotations

import os
import urllib.request


def _hf_reachable(timeout: float = 4.0) -> bool:
    try:
        req = urllib.request.Request("https://huggingface.co", method="HEAD")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


if os.getenv("INTENT_HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("INTENT_HF_ENDPOINT")
elif "HF_ENDPOINT" not in os.environ and not _hf_reachable():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 在设置好 HF_ENDPOINT 之后再导入 transformers，镜像配置才会生效
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .config import STUDENT_MODEL_CANDIDATES

_student_name_cache: str | None = None


def student_model_name() -> str:
    """按候选顺序探测可用的小模型名（hfl/chinese-bert-wwm-ext-tiny 优先，缺省回退 hfl/rbt3）。"""
    global _student_name_cache
    if _student_name_cache:
        return _student_name_cache
    for name in STUDENT_MODEL_CANDIDATES:
        try:
            AutoConfig.from_pretrained(name)
            _student_name_cache = name
            return name
        except Exception:
            continue
    # 探测全失败（离线且无缓存）时返回首选名，让上层报出原始错误
    return STUDENT_MODEL_CANDIDATES[0]


__all__ = [
    "AutoConfig",
    "AutoModel",
    "AutoModelForSequenceClassification",
    "AutoTokenizer",
    "student_model_name",
]
