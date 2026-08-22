"""三层蒸馏意图识别：规则引擎 → tiny-bert 小模型 → LLM 精判。"""
from .config import PrimaryIntent, SecondaryIntent  # noqa: F401
from .schemas import IntentResult, RiskFlag, Slots  # noqa: F401


def build_pipeline():
    """库用法入口：from intent_classifier import build_pipeline"""
    from .intent_node import IntentPipeline
    return IntentPipeline()
