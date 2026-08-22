"""三层流水线编排：规则引擎 → 学生小模型 → LLM 精判。"""
from __future__ import annotations

import time

from .config import CONFIDENCE_HIGH, PrimaryIntent
from .llm_refiner import llm_refiner
from .rule_engine import rule_engine
from .schemas import IntentResult
from .small_classifier import get_small_classifier


class IntentPipeline:
    """生产入口：classify(query) → IntentResult。"""

    def __init__(self, warmup: bool = True):
        self._small = None
        if warmup:
            _ = self.small  # 预加载小模型，避免首条请求慢

    @property
    def small(self):
        if self._small is None:
            self._small = get_small_classifier()
        return self._small

    def classify(self, query: str) -> IntentResult:
        query = (query or "").strip()
        if not query:
            return IntentResult(
                query=query, primary_intent=PrimaryIntent.UNKNOWN,
                handled_by="RULE", confidence=1.0,
                decision_trace=["空输入直接判 UNKNOWN"],
            )

        t0 = time.perf_counter()

        # ---- 第一层：规则引擎 ----
        hit = rule_engine.check(query)
        if hit is not None:
            hit.latency_ms = int((time.perf_counter() - t0) * 1000)
            return hit

        # ---- 第二层：学生小模型 ----
        out, small_ms = self.small.predict(query)
        trace = ["规则层未命中",
                 f"小模型({self.small.model_name.split('/')[-1]}, {small_ms}ms): "
                 f"{out.intent.value} conf={out.confidence}"]

        # ---- 高置信度直接返回（跳过昂贵的第三层）----
        if out.confidence >= CONFIDENCE_HIGH:
            return IntentResult(
                query=query,
                primary_intent=out.intent,
                confidence=out.confidence,
                handled_by="SMALL_MODEL",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                decision_trace=trace + [f"conf >= {CONFIDENCE_HIGH}，直接输出，跳过LLM"],
            )

        # ---- 第三层：LLM 精判 + 槽位 ----
        refined, llm_ms = llm_refiner.refine(
            query, {"intent": out.intent, "confidence": out.confidence}
        )
        refined.latency_ms = int((time.perf_counter() - t0) * 1000)
        refined.decision_trace = trace + [
            f"conf < {CONFIDENCE_HIGH}，进入LLM精判({llm_ms}ms)"
        ] + refined.decision_trace[1:]
        return refined


if __name__ == "__main__":
    r = IntentPipeline().classify("帮我看看这道二次函数的题怎么做")
    print(r.model_dump_json(indent=2))
