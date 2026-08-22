"""三层线性流水线编排：

L1 规则引擎：风险拦截(作弊/心理高危)，命中直接返回拒绝/安抚话术结束链路
L2 tiny-bert：仅输出意图+置信度（作为候选）
L3 LLM精判：复核意图 + 抽取全部槽位 subject/grade/question_text/... + 必填校验
   （无 LLM Key 时自动降级启发式，词典抽槽，流水线不中断）
"""
from __future__ import annotations

import time

from .config import PrimaryIntent
from .llm_refiner import LLMRefiner, llm_refiner
from .rule_engine import rule_engine
from .schemas import IntentResult
from .small_classifier import get_small_classifier


class IntentPipeline:
    """生产入口：classify(query) → IntentResult。

    use_llm: None=按配置(有Key即启用)；False=强制启发式(离线/省配额)；
             True=强制LLM(无Key时报错走降级)。
    """

    def __init__(self, warmup: bool = True, use_llm: bool | None = None):
        self._small = None
        if use_llm is None:
            self.refiner = llm_refiner
        else:
            self.refiner = LLMRefiner(api_key=llm_refiner.api_key if use_llm else "")
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
                handled_by="LLM_FALLBACK", confidence=1.0,
                decision_trace=["空输入直接判 UNKNOWN"],
            )

        t0 = time.perf_counter()

        # ---- L1 规则引擎：命中即结束链路 ----
        hit = rule_engine.check(query)
        if hit is not None:
            hit.latency_ms = int((time.perf_counter() - t0) * 1000)
            return hit

        # ---- L2 小模型：意图候选 ----
        out, small_ms = self.small.predict(query)
        trace = [
            "规则层未命中",
            f"小模型({self.small.model_name.split('/')[-1]}, {small_ms}ms): "
            f"{out.intent.value} conf={out.intent_confidence}",
        ]

        # ---- L3 LLM 精判：复核意图 + 全量槽位 + 必填校验 ----
        refined, llm_ms = self.refiner.refine(
            query, {"intent": out.intent, "confidence": out.intent_confidence}
        )
        refined.latency_ms = int((time.perf_counter() - t0) * 1000)
        refined.decision_trace = trace + [
            f"LLM精判({llm_ms}ms)" if refined.handled_by == "LLM_REFINE"
            else f"启发式精判({llm_ms}ms)"
        ] + refined.decision_trace[1:]
        return refined


if __name__ == "__main__":
    r = IntentPipeline().classify("帮我看看这道二次函数的题怎么做")
    print(r.model_dump_json(indent=2))
