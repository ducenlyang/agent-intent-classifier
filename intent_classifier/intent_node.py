"""三层流水线编排（联合多任务版）：

L1 规则拦截 + 词典提示槽 rule_hint_slots
L2 联合tiny-bert：意图头(8分类) + BIO槽位头(subject/grade)
   intent_conf ≥ 0.85 且已检出的短槽位置信度全 ≥ 0.80
   → 放行：合并 rule_hint + bert_short 作为候选槽位输出
   否则 → L3：LLM 对意图与短槽位终审，并抽取全部长开放槽位 + 必填校验
"""
from __future__ import annotations

import time

from .config import (
    CONFIDENCE_HIGH,
    REQUIRED_SLOTS,
    SLOT_CONF_HIGH,
    PrimaryIntent,
)
from .llm_refiner import llm_refiner
from .rule_engine import rule_engine
from .schemas import IntentResult, Slots
from .small_classifier import get_small_classifier


def _missing(slots: Slots, intent: PrimaryIntent) -> list[str]:
    return [f for f in REQUIRED_SLOTS.get(intent, []) if not getattr(slots, f)]


def _merge_short_slots(hints: dict[str, str], bert: dict[str, dict]) -> tuple[Slots, dict[str, float]]:
    """短槽位合并：BIO 高置信值优先，其次规则提示槽。"""
    merged: dict[str, str | None] = {}
    conf: dict[str, float] = {}
    for field in ("subject", "grade"):
        cand = bert.get(field)
        if cand and cand["confidence"] >= SLOT_CONF_HIGH:
            merged[field] = cand["value"]
            conf[field] = cand["confidence"]
        else:
            merged[field] = hints.get(field)
    return Slots(subject=merged["subject"], grade=merged["grade"]), conf


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
        hints = rule_engine.hint_slots(query)

        # ---- 第二层：联合多任务小模型 ----
        out, small_ms = self.small.predict(query)
        slot_desc = "、".join(
            f"{f}={v['value']}({v['confidence']})" for f, v in out.bert_short_slots.items()
        ) or "无"
        trace = [
            "规则层未命中",
            f"提示槽: {hints}",
            f"小模型({self.small.model_name.split('/')[-1]}, {small_ms}ms): "
            f"{out.intent.value} conf={out.intent_confidence}, BIO槽位[{slot_desc}]",
        ]

        # ---- 置信分支：意图置信 + 已检出短槽位置信全部达标才放行 ----
        intent_ok = out.intent_confidence >= CONFIDENCE_HIGH
        weak_slots = [
            f for f, v in out.bert_short_slots.items()
            if v["confidence"] < SLOT_CONF_HIGH
        ]

        if intent_ok and not weak_slots:
            slots, slot_conf = _merge_short_slots(hints, out.bert_short_slots)
            missing = _missing(slots, out.intent)
            return IntentResult(
                query=query,
                primary_intent=out.intent,
                confidence=out.intent_confidence,
                handled_by="SMALL_MODEL",
                slots=slots,
                slot_confidence=slot_conf,
                missing_slots=missing,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                decision_trace=trace + [
                    f"意图conf≥{CONFIDENCE_HIGH}且短槽位全≥{SLOT_CONF_HIGH}，"
                    f"放行(合并 rule_hint + bert 槽位)"
                ] + ([f"必填槽位缺失: {missing}，待下游追问"] if missing else []),
            )

        # ---- 第三层：LLM/启发式 终审 + 长槽位抽取 ----
        reason = []
        if not intent_ok:
            reason.append(f"意图conf={out.intent_confidence}<{CONFIDENCE_HIGH}")
        if weak_slots:
            reason.append(f"槽位低置信{weak_slots}")
        refined, llm_ms = llm_refiner.refine(query, {
            "intent": out.intent,
            "confidence": out.intent_confidence,
            "rule_hint_slots": hints,
            "bert_short_slots": out.bert_short_slots,
        })
        refined.latency_ms = int((time.perf_counter() - t0) * 1000)
        refined.decision_trace = trace + [
            f"{'；'.join(reason)} → 送入LLM终审({llm_ms}ms)"
        ] + refined.decision_trace[1:]
        return refined


if __name__ == "__main__":
    r = IntentPipeline().classify("帮我看看这道二次函数的题怎么做")
    print(r.model_dump_json(indent=2))
