"""三层流水线编排（联合多任务 + 置信短路版）：

L1 规则引擎：风险拦截(作弊/心理高危)，命中直接返回拒绝/安抚话术结束链路
L2 联合tiny-bert（双头）：意图+置信度 + BIO短槽位(subject/grade)+槽位置信度
    意图conf ≥ 0.85 且 已检出槽位conf ≥ 0.80 且 非复杂query
      → 短路放行：合并 rule_hint + BIO 槽位直接输出（几十毫秒，不调LLM）
    否则（低置信/槽位不稳/太复杂）
      → L3 LLM 精判：复核意图 + 终审短槽位 + 抽全部长槽位 + 必填校验
网关保持无状态：单句识别，多轮缓存归下游对话后端。
"""
from __future__ import annotations

import time

from .config import (
    COMPLEX_QUERY_LEN,
    CONFIDENCE_HIGH,
    REQUIRED_SLOTS,
    SLOT_CONF_HIGH,
    PrimaryIntent,
)
from .llm_refiner import LLMRefiner, llm_refiner
from .rule_engine import rule_engine
from .schemas import IntentResult, Slots
from .small_classifier import get_small_classifier
from .slot_lexicon import (
    GRADE_LEXICON,
    SUBJECT_LEXICON,
    extract_lexicon_slots,
    is_meta_request,
    rule_hint_slots,
)

_SHORT_SLOT_LEXICONS = {"subject": SUBJECT_LEXICON, "grade": GRADE_LEXICON}


def _normalize_bio_slots(bert: dict) -> list[str]:
    """BIO 短槽位词典归一化：span 不在词典时取内部最长词典子串（化学元素→化学），
    无子串命中则丢弃该槽位（防小模型编造词形污染下游与缓存）。"""
    fixes = []
    for field, lexicon in _SHORT_SLOT_LEXICONS.items():
        cand = bert.get(field)
        if not cand:
            continue
        raw = cand["value"]
        if raw in lexicon:
            continue
        inner = max((w for w in lexicon if w in raw), key=len, default=None)
        if inner:
            cand["value"] = inner
            fixes.append(f"{field}词形修正 {raw}→{inner}")
        else:
            del bert[field]
            fixes.append(f"{field}='{raw}'不在词典，丢弃")
    return fixes


def _merge_short_slots(hints: dict, bert: dict) -> tuple[Slots, dict[str, float]]:
    """短槽位合并：BIO 高置信值优先，其次 L1 词典提示。"""
    values: dict[str, str | None] = {}
    conf: dict[str, float] = {}
    for field in ("subject", "grade"):
        cand = bert.get(field)
        if cand and cand["confidence"] >= SLOT_CONF_HIGH:
            values[field] = cand["value"]
            conf[field] = cand["confidence"]
        else:
            values[field] = hints.get(field)
    return Slots(subject=values["subject"], grade=values["grade"]), conf


def _fill_question_text(slots: Slots, intent: PrimaryIntent, query: str) -> Slots:
    """答疑意图：question_text 缺省时补原文；"要题"元请求保持置空触发反问。"""
    if (intent is PrimaryIntent.QUESTION_SUBJECT
            and not slots.question_text and not is_meta_request(query)):
        slots.question_text = query
    return slots


def _finalize(result: IntentResult) -> IntentResult:
    """统一收尾：question_text 策略 + 必填校验 + guide 标记。"""
    result.slots = _fill_question_text(result.slots, result.primary_intent, result.query)
    result.missing_slots = [
        f for f in REQUIRED_SLOTS.get(result.primary_intent, [])
        if not getattr(result.slots, f)
    ]
    result.need_guide_only = result.primary_intent is PrimaryIntent.QUESTION_SUBJECT
    return result


class IntentPipeline:
    """生产入口：classify(query) → IntentResult。网关无状态。

    use_llm: None=按配置(有Key即启用)；False=强制启发式(离线/省配额)。
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
                handled_by="SMALL_MODEL", confidence=1.0,
                decision_trace=["空输入直接判 UNKNOWN"],
            )

        t0 = time.perf_counter()

        # ---- L1 规则引擎：命中即结束链路 ----
        hit = rule_engine.check(query)
        if hit is not None:
            hit.latency_ms = int((time.perf_counter() - t0) * 1000)
            return _finalize(hit)

        # ---- L2 联合双头：意图候选 + BIO 短槽位 ----
        hints = rule_hint_slots(query)
        out, small_ms = self.small.predict(query)
        fixes = _normalize_bio_slots(out.bert_short_slots)
        slot_desc = "、".join(
            f"{f}={v['value']}({v['confidence']})" for f, v in out.bert_short_slots.items()
        ) or "无"
        trace = [
            "规则层未命中",
            f"提示槽: { {k: v for k, v in hints.items() if v} or '无' }",
            f"小模型({self.small.model_name.split('/')[-1]}, {small_ms}ms): "
            f"{out.intent.value} conf={out.intent_confidence}, BIO槽位[{slot_desc}]",
        ]
        if fixes:
            trace.append(f"BIO词形归一化: {'；'.join(fixes)}")

        # ---- 置信短路：全部达标且非复杂query → 不调LLM直接输出 ----
        intent_ok = out.intent_confidence >= CONFIDENCE_HIGH
        weak_slots = [f for f, v in out.bert_short_slots.items()
                      if v["confidence"] < SLOT_CONF_HIGH]
        complex_q = len(query) > COMPLEX_QUERY_LEN
        # 分歧即升级：规则词典与BIO在归一化后仍不一致 → 不短路，交LLM终审
        # 多学科歧义升级："数学物理都不好先看物理"词典取首个命中(数学)但语境
        # 指向另一个(物理) → 不短路，交LLM终审裁定
        multi_subject = sum(
            1 for w in SUBJECT_LEXICON[:10] if w in query
        ) >= 2
        disagree = [
            f for f in ("subject", "grade")
            if hints.get(f) and (out.bert_short_slots.get(f) or {}).get("value")
            and out.bert_short_slots[f]["value"] != hints[f]
        ]

        if intent_ok and not weak_slots and not complex_q and not disagree and not multi_subject:
            slots, slot_conf = _merge_short_slots(hints, out.bert_short_slots)
            # 短路跳过了L3，长槽位用零成本词典抽取补齐(否则"寒假"类时间/主题丢失)
            lex = extract_lexicon_slots(query)
            slots.topic = slots.topic or lex["topic"]
            slots.emotion = slots.emotion or lex["emotion"]
            slots.time_horizon = slots.time_horizon or lex["time_horizon"]
            result = IntentResult(
                query=query,
                primary_intent=out.intent,
                confidence=out.intent_confidence,
                handled_by="SMALL_MODEL",
                slots=slots,
                slot_confidence=slot_conf,
                decision_trace=trace + [
                    f"意图conf≥{CONFIDENCE_HIGH}且槽位全≥{SLOT_CONF_HIGH}且非复杂query，"
                    f"短路放行(合并 rule_hint + BIO 槽位，未调LLM)"
                ],
            )
            _finalize(result)
            result.latency_ms = int((time.perf_counter() - t0) * 1000)
            return result

        # ---- L3 LLM 精判：复核意图 + 终审短槽位 + 长槽位 ----
        reasons = []
        if not intent_ok:
            reasons.append(f"意图conf={out.intent_confidence}<{CONFIDENCE_HIGH}")
        if weak_slots:
            reasons.append(f"槽位低置信{weak_slots}")
        if complex_q:
            reasons.append(f"复杂query(>{COMPLEX_QUERY_LEN}字)")
        if multi_subject:
            reasons.append("多学科命中(语境指向歧义)")
        if disagree:
            reasons.append(f"规则槽与BIO槽分歧{disagree}(rule={ {f: hints[f] for f in disagree} },"
                           f"bio={ {f: out.bert_short_slots[f]['value'] for f in disagree} })")
        refined, llm_ms = self.refiner.refine(query, {
            "intent": out.intent,
            "confidence": out.intent_confidence,
            "rule_hint_slots": hints,
            "bert_short_slots": out.bert_short_slots,
        })
        # 终审后的槽位置信：未被LLM改写的BIO槽位保留其置信度
        refined.slot_confidence = {
            f: v["confidence"] for f, v in out.bert_short_slots.items()
            if getattr(refined.slots, f) == v["value"]
        }
        refined.latency_ms = int((time.perf_counter() - t0) * 1000)
        refined.decision_trace = trace + [
            f"{'；'.join(reasons)} → 升级LLM精判({llm_ms}ms)"
        ] + refined.decision_trace[1:]
        return _finalize(refined)


if __name__ == "__main__":
    r = IntentPipeline().classify("帮我看看这道二次函数的题怎么做")
    print(r.model_dump_json(indent=2))
