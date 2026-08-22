"""第一层：规则引擎（硬拦截 + 词典提示槽）。

作弊类关键词 → REFUSE_CHEAT；心理高危关键词 → CHAT_EMOTION + psych_risk。
命中直接返回 IntentResult；未命中放行进入第二层。
同时提供 rule_hint_slots：词典捞取 subject/grade 提示槽位（供 L2 合并、L3 终审）。
"""
from __future__ import annotations

import time

from .config import PrimaryIntent, SecondaryIntent
from .schemas import IntentResult, RiskFlag, Slots
from .slot_lexicon import rule_hint_slots as _rule_hint_slots

# 作弊类关键词：直接判 REFUSE_CHEAT（下游回复器负责礼貌拒绝+引导）
CHEAT_KEYWORDS: list[str] = [
    "作弊", "代考", "代写", "替考", "枪手", "买答案", "卖答案", "求答案",
    "泄题", "漏题", "押题答案", "试题答案", "考试答案", "真题答案", "答案发我",
    "小抄", "抄袭", "传答案", "发答案", "发我答案", "拍答案", "偷看",
    "作弊器", "作弊软件", "暗号", "作弊码", "保过包过", "内部试卷",
    "提前拿到卷子", "弄到试卷", "隐形耳机", "替我去考试",
]

# 心理高危关键词：高危词直接命中并打 psych_risk=high
PSYCH_HIGH_KEYWORDS: list[str] = [
    "自杀", "自残", "自伤", "轻生", "不想活", "不想活了", "活不下去",
    "想死", "去死", "结束生命", "了结自己", "割腕", "安眠药自杀",
    "跳楼", "没有意义活着", "活着没意思", "消失了算了",
]

# 中危情感词：仅在进入 LLM 精判时提示（不在第一层拦截）
PSYCH_LOW_KEYWORDS: list[str] = [
    "压力大", "崩溃", "绝望", "压抑", "失眠", "焦虑", "自我怀疑",
]


def _matched(text: str, keywords: list[str]) -> list[str]:
    return [kw for kw in keywords if kw in text]


class RuleEngine:
    """无状态关键词拦截器。"""

    def hint_slots(self, query: str) -> dict[str, str]:
        """L1 词典提示槽位（{"subject": .., "grade": ..}，可能为 None）。"""
        return _rule_hint_slots(query)

    def check(self, query: str) -> IntentResult | None:
        t0 = time.perf_counter()
        hints = self.hint_slots(query)  # 拦截结果也附带提示槽位，供下游参考

        hit_cheat = _matched(query, CHEAT_KEYWORDS)
        if hit_cheat:
            return IntentResult(
                query=query,
                primary_intent=PrimaryIntent.REFUSE_CHEAT,
                secondary_intent=SecondaryIntent.UNCLEAR,
                confidence=1.0,
                handled_by="RULE",
                slots=Slots(subject=hints["subject"], grade=hints["grade"]),
                risk=RiskFlag(cheat_risk=True, matched_keywords=hit_cheat),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                decision_trace=[f"规则层命中作弊关键词: {hit_cheat}"],
                reply_hint="礼貌拒绝作弊请求，引导到正当备考方式",
            )

        hit_psych = _matched(query, PSYCH_HIGH_KEYWORDS)
        if hit_psych:
            return IntentResult(
                query=query,
                primary_intent=PrimaryIntent.CHAT_EMOTION,
                secondary_intent=SecondaryIntent.EMOTION_CRISIS,
                confidence=1.0,
                handled_by="RULE",
                slots=Slots(subject=hints["subject"], grade=hints["grade"]),
                risk=RiskFlag(psych_risk="high", matched_keywords=hit_psych),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                decision_trace=[f"规则层命中心理高危关键词: {hit_psych}"],
                reply_hint="高危！先暖心安抚，提示心理援助热线(12356)，必要时人工介入",
            )

        return None  # 放行进入第二层


rule_engine = RuleEngine()
