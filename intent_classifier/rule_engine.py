"""第一层：规则引擎（风险拦截，命中直接返回拒绝话术结束链路）。

作弊关键词 → REFUSE_CHEAT + 拒绝话术；心理高危关键词 → CHAT_EMOTION + 安抚话术。
未命中 → 放行进入第二层。
"""
from __future__ import annotations

import time

from .config import PrimaryIntent, SecondaryIntent
from .schemas import IntentResult, RiskFlag

# 作弊类关键词：直接判 REFUSE_CHEAT
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

# L1 直接返回的话术（可直接下发给用户）
CHEAT_REPLY = (
    "这个忙帮不了哦～作弊一旦被发现，轻则成绩作废，重则记入诚信档案，"
    "影响升学太不划算了。与其冒这个险，不如告诉我你在备考哪一科、"
    "哪块最没底，我帮你安排突击计划，稳稳提分更踏实。"
)
PSYCH_REPLY = (
    "听到你这么说，我很担心你，也很谢谢你愿意讲出来。你能说出来就已经很勇敢了。"
    "现在先慢慢做三次深呼吸，让自己缓一缓。如果这种难受一直压着你，"
    "请一定拨打心理援助热线 12356（24小时），或把心里的话告诉信任的家人、老师。"
    "我也可以一直在这里陪你聊聊，你想说什么都可以。"
)


def _matched(text: str, keywords: list[str]) -> list[str]:
    return [kw for kw in keywords if kw in text]


class RuleEngine:
    """无状态关键词拦截器。"""

    def check(self, query: str) -> IntentResult | None:
        t0 = time.perf_counter()

        hit_cheat = _matched(query, CHEAT_KEYWORDS)
        if hit_cheat:
            return IntentResult(
                query=query,
                primary_intent=PrimaryIntent.REFUSE_CHEAT,
                secondary_intent=SecondaryIntent.UNCLEAR,
                confidence=1.0,
                handled_by="RULE",
                risk=RiskFlag(cheat_risk=True, matched_keywords=hit_cheat),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                decision_trace=[f"规则层命中作弊关键词: {hit_cheat}，直接返回拒绝话术"],
                reply=CHEAT_REPLY,
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
                risk=RiskFlag(psych_risk="high", matched_keywords=hit_psych),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                decision_trace=[f"规则层命中心理高危关键词: {hit_psych}，直接返回安抚话术"],
                reply=PSYCH_REPLY,
                reply_hint="高危！已下发安抚话术+热线12356，建议记录并视情况人工介入",
            )

        return None  # 放行进入第二层


rule_engine = RuleEngine()
