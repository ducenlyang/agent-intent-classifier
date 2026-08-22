"""路由分发 Router：按 IntentResult 决定本轮走向。

├─ 风险拦截分支(REFUSE_CHEAT等) → 直接返回话术，结束
├─ missing_slots 非空 → 反问用户补齐信息，结束本轮
└─ 槽位齐全 → 分发到对应业务 Agent
"""
from __future__ import annotations

from dataclasses import dataclass

from .agents import Agent, get_agent
from .config import PrimaryIntent
from .schemas import IntentResult

# 缺槽反问话术（每个必填槽位对应一句自然追问）
CLARIFY_QUESTIONS: dict[str, str] = {
    "subject": "想让我帮你重点抓哪一科呢？（数学/语文/英语/物理/化学…）",
    "grade": "方便说下现在读几年级吗？（比如初三、高一、高三）",
    "question_text": "请把题目发给我（题目文字直接打出来就行）～",
    "time_horizon": "你想规划多长时间？（比如寒假、一个月、90天）",
}


@dataclass
class RouteDecision:
    kind: str          # intercept / clarify / agent
    reply: str = ""    # intercept/clarify 时直接可发的话术
    agent: Agent | None = None
    missing: list[str] | None = None


def route(result: IntentResult) -> RouteDecision:
    # 风险拦截分支：L1 已带可直接下发的话术，直接结束链路
    if result.primary_intent == PrimaryIntent.REFUSE_CHEAT or result.risk.cheat_risk:
        return RouteDecision(kind="intercept", reply=result.reply or "这个忙帮不了哦～")

    # 缺槽反问分支：结束本轮，等用户补充（由 Assistant 合并重识别）
    if result.missing_slots:
        questions = " ".join(
            CLARIFY_QUESTIONS.get(f, f"请补充{f}") for f in result.missing_slots
        )
        return RouteDecision(
            kind="clarify",
            reply=f"好嘞，先确认两个信息：{questions}",
            missing=result.missing_slots,
        )

    # 槽位齐全 → 分发业务 Agent（含 UNKNOWN 兜底 Agent）
    return RouteDecision(kind="agent", agent=get_agent(result.primary_intent))
