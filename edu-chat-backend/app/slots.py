"""后端槽位合并策略（对话后端独有职责，网关无此概念）。

合并规则（方案约定）：本轮抽取值不为 None 优先覆盖；为 None 继承会话缓存。
二次判断：网关的单轮 missing_slots 中，合并后仍为空的才算真缺失 → 反问。
"""
from __future__ import annotations

from intent_classifier.schemas import IntentResult, Slots

# 允许跨轮记忆的槽位（question_text 每题不同，不缓存）
CACHEABLE_FIELDS = ("subject", "grade", "topic", "time_horizon")

# 缺槽反问话术（后端职责：生成反问向用户索要信息）
CLARIFY_QUESTIONS: dict[str, str] = {
    "subject": "想让我帮你重点抓哪一科呢？（数学/语文/英语/物理/化学…）",
    "grade": "方便说下现在读几年级吗？（比如初三、高一、高二）",
    "question_text": "请把题目发给我（题目文字直接打出来就行）～",
    "time_horizon": "你想规划多长时间？（比如寒假、一个月、90天）",
}


def merge_slots(ir: IntentResult, cached: Slots | None) -> Slots:
    """本轮非None优先覆盖；None继承缓存。返回完整 Slots(含question_text)。"""
    data = (cached or Slots()).model_dump()
    for k, v in ir.slots.model_dump().items():
        if v is not None:
            data[k] = v
    return Slots.model_validate(data)


def still_missing(ir: IntentResult, merged: Slots) -> list[str]:
    """网关单轮缺失中，缓存合并后仍为空的（消除可由记忆补上的反问）。"""
    return [f for f in (ir.missing_slots or []) if not getattr(merged, f, None)]


def cacheable(merged: Slots) -> Slots:
    """从合并结果提取下一轮要记住的槽位。"""
    return Slots.model_validate(
        {f: getattr(merged, f) for f in CACHEABLE_FIELDS if getattr(merged, f)}
    )
