"""输出守卫校验层：Agent 生成结果返回用户前的安全检查。

检查项：空/异常输出、作弊协助话术、心理高危场景的话术底线。
未通过 → 替换为安全兜底话术；软性问题 → 追加补丁而非拦截。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import IntentResult

# Agent 输出中不允许出现的内容（协助作弊类话术）
CHEAT_FACILITATION = [
    "帮你代写", "代写好了", "直接抄", "抄我的", "作弊方法", "作弊技巧",
    "作弊不被发现", "暗号是", "答案发你", "进考场带小抄", "枪手联系方式",
]

# 心理高危场景的兜底要素
HOTLINE = "12356"
CARE_WORDS = ("热线", "深呼吸", "陪着你", "关心你", "勇敢", "不是你的错", "倾诉")

SAFE_FALLBACK = (
    "这个问题我需要更谨慎地回答。换个角度：如果你想提升这门课的成绩，"
    "我可以帮你分析薄弱点、制定复习计划，靠实力拿分才是最稳的路～"
)


@dataclass
class GuardVerdict:
    passed: bool = True
    actions: list[str] = field(default_factory=list)  # 采取的处置说明


def output_guard(text: str, result: IntentResult) -> tuple[str, GuardVerdict]:
    """返回 (安全文本, 守卫结论)。"""
    verdict = GuardVerdict()

    # 1) 空/异常输出
    if not text or not text.strip() or len(text.strip()) < 5:
        verdict.passed = False
        verdict.actions.append("输出为空，替换为兜底话术")
        return "抱歉，我刚才走神了，能再说一遍你的问题吗？", verdict

    # 2) 作弊协助话术硬拦截
    hits = [kw for kw in CHEAT_FACILITATION if kw in text]
    if hits:
        verdict.passed = False
        verdict.actions.append(f"检出违规话术{hits}，已替换安全回复")
        return SAFE_FALLBACK, verdict

    # 3) 心理高危场景软校验：输出应包含热线或安抚要素
    if result.risk.psych_risk == "high":
        if HOTLINE not in text and not any(w in text for w in CARE_WORDS):
            text += (
                "\n\n（如果心里的难受一直压着你，记得可以拨打心理援助热线 12356，"
                "24小时都有人愿意听你说。）"
            )
            verdict.actions.append("心理高危输出缺少安抚要素，已追加热线提示")

    return text, verdict
