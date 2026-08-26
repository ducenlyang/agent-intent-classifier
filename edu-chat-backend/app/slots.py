"""后端槽位合并策略（对话后端独有职责，网关无此概念）。

合并规则（方案约定）：本轮抽取值不为 None 优先覆盖；为 None 继承会话缓存。
二次判断：网关的单轮 missing_slots 中，合并后仍为空的才算真缺失 → 反问。
"""
from __future__ import annotations

import re

from intent_classifier.schemas import IntentResult, Slots
from intent_classifier.slot_lexicon import GRADE_LEXICON, SUBJECT_LEXICON

# 纯值词表：回复恰好是学科/年级词本身时，视作对 AI 提问的回答而非新问题
SUBJECT_VALUES = set(SUBJECT_LEXICON)
GRADE_VALUES = set(GRADE_LEXICON)

# 新题信号启发式（无OCR上游，用题干特征替代）：题干套话/数学符号/算式/超长粘贴
_NEW_Q_PATTERNS = (
    "已知", "如图", "求证", "求解", "解方程", "解下列", "下列", "计算下", "求下",
    "化简", "证明", "选择题", "填空题", "应用题",
    "①", "②", "③", "√", "≥", "≤", "≠", "²", "³", "π", "cm²", "cm2",
)
_NEW_Q_EXPR = re.compile(r"[0-9a-zA-Zx)）²³]\s*[+\-*/×÷=]\s*[0-9a-zA-Zx(（²³]")
# 追问语气结尾：短句以这些结尾时是对答案/求解释的追问，不是贴新题("x=2对吗")
_FOLLOWUP_TAILS = ("对吗", "对不对", "对吧", "是吗", "错了吗", "为什么",
                   "怎么办", "怎么想", "行吗", "可以吗")


def has_new_question(q: str) -> bool:
    """本轮输入是否携带一道新题目(题干特征/算式/长文本粘贴)。"""
    if not q:
        return False
    if len(q) >= 40:
        return True
    stripped = q.rstrip("？?！!。.，, ")
    if len(stripped) <= 20 and any(stripped.endswith(t) for t in _FOLLOWUP_TAILS):
        return False
    return any(p in q for p in _NEW_Q_PATTERNS) or bool(_NEW_Q_EXPR.search(q))


# 学科内容推断：题目原文未点名学科时，从题面关键词推断(subject必填槽位兜底，
# 避免用户贴纯题干却被反复反问"哪一科")
SUBJECT_INFER_RULES: dict[str, tuple] = {
    "数学": ("方程", "函数", "开方", "平方", "几何", "代数", "概率", "数列",
             "向量", "三角形", "勾股", "导数", "不等式", "坐标", "求x", "求解"),
    "物理": ("受力", "速度", "加速度", "电路", "电流", "电压", "电阻", "浮力",
             "杠杆", "功率", "牛顿", "电磁", "摩擦", "惯性"),
    "化学": ("化学式", "元素", "分子", "原子", "反应", "氧化", "还原", "周期表",
             "摩尔", "化学方程"),
    "英语": ("英语", "语法", "时态", "从句", "单词", "翻译", "完形"),
    "语文": ("文言文", "古诗", "阅读理解", "作文", "修辞", "拼音"),
    "生物": ("细胞", "光合", "遗传", "呼吸作用", "生态系统", "染色体"),
}


def infer_subject(text: str) -> str | None:
    """从题目原文关键词推断学科。"""
    for subj, kws in SUBJECT_INFER_RULES.items():
        if any(k in text for k in kws):
            return subj
    return None

# 槽位生命周期分级（对话状态跟踪惯例）：
#   用户稳定属性：跨话题长期有效（换了话题学生还是高二）
#   任务作用域槽：只在当前任务话题内有效，话题切换即失效，防跨话题错配/污染传染
# question_text 每题不同，不缓存
USER_STABLE_FIELDS = ("grade",)
TASK_SCOPED_FIELDS = ("subject", "topic", "time_horizon")
CACHEABLE_FIELDS = USER_STABLE_FIELDS + TASK_SCOPED_FIELDS

# 学科强相关的任务意图组：组内切换（讲题→错题→计划）视作同一话题链，保留学科槽；
# 跨出该组的任务意图切换则任务作用域槽全部失效
SUBJECT_TASK_INTENTS = {
    "QUESTION_SUBJECT", "REQUEST_ERROR_ANALYSIS", "REQUEST_STUDY_PLAN",
}


def drop_task_scoped(cached: Slots | None) -> Slots:
    """话题切换时丢弃任务作用域槽位，仅保留用户稳定属性。"""
    if not cached:
        return Slots()
    return Slots.model_validate(
        {f: getattr(cached, f) for f in USER_STABLE_FIELDS if getattr(cached, f)}
    )

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
