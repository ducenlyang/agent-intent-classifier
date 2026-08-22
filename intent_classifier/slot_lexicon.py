"""槽位词典（单一数据源）：规则层提示槽、BIO远程监督标注、兜底槽位抽取共用。

改词表只改这一个文件，L1提示 / L2训练标签 / L3兜底 三处自动一致。
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 实体词典（短槽位：subject / grade）——BIO 头的训练标签即由此生成
# ---------------------------------------------------------------------------
SUBJECT_LEXICON: list[str] = [
    "数学", "语文", "英语", "物理", "化学", "生物", "历史", "地理", "政治",
    "文言文", "作文", "阅读理解", "完形填空", "听力", "口语", "函数", "几何",
    "导数", "数列", "概率", "三角函数", "向量", "不等式", "力学", "电学",
    "电磁感应", "有机化学", "元素周期表", "遗传", "光合作用", "细胞分裂",
    "定语从句", "虚拟语气", "被动语态", "议论文",
]

GRADE_LEXICON: list[str] = [
    "高三", "高二", "高一", "初三", "初二", "初一", "六年级", "小学",
    "大一", "大二", "大三", "大四", "复读", "考研", "初升高", "小升初",
]

# ---------------------------------------------------------------------------
# 长槽位词典（仅第三层使用）
# ---------------------------------------------------------------------------
TOPIC_LEXICON: list[str] = [
    "高考", "中考", "月考", "期末", "期中", "模拟考", "一模", "二模",
    "考研", "会考", "单招", "强基计划", "艺考", "体考", "自主招生",
]

EMOTION_LEXICON: list[str] = [
    "焦虑", "紧张", "压力", "崩溃", "烦躁", "低落", "难过", "伤心", "孤独",
    "委屈", "挫败", "绝望", "疲惫", "累", "想哭", "丢脸", "自责", "emo",
    "心慌", "慌", "失眠", "睡不着",
]

TIME_PATTERNS: list[str] = [
    r"\d+\s*天", "寒假", "暑假", "周末", "一个月", "三个月", "半年",
    "一轮复习", "二轮复习", "冲刺阶段", "晚自习", "每天", "两个月",
]

# 各意图的启发式证据词（兜底精判校验小模型预测是否可信）
POLICY_HINTS = ["报名", "录取", "分数线", "复读", "志愿", "政策", "招生",
                "单招", "强基", "调剂", "专升本", "毕业", "学位", "考试时间"]
PLAN_HINTS = ["计划", "规划", "安排", "怎么学", "提分", "提高", "逆袭",
              "复习", "冲刺", "作息", "时间表", "利用"]
ERROR_HINTS = ["错题", "丢分", "马虎", "分析", "考了", "没考好", "下滑",
               "审题", "粗心", "压轴题", "步骤分", "错在哪", "诊断"]
SUBJECT_HINTS = SUBJECT_LEXICON + ["知识点", "公式", "定义", "怎么解",
                                    "怎么做", "讲解", "考点"]


# ---------------------------------------------------------------------------
# 最长匹配找实体字符区间（供规则提示槽 + BIO 远程监督共用）
# ---------------------------------------------------------------------------
def find_entity_spans(text: str, lexicon: list[str]) -> list[tuple[int, int]]:
    """词典最长匹配，返回互不重叠的 (start, end) 字符区间。"""
    spans: list[tuple[int, int]] = []
    used = [False] * len(text)
    for term in sorted(lexicon, key=len, reverse=True):
        start = 0
        while (i := text.find(term, start)) != -1:
            if not any(used[i:i + len(term)]):
                for k in range(i, i + len(term)):
                    used[k] = True
                spans.append((i, i + len(term)))
            start = i + 1
    return spans


def match_subject(text: str) -> str | None:
    spans = find_entity_spans(text, SUBJECT_LEXICON)
    return text[spans[0][0]:spans[0][1]] if spans else None


def match_grade(text: str) -> str | None:
    spans = find_entity_spans(text, GRADE_LEXICON)
    return text[spans[0][0]:spans[0][1]] if spans else None


def rule_hint_slots(query: str) -> dict[str, str]:
    """L1 词典/正则捞取的提示槽位（只管 subject/grade 两个短槽位）。"""
    return {"subject": match_subject(query), "grade": match_grade(query)}


def extract_lexicon_slots(query: str) -> dict:
    """L3 兜底：全量长槽位词典抽取。"""
    subject = match_subject(query)
    grade = match_grade(query)
    topic = next((t for t in TOPIC_LEXICON if t in query), None)
    emotion = next((e for e in EMOTION_LEXICON if e in query), None)
    time_horizon = None
    for pat in TIME_PATTERNS:
        m = re.search(pat, query)
        if m:
            time_horizon = m.group(0)
            break
    return {
        "subject": subject, "grade": grade, "topic": topic,
        "emotion": emotion, "time_horizon": time_horizon,
    }
