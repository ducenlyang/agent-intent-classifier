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


# "元请求"特征：求题/求讲解但没带真实题目（如"帮我解一道高二数学题"）。
# 此类 question_text 置空 → missing_slots 触发下游反问"请把题目发给我"。
META_REQUEST_PATTERNS = [
    "帮我解", "帮我做", "出一道", "考我一", "来一道",
    "一道题", "一个题", "几道题", "道数学题", "道题",
    "出一个", "出几道", "给我出", "帮我出", "出题", "考我",
]


# 题干实体特征：句中出现即认定带了真实题目内容，不按元请求处理
# (防"帮我讲讲这道题：已知x²-4=0"被"道题"模式误判为只要题)
_PROBLEM_FEATURES = ("已知", "如图", "求证", "求解", "化简", "证明",
                     "计算", "解方程", "方程", "=", "²", "√", "≥", "≤")

# 出题请求组合语法(动词×名词)，替代纯子串枚举的打地鼠：
#   "来一个化学题" / "来个题" / "练几道" / "想学化学出点题" 一次接住
_PROBLEM_REQ_VERBS = ("来", "出", "给", "发", "考", "练", "想学", "想练")
_PROBLEM_REQ_NOUNS = ("题", "卷", "几道", "单词", "练习")


def is_problem_request(query: str) -> bool:
    """True=用户要求助手【出一道题】(而非自己带题求解答)。
    出题动词组 × 题类名词组合命中，且无题干实体特征。"""
    if any(f in query for f in _PROBLEM_FEATURES):
        return False
    return (any(v in query for v in _PROBLEM_REQ_VERBS)
            and any(n in query for n in _PROBLEM_REQ_NOUNS))


def is_meta_request(query: str) -> bool:
    """True=用户在"要题/求讲解"但没带真实题目（如"帮我解一道高二数学题"）。
    含出题请求；两者下游处理不同：解题类反问收题，出题类直接出题。"""
    if any(f in query for f in _PROBLEM_FEATURES):
        return False  # 带题干特征：题目就在句子里，不是元请求
    return is_problem_request(query) or any(
        p in query for p in META_REQUEST_PATTERNS)


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
