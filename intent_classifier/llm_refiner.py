"""第三层：LLM 精判兜底 + 完整槽位抽取。

配置了 INTENT_LLM_API_KEY 时调用 OpenAI 兼容接口（默认智谱 glm-4-flash）；
未配置或调用失败时，自动降级为启发式精判（关键词槽位抽取），保证流水线永不中断。
"""
from __future__ import annotations

import json
import re
import time

from .config import (
    ALLOWED_SECONDARY,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT,
    PrimaryIntent,
    SecondaryIntent,
)
from .rule_engine import CHEAT_KEYWORDS, PSYCH_HIGH_KEYWORDS, _matched
from .schemas import IntentResult, RiskFlag, Slots

# ---------------------------------------------------------------------------
# 启发式槽位抽取词典（fallback 与 LLM 结果校验共用）
# ---------------------------------------------------------------------------
SUBJECT_PATTERNS = [
    "数学", "语文", "英语", "物理", "化学", "生物", "历史", "地理", "政治",
    "文言文", "作文", "阅读理解", "完形填空", "听力", "口语", "函数", "几何",
    "力学", "电学", "有机化学", "遗传", "概率", "三角函数", "数列",
]
GRADE_PATTERNS = [
    "高三", "高二", "高一", "初三", "初二", "初一", "六年级", "小学",
    "大一", "大二", "大三", "大四", "复读", "考研", "初升高", "小升初",
]
TOPIC_PATTERNS = [
    "高考", "中考", "月考", "期末", "期中", "模拟考", "一模", "二模",
    "考研", "会考", "单招", "强基计划", "艺考", "体考", "自主招生",
]
EMOTION_LEXICON = [
    "焦虑", "紧张", "压力", "崩溃", "烦躁", "低落", "难过", "伤心", "孤独",
    "委屈", "挫败", "绝望", "疲惫", "累", "想哭", "丢脸", "自责", "emo",
]
TIME_PATTERNS = [
    r"\d+\s*天", "寒假", "暑假", "周末", "一个月", "三个月", "半年",
    "一轮复习", "二轮复习", "冲刺阶段", "晚自习", "每天",
]

# 各意图的启发式证据词（兜底精判时校验小模型预测是否可信）
POLICY_HINTS = ["报名", "录取", "分数线", "复读", "志愿", "政策", "招生",
                "单招", "强基", "调剂", "专升本", "毕业", "学位", "考试时间"]
PLAN_HINTS = ["计划", "规划", "安排", "怎么学", "提分", "提高", "逆袭",
              "复习", "冲刺", "作息", "时间表", "利用"]
ERROR_HINTS = ["错题", "丢分", "马虎", "分析", "考了", "没考好", "下滑",
               "审题", "粗心", "压轴题", "步骤分", "错在哪", "诊断"]
SUBJECT_HINTS = SUBJECT_PATTERNS + ["知识点", "公式", "定义", "怎么解",
                                    "怎么做", "讲解", "考点"]

_SYSTEM_PROMPT = """你是一个教育助手的意图识别引擎。对用户输入做意图分类与槽位抽取。

一级意图(只能选一个):
- QUESTION_SUBJECT: 学科知识提问(概念/题目求解)
- QUESTION_POLICY: 升学/考试政策提问(报名、录取、分数线等)
- REQUEST_STUDY_PLAN: 请求制定学习计划/提分规划
- REQUEST_ERROR_ANALYSIS: 请求分析错题/丢分/试卷问题
- CHAT_EMOTION: 情感倾诉、心理压力表达
- REFUSE_CHEAT: 寻求作弊、代写代考、买答案等违规帮助
- GENERAL_CHAT: 与学习无关的日常闲聊/信息查询
- UNKNOWN: 无实质内容、乱码、无法理解

二级意图: CONCEPT_EXPLAIN, SOLVE_PROBLEM, EXAM_POLICY, ADMISSION_POLICY,
SCHEDULE_PLANNING, GRADE_IMPROVE, MISTAKE_DIAGNOSIS, PAPER_ANALYSIS,
EMOTION_VENT, EMOTION_CRISIS, SMALL_TALK, INFO_SEEK, UNCLEAR

只输出 JSON,不要任何解释,格式:
{"primary_intent":"...","secondary_intent":"...","confidence":0.0,
 "slots":{"subject":null,"grade":null,"knowledge_points":[],"topic":null,"emotion":null,"time_horizon":null},
 "risk":{"cheat":false,"psych":"none"}}"""


def _heuristic_slots(query: str) -> Slots:
    subject = next((s for s in SUBJECT_PATTERNS if s in query), None)
    grade = next((g for g in GRADE_PATTERNS if g in query), None)
    topic = next((t for t in TOPIC_PATTERNS if t in query), None)
    emotion = next((e for e in EMOTION_LEXICON if e in query), None)
    time_horizon = None
    for pat in TIME_PATTERNS:
        m = re.search(pat, query)
        if m:
            time_horizon = m.group(0)
            break
    return Slots(
        subject=subject, grade=grade, topic=topic,
        emotion=emotion, time_horizon=time_horizon,
    )


def _validate(raw: dict, query: str, layer2: dict) -> dict:
    """LLM 输出白名单校验：非法枚举一律回退，绝不透传脏数据。"""
    try:
        primary = PrimaryIntent(raw.get("primary_intent", "").strip())
    except ValueError:
        primary = layer2.get("intent") or PrimaryIntent.UNKNOWN

    secondary = None
    try:
        cand = SecondaryIntent(raw.get("secondary_intent", "").strip())
        if cand in ALLOWED_SECONDARY[primary]:
            secondary = cand
    except ValueError:
        pass

    try:
        conf = float(raw.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = 0.5

    s = raw.get("slots") or {}
    slots = Slots(
        subject=s.get("subject"),
        grade=s.get("grade"),
        knowledge_points=[str(k) for k in (s.get("knowledge_points") or [])][:8],
        topic=s.get("topic"),
        emotion=s.get("emotion"),
        time_horizon=s.get("time_horizon"),
    )

    r = raw.get("risk") or {}
    cheat = bool(r.get("cheat", False)) or bool(_matched(query, CHEAT_KEYWORDS))
    psych = r.get("psych", "none")
    if psych not in ("none", "low", "high"):
        psych = "none"
    if _matched(query, PSYCH_HIGH_KEYWORDS):
        psych = "high"
    return {
        "primary": primary, "secondary": secondary, "conf": conf,
        "slots": slots, "risk": RiskFlag(cheat_risk=cheat, psych_risk=psych),
    }


class LLMRefiner:
    def __init__(self, api_key: str | None = None):
        self.api_key = (api_key or LLM_API_KEY).strip()
        self.available = bool(self.api_key)

    # ------------------------------------------------------------------
    def refine(self, query: str, layer2: dict) -> tuple[IntentResult, int]:
        """layer2: {"intent": PrimaryIntent, "confidence": float}"""
        t0 = time.perf_counter()
        if self.available:
            try:
                parsed = self._call_llm(query, layer2)
                result = self._build_result(query, layer2, parsed, "LLM_REFINE")
                return result, int((time.perf_counter() - t0) * 1000)
            except Exception as e:  # 网络/解析异常 → 启发式兜底
                print(f"[LLMRefiner] LLM 调用失败({e})，降级启发式精判")
        parsed = self._heuristic_refine(query, layer2)
        result = self._build_result(query, layer2, parsed, "LLM_FALLBACK")
        return result, int((time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------
    def _call_llm(self, query: str, layer2: dict) -> dict:
        import requests  # 局部导入，无网络场景不依赖

        user_msg = (
            f"用户输入: {query}\n"
            f"小模型预测: {layer2.get('intent')} (置信度{layer2.get('confidence')})\n"
            f"请给出最终判断。"
        )
        resp = requests.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.1,
                "max_tokens": 300,
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # 容忍 markdown 代码块包裹
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise ValueError(f"LLM 未返回 JSON: {content[:120]}")
        return json.loads(m.group(0))

    def _heuristic_refine(self, query: str, layer2: dict) -> dict:
        """无 API Key 时的离线精判：校验小模型预测 + 关键词槽位抽取。

        低置信(才进第三层)且无启发式证据支撑时，宁可判 UNKNOWN
        也不沿用可能错误的预测——域外问题的安全出口。
        """
        primary: PrimaryIntent = layer2.get("intent") or PrimaryIntent.UNKNOWN
        conf: float = layer2.get("confidence", 0.5)
        slots = _heuristic_slots(query)

        support = {
            PrimaryIntent.QUESTION_SUBJECT: any(h in query for h in SUBJECT_HINTS),
            PrimaryIntent.QUESTION_POLICY: any(h in query for h in POLICY_HINTS),
            PrimaryIntent.REQUEST_STUDY_PLAN: any(h in query for h in PLAN_HINTS),
            PrimaryIntent.REQUEST_ERROR_ANALYSIS: any(h in query for h in ERROR_HINTS),
            PrimaryIntent.CHAT_EMOTION: bool(slots.emotion),
            PrimaryIntent.REFUSE_CHEAT: bool(_matched(query, CHEAT_KEYWORDS)),
        }.get(primary, primary in (PrimaryIntent.GENERAL_CHAT, PrimaryIntent.UNKNOWN))

        if not support and conf < 0.6:
            primary = PrimaryIntent.UNKNOWN
            conf = min(conf, 0.5)

        secondary = next(iter(ALLOWED_SECONDARY[primary]))
        if primary is PrimaryIntent.CHAT_EMOTION and slots.emotion:
            secondary = SecondaryIntent.EMOTION_VENT
        return {
            "primary": primary, "secondary": secondary,
            "conf": conf, "slots": slots,
            "risk": RiskFlag(
                cheat_risk=bool(_matched(query, CHEAT_KEYWORDS)),
                psych_risk="low" if slots.emotion else "none",
            ),
        }

    def _build_result(self, query, layer2, parsed, handled_by) -> IntentResult:
        hint = None
        if parsed["risk"].psych_risk == "high":
            hint = "高危！先暖心安抚，提示心理援助热线(12356)，必要时人工介入"
        elif parsed["primary"] is PrimaryIntent.REFUSE_CHEAT:
            hint = "礼貌拒绝作弊请求，引导到正当备考方式"
        return IntentResult(
            query=query,
            primary_intent=parsed["primary"],
            secondary_intent=parsed["secondary"],
            confidence=parsed["conf"],
            handled_by=handled_by,
            slots=parsed["slots"],
            risk=parsed["risk"],
            decision_trace=[
                f"小模型: {layer2.get('intent')} conf={layer2.get('confidence')} < 0.85",
                f"{'LLM精判' if handled_by == 'LLM_REFINE' else '启发式精判'}: "
                f"{parsed['primary'].value}",
            ],
            reply_hint=hint,
        )


llm_refiner = LLMRefiner()
