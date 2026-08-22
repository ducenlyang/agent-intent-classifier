"""第三层：LLM 精判层——复核意图 + 抽取全部槽位 + 必填校验。

线性流水线：所有未被 L1 拦截的请求都会经过本层（L2 结果只作意图候选参考）。
任务：①复核 L2 意图候选 ②抽取全部槽位 subject/grade/question_text/
     knowledge_points/topic/emotion/time_horizon ③必填槽位校验 missing_slots
未配置 INTENT_LLM_API_KEY 或调用失败时，自动降级为启发式精判（词典槽位），不中断。
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
    REQUIRED_SLOTS,
    PrimaryIntent,
    SecondaryIntent,
)
from .rule_engine import CHEAT_KEYWORDS, PSYCH_HIGH_KEYWORDS, _matched
from .schemas import IntentResult, RiskFlag, Slots
from .slot_lexicon import (
    EMOTION_LEXICON,
    ERROR_HINTS,
    PLAN_HINTS,
    POLICY_HINTS,
    SUBJECT_HINTS,
    extract_lexicon_slots,
)

_SYSTEM_PROMPT = """你是一个教育助手的意图识别引擎。上游小模型给出意图候选，你负责终审并抽取全部槽位。

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

槽位说明:
- subject: 学科(数学/语文/英语/物理/化学/生物/历史/地理/政治等)
- grade: 年级(高一~高三/初一~初三/小学/大学年级等)
- question_text: 题目/问题原文(仅学科提问时有)
- knowledge_points: 涉及知识点列表
- topic: 考试/场景主题(高考/中考/月考/期末等)
- emotion: 情绪词(焦虑/压力/低落等)
- time_horizon: 时间范围(90天/寒假/一个月等)

必填槽位规则: QUESTION_SUBJECT必填subject；REQUEST_STUDY_PLAN必填subject+grade；
REQUEST_ERROR_ANALYSIS必填subject。缺失的字段名填入 missing_slots。

只输出 JSON,不要任何解释,格式:
{"primary_intent":"...","secondary_intent":"...","confidence":0.0,
 "slots":{"subject":null,"grade":null,"question_text":null,"knowledge_points":[],
          "topic":null,"emotion":null,"time_horizon":null},
 "missing_slots":[],
 "risk":{"cheat":false,"psych":"none"}}"""


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
    qt = s.get("question_text")
    slots = Slots(
        subject=s.get("subject") or None,
        grade=s.get("grade") or None,
        question_text=(str(qt)[:200] or None) if qt else None,
        knowledge_points=[str(k) for k in (s.get("knowledge_points") or [])][:8] or None,
        topic=s.get("topic") or None,
        emotion=s.get("emotion") or None,
        time_horizon=s.get("time_horizon") or None,
    )
    missing_raw = [str(m) for m in (raw.get("missing_slots") or [])][:8]
    # 只保留当前意图下确实为空的必填槽位名，防止 LLM 幻觉
    missing = [f for f in missing_raw if f in REQUIRED_SLOTS.get(primary, [])
               and not getattr(slots, f)]

    r = raw.get("risk") or {}
    cheat = bool(r.get("cheat", False)) or bool(_matched(query, CHEAT_KEYWORDS))
    psych = r.get("psych", "none")
    if psych not in ("none", "low", "high"):
        psych = "none"
    if _matched(query, PSYCH_HIGH_KEYWORDS):
        psych = "high"
    return {
        "primary": primary, "secondary": secondary, "conf": conf,
        "slots": slots, "missing": missing,
        "risk": RiskFlag(cheat_risk=cheat, psych_risk=psych),
    }


class LLMRefiner:
    def __init__(self, api_key: str | None = None):
        # None=取配置；显式传空串=强制禁用LLM（离线/评估省配额场景）
        self.api_key = (LLM_API_KEY if api_key is None else api_key).strip()
        self.available = bool(self.api_key)

    # ------------------------------------------------------------------
    def refine(self, query: str, layer2: dict) -> tuple[IntentResult, int]:
        """layer2: {"intent": PrimaryIntent, "confidence": float}"""
        t0 = time.perf_counter()
        if self.available:
            try:
                raw = self._call_llm(query, layer2)
                parsed = _validate(raw, query, layer2)  # 白名单校验后再构建结果
                result = self._build_result(query, layer2, parsed, "LLM_REFINE")
                return result, int((time.perf_counter() - t0) * 1000)
            except Exception as e:  # 网络/解析异常 → 启发式兜底
                print(f"[LLMRefiner] LLM 调用失败({e})，降级启发式精判")
        parsed = self._heuristic_refine(query, layer2)
        result = self._build_result(query, layer2, parsed, "LLM_FALLBACK")
        return result, int((time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------
    def _call_llm(self, query: str, layer2: dict) -> dict:
        from .llm_client import chat_completion

        user_msg = (
            f"用户输入: {query}\n"
            f"上游小模型意图候选: {getattr(layer2.get('intent'), 'value', layer2.get('intent'))}"
            f" (置信度{layer2.get('confidence')})\n"
            f"请终审意图并抽取全部槽位。"
        )
        content = chat_completion(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=400,
            timeout=LLM_TIMEOUT,
        )
        # 容忍 markdown 代码块包裹
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise ValueError(f"LLM 未返回 JSON: {content[:120]}")
        return json.loads(m.group(0))

    def _heuristic_refine(self, query: str, layer2: dict) -> dict:
        """无 API Key 时的离线精判：证据词校验 + 词典全量槽位抽取。"""
        primary: PrimaryIntent = layer2.get("intent") or PrimaryIntent.UNKNOWN
        conf: float = layer2.get("confidence", 0.5)

        support = {
            PrimaryIntent.QUESTION_SUBJECT: any(h in query for h in SUBJECT_HINTS),
            PrimaryIntent.QUESTION_POLICY: any(h in query for h in POLICY_HINTS),
            PrimaryIntent.REQUEST_STUDY_PLAN: any(h in query for h in PLAN_HINTS),
            PrimaryIntent.REQUEST_ERROR_ANALYSIS: any(h in query for h in ERROR_HINTS),
            PrimaryIntent.CHAT_EMOTION: any(e in query for e in EMOTION_LEXICON),
            PrimaryIntent.REFUSE_CHEAT: bool(_matched(query, CHEAT_KEYWORDS)),
        }.get(primary, primary in (PrimaryIntent.GENERAL_CHAT, PrimaryIntent.UNKNOWN))

        if not support and conf < 0.6:
            primary = PrimaryIntent.UNKNOWN  # 域外安全出口
            conf = min(conf, 0.5)

        lex = extract_lexicon_slots(query)
        question_text = query if primary is PrimaryIntent.QUESTION_SUBJECT else None
        slots = Slots(
            subject=lex["subject"], grade=lex["grade"],
            question_text=question_text, topic=lex["topic"],
            emotion=lex["emotion"], time_horizon=lex["time_horizon"],
        )
        missing = [f for f in REQUIRED_SLOTS.get(primary, []) if not getattr(slots, f)]

        secondary = next(iter(ALLOWED_SECONDARY[primary]))
        if primary is PrimaryIntent.CHAT_EMOTION and slots.emotion:
            secondary = SecondaryIntent.EMOTION_VENT
        return {
            "primary": primary, "secondary": secondary,
            "conf": conf, "slots": slots, "missing": missing,
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
        missing = parsed["missing"] or [
            f for f in REQUIRED_SLOTS.get(parsed["primary"], [])
            if not getattr(parsed["slots"], f)
        ]
        return IntentResult(
            query=query,
            primary_intent=parsed["primary"],
            secondary_intent=parsed["secondary"],
            confidence=parsed["conf"],
            handled_by=handled_by,
            slots=parsed["slots"],
            missing_slots=missing,
            risk=parsed["risk"],
            decision_trace=[
                f"小模型候选: {layer2.get('intent')} conf={layer2.get('confidence')}",
                f"{'LLM终审' if handled_by == 'LLM_REFINE' else '启发式终审'}: "
                f"{parsed['primary'].value}",
            ],
            reply_hint=hint,
        )


llm_refiner = LLMRefiner()
