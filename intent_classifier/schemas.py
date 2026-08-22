"""数据结构定义（pydantic v2）：槽位、风险标记、意图结果。"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .config import PrimaryIntent, SecondaryIntent


class Slots(BaseModel):
    """槽位：短槽位(subject/grade)可由 L2 填充；长槽位由第三层填充。"""

    subject: Optional[str] = Field(None, description="学科，如 数学/物理/英语")
    grade: Optional[str] = Field(None, description="年级，如 高三/初二")
    question_text: Optional[str] = Field(None, description="长开放槽位：题目/问题原文")
    knowledge_points: Optional[list[str]] = Field(None, description="知识点列表(仅LLM)")
    topic: Optional[str] = Field(None, description="主题，如 高考/中考/寒假")
    emotion: Optional[str] = Field(None, description="情绪极性，如 焦虑/低落/压力")
    time_horizon: Optional[str] = Field(None, description="时间范围，如 90天/寒假")


class RiskFlag(BaseModel):
    cheat_risk: bool = False
    psych_risk: Literal["none", "low", "high"] = "none"
    matched_keywords: list[str] = Field(default_factory=list)


class IntentResult(BaseModel):
    """三层流水线最终输出。"""

    query: str
    primary_intent: PrimaryIntent
    secondary_intent: Optional[SecondaryIntent] = None
    confidence: float = 1.0
    handled_by: Literal["RULE", "SMALL_MODEL", "LLM_REFINE", "LLM_FALLBACK"]
    slots: Slots = Field(default_factory=Slots)
    slot_confidence: dict[str, float] = Field(
        default_factory=dict, description="BIO头输出的短槽位置信度 {slot: conf}"
    )
    missing_slots: list[str] = Field(
        default_factory=list, description="必填槽位缺失项，供下游Agent追问"
    )
    risk: RiskFlag = Field(default_factory=RiskFlag)
    latency_ms: int = 0
    decision_trace: list[str] = Field(default_factory=list)  # 各层决策路径，便于排查
    reply_hint: Optional[str] = Field(
        None, description="给下游回复器的提示，如心理高危需暖心话术+人工介入"
    )
