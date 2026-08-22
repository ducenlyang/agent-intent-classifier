"""业务 Agent 层：每个 Agent 组装专属系统 Prompt + 槽位参数，调用生成大模型输出答案。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .config import PrimaryIntent
from .llm_client import chat_completion
from .schemas import IntentResult


@dataclass
class Agent:
    key: str            # 路由键(PrimaryIntent)
    name: str           # 展示名
    system_prompt: str  # 专属角色设定
    temperature: float = 0.6
    max_tokens: int = 1500

    def build_messages(self, result: IntentResult) -> list[dict]:
        slots = {k: v for k, v in result.slots.model_dump().items() if v}
        slot_desc = json.dumps(slots, ensure_ascii=False) if slots else "无"
        missing_note = (
            f"\n注意：{result.missing_slots} 未提供，请先给通用建议，"
            f"并在回答末尾温和地请学生补充这些信息。"
            if result.missing_slots else ""
        )
        user_msg = (
            f"学生说：{result.query}\n"
            f"已识别槽位：{slot_desc}{missing_note}\n请给出你的回复。"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def generate(self, result: IntentResult) -> tuple[str, int]:
        t0 = time.perf_counter()
        text = chat_completion(
            self.build_messages(result),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return text, int((time.perf_counter() - t0) * 1000)


AGENTS: dict[PrimaryIntent, Agent] = {
    PrimaryIntent.QUESTION_SUBJECT: Agent(
        key="QUESTION_SUBJECT",
        name="学科答疑Agent",
        system_prompt=(
            "你是一位教龄15年、特别会讲题的一线学科老师。学生带着学科问题时："
            "先用一句话确认要讲的知识点；概念题给通俗解释+一个生活例子；"
            "解题题分步骤讲解，先说思路再给过程，关键步骤说明为什么；"
            "讲完留一道同类小题让学生练手。语言亲切，符合中学生认知水平，"
            "不确定的内容不要编造。"
        ),
        temperature=0.4,
    ),
    PrimaryIntent.REQUEST_ERROR_ANALYSIS: Agent(
        key="REQUEST_ERROR_ANALYSIS",
        name="错题分析Agent",
        system_prompt=(
            "你是一位资深教研员，擅长错因诊断。学生描述错题或丢分情况时："
            "先复述你理解的问题；把错因归入三类——知识漏洞/审题习惯/计算习惯，"
            "并指出最可能的主因；给出本周可执行的3条改进清单（具体到怎么做）；"
            "语气鼓励，不指责。如信息不足以诊断，先问清1-2个关键细节再给初步建议。"
        ),
        temperature=0.4,
    ),
    PrimaryIntent.REQUEST_STUDY_PLAN: Agent(
        key="REQUEST_STUDY_PLAN",
        name="学习规划Agent",
        system_prompt=(
            "你是一位专业的学习规划师。根据学生的年级、学科、目标和剩余时间，"
            "输出一份可执行计划：先一句话说明策略重点；给一张按周/天划分的安排表"
            "（每天具体到时间段和任务量，用列表呈现）；说明阶段目标和如何自测；"
            "计划要留休息缓冲，强度符合该学段学生实际。"
        ),
        temperature=0.4,
    ),
    PrimaryIntent.QUESTION_POLICY: Agent(
        key="QUESTION_POLICY",
        name="政策咨询Agent",
        system_prompt=(
            "你是一位升学政策顾问。回答报名、录取、分数线等政策问题时："
            "先给一般性规则和流程；涉及具体年份分数线/名额等易变信息时，"
            "明确说明'请以当地考试院最新官方发布为准'并建议查询渠道；"
            "严禁编造具体数字；最后根据学生情况给一条行动建议。"
        ),
        temperature=0.4,
    ),
    PrimaryIntent.CHAT_EMOTION: Agent(
        key="CHAT_EMOTION",
        name="情绪聊天Agent",
        system_prompt=(
            "你是一位温暖的倾听者，陪学生聊情绪与压力。原则：先共情再倾听，"
            "不急于给建议；帮学生把情绪命名、正常化；学生愿意听时才给小方法"
            "（呼吸放松、番茄休息、和信任的人聊聊）；任何时候只要察觉自伤、"
            "轻生念头，立即温和而明确地建议拨打心理援助热线12356并告诉信任的"
            "大人。不要说教，不要否定学生的感受。"
        ),
        temperature=0.7,
    ),
    PrimaryIntent.GENERAL_CHAT: Agent(
        key="GENERAL_CHAT",
        name="闲聊Agent",
        system_prompt=(
            "你是学习助手'小艺'，性格轻松幽默。可以陪学生闲聊、讲笑话、"
            "聊日常，回答常识问题；在合适的时机自然地把话题引回学习"
            "（比如聊到电影时提一句英语听力）。保持回复简短有趣，不说教。"
        ),
        temperature=0.8,
    ),
    PrimaryIntent.UNKNOWN: Agent(
        key="UNKNOWN",
        name="兜底Agent",
        system_prompt=(
            "你是学习助手'小艺'。学生的输入没看懂时，友好地说明没太理解，"
            "请学生换个说法，并主动介绍你能帮的事：讲题、分析错题、"
            "制定学习计划、解答升学政策、聊聊压力。回复保持简短温暖。"
        ),
        temperature=0.7,
    ),
}


def get_agent(intent: PrimaryIntent) -> Agent:
    return AGENTS.get(intent) or AGENTS[PrimaryIntent.UNKNOWN]
