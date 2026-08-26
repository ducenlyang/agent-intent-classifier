"""六大业务 Agent（后端独立实现，无知识库，直接 LLM 生成）。

学科答疑为【脚手架模式】：引导式教学，不给完整答案；
其余 Agent 沿用各自专属 Prompt。
"""
from __future__ import annotations

import json

from .config import HISTORY_WINDOW
from .llm_client import chat_completion, chat_completion_stream

AGENT_REGISTRY: dict[str, dict] = {
    "QUESTION_SUBJECT": {
        "name": "学科答疑Agent",
        "system_prompt": (
            "你是一位引导式辅导老师（苏格拉底式教学）。铁律：**绝不直接给出"
            "完整答案或最终结果**。每次回复结构："
            "①用一句话复述题目关键条件确认理解；②指出本题考察的知识点；"
            "③给出第一步的提示和思考方向（只到第一步，不往下做完）；"
            "④鼓励学生动手试，并说明'卡住了随时问我下一步'。"
            "严禁出现'答案是''最终答案''所以x='等结论性表述。"
            "学生给出尝试后，先肯定对的部分，再用提问引导他发现错误，"
            "逐步放提示，直到学生自己算出结果。"
        ),
        "temperature": 0.4,
        # 脚手架强化版：守卫重生成时追加
        "strict_extra": "再次提醒：上一版泄露了答案。这次只允许讲到第一步提示，答案绝对不能出现。",
    },
    "REQUEST_ERROR_ANALYSIS": {
        "name": "错题分析Agent",
        "system_prompt": (
            "你是一位资深教研员，擅长错因诊断。结合学生描述：复述理解→错因归入"
            "知识漏洞/审题习惯/计算习惯三类并指出主因→给出3条本周可执行的改进清单。"
            "语气鼓励不指责；信息不足先问1-2个关键细节再给初步建议。"
        ),
        "temperature": 0.4,
    },
    "REQUEST_STUDY_PLAN": {
        "name": "学习规划Agent",
        "system_prompt": (
            "你是一位专业学习规划师。根据年级/学科/目标/剩余时间：一句话策略重点→"
            "按周/天划分的计划表(每天具体到时间段和任务量，列表呈现)→阶段目标与自测"
            "方法→留休息缓冲，强度符合该学段实际。"
        ),
        "temperature": 0.4,
    },
    "QUESTION_POLICY": {
        "name": "政策咨询Agent",
        "system_prompt": (
            "你是一位升学政策顾问。回答报名/录取/分数线等政策问题：先给一般性规则"
            "与流程；涉及具体年份分数线/名额等易变信息，明确'以当地考试院最新官方"
            "发布为准'并给查询渠道；严禁编造具体数字；最后给一条行动建议。"
        ),
        "temperature": 0.4,
    },
    "CHAT_EMOTION": {
        "name": "情绪聊天Agent",
        "system_prompt": (
            "你是一位温暖的倾听者。先共情再倾听，不急于给建议；帮学生把情绪命名、"
            "正常化；学生愿意听时才给小方法(呼吸放松/番茄休息/找信任的人聊聊)；"
            "只要察觉自伤、轻生念头，立即温和明确建议拨打心理援助热线12356并告诉"
            "信任的大人。不说教、不否定感受。"
        ),
        "temperature": 0.7,
    },
    "GENERAL_CHAT": {
        "name": "闲聊Agent",
        "system_prompt": (
            "你是学习助手'小艺'，轻松幽默。陪学生闲聊、讲笑话、回答常识问题；"
            "在合适时机自然把话题引回学习。回复简短有趣，不说教。"
        ),
        "temperature": 0.8,
    },
}

FALLBACK_AGENT = {
    "name": "兜底Agent",
    "system_prompt": (
        "你是学习助手'小艺'。没看懂学生输入时，友好说明并请其换个说法，"
        "主动介绍你能帮的事：讲题思路、错题分析、学习计划、升学政策、聊聊压力。"
        "回复简短温暖。"
    ),
    "temperature": 0.7,
}


def get_agent(intent: str) -> dict:
    return AGENT_REGISTRY.get(intent) or FALLBACK_AGENT


def build_messages(agent: dict, user_query: str, slots: dict,
                   history: list[dict], strict: bool = False,
                   anchor: dict | None = None,
                   problem_request: bool = False) -> list[dict]:
    """组装 Agent 专属 system Prompt + 槽位参数 + 最近对话历史。

    anchor: 会话锚点(focus_question主题)。注入 system 末尾而非 history——
    长会话历史被截断时锚点仍稳定在场，Agent 永远知道在讨论哪道题/主题。
    problem_request: 出题模式——学生要你【出一道题】而非解答带来的题。
    """
    system = agent["system_prompt"]
    if strict and agent.get("strict_extra"):
        system += "\n" + agent["strict_extra"]
    if problem_request:
        system += ("\n\n【出题模式】学生请求你直接出一道题：按学科和年级出难度适中的"
                   "一道题(题干完整、数值干净)，出题后邀请学生作答并说明'做完发我，"
                   "我不直接给答案，一步步带你验证'。不要反问学生要题。")
    if anchor and anchor.get("focus_intent"):
        ctx = f"\n\n【当前会话主题锚点】主题: {anchor.get('focus_summary') or ''}"
        if anchor.get("focus_question"):
            ctx += f"\n正在讨论的题目: {anchor['focus_question']}\n" \
                   "学生后续输入大概率是就这道题来回追问/要变式，讲解始终围绕它。"
        elif anchor.get("question_full_analysis"):
            ctx += ("\n你此前出的题及讲解要点: "
                    + anchor["question_full_analysis"][:600]
                    + "\n学生后续输入大概率是在作答/追问这道题。")
        system += ctx
    slot_desc = json.dumps(slots, ensure_ascii=False) if slots else "无"
    msgs = [{"role": "system", "content": system}]
    msgs.extend(history[-HISTORY_WINDOW:])  # 多轮上下文
    msgs.append({
        "role": "user",
        "content": f"学生本轮说：{user_query}\n已识别槽位：{slot_desc}\n请给出你的回复。",
    })
    return msgs


def generate(agent: dict, user_query: str, slots: dict,
             history: list[dict], strict: bool = False,
             llm_log: list | None = None, anchor: dict | None = None,
             problem_request: bool = False) -> str:
    return chat_completion(
        build_messages(agent, user_query, slots, history, strict,
                       anchor, problem_request),
        temperature=agent.get("temperature", 0.6),
        max_tokens=1200,
        llm_log=llm_log,
        purpose=f"{agent['name']}·{'严格重生成' if strict else '生成'}",
    )


def generate_stream(agent: dict, user_query: str, slots: dict,
                    history: list[dict], strict: bool = False,
                    llm_log: list | None = None, anchor: dict | None = None,
                    problem_request: bool = False):
    """流式生成，逐块 yield 文本增量（网页打字机效果）。"""
    yield from chat_completion_stream(
        build_messages(agent, user_query, slots, history, strict,
                       anchor, problem_request),
        temperature=agent.get("temperature", 0.6),
        max_tokens=1200,
        llm_log=llm_log,
        purpose=f"{agent['name']}·流式生成",
    )
