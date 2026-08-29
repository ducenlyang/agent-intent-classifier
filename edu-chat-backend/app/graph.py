"""LangGraph 对话编排（方案版 5 节点工作流，支持流式自定义事件）：

START → ①call_intent_gateway(HTTP调网关,无状态单句识别)
      → ②slot_merge_and_router(缓存槽位合并+二次判缺+三路分支)
           ├─ A 风险 → ③risk_reply → END
           ├─ B 缺槽 → ②内生成反问话术 → END
           └─ C 齐全 → ④dispatch_agent(流式生成) → ⑤output_guard → END

流式协议（graph.stream(stream_mode="custom")，非流式 invoke 下自动忽略）：
  {"type":"meta", intent/route_kind/slots/missing_slots}   路由决策完成即推送
  {"type":"delta","text":...}                               答案逐块推送
  {"type":"replace","text":...}                             守卫兜底整段替换
  {"type":"done", reply/guard}                              本轮结束
"""
from __future__ import annotations

import re
import time
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import agents, gateway, slots
from .guard import SAFE_FALLBACK, guard, stream_violation
from .gateway import IntentResult, Slots
from .llm_client import chat_completion
from intent_classifier.config import PrimaryIntent, REQUIRED_SLOTS
from intent_classifier.slot_lexicon import is_problem_request

INTENT_ZH = {
    "QUESTION_SUBJECT": "学科问题", "QUESTION_POLICY": "政策咨询",
    "REQUEST_STUDY_PLAN": "学习计划", "REQUEST_ERROR_ANALYSIS": "错题分析",
    "CHAT_EMOTION": "情感倾诉", "REFUSE_CHEAT": "作弊拒绝",
    "GENERAL_CHAT": "闲聊", "UNKNOWN": "未识别",
}

# 对话行为(dialog act)：与域意图正交的维度——用户这句话在会话结构里扮演什么角色。
# 网关只做单句域意图识别；对话行为由DM结合锚点/状态消解，CONTINUE_CHAT即"继续锚点主题来回讨论"
DIALOG_ACT_ZH = {
    "CONTINUE_CHAT": "继续锚点对话",
    "TOPIC_SWITCH": "同意图切主题",
    "SLOT_ANSWER": "回答反问",
    "AFFIRM": "承接确认",
    "REFUSE": "拒绝终止",
    "CORRECTED": "矫正意图",
    "NEW_TASK": "新任务",
}


def _emit(event: dict) -> None:
    """向流式消费方推送自定义事件；非流式(invoke)上下文为 no-op。"""
    try:
        get_stream_writer()(event)
    except Exception:
        pass


class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # 对话历史自动追加
    session_id: str
    user_query: Optional[str]
    intent_result: Optional[IntentResult]  # 网关单轮原始结果
    cached_slots: Slots                     # 后端独有的多轮槽位记忆
    # ---- 会话焦点与任务状态机（DM独有，网关无状态）----
    last_active_intent: str                 # 会话焦点：上一轮正在运行的业务意图
    task_status: str                        # IDLE / IN_TASK / AWAIT_USER_REPLY
    last_prompt_type: str                   # 上一轮AI提问类型: ASK_SLOT/ASK_CONFIRM/NONE
    awaiting_asked: list[str]               # ASK_SLOT 时等待用户补充的槽位清单
    slot_states: dict[str, str]             # 槽位三态: uncollected/filled/pending_correction
    semantic_memory: dict                   # 会话锚点(focus_question/summary)，不进messages不参与截断
    problem_request: bool                   # 出题请求：路由答疑Agent出题模式(不反问收题)
    pending_direct_reply: str               # 消歧层直接生成的回复(拒绝/澄清)，跳过Agent
    # 会话锚点(semantic_memory)：当前讨论主题的长期记忆。不进LLM messages、
    # 不参与历史窗口截断，由 prompt 组装时手动注入，保障超长对话不"忘题"。
    # {focus_intent, focus_question(答疑锚点题干), focus_summary(主题一句话摘要)}
    semantic_memory: dict
    merged_slots: Slots                     # 合并后的完整槽位
    still_missing: list[str]
    route_kind: str                         # risk / clarify / agent
    draft_answer: Optional[str]
    final_answer: Optional[str]
    guard_info: dict
    llm_calls: list                         # 本轮全部LLM调用留痕(网关L3+消歧+Agent生成)
    latency_ms: int


def _messages_view(state: ChatState) -> list[dict]:
    """当前累积的多轮对话记录（含本轮user，assistant回复由done事件补充）。"""
    out = []
    for m in state.get("messages", []):
        role = getattr(m, "type", "user")
        out.append({"role": "assistant" if role == "ai" else "user",
                    "content": str(getattr(m, "content", ""))})
    return out


def _meta_event(ir: IntentResult, merged: Slots, missing: list[str],
                route_kind: str, state: ChatState = None) -> dict:
    return {
        "type": "meta",
        "intent": {
            "primary_intent": ir.primary_intent.value,
            "primary_intent_zh": INTENT_ZH.get(ir.primary_intent.value, "?"),
            "secondary_intent": ir.secondary_intent.value if ir.secondary_intent else None,
            "confidence": ir.confidence,
            "need_guide_only": ir.need_guide_only,
        },
        "route_kind": route_kind,
        "slots": {k: v for k, v in merged.model_dump().items() if v},
        "missing_slots": missing,
        "handled_by": ir.handled_by,
        "decision_trace": list(ir.decision_trace),  # 网关逐层决策轨迹(前端侧栏展示)
        "history": _messages_view(state) if state else [],  # 累积多轮messages
        "llm_calls": ([c.model_dump() for c in ir.llm_calls]
                      + list((state or {}).get("llm_calls") or [])),  # 网关L3+消歧留痕
        "anchor": _anchor_view(state),  # 会话锚点(focus_question主题)
        "dialog_act": ir.dialog_act,  # 对话行为(CONTINUE_CHAT继续锚点对话等)
    }


def _done_event(state: ChatState, reply: str, guard_info: dict,
                llm_calls: list | None = None) -> dict:
    hist = _messages_view(state)
    hist.append({"role": "assistant", "content": reply})
    return {"type": "done", "reply": reply, "guard": guard_info, "history": hist,
            "llm_calls": llm_calls or []}


# ---------------------------------------------------------------------------
# 节点1：调用意图网关（HTTP，无状态单句识别）
# ---------------------------------------------------------------------------
def call_intent_gateway(state: ChatState) -> dict:
    t0 = time.perf_counter()
    ir = gateway.classify(state["user_query"])
    print(f"[graph] 网关返回: intent={ir.primary_intent.value} "
          f"slots={ {k: v for k, v in ir.slots.model_dump().items() if v} } "
          f"missing={ir.missing_slots} ({time.perf_counter()-t0:.1f}s)")
    return {"intent_result": ir, "latency_ms": int((time.perf_counter() - t0) * 1000)}


_ELICIT_CUES = (  # 结尾含这些 → 视为 AI 在等待用户回答(宁可多检：多设的后果只是
    # UNKNOWN 回复继承原任务而非送兜底，通常更优；漏设则会错当新 query 路由)
    "？", "?", "告诉我", "说说", "说下", "补充", "你觉得", "吗", "呢",
    "哪", "什么", "怎么", "为什么", "多少", "几岁", "几年", "几年级",
    "选一个", "挑一个", "试试看", "发我", "发给", "贴一下",
)
# 消歧L1 词库：正向承接确认(委托/顺从) + 负向拒绝
AFFIRM_WORDS = (
    "好的", "好呀", "好嘞", "行", "行吧", "没问题", "可以", "ok", "OK",
    "嗯", "嗯嗯", "就这么办", "就这么定", "同意", "按你的建议来", "按你的思路来",
    "按你", "你决定", "你选", "你来定", "听你的", "随你", "你看着办", "都行",
    "就按", "按这个", "就这么", "照你", "就这样",
)
REFUSE_WORDS = ("不行", "不要", "不用了", "不了", "算了", "换一个", "换个", "重新来")

# 消歧L3 可矫正意图白名单(与网关 PrimaryIntent 对齐，排除 REFUSE_CHEAT/UNKNOWN)
_DISAMBIG_INTENTS = (
    "QUESTION_SUBJECT", "QUESTION_POLICY", "REQUEST_STUDY_PLAN",
    "REQUEST_ERROR_ANALYSIS", "CHAT_EMOTION", "GENERAL_CHAT",
)
_DISAMBIG_SYSTEM = (
    "你是对话系统的意图矫正器。下面给出对话历史和用户当前输入，"
    "请结合上下文判断用户本轮的真实意图。\n"
    "只能从这些意图中选择: " + ", ".join(_DISAMBIG_INTENTS) + "\n"
    "无法判断时输出 UNKNOWN。禁止编造列表外的意图。\n"
    '只输出 JSON: {"corrected_intent": "..."}'
)


def _ends_with_elicitation(text: str) -> bool:
    """回复结尾(最后80字)是否在向用户索要信息。"""
    tail = (text or "").rstrip()[-80:]
    return any(c in tail for c in _ELICIT_CUES)


def _is_affirm(q: str) -> bool:
    return bool(q) and len(q) <= 12 and any(w in q for w in AFFIRM_WORDS)


def _is_refuse(q: str) -> bool:
    return bool(q) and len(q) <= 12 and any(w in q for w in REFUSE_WORDS)


_PUNCT = "，。！？,.!?~～ 、"


def _pure_refuse(q: str, ir: IntentResult) -> bool:
    """区分纯拒绝与"放弃旧+转向新"的复合话语：
    "算了不用了"→纯拒绝，终止任务；
    "算了，我想看物理的"→拒绝词只是话语标记，句中还有新主题(subject=物理)，
    应放行正常路由由主题指纹切换，不能一刀切终止。
    判定：剥离拒绝词与标点后无实质内容、且网关未抽到任何主题槽位。"""
    residual = q
    for w in sorted(REFUSE_WORDS, key=len, reverse=True):
        residual = residual.replace(w, "")
    residual = residual.strip(_PUNCT)
    has_topic = bool(ir.slots.subject or ir.slots.topic
                     or ir.slots.time_horizon or ir.slots.grade)
    return not has_topic and len(residual) <= 2


# ---- 会话锚点(focus_question)配套信号 ----
VARIANT_KEYWORDS = ("变式", "换一道", "换一个题", "类似的题", "再出一道",
                    "再来一道题", "再来一个", "再出一个", "再来一题",
                    "同样的题", "同类型的题")
KNOWLEDGE_KEYWORDS = ("知识点", "定义", "概念", "什么叫", "什么是", "怎么理解", "是什么意思")


def _anchor_summary(intent: str, slot_dict: dict) -> str:
    """锚点主题摘要：如「高二·数学·学习计划」。"""
    parts = [slot_dict.get("grade"), slot_dict.get("subject"),
             INTENT_ZH.get(intent, intent)]
    return "·".join(p for p in parts if p)


def _anchor_view(state: ChatState) -> dict:
    """给前端/meta事件的可视化锚点(仅展示字段)。"""
    a = state.get("semantic_memory") or {}
    if not a.get("focus_intent"):
        return {}
    return {"summary": a.get("focus_summary") or "",
            "question": a.get("focus_question") or "",
            "analysis": (a.get("question_full_analysis") or "")[:120]}


def _clarify_reply(active: str | None) -> str:
    """消歧L4 定向澄清：结合会话焦点给出二选一，而非干巴巴的没听懂。"""
    if active and active in INTENT_ZH and active != "UNKNOWN":
        return (f"我有点没理解你的意思～你是想继续刚才的「{INTENT_ZH[active]}」，"
                f"还是有别的新问题呢？")
    return "我没太理解你的意思，可以换个说法吗？讲题思路、错题分析、学习计划、升学政策、聊聊天都可以～"


def _inherit(ir: IntentResult, target: str, note: str,
             act: str | None = None) -> IntentResult:
    """继承会话焦点意图：替换意图+清除误填的question_text+按目标意图重算缺失。
    act: 对话行为标签(CONTINUE_CHAT等)，与域意图正交，仅用于展示/埋点。"""
    ir2 = ir.model_copy(update={
        "primary_intent": PrimaryIntent(target), "handled_by": "CONTEXT_INHERIT",
        "dialog_act": act,
        # 只清question_text(防把追问文本当题目)；subject/grade等本轮显式值
        # 是用户最新意志(如"改成数学")，保留供槽位合并与主题指纹比对
        "slots": ir.slots.model_copy(update={"question_text": None})})
    ir2.decision_trace = ir.decision_trace + [note]
    ir2.missing_slots = [f for f in REQUIRED_SLOTS.get(ir2.primary_intent, [])
                         if not getattr(ir2.slots, f, None)]
    return ir2


# ---------------------------------------------------------------------------
# 节点1.5：四层 UNKNOWN 消歧流水线（DST+Policy，全部在后端，网关保持无状态）
#   L1 规则词库(承接确认/拒绝) → L2 会话焦点继承(状态机门控) 
#   → L3 上下文LLM矫正 → L4 定向澄清反问。命中即终止。
# ---------------------------------------------------------------------------
# ---- 槽位修正(TODO-oriented经典缺陷修复)：用户要改已填槽位 ----
# 三态: uncollected未收集 / filled已收集 / pending_correction待修改
CORRECT_WORDS = ("不对", "说错了", "改一下", "换个", "换一个", "换科目",
                 "重新说", "不是这个", "改成", "我要改", "填错了")


def _wants_correction(q: str) -> bool:
    return any(w in q for w in CORRECT_WORDS)


def _slot_states_of(state: dict, merged_slots) -> dict:
    """必填槽位三态快照: filled / uncollected（pending_correction 只在修正轮出现）。"""
    from .config import REQUIRED_SLOTS
    from .gateway import IntentResult
    ir = state.get("intent_result")
    intent = ir.primary_intent.value if ir else "UNKNOWN"
    return {f: ("filled" if getattr(merged_slots, f, None) else "uncollected")
            for f in REQUIRED_SLOTS.get(intent, [])}


def contextual_resolve_node(state: ChatState) -> dict:
    ir: IntentResult = state["intent_result"]
    q = (state.get("user_query") or "").strip()
    cur = ir.primary_intent.value
    status = state.get("task_status") or "IDLE"
    active = state.get("last_active_intent")
    prompt_type = state.get("last_prompt_type") or "NONE"
    asked = state.get("awaiting_asked") or []
    sid = state.get("session_id", "-")
    busy = status in ("IN_TASK", "AWAIT_USER_REPLY")

    # -- 槽位修正：等待补充期间用户要改已答内容("不对，是高一"/"换个学科")
    #    处理：命中修正词→本轮能抽到的新值直接覆盖(该槽=filled)，
    #    抽不到新值的等待槽=pending_correction并单独重问；任务绝不终止，其余槽位保留。
    if status == "AWAIT_USER_REPLY" and prompt_type == "ASK_SLOT" and asked \
            and _wants_correction(q):
        new_vals = {f: getattr(ir.slots, f) for f in asked
                    if getattr(ir.slots, f, None)}
        reask = [f for f in asked if f not in new_vals]
        if new_vals:
            ir2 = _inherit(ir, active, f"消歧: 槽位修正，覆盖{list(new_vals)}，"
                                       f"任务继续(其余槽位保留)", act="CORRECTED")
            for f, v in new_vals.items():
                setattr(ir2.slots, f, v)
            ir2.missing_slots = reask
            if reask:  # 还有没给新值的槽位 → 单独重问
                from .slots import CLARIFY_QUESTIONS
                reply = "好的，已修改。那" + " ".join(
                    CLARIFY_QUESTIONS.get(f, f"请补充{f}") for f in reask)
                print(f"[dm] 消歧 session={sid} layer=SLOT_CORRECT "
                      f"覆盖={new_vals} 待重问={reask}")
                return {"intent_result": ir2, "awaiting_asked": reask,
                        "task_status": "AWAIT_USER_REPLY",
                        "last_prompt_type": "ASK_SLOT",
                        "pending_direct_reply": reply, "llm_calls": []}
            print(f"[dm] 消歧 session={sid} layer=SLOT_CORRECT 覆盖={new_vals} 任务继续")
            return {"intent_result": ir2, "awaiting_asked": [],
                    "last_prompt_type": "NONE", "llm_calls": []}
        # 修正词但没说新值 → 全部等待槽位转 pending_correction，重问第一等待槽
        f0 = asked[0]
        from .slots import CLARIFY_QUESTIONS
        reply = "没问题，我们改一下。" + CLARIFY_QUESTIONS.get(f0, f"请补充{f0}")
        print(f"[dm] 消歧 session={sid} layer=SLOT_CORRECT 重问={f0}")
        return {"pending_direct_reply": reply,
                "task_status": "AWAIT_USER_REPLY", "last_prompt_type": "ASK_SLOT",
                "awaiting_asked": [f0], "llm_calls": [],
                "intent_result": ir.model_copy(update={
                    "dialog_act": "CORRECTED",
                    "decision_trace": ir.decision_trace + [
                        f"消歧: 修正请求(未带新值)，槽位{asked}转待修改并重问"]})}

    # -- 槽位反问等待：纯值/短回答直接绑定槽位("化学"/"高二数学不好")
    if status == "AWAIT_USER_REPLY" and prompt_type == "ASK_SLOT" and asked:
        filled = {f: getattr(ir.slots, f) for f in asked if getattr(ir.slots, f, None)}
        asked_hit = any(getattr(ir.slots, f, None) == q for f in asked)
        if filled and (asked_hit or len(q) <= 10):
            ir2 = _inherit(ir, active, f"上下文消解: 反问等待中，回复绑定槽位{filled}，"
                                       f"继承意图{active}继续原任务", act="SLOT_ANSWER")
            fq0 = (state.get("semantic_memory") or {}).get("focus_question")
            for f, v in filled.items():  # 恢复刚绑定的槽位值(_inherit默认清question_text)
                if f == "question_text" and fq0 and not slots.has_new_question(v):
                    continue  # 绑的值不像题(实为追问)且锚点有真题 → 让锚点恢复接管
                setattr(ir2.slots, f, v)
            ir2.missing_slots = [f for f in asked if f not in filled]
            print(f"[dm] 消歧 session={sid} layer=SLOT_BIND bound={filled}")
            return {"intent_result": ir2, "awaiting_asked": [],
                    "last_prompt_type": "NONE", "llm_calls": [],
                    "slot_states": {f: "filled" for f in asked if f in filled}
                    | {f: "uncollected" for f in asked if f not in filled}}

    if cur != "UNKNOWN":
        # 网关判明意图。若与锚点主题一致且无新题信号 → 印证为继续锚点对话
        # (CONTINUE_CHAT是对话行为标签，域意图仍由网关结果决定，不改路由)
        anchor0 = state.get("semantic_memory") or {}
        fi0 = anchor0.get("focus_intent")
        cleanup = ({"awaiting_asked": [], "last_prompt_type": "NONE"}
                   if status == "AWAIT_USER_REPLY" else {})
        # 主题指纹比对：同意图但学科漂移(数学→化学) → TOPIC_SWITCH 非继续对话
        subj_now = ir.slots.subject or (state.get("cached_slots") or Slots()).subject
        subj_anchor = anchor0.get("focus_subject")
        drifted = bool(subj_now and subj_anchor and subj_now != subj_anchor)
        if fi0 and cur == fi0 and not slots.has_new_question(q) and not drifted:
            ir2 = ir.model_copy(update={"dialog_act": "CONTINUE_CHAT"})
            ir2.decision_trace = ir.decision_trace + [
                f"对话行为: 网关意图与锚点({anchor0.get('focus_summary') or fi0})"
                f"一致且无新题信号，判定 CONTINUE_CHAT 继续锚点对话"
            ]
            print(f"[dm] 对话行为 session={sid} act=CONTINUE_CHAT(锚点印证) intent={cur}")
            cleanup["intent_result"] = ir2
        elif (fi0 in slots.SUBJECT_TASK_INTENTS and cur in slots.SUBJECT_TASK_INTENTS
                and cur != fi0 and not slots.has_new_question(q)
                and len(q) <= 14
                and any(w in q for w in ("这", "那", "刚才", "为什么", "怎么"))
                and not any(w in q for w in ("帮我", "给我", "我要", "计划", "分析",
                                              "总结", "错题", "出题", "再来", "换"))):
            # 指代式追问("为什么这步错了")：网关判成相邻意图，但语义是追问锚点
            # 正在讨论的内容 → 跟随锚点意图而非开启新任务
            ir4 = _inherit(ir, fi0,
                           f"对话行为: 指代式追问命中相邻意图{cur}，跟随锚点意图{fi0}",
                           act="CONTINUE_CHAT")
            cleanup["intent_result"] = ir4
            print(f"[dm] 对话行为 session={sid} act=CONTINUE_CHAT(指代追问) {cur}->{fi0}")
        elif drifted and cur == fi0:
            ir2 = ir.model_copy(update={"dialog_act": "TOPIC_SWITCH"})
            ir2.decision_trace = ir.decision_trace + [
                f"对话行为: 学科漂移 {subj_anchor}→{subj_now}，判定 TOPIC_SWITCH "
                f"同意图切主题(旧题锚点将在路由时清空)"
            ]
            print(f"[dm] 对话行为 session={sid} act=TOPIC_SWITCH {subj_anchor}->{subj_now}")
            cleanup["intent_result"] = ir2
        elif fi0 and cur != "GENERAL_CHAT":
            ir3 = ir.model_copy(update={"dialog_act": "NEW_TASK"})
            ir3.decision_trace = ir.decision_trace + [
                "对话行为: 网关识别到锚点外新意图，判定 NEW_TASK 开启新任务"
            ]
            cleanup["intent_result"] = ir3
        return cleanup

    # ---- L0前哨：明确对话行为最优先(标签准确) ----
    # 承接确认("就按这个执行")优先于锚点继承
    if busy and active and _is_affirm(q) and not _is_refuse(q):
        print(f"[dm] 消歧 session={sid} layer=L1_AFFIRM -> {active}")
        return {"intent_result": _inherit(
            ir, active, f"消歧L1: 承接确认词命中，继续会话焦点{active}", act="AFFIRM"),
            "awaiting_asked": [], "last_prompt_type": "NONE"}
    # 纯拒绝终止("算了不用了")；"算了，我想看物理的"类复合话语(拒绝词+新主题)
    # 不在此终止，放行正常路由由主题指纹切换
    if _is_refuse(q) and _pure_refuse(q, ir):
        print(f"[dm] 消歧 session={sid} layer=L1_REFUSE -> 终止任务")
        return {"pending_direct_reply": "好的，那咱们就先到这里～后续有别的需求随时告诉我。",
                "task_status": "IDLE", "last_prompt_type": "NONE",
                "awaiting_asked": [],
                "intent_result": ir.model_copy(update={
                    "dialog_act": "REFUSE",
                    "decision_trace": ir.decision_trace + ["消歧L1: 拒绝词命中，终止当前任务"]})}

    # ---- L0 锚点门控 FOLLOW_UP：有锚点+无新题信号 → 省略/含糊回复=继续锚点主题 ----
    # 锚点(semantic_memory.focus_intent)是比task_status更强的会话状态：锚点只在
    # 主题/意图切换时变更，故无锚点时(闲聊/新会话)本层不点火，歧义下沉L1-L4。
    # 网关明确判出非UNKNOWN意图时不会到达这里——新主题由网关负责，锚点只管"来回讨论"。
    anchor = state.get("semantic_memory") or {}
    focus_intent = anchor.get("focus_intent")
    # 本轮显式切换学科("改成数学")：用户明确意志优先于锚点继承，跳过L0
    explicit_switch = (ir.slots.subject and anchor.get("focus_subject")
                       and ir.slots.subject != anchor["focus_subject"])
    if focus_intent and not explicit_switch and not slots.has_new_question(q):
        variant = any(k in q for k in VARIANT_KEYWORDS)
        print(f"[dm] 消歧 session={sid} layer=L0_ANCHOR -> {focus_intent}"
              f"{'(变式请求)' if variant else ''}")
        return {"intent_result": _inherit(
            ir, focus_intent,
            f"消歧L0: 锚点存在(主题:{anchor.get('focus_summary') or focus_intent})"
            f"且无新题信号{'，变式请求' if variant else ''}，"
            f"判CONTINUE_CHAT继续锚点主题(跟随锚点意图{focus_intent})",
            act="CONTINUE_CHAT"),
            "awaiting_asked": [], "last_prompt_type": "NONE"}

    # ---- L2 会话焦点继承(有限状态机门控：仅任务中，IDLE 禁止继承防串任务) ----
    if busy and active:
        # 继承意图，但若本轮显式说了新学科("算了，想看物理")→ 标 TOPIC_SWITCH，
        # 主题指纹稍后在路由层完成切换(清旧题+换主题)
        drift_here = (ir.slots.subject and focus_intent
                      and (state.get("semantic_memory") or {}).get("focus_subject")
                      and ir.slots.subject != (state["semantic_memory"] or {}).get("focus_subject"))
        act2 = ("TOPIC_SWITCH" if drift_here
                else "CONTINUE_CHAT" if focus_intent and active == focus_intent else None)
        print(f"[dm] 消歧 session={sid} layer=L2_INHERIT -> {active}"
              f"{'(学科漂移)' if drift_here else ''}")
        return {"intent_result": _inherit(
            ir, active, f"消歧L2: 会话焦点继承{active}(task_status={status})，"
                        f"连同历史交回原Agent", act=act2),
            "awaiting_asked": [], "last_prompt_type": "NONE"}

    # ---- L3 上下文LLM矫正(后端独立调用，网关不参与；仅UNKNOWN触发控制成本) ----
    log: list = []
    corrected = None
    try:
        history = _messages_view(state)[-4:]
        msgs = ([{"role": "system", "content": _DISAMBIG_SYSTEM}]
                + history + [{"role": "user", "content": q}])
        content = chat_completion(msgs, temperature=0.1, max_tokens=50,
                                  llm_log=log, purpose="上下文消歧(L3)")
        m = re.search(r'"corrected_intent"\s*:\s*"([^"]+)"', content)
        corrected = m.group(1) if m and m.group(1) in _DISAMBIG_INTENTS else None
    except Exception as e:
        print(f"[dm] 消歧L3 LLM调用失败: {e}")
    rec = log[0] if log else {}
    if corrected:
        print(f"[dm] 消歧 session={sid} layer=L3_LLM -> {corrected}")
        return {"intent_result": _inherit(
            ir, corrected, f"消歧L3: 上下文LLM矫正意图→{corrected}", act="CORRECTED"),
            "task_status": "IN_TASK", "last_active_intent": corrected,
            "awaiting_asked": [], "last_prompt_type": "NONE",
            "llm_calls": [rec]}

    # ---- L4 定向澄清：结合会话焦点二选一。任务进行中保留任务状态
    #      (用户重试即可继续，不清空已收集进度——修复"未识别直接结束任务"缺陷) ----
    print(f"[dm] 消歧 session={sid} layer=L4_CLARIFY")
    keep = {"task_status": status, "last_active_intent": active} if busy else            {"task_status": "IDLE"}
    return {"pending_direct_reply": _clarify_reply(active),
            **keep, "last_prompt_type": "NONE", "awaiting_asked": [],
            "llm_calls": [rec] if rec else [],
            "intent_result": ir.model_copy(update={
                "decision_trace": ir.decision_trace + ["消歧L4: 各层未命中，定向澄清反问"]})}


# ---------------------------------------------------------------------------
# 节点2：槽位合并 + 分支路由（后端独有逻辑）
# ---------------------------------------------------------------------------
def slot_merge_and_router(state: ChatState) -> dict:
    ir: IntentResult = state["intent_result"]
    # 话题切换检测：任务意图跨组切换 → 任务作用域槽位(subject/topic/time_horizon)失效，
    # 仅保留用户稳定属性(grade)；学科任务组内切换(讲题→错题→计划)视为同一话题链
    cur = ir.primary_intent.value
    prev = state.get("last_active_intent")
    cached = state.get("cached_slots")
    same_chain = (cur in slots.SUBJECT_TASK_INTENTS and prev in slots.SUBJECT_TASK_INTENTS)
    # 话题切换(跨组) → 任务作用域槽位失效 + 会话锚点清空(旧题讨论结束)
    anchor_clear: dict = {}
    if cur and prev and cur != prev and not same_chain:
        cached = slots.drop_task_scoped(cached)
        if (state.get("semantic_memory") or {}).get("focus_intent"):
            anchor_clear = {"semantic_memory": {}}
            print(f"[dm] 锚点清空: 话题切换 {prev} -> {cur}")
    merged = slots.merge_slots(ir, cached)
    missing = slots.still_missing(ir, merged)
    anchor_mem = state.get("semantic_memory") or {}
    # 主题指纹裁决(DM权威位置)：同意图下 subject 漂移(数学→化学) → 主题切换：
    # 更新锚点主题字段并清掉旧题锚点(旧学科的题不属于新主题)，防"化学主题挂数学题"
    same_topic_group = (anchor_mem.get("focus_intent") in slots.SUBJECT_TASK_INTENTS
                        and cur in slots.SUBJECT_TASK_INTENTS)
    if ((anchor_mem.get("focus_intent") == cur or same_topic_group)
            and anchor_mem.get("focus_subject") and merged.subject
            and merged.subject != anchor_mem["focus_subject"]):
        old_subj = anchor_mem["focus_subject"]
        anchor_mem = {**anchor_mem, "focus_subject": merged.subject,
                      "focus_question": None, "question_full_analysis": None,
                      "focus_summary": _anchor_summary(cur, merged.model_dump())}
        anchor_update = {"semantic_memory": anchor_mem}
        ir.decision_trace = list(ir.decision_trace) + [
            f"锚点主题切换: 学科 {old_subj}→{merged.subject}，清空旧题锚点"
        ]
        print(f"[dm] 锚点主题切换: {old_subj} -> {merged.subject}(旧题已清)")
    else:
        anchor_update = {}
    # CONTINUE_CHAT 追问：锚点题目就是当前正在讲的题 → 恢复 question_text 免反问。
    # (继承路径会清question_text防污染，但对锚点主题的追问而言题目不该丢)
    anchor_q = anchor_mem.get("focus_question") or (
        (anchor_mem.get("question_full_analysis") or "")[:200])
    if (ir.dialog_act in ("CONTINUE_CHAT", "SLOT_ANSWER")
            and anchor_q and not merged.question_text
            and "question_text" in missing):
        merged.question_text = anchor_q
        missing = [f for f in missing if f != "question_text"]
        src = "锚点题目" if anchor_mem.get("focus_question") else "AI出题记录"
        ir.decision_trace = list(ir.decision_trace) + [
            f"锚点恢复: 追问沿用{src}「{anchor_q[:30]}」"
        ]
        print(f"[dm] 锚点恢复: question_text <- {src}「{anchor_q[:30]}」")
    # 学科推断兜底：题目原文在场但 subject 缺失 → 从题面关键词推断，免反问
    if "subject" in missing and merged.question_text:
        inferred = slots.infer_subject(merged.question_text)
        if inferred:
            merged.subject = inferred
            missing = [f for f in missing if f != "subject"]
            ir.decision_trace = list(ir.decision_trace) + [
                f"槽位推断: 题面关键词推断 subject={inferred}"
            ]
            print(f"[dm] 学科推断: subject={inferred}(来自题面关键词)")
    # 会话焦点：UNKNOWN 不作为焦点记录(沿用上一个有效焦点)，避免"继续刚才的未识别"
    keep_intent = cur if cur and cur != "UNKNOWN" else prev
    llm_calls = [c.model_dump() for c in ir.llm_calls] + list(state.get("llm_calls") or [])

    direct = state.get("pending_direct_reply")  # 消歧层直接回复(拒绝/澄清) → 跳过Agent
    if direct:
        _emit(_meta_event(ir, merged, [], "clarify", state))
        _emit(_done_event(state, direct, {}, llm_calls))
        return {"merged_slots": merged, "still_missing": [],
                "route_kind": "clarify", "final_answer": direct,
                "cached_slots": slots.cacheable(merged),
                "last_active_intent": keep_intent, "llm_calls": llm_calls,
                "task_status": "IDLE", "last_prompt_type": "NONE",
                "awaiting_asked": [], "pending_direct_reply": None,
                "messages": [{"role": "assistant", "content": direct}]}

    if ir.primary_intent.value == "REFUSE_CHEAT" or ir.risk.cheat_risk or (
            ir.risk.psych_risk == "high" and ir.reply):
        _emit(_meta_event(ir, merged, missing, "risk", state))
        return {"merged_slots": merged, "still_missing": missing,
                "route_kind": "risk", "cached_slots": slots.cacheable(merged),
                "last_active_intent": keep_intent, "llm_calls": llm_calls,
                "task_status": "IDLE", "last_prompt_type": "NONE",
                "awaiting_asked": [], **anchor_clear}

    if missing and cur == "QUESTION_SUBJECT" and missing == ["question_text"]             and is_problem_request(state.get("user_query") or ""):
        # 出题请求("来一个化学题")：不反问收题，直接路由答疑Agent出题模式
        ir.decision_trace = list(ir.decision_trace) + [
            "出题模式: 出题请求命中(动词×名词组合)，路由Agent直接出题"
        ]
        print(f"[dm] 出题模式: session={state.get('session_id','-')}")
        _emit(_meta_event(ir, merged, [], "agent", state))
        return {"merged_slots": merged, "still_missing": [],
                "route_kind": "agent", "cached_slots": slots.cacheable(merged),
                "last_active_intent": keep_intent, "llm_calls": llm_calls,
                "problem_request": True, "task_status": "IN_TASK",
                "last_prompt_type": "NONE", "awaiting_asked": [], **anchor_update}

    if missing:  # 分支B：反问话术直接生成并流式下发 → END
        questions = " ".join(
            slots.CLARIFY_QUESTIONS.get(f, f"请补充{f}") for f in missing
        )
        reply = f"好嘞，先确认一下：{questions}"
        _emit(_meta_event(ir, merged, missing, "clarify", state))
        _emit(_done_event(state, reply, {}, llm_calls))
        return {"merged_slots": merged, "still_missing": missing,
                "route_kind": "clarify", "final_answer": reply,
                "cached_slots": slots.cacheable(merged),
                "last_active_intent": keep_intent, "llm_calls": llm_calls,
                # 进入槽位等待状态：记录期待的槽位清单(有类型绑定供下轮消解)
                "task_status": "AWAIT_USER_REPLY", "last_prompt_type": "ASK_SLOT",
                "awaiting_asked": list(missing),
                "slot_states": {**{f: "filled" for f in missing if getattr(merged, f, None)},
                                **{f: "uncollected" for f in missing if not getattr(merged, f, None)}},
                "messages": [{"role": "assistant", "content": reply}]}

    _emit(_meta_event(ir, merged, [], "agent", state))
    return {"merged_slots": merged, "still_missing": [],
            "route_kind": "agent", "cached_slots": slots.cacheable(merged),
            "last_active_intent": keep_intent, "llm_calls": llm_calls,
            **anchor_clear, **anchor_update}


def route_after_merge(state: ChatState) -> str:
    kind = state["route_kind"]
    if kind == "risk":
        return "risk_reply"
    if kind == "clarify":
        return END
    return "dispatch_agent"


# ---------------------------------------------------------------------------
# 节点3：风险回复（网关L1话术直接下发）
# ---------------------------------------------------------------------------
def risk_reply_node(state: ChatState) -> dict:
    ir: IntentResult = state["intent_result"]
    reply = ir.reply or "这个忙帮不了哦～咱们聊聊学习吧。"
    _emit(_done_event(state, reply, {}, state.get("llm_calls")))
    return {"final_answer": reply, "task_status": "IDLE",
            "last_prompt_type": "NONE", "awaiting_asked": [],
            "messages": [{"role": "assistant", "content": reply}]}


# ---------------------------------------------------------------------------
# 节点4：Agent 分发（流式生成 + 增量守卫掐断）
# ---------------------------------------------------------------------------
def _history(state: ChatState) -> list[dict]:
    out = []
    for m in state.get("messages", [])[:-1]:  # 最后一条是本轮user，已单独传
        role = getattr(m, "type", "user")
        out.append({"role": "assistant" if role == "ai" else "user",
                    "content": str(getattr(m, "content", ""))})
    return out


def dispatch_agent_node(state: ChatState) -> dict:
    ir: IntentResult = state["intent_result"]
    agent = agents.get_agent(ir.primary_intent.value)
    slot_dict = {k: v for k, v in state["merged_slots"].model_dump().items() if v}
    history = _history(state)
    guide_only = ir.need_guide_only
    t0 = time.perf_counter()

    parts: list[str] = []
    llm_log: list = state.get("llm_calls") or []  # 续接网关L3留痕，追加Agent生成记录
    anchor = state.get("semantic_memory") or None  # 会话锚点注入system(不进history)
    prob_req = bool(state.get("problem_request"))
    try:
        for delta in agents.generate_stream(agent, state["user_query"],
                                            slot_dict, history,
                                            llm_log=llm_log, anchor=anchor,
                                            problem_request=prob_req):
            bad = stream_violation("".join(parts) + delta, need_guide_only=guide_only)
            if bad:  # 增量守卫命中：立即掐断并替换
                _emit({"type": "delta",
                       "text": "\n\n⚠️ [守卫拦截] 以上内容已撤回，替换为：\n\n"})
                _emit({"type": "replace", "text": SAFE_FALLBACK})
                parts = [SAFE_FALLBACK]
                break
            parts.append(delta)
            _emit({"type": "delta", "text": delta})
    except Exception as e:  # 流式失败 → 非流式重试一次
        print(f"[graph] 流式生成失败({e})，回退非流式")
        try:
            text = agents.generate(agent, state["user_query"], slot_dict, history,
                                   llm_log=llm_log)
            _emit({"type": "replace", "text": text})
            parts = [text]
        except Exception as e2:
            print(f"[graph] Agent生成失败({e2})")
            text = "哎呀，我这边开小差了，能再说一遍吗？"
            _emit({"type": "replace", "text": text})
            parts = [text]

    draft = "".join(parts).strip()
    print(f"[graph] {agent['name']} 生成 {time.perf_counter()-t0:.1f}s")
    return {"draft_answer": draft, "llm_calls": llm_log}


# ---------------------------------------------------------------------------
# 节点5：输出守卫（终检；罕见漏网时整段替换并通知前端）
# ---------------------------------------------------------------------------
def output_guard_node(state: ChatState) -> dict:
    ir: IntentResult = state["intent_result"]
    agent = agents.get_agent(ir.primary_intent.value)
    slot_dict = {k: v for k, v in state["merged_slots"].model_dump().items() if v}
    draft = state["draft_answer"]
    verdict = guard(draft, need_guide_only=ir.need_guide_only)

    text = draft
    if not verdict["passed"] and verdict["regenerate"]:
        try:  # 脚手架泄露 → 严格模式重生成一次
            text = agents.generate(agent, state["user_query"], slot_dict,
                                   _history(state), strict=True,
                                   llm_log=state.get("llm_calls"),
                                   anchor=state.get("semantic_memory") or None)
            verdict = guard(text, need_guide_only=ir.need_guide_only)
            if not verdict["passed"]:
                text = SAFE_FALLBACK
                verdict["actions"].append("重生成仍泄露，替换安全引导话术")
        except Exception as e:
            text = SAFE_FALLBACK
            verdict["actions"].append(f"重生成失败({e})，替换安全引导话术")
    elif not verdict["passed"]:
        text = SAFE_FALLBACK
    if text != draft:
        _emit({"type": "replace", "text": text})  # 整段替换已流出内容

    _emit(_done_event(state, text,
                      {"passed": verdict["passed"], "actions": verdict["actions"]},
                      state.get("llm_calls")))
    # Agent 回复结尾在向用户索要信息 → 进入追问等待(下轮承接/省略回复继承焦点)
    if _ends_with_elicitation(text):
        post = {"task_status": "AWAIT_USER_REPLY", "last_prompt_type": "ASK_CONFIRM"}
    else:
        post = {"task_status": "IN_TASK", "last_prompt_type": "NONE"}

    # ---- 锚点维护：任务型Agent成功回复 → 建立/刷新会话锚点 ----
    # 答疑意图：有新题信号(has_new_question)才锚定新题；追问保持旧题锚点不变。
    # 其余任务意图锚定到主题摘要。闲聊/未识别不设锚点。
    new_anchor: dict = {}
    intent_v = ir.primary_intent.value
    if intent_v not in ("GENERAL_CHAT", "UNKNOWN", "REFUSE_CHEAT"):
        s = state["merged_slots"].model_dump()
        old = state.get("semantic_memory") or {}
        fq = None
        if intent_v == "QUESTION_SUBJECT":
            if slots.has_new_question(state.get("user_query") or ""):
                # 新题 → 锚定新题(绑定/继承路径下question_text可能未被抽取，
                # 用户原句本身就是题干，兜底取原句)
                fq = s.get("question_text") or (state.get("user_query") or "").strip()
            else:
                # 追问/变式 → 保持旧题；首建锚时仅当本轮文本像题才兜底进锚
                qt = s.get("question_text")
                fq = old.get("focus_question") or (
                    qt if qt and slots.has_new_question(qt) else None)
        same_q = fq and old.get("focus_question") == fq
        if not same_q:  # 同题：不重建
            mem = {
                "focus_intent": intent_v,
                "focus_subject": s.get("subject"),
                "focus_grade": s.get("grade"),
                "focus_summary": _anchor_summary(intent_v, s),
                "focus_question": fq,
            }
            if state.get("problem_request"):
                # 出题轮：题目由Agent生成(在回复里)，存全文供后续追问锚定
                mem["question_full_analysis"] = text[:800]
            new_anchor = {"semantic_memory": mem}
            print(f"[dm] 锚点{'更新' if old else '建立'}: {_anchor_summary(intent_v, s)}"
                  f"{f' | 题目: {fq[:30]}' if fq else ' | (AI出题,存解析)'}")
    return {"final_answer": text,
            "guard_info": {"passed": verdict["passed"],
                           "actions": verdict["actions"]},
            "last_active_intent": ir.primary_intent.value,
            "awaiting_asked": [],
            **post, **new_anchor,
            "messages": [{"role": "assistant", "content": text}]}


def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("call_intent_gateway", call_intent_gateway)
    builder.add_node("contextual_resolve", contextual_resolve_node)
    builder.add_node("slot_merge_and_router", slot_merge_and_router)
    builder.add_node("risk_reply", risk_reply_node)
    builder.add_node("dispatch_agent", dispatch_agent_node)
    builder.add_node("output_guard", output_guard_node)

    builder.add_edge(START, "call_intent_gateway")
    builder.add_edge("call_intent_gateway", "contextual_resolve")
    builder.add_edge("contextual_resolve", "slot_merge_and_router")
    builder.add_conditional_edges(
        "slot_merge_and_router", route_after_merge,
        {"risk_reply": "risk_reply", END: END, "dispatch_agent": "dispatch_agent"},
    )
    builder.add_edge("risk_reply", END)
    builder.add_edge("dispatch_agent", "output_guard")
    builder.add_edge("output_guard", END)
    return builder.compile(checkpointer=MemorySaver())


# 会话存储：内存字典 {session_id: state}（MemorySaver 按 thread_id 管理）
chat_graph = build_graph()


def run_turn(session_id: str, user_query: str) -> dict:
    return chat_graph.invoke(
        {"session_id": session_id, "user_query": user_query,
         "messages": [{"role": "user", "content": user_query}]},
        config={"configurable": {"thread_id": session_id}},
    )


def stream_turn(session_id: str, user_query: str):
    """流式执行一轮对话，逐个 yield 自定义事件 dict（SSE 用）。"""
    yield from chat_graph.stream(
        {"session_id": session_id, "user_query": user_query,
         "messages": [{"role": "user", "content": user_query}]},
        config={"configurable": {"thread_id": session_id}},
        stream_mode="custom",
    )
