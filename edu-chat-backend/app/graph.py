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

INTENT_ZH = {
    "QUESTION_SUBJECT": "学科问题", "QUESTION_POLICY": "政策咨询",
    "REQUEST_STUDY_PLAN": "学习计划", "REQUEST_ERROR_ANALYSIS": "错题分析",
    "CHAT_EMOTION": "情感倾诉", "REFUSE_CHEAT": "作弊拒绝",
    "GENERAL_CHAT": "闲聊", "UNKNOWN": "未识别",
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
    merged_slots: Slots                     # 合并后的完整槽位
    still_missing: list[str]
    route_kind: str                         # risk / clarify / agent
    draft_answer: Optional[str]
    final_answer: Optional[str]
    guard_info: dict
    latency_ms: int


def _meta_event(ir: IntentResult, merged: Slots, missing: list[str],
                route_kind: str) -> dict:
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
    }


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


# ---------------------------------------------------------------------------
# 节点2：槽位合并 + 分支路由（后端独有逻辑）
# ---------------------------------------------------------------------------
def slot_merge_and_router(state: ChatState) -> dict:
    ir: IntentResult = state["intent_result"]
    merged = slots.merge_slots(ir, state.get("cached_slots"))
    missing = slots.still_missing(ir, merged)

    if ir.primary_intent.value == "REFUSE_CHEAT" or ir.risk.cheat_risk or (
            ir.risk.psych_risk == "high" and ir.reply):
        _emit(_meta_event(ir, merged, missing, "risk"))
        return {"merged_slots": merged, "still_missing": missing,
                "route_kind": "risk", "cached_slots": slots.cacheable(merged)}

    if missing:  # 分支B：反问话术直接生成并流式下发 → END
        questions = " ".join(
            slots.CLARIFY_QUESTIONS.get(f, f"请补充{f}") for f in missing
        )
        reply = f"好嘞，先确认一下：{questions}"
        _emit(_meta_event(ir, merged, missing, "clarify"))
        _emit({"type": "done", "reply": reply, "guard": {}})
        return {"merged_slots": merged, "still_missing": missing,
                "route_kind": "clarify", "final_answer": reply,
                "cached_slots": slots.cacheable(merged),
                "messages": [{"role": "assistant", "content": reply}]}

    _emit(_meta_event(ir, merged, [], "agent"))
    return {"merged_slots": merged, "still_missing": [],
            "route_kind": "agent", "cached_slots": slots.cacheable(merged)}


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
    _emit({"type": "done", "reply": reply, "guard": {}})
    return {"final_answer": reply,
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
    try:
        for delta in agents.generate_stream(agent, state["user_query"],
                                            slot_dict, history):
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
            text = agents.generate(agent, state["user_query"], slot_dict, history)
            _emit({"type": "replace", "text": text})
            parts = [text]
        except Exception as e2:
            print(f"[graph] Agent生成失败({e2})")
            text = "哎呀，我这边开小差了，能再说一遍吗？"
            _emit({"type": "replace", "text": text})
            parts = [text]

    draft = "".join(parts).strip()
    print(f"[graph] {agent['name']} 生成 {time.perf_counter()-t0:.1f}s")
    return {"draft_answer": draft}


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
                                   _history(state), strict=True)
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

    _emit({"type": "done", "reply": text,
           "guard": {"passed": verdict["passed"], "actions": verdict["actions"]}})
    return {"final_answer": text,
            "guard_info": {"passed": verdict["passed"],
                           "actions": verdict["actions"]},
            "messages": [{"role": "assistant", "content": text}]}


def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("call_intent_gateway", call_intent_gateway)
    builder.add_node("slot_merge_and_router", slot_merge_and_router)
    builder.add_node("risk_reply", risk_reply_node)
    builder.add_node("dispatch_agent", dispatch_agent_node)
    builder.add_node("output_guard", output_guard_node)

    builder.add_edge(START, "call_intent_gateway")
    builder.add_edge("call_intent_gateway", "slot_merge_and_router")
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
