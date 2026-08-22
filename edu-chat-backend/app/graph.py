"""LangGraph 对话编排（方案版 5 节点工作流）：

START → ①call_intent_gateway(HTTP调网关,无状态单句识别)
      → ②slot_merge_and_router(后端独有: 缓存槽位合并+二次判缺+三路分支)
           ├─ A 风险 → ③risk_reply → END
           ├─ B 缺槽 → ②内生成反问话术 → END
           └─ C 齐全 → ④dispatch_agent(6大Agent生成) → ⑤output_guard → END

会话状态：MemorySaver 内存字典（Demo 不上 Redis/DB）。
槽位分工：抽取/单轮missing=网关；缓存/合并/反问/注入Prompt=本后端。
"""
from __future__ import annotations

import time
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import agents, gateway, slots
from .guard import SAFE_FALLBACK, guard
from .gateway import IntentResult, Slots


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

    if ir.primary_intent.value == "REFUSE_CHEAT" or ir.risk.cheat_risk:
        return {"merged_slots": merged, "still_missing": missing,
                "route_kind": "risk", "cached_slots": slots.cacheable(merged)}
    if ir.risk.psych_risk == "high" and ir.reply:
        return {"merged_slots": merged, "still_missing": missing,
                "route_kind": "risk", "cached_slots": slots.cacheable(merged)}

    if missing:  # 分支B：反问话术在本节点直接生成 → END
        questions = " ".join(
            slots.CLARIFY_QUESTIONS.get(f, f"请补充{f}") for f in missing
        )
        reply = f"好嘞，先确认一下：{questions}"
        return {"merged_slots": merged, "still_missing": missing,
                "route_kind": "clarify", "final_answer": reply,
                "cached_slots": slots.cacheable(merged),
                "messages": [{"role": "assistant", "content": reply}]}

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
    return {"final_answer": reply,
            "messages": [{"role": "assistant", "content": reply}]}


# ---------------------------------------------------------------------------
# 节点4：Agent 分发（合并后的完整槽位注入 Prompt，直接 LLM 生成，无RAG）
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
    t0 = time.perf_counter()
    try:
        draft = agents.generate(agent, state["user_query"], slot_dict,
                                _history(state))
    except Exception as e:
        print(f"[graph] Agent生成失败({e})")
        draft = "哎呀，我这边开小差了，能再说一遍吗？"
    print(f"[graph] {agent['name']} 生成 {time.perf_counter()-t0:.1f}s")
    return {"draft_answer": draft}


# ---------------------------------------------------------------------------
# 节点5：输出守卫（need_guide_only 防答疑泄答案 + 基础安全过滤）
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
