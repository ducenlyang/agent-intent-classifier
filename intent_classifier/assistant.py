"""Assistant：端到端对话编排。

流程：意图三层流水线 → Router →(拦截|反问|Agent生成)→ 输出守卫 → 返回用户。
反问补槽的多轮处理：本轮反问后记录原 query，用户下一条消息视为补充信息，
与原 query 拼接后重新走完整识别（LLM 会合并抽取槽位），仅反问一次防死循环。
"""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from .agents import Agent
from .config import GEN_MODEL
from .guard import GuardVerdict, output_guard
from .intent_node import IntentPipeline
from .router import RouteDecision, route
from .schemas import IntentResult


class AssistantTurn(BaseModel):
    """一轮对话的完整记录（调试与统计用）。"""

    query: str
    intent: IntentResult
    route_kind: str                     # intercept / clarify / agent
    agent_name: str | None = None
    reply: str = Field(description="最终下发给用户的回复")
    guard: dict = Field(default_factory=dict)
    latency_ms: int = 0
    gen_model: str | None = None


class Assistant:
    def __init__(self, use_llm: bool | None = None):
        self.pipeline = IntentPipeline(use_llm=use_llm)
        self._pending: str | None = None  # 反问等待补充时的原 query

    # ------------------------------------------------------------------
    def chat(self, user_input: str) -> AssistantTurn:
        t0 = time.perf_counter()

        # 反问补槽轮：把补充信息并入原 query 重新识别
        if self._pending is not None:
            user_input = f"{self._pending}（补充：{user_input.strip()}）"
            self._pending = None

        result = self.pipeline.classify(user_input)
        decision: RouteDecision = route(result)
        gen_model = None
        guard_info: dict = {}

        if decision.kind == "intercept":
            reply = decision.reply
        elif decision.kind == "clarify":
            reply = decision.reply
            self._pending = user_input  # 记录，下轮合并；仅追问一次，下轮必放行
        else:
            reply, guard_info, gen_model = self._generate(decision.agent, result)

        return AssistantTurn(
            query=user_input,
            intent=result,
            route_kind=decision.kind,
            agent_name=decision.agent.name if decision.agent else None,
            reply=reply,
            guard=guard_info,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            gen_model=gen_model,
        )

    # ------------------------------------------------------------------
    def _generate(self, agent: Agent, result: IntentResult) -> tuple[str, dict, str]:
        """Agent 生成 + 输出守卫。生成失败降级为道歉+引导。"""
        try:
            text, gen_ms = agent.generate(result)
            gen_model = GEN_MODEL
        except Exception as e:
            print(f"[Assistant] Agent生成失败({e})，使用降级话术")
            text = (
                "哎呀，我这边网络开小差了，刚才没想好怎么回你。"
                "可以再发一次，或者先说说你的年级和科目，我们聊点学习上的事？"
            )
            gen_ms, gen_model = 0, None
        safe_text, verdict = output_guard(text, result)
        return safe_text, {
            "passed": verdict.passed,
            "actions": verdict.actions,
            "gen_ms": gen_ms,
        }, gen_model


if __name__ == "__main__":
    ast = Assistant()
    print(ast.chat("帮我讲一下二次函数顶点公式怎么用").reply[:200])
