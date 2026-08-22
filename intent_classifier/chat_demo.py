"""端到端对话演示（流式输出）：python -m intent_classifier.chat_demo

完整链路：意图三层流水线 → Router → 业务Agent(专属Prompt+槽位) → 生成大模型
        (SSE流式逐字输出) → 输出守卫 → 回复用户
命令: /debug 切换显示意图明细 | /stats 会话统计 | /new 重开会轮 | /quit 退出
"""
from __future__ import annotations

import sys
from collections import Counter

from .assistant import Assistant, AssistantTurn
from .config import GEN_MODEL, LLM_MODEL
from .demo_run import INTENT_ZH, print_result
from .schemas import IntentResult

ROUTE_ZH = {"intercept": "拦截返回", "clarify": "反问补槽", "agent": "分发Agent"}


def fmt_slot_line(t: AssistantTurn) -> str:
    s = t.intent.slots
    parts = [f"{zh}:{getattr(s, k)}" for k, zh in
             [("subject", "学科"), ("grade", "年级"), ("topic", "主题"),
              ("emotion", "情绪"), ("time_horizon", "时间")]
             if getattr(s, k)]
    if s.question_text:
        parts.append(f"问题:{s.question_text[:16]}…")
    return "  ".join(parts) if parts else "—"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    assistant = Assistant()
    debug = False
    turns = 0
    stats: Counter = Counter()

    def on_route(result: IntentResult, decision) -> None:
        # 路由决策后立即打印意图头行；Agent 轮接着开打字机
        agent_name = decision.agent.name if decision.agent else None
        print(f"🧭 {result.primary_intent.value}({INTENT_ZH[result.primary_intent]}) "
              f"{result.confidence:.0%} | 路由: {ROUTE_ZH[decision.kind]}"
              + (f"→{agent_name}" if agent_name else ""))
        if debug:
            print_result(result)
        if decision.kind == "agent":
            print(f"⏳ {agent_name} 流式生成中({GEN_MODEL})...")
            print("\n🤖 小助手 > ", end="", flush=True)

    def on_delta(chunk: str) -> None:
        print(chunk, end="", flush=True)  # 打字机效果：逐块打印不换行

    print("=" * 64)
    print("  教育助手 · 端到端对话演示（流式输出）")
    print(f"  意图精判: {LLM_MODEL} | 答案生成: {GEN_MODEL}")
    print("  /debug 意图明细 /stats 统计 /new 重开 /quit 退出")
    print("=" * 64)
    while True:
        try:
            user = input("🧑 你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user:
            continue
        if user == "/quit":
            print("再见！")
            break
        if user == "/debug":
            debug = not debug
            print(f"意图明细: {'开' if debug else '关'}")
            continue
        if user == "/new":
            assistant = Assistant()
            turns = 0
            stats.clear()
            print("已重开新会话\n")
            continue
        if user == "/stats":
            print(f"  本会话 {turns} 轮，路由分布: {dict(stats) or '—'}")
            continue

        t = assistant.chat(user, on_delta=on_delta, on_route=on_route)
        turns += 1
        stats[ROUTE_ZH[t.route_kind]] += 1

        if t.route_kind != "agent":
            # 拦截/反问：无生成过程，整段返回
            print(f"🤖 小助手 > {t.reply}\n")
            continue
        # Agent 轮：流式正文已逐字打出，这里补守卫结论收尾
        g = t.guard
        print(f"\n✅ 守卫{'通过' if g.get('passed', True) else '拦截'}"
              f" · 生成 {g.get('gen_ms', 0)}ms · 全链路 {t.latency_ms}ms"
              + (f" | {'; '.join(g['actions'])}" if g.get("actions") else ""))
        print()


if __name__ == "__main__":
    main()
