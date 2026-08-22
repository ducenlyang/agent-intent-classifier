"""端到端对话演示：python -m intent_classifier.chat_demo

完整链路：意图三层流水线 → Router → 业务Agent(专属Prompt+槽位) → 生成大模型
        → 输出守卫 → 回复用户
命令: /debug 切换显示意图明细 | /stats 会话统计 | /new 重开会轮 | /quit 退出
"""
from __future__ import annotations

import sys
from collections import Counter

from .assistant import Assistant, AssistantTurn
from .config import GEN_MODEL, LLM_MODEL
from .demo_run import INTENT_ZH, print_result

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


def print_turn(t: AssistantTurn, debug: bool) -> None:
    r = t.intent
    print(f"🧭 {r.primary_intent.value}({INTENT_ZH[r.primary_intent]}) "
          f"{r.confidence:.0%} | 槽位: {fmt_slot_line(t)} | "
          f"路由: {ROUTE_ZH[t.route_kind]}"
          + (f"→{t.agent_name}" if t.agent_name else ""))
    if debug:
        print_result(r)
    if t.route_kind == "agent":
        print(f"⏳ {t.agent_name} 生成中({t.gen_model or GEN_MODEL})...")
        print(f"✅ 守卫{'通过' if t.guard.get('passed', True) else '拦截'}"
              + (f" | {t.guard.get('gen_ms', 0)}ms"
                 f"{' | ' + '; '.join(t.guard['actions']) if t.guard.get('actions') else ''}"))
    print(f"\n🤖 小助手 > {t.reply}\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    assistant = Assistant()
    debug = False
    turns = 0
    stats: Counter = Counter()
    print("=" * 64)
    print("  教育助手 · 端到端对话演示")
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
        t = assistant.chat(user)
        turns += 1
        stats[ROUTE_ZH[t.route_kind]] += 1
        print_turn(t, debug)


if __name__ == "__main__":
    main()
