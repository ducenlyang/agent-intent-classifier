"""本地交互演示：python -m intent_classifier.demo_run

命令:
  直接输入文本  —— 走三层流水线，打印每层决策
  /stats        —— 查看会话统计(各层命中数/平均延迟)
  /help         —— 帮助
  /quit         —— 退出
可选参数:
  --once "query"  单条识别后退出(脚本联调用)
  --eval          在测试集上跑全流水线并输出精度报告
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .config import DATA_DIR, LLM_MODEL, PrimaryIntent, SecondaryIntent
from .intent_node import IntentPipeline
from .llm_refiner import llm_refiner
from .schemas import IntentResult

INTENT_ZH = {
    PrimaryIntent.QUESTION_SUBJECT: "学科问题",
    PrimaryIntent.QUESTION_POLICY: "政策咨询",
    PrimaryIntent.REQUEST_STUDY_PLAN: "学习计划请求",
    PrimaryIntent.REQUEST_ERROR_ANALYSIS: "错题分析请求",
    PrimaryIntent.CHAT_EMOTION: "情感倾诉",
    PrimaryIntent.REFUSE_CHEAT: "作弊请求(拒绝)",
    PrimaryIntent.GENERAL_CHAT: "通用闲聊",
    PrimaryIntent.UNKNOWN: "无法识别",
}
SECONDARY_ZH = {s.value: s.name for s in SecondaryIntent}
LAYER_ZH = {
    "RULE": "第1层·规则引擎",
    "LLM_REFINE": "第2层·小模型 → 第3层·LLM精判",
    "LLM_FALLBACK": "第2层·小模型 → 第3层·启发式降级",
}


def _fmt_slots(r: IntentResult) -> str:
    parts = []
    s = r.slots
    for key, zh in [("subject", "学科"), ("grade", "年级"), ("topic", "主题"),
                    ("emotion", "情绪"), ("time_horizon", "时间")]:
        v = getattr(s, key)
        if v:
            parts.append(f"{zh}:{v}")
    if s.question_text:
        q = s.question_text if len(s.question_text) <= 24 else s.question_text[:24] + "…"
        parts.append(f"问题:{q}")
    if s.knowledge_points:
        parts.append(f"知识点:{'、'.join(s.knowledge_points)}")
    return "  ".join(parts) if parts else "—"


def print_result(r: IntentResult) -> None:
    print("┌" + "─" * 62)
    print(f"│ 🧭 一级意图 : {r.primary_intent.value} ({INTENT_ZH[r.primary_intent]})")
    if r.secondary_intent:
        print(f"│ 🔎 二级意图 : {r.secondary_intent.value}")
    print(f"│ 🎯 置信度  : {r.confidence:.2%}")
    print(f"│ ⚙️  处理层  : {LAYER_ZH[r.handled_by]}   耗时 {r.latency_ms}ms")
    print(f"│ 🧩 槽位    : {_fmt_slots(r)}")
    if r.missing_slots:
        print(f"│ ❓ 缺槽待问: {', '.join(r.missing_slots)}")
    if r.risk.cheat_risk or r.risk.psych_risk != "none":
        print(f"│ ⚠️  风险    : 作弊={r.risk.cheat_risk} "
              f"心理={r.risk.psych_risk} 命中词={r.risk.matched_keywords or '—'}")
    if r.reply:
        print(f"│ 💬 直接回复: {r.reply}")
    if r.reply_hint:
        print(f"│ 💬 回复提示: {r.reply_hint}")
    print(f"│ 📋 决策路径: {' → '.join(r.decision_trace)}")
    print("└" + "─" * 62)


def run_repl(pipeline: IntentPipeline) -> None:
    stats: Counter = Counter()
    lat_sum = defaultdict(int)
    print("=" * 64)
    print("  三层蒸馏意图识别 · 本地演示")
    print(f"  小模型权重 : {pipeline.small.source}")
    print(f"  第三层LLM  : {LLM_MODEL + ' (已启用)' if llm_refiner.available else '未配置Key → 启发式精判兜底'}")
    print("  输入内容回车识别；/help 帮助；/quit 退出")
    print("=" * 64)
    while True:
        try:
            query = input("\n🧑 你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not query:
            continue
        if query == "/quit":
            print("再见！")
            break
        if query == "/help":
            print("输入任意文本进行意图识别；/stats 查看统计；/quit 退出")
            continue
        if query == "/stats":
            total = sum(stats.values()) or 1
            print(f"  共处理 {total} 条，平均延迟 {sum(lat_sum.values())/total:.0f}ms")
            for layer, n in stats.most_common():
                print(f"  {LAYER_ZH[layer]}: {n} 条 ({n/total:.0%})，平均 {lat_sum[layer]/n:.0f}ms")
            continue
        r = pipeline.classify(query)
        stats[r.handled_by] += 1
        lat_sum[r.handled_by] += r.latency_ms
        print("🤖 意图 >")
        print_result(r)


def run_eval(pipeline: IntentPipeline, use_llm: bool) -> None:
    test_csv = DATA_DIR / "test.csv"
    if not test_csv.exists():
        print(f"未找到测试集 {test_csv}，请先运行 distill_train.gen_data")
        return
    rows = list(csv.DictReader(open(test_csv, encoding="utf-8-sig")))
    mode = ("LLM终审(逐条调用API，较慢)" if use_llm
            else "启发式降级(不消耗LLM配额，加 --llm 启用LLM评估)")
    print(f"\n全流水线评估: {len(rows)} 条测试样本 | 第三层模式: {mode}")
    print("-" * 64)
    correct = 0
    layer_hits: Counter = Counter()
    per_class = defaultdict(lambda: [0, 0])  # label -> [正确, 总数]
    for i, row in enumerate(rows, 1):
        r = pipeline.classify(row["text"])
        layer_hits[r.handled_by] += 1
        ok = r.primary_intent.value == row["label"]
        correct += ok
        per_class[row["label"]][1] += 1
        per_class[row["label"]][0] += ok
        if i % 50 == 0:
            print(f"  ... 已评估 {i}/{len(rows)}", flush=True)
    print(f"总体准确率: {correct}/{len(rows)} = {correct/len(rows):.2%}")
    print("\n各层命中分布:")
    for layer, n in layer_hits.most_common():
        print(f"  {LAYER_ZH[layer]:<22} {n:>4} 条 ({n/len(rows):.0%})")
    print("\n分类别准确率:")
    for label in sorted(per_class):
        c, t = per_class[label]
        zh = INTENT_ZH.get(PrimaryIntent(label), label)
        print(f"  {label:<26} {zh:<10} {c}/{t} = {c/t:.0%}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底
    args = sys.argv[1:]
    if "--eval" in args:
        # 评估默认强制启发式(快且不耗LLM配额)；--llm 显式启用LLM终审评估
        use_llm = "--llm" in args
        run_eval(IntentPipeline(use_llm=use_llm), use_llm=use_llm)
        return
    pipeline = IntentPipeline()
    if "--once" in args:
        q = args[args.index("--once") + 1]
        print_result(pipeline.classify(q))
        return
    run_repl(pipeline)


if __name__ == "__main__":
    main()
