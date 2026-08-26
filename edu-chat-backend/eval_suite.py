"""系统性评测：单轮网关识别质量 + 多轮DM状态机行为。

python eval_suite.py            # 全量
python eval_suite.py --single   # 仅单轮(快,不调LLM生成)
"""
from __future__ import annotations

import sys
import time

import requests

GW = "http://127.0.0.1:8601"
BACKEND = "http://127.0.0.1:8600"
P = {"http": None, "https": None}


# ---------- 单轮：网关意图识别（真实学生口语表达） ----------
SINGLE_CASES = [
    # (query, 期望意图)
    ("帮我看看这道二次函数的题怎么做", "QUESTION_SUBJECT"),
    ("这道题第二步看不懂", "QUESTION_SUBJECT"),
    ("勾股定理怎么用啊", "QUESTION_SUBJECT"),
    ("高考多少分能上清华", "QUESTION_POLICY"),
    ("中考报名要什么材料", "QUESTION_POLICY"),
    ("帮我安排一下寒假复习", "REQUEST_STUDY_PLAN"),
    ("还有60天高考了怎么规划", "REQUEST_STUDY_PLAN"),
    ("我这次月考砸了", "CHAT_EMOTION|REQUEST_ERROR_ANALYSIS"),  # 歧义标注：孤立看两者皆可
    ("数学总是粗心丢分怎么办", "REQUEST_ERROR_ANALYSIS"),
    ("我心态崩了学不进去", "CHAT_EMOTION"),
    ("最近失眠压力好大", "CHAT_EMOTION"),
    ("讲个笑话", "GENERAL_CHAT"),
    ("你是机器人吗", "GENERAL_CHAT"),
    ("考试答案发我", "REFUSE_CHEAT"),
    ("帮我代写作文", "REFUSE_CHEAT"),
    ("我不想活了", "CHAT_EMOTION"),  # 心理高危,应带 psych=high
    ("asdfghjkl", "UNKNOWN"),
    ("数序题帮我讲讲", "QUESTION_SUBJECT"),  # 错别字
    ("帮我解个题呗", "QUESTION_SUBJECT"),
    ("换你你会怎么选", "GENERAL_CHAT"),
]


def eval_single() -> list[str]:
    fails = []
    print("\n===== 单轮: 网关意图识别 =====")
    for q, want in SINGLE_CASES:
        d = requests.post(f"{GW}/classify", json={"query": q},
                          proxies=P, timeout=30).json()
        got = d["primary_intent"]
        ok = got in want.split("|")
        mark = "✅" if ok else "❌"
        print(f"  {mark} {q[:24]:<26} got={got:<22} want={want} conf={d['confidence']:.2f}")
        if not ok:
            fails.append(f"[单轮] {q}: got={got} want={want} conf={d['confidence']:.2f}")
    return fails


# ---------- 多轮: DM状态机场景 ----------
def scene(sid_prefix, name, turns):
    """turns: [(query, 断言名->期望值...)] 用dict断言: route/act/anchor_summary_contains/anchor_question_contains/intent"""
    fails = []
    sid = f"{sid_prefix}_{int(time.time())}"
    print(f"\n===== {name} =====")
    for i, (q, exp) in enumerate(turns, 1):
        try:
            d = requests.post(f"{BACKEND}/api/chat",
                              json={"session_id": sid, "query": q},
                              proxies=P, timeout=180).json()
        except Exception as e:
            print(f"  ❌ T{i} {q[:20]} 异常 {e}")
            fails.append(f"[{name}]T{i} {q}: 异常{e}")
            continue
        a = d.get("anchor") or {}
        got = {"route": d.get("route_kind"), "act": d.get("dialog_act"),
               "intent": d.get("intent", {}).get("primary_intent"),
               "anchor": a.get("focus_summary") or "",
               "anchor_q": a.get("focus_question") or "",
               "slots": d.get("slots") or {}, "reply": d.get("reply") or ""}
        errs = []
        for k, v in exp.items():
            g = got.get(k, "")
            if k in ("anchor", "anchor_q"):
                ok = v in g
            else:
                ok = g == v
            if not ok:
                errs.append(f"{k}={g!r}≠{v!r}")
        mark = "✅" if not errs else "❌"
        print(f"  {mark} T{i} {q[:24]:<26} route={got['route']:<8} act={got['act'] or '-':<13} 锚点={got['anchor'] or '无'}")
        if errs:
            print(f"        └─ {'; '.join(errs)}")
            fails.append(f"[{name}]T{i} {q}: {'; '.join(errs)}")
    return fails


def eval_multi() -> list[str]:
    fails = []
    fails += scene("s1", "场景1 出题→作答→判定", [
        ("来一道初二数学题", {"route": "agent", "anchor": "数学"}),
        ("x=3", {"act": "CONTINUE_CHAT"}),  # 作答应继承闲聊/答疑锚点继续
        ("为什么这步错了", {"act": "CONTINUE_CHAT"}),
    ])
    fails += scene("s2", "场景2 讲题→追问→对答案", [
        ("帮我讲讲：已知2x+6=18，求解x", {"route": "agent", "anchor": "数学"}),
        ("第一步为什么移项要变号", {"act": "CONTINUE_CHAT", "anchor_q": "2x+6=18"}),
        ("我算出x=6对吗", {"act": "CONTINUE_CHAT", "anchor_q": "2x+6=18"}),
    ])
    fails += scene("s3", "场景3 计划确认执行", [
        ("帮我定个学习计划", {"route": "clarify"}),
        ("高二", {"route": "clarify", "act": "SLOT_ANSWER"}),
        ("英语", {"route": "agent", "intent": "REQUEST_STUDY_PLAN"}),
        ("就按这个执行", {"act": "AFFIRM", "intent": "REQUEST_STUDY_PLAN"}),
    ])
    fails += scene("s4", "场景4 改口修正", [
        ("帮我制定高二数学计划", {"route": "agent", "anchor": "数学"}),
        ("不对，我是高三", {"anchor": "高三"}),  # 改口年级,锚点应更新
        ("换物理吧", {"anchor": "物理"}),
    ])
    fails += scene("s5", "场景5 情感倾诉不被任务锚定", [
        ("我最近压力好大", {"route": "agent", "intent": "CHAT_EMOTION"}),
        ("就是感觉喘不过气", {"act": "CONTINUE_CHAT", "intent": "CHAT_EMOTION"}),
        ("谢谢你，聊学习吧", {"intent": "GENERAL_CHAT"}),
    ])
    fails += scene("s6", "场景6 混合意图", [
        ("我数学物理都不好，先帮我看看物理",
         {"intent": "REQUEST_ERROR_ANALYSIS", "anchor": "物理"}),
    ])
    fails += scene("s7", "场景7 追问中换题", [
        ("帮我讲讲：解方程3x=12", {"route": "agent", "anchor_q": "3x=12"}),
        ("再来一道难的", {"act": "CONTINUE_CHAT"}),
        ("换英语题吧", {"anchor": "英语"}),
    ])
    fails += scene("s8", "场景8 口语省略", [
        ("物理错题太多了", {"route": "agent"}),
        ("就是大题", {"act": "CONTINUE_CHAT", "intent": "REQUEST_ERROR_ANALYSIS"}),
        ("嗯嗯", {"act": "AFFIRM"}),
    ])
    return fails


def main():
    fails: list[str] = []
    t0 = time.time()
    if "--single" not in sys.argv:
        pass
    fails += eval_single()
    if "--single" not in sys.argv:
        fails += eval_multi()
    print(f"\n========== 汇总 ==========")
    print(f"耗时 {time.time()-t0:.0f}s | 失败 {len(fails)} 项")
    for f in fails:
        print(" ·", f)


if __name__ == "__main__":
    main()
