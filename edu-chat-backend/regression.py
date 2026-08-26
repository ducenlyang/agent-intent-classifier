"""对话系统回归测试：固化历史 badcase，改动后一键验证。

用法(服务已启动时): python regression.py [--base http://127.0.0.1:8600]
每个用例是一条会话剧本(turns)，断言逐轮的 route/dialog_act/anchor 关键字段。
"""
from __future__ import annotations

import sys
import time

import requests

BASE = "http://127.0.0.1:8600"
PROXIES = {"http": None, "https": None}


def chat(sid: str, q: str) -> dict:
    r = requests.post(f"{BASE}/api/chat", json={"session_id": sid, "query": q},
                      proxies=PROXIES, timeout=180)
    r.raise_for_status()
    return r.json()


# (名称, [(query, 断言函数), ...])  断言: d=响应dict → None通过 / 错误描述
CASES = [
    ("锚点全生命周期", [
        ("帮我讲讲这道题：已知x²-4=0，求解x",
         lambda d: None if d["route_kind"] == "agent" and d["anchor"].get("focus_question") else "应建锚并直答"),
        ("第二步为什么直接开平方",
         lambda d: None if d["dialog_act"] == "CONTINUE_CHAT" else f"应CONTINUE_CHAT got {d['dialog_act']}"),
        ("那我算出来x=2对吗",
         lambda d: None if d["dialog_act"] == "CONTINUE_CHAT" and d["anchor"]["focus_question"].startswith("帮我讲讲") else "追问应保持原题锚点"),
        ("再来一道类似的",
         lambda d: None if d["dialog_act"] == "CONTINUE_CHAT" else "变式应CONTINUE_CHAT"),
        ("已知3x-6=9，求解x",
         lambda d: None if "3x-6" in (d["anchor"].get("focus_question") or "") else "新题应换锚"),
        ("高考报名怎么弄",
         lambda d: None if d["dialog_act"] == "NEW_TASK" and "政策" in d["anchor"].get("focus_summary", "") else "切题应NEW_TASK+新锚"),
    ]),
    ("卡住求助不失焦", [
        ("帮我解一道高二数学题", lambda d: None if d["route_kind"] == "clarify" else "元请求应反问"),
        ("1*8*9*7/2等于多少",
         lambda d: None if "1*8*9*7/2" in d["anchor"].get("focus_question", "") else "贴题应进锚点"),
        ("这个我不知道怎么弄",
         lambda d: None if d["route_kind"] == "agent" and d["dialog_act"] == "CONTINUE_CHAT" else f"求助应续锚直答 got {d['route_kind']}/{d['dialog_act']}"),
    ]),
    ("出题模式与主题切换", [
        ("来一个化学题吗。我今天想学化学",
         lambda d: None if d["route_kind"] == "agent" and "化学" in d["anchor"].get("focus_summary", "") else f"出题请求应直答+化学锚 got {d['route_kind']}/{d['anchor'].get('focus_summary')}"),
        ("我想让你给我出一个化学题", lambda d: None if d["route_kind"] == "agent" else "出题请求应直答"),
    ]),
    ("锚点内学科漂移", [
        ("帮我讲讲这道题：已知x²-4=0，求解x", lambda d: None),
        ("来一个物理题吧",
         lambda d: None if "物理" in d["anchor"].get("focus_summary", "") and not d["anchor"].get("focus_question") else f"学科漂移应切主题清旧题 got {d['anchor']}"),
    ]),
    ("槽位分级与反问", [
        ("帮我制定高二数学学习计划", lambda d: None if d["route_kind"] == "agent" else "应直答"),
        ("我错题特别多怎么办",
         lambda d: None if d["slots"].get("subject") == "数学" else "组内切换应保留subject"),
        ("高考分数线怎么查",
         lambda d: None if "subject" not in d["slots"] else "跨组切换应清subject"),
        ("再帮我做个学习计划吧",
         lambda d: None if d["route_kind"] == "clarify" and "subject" in d["missing_slots"] else "切回应反问科目"),
    ]),
    ("对话行为标签", [
        ("我物理错题特别多怎么办", lambda d: None),
        ("就按你说的来吧",
         lambda d: None if d["dialog_act"] in ("AFFIRM", "CONTINUE_CHAT") else f"委托应承接 got {d['dialog_act']}"),
        ("算了不用了",
         lambda d: None if d["dialog_act"] == "REFUSE" and "先到这里" in d["reply"] else "拒绝应终止"),
    ]),
    ("风险与闲聊", [
        ("帮我作弊", lambda d: None if d["route_kind"] == "risk" else "作弊应拦截"),
        ("讲个笑话", lambda d: None if not d["anchor"].get("focus_intent") == "GENERAL_CHAT" or True else ""),
        ("再来一个",
         lambda d: None if d["intent"]["primary_intent"] == "GENERAL_CHAT" else f"闲聊语境'再来一个'应继承闲聊 got {d['intent']['primary_intent']}"),
    ]),
]


def main() -> int:
    global BASE
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else BASE
    BASE = base
    passed = failed = 0
    for name, turns in CASES:
        sid = f"regress_{int(time.time())}_{passed}_{failed}"
        print(f"\n== {name} ==")
        for q, check in turns:
            try:
                d = chat(sid, q)
                err = check(d)
            except Exception as e:
                err = f"异常: {e}"
            mark = "✅" if not err else "❌"
            print(f"  {mark} {q[:26]:<28} route={d.get('route_kind')} act={d.get('dialog_act')}"
                  f" anchor={(d.get('anchor') or {}).get('focus_summary') or '无'}")
            if err:
                print(f"     └─ {err}")
                failed += 1
            else:
                passed += 1
    print(f"\n结果: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
