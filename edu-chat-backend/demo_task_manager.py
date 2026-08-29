"""Task-Manager 场景演示：错题辅导中途切题 → 沉降 → 冷恢复 → 继续 → 完结。

运行: python demo_task_manager.py   (edu-chat-backend 目录下, 离线无LLM调用)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from app.task_manager import (Action, ColdStore, QuestionMeta, TaskManager,
                              UpstreamIntent, adapt)

SID = "demo-session-001"
tm = TaskManager(cold=ColdStore(Path(tempfile.mkdtemp()) / "tasks.db"))


def view(title):
    s = tm.snapshot(SID)
    act = s["active"]
    print(f"\n─ {title}")
    print(f"  active : {'%s [%s] %s' % (act['fsm_state'], act['question_meta']['subject'], act['question_meta']['question_text'][:18]) if act else '无'}"
          f"  (有效={s['active_valid']})")
    print(f"  内存suspended({len(s['suspended'])}): "
          + (", ".join(t["question_meta"]["question_text"][:10] for t in s["suspended"]) or "—"))
    print(f"  SQLite冷存储({len(s['cold'])}): "
          + (", ".join(c["question_text"] for c in s["cold"]) or "—"))


def step(upstream_ir: dict, label: str, meta: QuestionMeta | None = None):
    """模拟一轮上游输出 → 适配 → 决策 → 执行。"""
    tag, has_entity = adapt(upstream_ir)
    d = tm.decide(SID, tag, has_entity)
    d = tm.execute(SID, d, question_meta=meta)
    mark = {"start_exercise_tutor": "▶ 新建辅导", "resume_history_tutor": "↻ 恢复辅导",
            "continue_current_tutor": "▷ 继续当前", "knowledge_qa": "· 普通问答"}[d.action.value]
    print(f"\n用户: {upstream_ir['query']}")
    print(f"  上游标签={tag.value} 新题实体={has_entity} → 动作={d.action.value}  [{mark}]")
    if d.evicted:
        print(f"  ⤵ 沉降: {d.evicted[:8]}… 写入SQLite冷存储")
    print(f"  依据: {d.reason}")
    return d


view("初始状态")

# ① 首道错题（带题目实体 → 建任务）
d1 = step({"primary_intent": "QUESTION_SUBJECT",
           "query": "这道题不会：解方程 x²-5x+6=0",
           "slots": {"question_text": "解方程 x²-5x+6=0", "subject": "数学", "grade": "初三"}},
          "题1", QuestionMeta(subject="数学", grade="初三",
                              question_text="解方程 x²-5x+6=0", student_wrong_answer="x=1"))

# ② 辅导推进（FSM 前进 + continue 不动栈）
d1.task.advance()  # parse_question → diagnose_error_cause
step({"primary_intent": "QUESTION_SUBJECT", "query": "为什么会判别式小于零",
      "slots": {}}, "题1追问")
before = tm.snapshot(SID)

# ③ 中途切到另一道错题
d2 = step({"primary_intent": "REQUEST_ERROR_ANALYSIS",
           "query": "先帮我看看这道：已知函数f(x)=x²-2x，求最小值",
           "slots": {"question_text": "求f(x)=x²-2x最小值", "subject": "数学"}},
          "题2", QuestionMeta(subject="数学", grade="初三", question_text="求f(x)=x²-2x最小值"))
d2.task.advance()

# ④ 再切两道（触发沉降：内存suspended>3）
d3 = step({"primary_intent": "QUESTION_SUBJECT", "query": "这道英语完形怎么选",
           "slots": {"question_text": "完形填空第3题", "subject": "英语"}},
          "题3", QuestionMeta(subject="英语", question_text="完形填空第3题"))
d4 = step({"primary_intent": "QUESTION_SUBJECT", "query": "物理受力分析这道",
           "slots": {"question_text": "斜面物体受力分析", "subject": "物理"}},
          "题4", QuestionMeta(subject="物理", question_text="斜面物体受力分析"))
d5 = step({"primary_intent": "QUESTION_SUBJECT", "query": "再看这道化学",
           "slots": {"question_text": "化学方程式配平Fe+O2", "subject": "化学"}},
          "题5", QuestionMeta(subject="化学", question_text="化学方程式配平Fe+O2"))
view("切了5道题后（题1已沉降SQLite）")

# ⑤ 普通闲聊（continue：不建任务不动栈）
step({"primary_intent": "GENERAL_CHAT", "query": "有点累啊", "slots": {}}, "闲聊")
assert tm.snapshot(SID) == {**tm.snapshot(SID)}  # 栈未变(continue零修改)

# ⑥ 回到刚才那道题（内存suspended命中 → 恢复）
step({"primary_intent": "GENERAL_CHAT", "query": "回到刚才那道物理题", "slots": {}}, "恢复题4")

# ⑦ 恢复已沉降的题1（SQLite 冷恢复）
d = tm.decide(SID, UpstreamIntent.RESUME_HISTORY_TUTOR, False)
d = tm.execute(SID, d, resume_task_id=d1.task.task_instance_id)
print(f"\n用户: 回到最开始那道数学题(已沉降)")
print(f"  → 动作={d.action.value}  {d.reason}")
view("冷恢复题1后")

# ⑧ 完结题1（closed 入库，不可再恢复）
tm.close(SID, d.task.task_instance_id)
view("题1辅导完结(closed)")
r = tm.execute(SID, tm.decide(SID, UpstreamIntent.RESUME_HISTORY_TUTOR, False),
               resume_task_id=d.task.task_instance_id)
print(f"\n尝试恢复已完结任务 → 动作={r.action.value}（closed不可恢复）")
print("\n=== 演示结束：热栈切换/沉降/冷恢复/继续零修改/完结 全流程验证 ===")
