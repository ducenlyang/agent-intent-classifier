"""Task-Manager 单元测试（无网络依赖，LLM 兜底走 mock）。

运行: python -m app.test_task_manager   (edu-chat-backend 目录下)
"""
from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path

from app.task_manager import (
    MAX_SUSPENDED_TASK,
    Action,
    ColdStore,
    QuestionMeta,
    TaskManager,
    TeachingTask,
    UpstreamIntent,
    adapt,
)

PASS = 0


def ok(name):
    global PASS
    PASS += 1
    print(f"  PASS: {name}")


def fresh() -> TaskManager:
    """每个用例独立的 TaskManager + 临时SQLite，避免相互污染。"""
    tmp = Path(tempfile.mkdtemp()) / "tasks.db"
    return TaskManager(cold=ColdStore(tmp))


def q(text: str, subject: str = "数学") -> QuestionMeta:
    return QuestionMeta(subject=subject, question_text=text)


def start(tm, sid, text):
    d = tm.decide(sid, UpstreamIntent.START_EXERCISE_TUTOR, True)
    return tm.execute(sid, d, question_meta=q(text))


# ---------------------------------------------------------------------------
print("== 1. 决策表全行 ==")
tm, sid = fresh(), "s1"
d = tm.decide(sid, UpstreamIntent.START_EXERCISE_TUTOR, True)
assert d.action is Action.START; ok("start×新实体(True)→start")
d = tm.decide(sid, UpstreamIntent.RESUME_HISTORY_TUTOR, False)
assert d.action is Action.RESUME; ok("resume×无实体→resume")
start(tm, sid, "x²-3x+2=0求x")
d = tm.decide(sid, UpstreamIntent.KNOWLEDGE_QA, False)
assert d.action is Action.CONTINUE and d.task is tm.stack(sid).active
ok("qa×active有效→continue(带task上下文)")
tm2, s2 = fresh(), "s2"
d = tm2.decide(s2, UpstreamIntent.CHAT, False)
assert d.action is Action.KNOWLEDGE_QA; ok("chat×无active→knowledge_qa")

print("== 2. active失效 → knowledge_qa（超时/元数据不完整） ==")
tm3, s3 = fresh(), "s3"
tm3.execute(s3, tm3.decide(s3, UpstreamIntent.START_EXERCISE_TUTOR, True), q("题目A"))
tm3.stack(s3).active.last_interact_time -= 91 * 60        # 模拟91分钟无交互
assert not tm3.is_active_valid(s3)
assert tm3.decide(s3, UpstreamIntent.CHAT, False).action is Action.KNOWLEDGE_QA
ok("超时90min→active失效→knowledge_qa")
tm4, s4 = fresh(), "s4"
tm4.execute(s4, tm4.decide(s4, UpstreamIntent.START_EXERCISE_TUTOR, True),
            QuestionMeta(subject="数学"))                   # 无question_text
assert not tm4.is_active_valid(s4)
assert tm4.decide(s4, UpstreamIntent.KNOWLEDGE_QA, False).action is Action.KNOWLEDGE_QA
ok("question_meta不完整→active无效→knowledge_qa")

print("== 3. start: 旧active转suspended，新任务ongoing，唯一 ==")
tm, sid = fresh(), "s5"
t1 = start(tm, sid, "题1: 解方程x-1=0").task
t2 = start(tm, sid, "题2: 解方程x-2=0").task
st = tm.stack(sid)
assert t1.status == "suspended" and t2.status == "ongoing"
assert st.active.task_instance_id == t2.task_instance_id
assert len(st.suspended) == 1
ok("切题:题1挂起/题2成为active唯一ongoing")

print("== 4. continue 零修改（含 last_interact_time） ==")
before = copy.deepcopy(tm.stack(sid).model_dump())
time.sleep(0.01)
tm.execute(sid, tm.decide(sid, UpstreamIntent.CHAT, False))
after = tm.stack(sid).model_dump()
assert before == after, "continue修改了task_stack!"
ok("continue完全不修改task_stack(深拷贝逐字段一致)")

print("== 5. 淘汰沉降: suspended>3 最老一条入SQLite，ongoing永不沉降 ==")
tm, sid = fresh(), "s6"
ids = [start(tm, sid, f"题{i}").task.task_instance_id for i in range(1, 6)]
st = tm.stack(sid)
assert len(st.suspended) == MAX_SUSPENDED_TASK, len(st.suspended)
assert ids[0] in [c["task_instance_id"] for c in ColdStore.__dict__ and tm._cold.peek(sid)]
ok(f"第5题切入后: 最老的题1沉降SQLite, 内存suspended={len(st.suspended)}")
assert st.active.task_instance_id == ids[4]
ok("ongoing(active题5)始终在内存，未沉降")
cold_ids = [c["task_instance_id"] for c in tm._cold.peek(sid)]
assert ids[0] in cold_ids and ids[1] not in cold_ids
ok("冷库恰存被淘汰的题1(题2~4仍在内存)")

print("== 6. task_instance_id 去重 ==")
tm, sid = fresh(), "s7"
t1 = start(tm, sid, "题1").task
dup = tm.execute(sid, tm.decide(sid, UpstreamIntent.RESUME_HISTORY_TUTOR, False),
                 resume_task_id=t1.task_instance_id)
assert dup.task.task_instance_id == t1.task_instance_id
assert len([x for x in tm.stack(sid).suspended
            if x.task_instance_id == t1.task_instance_id]) == 0
ok("resume当前active命中去重(不重复进栈)")

print("== 7. resume: 内存优先 → SQLite冷恢复；closed不可恢复 ==")
tm, sid = fresh(), "s8"
t1 = start(tm, sid, "题1").task
t2 = start(tm, sid, "题2").task                       # 题1→suspended(内存)
r = tm.execute(sid, tm.decide(sid, UpstreamIntent.RESUME_HISTORY_TUTOR, False))
assert r.task.task_instance_id == t1.task_instance_id and r.task.status == "ongoing"
assert t2.status == "suspended"
ok("resume自内存suspended: 题1恢复ongoing, 原active题2挂起")
t3 = start(tm, sid, "题3").task
t4 = start(tm, sid, "题4").task
t5 = start(tm, sid, "题5").task                       # 超限沉降"最老"一条
cold1 = [c["task_instance_id"] for c in tm._cold.peek(sid)]
# 题1被resume刷新过last_interact_time → 此刻最老的是题2，沉降的应为题2
assert cold1 == [t2.task_instance_id], cold1
ok("超限沉降按last_interact_time取最老(resume会刷新时间, 题2为最老)")
t6 = start(tm, sid, "题6").task                       # 占位切出
r2 = tm.execute(sid, tm.decide(sid, UpstreamIntent.RESUME_HISTORY_TUTOR, False),
                resume_task_id=t2.task_instance_id)
assert r2.task and r2.task.task_instance_id == t2.task_instance_id
assert "sqlite" in r2.reason
assert t2.task_instance_id not in [c["task_instance_id"] for c in tm._cold.peek(sid)]
ok("内存找不到→SQLite冷恢复, 且冷库移除(禁止重复加载)")
# 冷恢复净增suspended也必须受上限约束（回归bug：曾出现4条）
ids3 = [start(tm, sid, f"补充题{i}").task.task_instance_id for i in range(7, 11)]
while len(tm.stack(sid).suspended) < 3 or tm.stack(sid).active is None:
    start(tm, sid, "垫题")
evicted_cold = tm._cold.peek(sid)
target = evicted_cold[0]["task_instance_id"]
before_len = len(tm.stack(sid).suspended)
tm.execute(sid, tm.decide(sid, UpstreamIntent.RESUME_HISTORY_TUTOR, False),
           resume_task_id=target)
assert len(tm.stack(sid).suspended) <= MAX_SUSPENDED_TASK, \
    f"冷恢复后suspended={len(tm.stack(sid).suspended)}超限!"
ok("冷恢复触发淘汰: suspended始终≤3")

print("== 8. close: 入SQLite且不可resume ==")
tm, sid = fresh(), "s9"
t = start(tm, sid, "题1").task
assert tm.close(sid, t.task_instance_id)
assert tm.stack(sid).active is None
closed_in_cold = any(c["status"] == "closed" for c in tm._cold.peek(sid))
assert closed_in_cold
r = tm.execute(sid, tm.decide(sid, UpstreamIntent.RESUME_HISTORY_TUTOR, False),
               resume_task_id=t.task_instance_id)
assert r.action is not Action.RESUME or r.task is None
ok("close→active清空, closed入冷库, resume被拒(closed仅按suspended捞取)")

print("== 9. LLM兜底: 低置信触发, 实体信号优先级高于LLM ==")
tm, sid = fresh(), "s10"
import app.task_manager as tmmod
calls = {}
def fake_llm(messages, **kw):
    calls["prompt"] = messages[0]["content"]
    return '{"action":"resume_history_tutor"}'
orig = None
try:
    from app import task_manager as _tm
    import app.llm_client as _lc
    orig = _lc.chat_completion
    _lc.chat_completion = fake_llm
    tmmod_llm_patched = True
except Exception:
    tmmod_llm_patched = False
# 打桩方式2：直接替换模块内引用
import app.llm_client as lc
lc.chat_completion = fake_llm
d = tm.decide_via_llm(sid, "帮我看看这道题", intent_confidence=0.4,
                      has_new_question_entity=True)
assert d.action is Action.START, "实体=True必须压过LLM的resume判断"
assert "禁止猜测" in calls["prompt"] and "True" in calls["prompt"]
ok("LLM兜底: prompt携带实体客观事实, 实体True覆盖LLM输出→start")
d2 = tm.decide_via_llm(sid, "随便聊聊", intent_confidence=0.4, has_new_question_entity=False)
assert d2.action is Action.RESUME  # fake_llm输出
ok("LLM兜底: 无实体时采纳LLM动作枚举")
assert tm.decide_via_llm(sid, "x", 0.9, False).reason.startswith("置信度充足")
ok("置信度充足时不调LLM")
if orig is not None:
    lc.chat_completion = orig

print("== 10. FSM 推进 ==")
task = TeachingTask(task_instance_id="t")
seq = []
for _ in range(7):
    seq.append(task.advance())
assert task.fsm_state == "finished"
ok(f"tutoring_flow FSM: 6态顺序推进至终态finished({len(set(seq))}次转移)")

print("== 11. 上游适配层(不改网关) ==")
ir = {"primary_intent": "QUESTION_SUBJECT", "query": "帮我解一道高二数学题",
      "slots": {"question_text": "已知x²-4=0求x", "subject": "数学", "grade": "高二"}}
label, ent = adapt(ir)
assert label is UpstreamIntent.START_EXERCISE_TUTOR and ent is True
ok("带题目实体→start_exercise_tutor+实体True")
ir2 = {"primary_intent": "QUESTION_SUBJECT", "query": "勾股定理是什么", "slots": {}}
label2, ent2 = adapt(ir2)
assert label2 is UpstreamIntent.KNOWLEDGE_QA and not ent2
ok("概念问答无实体→knowledge_qa(不建任务)")
ir3 = {"primary_intent": "GENERAL_CHAT", "query": "回到刚才那道题", "slots": {}}
label3, _ = adapt(ir3)
assert label3 is UpstreamIntent.RESUME_HISTORY_TUTOR
ok("回到刚才那道题→resume_history_tutor")
ir4 = {"primary_intent": "GENERAL_CHAT", "query": "直接告诉我答案", "slots": {}}
label4, _ = adapt(ir4)
assert label4 is UpstreamIntent.SIMPLE_ASK_ANSWER
ok("只求答案→simple_ask_answer(不建任务)")

print(f"\n=== 全部 {PASS} 项断言通过 ===")
