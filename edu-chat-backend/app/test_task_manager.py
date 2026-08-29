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
assert d.action is Action.CONTINUE
assert d.task.task_instance_id == tm.stack(sid).active.task_instance_id
ok("qa×active有效→continue(带task上下文)")
# continue 返回深拷贝：改副本不影响栈内任务
d.task.fsm_state = "finished"
assert tm.stack(sid).active.fsm_state != "finished"
ok("continue返回深拷贝(下游改副本不穿透)")

tm2, s2 = fresh(), "s2"
d = tm2.decide(s2, UpstreamIntent.CHAT, False)
assert d.action is Action.KNOWLEDGE_QA; ok("chat×无active→knowledge_qa")

print("== 2. active失效 → knowledge_qa（超时/元数据不完整） ==")
tm3, s3 = fresh(), "s3"
tm3.execute(s3, tm3.decide(s3, UpstreamIntent.START_EXERCISE_TUTOR, True), q("题目A"))
tm3._activity[s3] -= 91 * 60        # 模拟91分钟无交互(manager活动钟)
assert not tm3.is_active_valid(s3)
assert tm3.decide(s3, UpstreamIntent.CHAT, False).action is Action.KNOWLEDGE_QA
ok("超时90min(单调活动钟)→active失效→knowledge_qa")
# 复审#2语义: 无关问答(knowledge_qa)不得刷新活动钟复活过期任务
tm3.execute(s3, tm3.decide(s3, UpstreamIntent.KNOWLEDGE_QA, False))
assert not tm3.is_active_valid(s3), "无关问答不应复活过期任务"
ok("knowledge_qa不刷新活动钟(过期任务不被无关问答复活)")
# 边界内继续任务 → continue刷新活动钟 → 重新有效(#7核心修复写实断言)
tm3._activity[s3] = time.monotonic() - 89 * 60      # 89分钟: 仍在窗口内
d = tm3.decide(s3, UpstreamIntent.CHAT, False)
assert d.action is Action.CONTINUE
before = tm3._activity[s3]
tm3.execute(s3, d)
assert tm3._activity[s3] > before and tm3.is_active_valid(s3)
ok("窗口内continue刷新活动钟→长辅导不误失效(写实断言)")
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
assert st.active.task_instance_id == t2.task_instance_id
assert st.active.status == "ongoing"
assert len(st.suspended) == 1 and st.suspended[0].task_instance_id == t1.task_instance_id
assert st.suspended[0].status == "suspended"
ok("切题:题1挂起/题2成为active唯一ongoing")

print("== 4. continue 零修改（含 last_interact_time 与活动钟以外全部字段） ==")
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
cold_ids = [c["task_instance_id"] for c in tm._cold.peek(sid)]
assert ids[0] in cold_ids
ok(f"第5题切入后: 最老的题1沉降SQLite, 内存suspended={len(st.suspended)}")
assert st.active.task_instance_id == ids[4]
ok("ongoing(active题5)始终在内存，未沉降")

print("== 6. start 去重: 同题面复用现有任务 ==")
tm, sid = fresh(), "s7"
t1 = start(tm, sid, "题1: 解方程x-1=0").task
t2 = start(tm, sid, "题2: 解方程x-2=0").task
d = start(tm, sid, "题1:解方程x-1=0")          # 同题(空白差异规范化后同指纹)
assert d.task.task_instance_id == t1.task_instance_id
st = tm.stack(sid)
assert st.active.task_instance_id == t1.task_instance_id
assert sum(1 for t in st.suspended if t.fingerprint == t1.fingerprint) == 0
assert st.active.fsm_state == t1.fsm_state
ok("同题面再次start→复用t1并提回active, 不产生重复任务")
d2 = start(tm, sid, "完全不同的一道几何证明题").task
assert d2.task_instance_id not in (t1.task_instance_id, t2.task_instance_id)
ok("不同题面正常新建")

print("== 7. resume: 未指定id取最近挂起(LIFO)；指定id按id；冷库最近优先 ==")
tm, sid = fresh(), "s8"
t1 = start(tm, sid, "题1").task
time.sleep(0.01)
t2 = start(tm, sid, "题2").task                     # 题1→suspended(早), 题2=active
t3 = start(tm, sid, "题3").task                     # 题2→suspended(晚), 题3=active
r = tm.execute(sid, tm.decide(sid, UpstreamIntent.RESUME_HISTORY_TUTOR, False))
assert r.task.task_instance_id == t2.task_instance_id, "未指定id应恢复最近挂起的题2"
st = tm.stack(sid)
assert st.active.task_instance_id == t2.task_instance_id
assert [t.task_instance_id for t in st.suspended] == [t1.task_instance_id, t3.task_instance_id]
assert st.suspended[-1].status == "suspended"
ok("resume未指定id→最近挂起(LIFO):恢复题2, 当前题3挂起(栈内已suspended)")
t4 = start(tm, sid, "题4").task
r2 = tm.execute(sid, tm.decide(sid, UpstreamIntent.RESUME_HISTORY_TUTOR, False),
                resume_task_id=t1.task_instance_id)
assert r2.task.task_instance_id == t1.task_instance_id
ok("resume指定id→按id恢复(题1)")

print("== 8. 冷库沉降/冷恢复(文本去重)/TTL ==")
tm, sid = fresh(), "s9"
ids8 = [start(tm, sid, f"冷题{i}").task.task_instance_id for i in range(1, 6)]
start(tm, sid, "冷题6")                              # 第6题切入 → 冷题1(最老)沉降
cold_ids = [c["task_instance_id"] for c in tm._cold.peek(sid)]
assert ids8[0] in cold_ids and ids8[1] in cold_ids and len(cold_ids) == 2, cold_ids
ok("两次超限淘汰: 冷题1、冷题2(最老)依次沉降SQLite")
d = start(tm, sid, "冷题1")                          # 同题面再次start
assert d.task.task_instance_id == ids8[0] and "冷库" in d.reason
assert tm.stack(sid).active.task_instance_id == ids8[0]
ok("同题面在冷库→start冷恢复复用, 不新建")
assert all(c["task_instance_id"] != ids8[0] for c in tm._cold.peek(sid))
ok("冷恢复后冷库记录移除(禁止重复加载)")
# TTL: 伪造过期记录 → purge
tm._cold.put(sid, TeachingTask(task_instance_id="expired", status="suspended",
                               question_meta=q("过期题")))
from datetime import datetime, timedelta
tm._cold._db.execute("UPDATE cold_task SET cold_saved_at=? WHERE task_instance_id='expired'",
                     ((datetime.now() - timedelta(days=8)).isoformat(),))
tm._cold._db.commit()
n = tm._cold.purge_expired()
assert n >= 1 and all(c["task_instance_id"] != "expired" for c in tm._cold.peek(sid))
ok(f"TTL清理: purge_expired删除过期记录({n}条)")
tm2, s10 = fresh(), "s10"
assert tm2._cold.purge_expired() == 0
ok("TTL: 无过期记录时purge为空操作")

print("== 9. LLM兜底: 二选一相关性, 禁止LLM输出start/resume ==")
tm, sid = fresh(), "s11"
t = start(tm, sid, "解方程x²-5x+6=0，我算成x=1").task
import app.llm_client as lc
orig = lc.chat_completion
try:
    lc.chat_completion = lambda msgs, **kw: '{"relevant": true}'
    d = tm.decide_via_llm(sid, "第二步还是不明白", intent_confidence=0.4,
                          has_new_question_entity=False)
    assert d.action is Action.CONTINUE and d.task.task_instance_id == t.task_instance_id
    ok("LLM兜底: relevant=true→continue(携带active深拷贝)")
    lc.chat_completion = lambda msgs, **kw: '{"relevant": false}'
    d = tm.decide_via_llm(sid, "今天天气不错", intent_confidence=0.4,
                          has_new_question_entity=False)
    assert d.action is Action.KNOWLEDGE_QA and d.task is None
    ok("LLM兜底: relevant=false→knowledge_qa(不动栈)")
    lc.chat_completion = lambda msgs, **kw: '垃圾输出非JSON'
    d = tm.decide_via_llm(sid, "嗯嗯", intent_confidence=0.4,
                          has_new_question_entity=False)
    assert d.action is Action.KNOWLEDGE_QA
    ok("LLM兜底: 解析失败安全降级knowledge_qa")
    lc.chat_completion = lambda msgs, **kw: '{"relevant": true}'
    d = tm.decide_via_llm(sid, "新题", intent_confidence=0.4,
                          has_new_question_entity=True)
    assert d.action is Action.START and "实体覆盖" in d.reason
    ok("实体信号优先级高于LLM判断→START(未执行,meta由调用方携带)")
    assert tm.decide_via_llm(sid, "x", 0.9, False).reason.startswith("置信度充足")
    ok("置信度充足时不调LLM")
finally:
    lc.chat_completion = orig

print("== 9.5 START空meta降级 + advance门控 + relevant严格布尔 ==")
tm95, s95 = fresh(), "s95"
d = tm95.dispatch(s95, UpstreamIntent.START_EXERCISE_TUTOR, True,
                  question_meta=QuestionMeta())      # 实体True但没带题面
assert d.action is Action.KNOWLEDGE_QA and "题面" in d.reason
assert tm95.stack(s95).active is None
ok("START空meta→降级knowledge_qa且不动栈(无僵尸任务)")
d2 = tm95.dispatch(s95, UpstreamIntent.START_EXERCISE_TUTOR, True, question_meta=q("真题面"))
assert d2.action is Action.START and tm95.stack(s95).active is not None
ok("正常meta→START照常")
tm95._activity[s95] -= 91 * 60                       # 过期
assert tm95.advance_active(s95) is None
ok("advance门控: 过期active拒绝推进")
# relevant严格布尔: 字符串"false"必须按无关处理
import app.llm_client as lc2
orig2 = lc2.chat_completion
try:
    lc2.chat_completion = lambda msgs, **kw: '{"relevant": "false"}'
    tmx, sx = fresh(), "sx"
    start(tmx, sx, "锚定题目")                        # active保持【有效】→ 锁死严格布尔修复
    d = tmx.decide_via_llm(sx, "无关输入", 0.4, False)
    assert d.action is Action.KNOWLEDGE_QA, d
    ok('relevant="false"(字符串)按无关处理(严格布尔)')
finally:
    lc2.chat_completion = orig2

print("== 9.7 ColdStore put→purge 死锁回归(第50次写入) ==")
import threading as _th
tmp2 = Path(tempfile.mkdtemp()) / "deadlock.db"
store = ColdStore(tmp2)
store._puts_since_purge = 49
_t = _th.Thread(target=lambda: store.put(
    "sDead", TeachingTask(task_instance_id="trigger", question_meta=q("触发purge"))))
_t.start(); _t.join(timeout=5)
assert not _t.is_alive(), "DEADLOCK: put()第50次触发purge死锁"
assert store._puts_since_purge == 0
ok("put→purge 无死锁(RLock可重入), 计数器归零")

print("== 10. FSM 推进: manager内合法改栈 ==")
tm, sid = fresh(), "s12"
start(tm, sid, "题1").task
st0 = tm.stack(sid).active.fsm_state
new_state = tm.advance_active(sid)
assert new_state != st0 and tm.stack(sid).active.fsm_state == new_state
ok(f"advance_active: {st0}→{new_state}(栈内真实推进)")
# finished后不再推进
for _ in range(6):
    tm.advance_active(sid)
assert tm.advance_active(sid) == "finished"
ok("FSM终态finished稳定")
assert tm.advance_active("无任务会话") is None
ok("无active时advance_active返回None")

print("== 11. 会话栈LRU上限: 逐出会话全量沉降 ==")
tm = fresh()
for i in range(tm._max_sessions + 3):
    tm.stack(f"sess{i}")
assert len(tm._stacks) <= tm._max_sessions
ok(f"会话栈上限{tm._max_sessions}: 超出LRU逐出(当前{len(tm._stacks)})")

print("== 12. dispatch: decide+execute同锁入口 ==")
tm, sid = fresh(), "s13"
d = tm.dispatch(sid, UpstreamIntent.START_EXERCISE_TUTOR, True, question_meta=q("题A"))
assert d.task and tm.stack(sid).active.task_instance_id == d.task.task_instance_id
d = tm.dispatch(sid, UpstreamIntent.KNOWLEDGE_QA, False)
assert d.action is Action.CONTINUE
ok("dispatch锁定入口工作正常")

print("== 13. 上游适配层(不改网关) ==")
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
assert adapt(ir3)[0] is UpstreamIntent.RESUME_HISTORY_TUTOR
ok("回到刚才那道题→resume_history_tutor")
ir4 = {"primary_intent": "GENERAL_CHAT", "query": "直接告诉我答案", "slots": {}}
assert adapt(ir4)[0] is UpstreamIntent.SIMPLE_ASK_ANSWER
ok("只求答案→simple_ask_answer(不建任务)")

print(f"\n=== 全部 {PASS} 项断言通过 ===")
