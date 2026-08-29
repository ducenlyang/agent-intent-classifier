# 一次性修测试+demo v2（用后即删）
import pathlib

p = pathlib.Path("edu-chat-backend/app/test_task_manager.py")
s = p.read_text(encoding="utf-8")

# ---- test3: 副本status是拷贝时点值 → 改查栈内真实状态 ----
old = '''assert t1.status == "suspended" and t2.status == "ongoing"
assert st.active.task_instance_id == t2.task_instance_id
assert len(st.suspended) == 1
ok("切题:题1挂起/题2成为active唯一ongoing")'''
new = '''assert st.active.task_instance_id == t2.task_instance_id
assert st.active.status == "ongoing"
assert len(st.suspended) == 1 and st.suspended[0].task_instance_id == t1.task_instance_id
assert st.suspended[0].status == "suspended"
ok("切题:题1挂起/题2成为active唯一ongoing")'''
assert old in s, "test3 not found"
s = s.replace(old, new)

# ---- test8: 整块替换（起止标记法） ----
start_marker = 'print("== 8. 冷库沉降/冷恢复/TTL ==")'
end_marker = 'print("== 9. LLM兜底'
i0 = s.index(start_marker)
i1 = s.index(end_marker)
new_block = '''print("== 8. 冷库沉降/冷恢复(文本去重)/TTL ==")
tm, sid = fresh(), "s9"
ids8 = [start(tm, sid, f"冷题{i}").task.task_instance_id for i in range(1, 6)]
start(tm, sid, "冷题6")                              # 第6题切入 → 冷题1(最老)沉降
cold_ids = [c["task_instance_id"] for c in tm._cold.peek(sid)]
assert cold_ids == [ids8[0]], cold_ids
ok("冷题1(最老)已沉降SQLite")
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

'''
s = s[:i0] + new_block + s[i1:]

# ---- test9: LLM实体覆盖断言补reason ----
old = '''    assert d.action is Action.START
    ok("实体信号优先级高于LLM判断→START(未执行,meta由调用方携带)")'''
new = '''    assert d.action is Action.START and "实体覆盖" in d.reason
    ok("实体信号优先级高于LLM判断→START(未执行,meta由调用方携带)")'''
if old in s:
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")

# ---- demo: advance 改走 manager API ----
p2 = pathlib.Path("demo_task_manager.py")
s2 = p2.read_text(encoding="utf-8")
s2 = s2.replace("d1.task.advance()  # parse_question → diagnose_error_cause",
                "tm.advance_active(SID)  # FSM推进: parse_question → diagnose_error_cause (manager内合法改栈)")
s2 = s2.replace("d2.task.advance()", "tm.advance_active(SID)")
p2.write_text(s2, encoding="utf-8")
print("测试与demo更新完成 v2")
