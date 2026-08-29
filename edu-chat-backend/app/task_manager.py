"""Task-Manager 任务管理组件（错题辅导场景 · 独立新增，不侵入上游）。

职责边界：
- 上游（意图识别/实体检测/场景分发）保持原样，本组件只消费其输出；
- 业务联合决策层：依据【上游意图标签 + has_new_question_entity + 会话task_stack】
  推导 4 种调度动作（start/resume/continue/knowledge_qa），严格遵守决策表；
- 内存模拟热栈（每session一份，替代Redis）：active(ongoing)唯一 + suspended列表；
  超过 max_suspended_task=3 时最老的suspended沉降SQLite冷存储；ongoing永不沉降；
- 冷持久化SQLite：仅存被淘汰沉降的suspended任务与closed任务；ongoing不落库；
- 仅错题多步辅导生成 teaching_task；普通问答/闲聊/只求答案不建任务。

决策表（严格实现）：
| 上游意图标签                        | has_active & is_active_valid | has_new_question_entity | 调度动作               |
|------------------------------------|------------------------------|-------------------------|------------------------|
| start_exercise_tutor               | 任意                         | True                    | start_exercise_tutor   |
| resume_history_tutor               | 任意                         | False                   | resume_history_tutor   |
| knowledge_qa/simple_ask_answer/chat| True                         | False                   | continue_current_tutor |
| knowledge_qa/simple_ask_answer/chat| False                        | False                   | knowledge_qa           |
| knowledge_qa/simple_ask_answer/chat| True，但active失效            | False                   | knowledge_qa           |

注：has_new_question_entity 为客观实体信号，优先级最高——任何标签下命中均判
start_exercise_tutor（表外情形，按此原则闭合）。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .config import ROOT_DIR

# ---------------------------------------------------------------------------
# 常量与数据结构
# ---------------------------------------------------------------------------
ACTIVE_TIMEOUT_SECONDS = 90 * 60   # ongoing任务90分钟无交互即失效
MAX_SUSPENDED_TASK = 3             # 内存suspended上限，超出沉降最老一条
COLD_DB_PATH = ROOT_DIR / "data" / "tasks.db"

FSM_NAME = "tutoring_flow"
FSM_STATES = ["parse_question", "diagnose_error_cause", "difficult_analysis",
              "socratic_ask", "variant_practice", "finished"]


class UpstreamIntent(str, Enum):
    """上游意图标签（由适配层从网关 PrimaryIntent 映射而来，不改上游）。"""
    START_EXERCISE_TUTOR = "start_exercise_tutor"
    RESUME_HISTORY_TUTOR = "resume_history_tutor"
    KNOWLEDGE_QA = "knowledge_qa"
    SIMPLE_ASK_ANSWER = "simple_ask_answer"
    CHAT = "chat"
    OTHER = "other"


class Action(str, Enum):
    START = "start_exercise_tutor"
    RESUME = "resume_history_tutor"
    CONTINUE = "continue_current_tutor"
    KNOWLEDGE_QA = "knowledge_qa"


class QuestionMeta(BaseModel):
    question_id: str = ""
    subject: str = ""
    grade: str = ""
    question_text: str = ""
    student_wrong_answer: str = ""


class TaskSlots(BaseModel):
    error_cause: str = ""
    weak_knowledge: list[str] = Field(default_factory=list)


class TeachingTask(BaseModel):
    task_instance_id: str
    status: str = "ongoing"                      # ongoing | suspended | closed
    fsm_name: str = FSM_NAME
    fsm_state: str = FSM_STATES[0]
    question_meta: QuestionMeta = Field(default_factory=QuestionMeta)
    slots: TaskSlots = Field(default_factory=TaskSlots)
    last_interact_time: float = Field(default_factory=time.time)

    def advance(self) -> str:
        """FSM 单步推进（finished 为终态）。"""
        i = FSM_STATES.index(self.fsm_state)
        self.fsm_state = FSM_STATES[min(i + 1, len(FSM_STATES) - 1)]
        return self.fsm_state


class TaskStack(BaseModel):
    session_id: str
    active: TeachingTask | None = None
    suspended: list[TeachingTask] = Field(default_factory=list)
    max_suspended_task: int = MAX_SUSPENDED_TASK


class Decision(BaseModel):
    action: Action
    task: TeachingTask | None = None              # start/continue/resume 携带目标任务
    reason: str = ""                              # 决策依据（排查/轨迹用）
    evicted: str | None = None                    # 本次沉降到冷存储的 task_instance_id


# ---------------------------------------------------------------------------
# SQLite 冷存储（仅 suspended沉降 与 closed 任务）
# ---------------------------------------------------------------------------
class ColdStore:
    def __init__(self, db_path: Path = COLD_DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS cold_task (
                task_instance_id TEXT PRIMARY KEY,
                session_id       TEXT NOT NULL,
                status           TEXT NOT NULL,       -- suspended | closed
                fsm_name         TEXT, fsm_state      TEXT,
                question_meta    TEXT, slots           TEXT,
                last_interact_time REAL,
                cold_saved_at    TEXT
            )""")
        self._db.commit()

    def put(self, session_id: str, task: TeachingTask) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO cold_task VALUES (?,?,?,?,?,?,?,?,?)",
            (task.task_instance_id, session_id, task.status,
             task.fsm_name, task.fsm_state,
             task.question_meta.model_dump_json(), task.slots.model_dump_json(),
             task.last_interact_time, datetime.now().isoformat()))
        self._db.commit()

    def load(self, session_id: str, task_instance_id: str | None = None,
             status: str = "suspended") -> TeachingTask | None:
        """取出(并删除)冷存储任务；closed 默认不可恢复。"""
        if task_instance_id:
            row = self._db.execute(
                "SELECT * FROM cold_task WHERE session_id=? AND task_instance_id=? AND status=?",
                (session_id, task_instance_id, status)).fetchone()
        else:
            row = self._db.execute(
                "SELECT * FROM cold_task WHERE session_id=? AND status=? "
                "ORDER BY last_interact_time DESC LIMIT 1",
                (session_id, status)).fetchone()
        if not row:
            return None
        task = TeachingTask(
            task_instance_id=row[0], status=row[2], fsm_name=row[3], fsm_state=row[4],
            question_meta=QuestionMeta.model_validate_json(row[5]),
            slots=TaskSlots.model_validate_json(row[6]),
            last_interact_time=row[7])
        self._db.execute("DELETE FROM cold_task WHERE task_instance_id=?",
                         (task.task_instance_id,))
        self._db.commit()   # 移出冷存储 = 禁止重复加载同一task_instance_id
        return task

    def peek(self, session_id: str) -> list[dict]:
        return [dict(zip(("task_instance_id", "status", "fsm_state", "question_text"),
                         (r[0], r[2], r[4],
                          json.loads(r[5]).get("question_text", "")[:24])))
                for r in self._db.execute(
                    "SELECT * FROM cold_task WHERE session_id=?", (session_id,))]


# ---------------------------------------------------------------------------
# Task-Manager：业务联合决策层 + 内存热栈
# ---------------------------------------------------------------------------
class TaskManager:
    def __init__(self, cold: ColdStore | None = None):
        self._stacks: dict[str, TaskStack] = {}
        self._cold = cold or ColdStore()
        self._lock = threading.Lock()

    # ---------- 热栈访问 ----------
    def stack(self, session_id: str) -> TaskStack:
        if session_id not in self._stacks:
            self._stacks[session_id] = TaskStack(session_id=session_id)
        return self._stacks[session_id]

    # ---------- active 有效性 ----------
    def is_active_valid(self, session_id: str) -> bool:
        """校验active任务可用：ongoing、未超时(90min)、question_meta完整。"""
        t = self.stack(session_id).active
        if not t or t.status != "ongoing":
            return False
        if time.time() - t.last_interact_time > ACTIVE_TIMEOUT_SECONDS:
            return False
        if not t.question_meta.question_text:      # 元数据完整：必须有题目原文
            return False
        return True

    # ---------- 决策层（纯函数，不改状态） ----------
    def decide(self, session_id: str, intent: UpstreamIntent,
               has_new_question_entity: bool) -> Decision:
        valid = self.is_active_valid(session_id)
        # 客观实体信号优先级最高
        if has_new_question_entity:
            return Decision(action=Action.START,
                             reason=f"决策表:意图={intent.value}×新题目实体(True)→start")
        if intent is UpstreamIntent.RESUME_HISTORY_TUTOR:
            return Decision(action=Action.RESUME,
                             reason="决策表:resume_history_tutor×无新实体→resume")
        if valid:
            return Decision(action=Action.CONTINUE, task=self.stack(session_id).active,
                            reason="决策表:qa/chat×active有效×无新实体→continue")
        # 无active或active失效 → knowledge_qa（表尾两行；start标签但无实体的表外情形同此闭合）
        return Decision(action=Action.KNOWLEDGE_QA,
                         reason=f"决策表:无有效active(有效={valid})×无新实体→knowledge_qa")

    # ---------- 动作执行 ----------
    def execute(self, session_id: str, d: Decision,
                question_meta: QuestionMeta | None = None,
                resume_task_id: str | None = None) -> Decision:
        with self._lock:
            st = self.stack(session_id)
            if d.action is Action.START:
                return self._start(st, question_meta or QuestionMeta(), d)
            if d.action is Action.RESUME:
                return self._resume(st, d, resume_task_id)
            if d.action is Action.CONTINUE:
                # 强约束：continue 绝不修改 task_stack（含 last_interact_time），仅透传
                return d
            return d   # knowledge_qa：不走任务栈

    def _start(self, st: TaskStack, meta: QuestionMeta, d: Decision) -> Decision:
        self._suspend_active(st)
        task = TeachingTask(task_instance_id=uuid.uuid4().hex,
                            question_meta=meta, last_interact_time=time.time())
        st.active = task
        evicted = self._evict_if_over(st)
        d.task, d.evicted = task, evicted
        return d

    def _resume(self, st: TaskStack, d: Decision, task_id: str | None) -> Decision:
        # 去重0：目标即当前active(或未指定且无其他候选) → 直接复用，不重复进栈
        if st.active and (not task_id or st.active.task_instance_id == task_id) \
                and not [t for t in st.suspended if not task_id or t.task_instance_id == task_id]:
            d.task, d.reason = st.active, "resume命中当前active(去重)"
            return d
        # ① 内存suspended优先；② 找不到查SQLite冷存储；closed不可恢复(冷库只按suspended捞)
        task = next((t for t in st.suspended
                     if not task_id or t.task_instance_id == task_id), None)
        src = "memory"
        if task is None:
            task = self._cold.load(st.session_id, task_id, status="suspended")
            src = "sqlite"
        if task is None:
            d.action, d.task, d.reason = Action.KNOWLEDGE_QA, None, \
                "resume失败:无可恢复任务(内存/冷存储均无suspended)→降级knowledge_qa"
            return d
        st.suspended = [t for t in st.suspended if t.task_instance_id != task.task_instance_id]
        self._suspend_active(st)
        task.status, task.last_interact_time = "ongoing", time.time()
        st.active = task
        evicted = self._evict_if_over(st)   # 冷恢复净增一条suspended，同样受上限约束
        d.task, d.reason, d.evicted = task, d.reason + f"(自{src}恢复)", evicted
        return d

    def _suspend_active(self, st: TaskStack) -> None:
        if st.active and st.active.status == "ongoing":
            st.active.status = "suspended"
            st.suspended.append(st.active)
            st.active = None

    def _evict_if_over(self, st: TaskStack) -> str | None:
        """suspended超过上限：最老一条沉降SQLite并移出内存；ongoing永不沉降。"""
        if len(st.suspended) <= st.max_suspended_task:
            return None
        oldest = min(st.suspended, key=lambda t: t.last_interact_time)
        self._cold.put(st.session_id, oldest)
        st.suspended.remove(oldest)
        return oldest.task_instance_id

    # ---------- 任务完结 ----------
    def close(self, session_id: str, task_instance_id: str | None = None) -> bool:
        with self._lock:
            st = self.stack(session_id)
            task = st.active if (st.active and (not task_instance_id
                              or st.active.task_instance_id == task_instance_id)) else None
            if task is None:
                task = next((t for t in st.suspended
                             if t.task_instance_id == task_instance_id), None)
                if task:
                    st.suspended.remove(task)
            if task is None:
                return False
            task.status = "closed"
            self._cold.put(session_id, task)   # closed入冷存储，但不支持resume
            if st.active and st.active.task_instance_id == task.task_instance_id:
                st.active = None
            return True

    # ---------- 上游意图置信度不足时的LLM兜底 ----------
    def decide_via_llm(self, session_id: str, query: str, intent_confidence: float,
                       has_new_question_entity: bool,
                       confidence_threshold: float = 0.75) -> Decision:
        """上游置信度不足→组装上下文问LLM；实体信号为客观事实传入，禁止LLM猜测。"""
        if intent_confidence >= confidence_threshold:
            return Decision(action=Action.KNOWLEDGE_QA, task=None,
                            reason="置信度充足，无需LLM兜底(不应调用此路径)")
        from .llm_client import chat_completion
        st = self.stack(session_id)
        summary = {
            "active": (st.active.task_instance_id, st.active.fsm_state,
                       st.active.question_meta.question_text[:20]) if st.active else None,
            "suspended": [(t.task_instance_id, t.question_meta.question_text[:20])
                          for t in st.suspended],
            "active_valid": self.is_active_valid(session_id),
        }
        prompt = (
            "你是错题辅导的任务调度器。根据客观事实决定调度动作，只能输出以下JSON之一：\n"
            '{"action":"start_exercise_tutor"} 开始新的错题辅导(仅当用户明确要开始一道新题)\n'
            '{"action":"resume_history_tutor"} 恢复历史错题辅导(用户要求回到/继续之前某道题)\n'
            '{"action":"continue_current_tutor"} 继续当前辅导中的题(与当前题目相关的追问)\n'
            '{"action":"knowledge_qa"} 普通问答/闲聊，不涉及任务栈\n'
            f"客观事实(必须采信，禁止猜测)：本次请求是否携带新题目实体="
            f"{has_new_question_entity}；当前active任务={summary['active']}；"
            f"active有效={summary['active_valid']}；suspended={summary['suspended']}\n"
            f"用户输入：{query}\n只输出JSON。")
        content = chat_completion(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": query}], temperature=0.0, max_tokens=60)
        try:
            import re
            m = re.search(r'\{.*\}', content, re.S)
            action = Action(json.loads(m.group(0))["action"])
        except Exception:
            action = Action.KNOWLEDGE_QA   # 解析失败安全降级
        # 实体信号优先级高于LLM判断
        if has_new_question_entity:
            action = Action.START
        return Decision(action=action,
                        reason=f"LLM兜底决策(上游conf={intent_confidence})→{action.value}")

    # ---------- 观测 ----------
    def snapshot(self, session_id: str) -> dict:
        st = self.stack(session_id)
        return {
            "active": st.active.model_dump() if st.active else None,
            "active_valid": self.is_active_valid(session_id),
            "suspended": [t.model_dump() for t in st.suspended],
            "cold": self._cold.peek(session_id),
        }


# ---------------------------------------------------------------------------
# 上游适配层：网关 IntentResult → 本组件输入（不改上游任何代码）
# ---------------------------------------------------------------------------
_RESUME_PATTERNS = ("回到刚才", "继续刚才", "刚才那道题", "回到那道", "继续那道",
                    "切回", "接着讲那道", "刚才的题")
_ONLY_ANSWER_PATTERNS = ("直接告诉我答案", "只要答案", "答案是什么就行", "别引导")


def adapt(ir: dict) -> tuple[UpstreamIntent, bool]:
    """ir: 网关 /classify 返回的 IntentResult dict。返回(标签, 是否携带新题目实体)。"""
    pi = ir.get("primary_intent", "UNKNOWN")
    query = ir.get("query", "")
    slots = ir.get("slots") or {}
    # 客观实体信号：抽取到题目原文/题目ID 即视为携带新题（优先级最高）
    has_entity = bool(slots.get("question_text") or slots.get("question_id"))

    if any(p in query for p in _RESUME_PATTERNS) and not has_entity:
        return UpstreamIntent.RESUME_HISTORY_TUTOR, False
    if any(p in query for p in _ONLY_ANSWER_PATTERNS):
        return UpstreamIntent.SIMPLE_ASK_ANSWER, has_entity

    if pi in ("QUESTION_SUBJECT", "REQUEST_ERROR_ANALYSIS"):
        if has_entity:
            return UpstreamIntent.START_EXERCISE_TUTOR, True
        return UpstreamIntent.KNOWLEDGE_QA, False      # 概念问答，无题目实体
    if pi in ("GENERAL_CHAT", "CHAT_EMOTION"):
        return UpstreamIntent.CHAT, False
    if pi == "REFUSE_CHEAT":
        return UpstreamIntent.OTHER, False
    return UpstreamIntent.KNOWLEDGE_QA, has_entity     # 政策等按普通问答
