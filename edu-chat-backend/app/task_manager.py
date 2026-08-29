"""Task-Manager 任务管理组件（错题辅导场景 · 独立新增，不侵入上游）。

职责边界：
- 上游（意图识别/实体检测/场景分发）保持原样，本组件只消费其输出；
- 业务联合决策层：依据【上游意图标签 + has_new_question_entity + 会话task_stack】
  推导 4 种调度动作（start/resume/continue/knowledge_qa），严格遵守决策表；
- 内存模拟热栈（每session一份，替代Redis）：active(ongoing)唯一 + suspended列表；
  超过 max_suspended_task=3 时最老的suspended沉降SQLite冷存储；ongoing永不沉降；
- 冷持久化SQLite：仅存被淘汰沉降的suspended任务与closed任务，TTL 7天自动清理；
- 仅错题多步辅导生成 teaching_task；普通问答/闲聊/只求答案不建任务。

并发与健壮性约定：
- 运行时入口请使用 dispatch()（decide+execute 同锁，避免 TOCTOU）；
  decide()/execute() 保留为公开的纯函数/加锁对，供测试与演示分步调用；
- 90 分钟活动超时基于 manager 侧单调钟活动字段（execute 任何动作都会刷新），
  与「continue 不修改 task_stack」契约解耦——长会话追问不会误失效；
- continue/resume/start 返回的 task 均为深拷贝，下游不得也无法借此改栈；
  FSM 推进请走 advance_active()（manager 内改栈的唯一合法途径）；
- 会话栈数量有上限（LRU 淘汰，逐出前 active/suspended 全部沉降冷存储）。

接入待办（评审 #2/#8/#14，需先设计与 semantic_memory 双轨消解，本期不接入运行时）：
- 挂载点建议 slot_merge_and_router 之后、dispatch_agent 之前；
- 实体信号应统一取 merge 后槽位；AI 出题轮的建栈/锚定策略待定；
- FSM 推进与 close 的运行时触发者待定（当前由调用方显式调用）。
"""
from __future__ import annotations

import json
import re
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
ACTIVE_TIMEOUT_SECONDS = 90 * 60   # ongoing任务90分钟无交互即失效(基于manager活动钟)
MAX_SUSPENDED_TASK = 3             # 内存suspended上限，超出沉降最老一条
MAX_SESSIONS = 512                 # 内存会话栈上限(LRU逐出，逐出前全量沉降)
COLD_TTL_SECONDS = 7 * 24 * 3600   # 冷存储7天自动清理
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

    @property
    def fingerprint(self) -> str:
        """题面规范化指纹：用于 start 去重（同题不重复入栈）。"""
        return re.sub(r"\s+", "", self.question_meta.question_text)

    def advance(self) -> str:
        """FSM 单步推进（finished 为终态）。仅 TaskManager.advance_active 调用。"""
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
    task: TeachingTask | None = None              # start/continue/resume 携带目标任务(深拷贝)
    reason: str = ""                              # 决策依据（排查/轨迹用）
    evicted: str | None = None                    # 本次沉降到冷存储的 task_instance_id


# ---------------------------------------------------------------------------
# SQLite 冷存储（仅 suspended沉降 与 closed 任务；读写共用一把锁）
# ---------------------------------------------------------------------------
class ColdStore:
    def __init__(self, db_path: Path = COLD_DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._puts_since_purge = 0
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
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cold_session_status "
            "ON cold_task(session_id, status)")
        self._db.commit()

    def put(self, session_id: str, task: TeachingTask) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cold_task VALUES (?,?,?,?,?,?,?,?,?)",
                (task.task_instance_id, session_id, task.status,
                 task.fsm_name, task.fsm_state,
                 task.question_meta.model_dump_json(), task.slots.model_dump_json(),
                 task.last_interact_time, datetime.now().isoformat()))
            self._db.commit()
            self._puts_since_purge += 1
            if self._puts_since_purge >= 50:   # 长驻进程低频顺带清理过期行
                self._puts_since_purge = 0
                self.purge_expired()

    def load(self, session_id: str, task_instance_id: str | None = None,
             status: str = "suspended") -> TeachingTask | None:
        """取出(并删除)冷存储任务；closed 默认不可恢复——移出冷库即禁止重复加载。"""
        with self._lock:
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
            self._db.execute("DELETE FROM cold_task WHERE task_instance_id=?",
                             (row[0],))
            self._db.commit()
        return TeachingTask(
            task_instance_id=row[0], status=row[2], fsm_name=row[3], fsm_state=row[4],
            question_meta=QuestionMeta.model_validate_json(row[5]),
            slots=TaskSlots.model_validate_json(row[6]),
            last_interact_time=row[7])

    def peek(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM cold_task WHERE session_id=?", (session_id,)).fetchall()
        return [dict(zip(("task_instance_id", "status", "fsm_state", "question_text"),
                         (r[0], r[2], r[4],
                          json.loads(r[5]).get("question_text", "")[:24])))
                for r in rows]

    def purge_expired(self, ttl_seconds: float = COLD_TTL_SECONDS) -> int:
        """删除冷存储中超过 TTL 的记录，返回删除条数。"""
        cutoff = (datetime.now().timestamp() - ttl_seconds)
        with self._lock:
            cur = self._db.execute("DELETE FROM cold_task WHERE cold_saved_at < ?",
                                   (datetime.fromtimestamp(cutoff).isoformat(),))
            self._db.commit()
            return cur.rowcount

    def find_suspended_by_text(self, session_id: str, fingerprint: str) -> TeachingTask | None:
        """按题面指纹查找冷库中的suspended任务(找到即取出，语义同load)。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM cold_task WHERE session_id=? AND status='suspended'",
                (session_id,)).fetchall()
        for r in rows:
            task = TeachingTask(
                task_instance_id=r[0], status=r[2], fsm_name=r[3], fsm_state=r[4],
                question_meta=QuestionMeta.model_validate_json(r[5]),
                slots=TaskSlots.model_validate_json(r[6]),
                last_interact_time=r[7])
            if task.fingerprint == fingerprint:
                self.load(session_id, task.task_instance_id)
                return task
        return None


# ---------------------------------------------------------------------------
# Task-Manager：业务联合决策层 + 内存热栈
# ---------------------------------------------------------------------------
class TaskManager:
    def __init__(self, cold: ColdStore | None = None,
                 max_sessions: int = MAX_SESSIONS):
        self._stacks: dict[str, TaskStack] = {}
        self._activity: dict[str, float] = {}   # 会话活动钟(time.monotonic)，与栈解耦
        self._cold = cold or ColdStore()
        self._lock = threading.RLock()
        self._max_sessions = max_sessions
        self._cold.purge_expired()              # 进程启动时清理过期冷数据

    # ---------- 热栈访问 ----------
    def stack(self, session_id: str) -> TaskStack:
        with self._lock:
            if session_id not in self._stacks:
                self._stacks[session_id] = TaskStack(session_id=session_id)
                self._activity.setdefault(session_id, time.monotonic())
                self._evict_sessions_if_over()
            return self._stacks[session_id]

    def _evict_sessions_if_over(self) -> None:
        """会话栈超上限：LRU逐出最久不活跃的会话，栈内任务全量沉降冷存储。
        约定4保护：含【有效ongoing】的会话不参与逐出（有效任务永不离内存）；
        仅当全部会话都持有有效ongoing（512上限下极端罕见）时才回退全局LRU，
        此时active以suspended身份落库保证可resume——此为约定4的唯一显式例外。"""
        if len(self._stacks) <= self._max_sessions:
            return
        candidates = [x for x in self._stacks if not self.is_active_valid(x)]
        pool = candidates or list(self._stacks)
        oldest_sid = min(pool, key=lambda x: self._activity.get(x, 0))
        st = self._stacks.pop(oldest_sid)
        self._activity.pop(oldest_sid, None)
        for t in ([st.active] if st.active else []) + st.suspended:
            if t.status != "closed":
                t.status = "suspended"
                self._cold.put(oldest_sid, t)

    # ---------- active 有效性 ----------
    def is_active_valid(self, session_id: str) -> bool:
        """校验active任务可用：ongoing、未超时(90min活动钟)、question_meta完整
        (question_text+subject 必须非空)。"""
        t = self.stack(session_id).active
        if not t or t.status != "ongoing":
            return False
        last = self._activity.get(session_id)
        if last is not None and time.monotonic() - last > ACTIVE_TIMEOUT_SECONDS:
            return False
        if not t.question_meta.question_text or not t.question_meta.subject:
            return False
        return True

    # ---------- 决策层（不修改task_stack内容；首次建栈/LRU逐出为其副作用，
    # 运行时请统一走 dispatch() 入口） ----------
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
            return Decision(action=Action.CONTINUE,
                            task=self.stack(session_id).active.model_copy(deep=True),
                            reason="决策表:qa/chat×active有效×无新实体→continue")
        # 无active或active失效 → knowledge_qa（表尾两行；start标签但无实体的表外情形同此闭合）
        return Decision(action=Action.KNOWLEDGE_QA,
                        reason=f"决策表:无有效active(有效={valid})×无新实体→knowledge_qa")

    # ---------- 运行时统一入口（decide+execute 同锁，防TOCTOU） ----------
    def dispatch(self, session_id: str, intent: UpstreamIntent,
                 has_new_question_entity: bool,
                 question_meta: QuestionMeta | None = None,
                 resume_task_id: str | None = None) -> Decision:
        with self._lock:
            d = self.decide(session_id, intent, has_new_question_entity)
            return self._execute_locked(session_id, d, question_meta, resume_task_id)

    # ---------- 动作执行 ----------
    def execute(self, session_id: str, d: Decision,
                question_meta: QuestionMeta | None = None,
                resume_task_id: str | None = None) -> Decision:
        with self._lock:
            return self._execute_locked(session_id, d, question_meta, resume_task_id)

    def _execute_locked(self, session_id: str, d: Decision,
                        question_meta: QuestionMeta | None,
                        resume_task_id: str | None) -> Decision:
        with self._lock:
            st = self.stack(session_id)
            # 活动钟语义：仅任务相关动作(start/resume/continue)刷新。
            # "90分钟无交互"指对【任务】无交互——无关问答/闲聊不得复活过期任务。
            if d.action is Action.START:
                meta = question_meta or QuestionMeta()
                if not (meta.question_text and meta.question_text.strip()):
                    # 复审#4: START必须携带题面，否则降级knowledge_qa且不动栈
                    # (防僵尸任务+防误挂起有效active)
                    return Decision(action=Action.KNOWLEDGE_QA,
                                    reason="START缺少题面(question_meta.question_text为空)"
                                           "→降级knowledge_qa，task_stack未动")
                self._activity[session_id] = time.monotonic()
                return self._start(st, meta, d)
            if d.action is Action.RESUME:
                self._activity[session_id] = time.monotonic()
                return self._resume(st, d, resume_task_id)
            if d.action is Action.CONTINUE:
                # 强约束：continue 绝不修改 task_stack（含 last_interact_time），仅透传
                self._activity[session_id] = time.monotonic()
                return d
            return d   # knowledge_qa：不走任务栈，不刷新活动钟

    def _start(self, st: TaskStack, meta: QuestionMeta, d: Decision) -> Decision:
        new_task = TeachingTask(task_instance_id=uuid.uuid4().hex,
                                question_meta=meta, last_interact_time=time.time())
        # start 去重：同题面指纹已在 active/suspended/冷库 → 复用现有任务，禁止重复入栈
        same = [t for t in ([st.active] if st.active else []) + st.suspended
                if t.fingerprint == new_task.fingerprint]
        if same:
            exist = same[0]
            exist.last_interact_time = time.time()
            if exist.status == "suspended":     # 挂起中的同题 → 按start语义提回active
                st.suspended = [t for t in st.suspended
                                if t.task_instance_id != exist.task_instance_id]
                self._suspend_active(st)
                exist.status = "ongoing"
                st.active = exist
            d.task = exist.model_copy(deep=True)
            d.reason += "；start去重:同题面已存在，复用现有任务"
            return d
        cold_hit = self._cold.find_suspended_by_text(st.session_id, new_task.fingerprint)
        if cold_hit:
            self._suspend_active(st)
            cold_hit.status, cold_hit.last_interact_time = "ongoing", time.time()
            st.active = cold_hit
            evicted = self._evict_if_over(st)
            d.task, d.evicted = cold_hit.model_copy(deep=True), evicted
            d.reason += "；start去重:同题面在冷库，冷恢复复用"
            return d

        self._suspend_active(st)
        st.active = new_task
        evicted = self._evict_if_over(st)
        d.task, d.evicted = new_task.model_copy(deep=True), evicted
        return d

    def _resume(self, st: TaskStack, d: Decision, task_id: str | None) -> Decision:
        # 去重0：目标即当前active(或未指定且无其他候选) → 直接复用，不重复进栈
        if st.active and (not task_id or st.active.task_instance_id == task_id) \
                and not [t for t in st.suspended if not task_id or t.task_instance_id == task_id]:
            d.task = st.active.model_copy(deep=True)
            d.reason += "(去重:命中当前active)"
            return d
        # ① 内存suspended优先：未指定id时取【最近挂起】(用户"回到刚才"的预期，LIFO)
        #    ② 找不到查SQLite冷存储(最近优先)；closed不可恢复(冷库只按suspended捞)
        if task_id:
            task = next((t for t in st.suspended
                         if t.task_instance_id == task_id), None)
        else:
            task = max(st.suspended, key=lambda t: t.last_interact_time) \
                if st.suspended else None
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
        d.task = task.model_copy(deep=True)  # 深拷贝透传，隔离下游写
        d.reason, d.evicted = d.reason + f"(自{src}恢复)", evicted
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

    # ---------- FSM 推进（manager 内改栈的唯一合法途径） ----------
    def advance_active(self, session_id: str) -> str | None:
        """推进active任务FSM一步。active无效(不存在/过期/元数据不完整)时
        返回None拒绝推进——调用方可用is_active_valid区分原因。"""
        with self._lock:
            if not self.is_active_valid(session_id):
                return None
            st = self.stack(session_id)
            self._activity[session_id] = time.monotonic()
            return st.active.advance()

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
            self._activity[session_id] = time.monotonic()
            return True

    # ---------- 上游意图置信度不足时的LLM兜底（二选一：相关性判定） ----------
    def decide_via_llm(self, session_id: str, query: str, intent_confidence: float,
                       has_new_question_entity: bool,
                       confidence_threshold: float = 0.75) -> Decision:
        """上游置信度不足→组装上下文问LLM。设计约束：
        - LLM只做【与当前辅导是否相关】的二选一(relevant)，4动作由客观信号推导，
          严禁LLM输出start/resume，也严禁LLM猜测实体；
        - has_new_question_entity=True 时无条件START(客观信号优先级最高)。"""
        if intent_confidence >= confidence_threshold:
            return Decision(action=Action.KNOWLEDGE_QA,
                            reason="置信度充足，无需LLM兜底(不应调用此路径)")
        from .llm_client import chat_completion
        st = self.stack(session_id)
        valid = self.is_active_valid(session_id)
        summary = {
            "active": (st.active.question_meta.question_text[:24],
                       st.active.fsm_state) if st.active else None,
            "active_valid": valid,
            "has_new_question_entity": has_new_question_entity,
        }
        prompt = (
            "你是错题辅导任务调度器。判断用户新输入与当前正在进行的错题辅导是否相关，"
            "只输出JSON: {\"relevant\": true} 或 {\"relevant\": false}。\n"
            "true=继续当前辅导(追问/补充/确认等)；false=与当前题目无关。\n"
            f"客观事实(必须采信)：当前辅导题目={summary['active']}；"
            f"本次输入是否携带新题目实体={has_new_question_entity}\n"
            f"用户输入：{query}\n只输出JSON。")
        content = chat_completion(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": query}], temperature=0.0, max_tokens=30)
        try:
            m = re.search(r"\{.*\}", content, re.S)
            parsed = json.loads(m.group(0)) if m else {}
            relevant = parsed.get("relevant") is True   # 严格布尔：字符串"false"按无关处理
        except Exception:
            relevant = False   # 解析失败按无关处理(安全降级，不会动栈)
        if has_new_question_entity:
            # 实体覆盖→START，但不在此执行：meta 由持有新题信息的调用方提供
            return Decision(action=Action.START,
                            reason=f"LLM兜底:实体覆盖→start(上游conf={intent_confidence})，"
                                   f"待调用方execute时携带question_meta")
        if relevant and valid:
            return Decision(action=Action.CONTINUE,
                            task=st.active.model_copy(deep=True),
                            reason=f"LLM兜底:相关(relevant=true)→continue"
                                   f"(上游conf={intent_confidence})")
        return Decision(action=Action.KNOWLEDGE_QA,
                        reason=f"LLM兜底:无关或无有效任务(relevant={relevant},"
                               f"active_valid={valid})→knowledge_qa(上游conf={intent_confidence})")

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
# 注：resume/only-answer 词表是适配层的显式白名单(语义补充)，NLU标签仍为主信号。
# ---------------------------------------------------------------------------
_RESUME_PATTERNS = ("回到刚才", "继续刚才", "刚才那道题", "回到那道", "切回", "接着讲那道",
                    "刚才的题")
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
