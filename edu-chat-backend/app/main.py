"""FastAPI 入口：聊天接口 + 静态页面托管。

启动: uvicorn app.main:app --port 8600  (项目根目录执行)
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import gateway
from .config import GATEWAY_URL
from .gateway import IntentResult
from .graph import run_turn

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:  # 启动时探测网关服务是否在线（模型预热由网关自己的 lifespan 完成）
        ok = requests.get(f"{GATEWAY_URL.rstrip('/')}/health", timeout=5).json()
        print(f"[startup] 意图网关在线: {ok}")
    except Exception as e:
        print(f"[startup] 警告: 意图网关({GATEWAY_URL})不可达({e})，"
              f"请先启动: uvicorn intent_classifier.api:app --port 8601")
    yield


app = FastAPI(title="edu-chat-backend", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    route_kind: str                      # intercept / clarify / agent
    intent: dict                         # primary/secondary/confidence/handled_by
    slots: dict
    missing_slots: list[str]
    guard: dict = Field(default_factory=dict)
    latency_ms: int = 0


INTENT_ZH = {
    "QUESTION_SUBJECT": "学科问题", "QUESTION_POLICY": "政策咨询",
    "REQUEST_STUDY_PLAN": "学习计划", "REQUEST_ERROR_ANALYSIS": "错题分析",
    "CHAT_EMOTION": "情感倾诉", "REFUSE_CHEAT": "作弊拒绝",
    "GENERAL_CHAT": "闲聊", "UNKNOWN": "未识别",
}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/sessions")
def new_session() -> dict:
    """新建会话，返回唯一 session_id。"""
    return {"session_id": uuid.uuid4().hex}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    try:
        state = run_turn(req.session_id, req.query.strip())
    except Exception as e:  # 图执行异常兜底，接口永不500
        return ChatResponse(
            session_id=req.session_id,
            reply=f"系统开小差了({type(e).__name__})，请稍后再试或换个说法～",
            route_kind="error", intent={"primary_intent": "ERROR"},
            slots={}, missing_slots=[],
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    ir: IntentResult = state["intent_result"]
    return ChatResponse(
        session_id=req.session_id,
        reply=state["final_answer"],
        route_kind=state["route_kind"],
        intent={
            "primary_intent": ir.primary_intent.value,
            "primary_intent_zh": INTENT_ZH.get(ir.primary_intent.value, "?"),
            "secondary_intent": ir.secondary_intent.value if ir.secondary_intent else None,
            "confidence": ir.confidence,
            "handled_by": ir.handled_by,
            "need_guide_only": ir.need_guide_only,
        },
        slots={k: v for k, v in state["merged_slots"].model_dump().items() if v},
        missing_slots=state.get("still_missing") or [],
        guard=state.get("guard_info") or {},
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
