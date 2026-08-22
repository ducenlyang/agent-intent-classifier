"""意图网关 HTTP 服务（无状态）：供下游对话后端 HTTP 调用。

启动（项目根目录）:
  uvicorn intent_classifier.api:app --host 0.0.0.0 --port 8601

接口:
  GET  /health   健康检查
  POST /classify {"query": "..."} → IntentResult JSON
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .intent_node import IntentPipeline

_pipeline: IntentPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = IntentPipeline()  # 启动预热(含tiny-bert)，避免首请求慢
    yield


app = FastAPI(title="agent-intent-classifier", lifespan=lifespan)


class ClassifyRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "intent-gateway"}


@app.post("/classify")
def classify(req: ClassifyRequest) -> dict:
    t0 = time.perf_counter()
    result = _pipeline.classify(req.query.strip())
    data = result.model_dump()
    data["gateway_latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return data
