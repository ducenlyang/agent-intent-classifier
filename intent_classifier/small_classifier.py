"""第二层：学生小模型（tiny-bert）离线推理。

只输出【一级意图 + 置信度】，不做槽位抽取（槽位由第三层 LLM 填充）。
加载优先级：ckpt/student_final(save_pretrained) → ckpt/student_best.pt → 原始预训练模型(未蒸馏,告警)。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from pydantic import BaseModel

from .config import (
    LABEL_MAP_PATH,
    MAX_LEN,
    NUM_LABELS,
    PrimaryIntent,
    STUDENT_CKPT,
    STUDENT_SAVE_DIR,
)
from .model_hub import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    student_model_name,
)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
ID2LABEL = {int(k): v for k, v in LABEL_MAP.items()}
LABEL2ID = {v: int(k) for k, v in LABEL_MAP.items()}


class SmallModelOutput(BaseModel):
    intent: PrimaryIntent
    confidence: float


class SmallClassifier:
    def __init__(self, ckpt_path: str | Path | None = None):
        self.model_name = student_model_name()
        self.source = "untrained-pretrained"  # 记录权重来源，便于诊断
        self.device = torch.device("cpu")  # 生产定位：CPU 离线部署

        # 1) 优先加载蒸馏训练产物 save_pretrained 目录
        if STUDENT_SAVE_DIR.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(STUDENT_SAVE_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                STUDENT_SAVE_DIR
            )
            self.source = f"distilled:{STUDENT_SAVE_DIR}"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=NUM_LABELS
            )
            # 2) 其次加载 state_dict 蒸馏权重
            real_ckpt = Path(ckpt_path) if ckpt_path else STUDENT_CKPT
            if real_ckpt.exists():
                state = torch.load(real_ckpt, map_location="cpu")
                self.model.load_state_dict(state)
                self.source = f"distilled:{real_ckpt}"
            else:
                # 3) 都没有 → 用原始预训练分类头（演示可跑，但精度不保证）
                print(
                    f"[SmallClassifier] 警告: 未找到蒸馏权重({STUDENT_CKPT})，"
                    f"当前使用未训练的 {self.model_name}，请先执行 distill_train 训练。"
                )
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, query: str):
        # 单条推理无需 padding，截断到 MAX_LEN 即可
        return self.tokenizer(
            query, truncation=True, max_length=MAX_LEN, return_tensors="pt",
        )

    @torch.no_grad()
    def predict(self, query: str) -> SmallModelOutput:
        t0 = time.perf_counter()
        inputs = self._encode(query)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        conf, pred_id = torch.max(probs, dim=-1)
        return SmallModelOutput(
            intent=PrimaryIntent(ID2LABEL[int(pred_id.item())]),
            confidence=round(float(conf.item()), 4),
        ), int((time.perf_counter() - t0) * 1000)

    @torch.no_grad()
    def predict_batch(self, queries: list[str]) -> list[SmallModelOutput]:
        results = []
        for q in queries:  # CPU 单条推理已足够快，逐条保持与线上一致的行为
            out, _ = self.predict(q)
            results.append(out)
        return results


_classifier: SmallClassifier | None = None


def get_small_classifier() -> SmallClassifier:
    """进程内单例，首次调用才加载模型（避免 import 时下载）。"""
    global _classifier
    if _classifier is None:
        _classifier = SmallClassifier()
    return _classifier
