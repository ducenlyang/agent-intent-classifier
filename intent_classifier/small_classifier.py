"""第二层：小模型推理——仅输出【一级意图 + 置信度】，不做槽位抽取。

复用联合训练的 ckpt（意图头即独立训练产物，BIO头推理时休眠），无需重训。
全部槽位由第三层 LLM 抽取（分层职责分离）。
"""
from __future__ import annotations

import json
import time

import torch
from pydantic import BaseModel

from .config import (
    LABEL_MAP_PATH,
    MAX_LEN,
    NUM_LABELS,
    PrimaryIntent,
    STUDENT_JOINT_CKPT,
)
from .joint_model import JointIntentSlotModel
from .model_hub import AutoTokenizer, student_model_name

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
ID2LABEL = {int(k): v for k, v in LABEL_MAP.items()}
LABEL2ID = {v: int(k) for k, v in LABEL_MAP.items()}


class SmallModelOutput(BaseModel):
    intent: PrimaryIntent
    intent_confidence: float


class SmallClassifier:
    def __init__(self, ckpt_path=None):
        if not STUDENT_JOINT_CKPT.exists():
            raise FileNotFoundError(
                f"未找到小模型权重 {STUDENT_JOINT_CKPT}，请先运行:\n"
                f"  python -m intent_classifier.distill_train.train_student_joint"
            )
        self.device = torch.device("cpu")  # 生产定位：CPU 离线部署
        bundle = torch.load(STUDENT_JOINT_CKPT, map_location="cpu")
        self.model_name = bundle.get("base_model_name", student_model_name())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = JointIntentSlotModel(self.model_name, num_intents=NUM_LABELS)
        self.model.load_state_dict(bundle["state_dict"])
        self.model.to(self.device).eval()
        self.source = f"joint-distilled:{STUDENT_JOINT_CKPT}"

    @torch.no_grad()
    def predict(self, query: str) -> tuple[SmallModelOutput, int]:
        t0 = time.perf_counter()
        enc = self.tokenizer(query, truncation=True, max_length=MAX_LEN,
                             return_tensors="pt")
        intent_logits, _ = self.model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(intent_logits, dim=-1)
        conf, pred_id = torch.max(probs, dim=-1)
        out = SmallModelOutput(
            intent=PrimaryIntent(ID2LABEL[int(pred_id.item())]),
            intent_confidence=round(float(conf.item()), 4),
        )
        return out, int((time.perf_counter() - t0) * 1000)


_classifier: SmallClassifier | None = None


def get_small_classifier() -> SmallClassifier:
    """进程内单例，首次调用才加载模型。"""
    global _classifier
    if _classifier is None:
        _classifier = SmallClassifier()
    return _classifier
