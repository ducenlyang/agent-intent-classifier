"""联合多任务小模型：意图分类头(A) + BIO 序列标注头(B)。

头A: 8分类意图 + intent_confidence
头B: subject/grade 实体的 BIO 标注（O / B-SUBJECT / I-SUBJECT / B-GRADE / I-GRADE）
推理与训练共用本模块（编码、远程监督打标、解码）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model_hub import AutoModel
from .slot_lexicon import GRADE_LEXICON, SUBJECT_LEXICON, find_entity_spans

BIO_LABELS = ["O", "B-SUBJECT", "I-SUBJECT", "B-GRADE", "I-GRADE"]
BIO2ID = {l: i for i, l in enumerate(BIO_LABELS)}
ID2BIO = {i: l for l, i in BIO2ID.items()}
NER_TYPES = ("SUBJECT", "GRADE")  # 实体类型 → 槽位字段
TYPE2SLOT = {"SUBJECT": "subject", "GRADE": "grade"}


class JointIntentSlotModel(nn.Module):
    """encoder + 双输出头。"""

    def __init__(self, base_model_name: str, num_intents: int = 8,
                 num_bio: int = len(BIO_LABELS)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.intent_head = nn.Linear(hidden, num_intents)
        self.bio_head = nn.Linear(hidden, num_bio)

    def forward(self, input_ids, attention_mask):
        seq = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        seq = self.dropout(seq)
        pooled = seq[:, 0]  # [CLS]
        return self.intent_head(pooled), self.bio_head(seq)


# ---------------------------------------------------------------------------
# BIO 远程监督打标（词典最长匹配 → token 级 BIO id，特殊/补齐位 = -100）
# ---------------------------------------------------------------------------
def build_bio_labels(text: str, offsets: list[tuple[int, int]],
                     special_mask: list[bool]) -> list[int]:
    spans = {
        "SUBJECT": find_entity_spans(text, SUBJECT_LEXICON),
        "GRADE": find_entity_spans(text, GRADE_LEXICON),
    }
    labels = []
    for (cs, ce), is_special in zip(offsets, special_mask):
        if is_special or cs == ce:  # [CLS]/[SEP]/[PAD]
            labels.append(-100)
            continue
        tag = "O"
        for etype, espans in spans.items():
            for ss, se in espans:
                if ss <= cs < se:
                    tag = f"B-{etype}" if cs == ss else f"I-{etype}"
                    break
            if tag != "O":
                break
        labels.append(BIO2ID[tag])
    return labels


# ---------------------------------------------------------------------------
# BIO 解码：每个实体类型取置信度最高的 span
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_slots(bio_logits: torch.Tensor, offsets: list[tuple[int, int]],
                 query: str) -> dict[str, dict]:
    """返回 {slot_field: {"value": str, "confidence": float}}，仅含检测到的实体。"""
    probs = torch.softmax(bio_logits, dim=-1)  # [seq, 5]
    pred = probs.argmax(-1).tolist()
    result: dict[str, dict] = {}
    for etype in NER_TYPES:
        b_id, i_id = BIO2ID[f"B-{etype}"], BIO2ID[f"I-{etype}"]
        spans: list[tuple[float, int, int]] = []  # (score, tok_start, tok_end)
        i = 0
        while i < len(pred):
            if pred[i] in (b_id, i_id) and offsets[i][0] < offsets[i][1]:
                start = i
                i += 1
                while i < len(pred) and pred[i] == i_id and offsets[i][0] < offsets[i][1]:
                    i += 1
                token_ids = list(range(start, i))
                score = sum(probs[t, pred[t]].item() for t in token_ids) / len(token_ids)
                spans.append((score, start, i - 1))
            else:
                i += 1
        if spans:
            score, ts, te = max(spans, key=lambda s: (s[0], s[2] - s[1]))
            value = query[offsets[ts][0]:offsets[te][1]]
            if value:
                result[TYPE2SLOT[etype]] = {"value": value, "confidence": round(score, 4)}
    return result
