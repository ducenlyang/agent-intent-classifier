"""数据集：IntentDataset(编码后) 供教师/学生训练，TextDataset(原文) 供蒸馏双编码。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..config import DATA_DIR, LABEL_MAP_PATH

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
ID2LABEL = {int(k): v for k, v in LABEL_MAP.items()}
LABEL2ID = {v: int(k) for k, v in LABEL_MAP.items()}


def load_csv(path: str | Path) -> tuple[list[str], list[int]]:
    """读取 data/*.csv (text,label) → texts, label_ids

    用 utf-8-sig 读取：自动剥离 BOM（gen_data 写出的 Excel 友好格式），
    也兼容不带 BOM 的普通 UTF-8 文件（如人工标注导出的 csv）。
    """
    texts, labels = [], []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(LABEL2ID[row["label"]])
    return texts, labels


def load_split(split: str) -> tuple[list[str], list[int]]:
    return load_csv(DATA_DIR / f"{split}.csv")


class IntentDataset(Dataset):
    """编码后的数据集（教师常规训练 / 学生验证）。

    dynamic=True 时不 pad 到 max_len，配合 collate_dynamic 按批内最长补齐，
    短句场景（中位句长~16）CPU 训练提速约 2.5 倍。
    """

    def __init__(self, texts, labels, tokenizer, max_len=64, dynamic=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dynamic = dynamic

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding=False if self.dynamic else "max_length",
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_dynamic(batch: list[dict]) -> dict:
    """按批内最长序列补齐（pad_token_id=0, BERT 系通用）。"""
    maxlen = max(b["input_ids"].size(0) for b in batch)
    ids = torch.zeros(len(batch), maxlen, dtype=torch.long)
    mask = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        ids[i, :n] = b["input_ids"]
        mask[i, :n] = b["attention_mask"]
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "label": torch.stack([b["label"] for b in batch]),
    }


class TextDataset(Dataset):
    """保留原文的数据集（蒸馏时师生各自 tokenize，避免词表不一致问题）。"""

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}


def collate_text(batch: list[dict]) -> tuple[list[str], torch.Tensor]:
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return texts, labels
