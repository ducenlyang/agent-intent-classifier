"""学生联合多任务蒸馏训练：意图分类(蒸馏) + BIO槽位标注(远程监督)。

头A 意图: 硬标签CE + 教师软标签KL (T=2.0, α=0.5)，教师为已训练的 bert-base-chinese
头B 槽位: 词典最长匹配自动生成 BIO 标签(远程监督)，把词典能力"蒸馏"进小模型
损失: L = α·CE_hard + (1-α)·T²·KL_soft + λ·CE_ner

用法: 先完成 train_teacher，再运行
  python -m intent_classifier.distill_train.train_student_joint
产物: ckpt/student_joint.pt (推理直接加载)
"""
from __future__ import annotations

import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..config import (
    CKPT_DIR,
    MAX_LEN,
    NUM_LABELS,
    STUDENT_JOINT_CKPT,
    TEACHER_CKPT,
    TEACHER_MODEL_NAME,
)
from ..joint_model import JointIntentSlotModel, build_bio_labels, decode_slots
from ..model_hub import AutoModelForSequenceClassification, AutoTokenizer, student_model_name
from .dataset import TextDataset, load_split

EPOCHS = 4
BATCH_SIZE = 16
LR = 3e-5
TEMPERATURE = 2.0
ALPHA = 0.5     # 意图硬标签权重
NER_W = 1.0     # BIO 头损失权重
SEED = 42
RESUME_STATE = CKPT_DIR / "student_joint_resume.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_teacher(device):
    if (CKPT_DIR / "teacher_final").exists():
        model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR / "teacher_final")
        tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR / "teacher_final")
    elif TEACHER_CKPT.exists():
        tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            TEACHER_MODEL_NAME, num_labels=NUM_LABELS
        )
        model.load_state_dict(torch.load(TEACHER_CKPT, map_location="cpu"))
    else:
        raise FileNotFoundError("未找到教师权重，请先运行 train_teacher.py")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def make_collate(tokenizer):
    """动态padding + 意图标签 + 词典远程监督 BIO 标签。"""

    def collate(batch: list[dict]) -> dict:
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts, truncation=True, max_length=MAX_LEN, padding=True,
            return_tensors="pt", return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        offsets = enc.pop("offset_mapping").tolist()
        special = enc.pop("special_tokens_mask").tolist()
        bio_labels = [
            build_bio_labels(text, off, sp)
            for text, off, sp in zip(texts, offsets, special)
        ]
        max_len = enc["input_ids"].size(1)
        bio = torch.full((len(texts), max_len), -100, dtype=torch.long)
        for i, labels in enumerate(bio_labels):
            bio[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)
        return {
            "texts": texts,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "offsets": offsets,
            "intent_label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "bio_label": bio,
        }

    return collate


@torch.no_grad()
def evaluate(student, tokenizer, texts, labels, device, batch=64) -> tuple[float, float]:
    """返回 (意图acc, 短槽位精确匹配acc)。槽位gold = 词典远程监督标签。"""
    from ..slot_lexicon import GRADE_LEXICON, SUBJECT_LEXICON, find_entity_spans

    student.eval()
    correct = slot_ok = n = 0
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tokenizer(chunk, truncation=True, max_length=MAX_LEN, padding=True,
                        return_tensors="pt", return_offsets_mapping=True).to(device)
        offsets = enc.pop("offset_mapping").tolist()
        intent_logits, bio_logits = student(enc["input_ids"], enc["attention_mask"])
        lab = torch.tensor(labels[i:i + batch], device=device)
        correct += (intent_logits.argmax(-1) == lab).sum().item()

        for j, text in enumerate(chunk):
            pred = decode_slots(bio_logits[j], offsets[j], text)
            for lex, field in ((SUBJECT_LEXICON, "subject"), (GRADE_LEXICON, "grade")):
                gold_spans = find_entity_spans(text, lex)
                gold = text[gold_spans[0][0]:gold_spans[0][1]] if gold_spans else None
                pv = pred.get(field)
                got = pv["value"] if pv and pv["confidence"] >= 0.5 else None
                slot_ok += (gold == got)
                n += 1
    return correct / len(texts), slot_ok / n


def main() -> None:
    set_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = student_model_name()
    print(f"设备: {device} | 学生: {base_name} (联合多任务: 意图8分类 + BIO槽位)")

    teacher_model, teacher_tok = load_teacher(device)
    student_tok = AutoTokenizer.from_pretrained(base_name)
    student = JointIntentSlotModel(base_name, num_intents=NUM_LABELS).to(device)

    texts_train, labels_train = load_split("train")
    texts_val, labels_val = load_split("val")
    texts_test, labels_test = load_split("test")
    print(f"训练 {len(texts_train)} / 验证 {len(texts_val)} / 测试 {len(texts_test)}")

    loader = DataLoader(
        TextDataset(texts_train, labels_train), batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=make_collate(student_tok),
    )
    opt = AdamW(student.parameters(), lr=LR, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()

    best_score, start_epoch = 0.0, 0
    if RESUME_STATE.exists():  # ---- 断点续训 ----
        state = torch.load(RESUME_STATE, map_location=device)
        student.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        best_score, start_epoch = state["best_score"], state["epoch"] + 1
        print(f"↩ 从 epoch {start_epoch} 续训 (历史 best_score={best_score:.4f})")

    for epoch in range(start_epoch, EPOCHS):
        student.train()
        t0, total_loss, steps = time.time(), 0.0, 0
        for batch in loader:
            # 师生各自编码（词表可能不同）
            with torch.no_grad():
                t_enc = teacher_tok(batch["texts"], truncation=True, max_length=MAX_LEN,
                                    padding=True, return_tensors="pt").to(device)
                t_logits = teacher_model(**t_enc).logits
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            hard = batch["intent_label"].to(device)
            bio = batch["bio_label"].to(device)

            s_intent, s_bio = student(ids, mask)
            loss_hard = ce(s_intent, hard)
            loss_soft = F.kl_div(
                F.log_softmax(s_intent / TEMPERATURE, dim=-1),
                F.softmax(t_logits / TEMPERATURE, dim=-1),
                reduction="batchmean",
            ) * (TEMPERATURE ** 2)
            loss_ner = F.cross_entropy(
                s_bio.view(-1, s_bio.size(-1)), bio.view(-1), ignore_index=-100
            )
            loss = ALPHA * loss_hard + (1 - ALPHA) * loss_soft + NER_W * loss_ner

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            steps += 1

        acc_intent, acc_slot = evaluate(student, student_tok, texts_val, labels_val, device)
        score = (acc_intent + acc_slot) / 2
        print(f"epoch {epoch} | loss={total_loss/steps:.4f} "
              f"val_intent_acc={acc_intent:.4f} val_slot_acc={acc_slot:.4f} "
              f"耗时{time.time()-t0:.0f}s", flush=True)
        if score > best_score:
            best_score = score
            torch.save(
                {"state_dict": student.state_dict(),
                 "base_model_name": base_name,
                 "num_intents": NUM_LABELS},
                STUDENT_JOINT_CKPT,
            )
            print(f"  ↑ 最优，已保存到 {STUDENT_JOINT_CKPT}")
        torch.save(
            {"model": student.state_dict(), "opt": opt.state_dict(),
             "epoch": epoch, "best_score": best_score},
            RESUME_STATE,
        )

    # ---- 最终测试集评估（教师无槽位头，仅评学生联合模型）----
    s_intent, s_slot = evaluate(student, student_tok, texts_test, labels_test, device)
    print(f"\n测试集(学生联合模型): 意图acc={s_intent:.4f} | 短槽位acc={s_slot:.4f}")
    RESUME_STATE.unlink(missing_ok=True)
    print("联合多任务蒸馏训练完成")


if __name__ == "__main__":
    main()
