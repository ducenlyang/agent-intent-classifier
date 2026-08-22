"""学生模型蒸馏训练：tiny-bert + (硬标签CE + 教师软标签KL)。

特性: 师生各自编码(词表可能不同) / 断点续训 / 保存最优权重 + 测试集师生对比。
用法: 先完成 train_teacher，再运行
  python -m intent_classifier.distill_train.train_student_distill
产物: ckpt/student_best.pt + ckpt/student_final/ (推理加载目录)
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
    STUDENT_CKPT,
    STUDENT_SAVE_DIR,
    TEACHER_CKPT,
    TEACHER_MODEL_NAME,
)
from ..model_hub import AutoModelForSequenceClassification, AutoTokenizer, student_model_name
from .dataset import ID2LABEL, IntentDataset, TextDataset, collate_text, load_split

EPOCHS = 4
BATCH_SIZE = 16
LR = 3e-5
TEMPERATURE = 2.0
ALPHA = 0.5  # 硬标签损失权重
SEED = 42
RESUME_STATE = CKPT_DIR / "student_resume.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_teacher(device) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """优先加载训练好的 teacher_final/，其次 teacher_best.pt。"""
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL_NAME, num_labels=NUM_LABELS
    )
    if (CKPT_DIR / "teacher_final").exists():
        model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR / "teacher_final")
        tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR / "teacher_final")
    elif TEACHER_CKPT.exists():
        model.load_state_dict(torch.load(TEACHER_CKPT, map_location="cpu"))
    else:
        raise FileNotFoundError("未找到教师权重，请先运行 train_teacher.py")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


@torch.no_grad()
def evaluate(model, tokenizer, texts, labels, device, batch=64) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i:i + batch], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(device)
        lab = torch.tensor(labels[i:i + batch], device=device)
        pred = model(**enc).logits.argmax(-1)
        correct += (pred == lab).sum().item()
    return correct / len(texts)


def main() -> None:
    set_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_name = student_model_name()
    print(f"设备: {device} | 学生模型: {student_name}")

    teacher_model, teacher_tok = load_teacher(device)
    student_tok = AutoTokenizer.from_pretrained(student_name)
    student = AutoModelForSequenceClassification.from_pretrained(
        student_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
    ).to(device)

    texts_train, labels_train = load_split("train")
    texts_val, labels_val = load_split("val")
    texts_test, labels_test = load_split("test")
    print(f"训练 {len(texts_train)} / 验证 {len(texts_val)} / 测试 {len(texts_test)}")

    loader = DataLoader(
        TextDataset(texts_train, labels_train), batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_text,
    )
    opt = AdamW(student.parameters(), lr=LR, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()

    best_acc, start_epoch = 0.0, 0
    if RESUME_STATE.exists():  # ---- 断点续训 ----
        state = torch.load(RESUME_STATE, map_location=device)
        student.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        best_acc, start_epoch = state["best_acc"], state["epoch"] + 1
        print(f"↩ 从 epoch {start_epoch} 续训 (历史 best_acc={best_acc:.4f})")

    for epoch in range(start_epoch, EPOCHS):
        student.train()
        t0, total_loss, steps = time.time(), 0.0, 0

        for texts, hard in loader:
            # 师生各自编码（词表可能不同，不能共用 input_ids）
            t_enc = teacher_tok(list(texts), truncation=True, max_length=MAX_LEN,
                                padding=True, return_tensors="pt").to(device)
            s_enc = student_tok(list(texts), truncation=True, max_length=MAX_LEN,
                                padding=True, return_tensors="pt").to(device)
            hard = hard.to(device)

            with torch.no_grad():
                t_logits = teacher_model(**t_enc).logits
            s_logits = student(**s_enc).logits

            # 1) 硬标签损失（真实标注）
            loss_hard = ce(s_logits, hard)
            # 2) 软标签蒸馏损失（教师概率分布 KL 散度）
            loss_soft = F.kl_div(
                F.log_softmax(s_logits / TEMPERATURE, dim=-1),
                F.softmax(t_logits / TEMPERATURE, dim=-1),
                reduction="batchmean",
            ) * (TEMPERATURE ** 2)

            loss = ALPHA * loss_hard + (1 - ALPHA) * loss_soft
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            steps += 1

        acc = evaluate(student, student_tok, texts_val, labels_val, device)
        print(f"epoch {epoch} | loss={total_loss/steps:.4f} val_acc={acc:.4f} "
              f"耗时{time.time()-t0:.0f}s")
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), STUDENT_CKPT)
            student.save_pretrained(STUDENT_SAVE_DIR)
            student_tok.save_pretrained(STUDENT_SAVE_DIR)
            print(f"  ↑ 最优，已保存到 {STUDENT_SAVE_DIR}")
        torch.save(
            {"model": student.state_dict(), "opt": opt.state_dict(),
             "epoch": epoch, "best_acc": best_acc},
            RESUME_STATE,
        )

    # ---- 最终测试集对比 ----
    t_acc = evaluate(teacher_model, teacher_tok, texts_test, labels_test, device)
    s_acc = evaluate(student, student_tok, texts_test, labels_test, device)
    print(f"\n测试集对比: 教师(12层) acc={t_acc:.4f} | 蒸馏后学生 acc={s_acc:.4f}")
    RESUME_STATE.unlink(missing_ok=True)
    print("学生蒸馏训练完成")


if __name__ == "__main__":
    main()
