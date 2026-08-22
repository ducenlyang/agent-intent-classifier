"""教师模型训练：bert-base-chinese 12层 → 8分类意图识别。

特性: 动态padding(CPU提速) / 每epoch存档断点续训 / 保存最优权重。
用法: python -m intent_classifier.distill_train.train_teacher
      (中途被打断后重新运行同一命令即可续训)
产物: ckpt/teacher_best.pt (state_dict) + ckpt/teacher_final/ (save_pretrained)
"""
from __future__ import annotations

import os
import random
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..config import (
    CKPT_DIR,
    MAX_LEN,
    NUM_LABELS,
    TEACHER_CKPT,
    TEACHER_MODEL_NAME,
)
from ..model_hub import AutoModelForSequenceClassification, AutoTokenizer
from .dataset import ID2LABEL, IntentDataset, collate_dynamic, load_split

EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
SEED = 42
RESUME_STATE = CKPT_DIR / "teacher_resume.pt"  # model+opt+epoch 存档


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    correct = loss_sum = n = 0
    ce = nn.CrossEntropyLoss()
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lab = batch["label"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss_sum += ce(logits, lab).item() * len(lab)
        correct += (logits.argmax(-1) == lab).sum().item()
        n += len(lab)
    model.train()
    return correct / n, loss_sum / n


def main() -> None:
    set_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 线程: {torch.get_num_threads()} | 教师: {TEACHER_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
    ).to(device)

    texts_train, labels_train = load_split("train")
    texts_val, labels_val = load_split("val")
    print(f"训练集 {len(texts_train)} 条 / 验证集 {len(texts_val)} 条")

    train_loader = DataLoader(
        IntentDataset(texts_train, labels_train, tokenizer, MAX_LEN, dynamic=True),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_dynamic,
    )
    val_loader = DataLoader(
        IntentDataset(texts_val, labels_val, tokenizer, MAX_LEN, dynamic=True),
        batch_size=BATCH_SIZE * 2, collate_fn=collate_dynamic,
    )

    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()

    best_acc, start_epoch = 0.0, 0
    if RESUME_STATE.exists():  # ---- 断点续训 ----
        state = torch.load(RESUME_STATE, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        best_acc, start_epoch = state["best_acc"], state["epoch"] + 1
        print(f"↩ 从 epoch {start_epoch} 续训 (历史 best_acc={best_acc:.4f})")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t0, total_loss = time.time(), 0.0
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab = batch["label"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = ce(logits, lab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            if (step + 1) % 50 == 0:
                print(f"  step {step+1}/{len(train_loader)} loss={loss.item():.4f}")
        acc, val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch} | train_loss={total_loss/len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} val_acc={acc:.4f} "
              f"耗时{time.time()-t0:.0f}s", flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), TEACHER_CKPT)
            model.save_pretrained(CKPT_DIR / "teacher_final")
            tokenizer.save_pretrained(CKPT_DIR / "teacher_final")
            print(f"  ↑ 最优，已保存到 {TEACHER_CKPT}")
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(),
             "epoch": epoch, "best_acc": best_acc},
            RESUME_STATE,
        )

    RESUME_STATE.unlink(missing_ok=True)  # 训练完成清理存档
    print(f"教师训练完成，最佳 val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
