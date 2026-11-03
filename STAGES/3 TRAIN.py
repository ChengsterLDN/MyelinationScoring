"""
Stage 3: fine-tune an image classifier on per-pillar wrap
scores (0-3).

Splits by series, weights the loss by class frequency, uses exact dihedral augmentation, and
selects the best epoch by quadratic-weighted kappa.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

import config
from wrap_common import DihedralTransform

NUM_LABELS = config.NUM_LABELS


class CropDataset(TorchDataset):
    """Paths only; images are decoded on demand."""

    def __init__(self, root: Path, rows: list[dict], augment: bool, seed: int = 0):
        self.root = root
        self.rows = rows
        # Random dihedral transform for training, none for evaluation
        self.aug = DihedralTransform(np.random.default_rng(seed)) if augment else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        # Load one crop from disk and apply augmentation if enabled
        row = self.rows[i]
        img = Image.open(self.root / row["filepath"]).convert("RGB")
        if self.aug is not None:
            img = self.aug(img)
        return {"image": img, "label": int(row["label"])}


def make_collator(processor):
    """Batch function: run the backbone's image processor and attach labels."""
    def collate(batch):
        enc = processor(images=[b["image"] for b in batch], return_tensors="pt")
        enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return enc
    return collate


def split_by_group(rows: list[dict], group_key: str, holdout_frac: float, seed: int):
    """Assign whole groups (series by default) to train or eval."""
    groups = sorted({r[group_key] for r in rows})

    # With a single series, fall back to splitting by site
    if len(groups) < 2:
        print(f"! Only one distinct '{group_key}' - falling back to site-level "
              f"grouping. Eval will overstate cross-series performance.")
        group_key = "site_id"
        groups = sorted({r[group_key] for r in rows})

    # Randomly choose which groups are held out
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_hold = max(1, int(round(len(groups) * holdout_frac)))
    hold = {groups[i] for i in order[:n_hold]}

    # Partition rows by group membership
    train = [r for r in rows if r[group_key] not in hold]
    ev = [r for r in rows if r[group_key] in hold]
    print(f"Split on '{group_key}': {len(groups) - n_hold} train groups, "
          f"{n_hold} held out ({sorted(hold)[:6]}{'...' if n_hold > 6 else ''})")
    return train, ev


def quadratic_weighted_kappa(y_true, y_pred, n_classes=NUM_LABELS) -> float:
    """Agreement metric for ordinal labels: large errors cost more."""
    # Observed confusion matrix
    O = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1

    # Quadratic disagreement weights, 0 on the diagonal
    w = np.array([[(i - j) ** 2 for j in range(n_classes)]
                  for i in range(n_classes)]) / (n_classes - 1) ** 2

    # Expected matrix under chance, scaled to the same total
    E = np.outer(np.bincount(y_true, minlength=n_classes),
                 np.bincount(y_pred, minlength=n_classes)).astype(float)
    E *= O.sum() / max(E.sum(), 1e-9)

    denom = (w * E).sum()
    return 1.0 - (w * O).sum() / denom if denom > 0 else 0.0


def compute_metrics(eval_pred):
    """Metrics reported by the Trainer at each evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float((preds == labels).mean()),
        "mae": float(np.abs(preds - labels).mean()),
        "qwk": float(quadratic_weighted_kappa(labels.astype(int), preds.astype(int))),
        "adjacent_acc": float((np.abs(preds - labels) <= 1).mean()),
    }


class WeightedTrainer(Trainer):
    """HF Trainer with a class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Take the labels out so the model doesn't compute its own loss
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Move weights to the same device as the logits
        weights = (self.class_weights.to(outputs.logits.device)
                   if self.class_weights is not None else None)
        loss = nn.CrossEntropyLoss(weight=weights)(
            outputs.logits.view(-1, NUM_LABELS), labels.view(-1))
        # Put labels back for the Trainer's evaluation loop
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    # --- command-line overrides; defaults from config.py ---
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crops", type=Path, default=config.CROPS_DIR)
    p.add_argument("--out", type=Path, default=config.MODEL_DIR)
    p.add_argument("--backbone", default=config.BACKBONE)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--holdout-frac", type=float, default=config.HOLDOUT_FRAC)
    p.add_argument("--group-key", default=config.GROUP_KEY,
                   choices=["series", "site_id"])
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--seed", type=int, default=config.TRAIN_SEED)
    args = p.parse_args()

    labels_csv = args.crops / "labels.csv"
    if not labels_csv.exists():
        raise SystemExit(f"No labels at {labels_csv}\nRun: python -m label_crops")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}   Backbone: {args.backbone}")

    # Load labels, keeping only rows whose crop file still exists
    with open(labels_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (args.crops / r["filepath"]).exists()]
    if not rows:
        raise SystemExit(f"No usable rows in {labels_csv}")

    # Report class and source distributions
    dist = Counter(int(r["label"]) for r in rows)
    src = Counter(r.get("source", "detected") for r in rows)
    print(f"{len(rows)} labelled crops   sources: {dict(src)}")
    print("Distribution: " + "  ".join(
        f"{k}={dist.get(k, 0)} ({config.CLASS_NAMES[k]})" for k in range(NUM_LABELS)))

    # Warn about under-represented classes
    thin = [k for k in range(NUM_LABELS) if dist.get(k, 0) < 20]
    if thin:
        print(f"! Classes {thin} have <20 examples - the model will be unreliable "
              f"on them. Label more before trusting counts for those classes.")

    # Group-level train/eval split
    train_rows, eval_rows = split_by_group(rows, args.group_key,
                                           args.holdout_frac, args.seed)
    print(f"Train: {len(train_rows)}  |  Eval: {len(eval_rows)}")

    # Pretrained backbone with a fresh 4-class head
    processor = AutoImageProcessor.from_pretrained(args.backbone)
    model = AutoModelForImageClassification.from_pretrained(
        args.backbone,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        id2label={i: config.CLASS_NAMES[i] for i in range(NUM_LABELS)},
        label2id={v: k for k, v in config.CLASS_NAMES.items()},
    ).to(device)

    # Inverse-frequency class weights (empty classes treated as 1)
    class_weights = None
    if config.CLASS_WEIGHTS and not args.no_class_weights:
        counts = np.array([max(dist.get(i, 0), 1) for i in range(NUM_LABELS)], float)
        w = counts.sum() / (NUM_LABELS * counts)
        class_weights = torch.tensor(w, dtype=torch.float)
        print(f"Class weights: {np.round(w, 3).tolist()}")

    # Evaluate and checkpoint each epoch; keep the best by qwk
    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
        remove_unused_columns=False,
        fp16=(device == "cuda"),
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=CropDataset(args.crops, train_rows, True, args.seed),
        eval_dataset=CropDataset(args.crops, eval_rows, False),
        data_collator=make_collator(processor),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # Train, then save the best model and its processor
    trainer.train()
    trainer.save_model(str(args.out))
    processor.save_pretrained(str(args.out))

    # Carry the crop geometry with the weights so scoring cannot drift.
    src_pre = args.crops / "preproc.json"
    if src_pre.exists():
        shutil.copy(src_pre, args.out / "preproc.json")
    else:
        print("! No preproc.json in the crops directory - score_series will "
              "fall back to config defaults, which may not match these crops.")

    # Final metrics on the held-out set
    print("\nFinal eval:")
    for k, v in trainer.evaluate().items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Confusion matrix on the held-out set
    pred = trainer.predict(CropDataset(args.crops, eval_rows, False))
    yp = np.argmax(pred.predictions, axis=1)
    yt = pred.label_ids.astype(int)
    cm = np.zeros((NUM_LABELS, NUM_LABELS), int)
    for t, q in zip(yt, yp):
        cm[t, q] += 1

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print("       " + "".join(f"{i:>7}" for i in range(NUM_LABELS)))
    for i in range(NUM_LABELS):
        print(f"  {i}  " + "".join(f"{c:>7}" for c in cm[i]))
    np.savetxt(args.out / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    print(f"\nModel saved to {args.out}")
    print("Next: python -m score_series")


if __name__ == "__main__":
    main()
