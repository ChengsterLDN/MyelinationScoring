"""
Stage 3b: continue training an existing wrap classifier on a new batch of labelled crops.

    python -m "3B CONTRAIN" --old-crops ./Crops_run1 --crops ./Crops_run2 \
                             --model ./Model --out ./Model_v2

"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import EarlyStoppingCallback, TrainingArguments, AutoImageProcessor
from transformers import AutoModelForImageClassification

import config

NUM_LABELS = config.NUM_LABELS
# preproc.json keys that must match between model and crops
GEOM_KEYS = ("context_pitches", "out_size", "pillar_r_frac", "slide_at_edges")
# Filenames the stage-3 training script may be saved under
TRAIN_MODULE_NAMES = ("train_wrap", "3_TRAIN", "3 TRAIN")


def load_train_module(explicit: Path | None):
    """Import stage 3 so its dataset/split/metrics are reused, not copied.
    Falls back to loading by file path, since '3 TRAIN.py' is not a legal
    module name."""
    # Normal import first, unless a path was given explicitly
    if explicit is None:
        try:
            return __import__("train_wrap")
        except ImportError:
            pass
    # Otherwise try each candidate file path
    candidates = ([explicit] if explicit else
                  [Path(f"{n}.py") for n in TRAIN_MODULE_NAMES])
    for path in candidates:
        if path and path.exists():
            spec = importlib.util.spec_from_file_location("train_wrap_loaded", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["train_wrap_loaded"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(
        "! could not import the stage-3 training script. Put train_wrap.py on the "
        "path or pass --train-module '3 TRAIN.py'.")


# ---------------------------------------------------------------------------
# Labels from two crop directories
# ---------------------------------------------------------------------------

def read_rows(crops: Path, tag: str) -> list[dict]:
    """labels.csv from one crop set, with absolute filepaths (so one dataset
    can span two folders) and series prefixed by the batch tag (so names
    can't collide and eval can be reported per batch)."""
    labels_csv = crops / "labels.csv"
    if not labels_csv.exists():
        raise SystemExit(f"! no labels at {labels_csv}")
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    out = []
    for r in rows:
        # Skip rows whose crop file is missing
        p = (crops / r["filepath"]).resolve()
        if not p.exists():
            continue
        # Copy and rewrite path, series and batch tag
        r = dict(r)
        r["filepath"] = str(p)
        r["series"] = f"{tag}:{r.get('series', 'unknown')}"
        r["batch"] = tag
        out.append(r)
    if not out:
        raise SystemExit(f"! no usable rows in {labels_csv}")

    # Report class distribution for this batch
    dist = Counter(int(r["label"]) for r in out)
    print(f"  {tag}: {len(out)} crops   " + "  ".join(
        f"{config.CLASS_NAMES[k]}={dist.get(k, 0)}" for k in range(NUM_LABELS)))
    return out


def read_preproc(path: Path) -> dict | None:
    """preproc.json from a crops or model folder, or None if absent."""
    f = path / "preproc.json"
    return json.loads(f.read_text()) if f.exists() else None


def check_geometry(model_pre: dict | None, crops_pre: dict | None, tag: str,
                    allow: bool) -> None:
    """Stop if the crops were cut with different geometry from the model."""
    if model_pre is None or crops_pre is None:
        print(f"! no preproc.json for {tag} - cannot verify the crop geometry matches "
              f"the model's. Continuing blind.")
        return
    # Collect every geometry key whose value differs
    diff = {k: (model_pre.get(k), crops_pre.get(k)) for k in GEOM_KEYS
            if k in model_pre and k in crops_pre and model_pre[k] != crops_pre[k]}
    if not diff:
        return
    msg = "  ".join(f"{k}: model={a} {tag}={b}" for k, (a, b) in diff.items())
    if not allow:
        raise SystemExit(
            f"! {tag} crops were cut with different geometry -- {msg}\n"
            f"  They are not the same kind of image as the model was trained on. "
            f"Recut them with the model's settings, or pass --allow-preproc-mismatch.")
    print(f"! proceeding with mismatched crop geometry for {tag} -- {msg}")


def write_merged(rows: list[dict], out_dir: Path, preproc: dict | None) -> None:
    """Write a merged labels.csv that train_wrap can read as-is (absolute
    filepaths, so no images are copied)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Union of all column names across both batches
    fields = sorted({k for r in rows for k in r})
    with open(out_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # Carry geometry along too
    if preproc is not None:
        (out_dir / "preproc.json").write_text(json.dumps(preproc, indent=2))
    print(f"Merged {len(rows)} rows -> {out_dir / 'labels.csv'}")
    print(f"Now: python -m train_wrap --crops {out_dir} --out <new model dir>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- command-line options ---
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crops", type=Path, default=config.CROPS_DIR,
                   help="the NEW crop set to add")
    p.add_argument("--old-crops", type=Path, default=None,
                   help="the crop set the model was trained on - pass it if you have it")
    p.add_argument("--model", type=Path, default=config.MODEL_DIR,
                   help="the trained model to continue from")
    p.add_argument("--out", type=Path, default=None, help="new model directory")
    p.add_argument("--write-merged", type=Path, default=None,
                   help="write a merged labels.csv here and exit (no training)")
    p.add_argument("--train-module", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="deliberately below config.LEARNING_RATE: this is a "
                        "continuation, not a fresh fit")
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="freeze embeddings + the first N encoder blocks")
    p.add_argument("--holdout-frac", type=float, default=config.HOLDOUT_FRAC)
    p.add_argument("--group-key", default=config.GROUP_KEY,
                   choices=["series", "site_id"])
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--eval-only", action="store_true",
                   help="score the current model on both crop sets and stop")
    p.add_argument("--allow-preproc-mismatch", action="store_true")
    p.add_argument("--seed", type=int, default=config.TRAIN_SEED)
    args = p.parse_args()

    # Reuse stage 3's dataset, split, metrics and trainer
    tw = load_train_module(args.train_module)

    # --- crop sets -----------------------------------------------------------
    print("Crop sets:")
    batches = {"new": read_rows(args.crops, "new")}
    if args.old_crops:
        batches["old"] = read_rows(args.old_crops, "old")
    else:
        print("  ! no --old-crops: training on the new batch alone. Whatever the "
              "first batch taught the model is free to drift, and there is no old "
              "held-out data here to detect it.")

    # Verify both crop sets match the model's geometry
    model_pre = read_preproc(args.model)
    check_geometry(model_pre, read_preproc(args.crops), "new",
                   args.allow_preproc_mismatch)
    if args.old_crops:
        check_geometry(model_pre, read_preproc(args.old_crops), "old",
                       args.allow_preproc_mismatch)

    # Merge-only mode: write the union and exit
    all_rows = [r for rows in batches.values() for r in rows]
    if args.write_merged:
        write_merged(all_rows, args.write_merged,
                     read_preproc(args.crops) or model_pre)
        return
    if args.out is None:
        raise SystemExit("! --out is required (unless --write-merged)")

    # --- split each batch on its own series ---------------------------------
    # splitting the pool as one would let a batch with more series dominate
    # eval, and would make per-batch metrics depend on the draw
    train_rows: list[dict] = []
    eval_rows: dict[str, list[dict]] = {}
    for tag, rows in batches.items():
        print(f"\n[{tag}] ", end="")
        tr, ev = tw.split_by_group(rows, args.group_key, args.holdout_frac, args.seed)
        train_rows += tr
        eval_rows[tag] = ev
    eval_all = [r for ev in eval_rows.values() for r in ev]
    print(f"\nTrain: {len(train_rows)}  |  Eval: " +
          "  ".join(f"{k}={len(v)}" for k, v in eval_rows.items()))

    # Warn about thin classes in the combined training set
    dist = Counter(int(r["label"]) for r in train_rows)
    thin = [k for k in range(NUM_LABELS) if dist.get(k, 0) < 20]
    if thin:
        print(f"! Classes {thin} have <20 training examples in the union - the model "
              f"will stay unreliable on them however long this runs.")

    # --- model ---------------------------------------------------------------
    # Warm start from the existing model (not the backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).to(device)
    if model.config.num_labels != NUM_LABELS:
        raise SystemExit(f"! model has {model.config.num_labels} labels, config wants "
                         f"{NUM_LABELS} - this is not the same task")
    print(f"Device: {device}   Continuing from: {args.model}")

    # Optionally freeze embeddings + the first N encoder blocks
    if args.freeze_layers > 0:
        base = getattr(model, model.base_model_prefix, None)
        layers = getattr(getattr(base, "encoder", None), "layer", None)
        if layers is None:
            print(f"! --freeze-layers: no encoder.layer on this backbone, skipping")
        else:
            for prm in base.embeddings.parameters():
                prm.requires_grad = False
            for blk in layers[:args.freeze_layers]:
                for prm in blk.parameters():
                    prm.requires_grad = False
            n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
            print(f"  froze embeddings + first {args.freeze_layers} blocks "
                  f"({n_train / 1e6:.1f}M params still training)")

    # Inverse-frequency class weights from the combined training set
    class_weights = None
    if config.CLASS_WEIGHTS and not args.no_class_weights:
        counts = np.array([max(dist.get(i, 0), 1) for i in range(NUM_LABELS)], float)
        w = counts.sum() / (NUM_LABELS * counts)
        class_weights = torch.tensor(w, dtype=torch.float)
        print(f"Class weights (union): {np.round(w, 3).tolist()}")

    # Same training setup as stage 3, with a lower learning rate
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

    # Paths are absolute, so the dataset root is just "."
    trainer = tw.WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tw.CropDataset(Path("."), train_rows, True, args.seed),
        eval_dataset=tw.CropDataset(Path("."), eval_all, False),
        data_collator=tw.make_collator(processor),
        compute_metrics=tw.compute_metrics,
        class_weights=class_weights,
        callbacks=None if args.no_early_stopping
        else [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    def per_batch_eval(tag: str) -> dict:
        """Metrics for each crop set separately - the pooled number can hide
        a collapse on one of them."""
        report = {}
        for name, rows in eval_rows.items():
            # Predict on this batch's held-out rows
            pred = trainer.predict(tw.CropDataset(Path("."), rows, False))
            yp = np.argmax(pred.predictions, axis=1)
            yt = pred.label_ids.astype(int)
            m = {"n": len(rows),
                 "accuracy": float((yp == yt).mean()),
                 "adjacent_acc": float((np.abs(yp - yt) <= 1).mean()),
                 "mae": float(np.abs(yp - yt).mean()),
                 "qwk": float(tw.quadratic_weighted_kappa(yt, yp))}
            report[name] = m
            print(f"  [{tag}] {name:<4} n={m['n']:<5} acc={m['accuracy']:.3f} "
                  f"adj={m['adjacent_acc']:.3f} mae={m['mae']:.3f} qwk={m['qwk']:.3f}")
            # After training, also save a confusion matrix per batch
            if tag == "after":
                cm = np.zeros((NUM_LABELS, NUM_LABELS), int)
                for t, q in zip(yt, yp):
                    cm[t, q] += 1
                np.savetxt(args.out / f"confusion_matrix_{name}.csv", cm,
                           fmt="%d", delimiter=",")
        return report

    # Baseline metrics with the unchanged model
    print("\nBefore training:")
    before = per_batch_eval("before")
    if args.eval_only:
        print("\n--eval-only: stopping here. The 'new' row is how the current model "
              "does on the new crops as it stands - if that is already good enough, "
              "the safest change is no change.")
        return

    trainer.train()

    # Metrics after training
    print("\nAfter training:")
    args.out.mkdir(parents=True, exist_ok=True)
    after = per_batch_eval("after")

    # --- save ----------------------------------------------------------------
    trainer.save_model(str(args.out))
    processor.save_pretrained(str(args.out))
    # Copy geometry from the new crops, else from the parent model
    src_pre = (args.crops / "preproc.json" if (args.crops / "preproc.json").exists()
               else args.model / "preproc.json")
    if src_pre.exists():
        shutil.copy(src_pre, args.out / "preproc.json")
    else:
        print("! no preproc.json to carry forward - score_series will fall back to "
              "config defaults, which may not match these crops.")

    # Provenance record: parent, data, hyperparameters, before/after metrics
    (args.out / "train_manifest.json").write_text(json.dumps({
        "parent_model": str(args.model),
        "new_crops": str(args.crops),
        "old_crops": str(args.old_crops) if args.old_crops else None,
        "replay": bool(args.old_crops),
        "n_train": len(train_rows),
        "n_eval": {k: len(v) for k, v in eval_rows.items()},
        "train_class_counts": {config.CLASS_NAMES[k]: dist.get(k, 0)
                               for k in range(NUM_LABELS)},
        "hyperparams": {"lr": args.lr, "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "freeze_layers": args.freeze_layers,
                        "group_key": args.group_key,
                        "holdout_frac": args.holdout_frac,
                        "class_weights": class_weights is not None,
                        "seed": args.seed},
        "metrics_before": before,
        "metrics_after": after,
    }, indent=2))

    # qwk change per batch; flag forgetting on the old batch
    print("\nqwk, before -> after:")
    for name in eval_rows:
        d = after[name]["qwk"] - before[name]["qwk"]
        print(f"  {name:<4} {before[name]['qwk']:.3f} -> {after[name]['qwk']:.3f}  ({d:+.3f})")
    if "old" in after and after["old"]["qwk"] < before["old"]["qwk"] - 0.02:
        print("  ! the old batch got worse - that is forgetting, not noise. Lower "
              "--lr, fewer --epochs, or --freeze-layers 6.")

    print(f"\nModel saved to {args.out}")
    print("Re-score EVERY series with this model before comparing conditions - counts "
          "from the parent model and this one are not comparable.")


if __name__ == "__main__":
    main()
