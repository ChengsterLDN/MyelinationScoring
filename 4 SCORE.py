"""
Stage 4: score every pillar on every slice, aggregate over z, and count.

Writes to RESULTS_DIR: sites/, per_slice/, per_pillar/ (every aggregation rule side by side), qc/ overlays, per_stack_counts.csv, per_series_counts.csv and directory_summary.csv. 
Counts are reported for both "detected" and "all" (incl. inferred) denominators. 

Note: "max" can only push counts up; compare with consec2.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForImageClassification

import config
from wrap_common import (
    build_crop, build_site_table, describe_series, discover_series,
    image_shape, list_slices, load_slice_smoothed, normalise, select_slices,
    site_table_rows, write_csv,
)

NUM_LABELS = config.NUM_LABELS
CLASS_NAMES = config.CLASS_NAMES
# Ways of collapsing per-slice predictions into one class per pillar
AGG_RULES = ["max", "consec2", "consec3", "meanprob", "q90"]
# Ring colour per predicted class in the QC overlay
CLASS_COLOURS = {0: (90, 90, 90), 1: (60, 140, 255),
                 2: (255, 190, 40), 3: (60, 220, 90)}



def _consec(preds: np.ndarray, k: int) -> int:
    """Highest class held for k consecutive slices."""
    # Try classes from highest down; return the first sustained for k slices
    for c in range(NUM_LABELS - 1, 0, -1):
        run = 0
        for p in preds:
            run = run + 1 if p >= c else 0
            if run >= k:
                return c
    return 0


# Each rule takes (per-slice preds, per-slice probs) for one site
AGG_FUNCS = {
    "max":      lambda pr, pb: int(pr.max()),                       # best slice
    "consec2":  lambda pr, pb: _consec(pr, 2),                      # held 2 slices
    "consec3":  lambda pr, pb: _consec(pr, 3),                      # held 3 slices
    "meanprob": lambda pr, pb: int(np.argmax(pb.mean(axis=0))),     # mean probability
    "q90":      lambda pr, pb: int(np.floor(np.percentile(pr, 90))),  # 90th percentile
}


def counts_of(classes) -> dict[int, int]:
    """Number of sites in each class (zeros included)."""
    c = Counter(int(v) for v in classes)
    return {k: c.get(k, 0) for k in range(NUM_LABELS)}

# Inference

@torch.no_grad()
def score_field(model, processor, field8, all_xy, pitch, geo, device, batch_size):
    """(n_sites, 4) probability array for one slice."""
    out = np.zeros((len(all_xy), NUM_LABELS), dtype=np.float32)
    # Process sites in batches
    for start in range(0, len(all_xy), batch_size):
        idxs = range(start, min(start + batch_size, len(all_xy)))
        # Build crops with the model's own geometry
        crops = [build_crop(field8, all_xy, i, pitch, **geo) for i in idxs]
        enc = processor(images=crops, return_tensors="pt").to(device)
        logits = model(**enc).logits
        # Softmax -> probabilities, back to numpy
        out[start:start + len(crops)] = torch.softmax(logits, -1).float().cpu().numpy()
    return out


def qc_overlay(field8, table, classes, path: Path, pitch: float) -> None:
    """Field image with a coloured ring on each site, sized to the pillar."""
    img = Image.merge("RGB", [Image.fromarray(field8)] * 3)
    draw = ImageDraw.Draw(img)
    r = max(4.0, pitch * config.PILLAR_R_FRAC * 1.35)
    for s, c in zip(table.sites, classes):
        col = CLASS_COLOURS[int(c)]
        width = 3 if s.source == "detected" else 6      # thick = inferred/off-lattice
        draw.ellipse([s.x - r, s.y - r, s.x + r, s.y + r], outline=col, width=width)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

def main() -> None:
    # --- command-line overrides; defaults from config.py ---
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=config.ROOT_DIR)
    p.add_argument("--model", type=Path, default=config.MODEL_DIR)
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    p.add_argument("--agg", default=config.AGG_RULE, choices=AGG_RULES)
    p.add_argument("--denominator", default=config.PRIMARY_DENOMINATOR,
                   choices=["detected", "all"])
    p.add_argument("--z-range", default=None, help="e.g. 5:45")
    p.add_argument("--batch-size", type=int, default=config.INFER_BATCH)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    # Crop geometry saved with the model (fallback: config)
    pre_path = args.model / "preproc.json"
    if pre_path.exists():
        pre = json.loads(pre_path.read_text())
    else:
        pre = {"context_pitches": config.CONTEXT_PITCHES, "out_size": config.OUT_SIZE,
               "pillar_r_frac": config.PILLAR_R_FRAC,
               "slide_at_edges": config.SLIDE_AT_EDGES, "z_smooth": config.Z_SMOOTH}
        print("! No preproc.json with the model - using config defaults. If the "
              "crops were made with different settings, scores will be wrong.")

    # Split into build_crop kwargs and the z-smoothing setting
    geo = {k: pre[k] for k in
           ("context_pitches", "out_size", "pillar_r_frac", "slide_at_edges")
           if k in pre}
    z_smooth = pre.get("z_smooth", config.Z_SMOOTH)

    # z-range: config default, overridden by "--z-range lo:hi"
    z_range = config.Z_RANGE
    if args.z_range:
        lo, hi = args.z_range.split(":")
        z_range = (int(lo), int(hi))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   Model: {args.model}   Aggregation: {args.agg}")

    # Load trained model in eval mode
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).to(device).eval()

    stack_rows: list[dict] = []   # one row per slice
    series_rows: list[dict] = []  # one row per series/stack

    for series in discover_series(args.root):
        # Load slices and build this stack's site registry
        refs = list_slices(series.mbp_dir)
        shape = image_shape(refs)
        table = build_site_table(series.coords(), shape)
        sel_z = select_slices(refs, z_range)

        describe_series(series, table, len(refs), len(sel_z))
        write_csv(args.out / "sites" / f"{series.name}.csv", site_table_rows(table))

        all_xy = table.xy
        pitch = table.lattice.pitch
        # probs[slice, site, class]
        probs = np.zeros((len(sel_z), len(table), NUM_LABELS), dtype=np.float32)
        last_field = None

        # Score every site on every selected slice
        for n, zi in enumerate(sel_z):
            field8 = normalise(load_slice_smoothed(refs, zi, z_smooth))
            probs[n] = score_field(model, processor, field8, all_xy, pitch,
                                   geo, device, args.batch_size)
            last_field = field8
            print(f"    z={zi:>3}  ({n + 1}/{len(sel_z)})", end="\r")
        print(" " * 40, end="\r")

        # Most likely class per slice per site
        preds = probs.argmax(axis=2)
        src = [s.source for s in table.sites]

        #  per-slice detail 
        write_csv(args.out / "per_slice" / f"{series.name}.csv", [
            {"site_id": sid, "source": src[sid], "slice_idx": zi,
             "slice_key": refs[zi].key, "pred": int(preds[n, sid]),
             **{f"p{c}": round(float(probs[n, sid, c]), 4) for c in range(NUM_LABELS)}}
            for n, zi in enumerate(sel_z) for sid in range(len(table))
        ])

        # Class counts per slice, for both denominators
        det = table.detected_mask
        for n, zi in enumerate(sel_z):
            c_all = counts_of(preds[n])
            c_det = counts_of(preds[n][det])
            stack_rows.append({
                "series": series.name, "slice_idx": zi, "slice_key": refs[zi].key,
                **{f"det_class_{k}": c_det[k] for k in range(NUM_LABELS)},
                **{f"all_class_{k}": c_all[k] for k in range(NUM_LABELS)},
            })

        # aggregate over z, every rule 
        agg = {rule: np.array([fn(preds[:, s], probs[:, s, :])
                               for s in range(len(table))], dtype=int)
               for rule, fn in AGG_FUNCS.items()}

        # One row per pillar with every rule's result
        write_csv(args.out / "per_pillar" / f"{series.name}.csv", [
            {"site_id": s.site_id, "x": s.x, "y": s.y, "source": s.source,
             "grid_i": s.grid_i, "grid_j": s.grid_j, "n_slices": len(sel_z),
             **{f"agg_{r}": int(agg[r][s.site_id]) for r in AGG_RULES},
             "mean_conf": round(float(probs[:, s.site_id, :].max(axis=1).mean()), 4)}
            for s in table.sites
        ])

        # QC overlay coloured by the chosen rule
        if config.WRITE_QC_OVERLAY and last_field is not None:
            qc_overlay(last_field, table, agg[args.agg],
                       args.out / "qc" / f"{series.name}.png", pitch)

        # Headline counts under the chosen rule and denominator
        c_det = counts_of(agg[args.agg][det])
        c_all = counts_of(agg[args.agg])
        n_det, n_all = int(det.sum()), len(table)
        base = c_det if args.denominator == "detected" else c_all
        n_base = n_det if args.denominator == "detected" else n_all

        row = {
            "series": series.name,
            "n_detected": n_det,
            "n_inferred": table.counts_by_source()["inferred"],
            "n_off_lattice": table.counts_by_source()["off_lattice"],
            "n_all_sites": n_all,
            "lattice_ok": table.lattice_ok,
            "pitch_px": round(pitch, 1),
            "rotation_deg": round(table.lattice.rotation_deg, 2),
            "n_slices": len(sel_z),
            "agg_rule": args.agg,
            "denominator": args.denominator,
            **{f"class_{k}_{CLASS_NAMES[k]}": base[k] for k in range(NUM_LABELS)},
            **{f"pct_class_{k}": round(100 * base[k] / max(n_base, 1), 2)
               for k in range(NUM_LABELS)},
            **{f"det_class_{k}": c_det[k] for k in range(NUM_LABELS)},
            **{f"all_class_{k}": c_all[k] for k in range(NUM_LABELS)},
        }
        # Add counts under every other rule for comparison
        for rule in AGG_RULES:
            if rule != args.agg:
                c = counts_of(agg[rule][det] if args.denominator == "detected"
                              else agg[rule])
                row.update({f"alt_{rule}_class_{k}": c[k] for k in range(NUM_LABELS)})
        series_rows.append(row)

        print(f"  [{args.agg}, {args.denominator}] " +
              "  ".join(f"{CLASS_NAMES[k]}={base[k]}" for k in range(NUM_LABELS)))

        # Warn if max and consec2 disagree on >10% of pillars (single-slice misfires)
        c2 = counts_of(agg["consec2"][det] if args.denominator == "detected"
                       else agg["consec2"])
        drift = sum(abs(base[k] - c2[k]) for k in range(NUM_LABELS)) / 2
        if args.agg == "max" and drift > 0.1 * max(n_base, 1):
            print(f"  ! max and consec2 disagree on ~{int(drift)} pillars "
                  f"({100 * drift / n_base:.0f}%) - likely single-slice misfires")

    # Write per-slice and per-series tables
    write_csv(args.out / "per_stack_counts.csv", stack_rows)
    write_csv(args.out / "per_series_counts.csv", series_rows)

    # --- TOTAL row summed over all series ---
    total = {"series": "TOTAL", "agg_rule": args.agg,
             "denominator": args.denominator}
    for k in ("n_detected", "n_inferred", "n_off_lattice", "n_all_sites", "n_slices"):
        total[k] = sum(r[k] for r in series_rows)
    for k in range(NUM_LABELS):
        total[f"class_{k}_{CLASS_NAMES[k]}"] = sum(
            r[f"class_{k}_{CLASS_NAMES[k]}"] for r in series_rows)
    n_base = total["n_detected"] if args.denominator == "detected" else total["n_all_sites"]
    for k in range(NUM_LABELS):
        total[f"pct_class_{k}"] = round(
            100 * total[f"class_{k}_{CLASS_NAMES[k]}"] / max(n_base, 1), 2)

    # Compact summary: selected columns, one row per series + TOTAL
    cols = (["series", "n_detected", "n_inferred", "n_off_lattice", "n_all_sites",
             "n_slices", "agg_rule", "denominator"] +
            [f"class_{k}_{CLASS_NAMES[k]}" for k in range(NUM_LABELS)] +
            [f"pct_class_{k}" for k in range(NUM_LABELS)])
    write_csv(args.out / "directory_summary.csv",
              [{k: r.get(k, "") for k in cols} for r in series_rows] +
              [{k: total.get(k, "") for k in cols}])

    # Console table of the same
    print(f"\n{'series':<30}{'sites':>7}" +
          "".join(f"{CLASS_NAMES[k]:>14}" for k in range(NUM_LABELS)))
    for r in series_rows + [total]:
        n = r.get("n_detected" if args.denominator == "detected" else "n_all_sites", "")
        print(f"{r['series']:<30}{n:>7}" +
              "".join(f"{r[f'class_{k}_{CLASS_NAMES[k]}']:>14}"
                      for k in range(NUM_LABELS)))
    print(f"\nResults -> {args.out}")


if __name__ == "__main__":
    main()
