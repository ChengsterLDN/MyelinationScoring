"""
Stage 4: score every pillar on every slice, then score a per-site projection (MIP) of the slices that showed signal, and
count.


Default rule agg_maxmip = max(per-slice max, MIP class). 
The MIP is dulled back to single-slice intensity (default: histogram match) so brightness alone can't inflate it. 
Outputs as the standard scorer, plus MIP columns and qc/mips/ comparisons.
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
# Aggregation rules available on the command line
AGG_RULES = ["max", "maxmip", "consec2", "consec3", "meanprob", "q90"]
# Ring colour per predicted class in the QC overlay
CLASS_COLOURS = {0: (90, 90, 90), 1: (60, 140, 255),
                 2: (255, 190, 40), 3: (60, 220, 90)}
MIP_PROJ = ["max", "topk"]                     # projection methods
MIP_DULL = ["match", "gamma", "scale", "none"]  # intensity-dulling methods

# Aggregation rules

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


# maxmip is not in here: it needs the MIP pass and is folded in afterwards.
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

# Projection and dulling  (MIP path only - per-slice scoring is untouched)


def _vmax(ref, st: np.ndarray) -> float:
    """Full-scale value for the crop dtype."""
    a = np.asarray(ref)
    # Integer images: dtype maximum (e.g. 255)
    if np.issubdtype(a.dtype, np.integer):
        return float(np.iinfo(a.dtype).max)
    # Float images: guess 0-1 or 0-255 from the data
    return 1.0 if float(st.max()) <= 1.0 else 255.0


def _as_like(ref, arr: np.ndarray):
    """Rebuild arr in the container and dtype build_crop handed us."""
    a = np.asarray(ref)
    # Round and clip back into integer range if needed
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        arr = np.clip(np.rint(arr), info.min, info.max).astype(a.dtype)
    else:
        arr = arr.astype(a.dtype)
    # Return a PIL image if that's what came in
    if isinstance(ref, Image.Image):
        return Image.fromarray(arr, mode=ref.mode)
    return arr


def _match_1(m: np.ndarray, st: np.ndarray) -> np.ndarray:
    """Histogram-match m to the average single-slice histogram of st:
    which pixels are bright is kept, how bright they are is not."""
    # Reference: sorted pixel values of each slice, averaged across slices
    ref = np.sort(st.reshape(len(st), -1), axis=1).mean(axis=0)
    flat = m.ravel()
    out = np.empty_like(flat, dtype=np.float32)
    # Give the k-th darkest MIP pixel the k-th reference value
    out[np.argsort(flat, kind="stable")] = ref
    return out.reshape(m.shape)


def match_to_slices(m: np.ndarray, st: np.ndarray) -> np.ndarray:
    """Histogram-match, channel by channel for colour images."""
    if m.ndim == 3:                                   # per channel
        out = np.empty_like(m, dtype=np.float32)
        for ch in range(m.shape[2]):
            out[..., ch] = _match_1(m[..., ch], st[..., ch])
        return out
    return _match_1(m, st)


def project_crops(crops, proj="max", dull="match", gamma=1.6, scale=0.7, topk=2):
    """Project aligned crops of one site (same x,y on different slices) to a
    single dulled crop."""
    # Stack crops: (n_slices, H, W[, C])
    st = np.stack([np.asarray(c).astype(np.float32) for c in crops])

    # Projection: mean of top-k brightest values, or plain maximum
    if proj == "topk" and len(st) > 1:
        k = min(max(int(topk), 1), len(st))
        m = np.sort(st, axis=0)[-k:].mean(axis=0)
    else:
        m = st.max(axis=0)

    # Dulling: bring intensities back towards single-slice levels
    if dull == "match" and len(st) > 1:
        m = match_to_slices(m, st)
    elif dull == "gamma":
        v = _vmax(crops[0], st)
        m = v * np.clip(m / v, 0.0, 1.0) ** float(gamma)
    elif dull == "scale":
        m = m * float(scale)

    # Same type/dtype as the input crops
    return _as_like(crops[0], m)


def normalise_stack(raws, p_lo=1.0, p_hi=99.5, step=4) -> list:
    """8-bit stack normalised against shared percentiles, not per slice."""
    # Subsample every step-th pixel of every slice to estimate percentiles
    sample = np.concatenate([np.asarray(r)[::step, ::step].ravel() for r in raws])
    lo, hi = np.percentile(sample.astype(np.float32), [p_lo, p_hi])
    if hi <= lo:
        hi = lo + 1.0
    # Apply the same stretch to every slice
    out = []
    for r in raws:
        a = (np.asarray(r).astype(np.float32) - lo) / (hi - lo)
        out.append((np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8))
    return out


# Inference

@torch.no_grad()
def score_crops(model, processor, crops, device) -> np.ndarray:
    """(len(crops), 4) probability array for a batch of ready-made crops."""
    enc = processor(images=crops, return_tensors="pt").to(device)
    logits = model(**enc).logits
    return torch.softmax(logits, -1).float().cpu().numpy()


@torch.no_grad()
def score_field(model, processor, field8, all_xy, pitch, geo, device, batch_size):
    """(n_sites, 4) probability array for one slice."""
    out = np.zeros((len(all_xy), NUM_LABELS), dtype=np.float32)
    # Build and score crops in batches
    for start in range(0, len(all_xy), batch_size):
        idxs = range(start, min(start + batch_size, len(all_xy)))
        crops = [build_crop(field8, all_xy, i, pitch, **geo) for i in idxs]
        out[start:start + len(crops)] = score_crops(model, processor, crops, device)
    return out


def save_mip_qc(path: Path, ref, mip) -> None:
    """best single slice | projection, side by side."""
    def rgb8(x):
        # Convert any crop to an 8-bit RGB array
        a = np.asarray(x)
        if np.issubdtype(a.dtype, np.floating):
            a = a * 255.0 if a.max() <= 1.0 else a
        a = np.clip(a, 0, 255).astype(np.uint8)
        return np.stack([a] * 3, -1) if a.ndim == 2 else a[..., :3]

    # Concatenate with a 4-px white gap between them
    a, b = rgb8(ref), rgb8(mip)
    gap = np.full((a.shape[0], 4, 3), 255, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([a, gap, b], axis=1)).save(path)


@torch.no_grad()
def score_site_mips(model, processor, fields, preds, all_xy, pitch, geo,
                    device, batch_size, proj="max", dull="match", gamma=1.6,
                    scale=0.7, topk=2, qc_dir: Path | None = None, qc_limit=0):
    """Project and score one crop per site, over the slices where that site
    scored > 0. Sites with no such slice stay class 0 with n_used = 0.

    Returns (probs (n_sites, 4), n_used (n_sites,)).
    """
    n_sites = len(all_xy)
    # Default: every site certain class 0
    probs = np.zeros((n_sites, NUM_LABELS), dtype=np.float32)
    probs[:, 0] = 1.0
    n_used = np.zeros(n_sites, dtype=int)

    pend: list[tuple] = []        # (site_id, mip crop, best slice crop, slice max)
    n_qc = 0                      # QC images saved so far

    def flush() -> None:
        """Score all pending MIP crops in one batch."""
        nonlocal n_qc
        if not pend:
            return
        p = score_crops(model, processor, [c for _, c, _, _ in pend], device)
        for k, (sid, mip, ref, smax) in enumerate(pend):
            probs[sid] = p[k]
            cls = int(p[k].argmax())
            # Save a QC comparison when the MIP raised the class
            if qc_dir is not None and n_qc < qc_limit and cls > smax:
                save_mip_qc(qc_dir / f"site{sid:04d}_slice{smax}_mip{cls}.png",
                            ref, mip)
                n_qc += 1
        pend.clear()

    for sid in range(n_sites):
        # Slices where this site scored non-zero
        keep = np.nonzero(preds[:, sid] > 0)[0]
        n_used[sid] = len(keep)
        if len(keep) == 0:
            continue
        # Crop this site on each kept slice
        crops = [build_crop(fields[n], all_xy, sid, pitch, **geo) for n in keep]
        # Queue: MIP crop, the best single-slice crop, and its class
        pend.append((
            sid,
            project_crops(crops, proj=proj, dull=dull, gamma=gamma,
                          scale=scale, topk=topk),
            crops[int(np.argmax(preds[keep, sid]))],
            int(preds[keep, sid].max()),
        ))
        # Score once a full batch has accumulated
        if len(pend) >= batch_size:
            flush()
        print(f"    MIP {sid + 1}/{n_sites}", end="\r")
    flush()  # score any remainder
    print(" " * 40, end="\r")
    return probs, n_used


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
    # --- command-line options ---
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=config.ROOT_DIR)
    p.add_argument("--model", type=Path, default=config.MODEL_DIR)
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    # maxmip, not config.AGG_RULE: the MIP-augmented max is the headline rule
    # now. Pass --agg max (or the config value) to get the old behaviour.
    p.add_argument("--agg", default="maxmip", choices=AGG_RULES)
    p.add_argument("--no-mip", action="store_true",
                   help="skip the per-site MIP pass; agg_maxmip becomes agg_max")
    p.add_argument("--mip-proj", default="max", choices=MIP_PROJ,
                   help="per-pixel max, or mean of the k brightest slices")
    p.add_argument("--mip-topk", type=int, default=2, help="k for --mip-proj topk")
    p.add_argument("--mip-dull", default="match", choices=MIP_DULL,
                   help="how to pull MIP intensities back to single-slice levels")
    p.add_argument("--mip-gamma", type=float, default=1.6,
                   help=">1 darkens, for --mip-dull gamma")
    p.add_argument("--mip-scale", type=float, default=0.7,
                   help="multiplier for --mip-dull scale")
    p.add_argument("--mip-global-norm", action="store_true",
                   help="renormalise the sub-stack on shared percentiles for the "
                        "MIP instead of using the per-slice stretch (2nd copy in RAM)")
    p.add_argument("--mip-qc", type=int, default=12,
                   help="save up to N slice|MIP comparisons per series (0 = off)")
    p.add_argument("--denominator", default=config.PRIMARY_DENOMINATOR,
                   choices=["detected", "all"])
    p.add_argument("--z-range", default=None, help="e.g. 5:45")
    p.add_argument("--batch-size", type=int, default=config.INFER_BATCH)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.no_mip and args.agg == "maxmip":
        print("! --no-mip with --agg maxmip: counts will be identical to max.")

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

    # Split into build_crop kwargs and z-smoothing
    geo = {k: pre[k] for k in
           ("context_pitches", "out_size", "pillar_r_frac", "slide_at_edges")
           if k in pre}
    z_smooth = pre.get("z_smooth", config.Z_SMOOTH)

    # z-range: config default, overridden by "--z-range lo:hi"
    z_range = config.Z_RANGE
    if args.z_range:
        lo, hi = args.z_range.split(":")
        z_range = (int(lo), int(hi))

    # Report device and MIP settings
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   Model: {args.model}   Aggregation: {args.agg}")
    if args.no_mip:
        print("MIP pass: off")
    else:
        print(f"MIP: proj={args.mip_proj}"
              f"{f'(k={args.mip_topk})' if args.mip_proj == 'topk' else ''}  "
              f"dull={args.mip_dull}"
              f"{f'(g={args.mip_gamma})' if args.mip_dull == 'gamma' else ''}"
              f"{f'(s={args.mip_scale})' if args.mip_dull == 'scale' else ''}  "
              f"global_norm={args.mip_global_norm}")

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
        # kept only when the MIP pass needs to revisit slices
        fields: list = [None] * len(sel_z)
        raws: list = [None] * len(sel_z)

        # Score every site on every selected slice
        for n, zi in enumerate(sel_z):
            raw = load_slice_smoothed(refs, zi, z_smooth)
            field8 = normalise(raw)
            probs[n] = score_field(model, processor, field8, all_xy, pitch,
                                   geo, device, args.batch_size)
            # Keep raw or normalised slice for the MIP pass
            if not args.no_mip:
                if args.mip_global_norm:
                    raws[n] = raw
                else:
                    fields[n] = field8
            last_field = field8
            print(f"    z={zi:>3}  ({n + 1}/{len(sel_z)})", end="\r")
        print(" " * 40, end="\r")

        # Most likely class per slice per site
        preds = probs.argmax(axis=2)
        src = [s.source for s in table.sites]

        # --- per-site MIP over the slices that scored non-zero ---
        if args.no_mip:
            # Placeholder: everything class 0, no slices used
            mip_probs = np.zeros((len(table), NUM_LABELS), dtype=np.float32)
            mip_probs[:, 0] = 1.0
            n_mip = np.zeros(len(table), dtype=int)
        else:
            # Optionally renormalise the whole sub-stack together first
            if args.mip_global_norm:
                fields = normalise_stack(raws)
                raws = []
            mip_probs, n_mip = score_site_mips(
                model, processor, fields, preds, all_xy, pitch, geo,
                device, args.batch_size, proj=args.mip_proj, dull=args.mip_dull,
                gamma=args.mip_gamma, scale=args.mip_scale, topk=args.mip_topk,
                qc_dir=args.out / "qc" / "mips" / series.name,
                qc_limit=max(args.mip_qc, 0))
        fields, raws = [], []                          # release the stack
        mip_pred = mip_probs.argmax(axis=1).astype(int)

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
        # the MIP only ever raises a pillar, and only if it beats the max
        agg["maxmip"] = np.maximum(agg["max"], mip_pred).astype(int)
        upgraded = mip_pred > agg["max"]

        # One row per pillar: every rule plus MIP details
        write_csv(args.out / "per_pillar" / f"{series.name}.csv", [
            {"site_id": s.site_id, "x": s.x, "y": s.y, "source": s.source,
             "grid_i": s.grid_i, "grid_j": s.grid_j, "n_slices": len(sel_z),
             **{f"agg_{r}": int(agg[r][s.site_id]) for r in AGG_RULES},
             "mip_class": int(mip_pred[s.site_id]),
             "n_mip_slices": int(n_mip[s.site_id]),
             "mip_conf": round(float(mip_probs[s.site_id].max()), 4),
             "mip_upgraded": bool(upgraded[s.site_id]),
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
        # Mask of sites included in the chosen denominator
        keep = det if args.denominator == "detected" else np.ones(n_all, bool)

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
            "mip_proj": "off" if args.no_mip else args.mip_proj,
            "mip_topk": args.mip_topk if args.mip_proj == "topk" else "",
            "mip_dull": "off" if args.no_mip else args.mip_dull,
            "mip_global_norm": bool(args.mip_global_norm and not args.no_mip),
            "n_mip_scored": int((n_mip[keep] > 0).sum()),
            "n_mip_upgraded": int(upgraded[keep].sum()),
            **{f"class_{k}_{CLASS_NAMES[k]}": base[k] for k in range(NUM_LABELS)},
            **{f"pct_class_{k}": round(100 * base[k] / max(n_base, 1), 2)
               for k in range(NUM_LABELS)},
            **{f"det_class_{k}": c_det[k] for k in range(NUM_LABELS)},
            **{f"all_class_{k}": c_all[k] for k in range(NUM_LABELS)},
        }
        # Add counts under every other rule for comparison
        for rule in AGG_RULES:
            if rule != args.agg:
                c = counts_of(agg[rule][keep])
                row.update({f"alt_{rule}_class_{k}": c[k] for k in range(NUM_LABELS)})
        series_rows.append(row)

        print(f"  [{args.agg}, {args.denominator}] " +
              "  ".join(f"{CLASS_NAMES[k]}={base[k]}" for k in range(NUM_LABELS)))

        # Report how much the MIP changed; warn if it drives >10% of pillars
        if not args.no_mip:
            n_up = int(upgraded[keep].sum())
            print(f"  MIP: {int((n_mip[keep] > 0).sum())} pillars projected, "
                  f"{n_up} raised above the per-slice max")
            if n_up > 0.1 * max(n_base, 1):
                print(f"  ! the MIP is driving {100 * n_up / max(n_base, 1):.0f}% of "
                      f"the pillars - check qc/mips/{series.name}/ before trusting "
                      f"the headline count")

        # Warn if the chosen rule and consec2 disagree on >10% of pillars
        c2 = counts_of(agg["consec2"][keep])
        drift = sum(abs(base[k] - c2[k]) for k in range(NUM_LABELS)) / 2
        if args.agg in ("max", "maxmip") and drift > 0.1 * max(n_base, 1):
            print(f"  ! {args.agg} and consec2 disagree on ~{int(drift)} pillars "
                  f"({100 * drift / n_base:.0f}%) - likely single-slice misfires")

    # Write per-slice and per-series tables
    write_csv(args.out / "per_stack_counts.csv", stack_rows)
    write_csv(args.out / "per_series_counts.csv", series_rows)

    # TOTAL row summed over all series 
    total = {"series": "TOTAL", "agg_rule": args.agg,
             "denominator": args.denominator,
             "mip_proj": "off" if args.no_mip else args.mip_proj,
             "mip_dull": "off" if args.no_mip else args.mip_dull}
    for k in ("n_detected", "n_inferred", "n_off_lattice", "n_all_sites", "n_slices",
              "n_mip_scored", "n_mip_upgraded"):
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
             "n_slices", "agg_rule", "denominator", "mip_proj", "mip_dull",
             "n_mip_scored", "n_mip_upgraded"] +
            [f"class_{k}_{CLASS_NAMES[k]}" for k in range(NUM_LABELS)] +
            [f"pct_class_{k}" for k in range(NUM_LABELS)])
    write_csv(args.out / "directory_summary.csv",
              [{k: r.get(k, "") for k in cols} for r in series_rows] +
              [{k: total.get(k, "") for k in cols}])

    # Console table of the same
    print(f"\n{'series':<30}{'sites':>7}" +
          "".join(f"{CLASS_NAMES[k]:>14}" for k in range(NUM_LABELS)) +
          f"{'mip_up':>9}")
    for r in series_rows + [total]:
        n = r.get("n_detected" if args.denominator == "detected" else "n_all_sites", "")
        print(f"{r['series']:<30}{n:>7}" +
              "".join(f"{r[f'class_{k}_{CLASS_NAMES[k]}']:>14}"
                      for k in range(NUM_LABELS)) +
              f"{r.get('n_mip_upgraded', ''):>9}")
    print(f"\nResults -> {args.out}")


if __name__ == "__main__":
    main()
