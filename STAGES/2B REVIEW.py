"""
Re-open already-labelled crops, filtered, to audit and correct

    python -m review_labels --class 2 3 [--source from_mask]
                            [--series Dose10-1] [--shuffle]

"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

import config
from label_crops import Labeller


def parse_args() -> argparse.Namespace:
    """Filters for which labelled crops to review."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", type=Path, default=config.CROPS_DIR)
    p.add_argument("--class", dest="classes", type=int, nargs="+",
                   choices=range(config.NUM_LABELS), metavar="N",
                   help="Only show crops currently labelled N (one or more)")
    p.add_argument("--source", nargs="+", default=None,
                   help="Only show sites of this kind: detected, off_lattice, "
                        "from_mask, inferred")
    p.add_argument("--series", nargs="+", default=None,
                   help="Only show these series")
    p.add_argument("--shuffle", action="store_true",
                   help="Randomise order. Use when auditing a class - manifest "
                        "order groups crops by series and z, which makes "
                        "neighbouring crops look artificially consistent.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Both the manifest and existing labels are needed
    manifest_path = args.crops / "manifest.csv"
    labels_path = args.crops / "labels.csv"
    if not labels_path.exists():
        raise SystemExit(f"No labels at {labels_path}\nRun: python -m label_crops")

    # Load crop metadata and current labels
    with open(manifest_path, newline="") as f:
        all_rows = list(csv.DictReader(f))
    with open(labels_path, newline="") as f:
        labels = {r["filepath"]: int(r["label"]) for r in csv.DictReader(f)}

    if not labels:
        raise SystemExit(f"{labels_path} is empty - nothing to review")

    # Rows the manifest doesn't know about (labels from an older run) still
    # need to survive the rewrite, so carry them as metadata.
    known = {r["filepath"] for r in all_rows}
    for fp in labels:
        if fp not in known:
            all_rows.append({"filepath": fp, "series": "", "site_id": "",
                             "source": "", "slice_key": ""})

    # Start from labelled crops only
    rows = [r for r in all_rows if r["filepath"] in labels]

    # Apply the class / source / series filters
    if args.classes:
        rows = [r for r in rows if labels[r["filepath"]] in set(args.classes)]
    if args.source:
        rows = [r for r in rows if r.get("source", "") in set(args.source)]
    if args.series:
        rows = [r for r in rows if r.get("series", "") in set(args.series)]

    # Nothing matched: print what does exist to help pick a filter
    if not rows:
        print("Nothing matches that filter. Labels currently on file:")
        by_class = Counter(labels.values())
        for k in range(config.NUM_LABELS):
            print(f"  class {k} ({config.CLASS_NAMES[k]}): {by_class.get(k, 0)}")
        by_source = Counter(r.get("source", "") for r in all_rows
                            if r["filepath"] in labels)
        print(f"  sources: {dict(by_source)}")
        return

    # Remember what each crop came in as, so the GUI can flag edits.
    for r in rows:
        r["_original_label"] = labels[r["filepath"]]

    # Optional random order
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rows = [rows[i] for i in rng.permutation(len(rows))]

    # Describe the filter for the console and window title
    before = Counter(labels[r["filepath"]] for r in rows)
    filt = []
    if args.classes:
        filt.append("class " + "/".join(str(c) for c in args.classes))
    if args.source:
        filt.append("source " + "/".join(args.source))
    if args.series:
        filt.append("series " + "/".join(args.series))
    scope = ", ".join(filt) if filt else "all labelled crops"

    print(f"Reviewing {len(rows)} crops  ({scope})")
    print("  in: " + "  ".join(
        f"{k}={before.get(k, 0)}" for k in range(config.NUM_LABELS) if before.get(k)))

    # Open the labelling GUI on the filtered rows, from the first one
    app = Labeller(args.crops, rows, labels_path, all_rows=all_rows,
                   start_at_unlabelled=False,
                   title=f"Review - {scope}")
    app.win.mainloop()

    # Find crops whose label changed this session
    after = {r["filepath"]: app.labels.get(r["filepath"]) for r in rows}
    changed = [(r, after[r["filepath"]]) for r in rows
               if after[r["filepath"]] is not None
               and after[r["filepath"]] != r["_original_label"]]

    # Summarise changes as old -> new counts
    print(f"\n{len(changed)} of {len(rows)} reviewed crops changed")
    if changed:
        moves = Counter((r["_original_label"], new) for r, new in changed)
        for (old, new), n in sorted(moves.items()):
            print(f"  {old} -> {new}: {n}")

    # Full label distribution after review
    total = Counter(app.labels.values())
    print("Full label set is now: " + "  ".join(
        f"{k}={total.get(k, 0)} ({config.CLASS_NAMES[k]})"
        for k in range(config.NUM_LABELS)))
    print(f"Written to {labels_path}")


if __name__ == "__main__":
    main()
