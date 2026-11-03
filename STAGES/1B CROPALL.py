"""
Stage 1 ALL: sample per-pillar context crops for hand labelling.

CROPS ALL

Runs over every series/stack. Writes to CROPS_DIR: manifest.csv (sorted
into labelling order), preproc.json (crop geometry), sites/<series>.csv
and the crop PNGs. Lattice geometry is measured per stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config
from wrap_common import (
    build_crop, build_site_table, describe_series, discover_series,
    image_shape, list_slices, load_slice_smoothed, normalise, preproc_dict,
    select_slices, site_table_rows, write_csv,
)


def parse_args() -> argparse.Namespace:
    """Command-line overrides; every default comes from config.py."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=config.ROOT_DIR)
    p.add_argument("--out", type=Path, default=config.CROPS_DIR)
    p.add_argument("--pillars-per-series", type=int, default=config.PILLARS_PER_SERIES,
                   help="how many sites to sample per series; 0 or a negative "
                        "number means every site (same as --all-sites)")
    p.add_argument("--all-sites", action="store_true",
                   help="crop EVERY site in the registry, not a sample. The "
                        "stratified sample still sets the labelling order.")
    p.add_argument("--slices-per-pillar", type=int, default=config.SLICES_PER_PILLAR,
                   help="z-slices per pillar, spread evenly through the usable "
                        "range; 0 or a negative number means every slice "
                        "(same as --all-slices)")
    p.add_argument("--all-slices", action="store_true",
                   help="crop EVERY usable z-slice, not an even spread. This is "
                        "the setting that multiplies fastest - 40 slices instead "
                        "of 6 is 7x the crops.")
    p.add_argument("--z-range", type=str, default=None, help="e.g. 5:45")
    p.add_argument("--seed", type=int, default=config.SAMPLE_SEED)
    return p.parse_args()


def stratified_sample(table, n_wanted: int, rng: np.random.Generator) -> list[int]:
    """Choose which sites to label FIRST, stratified by source so off-lattice
    and inferred sites appear in training at their inference-time rate."""
    # Group site ids by source (detected / off_lattice / inferred)
    by_source: dict[str, list[int]] = {}
    for s in table.sites:
        by_source.setdefault(s.source, []).append(s.site_id)

    # Can't sample more sites than exist
    n_wanted = min(n_wanted, len(table))
    picked: list[int] = []
    total = len(table)

    # Proportional share per source, at least one each
    for source, ids in by_source.items():
        share = max(1, int(round(n_wanted * len(ids) / total)))
        share = min(share, len(ids))
        picked += rng.choice(ids, size=share, replace=False).tolist()

    # Rounding can overshoot; trim back at random
    if len(picked) > n_wanted:
        picked = rng.choice(picked, size=n_wanted, replace=False).tolist()
    return sorted(int(p) for p in picked)


def select_sites(table, n_wanted: int, all_sites: bool,
                  rng: np.random.Generator) -> tuple[list[int], dict[int, int]]:
    """Which sites to crop, and a labelling priority for each.

    The stratified sample is priority 0. With all_sites, the remaining
    sites follow in shuffled (not lattice) order, so stopping early still
    leaves a representative labelled set.

    Returns (site_ids to crop, site_id -> priority).
    """
    # Stratified sample always forms the first labelling batch
    first = stratified_sample(table, n_wanted, rng)
    if not all_sites:
        return first, {sid: 0 for sid in first}

    # Remaining sites, shuffled, get priorities 1, 2, 3, ...
    rest = [s.site_id for s in table.sites if s.site_id not in set(first)]
    rng.shuffle(rest)
    priority = {sid: 0 for sid in first}
    priority.update({sid: i + 1 for i, sid in enumerate(rest)})
    return sorted(first + rest), priority


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # z-range: config default, overridden by "--z-range lo:hi"
    z_range = config.Z_RANGE
    if args.z_range:
        lo, hi = args.z_range.split(":")
        z_range = (int(lo), int(hi))

    # Record crop geometry next to the crops so training/scoring can match it
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preproc.json").write_text(json.dumps(preproc_dict(), indent=2))

    # Find every series/stack with both coords.json and an MBP stack
    series_list = discover_series(args.root)
    if not series_list:
        print(f"No series/stack found under {args.root} with both a "
              f"detect_pillars.py coords.json and a matching '{config.MBP_DIRNAME}' "
              f"stack. Run detect_pillars first if you haven't.")
        return

    manifest: list[dict] = []  # one row per crop written
    n_total_sites = 0
    # A non-positive count is shorthand for "all"
    all_sites = args.all_sites or args.pillars_per_series <= 0
    all_slices = args.all_slices or args.slices_per_pillar <= 0
    print(f"Root : {args.root}")
    print(f"Out  : {args.out}")
    print(f"Series/stacks found: {len(series_list)}")
    if all_sites:
        print("Cropping EVERY site in each registry. The stratified sample still "
              "runs and marks the first sites to label as priority 0 in the "
              "manifest; the rest follow in shuffled order.")
    if all_slices:
        print("Cropping EVERY usable z-slice per site. Note that neighbouring "
              "slices of one pillar are near-duplicates: they multiply the disk "
              "and the labelling, not the information. GROUP_KEY='series' keeps "
              "them out of both sides of the train/eval split.")

    for series in series_list:
        # series.mbp_dir can hold more than one stack's worth of files --
        # mbp_stack_id picks out just the slices for THIS series/stack
        refs = list_slices(series.mbp_dir, series.mbp_stack_id)
        if not refs:
            print(f"  ! no mbp slices found for {series.name} in {series.mbp_dir} -- skipping")
            continue

        # Build the site registry and choose which slices are in range
        shape = image_shape(refs)
        table = build_site_table(series.coords(), shape)
        sel_z = select_slices(refs, z_range)

        # Log a summary and save the registry
        describe_series(series, table, len(refs), len(sel_z))
        write_csv(args.out / "sites" / f"{series.name}.csv", site_table_rows(table))

        if len(table) == 0:
            print(f"    ! no sites for {series.name} -- skipping crops")
            continue

        all_xy = table.xy
        pitch = table.lattice.pitch
        # how many sites go in the first labelling batch; with --all-sites
        # everything is cropped regardless and this only sets the order
        n_first = args.pillars_per_series if args.pillars_per_series > 0 else (
            config.PILLARS_PER_SERIES if config.PILLARS_PER_SERIES > 0 else 40)
        picked, priority = select_sites(table, min(n_first, len(table)),
                                         all_sites, rng)

        if all_slices:
            # Every in-range slice
            z_picks = list(sel_z)
        else:
            # an even spread, not the middle: a pillar's sheath appears over a
            # few microns and the ends of the range are where the ambiguous
            # crops live, which is what a labelled set needs to cover
            n_slices = min(args.slices_per_pillar, len(sel_z))
            z_picks = sorted({sel_z[i] for i in
                              np.linspace(0, len(sel_z) - 1, n_slices).round().astype(int)})

        (args.out / series.name).mkdir(parents=True, exist_ok=True)
        by_source = {s.site_id: s.source for s in table.sites}

        for zi in z_picks:
            # Load (optionally z-smoothed) slice and stretch to 8-bit
            field = normalise(load_slice_smoothed(refs, zi, config.Z_SMOOTH))
            for sid in picked:
                # Build the 3-channel crop for this site and save it
                crop = build_crop(field, all_xy, sid, pitch)
                rel = Path(series.name) / f"s{sid:04d}_z{zi:03d}.png"
                crop.save(args.out / rel)
                # Record it, including its labelling priority
                manifest.append({
                    "filepath": rel.as_posix(), "series": series.name,
                    "site_id": sid, "source": by_source[sid],
                    "priority": priority[sid],
                    "slice_idx": zi, "slice_key": refs[zi].key,
                    "x": int(all_xy[sid, 0]), "y": int(all_xy[sid, 1]),
                })
            # Progress: overwrite one line when cropping all slices
            print(f"    z={zi:>3} ({refs[zi].key}): {len(picked)} crops", end="\r"
                  if all_slices else "\n")
        if all_slices:
            print(f"    {len(picked)} sites x {len(z_picks)} slices = "
                  f"{len(picked) * len(z_picks)} crops" + " " * 20)
        n_total_sites += len(picked)

    # labelling order, not lattice order: priority first, then series/site/z
    manifest.sort(key=lambda r: (r["priority"], r["series"], r["site_id"],
                                  r["slice_idx"]))
    write_csv(args.out / "manifest.csv", manifest)

    # Summary and warnings
    n_first = sum(1 for r in manifest if r["priority"] == 0)
    print(f"\n{len(manifest)} crops from {n_total_sites} sites written to {args.out}")
    if all_sites and len(manifest) > n_first:
        print(f"  the first {n_first} rows of manifest.csv are the stratified "
              f"sample - label those before working into the remaining "
              f"{len(manifest) - n_first}")
    if len(manifest) > 5000:
        print(f"  ! {len(manifest)} crops is far more than anyone hand-labels. "
              f"That is fine if you wanted them for scoring or pseudo-labelling; "
              f"if the goal is a training set, label in manifest order and stop "
              f"when the classes stop improving.")
    print("Next: python -m label_crops")


if __name__ == "__main__":
    main()
