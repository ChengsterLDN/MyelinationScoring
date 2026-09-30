"""
Stage 1: sample per-pillar context crops for hand labelling.

Runs over every series/stack. Writes to CROPS_DIR: manifest.csv,
preproc.json (crop geometry), sites/<series>.csv and the crop PNGs.
Lattice geometry is measured per stack from detect_pillars' coords.json.
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
    p.add_argument("--pillars-per-series", type=int, default=config.PILLARS_PER_SERIES)
    p.add_argument("--slices-per-pillar", type=int, default=config.SLICES_PER_PILLAR)
    p.add_argument("--z-range", type=str, default=None, help="e.g. 5:45")
    p.add_argument("--seed", type=int, default=config.SAMPLE_SEED)
    return p.parse_args()


def sample_sites(table, n_wanted: int, rng: np.random.Generator) -> list[int]:
    """
    Choose which sites to label.

    Sampling is stratified by source so that off-lattice and inferred sites
    appear in training at roughly the rate they appear at inference.
    """
    # Group site ids by their source (detected / off_lattice / inferred)
    by_source: dict[str, list[int]] = {}
    for s in table.sites:
        by_source.setdefault(s.source, []).append(s.site_id)

    # Can't sample more sites than exist
    n_wanted = min(n_wanted, len(table))
    picked: list[int] = []
    total = len(table)

    # Each source gets a share proportional to its size (at least one site)
    for source, ids in by_source.items():
        share = max(1, int(round(n_wanted * len(ids) / total)))
        share = min(share, len(ids))
        picked += rng.choice(ids, size=share, replace=False).tolist()

    # Rounding can overshoot; trim back to n_wanted at random
    if len(picked) > n_wanted:
        picked = rng.choice(picked, size=n_wanted, replace=False).tolist()
    return sorted(int(p) for p in picked)


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
    print(f"Root : {args.root}")
    print(f"Out  : {args.out}")
    print(f"Series/stacks found: {len(series_list)}")

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
        # Stratified sample of sites to crop
        picked = sample_sites(table, args.pillars_per_series, rng)

        # Evenly spaced slice indices through the selected range
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
                # Record it in the manifest
                manifest.append({
                    "filepath": rel.as_posix(), "series": series.name,
                    "site_id": sid, "source": by_source[sid],
                    "slice_idx": zi, "slice_key": refs[zi].key,
                    "x": int(all_xy[sid, 0]), "y": int(all_xy[sid, 1]),
                })
            print(f"    z={zi:>3} ({refs[zi].key}): {len(picked)} crops")

    # Write the manifest that label_crops reads
    write_csv(args.out / "manifest.csv", manifest)
    print(f"\n{len(manifest)} crops written to {args.out}")
    print("Next: python -m label_crops")


if __name__ == "__main__":
    main()
