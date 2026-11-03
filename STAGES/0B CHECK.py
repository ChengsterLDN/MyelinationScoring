"""
Checking the sites produced from PIL_DET
Writes to RESULTS_DIR/site_check: sites/<series>.csv, <series>.png (first
slice with sites ringed: grey = detected, blue = off-lattice, red =
inferred) and site_check.csv (one row per stack). Red rings on empty gel
mean the lattice fit is extrapolating.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import config
from wrap_common import (
    build_site_table, discover_series, image_shape, list_slices, load_slice,
    normalise, select_slices, site_table_rows, write_csv,
)

# Ring colour (RGB) and line width for each site source
RING = {"detected": ((160, 160, 160), 2),
        "off_lattice": ((70, 150, 255), 5),
        "inferred": ((255, 70, 70), 5)}


def main() -> None:
    # --- command-line overrides for the input root and output folder ---
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=config.ROOT_DIR)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    # Default output folder sits inside RESULTS_DIR
    out = args.out or (config.RESULTS_DIR / "site_check")
    out.mkdir(parents=True, exist_ok=True)

    # Find every (series, stack) pair with both coords.json and an MBP stack
    series_list = discover_series(args.root)
    if not series_list:
        print(f"No series/stack found under {args.root} with both a "
              f"detect_pillars.py coords.json and a matching '{config.MBP_DIRNAME}' "
              f"stack. Run detect_pillars first if you haven't.")
        return

    rows = []  # one summary row per series/stack
    for series in series_list:
        # series.mbp_dir can hold more than one stack's worth of files --
        # mbp_stack_id picks out just the slices for THIS series/stack
        refs = list_slices(series.mbp_dir, series.mbp_stack_id)
        if not refs:
            print(f"  ! no mbp slices found for {series.name} in {series.mbp_dir} -- skipping")
            continue

        # Build the site registry from the coords file and image size
        shape = image_shape(refs)
        coords = series.coords()
        table = build_site_table(coords, shape)
        lat = table.lattice
        c = table.counts_by_source()

        # Save the registry as CSV
        write_csv(out / "sites" / f"{series.name}.csv", site_table_rows(table))

        # Pick the middle slice of the selected range as the overlay background
        sel = select_slices(refs)
        mid = sel[len(sel) // 2]
        field = normalise(load_slice(refs[mid]))

        # Grey-scale -> RGB so coloured rings can be drawn on it
        img = Image.merge("RGB", [Image.fromarray(field)] * 3)
        draw = ImageDraw.Draw(img)

        # Ring radius: slightly larger than the pillar, at least 4 px
        r = max(4.0, lat.pitch * config.PILLAR_R_FRAC * 1.35)
        for s in table.sites:
            col, width = RING[s.source]
            draw.ellipse([s.x - r, s.y - r, s.x + r, s.y + r], outline=col, width=width)

        # Save the overlay PNG
        img_path = out / f"{series.name}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(img_path)

        # Record lattice-fit statistics for this stack
        rows.append({
            "series": series.name,
            "image_w": shape[1], "image_h": shape[0],
            "n_slices": len(refs),
            "n_in_coord_file": len(coords),
            "n_detected": c["detected"], "n_off_lattice": c["off_lattice"],
            "n_inferred": c["inferred"], "n_all_sites": len(table),
            "pitch_px": round(lat.pitch, 1),
            "spacing_a_px": round(float(np.linalg.norm(lat.a)), 1),
            "spacing_b_px": round(float(np.linalg.norm(lat.b)), 1),
            "rotation_deg": round(lat.rotation_deg, 2),
            "median_resid_px": round(lat.median_resid, 2),
            "lattice_ok": table.lattice_ok,
            "note": table.lattice_note,
        })

        # One-line console summary; flag rejected lattice fits
        flag = "" if table.lattice_ok else "   <-- FIT REJECTED"
        print(f"{series.name:<30} {len(coords):>4} in file -> {len(table):>4} sites "
              f"({c['inferred']:>3} inferred, {c['off_lattice']:>2} off-lattice)  "
              f"pitch {lat.pitch:6.1f}  rot {lat.rotation_deg:+6.2f}  "
              f"resid {lat.median_resid:5.2f}{flag}")

    # Write the combined summary table
    write_csv(out / "site_check.csv", rows)
    print(f"\nOverlays and registry -> {out}")


if __name__ == "__main__":
    main()
