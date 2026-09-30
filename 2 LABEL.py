"""
Stage 2: hand-label the crops from make_crops (Tk GUI).

Display settings are cosmetic; crop files are never modified. Labels save
after every keystroke to CROPS_DIR/labels.csv; re-running resumes at the
first unlabelled crop.
"""

from __future__ import annotations

import argparse
import csv
import tkinter as tk
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk

import config

# On-screen text for each class key
CLASS_HELP = {
    0: "0  not wrapped",
    1: "1  somewhat  (<50%)",
    2: "2  mostly    (>=50%, gap visible)",
    3: "3  full      (complete ring)",
}
DISPLAY = 720                                        # display size in px
VIEWS = ("cyan", "mbp", "raw")                       # views cycled with "v"
YELLOW = np.array([1.00, 0.85, 0.15], dtype=np.float32)  # pillar overlay colour

# Rendering

def marker_centre(marker: np.ndarray) -> tuple[float, float]:
    """Intensity centroid of the target-marker channel (the target is
    off-centre at field edges, so it is read rather than assumed)."""
    w = marker.astype(np.float64)
    total = w.sum()
    # Empty marker: fall back to the crop centre
    if total <= 0:
        return marker.shape[1] / 2, marker.shape[0] / 2
    # Weighted mean of pixel coordinates
    ys, xs = np.mgrid[0:marker.shape[0], 0:marker.shape[1]]
    return float((xs * w).sum() / total), float((ys * w).sum() / total)


def _edge(mask01: np.ndarray, grow: int = 0) -> np.ndarray:
    """Boundary band of a binary mask, optionally dilated outward first."""
    img = Image.fromarray((mask01 * 255).astype(np.uint8))
    # Optional outward dilation
    if grow > 0:
        img = img.filter(ImageFilter.MaxFilter(2 * grow + 1))
    # Edge = mask minus its erosion
    outer = np.asarray(img).astype(np.float32) / 255.0
    inner = np.asarray(img.filter(ImageFilter.MinFilter(5))).astype(np.float32) / 255.0
    return np.clip(outer - inner, 0, 1)


def render(crop: Image.Image, view: str = "cyan", gamma: float = 1.0,
           fill_alpha: float = 0.18, edge_alpha: float = 0.5,
           out: int = DISPLAY) -> Image.Image:
    """Composite one crop for display (never modifies the file). Fill and
    outline of the pillar mask are separately adjustable."""
    # Split the crop into its three channels, scaled to 0-1
    a = np.asarray(crop).astype(np.float32) / 255.0
    mbp, mask, marker = a[..., 0], a[..., 1], a[..., 2]

    # Display gamma on MBP only (>1 brightens faint signal)
    if gamma != 1.0:
        mbp = np.power(np.clip(mbp, 0, 1), 1.0 / gamma)

    if view == "raw":
        # Exactly what the model sees
        rgb = a.copy()
    else:
        rgb = np.stack([np.zeros_like(mbp), mbp, mbp], axis=-1)      # cyan MBP

        if view == "cyan":
            m01 = (mask > 0.5).astype(np.float32)

            # Translucent yellow fill over the pillar shafts
            if fill_alpha > 0:
                f = (m01 * fill_alpha)[..., None]
                rgb = rgb * (1 - f) + YELLOW * f

            # Yellow outline around the pillar shafts
            if edge_alpha > 0:
                e = (_edge(m01) * edge_alpha)[..., None]
                rgb = rgb * (1 - e) + YELLOW * e

            # Target outline: the mask component under the marker, grown a few
            # pixels so the white line sits just outside the real shaft.
            cx, cy = marker_centre(marker)
            yy, xx = np.mgrid[0:m01.shape[0], 0:m01.shape[1]]
            near = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (0.42 * m01.shape[0] / 2.5) ** 2
            target = m01 * near
            if target.sum() > 0:
                t = _edge(target, grow=3)[..., None]
                rgb = rgb * (1 - t) + np.array([1.0, 1.0, 1.0], np.float32) * t

    # Back to 8-bit and upscale for display
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize((out, out), Image.LANCZOS)

    if view == "mbp":
        # No mask drawn in this view, so fall back to a plain circle to show
        # which pillar is being scored.
        cx, cy = marker_centre(marker)
        k = out / marker.shape[0]
        r = out * 0.20 / 2.5 * 1.9
        ImageDraw.Draw(img).ellipse(
            [cx * k - r, cy * k - r, cx * k + r, cy * k + r],
            outline=(255, 255, 255), width=3)
    return img

# GUI - *Claude Generated*

class Labeller:
    """Tk window that shows crops one at a time and records key-press labels."""

    def __init__(self, root_dir: Path, rows: list[dict], labels_path: Path,
                 all_rows: list[dict] | None = None,
                 start_at_unlabelled: bool = True, title: str = "Wrap scoring"):
        self.root_dir = root_dir
        self.rows = rows            # crops to show, in order
        # Metadata for EVERY crop, not just the ones on screen. save() rewrites
        # the whole labels.csv, so a filtered view must still be able to look
        # up the series/site/source of labels it is not currently showing -
        # otherwise reviewing one class would blank those columns for the rest.
        self.all_rows = all_rows if all_rows is not None else rows
        self.labels_path = labels_path
        self.title = title
        self.labels: dict[str, int] = {}   # filepath -> label
        self.history: list[str] = []       # labelled filepaths, for undo

        # Initial display settings
        self.view = 0
        self.gamma = 1.6
        self.fill = 0.10
        self.edge = 0.55

        # Load any existing labels so work resumes
        if labels_path.exists():
            with open(labels_path, newline="") as f:
                for r in csv.DictReader(f):
                    self.labels[r["filepath"]] = int(r["label"])

        # Start at the first unlabelled crop (or the first crop)
        self.i = (next((k for k, r in enumerate(rows)
                        if r["filepath"] not in self.labels), 0)
                  if start_at_unlabelled else 0)

        # --- build the window: image, status line, two help lines ---
        self.win = tk.Tk()
        self.win.title(self.title)
        self.win.configure(bg="#111")
        self.canvas = tk.Label(self.win, bg="#111")
        self.canvas.pack()
        self.status = tk.Label(self.win, font=("TkDefaultFont", 11),
                               justify="left", bg="#111", fg="#eee")
        self.status.pack(fill="x")
        tk.Label(self.win, text="   ".join(CLASS_HELP.values()),
                 font=("TkDefaultFont", 9), bg="#111", fg="#888").pack(fill="x")
        tk.Label(self.win, text="v view   + - gamma   [ ] fill   , . outline"
                               "   u undo   q quit",
                 font=("TkDefaultFont", 9), bg="#111", fg="#666").pack(fill="x")

        # Route every key press to on_key, then draw the first crop
        self.win.bind("<Key>", self.on_key)
        self.show()

    def show(self) -> None:
        """Render the current crop and refresh the status line."""
        if not (0 <= self.i < len(self.rows)):
            self.status.config(text="Done - press q to quit")
            return

        # Load and render the current crop
        row = self.rows[self.i]
        crop = Image.open(self.root_dir / row["filepath"]).convert("RGB")
        img = render(crop, VIEWS[self.view], self.gamma,
                     self.fill, self.edge, DISPLAY)

        # Keep a reference so Tk doesn't garbage-collect the image
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.photo)

        # Status flags: non-detected source, and edits made in review mode
        cur = self.labels.get(row["filepath"])
        src = row.get("source", "detected")
        flag = "" if src == "detected" else f"  [{src}]"
        was = row.get("_original_label")
        if was is not None and cur is not None and int(was) != int(cur):
            flag += f"   CHANGED {was} -> {cur}"
        self.status.config(
            text=f"[{self.i + 1}/{len(self.rows)}]  labelled {len(self.labels)}   "
                 f"{row['series']}  site={row['site_id']}{flag}  "
                 f"z={row['slice_key']}   label={'-' if cur is None else cur}   "
                 f"view={VIEWS[self.view]}  gamma={self.gamma:.1f}  "
                 f"fill={self.fill:.2f}  outline={self.edge:.2f}"
        )

    def save(self) -> None:
        """Rewrite labels.csv with every label plus its crop metadata."""
        by_path = {r["filepath"]: r for r in self.all_rows}
        with open(self.labels_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "label", "series", "site_id", "source",
                        "slice_key"])
            for fp, lab in self.labels.items():
                r = by_path.get(fp, {})
                w.writerow([fp, lab, r.get("series", ""), r.get("site_id", ""),
                            r.get("source", ""), r.get("slice_key", "")])

    def on_key(self, event) -> None:
        """Handle one key press, then redraw."""
        k = event.keysym

        if k in ("0", "1", "2", "3") and 0 <= self.i < len(self.rows):
            # Assign label, save immediately, advance
            fp = self.rows[self.i]["filepath"]
            self.labels[fp] = int(k)
            self.history.append(fp)
            self.save()
            self.i += 1
        elif k == "Right":
            # Next crop without labelling
            self.i = min(self.i + 1, len(self.rows) - 1)
        elif k == "Left":
            # Previous crop
            self.i = max(self.i - 1, 0)
        elif k == "v":
            # Cycle display view
            self.view = (self.view + 1) % len(VIEWS)
        elif k in ("plus", "equal", "KP_Add"):
            # Brighten MBP
            self.gamma = min(4.0, self.gamma + 0.2)
        elif k in ("minus", "KP_Subtract"):
            # Darken MBP
            self.gamma = max(0.4, self.gamma - 0.2)
        elif k == "bracketright":
            # More opaque pillar fill
            self.fill = min(0.9, self.fill + 0.05)
        elif k == "bracketleft":
            # Less opaque pillar fill
            self.fill = max(0.0, self.fill - 0.05)
        elif k == "period":
            # More opaque outline
            self.edge = min(1.0, self.edge + 0.1)
        elif k == "comma":
            # Less opaque outline
            self.edge = max(0.0, self.edge - 0.1)
        elif k == "u" and self.history:
            # Undo last label and jump back to that crop
            fp = self.history.pop()
            self.labels.pop(fp, None)
            self.i = next((n for n, r in enumerate(self.rows)
                           if r["filepath"] == fp), self.i)
            self.save()
        elif k == "q":
            # Save and close
            self.save()
            self.win.destroy()
            return

        self.show()


def main() -> None:
    # Crops folder (default from config)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crops", type=Path, default=config.CROPS_DIR)
    args = p.parse_args()

    # The manifest from make_crops lists every crop in labelling order
    manifest = args.crops / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"No manifest at {manifest}\nRun: python -m make_crops")

    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    # Run the GUI until the user quits
    app = Labeller(args.crops, rows, args.crops / "labels.csv")
    app.win.mainloop()

    # Print the final class distribution
    dist = Counter(app.labels.values())
    print(f"Saved {len(app.labels)} labels to {args.crops / 'labels.csv'}")
    print("Distribution: " + "  ".join(
        f"{k}={dist.get(k, 0)} ({config.CLASS_NAMES[k]})"
        for k in range(config.NUM_LABELS)))


if __name__ == "__main__":
    main()
