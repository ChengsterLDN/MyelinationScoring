"""
DETECT PILLARS 

For every "pillar" folder under ROOT_DIR, each stack (files grouped by stripping "_z<n>") gets <stack>_grid_mask.png (uniform disks on the fitted lattice), <stack>_coords.json (site positions, types, fit), a raw mask and optional QC / focus-stack images. 
Slices are judged individually by how many pillar-sized round blobs they contain; unusable slices are dropped. 
Missing lattice sites are filled as "inferred"; off-lattice blobs are discarded.
Edge sites are kept while <= PILLAR_EDGE_MAX_CUT_FRAC of the disk is cut off.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure, filters
from skimage.draw import disk as draw_disk
from skimage.morphology import disk as disk_selem

import config


def cfg(name: str, default):
    """config.<name> if it exists, else default (so newer settings don't
    require editing config.py)."""
    return getattr(config, name, default)

# Filename parsing / stack grouping


# Matches the z-index token so it can be stripped to recover the stack id,
# e.g. "1_2_c2_yellow_z7_t0.png" -> stack id "1_2_c2_yellow_t0", z=7
_Z_TOKEN_RE = re.compile(r"_z(\d+)(?=[_.])", re.IGNORECASE)


def parse_z_stack_filename(filename: str):
    """Return (stack_id, z_index) for one pillar-channel PNG, or None if
    the filename has no recognisable '_z<n>' token."""
    match = _Z_TOKEN_RE.search(filename)
    if not match:
        return None
    z_index = int(match.group(1))
    # Remove the z token and ".png" to get the stack id
    stack_id = filename[: match.start()] + filename[match.end():]
    stack_id = re.sub(r"\.png$", "", stack_id, flags=re.IGNORECASE)
    return stack_id, z_index


def find_pillar_folders(root_dir: Path) -> list[Path]:
    """Recursively find every folder literally named PILLAR_DIRNAME."""
    name = config.PILLAR_DIRNAME.lower()
    return sorted(p for p in root_dir.rglob("*") if p.is_dir() and p.name.lower() == name)


def group_into_stacks(pillar_dir: Path):
    """Group every PNG in pillar_dir into {stack_id: [(z_index, path), ...]},
    sorted by z. Returns (stacks, skipped_filenames)."""
    stacks: dict[str, list[tuple[int, Path]]] = {}
    skipped = []
    for png_path in sorted(pillar_dir.glob("*.png")):
        parsed = parse_z_stack_filename(png_path.name)
        # Files with no z token are reported and ignored
        if parsed is None:
            skipped.append(png_path.name)
            continue
        stack_id, z_index = parsed
        stacks.setdefault(stack_id, []).append((z_index, png_path))
    # Order each stack's slices by z
    for slices in stacks.values():
        slices.sort(key=lambda t: t[0])
    return stacks, skipped

# Loading, debris exclusion, normalisation


def load_intensity(png_path: Path) -> np.ndarray:
    """Load a PNG as one float32 intensity plane (per-pixel channel MAX, so
    a coloured LUT isn't diluted by its empty channels)."""
    arr = np.array(Image.open(png_path).convert("RGB"), dtype=np.float32)
    return arr.max(axis=-1)


def find_debris_mask(intensity: np.ndarray, seed_percentile: float, min_area_frac: float,
                      dilate_px: int) -> np.ndarray:
    """Mask of debris/cells: bright connected regions (top seed_percentile
    of pixels) larger than min_area_frac of the frame -- far bigger than any
    pillar -- dilated to cover their halo."""
    # Candidate bright pixels (threshold at least 1)
    seed_thresh = max(np.percentile(intensity, seed_percentile), 1.0)
    seed_mask = intensity >= seed_thresh
    # Label 8-connected bright regions
    struct = ndi.generate_binary_structure(2, 2)
    labels, n = ndi.label(seed_mask, structure=struct)
    if n == 0:
        return np.zeros_like(seed_mask)
    # Region sizes; keep only the large ones as debris
    sizes = ndi.sum(seed_mask, labels, index=range(1, n + 1))
    min_area = min_area_frac * intensity.size
    debris_ids = [i + 1 for i, s in enumerate(sizes) if s >= min_area]
    if not debris_ids:
        return np.zeros_like(seed_mask)
    debris_mask = np.isin(labels, debris_ids)
    # Grow to include the glow around each region
    if dilate_px > 0:
        debris_mask = ndi.binary_dilation(debris_mask, iterations=dilate_px)
    return debris_mask


def normalize_slice(intensity: np.ndarray, bg_pct: float, ref_pct: float, min_scale: float,
                     debris_mask: np.ndarray | None = None) -> np.ndarray:
    """Background-subtract and rescale by a percentile range (median/MAD
    collapses to 0 on these images). Debris is excluded from the statistics
    and set to -1 so it never reads as a pillar."""
    # Statistics from non-debris pixels only
    valid = intensity[~debris_mask] if (debris_mask is not None and debris_mask.any()) else intensity
    bg = np.percentile(valid, bg_pct)
    ref = np.percentile(valid, ref_pct)
    scale = max(ref - bg, min_scale)
    norm = (intensity - bg) / scale
    # Force debris strongly negative
    if debris_mask is not None:
        norm[debris_mask] = -1.0
    return norm

# Per-slice pillar finding 


def find_slice_pillars(intensity: np.ndarray, debris_mask: np.ndarray, min_diameter_px: float,
                        max_diameter_px: float, max_eccentricity: float) -> list[dict]:
    """Otsu-threshold ONE slice (debris excluded) and keep connected
    components whose equivalent diameter is in range and which are round
    enough to be a pillar."""
    # Threshold computed from non-debris pixels
    valid = intensity[~debris_mask] if debris_mask.any() else intensity
    if valid.size == 0 or valid.max() <= valid.min():
        return []
    try:
        thresh = filters.threshold_otsu(valid)
    except ValueError:
        return []
    # Foreground = above threshold and not debris
    bright = (intensity >= thresh) & (~debris_mask)
    struct = ndi.generate_binary_structure(2, 2)
    # Close small gaps and fill holes so each pillar is one solid blob
    bright = ndi.binary_closing(bright, structure=disk_selem(2))
    bright = ndi.binary_fill_holes(bright, structure=struct)
    labels, n = ndi.label(bright, structure=struct)
    if n == 0:
        return []
    found = []
    for p in measure.regionprops(labels):
        # Diameter of a circle with the same area
        diameter = 2.0 * math.sqrt(p.area / math.pi)
        # Keep blobs that pass the size and roundness tests
        if min_diameter_px <= diameter <= max_diameter_px and p.eccentricity <= max_eccentricity:
            found.append({
                "x": float(p.centroid[1]), "y": float(p.centroid[0]),
                "area": int(p.area), "diameter": diameter,
            })
    return found


def analyze_slice(intensity: np.ndarray, debris_mask: np.ndarray, min_diameter_px: float,
                   max_diameter_px: float, max_eccentricity: float) -> tuple[np.ndarray, list[dict], bool]:
    """Try a slice as-is and inverted; keep whichever finds more valid
    pillar blobs (this decides contrast polarity).

    Returns (corrected_intensity, blobs_found, was_flipped).
    """
    # Blobs with original contrast
    blobs_asis = find_slice_pillars(intensity, debris_mask, min_diameter_px, max_diameter_px,
                                     max_eccentricity)
    # Blobs with inverted contrast
    flipped = float(intensity.max()) - intensity
    blobs_flipped = find_slice_pillars(flipped, debris_mask, min_diameter_px, max_diameter_px,
                                        max_eccentricity)
    # Inverted wins only if strictly better
    if len(blobs_flipped) > len(blobs_asis):
        return flipped, blobs_flipped, True
    return intensity, blobs_asis, False


# Lattice fit 

def cluster_candidates(candidates: list[dict], radius_px: float) -> list[dict]:
    """Merge detections of the same pillar from different slices (points
    within radius_px, via union-find). Must run before the lattice fit, or
    near-duplicate pairs corrupt the pitch estimate."""
    if not candidates:
        return []
    # All pairs of candidates closer than radius_px
    pts = np.array([(c["x"], c["y"]) for c in candidates])
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=radius_px)
    n = len(candidates)
    parent = list(range(n))

    def find(a: int) -> int:
        """Root of a's cluster (with path compression)."""
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        """Join the clusters of a and b."""
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Link every close pair
    for i, j in pairs:
        union(i, j)

    # Collect candidates by cluster root
    groups: dict[int, list[dict]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(candidates[i])

    # One merged record per cluster: mean position, largest blob's details
    merged = []
    for group in groups.values():
        best = max(group, key=lambda c: c["diameter"])
        merged.append({
            "x": float(np.mean([c["x"] for c in group])),
            "y": float(np.mean([c["y"] for c in group])),
            "best_x": best["x"], "best_y": best["y"],
            "area": best["area"], "diameter": best["diameter"], "z": best["z"],
            "n_slice_detections": len(group),
        })
    return merged


def estimate_lattice(centroids_xy: list[tuple[float, float]]) -> dict | None:
    """Initial axis-aligned lattice estimate (pitch and phase in x and y
    separately). Pitch from neighbour spacing; phase by circular mean of
    positions modulo pitch. None if fewer than 6 points."""
    pts = np.asarray(centroids_xy, dtype=np.float64)
    if len(pts) < 6:
        return None

    # Rough pitch = median nearest-neighbour distance
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    rough_pitch = float(np.median(dists[:, 1]))
    if not np.isfinite(rough_pitch) or rough_pitch <= 0:
        return None

    # Separate spacings for same-row pairs (x) and same-column pairs (y)
    x_gaps, y_gaps = [], []
    for i, j in tree.query_pairs(r=1.5 * rough_pitch):
        dx = abs(pts[j, 0] - pts[i, 0])
        dy = abs(pts[j, 1] - pts[i, 1])
        if dy < 0.3 * rough_pitch and dx > 0.3 * rough_pitch:
            x_gaps.append(dx)
        if dx < 0.3 * rough_pitch and dy > 0.3 * rough_pitch:
            y_gaps.append(dy)

    # Per-axis pitch, falling back to the rough value if too few pairs
    pitch_x = float(np.median(x_gaps)) if len(x_gaps) >= 5 else rough_pitch
    pitch_y = float(np.median(y_gaps)) if len(y_gaps) >= 5 else rough_pitch

    def circular_phase(vals: np.ndarray, pitch: float) -> float:
        """Mean of vals modulo pitch, treating it as an angle (handles wrap)."""
        theta = 2 * np.pi * (vals % pitch) / pitch
        mean_angle = math.atan2(np.sin(theta).sum(), np.cos(theta).sum())
        return (mean_angle / (2 * np.pi) * pitch) % pitch

    phase_x = circular_phase(pts[:, 0], pitch_x)
    phase_y = circular_phase(pts[:, 1], pitch_y)
    return {"pitch_x": pitch_x, "pitch_y": pitch_y, "phase_x": phase_x, "phase_y": phase_y}


def refine_lattice(centroids_xy: list[tuple[float, float]], lattice: dict,
                    off_lattice_frac: float, iters: int,
                    max_pitch_drift: float) -> tuple[dict, np.ndarray, float]:
    """Iteratively re-fit pitch and phase by least squares on inliers only
    (points within off_lattice_frac pitch of a site), so outliers don't pull
    the grid. A step changing pitch by more than max_pitch_drift is rejected.

    Returns (lattice, inlier_mask, rms_residual_px).
    """
    pts = np.asarray(centroids_xy, dtype=np.float64)
    lat = dict(lattice)

    def fit_state(lat: dict):
        """Snap points to lat; return rows, cols, inlier mask, inlier RMS."""
        px, py = lat["pitch_x"], lat["pitch_y"]
        cols = np.round((pts[:, 0] - lat["phase_x"]) / px)
        rows = np.round((pts[:, 1] - lat["phase_y"]) / py)
        # Distance from each point to its nearest lattice site
        resid = np.hypot(pts[:, 0] - (lat["phase_x"] + cols * px),
                         pts[:, 1] - (lat["phase_y"] + rows * py))
        keep = resid <= off_lattice_frac * (px + py) / 2
        r = float(np.sqrt((resid[keep] ** 2).mean())) if keep.any() else float("nan")
        return rows, cols, keep, r

    rows, cols, inliers, rms = fit_state(lat)

    for _ in range(max(int(iters), 0)):
        px, py = lat["pitch_x"], lat["pitch_y"]
        rows, cols, inliers, rms = fit_state(lat)

        # Need enough inliers spanning at least 2 rows and 2 columns
        if inliers.sum() < 6:
            break
        if len(np.unique(cols[inliers])) < 2 or len(np.unique(rows[inliers])) < 2:
            break

        # Linear fit: x = pitch_x * col + offset, y = pitch_y * row + offset
        new_px, new_ox = np.polyfit(cols[inliers], pts[inliers, 0], 1)
        new_py, new_oy = np.polyfit(rows[inliers], pts[inliers, 1], 1)
        # Reject invalid or too-large pitch changes
        if not (np.isfinite(new_px) and np.isfinite(new_py)) or new_px <= 0 or new_py <= 0:
            break
        if (abs(new_px - px) > max_pitch_drift * px or
                abs(new_py - py) > max_pitch_drift * py):
            break

        # Accept the new fit; stop once it no longer changes
        cand = {"pitch_x": float(new_px), "pitch_y": float(new_py),
                "phase_x": float(new_ox % new_px), "phase_y": float(new_oy % new_py)}
        if cand == lat:
            break
        lat = cand

    return lat, inliers, rms


def snap_to_lattice(centroids_xy: list[tuple[float, float]], lattice: dict,
                     off_lattice_frac: float) -> list[dict]:
    """Assign each centre to its nearest lattice (row, col) and flag whether
    it is close enough to count as on-lattice."""
    px, py = lattice["pitch_x"], lattice["pitch_y"]
    ox, oy = lattice["phase_x"], lattice["phase_y"]
    avg_pitch = (px + py) / 2
    out = []
    for x, y in centroids_xy:
        # Nearest grid indices and that site's position
        col = round((x - ox) / px)
        row = round((y - oy) / py)
        sx, sy = ox + col * px, oy + row * py
        resid = math.hypot(x - sx, y - sy)
        out.append({
            "row": row, "col": col,
            "on_lattice": resid <= off_lattice_frac * avg_pitch,
            "residual_px": resid,
        })
    return out


def segment_cut_fraction(d: float, r: float) -> float:
    """Fraction of a disk of radius r lying beyond a straight line at
    signed distance d from its centre (d > 0 = centre inside)."""
    if r <= 0:
        return 0.0 if d >= 0 else 1.0
    # Entirely inside / entirely outside
    if d >= r:
        return 0.0
    if d <= -r:
        return 1.0
    # Circular-segment area divided by disk area
    return (r * r * math.acos(d / r) - d * math.sqrt(max(r * r - d * d, 0.0))) / (math.pi * r * r)


def edge_centre_margin(r: float, max_cut_frac: float) -> float:
    """How far inside the border a centre must sit so that at most
    max_cut_frac of the disk is outside (0 at 0.5; negative above 0.5).
    Solved by bisection."""
    if max_cut_frac <= 0:
        return r
    if max_cut_frac >= 1:
        return -r
    lo, hi = -r, r                      # cut fraction decreases as d grows
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if segment_cut_fraction(mid, r) > max_cut_frac:
            lo = mid
        else:
            hi = mid
    return hi


def disk_cut_fraction(cx: float, cy: float, r: float,
                       image_shape: tuple[int, int], samples: int = 48) -> float:
    """Fraction of a pillar's disk outside the image, estimated by sampling
    (so corner sites cut by two edges are not double-counted)."""
    h, w = image_shape
    if r <= 0:
        return 0.0 if (0 <= cx <= w - 1 and 0 <= cy <= h - 1) else 1.0
    # Fully inside: nothing cut
    if cx - r >= 0 and cy - r >= 0 and cx + r <= w - 1 and cy + r <= h - 1:
        return 0.0
    # Sample grid over the disk's bounding box
    t = np.linspace(-r, r, samples)
    in_disk = (t[None, :] ** 2 + t[:, None] ** 2) <= r * r
    xs, ys = cx + t[None, :], cy + t[:, None]
    in_img = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    n = int(in_disk.sum())
    if n == 0:
        return 1.0
    # 1 - (disk samples inside image / all disk samples)
    return float(1.0 - (in_disk & in_img).sum() / n)


def lattice_index_range(lattice: dict, image_shape: tuple[int, int],
                         margin_px: float) -> tuple[int, int, int, int]:
    """Row/col index range of sites whose centres are at least margin_px
    inside the border (negative margin allows centres just outside)."""
    h, w = image_shape
    px, py = lattice["pitch_x"], lattice["pitch_y"]
    ox, oy = lattice["phase_x"], lattice["phase_y"]
    col_min = math.ceil((margin_px - ox) / px)
    col_max = math.floor((w - 1 - margin_px - ox) / px)
    row_min = math.ceil((margin_px - oy) / py)
    row_max = math.floor((h - 1 - margin_px - oy) / py)
    return row_min, row_max, col_min, col_max


def grid_extent(lattice: dict, image_shape: tuple[int, int],
                 detected_rc: list[tuple[int, int]], mode: str, pad_cells: int,
                 radius_px: float, max_cut_frac: float,
                 extend_to_edge: bool = True) -> tuple[int, int, int, int]:
    """The (row, col) box to fill with sites.

    mode="detections": bounding box of detected sites (+ pad_cells), grown
    outward while the next row/col is clipped by the frame (extend_to_edge),
    since half-cut edge pillars are usually missed by the detector.
    mode="image": every lattice site in frame.
    Always clipped to sites within max_cut_frac of the border.
    """
    # Largest admissible box given the edge-cut limit
    margin = edge_centre_margin(radius_px, max_cut_frac)
    img_r0, img_r1, img_c0, img_c1 = lattice_index_range(lattice, image_shape, margin)
    if mode == "image" or not detected_rc:
        return img_r0, img_r1, img_c0, img_c1

    # Bounding box of detected sites, padded and clipped
    rows = [r for r, _ in detected_rc]
    cols = [c for _, c in detected_rc]
    r0, r1 = max(min(rows) - pad_cells, img_r0), min(max(rows) + pad_cells, img_r1)
    c0, c1 = max(min(cols) - pad_cells, img_c0), min(max(cols) + pad_cells, img_c1)

    if extend_to_edge:
        px, py = lattice["pitch_x"], lattice["pitch_y"]
        ox, oy = lattice["phase_x"], lattice["phase_y"]

        def clipped_row(row: int, lo: int, hi: int) -> bool:
            """True if any site in this row (cols lo..hi) is cut by the frame."""
            sy = oy + row * py
            return any(disk_cut_fraction(ox + c * px, sy, radius_px, image_shape) > 0.0
                       for c in range(lo, hi + 1))

        def clipped_col(col: int, lo: int, hi: int) -> bool:
            """True if any site in this column (rows lo..hi) is cut by the frame."""
            sx = ox + col * px
            return any(disk_cut_fraction(sx, oy + r * py, radius_px, image_shape) > 0.0
                       for r in range(lo, hi + 1))

        # Grow each side one cell at a time while the new rank is edge-clipped
        while r0 - 1 >= img_r0 and clipped_row(r0 - 1, c0, c1):
            r0 -= 1
        while r1 + 1 <= img_r1 and clipped_row(r1 + 1, c0, c1):
            r1 += 1
        while c0 - 1 >= img_c0 and clipped_col(c0 - 1, r0, r1):
            c0 -= 1
        while c1 + 1 <= img_c1 and clipped_col(c1 + 1, r0, r1):
            c1 += 1

    return r0, r1, c0, c1



def local_site_evidence(norm_stack: np.ndarray, x: float, y: float, radius_px: float,
                         vote_thresh: float, min_vote_fraction: float) -> bool:
    """Is there signal at an inferred site? Each usable slice votes if the
    mean of a small patch there exceeds vote_thresh; True if enough vote."""
    n_slices, h, w = norm_stack.shape
    # Square patch around (x, y), clipped to the image
    x0, x1 = max(0, int(x - radius_px)), min(w, int(x + radius_px) + 1)
    y0, y1 = max(0, int(y - radius_px)), min(h, int(y + radius_px) + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    # Patch mean per slice, then fraction of slices voting
    patch = norm_stack[:, y0:y1, x0:x1].reshape(n_slices, -1)
    slice_means = patch.mean(axis=1)
    frac_voting = float((slice_means > vote_thresh).mean())
    return frac_voting >= min_vote_fraction


def build_grid_mask(image_shape: tuple[int, int], sites: list[dict], radius_px: float) -> np.ndarray:
    """Binary mask: uniform filled circle at each site's lattice position."""
    img = Image.new("L", (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(img)
    r = radius_px
    for s in sites:
        cx, cy = s["snapped_x"], s["snapped_y"]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    return np.array(img)



# QC overlay / focus stack / raw mask


def write_qc_overlay(norm_stack: np.ndarray, pillars: list[dict], out_path: Path,
                      discarded: list[dict] | None = None) -> None:
    """Max projection of usable slices with sites circled (red = detected,
    yellow = inferred, smaller yellow = inferred with no local signal,
    blue = off-lattice) and discarded candidates crossed in grey."""
    # Contrast-stretched max projection as background
    proj = norm_stack.max(axis=0)
    lo, hi = np.percentile(proj, [1, 99.5])
    scaled = np.clip((proj - lo) / max(hi - lo, 1e-6), 0, 1)
    img = Image.fromarray((scaled * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    r = 6
    colors = {"detected": (255, 0, 0), "inferred": (255, 255, 0), "off_lattice": (0, 140, 255)}
    # Circle every site
    for p in pillars:
        x, y = p["x"], p["y"]
        color = colors.get(p.get("site_type"), (255, 0, 0))
        rr = r if p.get("site_type") != "inferred" or p.get("has_local_evidence") else r - 2
        draw.ellipse([x - rr, y - rr, x + rr, y + rr], outline=color, width=2)
    # Grey X on each discarded off-lattice candidate
    for c in discarded or []:
        x, y = c["x"], c["y"]
        grey = (150, 150, 150)
        draw.line([x - r, y - r, x + r, y + r], fill=grey, width=2)
        draw.line([x - r, y + r, x + r, y - r], fill=grey, width=2)
    img.save(out_path)


def build_focus_stack(norm_stack: np.ndarray, usable_z: list[int], pillars: list[dict],
                       default_radius_px: float) -> np.ndarray:
    """Max projection background, with each grid site's disk repainted from
    its own best-focus slice. Off-lattice entries are skipped."""
    canvas = norm_stack.max(axis=0).copy()
    # z value -> index into norm_stack
    z_to_idx = {z: i for i, z in enumerate(usable_z)}
    for p in pillars:
        if p.get("site_type") not in ("detected", "inferred"):
            continue
        z = p.get("best_z")
        if z is None or z not in z_to_idx:
            continue
        idx = z_to_idx[z]
        # Copy that slice's pixels over the pillar's disk
        r = p["radius_px"] if p.get("radius_px") else default_radius_px
        rr, cc = draw_disk((p["y"], p["x"]), r, shape=canvas.shape)
        canvas[rr, cc] = norm_stack[idx][rr, cc]
    return canvas


def save_scaled(img: np.ndarray, out_path: Path) -> None:
    """Percentile-stretch a float image to 8-bit and save it."""
    lo, hi = np.percentile(img, [1, 99.5])
    scaled = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    Image.fromarray((scaled * 255).astype(np.uint8)).save(out_path)


def build_raw_mask(image_shape: tuple[int, int], usable_idx: list[int],
                    slice_blobs: list[list[dict]]) -> np.ndarray:
    """Union of every valid blob from every usable slice (debugging aid;
    grid_mask.png is the main output)."""
    mask = np.zeros(image_shape, dtype=bool)
    for i in usable_idx:
        for b in slice_blobs[i]:
            r = b["diameter"] / 2
            rr, cc = draw_disk((b["y"], b["x"]), r, shape=image_shape)
            mask[rr, cc] = True
    return mask

# Per-stack processing


def process_stack(stack_id: str, slice_paths: list[tuple[int, Path]], out_dir: Path,
                   series_name: str, pillar_dir: Path):
    """Detect pillars, fit the lattice and write all outputs for one stack."""
    if not slice_paths:
        return None

    # --- load slices, skipping any whose shape doesn't match the first ---
    planes = []
    z_indices = []
    ref_shape = None
    for z_idx, path in slice_paths:
        intensity = load_intensity(path)
        if ref_shape is None:
            ref_shape = intensity.shape
        elif intensity.shape != ref_shape:
            print(f"      ! skipping {path.name}: shape {intensity.shape} != {ref_shape}")
            continue
        planes.append(intensity)
        z_indices.append(z_idx)

    n_total = len(slice_paths)
    n_used = len(planes)
    if n_used == 0:
        print(f"      ! no usable slices for stack '{stack_id}', skipping")
        return None

    # --- debris mask per slice ---
    debris_masks = [
        find_debris_mask(p, config.PILLAR_DEBRIS_SEED_PERCENTILE,
                          config.PILLAR_DEBRIS_MIN_AREA_FRAC, config.PILLAR_DEBRIS_DILATE_PX)
        for p in planes
    ]
    debris_frac = float(np.mean([m.mean() for m in debris_masks]))
    if debris_frac > 0:
        print(f"      debris/contamination excluded from {debris_frac * 100:.2f}% of frame on average")

    # per-slice: decide polarity and find THIS slice's own candidate
    # pillars, judged purely on the known physical size/shape -- no
    # cross-slice statistics involved yet
    corrected, slice_blobs, flipped_z = [], [], []
    for i, (plane, dmask) in enumerate(zip(planes, debris_masks)):
        fixed, blobs, was_flipped = analyze_slice(
            plane, dmask, config.PILLAR_MIN_DIAMETER_PX, config.PILLAR_MAX_DIAMETER_PX,
            config.PILLAR_MAX_ECCENTRICITY)
        corrected.append(fixed)
        slice_blobs.append(blobs)
        if was_flipped:
            flipped_z.append(z_indices[i])

    # --- usable slices: blob count >= fraction of the best slice ---
    counts = [len(b) for b in slice_blobs]
    best_count = max(counts) if counts else 0
    usable_thresh = max(config.PILLAR_SLICE_MIN_COUNT,
                         math.ceil(config.PILLAR_SLICE_USABLE_FRAC * best_count))
    usable_idx = [i for i, c in enumerate(counts) if c >= usable_thresh]
    disregarded_z = [z_indices[i] for i in range(n_used) if i not in usable_idx]

    # If nothing passes, use everything rather than fail
    if not usable_idx:
        print("      ! every slice looked too noisy to trust -- falling back to using all of them")
        usable_idx = list(range(n_used))
        disregarded_z = []

    usable_z = [z_indices[i] for i in usable_idx]
    print(f"      usable slices: {len(usable_idx)}/{n_used} (z={usable_z})")
    if disregarded_z:
        print(f"      disregarded as too noisy: z={disregarded_z}")
    if flipped_z:
        print(f"      flipped (inverted contrast) z={flipped_z}")

    # Normalised stack of usable slices (for QC, focus stack, evidence check)
    norm_stack = np.stack(
        [normalize_slice(corrected[i], config.PILLAR_BG_PERCENTILE, config.PILLAR_REF_PERCENTILE,
                          config.PILLAR_MIN_SCALE, debris_masks[i]) for i in usable_idx],
        axis=0,
    ).astype(np.float32)

    # Pool blobs from all usable slices, tagged with their z
    raw_candidates = []
    for i in usable_idx:
        for b in slice_blobs[i]:
            raw_candidates.append({**b, "z": z_indices[i]})

    # merge near-duplicate detections of the same physical pillar
    # (typically one per usable slice) BEFORE fitting the lattice
    merged_candidates = cluster_candidates(raw_candidates, config.PILLAR_MIN_DIAMETER_PX)

    # Initial lattice estimate
    lattice = estimate_lattice([(c["x"], c["y"]) for c in merged_candidates])
    pillars: list[dict] = []
    off_lattice_records: list[dict] = []
    grid_mask = None
    grid_info = None
    fit_rms = None
    n_inferred = n_off_lattice = 0

    if lattice is None:
        # Too few points: output raw candidates only
        print(f"      ! too few raw candidates ({len(merged_candidates)}) to fit a lattice -- "
              f"coords.json has raw candidates only, no grid_mask")
        for c in merged_candidates:
            pillars.append({
                "x": round(c["best_x"], 2), "y": round(c["best_y"], 2),
                "grid_x": None, "grid_y": None, "area_px": c["area"],
                "radius_px": round(c["diameter"] / 2, 2), "best_z": c["z"],
                "row": None, "col": None, "n_slice_detections": c["n_slice_detections"],
                "site_type": "detected",
            })
    else:
        # sanity guard: a corrupted pitch estimate would blow up the
        # inferred-site index range to millions of cells; bail out
        # cleanly rather than hang if that ever happens again
        h, w = ref_shape
        max_cells = 4 * (w / max(lattice["pitch_x"], 1e-6)) * (h / max(lattice["pitch_y"], 1e-6))
        if max_cells > 200_000:
            print(f"      ! lattice fit looks wrong (pitch=({lattice['pitch_x']:.2f},"
                  f"{lattice['pitch_y']:.2f})px implies {max_cells:.0f} grid cells) -- "
                  f"discarding the fit, coords.json has raw candidates only")
            lattice = None
            for c in merged_candidates:
                pillars.append({
                    "x": round(c["best_x"], 2), "y": round(c["best_y"], 2),
                    "grid_x": None, "grid_y": None, "area_px": c["area"],
                    "radius_px": round(c["diameter"] / 2, 2), "best_z": c["z"],
                    "row": None, "col": None, "n_slice_detections": c["n_slice_detections"],
                    "site_type": "detected",
                })

    if lattice is not None:
        # align the lattice to the majority of the detections before
        # anything is snapped to it, so outliers do not shift the grid
        # that then decides what counts as an outlier
        lattice, _, fit_rms = refine_lattice(
            [(c["x"], c["y"]) for c in merged_candidates], lattice,
            config.OFF_LATTICE_FRAC, cfg("LATTICE_REFINE_ITERS", 3),
            cfg("LATTICE_MAX_PITCH_DRIFT", 0.25))

        # Snap candidates; group on-lattice ones by (row, col)
        snapped = snap_to_lattice([(c["x"], c["y"]) for c in merged_candidates], lattice,
                                   config.OFF_LATTICE_FRAC)
        on_site: dict[tuple[int, int], list[dict]] = {}
        off_site: list[dict] = []
        for c, s in zip(merged_candidates, snapped):
            if s["on_lattice"]:
                on_site.setdefault((s["row"], s["col"]), []).append(c)
            else:
                off_site.append({**c, "residual_px": round(s["residual_px"], 2)})

        # Derived sizes: evidence search radius, pillar radius, edge-cut limit
        avg_pitch = (lattice["pitch_x"] + lattice["pitch_y"]) / 2
        search_r = config.OFF_LATTICE_FRAC * avg_pitch
        radius_px = config.PILLAR_R_FRAC * avg_pitch
        max_cut = float(cfg("PILLAR_EDGE_MAX_CUT_FRAC", 0.5))

        # --- the grid: every cell in the extent, present or not ---
        row_min, row_max, col_min, col_max = grid_extent(
            lattice, ref_shape, list(on_site.keys()),
            cfg("GRID_EXTENT", "detections"), int(cfg("GRID_PAD_CELLS", 0)),
            radius_px, max_cut, bool(cfg("GRID_EXTEND_TO_EDGE", True)))

        n_rows = max(row_max - row_min + 1, 0)
        n_cols = max(col_max - col_min + 1, 0)
        n_clipped = n_dropped_cut = 0
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                # Ideal lattice position of this cell
                sx = lattice["phase_x"] + col * lattice["pitch_x"]
                sy = lattice["phase_y"] + row * lattice["pitch_y"]
                # a site at the image border is kept as long as enough of
                # the pillar is still in frame; the fraction that is not
                # is recorded so a downstream stage can weight or drop it
                cut = disk_cut_fraction(sx, sy, radius_px, ref_shape)
                if cut > max_cut:
                    n_dropped_cut += 1
                    continue
                if cut > 0:
                    n_clipped += 1
                # row/col re-based to 0 so they index the grid directly
                r0, c0 = row - row_min, col - col_min
                group = on_site.get((row, col))
                if group:
                    # position AND pixel-sampling both come from the same
                    # (best) observation -- averaging position across the
                    # group while sampling best_z's own pixels for
                    # painting would let them drift apart for a
                    # noisier/edge-clipped site, enough to miss the real
                    # bright disk in that slice's own data
                    best = max(group, key=lambda c: c["diameter"])
                    pillars.append({
                        "x": round(best["x"], 2), "y": round(best["y"], 2),
                        "grid_x": round(sx, 2), "grid_y": round(sy, 2),
                        "area_px": best["area"], "radius_px": round(best["diameter"] / 2, 2),
                        "best_z": best["z"], "row": r0, "col": c0,
                        "n_slice_detections": sum(c["n_slice_detections"] for c in group),
                        "has_local_evidence": True,
                        "cut_frac": round(cut, 3), "clipped": cut > 0,
                        "site_type": "detected",
                    })
                else:
                    # filled unconditionally - the local check is recorded
                    # for downstream filtering, it is no longer a gate
                    n_inferred += 1
                    pillars.append({
                        "x": round(sx, 2), "y": round(sy, 2),
                        "grid_x": round(sx, 2), "grid_y": round(sy, 2),
                        "area_px": None, "radius_px": None,
                        "best_z": None, "row": r0, "col": c0, "n_slice_detections": 0,
                        "has_local_evidence": bool(local_site_evidence(
                            norm_stack, sx, sy, search_r, config.PILLAR_VOTE_THRESH,
                            config.PILLAR_MIN_VOTE_FRACTION)),
                        "cut_frac": round(cut, 3), "clipped": cut > 0,
                        "site_type": "inferred",
                    })

        # --- off-lattice: discarded, not carried as sites ---
        n_off_lattice = len(off_site)
        off_lattice_records = [
            {"x": round(c["x"], 2), "y": round(c["y"], 2),
             "area_px": c["area"], "radius_px": round(c["diameter"] / 2, 2),
             "best_z": c["z"], "residual_px": c["residual_px"],
             "n_slice_detections": c["n_slice_detections"]}
            for c in off_site
        ]
        # Optionally keep them as sites after all
        if cfg("KEEP_OFF_LATTICE", False):
            for c in off_site:
                pillars.append({
                    "x": round(c["x"], 2), "y": round(c["y"], 2),
                    "grid_x": None, "grid_y": None,
                    "area_px": c["area"], "radius_px": round(c["diameter"] / 2, 2),
                    "best_z": c["z"], "row": None, "col": None,
                    "n_slice_detections": c["n_slice_detections"],
                    "has_local_evidence": None,
                    "cut_frac": None, "clipped": None,
                    "site_type": "off_lattice",
                })

        # Grid mask from the ideal positions of all grid sites
        grid_sites = [
            {"snapped_x": p["grid_x"], "snapped_y": p["grid_y"]}
            for p in pillars if p["site_type"] in ("detected", "inferred")
        ]
        grid_mask = build_grid_mask(ref_shape, grid_sites, radius_px)
        n_grid = len(grid_sites)
        n_detected = n_grid - n_inferred
        n_weak = sum(1 for p in pillars
                     if p["site_type"] == "inferred" and not p["has_local_evidence"])
        # Grid summary for coords.json
        grid_info = {
            "n_rows": n_rows, "n_cols": n_cols,
            "row_min": row_min, "col_min": col_min,
            "extent_mode": cfg("GRID_EXTENT", "detections"),
            "pad_cells": int(cfg("GRID_PAD_CELLS", 0)),
            "n_clipped": n_clipped,
            "n_dropped_over_cut": n_dropped_cut,
            "edge_max_cut_frac": max_cut,
            # complete = every admissible cell in the box is present;
            # cells dropped for hanging too far off the frame are counted
            # separately rather than silently making the grid "incomplete"
            "complete": n_grid == n_rows * n_cols - n_dropped_cut,
            "fit_rms_px": None if fit_rms is None or not math.isfinite(fit_rms)
            else round(fit_rms, 2),
        }

        #  console report and sanity warnings 
        rms_txt = (f"{fit_rms:.2f}px" if fit_rms is not None and math.isfinite(fit_rms)
                   else "n/a")
        print(f"      lattice pitch=({lattice['pitch_x']:.1f},{lattice['pitch_y']:.1f})px "
              f"rms={rms_txt} -> {n_rows}x{n_cols} = {n_grid} grid sites "
              f"({n_detected} detected + {n_inferred} filled), "
              f"{n_off_lattice} off-lattice discarded")
        if n_clipped or n_dropped_cut:
            print(f"      edge sites: {n_clipped} kept clipped by the frame "
                  f"(<={max_cut * 100:.0f}% cut off), {n_dropped_cut} dropped for being "
                  f"cut more than that")
        if n_weak:
            print(f"      note: {n_weak}/{n_inferred} filled sites show no local signal "
                  f"(kept anyway; 'has_local_evidence': false in coords.json)")
        if n_grid and n_off_lattice > 0.1 * n_grid:
            print(f"      ! {n_off_lattice} candidates ({100 * n_off_lattice / n_grid:.0f}% "
                  f"of the grid) missed the lattice - check {stack_id}_qc.png, the fit may "
                  f"be wrong rather than the detections")
        if n_grid and n_inferred > 0.5 * n_grid:
            print(f"      ! over half the grid is filled rather than detected - the grid is "
                  f"mostly extrapolation, check the QC overlay before using it")

    # Sort sites top-to-bottom, left-to-right and number them from 1
    pillars.sort(key=lambda p: (p["y"], p["x"]))
    for i, p in enumerate(pillars, start=1):
        p["id"] = i
    # Put "id" first in each record
    pillars = [{"id": p["id"], **{k: v for k, v in p.items() if k != "id"}} for p in pillars]

    out_dir.mkdir(parents=True, exist_ok=True)

    #  write raw mask 
    raw_mask = build_raw_mask(ref_shape, usable_idx, slice_blobs)
    mask_path = out_dir / f"{stack_id}_mask.png"
    Image.fromarray((raw_mask * 255).astype(np.uint8)).save(mask_path)

    #  write grid mask (if a lattice was fitted) 
    grid_mask_path = None
    if grid_mask is not None:
        grid_mask_path = out_dir / f"{stack_id}_grid_mask.png"
        Image.fromarray(grid_mask).save(grid_mask_path)

    #  coords.json: sites, lattice, grid info, slice usage and parameters 
    coords = {
        "series": series_name,
        "pillar_folder": str(pillar_dir),
        "stack_id": stack_id,
        "image_shape": list(ref_shape),
        "n_slices_found": n_total,
        "n_slices_used": n_used,
        "z_indices_used": z_indices,
        "usable_z": usable_z,
        "disregarded_z": disregarded_z,
        "flipped_z": flipped_z,
        "per_slice_blob_counts": dict(zip(z_indices, counts)),
        "lattice": lattice,
        "grid": grid_info,
        "off_lattice_discarded": off_lattice_records,
        "n_off_lattice_discarded": (0 if cfg("KEEP_OFF_LATTICE", False)
                                    else len(off_lattice_records)),
        "params": {
            "min_diameter_px": config.PILLAR_MIN_DIAMETER_PX,
            "max_diameter_px": config.PILLAR_MAX_DIAMETER_PX,
            "max_eccentricity": config.PILLAR_MAX_ECCENTRICITY,
            "slice_usable_frac": config.PILLAR_SLICE_USABLE_FRAC,
            "slice_min_count": config.PILLAR_SLICE_MIN_COUNT,
            "bg_percentile": config.PILLAR_BG_PERCENTILE,
            "ref_percentile": config.PILLAR_REF_PERCENTILE,
            "vote_thresh": config.PILLAR_VOTE_THRESH,
            "min_vote_fraction": config.PILLAR_MIN_VOTE_FRACTION,
            "debris_frac_excluded": round(debris_frac, 4),
            "off_lattice_frac": config.OFF_LATTICE_FRAC,
            "grid_extent": cfg("GRID_EXTENT", "detections"),
            "grid_pad_cells": int(cfg("GRID_PAD_CELLS", 0)),
            "grid_extend_to_edge": bool(cfg("GRID_EXTEND_TO_EDGE", True)),
            "edge_max_cut_frac": float(cfg("PILLAR_EDGE_MAX_CUT_FRAC", 0.5)),
            "keep_off_lattice": bool(cfg("KEEP_OFF_LATTICE", False)),
            "lattice_refine_iters": int(cfg("LATTICE_REFINE_ITERS", 3)),
            "lattice_max_pitch_drift": cfg("LATTICE_MAX_PITCH_DRIFT", 0.25),
            "full_grid": True,
        },
        "n_pillars": len(pillars),
        "n_grid_sites": (sum(1 for p in pillars
                             if p["site_type"] in ("detected", "inferred"))
                        if lattice is not None else None),
        "pillars": pillars,
    }
    json_path = out_dir / f"{stack_id}_coords.json"
    json_path.write_text(json.dumps(coords, indent=2))

    # --- optional QC images ---
    if config.WRITE_PILLAR_QC_OVERLAY:
        write_qc_overlay(norm_stack, pillars, out_dir / f"{stack_id}_qc.png",
                          discarded=off_lattice_records)

    if config.WRITE_PILLAR_FOCUS_STACK:
        default_r = (lattice["pitch_x"] + lattice["pitch_y"]) / 4 * config.PILLAR_R_FRAC \
            if lattice is not None else config.PILLAR_MIN_DIAMETER_PX / 2
        focus_stack = build_focus_stack(norm_stack, usable_z, pillars, default_r)
        save_scaled(focus_stack, out_dir / f"{stack_id}_focus_stack.png")

    # Number of grid sites (or raw candidates if no lattice)
    n_grid_sites = (sum(1 for p in pillars if p["site_type"] in ("detected", "inferred"))
                    if lattice is not None else len(pillars))
    return mask_path, json_path, grid_mask_path, n_grid_sites


def process_pillar_folder(pillar_dir: Path, root_dir: Path) -> dict[str, int]:
    """Process every stack in one pillar folder; return {stack_id: n_sites}."""
    # Series name = parent folder path relative to root
    try:
        series_name = pillar_dir.parent.relative_to(root_dir).as_posix()
    except ValueError:
        series_name = pillar_dir.parent.name
    if not series_name:
        series_name = pillar_dir.parent.name

    stacks, skipped = group_into_stacks(pillar_dir)
    if skipped:
        print(f"    (ignored {len(skipped)} file(s) with no recognisable z-index token)")
    if not stacks:
        print("    No z-stacks found.")
        return {}

    # Outputs go to PILLAR_DETECT_DIR/<series>
    out_dir = Path(config.PILLAR_DETECT_DIR) / series_name
    results = {}
    for stack_id, slice_paths in stacks.items():
        print(f"    Stack '{stack_id}': {len(slice_paths)} slice(s)")
        result = process_stack(stack_id, slice_paths, out_dir, series_name, pillar_dir)
        if result:
            _, json_path, grid_mask_path, n_grid_sites = result
            label = grid_mask_path.name if grid_mask_path else json_path.name
            print(f"      -> {n_grid_sites} grid sites -> {label}")
            results[stack_id] = n_grid_sites
    return results


def main():
    # Find every pillar folder under ROOT_DIR
    root_dir = Path(config.ROOT_DIR)
    print(f"Scanning {root_dir} for '{config.PILLAR_DIRNAME}' folders...")
    pillar_dirs = find_pillar_folders(root_dir)
    if not pillar_dirs:
        print(f"No '{config.PILLAR_DIRNAME}' folders found.")
        return

    print(f"Found {len(pillar_dirs)} pillar folder(s).")
    start = time.time()
    per_stack_counts: dict[str, int] = {}

    # Process each folder, collecting site counts per stack
    for pillar_dir in pillar_dirs:
        print(f"\n--- {pillar_dir.relative_to(root_dir)} ---")
        for stack_id, n_grid_sites in process_pillar_folder(pillar_dir, root_dir).items():
            per_stack_counts[f"{pillar_dir}::{stack_id}"] = n_grid_sites

    # Final summary
    elapsed = time.time() - start
    print(f"\n{'=' * 50}")
    print("PILLAR DETECTION COMPLETE")
    print(f"{'=' * 50}")
    print(f"Stacks processed  : {len(per_stack_counts)}")
    print(f"Total grid sites  : {sum(per_stack_counts.values())}")
    print(f"Time elapsed      : {elapsed:.2f}s")
    print(f"Output written to : {config.PILLAR_DETECT_DIR}")


if __name__ == "__main__":
    main()
