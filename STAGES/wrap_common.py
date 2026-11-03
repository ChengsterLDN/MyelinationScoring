"""
Shared utilities for the wrap-scoring pipeline
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

import config

# Filename parsing (mirrors PIL_DET's own regexes)


# "_z<n>" followed by "_" or "." -- the z-index token in a slice filename
_Z_TOKEN_RE = re.compile(r"_z(\d+)(?=[_.])", re.IGNORECASE)
# Matches the "_c<N>_<colour>" channel token so it can be stripped to get
# the field-of-view/position key shared between a pillar stack and its
# matching mbp stack, e.g. "..._c2_yellow_t0" and "..._c1_cyan_t0" both
# reduce to "..._t0". Adjust this if your MBP export uses a different
# channel-labelling convention.
CHANNEL_TOKEN_RE = re.compile(r"_c\d+_[A-Za-z]+", re.IGNORECASE)


def _parse_z_filename(filename: str) -> tuple[str, int] | None:
    """(stack_id, z) from a slice filename, or None if it has no z token."""
    match = _Z_TOKEN_RE.search(filename)
    if not match:
        return None
    z = int(match.group(1))
    # Stack id = filename with the z token and ".png" removed
    stack_id = filename[: match.start()] + filename[match.end():]
    stack_id = re.sub(r"\.png$", "", stack_id, flags=re.IGNORECASE)
    return stack_id, z


def strip_channel_token(stack_id: str) -> str:
    """Position key shared by pillar and mbp channels of one stack,
    e.g. "1_2_c2_yellow_t0" -> "1_2_t0"."""
    return CHANNEL_TOKEN_RE.sub("", stack_id)


# Z-stack slices


@dataclass(frozen=True)
class SliceRef:
    """One z-slice file."""
    path: Path
    z: int
    key: str  # human-readable label for logs/manifest, e.g. "z014"


def list_slices(directory: Path, stack_id: str | None = None) -> list[SliceRef]:
    """Every PNG z-slice in directory, sorted by z; if stack_id is given,
    only that stack's files (a folder can mix several stacks)."""
    directory = Path(directory)
    refs = []
    for png_path in directory.glob("*.png"):
        # Skip files with no z token
        parsed = _parse_z_filename(png_path.name)
        if parsed is None:
            continue
        sid, z = parsed
        # Skip other stacks' files
        if stack_id is not None and sid != stack_id:
            continue
        refs.append(SliceRef(path=png_path, z=z, key=f"z{z:03d}"))
    refs.sort(key=lambda r: r.z)
    return refs


def image_shape(refs: list[SliceRef]) -> tuple[int, int]:
    """(height, width) from the first slice."""
    if not refs:
        raise ValueError("no slices to determine image shape from")
    arr = np.array(Image.open(refs[0].path))
    return arr.shape[0], arr.shape[1]


def load_slice(ref: SliceRef) -> np.ndarray:
    """One slice as float32 intensity, using the per-pixel channel MAX so a
    coloured LUT export isn't diluted by its empty channels."""
    arr = np.array(Image.open(ref.path).convert("RGB"), dtype=np.float32)
    return arr.max(axis=-1)


def load_slice_smoothed(refs: list[SliceRef], zi: int, z_smooth: int) -> np.ndarray:
    """Mean of slice zi and up to z_smooth neighbours each side (a denoiser,
    not a projection). z_smooth = 0 loads the single slice."""
    if z_smooth <= 0:
        return load_slice(refs[zi])
    # Window clipped at the ends of the stack
    lo = max(0, zi - z_smooth)
    hi = min(len(refs), zi + z_smooth + 1)
    planes = [load_slice(refs[i]) for i in range(lo, hi)]
    return np.mean(planes, axis=0)


def select_slices(refs: list[SliceRef], z_range: tuple[int, int] | None = None) -> list[int]:
    """Indices into refs whose z value lies in z_range (inclusive); None = all."""
    if z_range is None:
        return list(range(len(refs)))
    lo, hi = z_range
    return [i for i, r in enumerate(refs) if lo <= r.z <= hi]


def normalise(plane: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.7) -> np.ndarray:
    """Percentile-stretch an intensity plane to 0-255 uint8 (crop/QC input)."""
    lo, hi = np.percentile(plane, [lo_pct, hi_pct])
    # Map [lo, hi] to [0, 1], clip, scale to 8-bit
    scaled = np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)



# Site registry


@dataclass
class Site:
    """One pillar position to score."""
    site_id: int
    x: float
    y: float
    source: str                    # "detected" | "off_lattice" | "inferred"
    grid_i: int | None = None      # lattice row  (None for off_lattice)
    grid_j: int | None = None      # lattice col  (None for off_lattice)
    area_px: int | None = None
    radius_px: float | None = None


@dataclass
class Lattice:
    """Fitted pillar lattice (axis-aligned)."""
    pitch: float            # mean of pitch_x, pitch_y
    a: np.ndarray           # lattice vector 1, (pitch_x, 0)
    b: np.ndarray           # lattice vector 2, (0, pitch_y)
    rotation_deg: float     # always 0.0 -- detect_pillars.py's fit is
                             # axis-aligned only; see build_site_table
    median_resid: float     # median detected-site distance from its own
                             # ideal lattice position, px


class SiteTable:
    """All sites for one stack, plus the lattice and a fit-quality flag."""

    def __init__(self, sites: list[Site], lattice: Lattice, lattice_ok: bool,
                 lattice_note: str):
        self.sites = sites
        self.lattice = lattice
        self.lattice_ok = lattice_ok
        self.lattice_note = lattice_note

    def __len__(self) -> int:
        return len(self.sites)

    @property
    def xy(self) -> np.ndarray:
        """(N, 2) array of every site's (x, y), indexed by site_id."""
        if not self.sites:
            return np.zeros((0, 2))
        # NaN for any unused site_id
        n = max(s.site_id for s in self.sites) + 1
        out = np.full((n, 2), np.nan)
        for s in self.sites:
            out[s.site_id] = (s.x, s.y)
        return out

    @property
    def detected_mask(self) -> np.ndarray:
        """Bool array by site_id: True where source == 'detected'."""
        n = max((s.site_id for s in self.sites), default=-1) + 1
        out = np.zeros(n, dtype=bool)
        for s in self.sites:
            out[s.site_id] = s.source == "detected"
        return out

    def counts_by_source(self) -> dict[str, int]:
        """Number of sites of each source type (zeros included)."""
        c = Counter(s.source for s in self.sites)
        return {"detected": c.get("detected", 0), "off_lattice": c.get("off_lattice", 0),
                "inferred": c.get("inferred", 0)}


@dataclass
class CoordsFile:
    """Thin wrapper around one parsed <stack>_coords.json."""
    pillars: list[dict]
    lattice: dict | None
    image_shape: tuple[int, int] | None
    stack_id: str
    path: Path

    def __len__(self) -> int:
        return len(self.pillars)


def _fallback_pitch(sites: list[Site]) -> float:
    """Median nearest-neighbour distance; used only when coords.json has no
    lattice fit (lattice_ok is then False)."""
    if len(sites) < 2:
        return 0.0
    pts = np.array([(s.x, s.y) for s in sites])
    tree = cKDTree(pts)
    # k=2: nearest neighbour other than the point itself
    dists, _ = tree.query(pts, k=2)
    return float(np.median(dists[:, 1]))


def build_site_table(coords: CoordsFile, image_shape: tuple[int, int],
                      min_sites_for_ok: int = 10,
                      include_sources: tuple[str, ...] | None = None) -> SiteTable:
    """Turn one coords.json into a SiteTable, trusting its positions and
    site types. include_sources optionally restricts which site types are
    kept at all; fit quality is always measured on the full file."""
    # Optionally drop unwanted site types
    pillars = coords.pillars
    if include_sources is not None:
        pillars = [p for p in pillars if p.get("site_type", "detected") in include_sources]

    # One Site per kept pillar, site_id = its index
    sites = [
        Site(
            site_id=i, x=float(p["x"]), y=float(p["y"]),
            source=p.get("site_type", "detected"),
            grid_i=p.get("row"), grid_j=p.get("col"),
            area_px=p.get("area_px"), radius_px=p.get("radius_px"),
        )
        for i, p in enumerate(pillars)
    ]

    lat_dict = coords.lattice
    if lat_dict is not None:
        # Lattice from detect_pillars' fit
        pitch = (lat_dict["pitch_x"] + lat_dict["pitch_y"]) / 2
        a = np.array([lat_dict["pitch_x"], 0.0])
        b = np.array([0.0, lat_dict["pitch_y"]])
        # detect_pillars.py fits an axis-aligned lattice only (independent
        # pitch in x/y, but no rotation or shear term) -- rotation_deg is
        # therefore always 0 here. If a rotated array ever needs fitting,
        # this is the one place that assumption would need to change.
        rotation_deg = 0.0
        # Distance of each detected pillar from its ideal grid position
        resids = [
            math.hypot(p["x"] - p["grid_x"], p["y"] - p["grid_y"])
            for p in coords.pillars
            if p.get("site_type") == "detected" and p.get("grid_x") is not None
        ]
        median_resid = float(np.median(resids)) if resids else float("nan")
    else:
        # No fit available: rough pitch from nearest neighbours
        pitch = _fallback_pitch(sites)
        a, b = np.array([pitch, 0.0]), np.array([0.0, pitch])
        rotation_deg = 0.0
        median_resid = float("nan")

    # Decide whether the lattice fit is trustworthy
    n_on_grid = sum(1 for s in sites if s.source in ("detected", "inferred"))
    if lat_dict is None:
        lattice_ok, note = False, "no lattice fit in coords.json (too few raw detections upstream)"
    elif n_on_grid < min_sites_for_ok:
        lattice_ok, note = False, f"only {n_on_grid} on-grid sites (< {min_sites_for_ok})"
    elif not np.isfinite(pitch) or pitch <= 0:
        lattice_ok, note = False, f"pitch looks wrong ({pitch!r})"
    else:
        lattice_ok, note = True, ""

    lattice = Lattice(pitch=pitch, a=a, b=b, rotation_deg=rotation_deg,
                       median_resid=median_resid)
    return SiteTable(sites, lattice, lattice_ok, note)


def site_table_rows(table: SiteTable) -> list[dict]:
    """SiteTable as a list of dicts for CSV output."""
    return [
        {"site_id": s.site_id, "x": round(s.x, 2), "y": round(s.y, 2), "source": s.source,
         "grid_i": s.grid_i, "grid_j": s.grid_j, "area_px": s.area_px, "radius_px": s.radius_px}
        for s in table.sites
    ]


# CSV / misc small helpers


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV (columns from the first row), creating folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # No rows: write an empty file
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def preproc_dict() -> dict:
    """Crop geometry saved with crops and copied into the model, so
    scoring uses exactly the settings the model was trained on."""
    return {
        "context_pitches": config.CONTEXT_PITCHES,
        "out_size": config.OUT_SIZE,
        "pillar_r_frac": config.PILLAR_R_FRAC,
        "slide_at_edges": config.SLIDE_AT_EDGES,
        "z_smooth": config.Z_SMOOTH,
    }


def describe_series(series: "Series", table: SiteTable, n_slices: int, n_sel_slices: int) -> None:
    """Print a one-line summary of a series/stack."""
    c = table.counts_by_source()
    flag = "" if table.lattice_ok else f"   <-- {table.lattice_note}"
    print(f"{series.name:<30} {len(table):>4} sites "
          f"({c['detected']} detected, {c['off_lattice']} off-lattice, {c['inferred']} inferred)  "
          f"{n_sel_slices}/{n_slices} slices in range  pitch={table.lattice.pitch:.1f}px{flag}")


# Crop building -- shared by make_crops.py and score_series.py


def build_crop(field: np.ndarray, all_xy: np.ndarray, site_id: int, pitch: float,
                context_pitches: float | None = None, out_size: int | None = None,
                pillar_r_frac: float | None = None,
                slide_at_edges: bool | None = None) -> Image.Image:
    """One RGB crop for site_id: R = MBP, G = pillar footprints, B = marker
    on the target pillar. Geometry defaults to config; near the border the
    window slides inward (so the target can be off-centre)."""
    # Fill unspecified geometry from config
    context_pitches = config.CONTEXT_PITCHES if context_pitches is None else context_pitches
    out_size = config.OUT_SIZE if out_size is None else out_size
    pillar_r_frac = config.PILLAR_R_FRAC if pillar_r_frac is None else pillar_r_frac
    slide_at_edges = config.SLIDE_AT_EDGES if slide_at_edges is None else slide_at_edges

    # Square window of side context_pitches * pitch, centred on the target
    h, w = field.shape
    side = context_pitches * pitch
    cx, cy = all_xy[site_id]

    x0, y0 = cx - side / 2, cy - side / 2
    x1, y1 = x0 + side, y0 + side

    # Shift the window back inside the image if it overhangs an edge
    if slide_at_edges:
        if x0 < 0:
            x1 -= x0; x0 = 0.0
        if y0 < 0:
            y1 -= y0; y0 = 0.0
        if x1 > w:
            x0 -= (x1 - w); x1 = float(w)
        if y1 > h:
            y0 -= (y1 - h); y1 = float(h)
    # Clip to image bounds regardless
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    x1 = min(float(w), x1)
    y1 = min(float(h), y1)

    # Integer pixel bounds, at least 1 px wide/high
    x0i, y0i = int(round(x0)), int(round(y0))
    x1i, y1i = max(x0i + 1, int(round(x1))), max(y0i + 1, int(round(y1)))
    x1i, y1i = min(w, x1i), min(h, y1i)

    # R channel: copy the MBP pixels
    crop_w, crop_h = x1i - x0i, y1i - y0i
    r_channel = np.zeros((crop_h, crop_w), dtype=np.uint8)
    src = field[y0i:y1i, x0i:x1i]
    r_channel[: src.shape[0], : src.shape[1]] = src

    # Blank G (pillar mask) and B (target marker) channels
    pillar_r = pillar_r_frac * pitch
    g_img = Image.new("L", (crop_w, crop_h), 0)
    gd = ImageDraw.Draw(g_img)
    b_img = Image.new("L", (crop_w, crop_h), 0)
    bd = ImageDraw.Draw(b_img)

    # Draw a disk for every site inside (or just outside) the window
    target_lx = target_ly = None
    margin = pillar_r + 2
    for sid in range(all_xy.shape[0]):
        sx, sy = all_xy[sid]
        if np.isnan(sx):
            continue
        if not (x0i - margin <= sx <= x1i + margin and y0i - margin <= sy <= y1i + margin):
            continue
        # Position relative to the crop
        lx, ly = sx - x0i, sy - y0i
        gd.ellipse([lx - pillar_r, ly - pillar_r, lx + pillar_r, ly + pillar_r], fill=255)
        if sid == site_id:
            target_lx, target_ly = lx, ly

    if target_lx is None:   # target itself fell outside the window bounds check above
        target_lx, target_ly = cx - x0i, cy - y0i
    # Small filled marker at the target's centre
    marker_r = max(2.0, pillar_r * 0.35)
    bd.ellipse([target_lx - marker_r, target_ly - marker_r,
                target_lx + marker_r, target_ly + marker_r], fill=255)

    # Stack channels and resize to the model's input size
    rgb = np.stack([r_channel, np.array(g_img), np.array(b_img)], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((out_size, out_size), Image.LANCZOS)


# Series discovery


@dataclass
class Series:
    """One field of view: an MBP stack paired with its pillar coords.json."""
    name: str            # "<series folder>/<position key>", unique per stack
    mbp_dir: Path         # the mbp folder (may hold more than one stack)
    mbp_stack_id: str     # which stack within mbp_dir this Series is
    coords_path: Path     # the matching detect_pillars.py coords.json

    def coords(self) -> CoordsFile:
        """Load and wrap the coords.json for this stack."""
        data = json.loads(self.coords_path.read_text())
        shape = tuple(data["image_shape"]) if data.get("image_shape") else None
        return CoordsFile(
            pillars=data.get("pillars", []), lattice=data.get("lattice"),
            image_shape=shape, stack_id=data.get("stack_id", ""), path=self.coords_path,
        )


def discover_series(root_dir: Path) -> list[Series]:
    """Every (series folder, stack) with both a coords.json under
    PILLAR_DETECT_DIR and a matching MBP stack (by position key)."""
    root_dir = Path(root_dir)
    series_list: list[Series] = []

    # Every folder named MBP_DIRNAME under root
    mbp_dirs = sorted(p for p in root_dir.rglob("*")
                       if p.is_dir() and p.name.lower() == config.MBP_DIRNAME.lower())

    for mbp_dir in mbp_dirs:
        # Series name = parent folder path relative to root
        series_folder = mbp_dir.parent
        try:
            series_name = series_folder.relative_to(root_dir).as_posix()
        except ValueError:
            series_name = series_folder.name
        series_name = series_name or series_folder.name

        # Map position key -> MBP stack id for stacks in this folder
        mbp_stacks: dict[str, str] = {}   # position key -> mbp stack id
        for png_path in mbp_dir.glob("*.png"):
            parsed = _parse_z_filename(png_path.name)
            if parsed is None:
                continue
            stack_id, _ = parsed
            mbp_stacks.setdefault(strip_channel_token(stack_id), stack_id)

        # detect_pillars output for this series
        coords_dir = Path(config.PILLAR_DETECT_DIR) / series_name
        if not coords_dir.exists():
            print(f"  ! no detect_pillars.py output for series '{series_name}' "
                  f"(expected {coords_dir}) -- run detect_pillars first, skipping")
            continue

        # Pair each pillar coords file with its MBP stack
        for coords_path in sorted(coords_dir.glob("*_coords.json")):
            pillar_stack_id = coords_path.name[: -len("_coords.json")]
            key = strip_channel_token(pillar_stack_id)
            mbp_stack_id = mbp_stacks.get(key)
            if mbp_stack_id is None:
                print(f"  ! no matching mbp stack for {series_name}/{pillar_stack_id} "
                      f"(looked for position key '{key}' in {mbp_dir}) -- skipping")
                continue
            series_list.append(Series(
                name=f"{series_name}/{key}", mbp_dir=mbp_dir,
                mbp_stack_id=mbp_stack_id, coords_path=coords_path,
            ))

    return series_list

# Augmentation


class DihedralTransform:
    """Random one of the 8 exact dihedral operations (4 rotations x optional
    mirror): lossless augmentation that matches the lattice's symmetry."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, img: Image.Image) -> Image.Image:
        # Rotate by 0/90/180/270 degrees
        k = int(self.rng.integers(0, 4))
        if k:
            img = img.rotate(90 * k)
        # Mirror with probability 0.5
        if self.rng.integers(0, 2):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
