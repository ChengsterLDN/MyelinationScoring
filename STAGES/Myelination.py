#!/usr/bin/env python
"""
wrap_pipeline.py -- the whole myelin wrap-scoring pipeline in one file.

    python wrap_pipeline.py                      settings window, then run
    python wrap_pipeline.py --headless           last saved settings, no window
    python wrap_pipeline.py --headless --mode score
    python wrap_pipeline.py --headless --mode none --nuclei     nuclei count only

Stages, in order:

    A  LIF extraction      .lif -> ROOT_DIR/<lif name>/<series>/{nuclei,mbp,pillar}/*.png
    0  Pillar detection    -> RESULTS_DIR/pillar_detection/<series>/<stack>_coords.json
    then ONE of
    TRAIN   1 Crops -> 2 Label (GUI) -> 3 Train (fresh) | 3B Continue training
            (optionally followed by 4 Score with the new model)
    SCORE   4 Score (per-slice + per-site MIP)
    NONE    (no myelin stages - only A / 0 / N as ticked)
    then, if ticked
    N  Nuclei count        nuclei/*.png -> one MIP per stack -> Fiji + StarDist
                           -> RESULTS_DIR/nuclei/<series>/..., nuclei_summary.csv

Every stage can be ticked off in the window, so a run can resume part-way
(e.g. skip extraction/detection when they already exist, or skip cropping
and go straight back into labelling).

Settings live in wrap_pipeline_settings.json next to this file and replace
config.py. torch / transformers / readlif are only imported by the stages
that need them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, measure
from skimage.draw import disk as draw_disk
from skimage.morphology import disk as disk_selem

try:                                    # GUI pieces are optional (headless scoring)
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk
except ImportError:                     # pragma: no cover
    tk = filedialog = messagebox = ttk = ImageTk = None


class PipelineError(Exception):
    """A stage cannot continue; the message says why. Stops the run."""


# =============================================================================
# CONFIGURATION  (replaces config.py; edited through the settings window)
# =============================================================================

NUM_LABELS = 4
CLASS_NAMES = {0: "none", 1: "partial_lt50", 2: "mostly", 3: "full"}
SETTINGS_PATH = Path(__file__).resolve().with_name("wrap_pipeline_settings.json")

BACKBONES = ["google/vit-base-patch16-224-in21k", "facebook/convnext-tiny-224",
             "microsoft/resnet-18", "facebook/dinov2-small"]
AGG_RULES = ["max", "maxmip", "consec2", "consec3", "meanprob", "q90"]
MIP_PROJ = ["max", "topk"]                      # projection methods
MIP_DULL = ["match", "gamma", "scale", "none"]  # intensity-dulling methods

DEFAULTS: dict = {
    # --- run plan ------------------------------------------------------------
    "MODE": "score",                 # "train" | "score" | "none"
    "RUN_EXTRACT": True,
    "LIF_FILES": [],
    "RUN_DETECT": True,
    "RUN_CROPS": True,
    "CROP_ALL_SITES": False,
    "CROP_ALL_SLICES": False,
    "RUN_LABEL": True,
    "RUN_TRAIN": True,
    "TRAIN_KIND": "fresh",           # "fresh" | "continue"
    "SCORE_AFTER_TRAIN": False,
    "RUN_NUCLEI": False,

    # --- paths ---------------------------------------------------------------
    "ROOT_DIR": r"G:\Training Datasets\Pup",
    "CROPS_DIR": r"G:\Training Datasets\Pup\Crops",
    "RESULTS_DIR": r"G:\Training Datasets\Pup\Results",
    "MODEL_DIR": r"G:\Training Datasets\Modelv4D",
    "PARENT_MODEL_DIR": "",          # continue-training only
    "OLD_CROPS_DIR": "",             # continue-training only (replay set)

    # --- pillar detection ----------------------------------------------------
    "PILLAR_DIRNAME": "pillar",
    "MBP_DIRNAME": "mbp",
    "PILLAR_MIN_DIAMETER_PX": 35,
    "PILLAR_MAX_DIAMETER_PX": 50,
    "PILLAR_MAX_ECCENTRICITY": 0.5,
    "PILLAR_BG_PERCENTILE": 50.0,
    "PILLAR_REF_PERCENTILE": 99.9,
    "PILLAR_MIN_SCALE": 1.0,
    "PILLAR_DEBRIS_SEED_PERCENTILE": 99.0,
    "PILLAR_DEBRIS_MIN_AREA_FRAC": 0.00005,
    "PILLAR_DEBRIS_DILATE_PX": 15,
    "PILLAR_SLICE_USABLE_FRAC": 0.3,
    "PILLAR_SLICE_MIN_COUNT": 5,
    "PILLAR_VOTE_THRESH": 0.3,
    "PILLAR_MIN_VOTE_FRACTION": 0.25,
    "OFF_LATTICE_FRAC": 0.35,
    "PILLAR_EDGE_MAX_CUT_FRAC": 0.5,
    "GRID_EXTENT": "detections",     # "detections" | "image"
    "GRID_PAD_CELLS": 0,
    "GRID_EXTEND_TO_EDGE": True,
    "KEEP_OFF_LATTICE": False,
    "LATTICE_REFINE_ITERS": 3,
    "LATTICE_MAX_PITCH_DRIFT": 0.25,
    "WRITE_PILLAR_QC_OVERLAY": True,
    "WRITE_PILLAR_FOCUS_STACK": True,

    # --- sites ---------------------------------------------------------------
    "INCLUDE_INFERRED_SITES": True,
    "INCLUDE_OFF_LATTICE_SITES": True,
    "PRIMARY_DENOMINATOR": "all",    # "detected" | "all"

    # --- crop geometry / z ---------------------------------------------------
    "CONTEXT_PITCHES": 2.5,
    "OUT_SIZE": 224,
    "PILLAR_R_FRAC": 0.15,
    "SLIDE_AT_EDGES": True,
    "Z_SMOOTH": 1,
    "Z_RANGE": "",                   # "" = all, or "5:45"

    # --- crop sampling -------------------------------------------------------
    "PILLARS_PER_SERIES": 40,
    "SLICES_PER_PILLAR": 6,
    "SAMPLE_SEED": 0,

    # --- training ------------------------------------------------------------
    "BACKBONE": BACKBONES[0],
    "EPOCHS": 8,
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 3e-5,
    "HOLDOUT_FRAC": 0.2,
    "GROUP_KEY": "series",           # "series" | "site_id"
    "CLASS_WEIGHTS": True,
    "TRAIN_SEED": 42,

    # --- continue training (3B) ----------------------------------------------
    "CT_EPOCHS": 3,
    "CT_LEARNING_RATE": 1e-5,
    "CT_FREEZE_LAYERS": 0,
    "CT_EARLY_STOPPING": True,
    "CT_EVAL_ONLY": False,
    "CT_ALLOW_PREPROC_MISMATCH": False,

    # --- scoring -------------------------------------------------------------
    "AGG_RULE": "maxmip",            # 4B's headline rule
    "INFER_BATCH": 32,
    "WRITE_QC_OVERLAY": True,
    "DEVICE": "auto",                # "auto" | "cuda" | "cpu"
    "MIP_ENABLED": True,
    "MIP_PROJ": "max",
    "MIP_TOPK": 2,
    "MIP_DULL": "match",
    "MIP_GAMMA": 1.6,
    "MIP_SCALE": 0.7,
    "MIP_GLOBAL_NORM": False,
    "MIP_QC": 12,

    # --- nuclei count (N) ----------------------------------------------------
    "FIJI_PATH": "",                 # ImageJ-win64.exe (or Fiji.app folder)
    "NUCLEI_DIRNAME": "nuclei",
    "NUCLEI_REMAKE_MIP": False,      # False = reuse a MIP newer than its slices
    "NUCLEI_PROB": 0.7,              # StarDist probability threshold
    "NUCLEI_MIN_DIAMETER_UM": 5.0,
    "NUCLEI_FILTER_HOLLOW": True,
    "NUCLEI_MIN_CENTRE_RATIO": 0.9,
    "NUCLEI_PIXEL_UM": 0.0,          # fallback when a series has no calibration.json
}

PATH_KEYS = ("ROOT_DIR", "CROPS_DIR", "RESULTS_DIR", "MODEL_DIR",
             "PARENT_MODEL_DIR", "OLD_CROPS_DIR", "FIJI_PATH")

# Every stage reads its parameters from here, exactly as the separate
# scripts read `import config`.
config = types.SimpleNamespace()


def parse_z_range(text) -> tuple[int, int] | None:
    """"" -> None; "5:45" -> (5, 45)."""
    text = (text or "").strip()
    if not text:
        return None
    lo, hi = text.split(":")
    return int(lo), int(hi)


def apply_settings(values: dict) -> None:
    """Load a settings dict into `config` (paths -> Path, derived values)."""
    merged = {**DEFAULTS, **{k: v for k, v in values.items() if k in DEFAULTS}}
    for k, v in merged.items():
        setattr(config, k, v)
    for k in PATH_KEYS:
        v = str(merged[k] or "").strip()
        setattr(config, k, Path(v) if v else None)
    config.Z_RANGE = parse_z_range(merged["Z_RANGE"])
    config.PILLAR_DETECT_DIR = (config.RESULTS_DIR / "pillar_detection"
                                if config.RESULTS_DIR else None)
    config.NUM_LABELS = NUM_LABELS
    config.CLASS_NAMES = CLASS_NAMES


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    """Saved settings merged over the defaults (unknown keys ignored)."""
    values = dict(DEFAULTS)
    if path.exists():
        try:
            saved = json.loads(path.read_text())
            values.update({k: v for k, v in saved.items() if k in DEFAULTS})
        except (OSError, json.JSONDecodeError) as e:
            print(f"! could not read {path} ({e}) - using defaults")
    return values


def save_settings(values: dict, path: Path = SETTINGS_PATH) -> None:
    path.write_text(json.dumps({k: values[k] for k in DEFAULTS if k in values},
                               indent=2))


def cfg(name: str, default):
    """config.<name> if it exists, else default."""
    return getattr(config, name, default)


def site_sources() -> tuple[str, ...] | None:
    """Site types kept in the registry, from INCLUDE_*_SITES (None = all)."""
    if config.INCLUDE_INFERRED_SITES and config.INCLUDE_OFF_LATTICE_SITES:
        return None
    keep = ["detected"]
    if config.INCLUDE_INFERRED_SITES:
        keep.append("inferred")
    if config.INCLUDE_OFF_LATTICE_SITES:
        keep.append("off_lattice")
    return tuple(keep)


apply_settings(DEFAULTS)


# =============================================================================
# SHARED UTILITIES  (wrap_common)
# =============================================================================

# "_z<n>" followed by "_" or "." -- the z-index token in a slice filename,
# e.g. "1_2_c2_yellow_z7_t0.png" -> stack id "1_2_c2_yellow_t0", z=7
_Z_TOKEN_RE = re.compile(r"_z(\d+)(?=[_.])", re.IGNORECASE)
# "_c<N>_<colour>" channel token; stripping it gives the position key shared
# by a pillar stack and its mbp stack ("..._c2_yellow_t0" / "..._c1_cyan_t0"
# -> "..._t0").
CHANNEL_TOKEN_RE = re.compile(r"_c\d+_[A-Za-z]+", re.IGNORECASE)


def parse_z_filename(filename: str) -> tuple[str, int] | None:
    """(stack_id, z) from a slice filename, or None if it has no z token."""
    match = _Z_TOKEN_RE.search(filename)
    if not match:
        return None
    z = int(match.group(1))
    stack_id = filename[: match.start()] + filename[match.end():]
    stack_id = re.sub(r"\.png$", "", stack_id, flags=re.IGNORECASE)
    return stack_id, z


def strip_channel_token(stack_id: str) -> str:
    """Position key shared by pillar and mbp channels of one stack."""
    return CHANNEL_TOKEN_RE.sub("", stack_id)


@dataclass(frozen=True)
class SliceRef:
    """One z-slice file."""
    path: Path
    z: int
    key: str  # e.g. "z014"


def list_slices(directory: Path, stack_id: str | None = None) -> list[SliceRef]:
    """Every PNG z-slice in directory, sorted by z; if stack_id is given,
    only that stack's files (a folder can mix several stacks)."""
    refs = []
    for png_path in Path(directory).glob("*.png"):
        parsed = parse_z_filename(png_path.name)
        if parsed is None:
            continue
        sid, z = parsed
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
    """One slice as float32 intensity (per-pixel channel MAX, so a coloured
    LUT export isn't diluted by its empty channels)."""
    arr = np.array(Image.open(ref.path).convert("RGB"), dtype=np.float32)
    return arr.max(axis=-1)


def load_slice_smoothed(refs: list[SliceRef], zi: int, z_smooth: int) -> np.ndarray:
    """Mean of slice zi and up to z_smooth neighbours each side."""
    if z_smooth <= 0:
        return load_slice(refs[zi])
    lo = max(0, zi - z_smooth)
    hi = min(len(refs), zi + z_smooth + 1)
    return np.mean([load_slice(refs[i]) for i in range(lo, hi)], axis=0)


def select_slices(refs: list[SliceRef], z_range: tuple[int, int] | None = None) -> list[int]:
    """Indices into refs whose z lies in z_range (inclusive); None = all."""
    if z_range is None:
        return list(range(len(refs)))
    lo, hi = z_range
    return [i for i, r in enumerate(refs) if lo <= r.z <= hi]


def normalise(plane: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.7) -> np.ndarray:
    """Percentile-stretch an intensity plane to 0-255 uint8."""
    lo, hi = np.percentile(plane, [lo_pct, hi_pct])
    scaled = np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


@dataclass
class Site:
    """One pillar position to score."""
    site_id: int
    x: float
    y: float
    source: str                    # "detected" | "off_lattice" | "inferred"
    grid_i: int | None = None
    grid_j: int | None = None
    area_px: int | None = None
    radius_px: float | None = None


@dataclass
class Lattice:
    """Fitted pillar lattice (axis-aligned)."""
    pitch: float
    a: np.ndarray
    b: np.ndarray
    rotation_deg: float            # always 0.0 -- the detector fit is axis-aligned
    median_resid: float


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
    """Median nearest-neighbour distance (only when there is no lattice fit)."""
    if len(sites) < 2:
        return 0.0
    pts = np.array([(s.x, s.y) for s in sites])
    dists, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(dists[:, 1]))


def build_site_table(coords: CoordsFile, image_shape: tuple[int, int],
                     min_sites_for_ok: int = 10,
                     include_sources: tuple[str, ...] | None = None) -> SiteTable:
    """Turn one coords.json into a SiteTable, trusting its positions and
    site types. include_sources optionally restricts which types are kept."""
    pillars = coords.pillars
    if include_sources is not None:
        pillars = [p for p in pillars if p.get("site_type", "detected") in include_sources]

    sites = [
        Site(site_id=i, x=float(p["x"]), y=float(p["y"]),
             source=p.get("site_type", "detected"),
             grid_i=p.get("row"), grid_j=p.get("col"),
             area_px=p.get("area_px"), radius_px=p.get("radius_px"))
        for i, p in enumerate(pillars)
    ]

    lat_dict = coords.lattice
    if lat_dict is not None:
        pitch = (lat_dict["pitch_x"] + lat_dict["pitch_y"]) / 2
        a = np.array([lat_dict["pitch_x"], 0.0])
        b = np.array([0.0, lat_dict["pitch_y"]])
        resids = [math.hypot(p["x"] - p["grid_x"], p["y"] - p["grid_y"])
                  for p in coords.pillars
                  if p.get("site_type") == "detected" and p.get("grid_x") is not None]
        median_resid = float(np.median(resids)) if resids else float("nan")
    else:
        pitch = _fallback_pitch(sites)
        a, b = np.array([pitch, 0.0]), np.array([0.0, pitch])
        median_resid = float("nan")

    n_on_grid = sum(1 for s in sites if s.source in ("detected", "inferred"))
    if lat_dict is None:
        lattice_ok, note = False, "no lattice fit in coords.json (too few raw detections upstream)"
    elif n_on_grid < min_sites_for_ok:
        lattice_ok, note = False, f"only {n_on_grid} on-grid sites (< {min_sites_for_ok})"
    elif not np.isfinite(pitch) or pitch <= 0:
        lattice_ok, note = False, f"pitch looks wrong ({pitch!r})"
    else:
        lattice_ok, note = True, ""

    lattice = Lattice(pitch=pitch, a=a, b=b, rotation_deg=0.0, median_resid=median_resid)
    return SiteTable(sites, lattice, lattice_ok, note)


def site_table_rows(table: SiteTable) -> list[dict]:
    return [{"site_id": s.site_id, "x": round(s.x, 2), "y": round(s.y, 2),
             "source": s.source, "grid_i": s.grid_i, "grid_j": s.grid_j,
             "area_px": s.area_px, "radius_px": s.radius_px}
            for s in table.sites]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV (columns from the first row), creating folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def preproc_dict() -> dict:
    """Crop geometry saved with crops and copied into the model."""
    return {"context_pitches": config.CONTEXT_PITCHES, "out_size": config.OUT_SIZE,
            "pillar_r_frac": config.PILLAR_R_FRAC,
            "slide_at_edges": config.SLIDE_AT_EDGES, "z_smooth": config.Z_SMOOTH}


def describe_series(series: "Series", table: SiteTable, n_slices: int,
                    n_sel_slices: int) -> None:
    c = table.counts_by_source()
    flag = "" if table.lattice_ok else f"   <-- {table.lattice_note}"
    print(f"{series.name:<30} {len(table):>4} sites "
          f"({c['detected']} detected, {c['off_lattice']} off-lattice, {c['inferred']} inferred)  "
          f"{n_sel_slices}/{n_slices} slices in range  pitch={table.lattice.pitch:.1f}px{flag}")


def build_crop(field: np.ndarray, all_xy: np.ndarray, site_id: int, pitch: float,
               context_pitches: float | None = None, out_size: int | None = None,
               pillar_r_frac: float | None = None,
               slide_at_edges: bool | None = None) -> Image.Image:
    """One RGB crop for site_id: R = MBP, G = pillar footprints, B = marker
    on the target pillar. Near the border the window slides inward."""
    context_pitches = config.CONTEXT_PITCHES if context_pitches is None else context_pitches
    out_size = config.OUT_SIZE if out_size is None else out_size
    pillar_r_frac = config.PILLAR_R_FRAC if pillar_r_frac is None else pillar_r_frac
    slide_at_edges = config.SLIDE_AT_EDGES if slide_at_edges is None else slide_at_edges

    h, w = field.shape
    side = context_pitches * pitch
    cx, cy = all_xy[site_id]
    x0, y0 = cx - side / 2, cy - side / 2
    x1, y1 = x0 + side, y0 + side

    if slide_at_edges:
        if x0 < 0:
            x1 -= x0; x0 = 0.0
        if y0 < 0:
            y1 -= y0; y0 = 0.0
        if x1 > w:
            x0 -= (x1 - w); x1 = float(w)
        if y1 > h:
            y0 -= (y1 - h); y1 = float(h)
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), x1), min(float(h), y1)

    x0i, y0i = int(round(x0)), int(round(y0))
    x1i, y1i = max(x0i + 1, int(round(x1))), max(y0i + 1, int(round(y1)))
    x1i, y1i = min(w, x1i), min(h, y1i)

    crop_w, crop_h = x1i - x0i, y1i - y0i
    r_channel = np.zeros((crop_h, crop_w), dtype=np.uint8)
    src = field[y0i:y1i, x0i:x1i]
    r_channel[: src.shape[0], : src.shape[1]] = src

    pillar_r = pillar_r_frac * pitch
    g_img = Image.new("L", (crop_w, crop_h), 0)
    gd = ImageDraw.Draw(g_img)
    b_img = Image.new("L", (crop_w, crop_h), 0)
    bd = ImageDraw.Draw(b_img)

    target_lx = target_ly = None
    margin = pillar_r + 2
    for sid in range(all_xy.shape[0]):
        sx, sy = all_xy[sid]
        if np.isnan(sx):
            continue
        if not (x0i - margin <= sx <= x1i + margin and y0i - margin <= sy <= y1i + margin):
            continue
        lx, ly = sx - x0i, sy - y0i
        gd.ellipse([lx - pillar_r, ly - pillar_r, lx + pillar_r, ly + pillar_r], fill=255)
        if sid == site_id:
            target_lx, target_ly = lx, ly

    if target_lx is None:
        target_lx, target_ly = cx - x0i, cy - y0i
    marker_r = max(2.0, pillar_r * 0.35)
    bd.ellipse([target_lx - marker_r, target_ly - marker_r,
                target_lx + marker_r, target_ly + marker_r], fill=255)

    rgb = np.stack([r_channel, np.array(g_img), np.array(b_img)], axis=-1)
    return Image.fromarray(rgb, mode="RGB").resize((out_size, out_size), Image.LANCZOS)


@dataclass
class Series:
    """One field of view: an MBP stack paired with its pillar coords.json."""
    name: str
    mbp_dir: Path
    mbp_stack_id: str
    coords_path: Path

    def coords(self) -> CoordsFile:
        data = json.loads(self.coords_path.read_text())
        shape = tuple(data["image_shape"]) if data.get("image_shape") else None
        return CoordsFile(pillars=data.get("pillars", []), lattice=data.get("lattice"),
                          image_shape=shape, stack_id=data.get("stack_id", ""),
                          path=self.coords_path)


def discover_series(root_dir: Path) -> list[Series]:
    """Every (series folder, stack) with both a coords.json under
    PILLAR_DETECT_DIR and a matching MBP stack (by position key)."""
    root_dir = Path(root_dir)
    series_list: list[Series] = []
    mbp_dirs = sorted(p for p in root_dir.rglob("*")
                      if p.is_dir() and p.name.lower() == config.MBP_DIRNAME.lower())

    for mbp_dir in mbp_dirs:
        series_folder = mbp_dir.parent
        try:
            series_name = series_folder.relative_to(root_dir).as_posix()
        except ValueError:
            series_name = series_folder.name
        series_name = series_name or series_folder.name

        mbp_stacks: dict[str, str] = {}
        for png_path in mbp_dir.glob("*.png"):
            parsed = parse_z_filename(png_path.name)
            if parsed is None:
                continue
            stack_id, _ = parsed
            mbp_stacks.setdefault(strip_channel_token(stack_id), stack_id)

        coords_dir = Path(config.PILLAR_DETECT_DIR) / series_name
        if not coords_dir.exists():
            print(f"  ! no pillar-detection output for series '{series_name}' "
                  f"(expected {coords_dir}) -- skipping")
            continue

        for coords_path in sorted(coords_dir.glob("*_coords.json")):
            pillar_stack_id = coords_path.name[: -len("_coords.json")]
            key = strip_channel_token(pillar_stack_id)
            mbp_stack_id = mbp_stacks.get(key)
            if mbp_stack_id is None:
                print(f"  ! no matching mbp stack for {series_name}/{pillar_stack_id} "
                      f"(looked for position key '{key}' in {mbp_dir}) -- skipping")
                continue
            series_list.append(Series(name=f"{series_name}/{key}", mbp_dir=mbp_dir,
                                      mbp_stack_id=mbp_stack_id, coords_path=coords_path))
    return series_list


class DihedralTransform:
    """Random one of the 8 exact dihedral operations (lossless augmentation
    matching the lattice's symmetry)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, img: Image.Image) -> Image.Image:
        k = int(self.rng.integers(0, 4))
        if k:
            img = img.rotate(90 * k)
        if self.rng.integers(0, 2):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


# =============================================================================
# STAGE A -- LIF EXTRACTION  (LIFEXTRACT)
# =============================================================================

CHANNEL_ROLES = ["nuclei", "mbp", "pillar", "skip"]


def _lut(colour: str):
    """black -> colour lookup for a 0-1 float image, as (H, W, 3) floats."""
    rgb = {"yellow": (1.0, 1.0, 0.0), "cyan": (0.0, 1.0, 1.0)}[colour]
    return lambda a: np.stack([a * c for c in rgb], axis=-1)


class LIFProcessor:
    """Extract LIF series to per-channel PNG z-slices, asking the user once per
    distinct channel setup which channel is nuclei / mbp / pillar."""

    def __init__(self):
        # Only pre-fills the confirmation dialog; never used silently.
        self.default_channel_guess = {0: "nuclei", 1: "mbp", 2: "pillar"}
        self.yellow_cmap = _lut("yellow")
        self.cyan_cmap = _lut("cyan")
        self.confirmed_mappings = {}     # channel signature -> {channel: role}
        self._tk_root = None

    @staticmethod
    def clean_filename(filename: str) -> str:
        filename = filename.strip()
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return re.sub(r'\s+', ' ', filename)

    # --- channel identification ---------------------------------------------

    def _collect_xml_image_elements(self, tree, return_list=None):
        """Walk the LIF XML in readlif's own order, so return_list[i] is the
        element for the i-th image from get_iter_image()."""
        if return_list is None:
            return_list = []
        children = tree.findall("./Children/Element")
        if len(children) < 1:
            children = tree.findall("./Element")
        for item in children:
            has_sub_children = len(item.findall("./Children/Element/Data")) > 0
            if len(item.findall("./Data/Image")) > 0:
                return_list.append(item)
            if has_sub_children:
                self._collect_xml_image_elements(item, return_list)
        return return_list

    def get_channel_hints(self, lif_file, img_index):
        """One LUT/dye hint string per channel, from Leica metadata."""
        try:
            item = self._collect_xml_image_elements(lif_file.xml_root)[img_index]
            hints = []
            for c in item.findall("./Data/Image/ImageDescription/Channels/ChannelDescription"):
                lut = c.attrib.get("LUTName", "")
                quantity = c.attrib.get("NameOfMeasuredQuantity", "")
                hints.append(" / ".join(x for x in [lut, quantity] if x) or "unknown")
            return hints
        except Exception as e:
            print(f"  Could not read channel metadata hints: {e}")
            return []

    def _make_preview_thumbnail(self, lif_image, channel, size=110):
        try:
            z_mid = (lif_image.dims.z if hasattr(lif_image.dims, 'z') else 1) // 2
            frame = lif_image.get_frame(t=0, z=z_mid, c=channel).convert("L")
            frame = ImageOps.autocontrast(frame)
            frame.thumbnail((size, size))
            return frame
        except Exception as e:
            print(f"  Could not build preview for channel {channel}: {e}")
            return Image.new("L", (size, size), color=40)

    def _get_root(self):
        if tk is None:
            raise PipelineError("tkinter is needed for the channel-assignment dialog")
        if self._tk_root is None:
            self._tk_root = tk.Tk()
            self._tk_root.withdraw()
        return self._tk_root

    def close(self):
        if self._tk_root is not None:
            self._tk_root.destroy()
            self._tk_root = None

    def confirm_channel_mapping(self, series_name, n_channels, hints, lif_image):
        """Dialog: thumbnail + hint per channel, user assigns each a role."""
        root = self._get_root()
        dialog = tk.Toplevel(root)
        dialog.title(f"Confirm channel assignment - {series_name}")
        dialog.grab_set()

        tk.Label(dialog, justify="left",
                 text=(f"Series: {series_name}\n"
                       "Check each channel's preview/metadata hint and assign its identity.\n"
                       "This mapping will be reused for other series with the same channel setup.")
                 ).grid(row=0, column=0, columnspan=n_channels, padx=10, pady=(10, 5), sticky="w")

        result, photo_refs, role_vars = {}, [], []
        for c in range(n_channels):
            col_frame = tk.Frame(dialog, bd=1, relief="groove", padx=6, pady=6)
            col_frame.grid(row=1, column=c, padx=6, pady=6)
            photo = ImageTk.PhotoImage(self._make_preview_thumbnail(lif_image, c))
            photo_refs.append(photo)
            tk.Label(col_frame, image=photo).pack()
            hint_text = hints[c] if c < len(hints) else "unknown"
            tk.Label(col_frame, text=f"Channel {c}\nLUT/dye hint: {hint_text}",
                     justify="center").pack(pady=(4, 4))
            guess = self.default_channel_guess.get(c, "skip")
            var = tk.StringVar(value=guess if guess in CHANNEL_ROLES else "skip")
            role_vars.append(var)
            ttk.OptionMenu(col_frame, var, var.get(), *CHANNEL_ROLES).pack()

        def on_confirm():
            for c, var in enumerate(role_vars):
                result[c] = var.get()
            dialog.destroy()

        tk.Button(dialog, text="Confirm mapping", command=on_confirm).grid(
            row=2, column=0, columnspan=n_channels, pady=10)
        dialog.protocol("WM_DELETE_WINDOW", on_confirm)
        _bring_to_front(dialog)
        root.wait_window(dialog)
        return result

    def resolve_channel_mapping(self, lif_file, img_index, series_name, lif_image):
        """{channel: role}; asks once per new channel signature."""
        n_channels = lif_image.channels
        hints = (self.get_channel_hints(lif_file, img_index) + ["unknown"] * n_channels)[:n_channels]
        signature = tuple(hints)
        if signature not in self.confirmed_mappings:
            print(f"  New channel setup detected {signature} - asking for confirmation")
            self.confirmed_mappings[signature] = self.confirm_channel_mapping(
                series_name, n_channels, hints, lif_image)
        return self.confirmed_mappings[signature]

    @staticmethod
    def calibration(lif_image) -> dict | None:
        """Pixel size in um from readlif's scale (px per um), or None."""
        scale = getattr(lif_image, "scale", None)
        try:
            sx, sy = scale[0], scale[1]
            sz = scale[2] if len(scale) > 2 else None
        except (TypeError, IndexError):
            return None
        if not sx or not sy:
            return None
        return {"pixel_width_um": 1.0 / sx, "pixel_height_um": 1.0 / sy,
                "z_step_um": (1.0 / sz) if sz else None,
                "source": "readlif LifImage.scale"}

    def browse_lif_files(self):
        root = self._get_root()
        return list(filedialog.askopenfilenames(
            parent=root, title="Select LIF file(s)",
            filetypes=[("LIF files", "*.lif"), ("All files", "*.*")]))

    # --- extraction ---------------------------------------------------------

    def process_lif_file(self, lif_file_path, out_root: Path) -> bool:
        """Extract one LIF into out_root/<lif name>/<series>/<role>/*.png."""
        from readlif.reader import LifFile
        try:
            print(f"Processing {os.path.basename(lif_file_path)}")
            lif_file = LifFile(lif_file_path)
            base_name = self.clean_filename(os.path.splitext(os.path.basename(lif_file_path))[0])
            output_dir = Path(out_root) / base_name
            output_dir.mkdir(parents=True, exist_ok=True)

            all_images = list(lif_file.get_iter_image())
            print(f"Found {len(all_images)} images in LIF file -> {output_dir}")

            for img_index, lif_image in enumerate(all_images):
                series_name = self.clean_filename(lif_image.name)
                dims = lif_image.dims
                print(f"Image {img_index + 1}/{len(all_images)}: {series_name}  "
                      f"dims={dims}  channels={lif_image.channels}")
                channel_role = self.resolve_channel_mapping(lif_file, img_index,
                                                            series_name, lif_image)
                print(f"  Channel mapping: {channel_role}")

                # PNGs carry no spatial calibration; keep it for the nuclei count
                cal = self.calibration(lif_image)
                (output_dir / series_name).mkdir(parents=True, exist_ok=True)
                if cal:
                    (output_dir / series_name / "calibration.json").write_text(
                        json.dumps(cal, indent=2))
                    print(f"  Pixel size: {cal['pixel_width_um']:.4f} x "
                          f"{cal['pixel_height_um']:.4f} um")
                else:
                    print("  ! no pixel size in the LIF metadata for this series")

                z_total = dims.z if hasattr(dims, 'z') else 1
                t_total = dims.t if hasattr(dims, 't') else 1
                for channel, role in channel_role.items():
                    if role == "skip":
                        continue
                    if channel >= lif_image.channels:
                        print(f"  Channel {channel} not available (only {lif_image.channels})")
                        continue
                    channel_dir = output_dir / series_name / role
                    channel_dir.mkdir(parents=True, exist_ok=True)

                    # mbp drops its first and last z-slice (follows the role,
                    # not a hard-coded channel index)
                    if role == "mbp":
                        z_range = range(1, z_total - 1) if z_total > 2 else range(0, z_total)
                    else:
                        z_range = range(z_total)

                    if role == "pillar":
                        cmap, channel_name = self.yellow_cmap, "yellow"
                    else:
                        cmap, channel_name = self.cyan_cmap, "cyan"

                    n_saved = 0
                    for z in z_range:
                        for t in range(t_total):
                            try:
                                np_image = np.array(lif_image.get_frame(t=t, z=z, c=channel))
                                np_image = np_image.astype(np.float64)
                                if np_image.max() > np_image.min():
                                    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
                                else:
                                    np_image = np.zeros_like(np_image)
                                rgb = (np.clip(cmap(np_image), 0, 1) * 255).astype(np.uint8)
                                filename = self.clean_filename(
                                    f'{series_name}_c{channel}_{channel_name}_z{z}_t{t}.png')
                                Image.fromarray(rgb, 'RGB').save(channel_dir / filename)
                                n_saved += 1
                            except Exception as e:
                                print(f"    Error on c{channel} z{z} t{t}: {e}")
                    print(f"  c{channel} -> {role}: {n_saved} slice(s)")

            print(f"Finished {os.path.basename(lif_file_path)}")
            return True
        except Exception:
            print(f"Error processing {lif_file_path}:")
            traceback.print_exc()
            return False


def run_lif_extraction() -> None:
    try:
        import readlif  # noqa: F401
    except ImportError:
        raise PipelineError("LIF extraction needs readlif:  pip install readlif")
    proc = LIFProcessor()
    try:
        lif_files = list(config.LIF_FILES) or proc.browse_lif_files()
        lif_files = [f for f in lif_files if f.lower().endswith(".lif")]
        if not lif_files:
            raise PipelineError("no .lif files selected")
        print(f"Output root: {config.ROOT_DIR}")
        ok = [proc.process_lif_file(f, config.ROOT_DIR) for f in lif_files]
    finally:
        proc.close()
    if not any(ok):
        raise PipelineError("no LIF file was extracted")
    if not all(ok):
        print(f"! {ok.count(False)} of {len(ok)} LIF file(s) failed - continuing with the rest")


# =============================================================================
# STAGE 0 -- PILLAR DETECTION  (0 PIL_DET)
# =============================================================================

def find_pillar_folders(root_dir: Path) -> list[Path]:
    name = config.PILLAR_DIRNAME.lower()
    return sorted(p for p in root_dir.rglob("*") if p.is_dir() and p.name.lower() == name)


def group_into_stacks(pillar_dir: Path):
    """{stack_id: [(z, path), ...]} sorted by z, plus skipped filenames."""
    stacks: dict[str, list[tuple[int, Path]]] = {}
    skipped = []
    for png_path in sorted(pillar_dir.glob("*.png")):
        parsed = parse_z_filename(png_path.name)
        if parsed is None:
            skipped.append(png_path.name)
            continue
        stack_id, z_index = parsed
        stacks.setdefault(stack_id, []).append((z_index, png_path))
    for slices in stacks.values():
        slices.sort(key=lambda t: t[0])
    return stacks, skipped


def load_intensity(png_path: Path) -> np.ndarray:
    arr = np.array(Image.open(png_path).convert("RGB"), dtype=np.float32)
    return arr.max(axis=-1)


def find_debris_mask(intensity, seed_percentile, min_area_frac, dilate_px) -> np.ndarray:
    """Large bright connected regions (cells, dust), dilated over their halo."""
    seed_thresh = max(np.percentile(intensity, seed_percentile), 1.0)
    seed_mask = intensity >= seed_thresh
    struct = ndi.generate_binary_structure(2, 2)
    labels, n = ndi.label(seed_mask, structure=struct)
    if n == 0:
        return np.zeros_like(seed_mask)
    sizes = ndi.sum(seed_mask, labels, index=range(1, n + 1))
    min_area = min_area_frac * intensity.size
    debris_ids = [i + 1 for i, s in enumerate(sizes) if s >= min_area]
    if not debris_ids:
        return np.zeros_like(seed_mask)
    debris_mask = np.isin(labels, debris_ids)
    if dilate_px > 0:
        debris_mask = ndi.binary_dilation(debris_mask, iterations=dilate_px)
    return debris_mask


def normalize_slice(intensity, bg_pct, ref_pct, min_scale, debris_mask=None) -> np.ndarray:
    """Percentile-range normalisation; debris excluded and set to -1."""
    valid = intensity[~debris_mask] if (debris_mask is not None and debris_mask.any()) else intensity
    bg = np.percentile(valid, bg_pct)
    ref = np.percentile(valid, ref_pct)
    norm = (intensity - bg) / max(ref - bg, min_scale)
    if debris_mask is not None:
        norm[debris_mask] = -1.0
    return norm


def find_slice_pillars(intensity, debris_mask, min_diameter_px, max_diameter_px,
                       max_eccentricity) -> list[dict]:
    """Otsu-threshold one slice and keep round blobs of pillar size."""
    valid = intensity[~debris_mask] if debris_mask.any() else intensity
    if valid.size == 0 or valid.max() <= valid.min():
        return []
    try:
        thresh = filters.threshold_otsu(valid)
    except ValueError:
        return []
    bright = (intensity >= thresh) & (~debris_mask)
    struct = ndi.generate_binary_structure(2, 2)
    bright = ndi.binary_closing(bright, structure=disk_selem(2))
    bright = ndi.binary_fill_holes(bright, structure=struct)
    labels, n = ndi.label(bright, structure=struct)
    if n == 0:
        return []
    found = []
    for p in measure.regionprops(labels):
        diameter = 2.0 * math.sqrt(p.area / math.pi)
        if min_diameter_px <= diameter <= max_diameter_px and p.eccentricity <= max_eccentricity:
            found.append({"x": float(p.centroid[1]), "y": float(p.centroid[0]),
                          "area": int(p.area), "diameter": diameter})
    return found


def analyze_slice(intensity, debris_mask, min_d, max_d, max_ecc):
    """Try as-is and inverted; keep whichever finds more pillar blobs."""
    blobs_asis = find_slice_pillars(intensity, debris_mask, min_d, max_d, max_ecc)
    flipped = float(intensity.max()) - intensity
    blobs_flipped = find_slice_pillars(flipped, debris_mask, min_d, max_d, max_ecc)
    if len(blobs_flipped) > len(blobs_asis):
        return flipped, blobs_flipped, True
    return intensity, blobs_asis, False


def cluster_candidates(candidates: list[dict], radius_px: float) -> list[dict]:
    """Merge detections of the same pillar from different slices (union-find)."""
    if not candidates:
        return []
    pts = np.array([(c["x"], c["y"]) for c in candidates])
    pairs = cKDTree(pts).query_pairs(r=radius_px)
    n = len(candidates)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    groups: dict[int, list[dict]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(candidates[i])

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


def estimate_lattice(centroids_xy) -> dict | None:
    """Initial axis-aligned lattice (pitch/phase per axis). None if < 6 points."""
    pts = np.asarray(centroids_xy, dtype=np.float64)
    if len(pts) < 6:
        return None
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    rough_pitch = float(np.median(dists[:, 1]))
    if not np.isfinite(rough_pitch) or rough_pitch <= 0:
        return None

    x_gaps, y_gaps = [], []
    for i, j in tree.query_pairs(r=1.5 * rough_pitch):
        dx = abs(pts[j, 0] - pts[i, 0])
        dy = abs(pts[j, 1] - pts[i, 1])
        if dy < 0.3 * rough_pitch and dx > 0.3 * rough_pitch:
            x_gaps.append(dx)
        if dx < 0.3 * rough_pitch and dy > 0.3 * rough_pitch:
            y_gaps.append(dy)
    pitch_x = float(np.median(x_gaps)) if len(x_gaps) >= 5 else rough_pitch
    pitch_y = float(np.median(y_gaps)) if len(y_gaps) >= 5 else rough_pitch

    def circular_phase(vals, pitch):
        theta = 2 * np.pi * (vals % pitch) / pitch
        mean_angle = math.atan2(np.sin(theta).sum(), np.cos(theta).sum())
        return (mean_angle / (2 * np.pi) * pitch) % pitch

    return {"pitch_x": pitch_x, "pitch_y": pitch_y,
            "phase_x": circular_phase(pts[:, 0], pitch_x),
            "phase_y": circular_phase(pts[:, 1], pitch_y)}


def refine_lattice(centroids_xy, lattice, off_lattice_frac, iters, max_pitch_drift):
    """Least-squares re-fit of pitch/phase on inliers only."""
    pts = np.asarray(centroids_xy, dtype=np.float64)
    lat = dict(lattice)

    def fit_state(lat):
        px, py = lat["pitch_x"], lat["pitch_y"]
        cols = np.round((pts[:, 0] - lat["phase_x"]) / px)
        rows = np.round((pts[:, 1] - lat["phase_y"]) / py)
        resid = np.hypot(pts[:, 0] - (lat["phase_x"] + cols * px),
                         pts[:, 1] - (lat["phase_y"] + rows * py))
        keep = resid <= off_lattice_frac * (px + py) / 2
        r = float(np.sqrt((resid[keep] ** 2).mean())) if keep.any() else float("nan")
        return rows, cols, keep, r

    rows, cols, inliers, rms = fit_state(lat)
    for _ in range(max(int(iters), 0)):
        px, py = lat["pitch_x"], lat["pitch_y"]
        rows, cols, inliers, rms = fit_state(lat)
        if inliers.sum() < 6:
            break
        if len(np.unique(cols[inliers])) < 2 or len(np.unique(rows[inliers])) < 2:
            break
        new_px, new_ox = np.polyfit(cols[inliers], pts[inliers, 0], 1)
        new_py, new_oy = np.polyfit(rows[inliers], pts[inliers, 1], 1)
        if not (np.isfinite(new_px) and np.isfinite(new_py)) or new_px <= 0 or new_py <= 0:
            break
        if abs(new_px - px) > max_pitch_drift * px or abs(new_py - py) > max_pitch_drift * py:
            break
        cand = {"pitch_x": float(new_px), "pitch_y": float(new_py),
                "phase_x": float(new_ox % new_px), "phase_y": float(new_oy % new_py)}
        if cand == lat:
            break
        lat = cand
    return lat, inliers, rms


def snap_to_lattice(centroids_xy, lattice, off_lattice_frac) -> list[dict]:
    px, py = lattice["pitch_x"], lattice["pitch_y"]
    ox, oy = lattice["phase_x"], lattice["phase_y"]
    avg_pitch = (px + py) / 2
    out = []
    for x, y in centroids_xy:
        col, row = round((x - ox) / px), round((y - oy) / py)
        resid = math.hypot(x - (ox + col * px), y - (oy + row * py))
        out.append({"row": row, "col": col,
                    "on_lattice": resid <= off_lattice_frac * avg_pitch,
                    "residual_px": resid})
    return out


def segment_cut_fraction(d: float, r: float) -> float:
    """Fraction of a disk beyond a line at signed distance d from its centre."""
    if r <= 0:
        return 0.0 if d >= 0 else 1.0
    if d >= r:
        return 0.0
    if d <= -r:
        return 1.0
    return (r * r * math.acos(d / r) - d * math.sqrt(max(r * r - d * d, 0.0))) / (math.pi * r * r)


def edge_centre_margin(r: float, max_cut_frac: float) -> float:
    """Distance inside the border a centre needs for <= max_cut_frac cut off."""
    if max_cut_frac <= 0:
        return r
    if max_cut_frac >= 1:
        return -r
    lo, hi = -r, r
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if segment_cut_fraction(mid, r) > max_cut_frac:
            lo = mid
        else:
            hi = mid
    return hi


def disk_cut_fraction(cx, cy, r, image_shape, samples: int = 48) -> float:
    """Fraction of a pillar's disk outside the image (sampled)."""
    h, w = image_shape
    if r <= 0:
        return 0.0 if (0 <= cx <= w - 1 and 0 <= cy <= h - 1) else 1.0
    if cx - r >= 0 and cy - r >= 0 and cx + r <= w - 1 and cy + r <= h - 1:
        return 0.0
    t = np.linspace(-r, r, samples)
    in_disk = (t[None, :] ** 2 + t[:, None] ** 2) <= r * r
    xs, ys = cx + t[None, :], cy + t[:, None]
    in_img = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    n = int(in_disk.sum())
    if n == 0:
        return 1.0
    return float(1.0 - (in_disk & in_img).sum() / n)


def lattice_index_range(lattice, image_shape, margin_px):
    h, w = image_shape
    px, py = lattice["pitch_x"], lattice["pitch_y"]
    ox, oy = lattice["phase_x"], lattice["phase_y"]
    return (math.ceil((margin_px - oy) / py), math.floor((h - 1 - margin_px - oy) / py),
            math.ceil((margin_px - ox) / px), math.floor((w - 1 - margin_px - ox) / px))


def grid_extent(lattice, image_shape, detected_rc, mode, pad_cells, radius_px,
                max_cut_frac, extend_to_edge=True):
    """(row_min, row_max, col_min, col_max) box to fill with sites."""
    margin = edge_centre_margin(radius_px, max_cut_frac)
    img_r0, img_r1, img_c0, img_c1 = lattice_index_range(lattice, image_shape, margin)
    if mode == "image" or not detected_rc:
        return img_r0, img_r1, img_c0, img_c1

    rows = [r for r, _ in detected_rc]
    cols = [c for _, c in detected_rc]
    r0, r1 = max(min(rows) - pad_cells, img_r0), min(max(rows) + pad_cells, img_r1)
    c0, c1 = max(min(cols) - pad_cells, img_c0), min(max(cols) + pad_cells, img_c1)

    if extend_to_edge:
        px, py = lattice["pitch_x"], lattice["pitch_y"]
        ox, oy = lattice["phase_x"], lattice["phase_y"]

        def clipped_row(row, lo, hi):
            sy = oy + row * py
            return any(disk_cut_fraction(ox + c * px, sy, radius_px, image_shape) > 0.0
                       for c in range(lo, hi + 1))

        def clipped_col(col, lo, hi):
            sx = ox + col * px
            return any(disk_cut_fraction(sx, oy + r * py, radius_px, image_shape) > 0.0
                       for r in range(lo, hi + 1))

        while r0 - 1 >= img_r0 and clipped_row(r0 - 1, c0, c1):
            r0 -= 1
        while r1 + 1 <= img_r1 and clipped_row(r1 + 1, c0, c1):
            r1 += 1
        while c0 - 1 >= img_c0 and clipped_col(c0 - 1, r0, r1):
            c0 -= 1
        while c1 + 1 <= img_c1 and clipped_col(c1 + 1, r0, r1):
            c1 += 1
    return r0, r1, c0, c1


def local_site_evidence(norm_stack, x, y, radius_px, vote_thresh, min_vote_fraction) -> bool:
    n_slices, h, w = norm_stack.shape
    x0, x1 = max(0, int(x - radius_px)), min(w, int(x + radius_px) + 1)
    y0, y1 = max(0, int(y - radius_px)), min(h, int(y + radius_px) + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    slice_means = norm_stack[:, y0:y1, x0:x1].reshape(n_slices, -1).mean(axis=1)
    return float((slice_means > vote_thresh).mean()) >= min_vote_fraction


def build_grid_mask(image_shape, sites, radius_px) -> np.ndarray:
    img = Image.new("L", (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(img)
    r = radius_px
    for s in sites:
        cx, cy = s["snapped_x"], s["snapped_y"]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    return np.array(img)


def write_pillar_qc_overlay(norm_stack, pillars, out_path, discarded=None) -> None:
    """Max projection with sites circled (red detected, yellow inferred,
    blue off-lattice) and discarded candidates crossed in grey."""
    proj = norm_stack.max(axis=0)
    lo, hi = np.percentile(proj, [1, 99.5])
    scaled = np.clip((proj - lo) / max(hi - lo, 1e-6), 0, 1)
    img = Image.fromarray((scaled * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    r = 6
    colors = {"detected": (255, 0, 0), "inferred": (255, 255, 0), "off_lattice": (0, 140, 255)}
    for p in pillars:
        x, y = p["x"], p["y"]
        color = colors.get(p.get("site_type"), (255, 0, 0))
        rr = r if p.get("site_type") != "inferred" or p.get("has_local_evidence") else r - 2
        draw.ellipse([x - rr, y - rr, x + rr, y + rr], outline=color, width=2)
    for c in discarded or []:
        x, y = c["x"], c["y"]
        draw.line([x - r, y - r, x + r, y + r], fill=(150, 150, 150), width=2)
        draw.line([x - r, y + r, x + r, y - r], fill=(150, 150, 150), width=2)
    img.save(out_path)


def build_focus_stack(norm_stack, usable_z, pillars, default_radius_px) -> np.ndarray:
    canvas = norm_stack.max(axis=0).copy()
    z_to_idx = {z: i for i, z in enumerate(usable_z)}
    for p in pillars:
        if p.get("site_type") not in ("detected", "inferred"):
            continue
        z = p.get("best_z")
        if z is None or z not in z_to_idx:
            continue
        r = p["radius_px"] if p.get("radius_px") else default_radius_px
        rr, cc = draw_disk((p["y"], p["x"]), r, shape=canvas.shape)
        canvas[rr, cc] = norm_stack[z_to_idx[z]][rr, cc]
    return canvas


def save_scaled(img: np.ndarray, out_path: Path) -> None:
    lo, hi = np.percentile(img, [1, 99.5])
    scaled = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    Image.fromarray((scaled * 255).astype(np.uint8)).save(out_path)


def build_raw_mask(image_shape, usable_idx, slice_blobs) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=bool)
    for i in usable_idx:
        for b in slice_blobs[i]:
            rr, cc = draw_disk((b["y"], b["x"]), b["diameter"] / 2, shape=image_shape)
            mask[rr, cc] = True
    return mask


def _raw_pillar_records(merged_candidates) -> list[dict]:
    """Raw candidates as sites, for when no lattice can be fitted."""
    return [{"x": round(c["best_x"], 2), "y": round(c["best_y"], 2),
             "grid_x": None, "grid_y": None, "area_px": c["area"],
             "radius_px": round(c["diameter"] / 2, 2), "best_z": c["z"],
             "row": None, "col": None, "n_slice_detections": c["n_slice_detections"],
             "site_type": "detected"} for c in merged_candidates]


def process_stack(stack_id, slice_paths, out_dir, series_name, pillar_dir):
    """Detect pillars, fit the lattice and write all outputs for one stack."""
    if not slice_paths:
        return None

    planes, z_indices, ref_shape = [], [], None
    for z_idx, path in slice_paths:
        intensity = load_intensity(path)
        if ref_shape is None:
            ref_shape = intensity.shape
        elif intensity.shape != ref_shape:
            print(f"      ! skipping {path.name}: shape {intensity.shape} != {ref_shape}")
            continue
        planes.append(intensity)
        z_indices.append(z_idx)

    n_total, n_used = len(slice_paths), len(planes)
    if n_used == 0:
        print(f"      ! no usable slices for stack '{stack_id}', skipping")
        return None

    debris_masks = [find_debris_mask(p, config.PILLAR_DEBRIS_SEED_PERCENTILE,
                                     config.PILLAR_DEBRIS_MIN_AREA_FRAC,
                                     config.PILLAR_DEBRIS_DILATE_PX) for p in planes]
    debris_frac = float(np.mean([m.mean() for m in debris_masks]))
    if debris_frac > 0:
        print(f"      debris/contamination excluded from {debris_frac * 100:.2f}% of frame on average")

    corrected, slice_blobs, flipped_z = [], [], []
    for i, (plane, dmask) in enumerate(zip(planes, debris_masks)):
        fixed, blobs, was_flipped = analyze_slice(
            plane, dmask, config.PILLAR_MIN_DIAMETER_PX, config.PILLAR_MAX_DIAMETER_PX,
            config.PILLAR_MAX_ECCENTRICITY)
        corrected.append(fixed)
        slice_blobs.append(blobs)
        if was_flipped:
            flipped_z.append(z_indices[i])

    counts = [len(b) for b in slice_blobs]
    best_count = max(counts) if counts else 0
    usable_thresh = max(config.PILLAR_SLICE_MIN_COUNT,
                        math.ceil(config.PILLAR_SLICE_USABLE_FRAC * best_count))
    usable_idx = [i for i, c in enumerate(counts) if c >= usable_thresh]
    disregarded_z = [z_indices[i] for i in range(n_used) if i not in usable_idx]
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

    norm_stack = np.stack(
        [normalize_slice(corrected[i], config.PILLAR_BG_PERCENTILE, config.PILLAR_REF_PERCENTILE,
                         config.PILLAR_MIN_SCALE, debris_masks[i]) for i in usable_idx],
        axis=0).astype(np.float32)

    raw_candidates = [{**b, "z": z_indices[i]} for i in usable_idx for b in slice_blobs[i]]
    merged_candidates = cluster_candidates(raw_candidates, config.PILLAR_MIN_DIAMETER_PX)

    lattice = estimate_lattice([(c["x"], c["y"]) for c in merged_candidates])
    pillars: list[dict] = []
    off_lattice_records: list[dict] = []
    grid_mask = grid_info = fit_rms = None
    n_inferred = n_off_lattice = 0

    if lattice is None:
        print(f"      ! too few raw candidates ({len(merged_candidates)}) to fit a lattice -- "
              f"coords.json has raw candidates only, no grid_mask")
        pillars = _raw_pillar_records(merged_candidates)
    else:
        h, w = ref_shape
        max_cells = 4 * (w / max(lattice["pitch_x"], 1e-6)) * (h / max(lattice["pitch_y"], 1e-6))
        if max_cells > 200_000:
            print(f"      ! lattice fit looks wrong (pitch=({lattice['pitch_x']:.2f},"
                  f"{lattice['pitch_y']:.2f})px implies {max_cells:.0f} grid cells) -- "
                  f"discarding the fit, coords.json has raw candidates only")
            lattice = None
            pillars = _raw_pillar_records(merged_candidates)

    if lattice is not None:
        lattice, _, fit_rms = refine_lattice(
            [(c["x"], c["y"]) for c in merged_candidates], lattice,
            config.OFF_LATTICE_FRAC, cfg("LATTICE_REFINE_ITERS", 3),
            cfg("LATTICE_MAX_PITCH_DRIFT", 0.25))

        snapped = snap_to_lattice([(c["x"], c["y"]) for c in merged_candidates], lattice,
                                  config.OFF_LATTICE_FRAC)
        on_site: dict[tuple[int, int], list[dict]] = {}
        off_site: list[dict] = []
        for c, s in zip(merged_candidates, snapped):
            if s["on_lattice"]:
                on_site.setdefault((s["row"], s["col"]), []).append(c)
            else:
                off_site.append({**c, "residual_px": round(s["residual_px"], 2)})

        avg_pitch = (lattice["pitch_x"] + lattice["pitch_y"]) / 2
        search_r = config.OFF_LATTICE_FRAC * avg_pitch
        radius_px = config.PILLAR_R_FRAC * avg_pitch
        max_cut = float(cfg("PILLAR_EDGE_MAX_CUT_FRAC", 0.5))

        row_min, row_max, col_min, col_max = grid_extent(
            lattice, ref_shape, list(on_site.keys()),
            cfg("GRID_EXTENT", "detections"), int(cfg("GRID_PAD_CELLS", 0)),
            radius_px, max_cut, bool(cfg("GRID_EXTEND_TO_EDGE", True)))

        n_rows = max(row_max - row_min + 1, 0)
        n_cols = max(col_max - col_min + 1, 0)
        n_clipped = n_dropped_cut = 0
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                sx = lattice["phase_x"] + col * lattice["pitch_x"]
                sy = lattice["phase_y"] + row * lattice["pitch_y"]
                cut = disk_cut_fraction(sx, sy, radius_px, ref_shape)
                if cut > max_cut:
                    n_dropped_cut += 1
                    continue
                if cut > 0:
                    n_clipped += 1
                r0, c0 = row - row_min, col - col_min
                group = on_site.get((row, col))
                if group:
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

        n_off_lattice = len(off_site)
        off_lattice_records = [
            {"x": round(c["x"], 2), "y": round(c["y"], 2),
             "area_px": c["area"], "radius_px": round(c["diameter"] / 2, 2),
             "best_z": c["z"], "residual_px": c["residual_px"],
             "n_slice_detections": c["n_slice_detections"]} for c in off_site]
        if cfg("KEEP_OFF_LATTICE", False):
            for c in off_site:
                pillars.append({
                    "x": round(c["x"], 2), "y": round(c["y"], 2),
                    "grid_x": None, "grid_y": None,
                    "area_px": c["area"], "radius_px": round(c["diameter"] / 2, 2),
                    "best_z": c["z"], "row": None, "col": None,
                    "n_slice_detections": c["n_slice_detections"],
                    "has_local_evidence": None, "cut_frac": None, "clipped": None,
                    "site_type": "off_lattice",
                })

        grid_sites = [{"snapped_x": p["grid_x"], "snapped_y": p["grid_y"]}
                      for p in pillars if p["site_type"] in ("detected", "inferred")]
        grid_mask = build_grid_mask(ref_shape, grid_sites, radius_px)
        n_grid = len(grid_sites)
        n_detected = n_grid - n_inferred
        n_weak = sum(1 for p in pillars
                     if p["site_type"] == "inferred" and not p["has_local_evidence"])
        grid_info = {
            "n_rows": n_rows, "n_cols": n_cols, "row_min": row_min, "col_min": col_min,
            "extent_mode": cfg("GRID_EXTENT", "detections"),
            "pad_cells": int(cfg("GRID_PAD_CELLS", 0)),
            "n_clipped": n_clipped, "n_dropped_over_cut": n_dropped_cut,
            "edge_max_cut_frac": max_cut,
            "complete": n_grid == n_rows * n_cols - n_dropped_cut,
            "fit_rms_px": None if fit_rms is None or not math.isfinite(fit_rms)
            else round(fit_rms, 2),
        }

        rms_txt = (f"{fit_rms:.2f}px" if fit_rms is not None and math.isfinite(fit_rms)
                   else "n/a")
        print(f"      lattice pitch=({lattice['pitch_x']:.1f},{lattice['pitch_y']:.1f})px "
              f"rms={rms_txt} -> {n_rows}x{n_cols} = {n_grid} grid sites "
              f"({n_detected} detected + {n_inferred} filled), "
              f"{n_off_lattice} off-lattice discarded")
        if n_clipped or n_dropped_cut:
            print(f"      edge sites: {n_clipped} kept clipped by the frame "
                  f"(<={max_cut * 100:.0f}% cut off), {n_dropped_cut} dropped")
        if n_weak:
            print(f"      note: {n_weak}/{n_inferred} filled sites show no local signal "
                  f"(kept anyway; 'has_local_evidence': false in coords.json)")
        if n_grid and n_off_lattice > 0.1 * n_grid:
            print(f"      ! {n_off_lattice} candidates ({100 * n_off_lattice / n_grid:.0f}% "
                  f"of the grid) missed the lattice - check {stack_id}_qc.png")
        if n_grid and n_inferred > 0.5 * n_grid:
            print("      ! over half the grid is filled rather than detected - check the "
                  "QC overlay before using it")

    pillars.sort(key=lambda p: (p["y"], p["x"]))
    pillars = [{"id": i, **p} for i, p in enumerate(pillars, start=1)]

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_mask = build_raw_mask(ref_shape, usable_idx, slice_blobs)
    mask_path = out_dir / f"{stack_id}_mask.png"
    Image.fromarray((raw_mask * 255).astype(np.uint8)).save(mask_path)

    grid_mask_path = None
    if grid_mask is not None:
        grid_mask_path = out_dir / f"{stack_id}_grid_mask.png"
        Image.fromarray(grid_mask).save(grid_mask_path)

    coords = {
        "series": series_name, "pillar_folder": str(pillar_dir), "stack_id": stack_id,
        "image_shape": list(ref_shape),
        "n_slices_found": n_total, "n_slices_used": n_used,
        "z_indices_used": z_indices, "usable_z": usable_z,
        "disregarded_z": disregarded_z, "flipped_z": flipped_z,
        "per_slice_blob_counts": dict(zip(z_indices, counts)),
        "lattice": lattice, "grid": grid_info,
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
        "n_grid_sites": (sum(1 for p in pillars if p["site_type"] in ("detected", "inferred"))
                         if lattice is not None else None),
        "pillars": pillars,
    }
    json_path = out_dir / f"{stack_id}_coords.json"
    json_path.write_text(json.dumps(coords, indent=2))

    if config.WRITE_PILLAR_QC_OVERLAY:
        write_pillar_qc_overlay(norm_stack, pillars, out_dir / f"{stack_id}_qc.png",
                                discarded=off_lattice_records)
    if config.WRITE_PILLAR_FOCUS_STACK:
        default_r = ((lattice["pitch_x"] + lattice["pitch_y"]) / 4 * config.PILLAR_R_FRAC
                     if lattice is not None else config.PILLAR_MIN_DIAMETER_PX / 2)
        save_scaled(build_focus_stack(norm_stack, usable_z, pillars, default_r),
                    out_dir / f"{stack_id}_focus_stack.png")

    n_grid_sites = (sum(1 for p in pillars if p["site_type"] in ("detected", "inferred"))
                    if lattice is not None else len(pillars))
    return mask_path, json_path, grid_mask_path, n_grid_sites


def process_pillar_folder(pillar_dir: Path, root_dir: Path) -> dict[str, int]:
    try:
        series_name = pillar_dir.parent.relative_to(root_dir).as_posix()
    except ValueError:
        series_name = pillar_dir.parent.name
    series_name = series_name or pillar_dir.parent.name

    stacks, skipped = group_into_stacks(pillar_dir)
    if skipped:
        print(f"    (ignored {len(skipped)} file(s) with no recognisable z-index token)")
    if not stacks:
        print("    No z-stacks found.")
        return {}

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


def run_pillar_detection() -> None:
    root_dir = Path(config.ROOT_DIR)
    print(f"Scanning {root_dir} for '{config.PILLAR_DIRNAME}' folders...")
    pillar_dirs = find_pillar_folders(root_dir)
    if not pillar_dirs:
        raise PipelineError(f"no '{config.PILLAR_DIRNAME}' folders found under {root_dir}")

    print(f"Found {len(pillar_dirs)} pillar folder(s).")
    per_stack_counts: dict[str, int] = {}
    for pillar_dir in pillar_dirs:
        print(f"\n--- {pillar_dir.relative_to(root_dir)} ---")
        for stack_id, n in process_pillar_folder(pillar_dir, root_dir).items():
            per_stack_counts[f"{pillar_dir}::{stack_id}"] = n

    print(f"\nStacks processed  : {len(per_stack_counts)}")
    print(f"Total grid sites  : {sum(per_stack_counts.values())}")
    print(f"Output written to : {config.PILLAR_DETECT_DIR}")
    if not per_stack_counts:
        raise PipelineError("pillar detection produced no stacks")


# =============================================================================
# STAGE 1 -- CROPS  (1B CROPALL; with both "all" options off it is 1 CROP)
# =============================================================================

def stratified_sample(table, n_wanted: int, rng: np.random.Generator) -> list[int]:
    """Sites to label first, stratified by source."""
    by_source: dict[str, list[int]] = {}
    for s in table.sites:
        by_source.setdefault(s.source, []).append(s.site_id)
    n_wanted = min(n_wanted, len(table))
    picked: list[int] = []
    total = len(table)
    for source, ids in by_source.items():
        share = min(max(1, int(round(n_wanted * len(ids) / total))), len(ids))
        picked += rng.choice(ids, size=share, replace=False).tolist()
    if len(picked) > n_wanted:
        picked = rng.choice(picked, size=n_wanted, replace=False).tolist()
    return sorted(int(p) for p in picked)


def select_sites(table, n_wanted, all_sites, rng):
    """(site_ids to crop, site_id -> labelling priority)."""
    first = stratified_sample(table, n_wanted, rng)
    if not all_sites:
        return first, {sid: 0 for sid in first}
    first_set = set(first)
    rest = [s.site_id for s in table.sites if s.site_id not in first_set]
    rng.shuffle(rest)
    priority = {sid: 0 for sid in first}
    priority.update({sid: i + 1 for i, sid in enumerate(rest)})
    return sorted(first + rest), priority


def run_crops() -> None:
    root, out = config.ROOT_DIR, config.CROPS_DIR
    rng = np.random.default_rng(config.SAMPLE_SEED)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preproc.json").write_text(json.dumps(preproc_dict(), indent=2))

    series_list = discover_series(root)
    if not series_list:
        raise PipelineError(f"no series/stack under {root} has both a pillar-detection "
                            f"coords.json and a matching '{config.MBP_DIRNAME}' stack")

    all_sites = config.CROP_ALL_SITES or config.PILLARS_PER_SERIES <= 0
    all_slices = config.CROP_ALL_SLICES or config.SLICES_PER_PILLAR <= 0
    n_first_cfg = config.PILLARS_PER_SERIES if config.PILLARS_PER_SERIES > 0 else 40
    print(f"Root : {root}\nOut  : {out}\nSeries/stacks found: {len(series_list)}")
    if all_sites:
        print("Cropping EVERY site. The stratified sample is priority 0 in the manifest; "
              "the rest follow in shuffled order.")
    if all_slices:
        print("Cropping EVERY usable z-slice. Neighbouring slices are near-duplicates: "
              "they multiply the labelling, not the information.")

    manifest: list[dict] = []
    n_total_sites = 0
    for series in series_list:
        refs = list_slices(series.mbp_dir, series.mbp_stack_id)
        if not refs:
            print(f"  ! no mbp slices for {series.name} in {series.mbp_dir} -- skipping")
            continue
        table = build_site_table(series.coords(), image_shape(refs),
                                 include_sources=site_sources())
        sel_z = select_slices(refs, config.Z_RANGE)
        describe_series(series, table, len(refs), len(sel_z))
        write_csv(out / "sites" / f"{series.name}.csv", site_table_rows(table))
        if len(table) == 0 or not sel_z:
            print(f"    ! no sites or no slices in range for {series.name} -- skipping crops")
            continue

        all_xy, pitch = table.xy, table.lattice.pitch
        picked, priority = select_sites(table, min(n_first_cfg, len(table)), all_sites, rng)

        if all_slices:
            z_picks = list(sel_z)
        else:
            # an even spread: the ambiguous crops live at the ends of the range
            n_slices = min(config.SLICES_PER_PILLAR, len(sel_z))
            z_picks = sorted({sel_z[i] for i in
                              np.linspace(0, len(sel_z) - 1, n_slices).round().astype(int)})

        (out / series.name).mkdir(parents=True, exist_ok=True)
        by_source = {s.site_id: s.source for s in table.sites}
        for zi in z_picks:
            field = normalise(load_slice_smoothed(refs, zi, config.Z_SMOOTH))
            for sid in picked:
                rel = Path(series.name) / f"s{sid:04d}_z{zi:03d}.png"
                build_crop(field, all_xy, sid, pitch).save(out / rel)
                manifest.append({
                    "filepath": rel.as_posix(), "series": series.name,
                    "site_id": sid, "source": by_source[sid], "priority": priority[sid],
                    "slice_idx": zi, "slice_key": refs[zi].key,
                    "x": int(all_xy[sid, 0]), "y": int(all_xy[sid, 1]),
                })
            print(f"    z={zi:>3} ({refs[zi].key}): {len(picked)} crops",
                  end="\r" if all_slices else "\n")
        if all_slices:
            print(f"    {len(picked)} sites x {len(z_picks)} slices = "
                  f"{len(picked) * len(z_picks)} crops" + " " * 20)
        n_total_sites += len(picked)

    # labelling order: priority first, then series/site/z
    manifest.sort(key=lambda r: (r["priority"], r["series"], r["site_id"], r["slice_idx"]))
    write_csv(out / "manifest.csv", manifest)
    if not manifest:
        raise PipelineError("no crops were written")

    n_first = sum(1 for r in manifest if r["priority"] == 0)
    print(f"\n{len(manifest)} crops from {n_total_sites} sites written to {out}")
    if all_sites and len(manifest) > n_first:
        print(f"  the first {n_first} manifest rows are the stratified sample - label those "
              f"before the remaining {len(manifest) - n_first}")
    if len(manifest) > 5000:
        print(f"  ! {len(manifest)} crops is far more than anyone hand-labels; label in "
              f"manifest order and stop when the classes stop improving.")


# =============================================================================
# STAGE 2 -- LABEL  (2 LABEL, Tk GUI)
# =============================================================================

CLASS_HELP = {0: "0  not wrapped", 1: "1  somewhat  (<50%)",
              2: "2  mostly    (>=50%, gap visible)", 3: "3  full      (complete ring)"}
DISPLAY = 720
VIEWS = ("cyan", "mbp", "raw")
YELLOW = np.array([1.00, 0.85, 0.15], dtype=np.float32)


def marker_centre(marker: np.ndarray) -> tuple[float, float]:
    """Intensity centroid of the target-marker channel."""
    w = marker.astype(np.float64)
    total = w.sum()
    if total <= 0:
        return marker.shape[1] / 2, marker.shape[0] / 2
    ys, xs = np.mgrid[0:marker.shape[0], 0:marker.shape[1]]
    return float((xs * w).sum() / total), float((ys * w).sum() / total)


def _edge(mask01: np.ndarray, grow: int = 0) -> np.ndarray:
    img = Image.fromarray((mask01 * 255).astype(np.uint8))
    if grow > 0:
        img = img.filter(ImageFilter.MaxFilter(2 * grow + 1))
    outer = np.asarray(img).astype(np.float32) / 255.0
    inner = np.asarray(img.filter(ImageFilter.MinFilter(5))).astype(np.float32) / 255.0
    return np.clip(outer - inner, 0, 1)


def render(crop, view="cyan", gamma=1.0, fill_alpha=0.18, edge_alpha=0.5,
           out=DISPLAY) -> Image.Image:
    """Composite one crop for display (the file is never modified)."""
    a = np.asarray(crop).astype(np.float32) / 255.0
    mbp, mask, marker = a[..., 0], a[..., 1], a[..., 2]
    if gamma != 1.0:
        mbp = np.power(np.clip(mbp, 0, 1), 1.0 / gamma)

    if view == "raw":
        rgb = a.copy()
    else:
        rgb = np.stack([np.zeros_like(mbp), mbp, mbp], axis=-1)
        if view == "cyan":
            m01 = (mask > 0.5).astype(np.float32)
            if fill_alpha > 0:
                f = (m01 * fill_alpha)[..., None]
                rgb = rgb * (1 - f) + YELLOW * f
            if edge_alpha > 0:
                e = (_edge(m01) * edge_alpha)[..., None]
                rgb = rgb * (1 - e) + YELLOW * e
            cx, cy = marker_centre(marker)
            yy, xx = np.mgrid[0:m01.shape[0], 0:m01.shape[1]]
            near = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (0.42 * m01.shape[0] / 2.5) ** 2
            target = m01 * near
            if target.sum() > 0:
                t = _edge(target, grow=3)[..., None]
                rgb = rgb * (1 - t) + np.array([1.0, 1.0, 1.0], np.float32) * t

    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize((out, out), Image.LANCZOS)
    if view == "mbp":
        cx, cy = marker_centre(marker)
        k = out / marker.shape[0]
        r = out * 0.20 / 2.5 * 1.9
        ImageDraw.Draw(img).ellipse([cx * k - r, cy * k - r, cx * k + r, cy * k + r],
                                    outline=(255, 255, 255), width=3)
    return img


def _bring_to_front(win) -> None:
    """Raise a Tk window above the console so its key bindings get focus."""
    try:
        win.lift()
        win.attributes("-topmost", True)
        win.after(300, lambda: win.attributes("-topmost", False))
        win.focus_force()
    except Exception:
        pass


class Labeller:
    """Tk window that shows crops one at a time and records key-press labels."""

    def __init__(self, root_dir: Path, rows: list[dict], labels_path: Path,
                 all_rows: list[dict] | None = None,
                 start_at_unlabelled: bool = True, title: str = "Wrap scoring"):
        self.root_dir = root_dir
        self.rows = rows
        # metadata for every crop, so save() never blanks rows not on screen
        self.all_rows = all_rows if all_rows is not None else rows
        self.labels_path = labels_path
        self.labels: dict[str, int] = {}
        self.history: list[str] = []
        self.view, self.gamma, self.fill, self.edge = 0, 1.6, 0.10, 0.55

        if labels_path.exists():
            with open(labels_path, newline="") as f:
                for r in csv.DictReader(f):
                    self.labels[r["filepath"]] = int(r["label"])

        self.i = (next((k for k, r in enumerate(rows) if r["filepath"] not in self.labels), 0)
                  if start_at_unlabelled else 0)

        self.win = tk.Tk()
        self.win.title(title)
        self.win.configure(bg="#111")
        self.canvas = tk.Label(self.win, bg="#111")
        self.canvas.pack()
        self.status = tk.Label(self.win, font=("TkDefaultFont", 11), justify="left",
                               bg="#111", fg="#eee")
        self.status.pack(fill="x")
        tk.Label(self.win, text="   ".join(CLASS_HELP.values()),
                 font=("TkDefaultFont", 9), bg="#111", fg="#888").pack(fill="x")
        tk.Label(self.win, text="v view   + - gamma   [ ] fill   , . outline   "
                                "<- -> move   u undo   q save & quit",
                 font=("TkDefaultFont", 9), bg="#111", fg="#666").pack(fill="x")
        self.win.bind("<Key>", self.on_key)
        self.win.protocol("WM_DELETE_WINDOW", self.quit)
        self.show()
        _bring_to_front(self.win)

    def show(self) -> None:
        if not (0 <= self.i < len(self.rows)):
            self.status.config(text="Done - press q to quit")
            return
        row = self.rows[self.i]
        crop = Image.open(self.root_dir / row["filepath"]).convert("RGB")
        self.photo = ImageTk.PhotoImage(render(crop, VIEWS[self.view], self.gamma,
                                               self.fill, self.edge, DISPLAY))
        self.canvas.config(image=self.photo)
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
                 f"fill={self.fill:.2f}  outline={self.edge:.2f}")

    def save(self) -> None:
        by_path = {r["filepath"]: r for r in self.all_rows}
        with open(self.labels_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "label", "series", "site_id", "source", "slice_key"])
            for fp, lab in self.labels.items():
                r = by_path.get(fp, {})
                w.writerow([fp, lab, r.get("series", ""), r.get("site_id", ""),
                            r.get("source", ""), r.get("slice_key", "")])

    def quit(self) -> None:
        self.save()
        self.win.destroy()

    def on_key(self, event) -> None:
        k = event.keysym
        if k in ("0", "1", "2", "3") and 0 <= self.i < len(self.rows):
            fp = self.rows[self.i]["filepath"]
            self.labels[fp] = int(k)
            self.history.append(fp)
            self.save()
            self.i += 1
        elif k == "Right":
            self.i = min(self.i + 1, len(self.rows) - 1)
        elif k == "Left":
            self.i = max(self.i - 1, 0)
        elif k == "v":
            self.view = (self.view + 1) % len(VIEWS)
        elif k in ("plus", "equal", "KP_Add"):
            self.gamma = min(4.0, self.gamma + 0.2)
        elif k in ("minus", "KP_Subtract"):
            self.gamma = max(0.4, self.gamma - 0.2)
        elif k == "bracketright":
            self.fill = min(0.9, self.fill + 0.05)
        elif k == "bracketleft":
            self.fill = max(0.0, self.fill - 0.05)
        elif k == "period":
            self.edge = min(1.0, self.edge + 0.1)
        elif k == "comma":
            self.edge = max(0.0, self.edge - 0.1)
        elif k == "u" and self.history:
            fp = self.history.pop()
            self.labels.pop(fp, None)
            self.i = next((n for n, r in enumerate(self.rows) if r["filepath"] == fp), self.i)
            self.save()
        elif k == "q":
            self.quit()
            return
        self.show()


def run_label() -> None:
    if tk is None:
        raise PipelineError("labelling needs tkinter (a desktop Python install)")
    crops = config.CROPS_DIR
    manifest = crops / "manifest.csv"
    if not manifest.exists():
        raise PipelineError(f"no manifest at {manifest} - run the crop stage first")
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise PipelineError(f"{manifest} is empty")

    print(f"{len(rows)} crops in {manifest}. Label with keys 0-3; q saves and quits "
          f"(training starts after the window closes).")
    app = Labeller(crops, rows, crops / "labels.csv")
    app.win.mainloop()

    dist = Counter(app.labels.values())
    print(f"Saved {len(app.labels)} labels to {crops / 'labels.csv'}")
    print("Distribution: " + "  ".join(f"{k}={dist.get(k, 0)} ({CLASS_NAMES[k]})"
                                       for k in range(NUM_LABELS)))


# =============================================================================
# STAGE 3 / 3B -- TRAIN  (3 TRAIN, 3B CONTRAIN)
# =============================================================================

def split_by_group(rows: list[dict], group_key: str, holdout_frac: float, seed: int):
    """Assign whole groups (series by default) to train or eval."""
    groups = sorted({r[group_key] for r in rows})
    if len(groups) < 2:
        print(f"! Only one distinct '{group_key}' - falling back to site-level "
              f"grouping. Eval will overstate cross-series performance.")
        group_key = "site_id"
        groups = sorted({r[group_key] for r in rows})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_hold = max(1, int(round(len(groups) * holdout_frac)))
    hold = {groups[i] for i in order[:n_hold]}
    train = [r for r in rows if r[group_key] not in hold]
    ev = [r for r in rows if r[group_key] in hold]
    print(f"Split on '{group_key}': {len(groups) - n_hold} train groups, "
          f"{n_hold} held out ({sorted(hold)[:6]}{'...' if n_hold > 6 else ''})")
    return train, ev


def quadratic_weighted_kappa(y_true, y_pred, n_classes=NUM_LABELS) -> float:
    """Agreement metric for ordinal labels: large errors cost more."""
    O = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    w = np.array([[(i - j) ** 2 for j in range(n_classes)]
                  for i in range(n_classes)]) / (n_classes - 1) ** 2
    E = np.outer(np.bincount(y_true, minlength=n_classes),
                 np.bincount(y_pred, minlength=n_classes)).astype(float)
    E *= O.sum() / max(E.sum(), 1e-9)
    denom = (w * E).sum()
    return 1.0 - (w * O).sum() / denom if denom > 0 else 0.0


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": float((preds == labels).mean()),
            "mae": float(np.abs(preds - labels).mean()),
            "qwk": float(quadratic_weighted_kappa(labels.astype(int), preds.astype(int))),
            "adjacent_acc": float((np.abs(preds - labels) <= 1).mean())}


_ML = None


def ml():
    """torch / transformers and the classes built on them, imported on first use."""
    global _ML
    if _ML is not None:
        return _ML
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset as TorchDataset
        from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                                  EarlyStoppingCallback, Trainer, TrainingArguments)
    except ImportError as e:
        raise PipelineError(f"training/scoring needs torch and transformers ({e}).  "
                            f"pip install torch transformers accelerate")

    class CropDataset(TorchDataset):
        """Paths only; images decoded on demand. Absolute filepaths ignore root."""

        def __init__(self, root: Path, rows: list[dict], augment: bool, seed: int = 0):
            self.root, self.rows = Path(root), rows
            self.aug = DihedralTransform(np.random.default_rng(seed)) if augment else None

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            row = self.rows[i]
            img = Image.open(self.root / row["filepath"]).convert("RGB")
            if self.aug is not None:
                img = self.aug(img)
            return {"image": img, "label": int(row["label"])}

    def make_collator(processor):
        def collate(batch):
            enc = processor(images=[b["image"] for b in batch], return_tensors="pt")
            enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
            return enc
        return collate

    class WeightedTrainer(Trainer):
        """HF Trainer with a class-weighted cross-entropy loss."""

        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weights = (self.class_weights.to(outputs.logits.device)
                       if self.class_weights is not None else None)
            loss = nn.CrossEntropyLoss(weight=weights)(
                outputs.logits.view(-1, NUM_LABELS), labels.view(-1))
            inputs["labels"] = labels
            return (loss, outputs) if return_outputs else loss

    _ML = types.SimpleNamespace(
        torch=torch, AutoImageProcessor=AutoImageProcessor,
        AutoModelForImageClassification=AutoModelForImageClassification,
        TrainingArguments=TrainingArguments, EarlyStoppingCallback=EarlyStoppingCallback,
        CropDataset=CropDataset, make_collator=make_collator,
        WeightedTrainer=WeightedTrainer)
    return _ML


def pick_device(L) -> str:
    if config.DEVICE and config.DEVICE != "auto":
        return config.DEVICE
    return "cuda" if L.torch.cuda.is_available() else "cpu"


def _training_args(L, out: Path, lr: float, epochs: int, device: str, seed: int):
    """Shared Trainer settings: evaluate/checkpoint each epoch, keep best qwk."""
    return L.TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=lr, warmup_ratio=0.1, num_train_epochs=epochs,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="qwk", greater_is_better=True,
        logging_steps=25, report_to="none", remove_unused_columns=False,
        fp16=(device == "cuda"), seed=seed)


def _class_weights(L, dist: Counter):
    if not config.CLASS_WEIGHTS:
        return None
    counts = np.array([max(dist.get(i, 0), 1) for i in range(NUM_LABELS)], float)
    w = counts.sum() / (NUM_LABELS * counts)
    print(f"Class weights: {np.round(w, 3).tolist()}")
    return L.torch.tensor(w, dtype=L.torch.float)


def _confusion(yt, yp) -> np.ndarray:
    cm = np.zeros((NUM_LABELS, NUM_LABELS), int)
    for t, q in zip(yt, yp):
        cm[t, q] += 1
    return cm


def run_train() -> None:
    """Stage 3: fine-tune BACKBONE on CROPS_DIR/labels.csv -> MODEL_DIR."""
    L = ml()
    crops, out = config.CROPS_DIR, config.MODEL_DIR
    labels_csv = crops / "labels.csv"
    if not labels_csv.exists():
        raise PipelineError(f"no labels at {labels_csv} - run the labelling stage first")

    device = pick_device(L)
    print(f"Device: {device}   Backbone: {config.BACKBONE}")

    with open(labels_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (crops / r["filepath"]).exists()]
    if not rows:
        raise PipelineError(f"no usable rows in {labels_csv}")

    dist = Counter(int(r["label"]) for r in rows)
    src = Counter(r.get("source", "detected") for r in rows)
    print(f"{len(rows)} labelled crops   sources: {dict(src)}")
    print("Distribution: " + "  ".join(f"{k}={dist.get(k, 0)} ({CLASS_NAMES[k]})"
                                       for k in range(NUM_LABELS)))
    thin = [k for k in range(NUM_LABELS) if dist.get(k, 0) < 20]
    if thin:
        print(f"! Classes {thin} have <20 examples - the model will be unreliable on them.")

    train_rows, eval_rows = split_by_group(rows, config.GROUP_KEY, config.HOLDOUT_FRAC,
                                           config.TRAIN_SEED)
    print(f"Train: {len(train_rows)}  |  Eval: {len(eval_rows)}")
    if not train_rows or not eval_rows:
        raise PipelineError("the train/eval split left one side empty - label more series")

    processor = L.AutoImageProcessor.from_pretrained(config.BACKBONE)
    model = L.AutoModelForImageClassification.from_pretrained(
        config.BACKBONE, num_labels=NUM_LABELS, ignore_mismatched_sizes=True,
        id2label={i: CLASS_NAMES[i] for i in range(NUM_LABELS)},
        label2id={v: k for k, v in CLASS_NAMES.items()}).to(device)

    trainer = L.WeightedTrainer(
        model=model,
        args=_training_args(L, out, config.LEARNING_RATE, config.EPOCHS, device,
                            config.TRAIN_SEED),
        train_dataset=L.CropDataset(crops, train_rows, True, config.TRAIN_SEED),
        eval_dataset=L.CropDataset(crops, eval_rows, False),
        data_collator=L.make_collator(processor),
        compute_metrics=compute_metrics,
        class_weights=_class_weights(L, dist))

    trainer.train()
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))

    # carry the crop geometry with the weights so scoring cannot drift
    if (crops / "preproc.json").exists():
        shutil.copy(crops / "preproc.json", out / "preproc.json")
    else:
        print("! no preproc.json in the crops folder - scoring will fall back to settings")

    print("\nFinal eval:")
    for k, v in trainer.evaluate().items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    pred = trainer.predict(L.CropDataset(crops, eval_rows, False))
    cm = _confusion(pred.label_ids.astype(int), np.argmax(pred.predictions, axis=1))
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print("       " + "".join(f"{i:>7}" for i in range(NUM_LABELS)))
    for i in range(NUM_LABELS):
        print(f"  {i}  " + "".join(f"{c:>7}" for c in cm[i]))
    np.savetxt(out / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    print(f"\nModel saved to {out}")


# preproc.json keys that must match between model and crops
GEOM_KEYS = ("context_pitches", "out_size", "pillar_r_frac", "slide_at_edges")


def read_rows(crops: Path, tag: str) -> list[dict]:
    """labels.csv with absolute filepaths and batch-tagged series names."""
    labels_csv = crops / "labels.csv"
    if not labels_csv.exists():
        raise PipelineError(f"no labels at {labels_csv}")
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        p = (crops / r["filepath"]).resolve()
        if not p.exists():
            continue
        r = dict(r)
        r["filepath"] = str(p)
        r["series"] = f"{tag}:{r.get('series', 'unknown')}"
        r["batch"] = tag
        out.append(r)
    if not out:
        raise PipelineError(f"no usable rows in {labels_csv}")
    dist = Counter(int(r["label"]) for r in out)
    print(f"  {tag}: {len(out)} crops   " + "  ".join(
        f"{CLASS_NAMES[k]}={dist.get(k, 0)}" for k in range(NUM_LABELS)))
    return out


def read_preproc(path: Path | None) -> dict | None:
    if path is None:
        return None
    f = path / "preproc.json"
    return json.loads(f.read_text()) if f.exists() else None


def check_geometry(model_pre, crops_pre, tag: str, allow: bool) -> None:
    """Stop if the crops were cut with different geometry from the model."""
    if model_pre is None or crops_pre is None:
        print(f"! no preproc.json for {tag} - cannot verify crop geometry. Continuing blind.")
        return
    diff = {k: (model_pre.get(k), crops_pre.get(k)) for k in GEOM_KEYS
            if k in model_pre and k in crops_pre and model_pre[k] != crops_pre[k]}
    if not diff:
        return
    msg = "  ".join(f"{k}: model={a} {tag}={b}" for k, (a, b) in diff.items())
    if not allow:
        raise PipelineError(f"{tag} crops were cut with different geometry -- {msg}. "
                            f"Recut them with the model's settings, or tick "
                            f"'Allow crop-geometry mismatch'.")
    print(f"! proceeding with mismatched crop geometry for {tag} -- {msg}")


def run_continue_train() -> None:
    """Stage 3B: continue PARENT_MODEL_DIR on CROPS_DIR (+ OLD_CROPS_DIR replay)
    -> MODEL_DIR."""
    L = ml()
    parent, out = config.PARENT_MODEL_DIR, config.MODEL_DIR
    new_crops, old_crops = config.CROPS_DIR, config.OLD_CROPS_DIR

    print("Crop sets:")
    batches = {"new": read_rows(new_crops, "new")}
    if old_crops:
        batches["old"] = read_rows(old_crops, "old")
    else:
        print("  ! no old crop set: training on the new batch alone. Whatever the first "
              "batch taught the model is free to drift, with no old held-out data to see it.")

    model_pre = read_preproc(parent)
    allow = config.CT_ALLOW_PREPROC_MISMATCH
    check_geometry(model_pre, read_preproc(new_crops), "new", allow)
    if old_crops:
        check_geometry(model_pre, read_preproc(old_crops), "old", allow)

    # split each batch on its own series
    train_rows: list[dict] = []
    eval_rows: dict[str, list[dict]] = {}
    for tag, rows in batches.items():
        print(f"\n[{tag}] ", end="")
        tr, ev = split_by_group(rows, config.GROUP_KEY, config.HOLDOUT_FRAC, config.TRAIN_SEED)
        train_rows += tr
        eval_rows[tag] = ev
    eval_all = [r for ev in eval_rows.values() for r in ev]
    print(f"\nTrain: {len(train_rows)}  |  Eval: " +
          "  ".join(f"{k}={len(v)}" for k, v in eval_rows.items()))

    dist = Counter(int(r["label"]) for r in train_rows)
    thin = [k for k in range(NUM_LABELS) if dist.get(k, 0) < 20]
    if thin:
        print(f"! Classes {thin} have <20 training examples in the union.")

    device = pick_device(L)
    processor = L.AutoImageProcessor.from_pretrained(parent)
    model = L.AutoModelForImageClassification.from_pretrained(parent).to(device)
    if model.config.num_labels != NUM_LABELS:
        raise PipelineError(f"parent model has {model.config.num_labels} labels, not "
                            f"{NUM_LABELS} - this is not the same task")
    print(f"Device: {device}   Continuing from: {parent}")

    if config.CT_FREEZE_LAYERS > 0:
        base = getattr(model, model.base_model_prefix, None)
        layers = getattr(getattr(base, "encoder", None), "layer", None)
        if layers is None:
            print("! freeze layers: no encoder.layer on this backbone, skipping")
        else:
            for prm in base.embeddings.parameters():
                prm.requires_grad = False
            for blk in layers[:config.CT_FREEZE_LAYERS]:
                for prm in blk.parameters():
                    prm.requires_grad = False
            n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
            print(f"  froze embeddings + first {config.CT_FREEZE_LAYERS} blocks "
                  f"({n_train / 1e6:.1f}M params still training)")

    trainer = L.WeightedTrainer(
        model=model,
        args=_training_args(L, out, config.CT_LEARNING_RATE, config.CT_EPOCHS, device,
                            config.TRAIN_SEED),
        train_dataset=L.CropDataset(Path("."), train_rows, True, config.TRAIN_SEED),
        eval_dataset=L.CropDataset(Path("."), eval_all, False),
        data_collator=L.make_collator(processor),
        compute_metrics=compute_metrics,
        class_weights=_class_weights(L, dist),
        callbacks=([L.EarlyStoppingCallback(early_stopping_patience=2)]
                   if config.CT_EARLY_STOPPING else None))

    def per_batch_eval(tag: str) -> dict:
        """Metrics per crop set - the pooled number can hide a collapse on one."""
        report = {}
        for name, rows in eval_rows.items():
            pred = trainer.predict(L.CropDataset(Path("."), rows, False))
            yp = np.argmax(pred.predictions, axis=1)
            yt = pred.label_ids.astype(int)
            m = {"n": len(rows), "accuracy": float((yp == yt).mean()),
                 "adjacent_acc": float((np.abs(yp - yt) <= 1).mean()),
                 "mae": float(np.abs(yp - yt).mean()),
                 "qwk": float(quadratic_weighted_kappa(yt, yp))}
            report[name] = m
            print(f"  [{tag}] {name:<4} n={m['n']:<5} acc={m['accuracy']:.3f} "
                  f"adj={m['adjacent_acc']:.3f} mae={m['mae']:.3f} qwk={m['qwk']:.3f}")
            if tag == "after":
                np.savetxt(out / f"confusion_matrix_{name}.csv", _confusion(yt, yp),
                           fmt="%d", delimiter=",")
        return report

    print("\nBefore training:")
    before = per_batch_eval("before")
    if config.CT_EVAL_ONLY:
        print("\nEval only: stopping here. The 'new' row is how the parent model does on "
              "the new crops - if that is already good enough, the safest change is none.")
        return

    trainer.train()
    print("\nAfter training:")
    out.mkdir(parents=True, exist_ok=True)
    after = per_batch_eval("after")

    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    src_pre = (new_crops / "preproc.json" if (new_crops / "preproc.json").exists()
               else parent / "preproc.json")
    if src_pre.exists():
        shutil.copy(src_pre, out / "preproc.json")
    else:
        print("! no preproc.json to carry forward - scoring will fall back to settings")

    (out / "train_manifest.json").write_text(json.dumps({
        "parent_model": str(parent), "new_crops": str(new_crops),
        "old_crops": str(old_crops) if old_crops else None, "replay": bool(old_crops),
        "n_train": len(train_rows), "n_eval": {k: len(v) for k, v in eval_rows.items()},
        "train_class_counts": {CLASS_NAMES[k]: dist.get(k, 0) for k in range(NUM_LABELS)},
        "hyperparams": {"lr": config.CT_LEARNING_RATE, "epochs": config.CT_EPOCHS,
                        "batch_size": config.BATCH_SIZE,
                        "freeze_layers": config.CT_FREEZE_LAYERS,
                        "group_key": config.GROUP_KEY, "holdout_frac": config.HOLDOUT_FRAC,
                        "class_weights": bool(config.CLASS_WEIGHTS),
                        "seed": config.TRAIN_SEED},
        "metrics_before": before, "metrics_after": after,
    }, indent=2))

    print("\nqwk, before -> after:")
    for name in eval_rows:
        d = after[name]["qwk"] - before[name]["qwk"]
        print(f"  {name:<4} {before[name]['qwk']:.3f} -> {after[name]['qwk']:.3f}  ({d:+.3f})")
    if "old" in after and after["old"]["qwk"] < before["old"]["qwk"] - 0.02:
        print("  ! the old batch got worse - that is forgetting, not noise. Lower the "
              "learning rate, fewer epochs, or freeze 6 layers.")
    print(f"\nModel saved to {out}")
    print("Re-score EVERY series with this model before comparing conditions.")


# =============================================================================
# STAGE 4 -- SCORE  (4B SCORE)
# =============================================================================

CLASS_COLOURS = {0: (90, 90, 90), 1: (60, 140, 255), 2: (255, 190, 40), 3: (60, 220, 90)}


def _consec(preds: np.ndarray, k: int) -> int:
    """Highest class held for k consecutive slices."""
    for c in range(NUM_LABELS - 1, 0, -1):
        run = 0
        for p in preds:
            run = run + 1 if p >= c else 0
            if run >= k:
                return c
    return 0


# maxmip is folded in after the MIP pass
AGG_FUNCS = {
    "max":      lambda pr, pb: int(pr.max()),
    "consec2":  lambda pr, pb: _consec(pr, 2),
    "consec3":  lambda pr, pb: _consec(pr, 3),
    "meanprob": lambda pr, pb: int(np.argmax(pb.mean(axis=0))),
    "q90":      lambda pr, pb: int(np.floor(np.percentile(pr, 90))),
}


def counts_of(classes) -> dict[int, int]:
    c = Counter(int(v) for v in classes)
    return {k: c.get(k, 0) for k in range(NUM_LABELS)}


def _vmax(ref, st: np.ndarray) -> float:
    a = np.asarray(ref)
    if np.issubdtype(a.dtype, np.integer):
        return float(np.iinfo(a.dtype).max)
    return 1.0 if float(st.max()) <= 1.0 else 255.0


def _as_like(ref, arr: np.ndarray):
    """Rebuild arr in the container/dtype build_crop produced."""
    a = np.asarray(ref)
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        arr = np.clip(np.rint(arr), info.min, info.max).astype(a.dtype)
    else:
        arr = arr.astype(a.dtype)
    if isinstance(ref, Image.Image):
        return Image.fromarray(arr, mode=ref.mode)
    return arr


def _match_1(m: np.ndarray, st: np.ndarray) -> np.ndarray:
    """Histogram-match m to the average single-slice histogram of st."""
    ref = np.sort(st.reshape(len(st), -1), axis=1).mean(axis=0)
    flat = m.ravel()
    out = np.empty_like(flat, dtype=np.float32)
    out[np.argsort(flat, kind="stable")] = ref
    return out.reshape(m.shape)


def match_to_slices(m: np.ndarray, st: np.ndarray) -> np.ndarray:
    if m.ndim == 3:
        out = np.empty_like(m, dtype=np.float32)
        for ch in range(m.shape[2]):
            out[..., ch] = _match_1(m[..., ch], st[..., ch])
        return out
    return _match_1(m, st)


def project_crops(crops, proj="max", dull="match", gamma=1.6, scale=0.7, topk=2):
    """Project aligned crops of one site to one dulled crop."""
    st = np.stack([np.asarray(c).astype(np.float32) for c in crops])
    if proj == "topk" and len(st) > 1:
        k = min(max(int(topk), 1), len(st))
        m = np.sort(st, axis=0)[-k:].mean(axis=0)
    else:
        m = st.max(axis=0)
    if dull == "match" and len(st) > 1:
        m = match_to_slices(m, st)
    elif dull == "gamma":
        v = _vmax(crops[0], st)
        m = v * np.clip(m / v, 0.0, 1.0) ** float(gamma)
    elif dull == "scale":
        m = m * float(scale)
    return _as_like(crops[0], m)


def normalise_stack(raws, p_lo=1.0, p_hi=99.5, step=4) -> list:
    """8-bit stack normalised against shared percentiles."""
    sample = np.concatenate([np.asarray(r)[::step, ::step].ravel() for r in raws])
    lo, hi = np.percentile(sample.astype(np.float32), [p_lo, p_hi])
    if hi <= lo:
        hi = lo + 1.0
    return [(np.clip((np.asarray(r).astype(np.float32) - lo) / (hi - lo), 0, 1) * 255
             ).astype(np.uint8) for r in raws]


def score_crops(L, model, processor, crops, device) -> np.ndarray:
    """(len(crops), 4) probabilities for ready-made crops."""
    with L.torch.no_grad():
        enc = processor(images=crops, return_tensors="pt").to(device)
        logits = model(**enc).logits
        return L.torch.softmax(logits, -1).float().cpu().numpy()


def score_field(L, model, processor, field8, all_xy, pitch, geo, device, batch_size):
    """(n_sites, 4) probabilities for one slice."""
    out = np.zeros((len(all_xy), NUM_LABELS), dtype=np.float32)
    for start in range(0, len(all_xy), batch_size):
        idxs = range(start, min(start + batch_size, len(all_xy)))
        crops = [build_crop(field8, all_xy, i, pitch, **geo) for i in idxs]
        out[start:start + len(crops)] = score_crops(L, model, processor, crops, device)
    return out


def save_mip_qc(path: Path, ref, mip) -> None:
    """best single slice | projection, side by side."""
    def rgb8(x):
        a = np.asarray(x)
        if np.issubdtype(a.dtype, np.floating):
            a = a * 255.0 if a.max() <= 1.0 else a
        a = np.clip(a, 0, 255).astype(np.uint8)
        return np.stack([a] * 3, -1) if a.ndim == 2 else a[..., :3]

    a, b = rgb8(ref), rgb8(mip)
    gap = np.full((a.shape[0], 4, 3), 255, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([a, gap, b], axis=1)).save(path)


def score_site_mips(L, model, processor, fields, preds, all_xy, pitch, geo, device,
                    batch_size, qc_dir: Path | None = None, qc_limit=0):
    """Project and score one crop per site over the slices where it scored > 0.
    Returns (probs (n_sites, 4), n_used (n_sites,))."""
    n_sites = len(all_xy)
    probs = np.zeros((n_sites, NUM_LABELS), dtype=np.float32)
    probs[:, 0] = 1.0
    n_used = np.zeros(n_sites, dtype=int)
    pend: list[tuple] = []
    n_qc = 0

    def flush():
        nonlocal n_qc
        if not pend:
            return
        p = score_crops(L, model, processor, [c for _, c, _, _ in pend], device)
        for k, (sid, mip, ref, smax) in enumerate(pend):
            probs[sid] = p[k]
            cls = int(p[k].argmax())
            if qc_dir is not None and n_qc < qc_limit and cls > smax:
                save_mip_qc(qc_dir / f"site{sid:04d}_slice{smax}_mip{cls}.png", ref, mip)
                n_qc += 1
        pend.clear()

    for sid in range(n_sites):
        keep = np.nonzero(preds[:, sid] > 0)[0]
        n_used[sid] = len(keep)
        if len(keep) == 0:
            continue
        crops = [build_crop(fields[n], all_xy, sid, pitch, **geo) for n in keep]
        pend.append((sid,
                     project_crops(crops, proj=config.MIP_PROJ, dull=config.MIP_DULL,
                                   gamma=config.MIP_GAMMA, scale=config.MIP_SCALE,
                                   topk=config.MIP_TOPK),
                     crops[int(np.argmax(preds[keep, sid]))],
                     int(preds[keep, sid].max())))
        if len(pend) >= batch_size:
            flush()
        print(f"    MIP {sid + 1}/{n_sites}", end="\r")
    flush()
    print(" " * 40, end="\r")
    return probs, n_used


def qc_overlay(field8, table, classes, path: Path, pitch: float) -> None:
    """Field image with a ring per site coloured by predicted class."""
    img = Image.merge("RGB", [Image.fromarray(field8)] * 3)
    draw = ImageDraw.Draw(img)
    r = max(4.0, pitch * config.PILLAR_R_FRAC * 1.35)
    for s, c in zip(table.sites, classes):
        width = 3 if s.source == "detected" else 6
        draw.ellipse([s.x - r, s.y - r, s.x + r, s.y + r],
                     outline=CLASS_COLOURS[int(c)], width=width)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def run_score() -> None:
    """Stage 4: score every series under ROOT_DIR with MODEL_DIR -> RESULTS_DIR."""
    L = ml()
    root, model_dir, out = config.ROOT_DIR, config.MODEL_DIR, config.RESULTS_DIR
    agg_rule, denominator = config.AGG_RULE, config.PRIMARY_DENOMINATOR
    use_mip, batch_size = bool(config.MIP_ENABLED), config.INFER_BATCH
    if not use_mip and agg_rule == "maxmip":
        print("! MIP pass off with rule maxmip: counts will be identical to max.")

    pre_path = model_dir / "preproc.json"
    if pre_path.exists():
        pre = json.loads(pre_path.read_text())
    else:
        pre = preproc_dict()
        print("! no preproc.json with the model - using the current crop settings. If the "
              "training crops were made differently, scores will be wrong.")
    geo = {k: pre[k] for k in ("context_pitches", "out_size", "pillar_r_frac",
                               "slide_at_edges") if k in pre}
    z_smooth = pre.get("z_smooth", config.Z_SMOOTH)

    device = pick_device(L)
    print(f"Device: {device}   Model: {model_dir}   Aggregation: {agg_rule}")
    if use_mip:
        print(f"MIP: proj={config.MIP_PROJ}"
              f"{f'(k={config.MIP_TOPK})' if config.MIP_PROJ == 'topk' else ''}  "
              f"dull={config.MIP_DULL}  global_norm={config.MIP_GLOBAL_NORM}")
    else:
        print("MIP pass: off")

    processor = L.AutoImageProcessor.from_pretrained(model_dir)
    model = L.AutoModelForImageClassification.from_pretrained(model_dir).to(device).eval()

    series_list = discover_series(root)
    if not series_list:
        raise PipelineError(f"no series/stack under {root} has both a pillar-detection "
                            f"coords.json and a matching '{config.MBP_DIRNAME}' stack")

    stack_rows: list[dict] = []
    series_rows: list[dict] = []
    for series in series_list:
        # only THIS stack's slices - an mbp folder can hold several stacks
        refs = list_slices(series.mbp_dir, series.mbp_stack_id)
        if not refs:
            print(f"  ! no mbp slices for {series.name} -- skipping")
            continue
        table = build_site_table(series.coords(), image_shape(refs),
                                 include_sources=site_sources())
        sel_z = select_slices(refs, config.Z_RANGE)
        describe_series(series, table, len(refs), len(sel_z))
        write_csv(out / "sites" / f"{series.name}.csv", site_table_rows(table))
        if len(table) == 0 or not sel_z:
            print(f"  ! no sites or no slices in range for {series.name} -- skipping")
            continue

        all_xy, pitch = table.xy, table.lattice.pitch
        probs = np.zeros((len(sel_z), len(table), NUM_LABELS), dtype=np.float32)
        last_field = None
        fields: list = [None] * len(sel_z)
        raws: list = [None] * len(sel_z)

        for n, zi in enumerate(sel_z):
            raw = load_slice_smoothed(refs, zi, z_smooth)
            field8 = normalise(raw)
            probs[n] = score_field(L, model, processor, field8, all_xy, pitch, geo,
                                   device, batch_size)
            if use_mip:
                if config.MIP_GLOBAL_NORM:
                    raws[n] = raw
                else:
                    fields[n] = field8
            last_field = field8
            print(f"    z={zi:>3}  ({n + 1}/{len(sel_z)})", end="\r")
        print(" " * 40, end="\r")

        preds = probs.argmax(axis=2)
        src = [s.source for s in table.sites]

        if use_mip:
            if config.MIP_GLOBAL_NORM:
                fields = normalise_stack(raws)
                raws = []
            mip_probs, n_mip = score_site_mips(
                L, model, processor, fields, preds, all_xy, pitch, geo, device, batch_size,
                qc_dir=out / "qc" / "mips" / series.name, qc_limit=max(config.MIP_QC, 0))
        else:
            mip_probs = np.zeros((len(table), NUM_LABELS), dtype=np.float32)
            mip_probs[:, 0] = 1.0
            n_mip = np.zeros(len(table), dtype=int)
        fields, raws = [], []
        mip_pred = mip_probs.argmax(axis=1).astype(int)

        write_csv(out / "per_slice" / f"{series.name}.csv", [
            {"site_id": sid, "source": src[sid], "slice_idx": zi,
             "slice_key": refs[zi].key, "pred": int(preds[n, sid]),
             **{f"p{c}": round(float(probs[n, sid, c]), 4) for c in range(NUM_LABELS)}}
            for n, zi in enumerate(sel_z) for sid in range(len(table))])

        det = table.detected_mask
        for n, zi in enumerate(sel_z):
            c_all, c_det = counts_of(preds[n]), counts_of(preds[n][det])
            stack_rows.append({
                "series": series.name, "slice_idx": zi, "slice_key": refs[zi].key,
                **{f"det_class_{k}": c_det[k] for k in range(NUM_LABELS)},
                **{f"all_class_{k}": c_all[k] for k in range(NUM_LABELS)}})

        agg = {rule: np.array([fn(preds[:, s], probs[:, s, :]) for s in range(len(table))],
                              dtype=int) for rule, fn in AGG_FUNCS.items()}
        agg["maxmip"] = np.maximum(agg["max"], mip_pred).astype(int)
        upgraded = mip_pred > agg["max"]

        write_csv(out / "per_pillar" / f"{series.name}.csv", [
            {"site_id": s.site_id, "x": s.x, "y": s.y, "source": s.source,
             "grid_i": s.grid_i, "grid_j": s.grid_j, "n_slices": len(sel_z),
             **{f"agg_{r}": int(agg[r][s.site_id]) for r in AGG_RULES},
             "mip_class": int(mip_pred[s.site_id]),
             "n_mip_slices": int(n_mip[s.site_id]),
             "mip_conf": round(float(mip_probs[s.site_id].max()), 4),
             "mip_upgraded": bool(upgraded[s.site_id]),
             "mean_conf": round(float(probs[:, s.site_id, :].max(axis=1).mean()), 4)}
            for s in table.sites])

        if config.WRITE_QC_OVERLAY and last_field is not None:
            qc_overlay(last_field, table, agg[agg_rule], out / "qc" / f"{series.name}.png", pitch)

        c_det, c_all = counts_of(agg[agg_rule][det]), counts_of(agg[agg_rule])
        n_det, n_all = int(det.sum()), len(table)
        base = c_det if denominator == "detected" else c_all
        n_base = n_det if denominator == "detected" else n_all
        keep = det if denominator == "detected" else np.ones(n_all, bool)

        row = {
            "series": series.name, "n_detected": n_det,
            "n_inferred": table.counts_by_source()["inferred"],
            "n_off_lattice": table.counts_by_source()["off_lattice"],
            "n_all_sites": n_all, "lattice_ok": table.lattice_ok,
            "pitch_px": round(pitch, 1),
            "rotation_deg": round(table.lattice.rotation_deg, 2),
            "n_slices": len(sel_z), "agg_rule": agg_rule, "denominator": denominator,
            "mip_proj": config.MIP_PROJ if use_mip else "off",
            "mip_topk": config.MIP_TOPK if config.MIP_PROJ == "topk" else "",
            "mip_dull": config.MIP_DULL if use_mip else "off",
            "mip_global_norm": bool(config.MIP_GLOBAL_NORM and use_mip),
            "n_mip_scored": int((n_mip[keep] > 0).sum()),
            "n_mip_upgraded": int(upgraded[keep].sum()),
            **{f"class_{k}_{CLASS_NAMES[k]}": base[k] for k in range(NUM_LABELS)},
            **{f"pct_class_{k}": round(100 * base[k] / max(n_base, 1), 2)
               for k in range(NUM_LABELS)},
            **{f"det_class_{k}": c_det[k] for k in range(NUM_LABELS)},
            **{f"all_class_{k}": c_all[k] for k in range(NUM_LABELS)},
        }
        for rule in AGG_RULES:
            if rule != agg_rule:
                c = counts_of(agg[rule][keep])
                row.update({f"alt_{rule}_class_{k}": c[k] for k in range(NUM_LABELS)})
        series_rows.append(row)

        print(f"  [{agg_rule}, {denominator}] " +
              "  ".join(f"{CLASS_NAMES[k]}={base[k]}" for k in range(NUM_LABELS)))
        if use_mip:
            n_up = int(upgraded[keep].sum())
            print(f"  MIP: {int((n_mip[keep] > 0).sum())} pillars projected, "
                  f"{n_up} raised above the per-slice max")
            if n_up > 0.1 * max(n_base, 1):
                print(f"  ! the MIP is driving {100 * n_up / max(n_base, 1):.0f}% of the "
                      f"pillars - check qc/mips/{series.name}/ before trusting the count")
        c2 = counts_of(agg["consec2"][keep])
        drift = sum(abs(base[k] - c2[k]) for k in range(NUM_LABELS)) / 2
        if agg_rule in ("max", "maxmip") and drift > 0.1 * max(n_base, 1):
            print(f"  ! {agg_rule} and consec2 disagree on ~{int(drift)} pillars "
                  f"({100 * drift / max(n_base, 1):.0f}%) - likely single-slice misfires")

    if not series_rows:
        raise PipelineError("no series could be scored")

    write_csv(out / "per_stack_counts.csv", stack_rows)
    write_csv(out / "per_series_counts.csv", series_rows)

    total = {"series": "TOTAL", "agg_rule": agg_rule, "denominator": denominator,
             "mip_proj": config.MIP_PROJ if use_mip else "off",
             "mip_dull": config.MIP_DULL if use_mip else "off"}
    for k in ("n_detected", "n_inferred", "n_off_lattice", "n_all_sites", "n_slices",
              "n_mip_scored", "n_mip_upgraded"):
        total[k] = sum(r[k] for r in series_rows)
    for k in range(NUM_LABELS):
        total[f"class_{k}_{CLASS_NAMES[k]}"] = sum(
            r[f"class_{k}_{CLASS_NAMES[k]}"] for r in series_rows)
    n_base = total["n_detected"] if denominator == "detected" else total["n_all_sites"]
    for k in range(NUM_LABELS):
        total[f"pct_class_{k}"] = round(
            100 * total[f"class_{k}_{CLASS_NAMES[k]}"] / max(n_base, 1), 2)

    cols = (["series", "n_detected", "n_inferred", "n_off_lattice", "n_all_sites",
             "n_slices", "agg_rule", "denominator", "mip_proj", "mip_dull",
             "n_mip_scored", "n_mip_upgraded"] +
            [f"class_{k}_{CLASS_NAMES[k]}" for k in range(NUM_LABELS)] +
            [f"pct_class_{k}" for k in range(NUM_LABELS)])
    write_csv(out / "directory_summary.csv",
              [{k: r.get(k, "") for k in cols} for r in series_rows] +
              [{k: total.get(k, "") for k in cols}])

    print(f"\n{'series':<30}{'sites':>7}" +
          "".join(f"{CLASS_NAMES[k]:>14}" for k in range(NUM_LABELS)) + f"{'mip_up':>9}")
    for r in series_rows + [total]:
        n = r.get("n_detected" if denominator == "detected" else "n_all_sites", "")
        print(f"{r['series']:<30}{n:>7}" +
              "".join(f"{r[f'class_{k}_{CLASS_NAMES[k]}']:>14}" for k in range(NUM_LABELS)) +
              f"{r.get('n_mip_upgraded', ''):>9}")
    print(f"\nResults -> {out}")


# =============================================================================
# STAGE N -- NUCLEI COUNT  (MIP per nuclei stack -> Fiji StarDist macro)
# =============================================================================

FIJI_LAUNCHERS = ("ImageJ-win64.exe", "fiji-windows-x64.exe", "ImageJ-win32.exe",
                  "ImageJ-macosx", "fiji-macos-arm64", "fiji-macos-x64",
                  "ImageJ-linux64", "fiji-linux-x64")

# Same StarDist settings and rejection filters as the two
# LIF_Nuclei_Count_ExcludeHollow_5um macros, but it reads the pipeline's MIPs
# instead of the LIF, so there is no channel pick and no Z projection here.
NUCLEI_MACRO = r"""// Written by wrap_pipeline.py (stage N) and rewritten on every run - edit
// the copy in wrap_pipeline.py, not this file.
// Same StarDist settings and filters as LIF_Nuclei_Count_ExcludeHollow_5um.ijm,
// applied to maximum projections the pipeline made from the nuclei channel.
// Argument: the jobs file. Line 1: #params, probability, minimum diameter (um),
// dark-centre filter (1/0), minimum centre ratio, done-marker path.
// Then one line per MIP: path, output prefix, pixel width um, pixel height um.

jobsPath = getArgument();
if (jobsPath == "")
    exit("This macro is started by wrap_pipeline.py with a jobs file.");

lines = split(File.openAsString(jobsPath), "\r\n");
params = split(lines[0], "\t");
probability = parseFloat(params[1]);
minimumDiameterUm = parseFloat(params[2]);
filterHollow = parseInt(params[3]);
minimumCentreRatio = parseFloat(params[4]);
doneFile = params[5];
minimumAreaUm2 = PI * minimumDiameterUm * minimumDiameterUm / 4;
nJobs = lines.length - 1;

close("*");
roiManager("reset");
print("\\Clear");
print("wrap_pipeline nuclei count: " + nJobs + " MIP(s)");

for (j = 1; j < lines.length; j++) {
    f = split(lines[j], "\t");
    mipPath = f[0];
    prefix = f[1];
    pixelAreaUm2 = parseFloat(f[2]) * parseFloat(f[3]);
    smallCount = 0;
    hollowCount = 0;
    showProgress(j - 1, nJobs);
    showStatus("Nuclei " + j + "/" + nJobs);

    open(mipPath);
    projectionID = getImageID();
    if (bitDepth() == 24)
        run("8-bit");
    // a fixed title keeps quotes in series names out of the StarDist args
    projectionTitle = "nuclei_mip_" + j;
    rename(projectionTitle);
    run("Properties...", "channels=1 slices=1 frames=1 unit=micron " +
        "pixel_width=" + f[2] + " pixel_height=" + f[3] + " voxel_depth=1");
    run("Select None");
    resetThreshold();
    run("Grays");
    roiManager("reset");

    run("Command From Macro",
        "command=[de.csbdresden.stardist.StarDist2D], " +
        "args=['input':'" + projectionTitle + "', " +
        "'modelChoice':'Versatile (fluorescent nuclei)', " +
        "'normalizeInput':'true', " +
        "'percentileBottom':'1.0', " +
        "'percentileTop':'99.8', " +
        "'probThresh':'" + probability + "', " +
        "'nmsThresh':'0.4', 'outputType':'Both', " +
        "'nTiles':'1', 'excludeBoundary':'2', " +
        "'roiPosition':'Automatic', 'verbose':'false', " +
        "'showCsbdeepProgress':'false', " +
        "'showProbAndDist':'false'], process=[false]");

    selectImage(projectionID);
    resetThreshold();
    rawCount = roiManager("count");

    if (rawCount > 0) {
        roiManager("Save", prefix + "_unfiltered_ROIs.zip");
        saveOutline(projectionID, prefix + "_unfiltered_outlines.png");
    }

    selectImage(projectionID);
    roiManager("Show None");

    diagnostics =
        "Detection,X_pixels,Y_pixels,Area_um2," +
        "Equivalent_diameter_um,Object_mean,Centre_mean," +
        "Centre_ratio,Too_small,Dark_centre,Rejected\n";

    // Work backwards so ROI deletion is safe.
    for (r = rawCount - 1; r >= 0; r--) {
        selectImage(projectionID);
        resetThreshold();
        roiManager("Select", r);
        getRawStatistics(objectPixels, objectMean, objectMin, objectMax, objectSD);
        getSelectionBounds(x, y, ow, oh);

        areaUm2 = objectPixels * pixelAreaUm2;
        diameterUm = 2 * sqrt(areaUm2 / PI);

        // Central ellipse: 30% of the bounding box.
        cw = maxOf(1, ow * 0.3);
        ch = maxOf(1, oh * 0.3);
        makeOval(x + (ow - cw) / 2, y + (oh - ch) / 2, cw, ch);
        getRawStatistics(centrePixels, centreMean, centreMin, centreMax, centreSD);

        ratio = 0;
        if (objectMean > 0)
            ratio = centreMean / objectMean;

        tooSmall = 0;
        if (areaUm2 < minimumAreaUm2)
            tooSmall = 1;
        darkCentre = 0;
        if (filterHollow) {
            if (objectMean <= 0 || ratio < minimumCentreRatio)
                darkCentre = 1;
        }
        reject = 0;
        if (tooSmall || darkCentre)
            reject = 1;
        if (tooSmall) smallCount++;
        if (darkCentre) hollowCount++;

        diagnostics = diagnostics +
            (r + 1) + "," +
            d2s(x + ow / 2, 2) + "," +
            d2s(y + oh / 2, 2) + "," +
            d2s(areaUm2, 4) + "," +
            d2s(diameterUm, 4) + "," +
            d2s(objectMean, 4) + "," +
            d2s(centreMean, 4) + "," +
            d2s(ratio, 4) + "," +
            tooSmall + "," + darkCentre + "," + reject + "\n";

        if (reject) {
            roiManager("Select", r);
            roiManager("Delete");
        }
    }

    File.saveString(diagnostics, prefix + "_detection_measurements.csv");
    count = roiManager("count");
    if (count > 0)
        roiManager("Save", prefix + "_filtered_ROIs.zip");
    saveOutline(projectionID, prefix + "_filtered_outlines.png");

    // read back by the pipeline: initial, too small, dark centre, retained
    File.saveString("" + rawCount + "," + smallCount + "," + hollowCount + "," +
                    count + "\n", prefix + "_count.csv");
    print("[" + j + "/" + nJobs + "] " + rawCount + " initial; " +
          (rawCount - count) + " rejected; " + count + " retained  <- " + mipPath);

    close("*");
    roiManager("reset");
    run("Collect Garbage");
}

File.saveString("done\n", doneFile);
run("Quit");

// Save a display image of the current detections.
function saveOutline(imageID, path) {
    selectImage(imageID);
    roiManager("Deselect");
    run("Select None");
    if (roiManager("count") > 0)
        roiManager("Show All");
    else
        roiManager("Show None");
    run("Flatten");
    saveAs("PNG", path);
    close();
}
"""


def resolve_fiji(path) -> Path:
    """The Fiji launcher, from the launcher itself or its Fiji.app folder."""
    if not path or not str(path).strip():
        raise PipelineError("Fiji path is empty (Nuclei tab) - point it at "
                            "ImageJ-win64.exe or the Fiji.app folder")
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        for base in (p, p / "Contents" / "MacOS", p / "Fiji.app",
                     p / "Fiji.app" / "Contents" / "MacOS"):
            for name in FIJI_LAUNCHERS:
                if (base / name).is_file():
                    return base / name
        raise PipelineError(f"no Fiji launcher in {p} (looked for "
                            f"{', '.join(FIJI_LAUNCHERS)})")
    raise PipelineError(f"Fiji not found: {p}")


def fiji_has_stardist(exe: Path) -> bool | None:
    """True/False if Fiji's plugins/jars folders do/don't hold StarDist,
    None if the Fiji folder layout isn't recognised."""
    for parent in exe.parents:
        if (parent / "plugins").is_dir():
            names = [f.name.lower() for sub in ("plugins", "jars")
                     if (parent / sub).is_dir() for f in (parent / sub).iterdir()]
            return any("stardist" in n for n in names)
    return None


def read_calibration(series_dir: Path) -> tuple[float, float] | None:
    """(pixel width, pixel height) in um from LIF extraction's calibration.json."""
    f = series_dir / "calibration.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text())
        pw = float(d["pixel_width_um"])
        ph = float(d.get("pixel_height_um") or pw)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (pw, ph) if pw > 0 and ph > 0 else None


def make_mip(slice_paths, mip_path: Path) -> int:
    """Max-intensity projection of one stack's PNG slices as an 8-bit TIFF.
    Returns the number of slices used."""
    mip, n_used = None, 0
    for _, path in slice_paths:
        plane = load_intensity(path)          # per-pixel channel max = intensity
        if mip is None:
            mip = plane
        elif plane.shape != mip.shape:
            print(f"      ! skipping {path.name}: shape {plane.shape} != {mip.shape}")
            continue
        else:
            np.maximum(mip, plane, out=mip)
        n_used += 1
    mip_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(mip, 0, 255).astype(np.uint8)).save(mip_path, format="TIFF")
    return n_used


def run_nuclei_count() -> None:
    """Stage N: one MIP per nuclei stack, then StarDist counts in Fiji."""
    # absolute paths: Fiji may start in its own folder, not this one
    root = Path(config.ROOT_DIR).resolve()
    out_root = Path(config.RESULTS_DIR).resolve() / "nuclei"
    fiji = resolve_fiji(config.FIJI_PATH)
    min_diam = float(config.NUCLEI_MIN_DIAMETER_UM)
    fallback_px = float(config.NUCLEI_PIXEL_UM or 0)

    dirname = config.NUCLEI_DIRNAME.lower()
    nuclei_dirs = sorted(p for p in root.rglob("*")
                         if p.is_dir() and p.name.lower() == dirname)
    if not nuclei_dirs:
        raise PipelineError(f"no '{config.NUCLEI_DIRNAME}' folders found under {root}")
    print(f"Found {len(nuclei_dirs)} nuclei folder(s). MIPs and counts -> {out_root}")

    jobs: list[dict] = []
    for ndir in nuclei_dirs:
        try:
            series_name = ndir.parent.relative_to(root).as_posix()
        except ValueError:
            series_name = ndir.parent.name
        series_name = series_name or ndir.parent.name
        stacks, skipped = group_into_stacks(ndir)
        print(f"\n--- {series_name}: {len(stacks)} stack(s)")
        if skipped:
            print(f"    (ignored {len(skipped)} file(s) with no z-index token)")

        cal, cal_src = read_calibration(ndir.parent), "calibration.json"
        if cal is None and fallback_px > 0:
            cal, cal_src = (fallback_px, fallback_px), "settings fallback"

        for stack_id, slices in stacks.items():
            out_dir = out_root / series_name
            mip_path = out_dir / f"{stack_id}_MIP.tif"
            newest = max(p.stat().st_mtime for _, p in slices)
            if (not config.NUCLEI_REMAKE_MIP and mip_path.exists()
                    and mip_path.stat().st_mtime >= newest):
                n_used = len(slices)
                print(f"    {stack_id}: MIP already made, reusing {mip_path.name}")
            else:
                n_used = make_mip(slices, mip_path)
                print(f"    {stack_id}: MIP of {n_used} slice(s) -> {mip_path.name}")

            job = {"series": series_name, "stack": stack_id, "n_slices": n_used,
                   "mip": mip_path, "prefix": str(out_dir / stack_id),
                   "pixel_width_um": cal[0] if cal else None,
                   "pixel_height_um": cal[1] if cal else None,
                   "calibration": cal_src if cal else "none", "status": "queued"}
            if cal is None:
                if min_diam > 0:
                    job["status"] = "skipped: no pixel size"
                    print(f"    ! {stack_id}: no calibration.json and no fallback pixel "
                          f"size - cannot apply the {min_diam:g} um filter, skipping")
                else:
                    job["pixel_width_um"] = job["pixel_height_um"] = 1.0
                    job["calibration"] = "none (sizes in pixels)"
            jobs.append(job)

    todo = [j for j in jobs if j["status"] == "queued"]
    if not todo:
        raise PipelineError("no MIP has a pixel size, so none can be counted. Re-extract "
                            "the LIF files (writes calibration.json) or set the fallback "
                            "pixel size on the Nuclei tab")

    # macro + jobs file for Fiji; stale counts are removed so a failed run can't
    # be mistaken for a fresh one
    out_root.mkdir(parents=True, exist_ok=True)
    macro_path = out_root / "_count_nuclei.ijm"
    jobs_path = out_root / "_fiji_jobs.tsv"
    done_path = out_root / "_fiji_done.txt"
    macro_path.write_text(NUCLEI_MACRO, encoding="utf-8")
    done_path.unlink(missing_ok=True)
    lines = ["\t".join(["#params", f"{float(config.NUCLEI_PROB):.4f}", f"{min_diam:.4f}",
                        "1" if config.NUCLEI_FILTER_HOLLOW else "0",
                        f"{float(config.NUCLEI_MIN_CENTRE_RATIO):.4f}", str(done_path)])]
    for j in todo:
        Path(j["prefix"] + "_count.csv").unlink(missing_ok=True)
        lines.append("\t".join([str(j["mip"]), j["prefix"], f"{j['pixel_width_um']:.8f}",
                                f"{j['pixel_height_um']:.8f}"]))
    with open(jobs_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    cmd = [str(fiji), "-port0", "-macro", str(macro_path), str(jobs_path)]
    print(f"\nStarting Fiji for {len(todo)} MIP(s): {fiji}")
    print("  Fiji opens, counts every MIP and closes itself. If it stops on an error "
          "dialog, read it, then close Fiji to carry on here.")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=False)
    except OSError as e:
        raise PipelineError(f"could not start Fiji ({e})")
    finished = done_path.exists()
    print(f"Fiji closed after {_fmt_secs(time.time() - t0)}"
          f"{'' if finished else ' WITHOUT finishing the macro'}")

    rows = []
    for j in jobs:
        res = {"initial": "", "small": "", "hollow": "", "retained": ""}
        status = j["status"]
        if status == "queued":
            count_file = Path(j["prefix"] + "_count.csv")
            if count_file.exists():
                try:
                    vals = [int(v) for v in count_file.read_text().strip().split(",")]
                    res = dict(zip(("initial", "small", "hollow", "retained"), vals))
                    status = "counted"
                except ValueError:
                    status = "unreadable count file"
            else:
                status = "not counted"
        rows.append({
            "series": j["series"], "stack": j["stack"], "status": status,
            "n_slices_in_mip": j["n_slices"],
            "initial_detections": res["initial"],
            "below_min_diameter": res["small"], "dark_centre": res["hollow"],
            "total_rejected": (res["initial"] - res["retained"]
                               if status == "counted" else ""),
            "retained_nuclei": res["retained"],
            "pixel_width_um": j["pixel_width_um"] or "",
            "pixel_height_um": j["pixel_height_um"] or "",
            "calibration": j["calibration"],
            "probability": config.NUCLEI_PROB, "min_diameter_um": min_diam,
            "dark_centre_filter": bool(config.NUCLEI_FILTER_HOLLOW),
            "min_centre_ratio": config.NUCLEI_MIN_CENTRE_RATIO,
            "mip": str(j["mip"]),
        })
    write_csv(out_root / "nuclei_summary.csv", rows)

    print(f"\n{'series / stack':<60}{'initial':>9}{'rejected':>10}{'nuclei':>8}")
    for r in rows:
        label = f"{r['series']}/{r['stack']}"[-59:]
        if r["status"] == "counted":
            print(f"{label:<60}{r['initial_detections']:>9}{r['total_rejected']:>10}"
                  f"{r['retained_nuclei']:>8}")
        else:
            print(f"{label:<60}  {r['status']}")
    n_counted = sum(r["status"] == "counted" for r in rows)
    print(f"\n{n_counted}/{len(rows)} stack(s) counted -> {out_root / 'nuclei_summary.csv'}")
    if not n_counted:
        raise PipelineError("Fiji counted nothing - check its Log window for the error "
                            "(StarDist and CSBDeep update sites must be enabled)")
    if not finished:
        print("! the macro did not reach its end - stacks marked 'not counted' were "
              "missed; rerun the stage to retry (MIPs are reused)")


# =============================================================================
# RUN PLAN
# =============================================================================

def _same_dir(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


def validate_plan(v: dict) -> tuple[list[str], list[str]]:
    """(errors that block the run, warnings worth a confirmation)."""
    errors, warnings = [], []
    def p(key):
        s = str(v.get(key) or "").strip()
        return Path(s) if s else None

    root, crops, results, model = p("ROOT_DIR"), p("CROPS_DIR"), p("RESULTS_DIR"), p("MODEL_DIR")
    if root is None:
        errors.append("Root folder is empty.")
    if results is None:
        errors.append("Results folder is empty (pillar detection writes there).")
    try:
        parse_z_range(v.get("Z_RANGE"))
    except ValueError:
        errors.append(f"Z range '{v.get('Z_RANGE')}' must be blank or lo:hi, e.g. 5:45.")

    if v.get("RUN_EXTRACT") and not v.get("LIF_FILES"):
        errors.append("LIF extraction is ticked but no .lif files were added.")
    if root is not None and not v.get("RUN_EXTRACT") and not root.exists():
        errors.append(f"Root folder does not exist: {root}")
    if v.get("MODE") != "none" and not v.get("RUN_DETECT") and results is not None and \
            not (results / "pillar_detection").exists():
        warnings.append(f"Pillar detection is off and {results / 'pillar_detection'} "
                        f"does not exist yet - later stages will find no series.")

    train = v.get("MODE") == "train"
    continuing = train and v.get("RUN_TRAIN") and v.get("TRAIN_KIND") == "continue"
    if train:
        if crops is None:
            errors.append("Crops folder is empty.")
        if model is None and v.get("RUN_TRAIN"):
            errors.append("Model folder (output) is empty.")
        if not v.get("RUN_CROPS") and crops is not None and \
                not (crops / "manifest.csv").exists():
            warnings.append(f"Cropping is off and there is no manifest in {crops}.")
        if continuing:
            parent = p("PARENT_MODEL_DIR")
            if parent is None or not (parent / "config.json").exists():
                errors.append("Continue training needs a parent model folder containing "
                              "a trained model (config.json).")
            elif model is not None and _same_dir(parent, model):
                errors.append("Parent model and output model folders must differ - the "
                              "parent would be overwritten mid-training.")
            old = p("OLD_CROPS_DIR")
            if old is not None and not (old / "labels.csv").exists():
                errors.append(f"Old crops folder has no labels.csv: {old}")
        if (v.get("RUN_TRAIN") and not (continuing and v.get("CT_EVAL_ONLY"))
                and model is not None and (model / "config.json").exists()):
            warnings.append(f"A model already exists in {model} and will be overwritten.")

    scoring = v.get("MODE") == "score"
    score_needed = scoring or (train and v.get("SCORE_AFTER_TRAIN") and v.get("RUN_TRAIN"))
    if scoring and (model is None or not (model / "config.json").exists()):
        errors.append(f"Scoring needs a trained model in the Model folder ({model}).")
    if score_needed and v.get("AGG_RULE") == "maxmip" and not v.get("MIP_ENABLED"):
        warnings.append("Rule 'maxmip' with the MIP pass off gives the same counts as 'max'.")

    if v.get("RUN_NUCLEI"):
        try:
            exe = resolve_fiji(v.get("FIJI_PATH"))
            if fiji_has_stardist(exe) is False:
                warnings.append(f"No StarDist plugin found in the Fiji at {exe.parent}. "
                                f"Enable the StarDist and CSBDeep update sites "
                                f"(Help > Update > Manage update sites) or the count will "
                                f"stop with an error.")
        except PipelineError as e:
            errors.append(f"Nuclei count: {e}")
        try:
            prob = float(v.get("NUCLEI_PROB"))
            if not 0 <= prob <= 1:
                errors.append("Nuclei StarDist probability must be between 0 and 1.")
        except (TypeError, ValueError):
            errors.append("Nuclei StarDist probability must be a number.")
        if float(v.get("NUCLEI_MIN_DIAMETER_UM") or 0) < 0 or \
                float(v.get("NUCLEI_MIN_CENTRE_RATIO") or 0) < 0:
            errors.append("Nuclei minimum diameter and centre ratio must be zero or greater.")
    return errors, warnings


def build_plan() -> list[tuple[str, callable]]:
    """Ordered (name, function) stages from the current config."""
    stages = []
    if config.RUN_EXTRACT:
        stages.append(("LIF extraction", run_lif_extraction))
    if config.RUN_DETECT:
        stages.append(("Pillar detection", run_pillar_detection))
    if config.MODE == "train":
        if config.RUN_CROPS:
            stages.append(("Crops", run_crops))
        if config.RUN_LABEL:
            stages.append(("Labelling", run_label))
        if config.RUN_TRAIN:
            if config.TRAIN_KIND == "continue":
                stages.append(("Continue training", run_continue_train))
            else:
                stages.append(("Training", run_train))
            eval_only = config.TRAIN_KIND == "continue" and config.CT_EVAL_ONLY
            if config.SCORE_AFTER_TRAIN and not eval_only:
                stages.append(("Scoring", run_score))
    elif config.MODE == "score":
        stages.append(("Scoring", run_score))
    # last, so a Fiji problem never blocks the myelin stages
    if config.RUN_NUCLEI:
        stages.append(("Nuclei count", run_nuclei_count))
    return stages


def run_pipeline(values: dict) -> bool:
    apply_settings(values)
    stages = build_plan()
    if not stages:
        print("Nothing ticked to run.")
        return True

    print("=" * 70)
    print(f"Wrap-scoring pipeline  -  mode: {config.MODE.upper()}")
    print("  " + "  ->  ".join(name for name, _ in stages))
    print("=" * 70)
    t_all = time.time()
    for i, (name, fn) in enumerate(stages, 1):
        print(f"\n{'#' * 70}\n# [{i}/{len(stages)}] {name}\n{'#' * 70}")
        t0 = time.time()
        try:
            fn()
        except PipelineError as e:
            print(f"\n! {name} stopped: {e}")
            _report_skipped(stages[i:])
            return False
        except KeyboardInterrupt:
            print(f"\n! interrupted during {name}")
            _report_skipped(stages[i:])
            return False
        except Exception:
            traceback.print_exc()
            print(f"\n! {name} failed (traceback above)")
            _report_skipped(stages[i:])
            return False
        print(f"\n{name} finished in {_fmt_secs(time.time() - t0)}")
    print(f"\nAll stages finished in {_fmt_secs(time.time() - t_all)}.")
    return True


def _report_skipped(rest) -> None:
    if rest:
        print("  not run: " + ", ".join(name for name, _ in rest))


def _fmt_secs(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# =============================================================================
# SETTINGS WINDOW
# =============================================================================

# (tab, key, label, kind, help, choices)
#   kind: dir | int | float | bool | str | choice (fixed list) | combo (editable list)
FIELDS = [
    ("Paths", "ROOT_DIR", "Root folder", "dir",
     "LIF extraction writes here; detection, crops and scoring read from here", None),
    ("Paths", "RESULTS_DIR", "Results folder", "dir",
     "pillar_detection/, scoring CSVs and QC images", None),
    ("Paths", "CROPS_DIR", "Crops folder", "dir",
     "crops, manifest.csv and labels.csv (train mode)", None),
    ("Paths", "MODEL_DIR", "Model folder", "dir",
     "train: where the new model is saved   score: the model used", None),
    ("Paths", "PARENT_MODEL_DIR", "Parent model", "dir",
     "continue training only: the model to start from", None),
    ("Paths", "OLD_CROPS_DIR", "Old crops (replay)", "dir",
     "continue training only, optional: the parent's crop set", None),

    ("Pillar detection", "PILLAR_DIRNAME", "Pillar folder name", "str", "", None),
    ("Pillar detection", "MBP_DIRNAME", "MBP folder name", "str", "", None),
    ("Pillar detection", "PILLAR_MIN_DIAMETER_PX", "Min pillar diameter (px)", "int", "", None),
    ("Pillar detection", "PILLAR_MAX_DIAMETER_PX", "Max pillar diameter (px)", "int", "", None),
    ("Pillar detection", "PILLAR_MAX_ECCENTRICITY", "Max eccentricity", "float",
     "0 = circle, ~1 = line", None),
    ("Pillar detection", "PILLAR_BG_PERCENTILE", "Background percentile", "float", "", None),
    ("Pillar detection", "PILLAR_REF_PERCENTILE", "Bright reference percentile", "float", "", None),
    ("Pillar detection", "PILLAR_MIN_SCALE", "Min normalisation scale", "float", "", None),
    ("Pillar detection", "PILLAR_DEBRIS_SEED_PERCENTILE", "Debris seed percentile", "float", "", None),
    ("Pillar detection", "PILLAR_DEBRIS_MIN_AREA_FRAC", "Debris min area (frame frac)", "float",
     "1.0 disables debris exclusion", None),
    ("Pillar detection", "PILLAR_DEBRIS_DILATE_PX", "Debris dilation (px)", "int", "", None),
    ("Pillar detection", "PILLAR_SLICE_USABLE_FRAC", "Usable slice fraction", "float",
     "slice kept if its blob count >= this x the best slice's", None),
    ("Pillar detection", "PILLAR_SLICE_MIN_COUNT", "Usable slice min blobs", "int", "", None),
    ("Pillar detection", "PILLAR_VOTE_THRESH", "Inferred-site vote threshold", "float", "", None),
    ("Pillar detection", "PILLAR_MIN_VOTE_FRACTION", "Inferred-site min vote fraction", "float", "", None),
    ("Pillar detection", "OFF_LATTICE_FRAC", "Off-lattice residual (x pitch)", "float", "", None),
    ("Pillar detection", "PILLAR_EDGE_MAX_CUT_FRAC", "Edge sites: max fraction cut off", "float", "", None),
    ("Pillar detection", "GRID_EXTENT", "Grid extent", "choice",
     "detections = around what was found; image = whole frame", ["detections", "image"]),
    ("Pillar detection", "GRID_PAD_CELLS", "Grid padding (cells)", "int", "", None),
    ("Pillar detection", "GRID_EXTEND_TO_EDGE", "Extend grid to clipped edges", "bool", "", None),
    ("Pillar detection", "KEEP_OFF_LATTICE", "Keep off-lattice blobs as sites", "bool", "", None),
    ("Pillar detection", "LATTICE_REFINE_ITERS", "Lattice refine iterations", "int", "", None),
    ("Pillar detection", "LATTICE_MAX_PITCH_DRIFT", "Max pitch drift per step", "float", "", None),
    ("Pillar detection", "WRITE_PILLAR_QC_OVERLAY", "Write QC overlay", "bool", "", None),
    ("Pillar detection", "WRITE_PILLAR_FOCUS_STACK", "Write focus stack", "bool", "", None),

    ("Sites & crops", "INCLUDE_INFERRED_SITES", "Include inferred sites", "bool",
     "sites the detector missed, filled from the lattice", None),
    ("Sites & crops", "INCLUDE_OFF_LATTICE_SITES", "Include off-lattice sites", "bool", "", None),
    ("Sites & crops", "PRIMARY_DENOMINATOR", "Headline denominator", "choice",
     "both count sets are always written", ["all", "detected"]),
    ("Sites & crops", "CONTEXT_PITCHES", "Crop size (pitches)", "float", "", None),
    ("Sites & crops", "OUT_SIZE", "Model input size (px)", "int", "", None),
    ("Sites & crops", "PILLAR_R_FRAC", "Pillar radius (x pitch)", "float", "", None),
    ("Sites & crops", "SLIDE_AT_EDGES", "Slide window in at edges", "bool", "", None),
    ("Sites & crops", "Z_SMOOTH", "Z smoothing (+-N slices)", "int", "0 = raw", None),
    ("Sites & crops", "Z_RANGE", "Z range", "str", "blank = all, or lo:hi e.g. 5:45", None),
    ("Sites & crops", "PILLARS_PER_SERIES", "Sites sampled per series", "int",
     "0 = every site", None),
    ("Sites & crops", "SLICES_PER_PILLAR", "Slices per site", "int",
     "spread evenly; 0 = every slice", None),
    ("Sites & crops", "SAMPLE_SEED", "Sampling seed", "int", "", None),

    ("Training", "BACKBONE", "Backbone", "combo", "any Hugging Face image classifier", BACKBONES),
    ("Training", "EPOCHS", "Epochs", "int", "", None),
    ("Training", "BATCH_SIZE", "Batch size", "int", "", None),
    ("Training", "LEARNING_RATE", "Learning rate", "float", "", None),
    ("Training", "HOLDOUT_FRAC", "Hold-out fraction", "float", "fraction of groups held out", None),
    ("Training", "GROUP_KEY", "Split by", "choice", "", ["series", "site_id"]),
    ("Training", "CLASS_WEIGHTS", "Class-weighted loss", "bool", "", None),
    ("Training", "TRAIN_SEED", "Training seed", "int", "", None),
    ("Training", "CT_EPOCHS", "Continue: epochs", "int", "", None),
    ("Training", "CT_LEARNING_RATE", "Continue: learning rate", "float",
     "deliberately below a fresh fit", None),
    ("Training", "CT_FREEZE_LAYERS", "Continue: freeze first N blocks", "int", "", None),
    ("Training", "CT_EARLY_STOPPING", "Continue: early stopping", "bool", "", None),
    ("Training", "CT_EVAL_ONLY", "Continue: evaluate only", "bool",
     "score the parent on both crop sets and stop", None),
    ("Training", "CT_ALLOW_PREPROC_MISMATCH", "Continue: allow crop-geometry mismatch", "bool", "", None),

    ("Scoring", "AGG_RULE", "Aggregation rule", "choice",
     "maxmip = max(per-slice max, MIP class)", AGG_RULES),
    ("Scoring", "INFER_BATCH", "Inference batch size", "int", "", None),
    ("Scoring", "DEVICE", "Device", "choice", "", ["auto", "cuda", "cpu"]),
    ("Scoring", "WRITE_QC_OVERLAY", "Write QC overlays", "bool", "", None),
    ("Scoring", "MIP_ENABLED", "MIP pass", "bool", "", None),
    ("Scoring", "MIP_PROJ", "MIP projection", "choice", "", MIP_PROJ),
    ("Scoring", "MIP_TOPK", "MIP top-k", "int", "for projection = topk", None),
    ("Scoring", "MIP_DULL", "MIP dulling", "choice",
     "pull MIP intensity back to single-slice levels", MIP_DULL),
    ("Scoring", "MIP_GAMMA", "MIP gamma", "float", "for dulling = gamma (>1 darkens)", None),
    ("Scoring", "MIP_SCALE", "MIP scale", "float", "for dulling = scale", None),
    ("Scoring", "MIP_GLOBAL_NORM", "MIP shared normalisation", "bool",
     "keeps a 2nd copy of the stack in RAM", None),
    ("Scoring", "MIP_QC", "MIP QC images per series", "int", "0 = off", None),

    ("Nuclei", "FIJI_PATH", "Fiji", "file",
     "ImageJ-win64.exe, or the Fiji.app folder; needs StarDist + CSBDeep", None),
    ("Nuclei", "NUCLEI_DIRNAME", "Nuclei folder name", "str", "", None),
    ("Nuclei", "NUCLEI_REMAKE_MIP", "Remake existing MIPs", "bool",
     "off = reuse a MIP that is newer than its slices", None),
    ("Nuclei", "NUCLEI_PROB", "StarDist probability threshold", "float", "0-1", None),
    ("Nuclei", "NUCLEI_MIN_DIAMETER_UM", "Min equivalent diameter (um)", "float",
     "0 = no size filter", None),
    ("Nuclei", "NUCLEI_FILTER_HOLLOW", "Filter dark-centred objects", "bool", "", None),
    ("Nuclei", "NUCLEI_MIN_CENTRE_RATIO", "Min centre / object brightness", "float", "", None),
    ("Nuclei", "NUCLEI_PIXEL_UM", "Fallback pixel size (um)", "float",
     "only for series extracted before calibration.json existed; 0 = none", None),
]


class SettingsWindow:
    """Tabbed settings + run-plan window. On Run, self.result holds the values."""

    def __init__(self, values: dict, settings_path: Path):
        self.values = dict(values)
        self.settings_path = settings_path
        self.vars: dict[str, tuple] = {}
        self.result = None

        self.root = tk.Tk()
        self.root.title("Wrap-scoring pipeline")
        style = ttk.Style(self.root)
        style.configure("Help.TLabel", foreground="#777")
        style.configure("Head.TLabel", font=("TkDefaultFont", 10, "bold"))

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self._build_run_tab(nb)

        tabs: dict[str, list] = {}
        for tab, key, label, kind, help_, choices in FIELDS:
            if tab not in tabs:
                frame = ttk.Frame(nb, padding=10)
                # only the Paths tab stretches its entry column; elsewhere
                # the help text sits right next to the field
                frame.columnconfigure(1, weight=1 if tab == "Paths" else 0)
                frame.columnconfigure(3, weight=0 if tab == "Paths" else 1)
                nb.add(frame, text=tab)
                tabs[tab] = [frame, 0]
            frame, row = tabs[tab]
            self._add_field(frame, row, key, label, kind, help_, choices)
            tabs[tab][1] += 1
        paths_frame, row = tabs["Paths"]
        ttk.Button(paths_frame, text="Put Crops and Results inside Root",
                   command=self._paths_under_root).grid(row=row, column=1, sticky="w",
                                                        pady=(10, 0))

        bar = ttk.Frame(self.root, padding=(8, 4, 8, 8))
        bar.pack(fill="x")
        ttk.Button(bar, text="Reset to defaults", command=self._reset).pack(side="left")
        ttk.Button(bar, text="Save settings", command=self._save).pack(side="left", padx=6)
        ttk.Button(bar, text="Quit", command=self.root.destroy).pack(side="right")
        ttk.Button(bar, text="Run  \u25B6", command=self._run).pack(side="right", padx=6)
        self._refresh_state()
        _bring_to_front(self.root)

    # --- widgets --------------------------------------------------------------

    def _add_field(self, frame, row, key, label, kind, help_, choices):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        v = self.values.get(key, DEFAULTS[key])
        if kind == "bool":
            var = tk.BooleanVar(value=bool(v))
            w = ttk.Checkbutton(frame, variable=var)
            sticky = "w"
        elif kind in ("choice", "combo"):
            var = tk.StringVar(value=str(v))
            w = ttk.Combobox(frame, textvariable=var, values=choices,
                             state="readonly" if kind == "choice" else "normal",
                             width=40 if kind == "combo" else 14)
            sticky = "w"
        else:
            var = tk.StringVar(value="" if v is None else str(v))
            wide = kind in ("dir", "str", "file") and key not in (
                "Z_RANGE", "PILLAR_DIRNAME", "MBP_DIRNAME", "NUCLEI_DIRNAME")
            w = ttk.Entry(frame, textvariable=var, width=60 if wide else 14)
            sticky = "we" if wide else "w"
        w.grid(row=row, column=1, sticky=sticky, pady=2)
        if kind == "dir":
            ttk.Button(frame, text="\u2026", width=3,
                       command=lambda: self._browse_dir(var)).grid(row=row, column=2, padx=(4, 0))
        elif kind == "file":
            ttk.Button(frame, text="\u2026", width=3,
                       command=lambda: self._browse_file(var)).grid(row=row, column=2, padx=(4, 0))
        if help_:
            ttk.Label(frame, text=help_, style="Help.TLabel").grid(
                row=row, column=3, sticky="w", padx=(10, 0))
        self.vars[key] = (var, kind)

    def _var(self, key, kind, factory):
        var = factory(value=self.values.get(key, DEFAULTS[key]))
        self.vars[key] = (var, kind)
        return var

    def _build_run_tab(self, nb):
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text="Run")
        f.columnconfigure(0, weight=1)

        mode = self._var("MODE", "str", tk.StringVar)
        top = ttk.Frame(f)
        top.grid(row=0, column=0, sticky="w")
        ttk.Label(top, text="Mode:", style="Head.TLabel").pack(side="left", padx=(0, 8))
        ttk.Radiobutton(top, text="Train a model", variable=mode, value="train",
                        command=self._refresh_state).pack(side="left")
        ttk.Radiobutton(top, text="Score with a trained model", variable=mode, value="score",
                        command=self._refresh_state).pack(side="left", padx=10)
        ttk.Radiobutton(top, text="Neither (only the stages below)", variable=mode,
                        value="none", command=self._refresh_state).pack(side="left")

        # --- common stages -----------------------------------------------------
        common = ttk.LabelFrame(f, text="Every mode", padding=8)
        common.grid(row=1, column=0, sticky="we", pady=(10, 4))
        common.columnconfigure(0, weight=1)
        ext = self._var("RUN_EXTRACT", "bool", tk.BooleanVar)
        ttk.Checkbutton(common, text="A. Extract LIF files into the Root folder",
                        variable=ext, command=self._refresh_state).grid(row=0, column=0, sticky="w")
        self.lif_frame = ttk.Frame(common)
        self.lif_frame.grid(row=1, column=0, sticky="we", padx=(22, 0), pady=4)
        self.lif_frame.columnconfigure(0, weight=1)
        self.lif_list = tk.Listbox(self.lif_frame, height=4, selectmode="extended")
        self.lif_list.grid(row=0, column=0, rowspan=3, sticky="we")
        for path in self.values.get("LIF_FILES", []):
            self.lif_list.insert("end", path)
        ttk.Button(self.lif_frame, text="Add .lif\u2026", command=self._add_lifs).grid(
            row=0, column=1, sticky="we", padx=(6, 0))
        ttk.Button(self.lif_frame, text="Remove", command=self._remove_lifs).grid(
            row=1, column=1, sticky="we", padx=(6, 0))
        ttk.Button(self.lif_frame, text="Clear",
                   command=lambda: self.lif_list.delete(0, "end")).grid(
            row=2, column=1, sticky="we", padx=(6, 0))
        ttk.Checkbutton(common, text="0. Detect pillars  (writes Results/pillar_detection)",
                        variable=self._var("RUN_DETECT", "bool", tk.BooleanVar)).grid(
            row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(common, text="N. Count nuclei: MIP of every nuclei stack, then "
                                     "StarDist in Fiji  (runs last; settings on the Nuclei tab)",
                        variable=self._var("RUN_NUCLEI", "bool", tk.BooleanVar)).grid(
            row=3, column=0, sticky="w", pady=(4, 0))

        # --- train ---------------------------------------------------------------
        self.train_frame = ttk.LabelFrame(f, text="Train", padding=8)
        self.train_frame.grid(row=2, column=0, sticky="we", pady=4)
        ttk.Checkbutton(self.train_frame, text="1. Make crops for labelling",
                        variable=self._var("RUN_CROPS", "bool", tk.BooleanVar)).grid(
            row=0, column=0, sticky="w")
        opts = ttk.Frame(self.train_frame)
        opts.grid(row=1, column=0, sticky="w", padx=(22, 0))
        ttk.Checkbutton(opts, text="every site (sample still sets the order)",
                        variable=self._var("CROP_ALL_SITES", "bool", tk.BooleanVar)).pack(side="left")
        ttk.Checkbutton(opts, text="every z-slice",
                        variable=self._var("CROP_ALL_SLICES", "bool", tk.BooleanVar)).pack(
            side="left", padx=12)
        ttk.Checkbutton(self.train_frame, text="2. Label crops  (resumes at the first unlabelled crop)",
                        variable=self._var("RUN_LABEL", "bool", tk.BooleanVar)).grid(
            row=2, column=0, sticky="w", pady=(4, 0))
        trn = ttk.Frame(self.train_frame)
        trn.grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(trn, text="3. Train:",
                        variable=self._var("RUN_TRAIN", "bool", tk.BooleanVar)).pack(side="left")
        kind = self._var("TRAIN_KIND", "str", tk.StringVar)
        ttk.Radiobutton(trn, text="fresh from the backbone", variable=kind,
                        value="fresh").pack(side="left", padx=(6, 0))
        ttk.Radiobutton(trn, text="continue from the parent model (Paths tab)",
                        variable=kind, value="continue").pack(side="left", padx=10)
        ttk.Checkbutton(self.train_frame, text="4. Then score every series with the new model",
                        variable=self._var("SCORE_AFTER_TRAIN", "bool", tk.BooleanVar)).grid(
            row=4, column=0, sticky="w", pady=(4, 0))

        # --- score -----------------------------------------------------------------
        self.score_frame = ttk.LabelFrame(f, text="Score", padding=8)
        self.score_frame.grid(row=3, column=0, sticky="we", pady=4)
        ttk.Label(self.score_frame, text="4. Score every series under Root with the model in "
                                         "the Model folder.\n    Rule, MIP and device options "
                                         "are on the Scoring tab.").grid(row=0, column=0, sticky="w")

        ttk.Label(f, style="Help.TLabel",
                  text=f"Settings are saved to {self.settings_path.name} on Run. "
                       f"Progress prints in the console window.").grid(
            row=4, column=0, sticky="w", pady=(8, 0))

    # --- actions --------------------------------------------------------------

    def _browse_dir(self, var):
        start = var.get() if var.get() and Path(var.get()).exists() else None
        d = filedialog.askdirectory(parent=self.root, initialdir=start)
        if d:
            var.set(str(Path(d)))

    def _browse_file(self, var):
        cur = Path(var.get()) if var.get() else None
        start = str(cur.parent) if cur and cur.parent.exists() else None
        f = filedialog.askopenfilename(parent=self.root, initialdir=start,
                                       title="Select the Fiji launcher (ImageJ-win64.exe)")
        if f:
            var.set(str(Path(f)))

    def _add_lifs(self):
        files = filedialog.askopenfilenames(
            parent=self.root, title="Select LIF file(s)",
            filetypes=[("LIF files", "*.lif"), ("All files", "*.*")])
        have = set(self.lif_list.get(0, "end"))
        for f in files:
            if f not in have:
                self.lif_list.insert("end", str(Path(f)))
        if files:
            self.vars["RUN_EXTRACT"][0].set(True)
            self._refresh_state()

    def _remove_lifs(self):
        for i in reversed(self.lif_list.curselection()):
            self.lif_list.delete(i)

    def _paths_under_root(self):
        root = self.vars["ROOT_DIR"][0].get().strip()
        if not root:
            messagebox.showinfo("Paths", "Set the Root folder first.", parent=self.root)
            return
        self.vars["CROPS_DIR"][0].set(str(Path(root) / "Crops"))
        self.vars["RESULTS_DIR"][0].set(str(Path(root) / "Results"))

    def _refresh_state(self):
        mode = self.vars["MODE"][0].get()
        _set_enabled(self.train_frame, mode == "train")
        _set_enabled(self.score_frame, mode == "score")
        _set_enabled(self.lif_frame, bool(self.vars["RUN_EXTRACT"][0].get()))

    def collect(self) -> dict | None:
        out = dict(self.values)
        for key, (var, kind) in self.vars.items():
            raw = var.get()
            try:
                if kind == "int":
                    val = int(str(raw).strip())
                elif kind == "float":
                    val = float(str(raw).strip())
                elif kind == "bool":
                    val = bool(raw)
                else:
                    val = str(raw).strip()
            except ValueError:
                label = next((f[2] for f in FIELDS if f[1] == key), key)
                messagebox.showerror("Invalid value",
                                     f"'{label}' must be a {kind}, got '{raw}'.", parent=self.root)
                return None
            out[key] = val
        out["LIF_FILES"] = list(self.lif_list.get(0, "end"))
        return out

    def _save(self):
        values = self.collect()
        if values is not None:
            save_settings(values, self.settings_path)
            self.values = values
            messagebox.showinfo("Saved", f"Settings saved to\n{self.settings_path}",
                                parent=self.root)

    def _reset(self):
        if not messagebox.askyesno("Reset", "Reset every field to its default?",
                                   parent=self.root):
            return
        keep_lifs = list(self.lif_list.get(0, "end"))
        for key, (var, kind) in self.vars.items():
            v = DEFAULTS[key]
            var.set(bool(v) if kind == "bool" else ("" if v is None else str(v)))
        self.lif_list.delete(0, "end")
        for p in keep_lifs:
            self.lif_list.insert("end", p)
        self._refresh_state()

    def _run(self):
        values = self.collect()
        if values is None:
            return
        errors, warnings = validate_plan(values)
        if errors:
            messagebox.showerror("Cannot run", "\n\n".join(errors), parent=self.root)
            return
        if warnings and not messagebox.askyesno(
                "Check before running", "\n\n".join(warnings) + "\n\nRun anyway?",
                parent=self.root):
            return
        save_settings(values, self.settings_path)
        self.result = values
        self.root.destroy()


def _set_enabled(widget, enabled: bool) -> None:
    for child in widget.winfo_children():
        try:
            child.configure(state="normal" if enabled else "disabled")
        except tk.TclError:
            pass
        _set_enabled(child, enabled)
    # comboboxes that should stay read-only are not inside these frames


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Wrap-scoring pipeline (all stages).")
    p.add_argument("--headless", action="store_true",
                   help="no settings window: run with the saved settings")
    p.add_argument("--mode", choices=["train", "score", "none"],
                   help="override the saved mode (none = only extraction/detection/nuclei)")
    p.add_argument("--nuclei", action="store_true",
                   help="also run the nuclei count (stage N) this time")
    p.add_argument("--settings", type=Path, default=SETTINGS_PATH,
                   help="settings JSON (default: next to this script)")
    args = p.parse_args()

    values = load_settings(args.settings)
    if args.mode:
        values["MODE"] = args.mode
    if args.nuclei:
        values["RUN_NUCLEI"] = True

    if args.headless or tk is None:
        if tk is None and not args.headless:
            print("tkinter is not available - running headless with saved settings.")
        errors, warnings = validate_plan(values)
        for w in warnings:
            print(f"! {w}")
        if errors:
            for e in errors:
                print(f"ERROR: {e}")
            sys.exit(2)
        sys.exit(0 if run_pipeline(values) else 1)

    win = SettingsWindow(values, args.settings)
    win.root.mainloop()
    if win.result is None:
        return
    ok = run_pipeline(win.result)
    if sys.stdin is not None and sys.stdin.isatty():
        input(f"\n{'Done' if ok else 'Stopped'}. Press Enter to close...")


if __name__ == "__main__":
    main()