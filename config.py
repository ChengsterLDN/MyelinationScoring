"""
CONFIGURATION - very path and parameter for the wrap-scoring pipeline.

"""

from pathlib import Path


# Paths  

# Directory holding the series subfolders (each with pillar/ + mbp/)
#ROOT_DIR = Path(r"G:\Training Datasets\Test")
ROOT_DIR = Path(r"G:\Training Datasets\Pup")

# Where make_crops writes crops + manifest, and label_crops writes labels.csv
#CROPS_DIR = Path(r"G:\Training Datasets\Test\Crops")
CROPS_DIR = Path(r"G:\Training Datasets\Pup\Crops")

# Where train_wrap saves the trained model (and score_series loads it from)
MODEL_DIR = Path(r"G:\Training Datasets\Modelv4D")

# Where score_series writes counts, per-pillar CSVs and QC overlays
#RESULTS_DIR = Path(r"G:\Training Datasets\Test\Results")
RESULTS_DIR = Path(r"G:\Training Datasets\Pup\Results")

# Where detect_pillars writes <stack>_mask.png + <stack>_coords.json,
# one subfolder per series
PILLAR_DETECT_DIR = RESULTS_DIR / "pillar_detection"


# PIL_DET


# Folder name (case-insensitive) holding the pillar-channel z-stacks.
# Files in it are grouped into stacks by stripping the "_z<n>" token.
PILLAR_DIRNAME = "pillar"

# Folder name holding the MBP (wrap-marker) channel. Paired to pillar
# stacks by position key (see wrap_common.CHANNEL_TOKEN_RE).
MBP_DIRNAME = "mbp"

# Minimum blob diameter (px) for a bright region to count as a pillar.
# Calibrate against real Otsu-segmented blobs at your pixel size.
PILLAR_MIN_DIAMETER_PX = 35

# Maximum blob diameter (px): rejects merged blobs / edge artefacts that
# are still round enough to pass the eccentricity test.
PILLAR_MAX_DIAMETER_PX = 50

# Reject blobs more elongated than this (0 = circle, ~1 = line)
PILLAR_MAX_ECCENTRICITY = 0.5

# Percentile-range normalisation (median/MAD collapses to 0 on these images)
PILLAR_BG_PERCENTILE  = 50.0   # background estimate
PILLAR_REF_PERCENTILE = 99.9   # "bright" reference level
PILLAR_MIN_SCALE      = 1.0    # floor on (ref - bg), raw intensity units

# Debris exclusion: large bright regions (cells, dust) are masked out of
# thresholding and normalisation. Set MIN_AREA_FRAC to 1.0 to disable.
PILLAR_DEBRIS_SEED_PERCENTILE = 99.0     # brightest-pixel seed for candidate debris
PILLAR_DEBRIS_MIN_AREA_FRAC   = 0.00005  # fraction of frame area -> counts as debris
PILLAR_DEBRIS_DILATE_PX       = 15       # grow flagged regions to cover their halo

# Per-slice usability: a slice is used only if its own count of valid
# pillar blobs is >= USABLE_FRAC x the best slice's count (and >= MIN_COUNT).
# Unusable slices are dropped from everything, not down-weighted.
PILLAR_SLICE_USABLE_FRAC = 0.3
PILLAR_SLICE_MIN_COUNT   = 5

# Local-evidence check for inferred (undetected) lattice sites: a slice
# "votes" if the patch mean exceeds VOTE_THRESH; the site has evidence if
# at least MIN_VOTE_FRACTION of usable slices vote.
PILLAR_VOTE_THRESH       = 0.3
PILLAR_MIN_VOTE_FRACTION = 0.25

# Save <stack>_qc.png: usable-slice max projection with every site circled
WRITE_PILLAR_QC_OVERLAY = True

# Save <stack>_focus_stack.png: same background, each pillar repainted
# from its own best-focus slice
WRITE_PILLAR_FOCUS_STACK = True

# PIL_UTIL
# File extensions treated as images when flattening (case-insensitive)
PILLAR_FLATTEN_EXTENSIONS = (".png",)

# Subfolder names left completely alone by the flattener
PILLAR_FLATTEN_EXCLUDE_DIRNAMES = ("")


# Site registry  
# Site types:
#   detected      on the lattice, present in the coords file
#   off_lattice   in the coords file but > OFF_LATTICE_FRAC pitch from a site
#   inferred      a lattice site with no detection, predicted from the fit

INCLUDE_INFERRED_SITES = True   # score sites the detector missed
OFF_LATTICE_FRAC       = 0.35   # residual (x pitch) above which a detection
                                # is flagged off_lattice
EDGE_MARGIN_FRAC       = 0.5    # inferred sites must sit at least this many
                                # pitches inside the image

# Keep off_lattice sites in scoring (they never enter the "detected"
# denominator either way). False drops them from every per-site output.
INCLUDE_OFF_LATTICE_SITES = True

# Which denominator drives the headline pct_* columns:
#   "detected"  detected sites only (conservative)
#   "all"       plus inferred sites (complete lattice)
# Both count sets are always written.
PRIMARY_DENOMINATOR = "all"


# Crop geometry


CONTEXT_PITCHES = 2.5    # crop side length in units of lattice pitch
OUT_SIZE        = 224    # model input resolution (px)
PILLAR_R_FRAC   = 0.15   # pillar radius as a fraction of pitch
SLIDE_AT_EDGES  = True   # at the image border, slide the window inward
                         # instead of padding (target then sits off-centre)


# Z handling

Z_SMOOTH = 1             # average +-N neighbouring slices (0 = raw)
Z_RANGE  = None          # e.g. (5, 45) to restrict slices; None = all


# Crop sampling for labelling

PILLARS_PER_SERIES = 40  # sites sampled per series/stack
SLICES_PER_PILLAR  = 6   # z-slices per sampled site, spread evenly
SAMPLE_SEED        = 0   # RNG seed for reproducible sampling

# TRAINING

# Pretrained backbone to fine-tune
BACKBONE      = "google/vit-base-patch16-224-in21k"
#BACKBONE      = "facebook/convnext-tiny-224"
#BACKBONE      = "microsoft/resnet-18"
#BACKBONE      = "facebook/dinov2-small"

NUM_LABELS    = 4            # wrap classes 0-3
EPOCHS        = 8
BATCH_SIZE    = 16
LEARNING_RATE = 3e-5
HOLDOUT_FRAC  = 0.2          # fraction of groups held out for evaluation
GROUP_KEY     = "series"     # "series" or "pillar_idx" -- unit of the split
CLASS_WEIGHTS = True         # weight the loss by inverse class frequency
TRAIN_SEED    = 42

# SCORING

AGG_RULE        = "max"       # max | consec2 | consec3 | meanprob | q90
INFER_BATCH     = 32          # crops per inference batch
WRITE_QC_OVERLAY = True       # per-series PNG with predicted class per pillar

# LABEL

# Class index -> short name used in CSV columns and logs
CLASS_NAMES = {0: "none", 1: "partial_lt50", 2: "mostly", 3: "full"}
