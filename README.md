# Myelin wrap scoring — pipeline

Per-pillar wrap classification (0 = none, 1 = <50%, 2 = mostly, 3 = full) across
a whole directory of series, with per-stack, per-series and directory-level counts.

wrap_common.py    shared: coord parsing, crop construction, augmentation
1 CROP.py     1. sample context crops for labelling
2 LABEL.py    2. hand-label them (tkinter GUI)
3 TRAIN.py     3. fine-tune ViT-base
4 SCORE.py   4. score everything, aggregate, count

## The crop

Each pillar becomes a 3-channel 224×224 image spanning 2.5 lattice pitches:

| channel | contents |
|---|---|
| R | MBP intensity, contrast-normalised over the whole field |
| G | synthetic lattice mask — filled discs at every pillar coordinate in view |
| B | soft marker on the one pillar being scored |

This is the answer to the context problem. The model sees the neighbourhood —
local background, how bright a ring looks in *this* field, whether a blob
respects the lattice or is debris lying across it — while B keeps the label
unambiguously attached to one pillar. Counts then come from tallying per-pillar
predictions, no internal counting required.

G is synthesised from your coordinate file rather than read from the pillar
channel, so it is noise-free and no extra files are needed.

On your data (137 px pitch, 2048 px field): the crop window is 342 px, the
centre pillar occupies 90 px of the 224, its body 36 px. If you find the model
struggling to resolve thin arcs, drop `--context-pitches` to 2.0 — that trades
the outer neighbours for ~25% more resolution on the target.

Contrast normalisation is per **field**, never per crop. Per-crop normalisation
would erase the brightness differences that separate a wrapped pillar from an
unwrapped one in the same image.

---

## 1. Extract crops

```bash
python "1 CROP" --root "G:/Data/Experiment" --out "G:/Training/crops" \
    --pillars-per-series 40 --slices-per-pillar 6 --z-smooth 1
```

`--z-smooth 1` averages z±1. This is a denoiser, not a projection — it raises
SNR without merging sheaths from different heights the way a max projection
would. Set `--z-smooth 0` for raw slices.

`--z-range 5:45` restricts to slices that actually intersect the pillar shafts.
Worth setting once you know where your stack's useful range is; slices above the
tops are pure noise and labelling them teaches the model nothing.

Slices are sampled spread through z rather than consecutively — adjacent slices
are near-duplicates, and six of them teach roughly what one does.

This writes `preproc.json` alongside the crops. It records the crop geometry,
gets copied into the model directory at training time, and is read back at
scoring time so inference can't silently drift from training.

## 2. Label

```bash
python "2 LABEL" --manifest "G:/Training/crops/manifest.csv" \
                      --labels   "G:/Training/crops/labels.csv"
```

`0`–`3` label and advance · `←`/`→` move without labelling · `r` toggle MBP-only
view · `u` undo · `q` quit. Saves after every keystroke and resumes where you
left off.

**Aim for 400–600 labelled crops spread across as many series as you can**, not
600 from one. Cross-batch variation is what the model has to survive.

**Before labelling in bulk**, have a second person blind-score 50 of the same
crops. Inter-rater agreement (quadratic-weighted kappa) is the ceiling the model
can be judged against, and on a 4-point wrap scale it is often lower than
people expect. If two people agree at κ = 0.7, a model at κ = 0.7 is done.

If a crop is unreadable rather than unwrapped, skip it with `→`. Labelling an
illegible crop `0` trains the model to guess.

## 3. Train

```bash
python "3 TRAIN".py --crops "G:/Training/crops" --out "./Modelv3" --epochs 8
```

- **Split by series, not by row.** Crops from the same pillar at neighbouring z
  are near-duplicates; a shuffled row-level split puts them on both sides of the
  fence and reports memorisation as accuracy. With one series only, it falls
  back to pillar-level grouping and says so.
- **Lazy loading.** The original held every decoded PIL image in RAM.
- **Class weights.** Wrap scores are heavily imbalanced. Unweighted
  cross-entropy predicts 0 everywhere and posts a fine accuracy.
- **Dihedral augmentation** (4 rotations × mirror) instead of
  `RandomRotation(180)`. Arbitrary angles tilt the lattice into orientations
  that never occur at test time, and interpolate the signal away. The 8
  dihedral ops are exact and preserve the square lattice.

Model selection uses quadratic-weighted kappa, not accuracy — accuracy scores a
0-vs-3 confusion the same as 2-vs-3. A confusion matrix is printed and saved to
`confusion_matrix.csv`. Read it before trusting any count.

## 4. Score and count

```bash
python "4 SCORE".py --root "G:/Data/Experiment" --model "./Modelv3" \
                       --out "G:/Results" --agg max
```

```
Results/
  per_slice/<series>.csv    pillar_idx, slice, pred, p0..p3   (full probabilities)
  per_pillar/<series>.csv   one row per pillar, every aggregation rule side by side
  per_stack_counts.csv      class counts for each individual slice
  per_series_counts.csv     class counts after aggregating over z
  directory_summary.csv     one row per series + TOTAL
```

`per_slice` holds the full probability output, so you can re-aggregate under a
different rule later without re-running the model.

---

## About the max rule

"Highest class assigned to that pillar across all slices" is the default, and
it's the aggregation most sensitive to classifier noise. The bias is
one-directional — max can only push counts up:

| slices in stack | truly-unwrapped pillars getting a spurious 3, at 2% per-slice FPR |
|---|---|
| 10 | 24% |
| 20 | 29% |
| 40 | 56% |
| 60 | 64% |

Every `per_pillar` CSV therefore reports five rules side by side:

| rule | behaviour |
|---|---|
| `max` | highest class on any slice |
| `consec2` | highest class held for ≥2 consecutive slices |
| `consec3` | ≥3 consecutive slices |
| `meanprob` | argmax of probability averaged over z |
| `q90` | 90th percentile of per-slice classes |

`consec2` keeps the spirit of "take the best view of the pillar" while requiring
the evidence to survive more than one slice — a real sheath persists over
several microns of z, a misfire usually doesn't. On a single-slice misfire, max
returns 3 and consec2 returns 0.

The script warns when max and consec2 disagree on more than 10% of pillars.
That gap is noise, not biology. Whichever rule you pick, apply the same one to
every condition you intend to compare.

## Missing lattice sites

Your example series detects 190 pillars in a field implying 225 lattice sites —
35 pillars with no centre, and therefore silently absent from every denominator.
Both `make_crops` and `score_series` report this per series, and
`directory_summary.csv` carries `n_pillars_detected` next to
`n_lattice_sites_expected`.

If the shortfall varies between conditions, your percentages are comparing
different populations. Fitting the lattice (two basis vectors + offset) to the
detected centres and snapping to it recovers the missing sites — worth doing
upstream in whatever produced the coordinate file.
