# Myelination pipeline

`Myelination.py` is the whole myelin wrap-scoring pipeline in one file. It takes Leica `.lif` confocal files and:

- splits them into per-channel PNG z-slices
- finds the micropillars and fits the pillar lattice
- cuts one image crop per pillar so you can label how far the MBP wraps it
- trains (or keeps training) an image classifier on those labels
- scores every pillar in every series with the trained model
- counts nuclei with StarDist in Fiji

Everything runs from one settings window, or from the command line with saved settings.

---

## Contents

1. [Requirements](#1-requirements)
2. [Starting the program](#2-starting-the-program)
3. [How the pipeline fits together](#3-how-the-pipeline-fits-together)
4. [Folder layout and file naming](#4-folder-layout-and-file-naming)
5. [The settings window](#5-the-settings-window)
   - [Bottom bar buttons](#51-bottom-bar-buttons)
   - [Run tab](#52-run-tab)
   - [Paths tab](#53-paths-tab)
   - [Pillar detection tab](#54-pillar-detection-tab)
   - [Sites & crops tab](#55-sites--crops-tab)
   - [Training tab](#56-training-tab)
   - [Scoring tab](#57-scoring-tab)
   - [Nuclei tab](#58-nuclei-tab)
6. [Stage by stage](#6-stage-by-stage)
   - [A. LIF extraction](#a-lif-extraction)
   - [0. Pillar detection](#0-pillar-detection)
   - [1. Crops](#1-crops)
   - [2. Labelling window](#2-labelling-window)
   - [3. Training (fresh)](#3-training-fresh)
   - [3B. Continue training](#3b-continue-training)
   - [4. Scoring](#4-scoring)
   - [N. Nuclei count](#n-nuclei-count)
7. [Checks before a run](#7-checks-before-a-run)
8. [Command line](#8-command-line)
9. [Output reference](#9-output-reference)
10. [Typical workflows](#10-typical-workflows)
11. [Troubleshooting](#11-troubleshooting)
12. [Running it as an application](#12-running-it-as-an-application)

---

## 1. Requirements

| What | Needed for | Install |
|---|---|---|
| Python 3.9 or newer | everything | – |
| numpy, Pillow, scipy, scikit-image | everything | `pip install numpy pillow scipy scikit-image` |
| tkinter | the settings window, the LIF channel dialog, the labelling window | comes with the standard Windows/macOS Python installers |
| readlif | A. LIF extraction | `pip install readlif` |
| torch, transformers, accelerate | 3 / 3B training, 4 scoring | `pip install torch transformers accelerate` (install the CUDA build of torch to use a GPU) |
| Fiji, with the **StarDist** and **CSBDeep** update sites enabled | N. nuclei count | In Fiji: *Help › Update… › Manage update sites*, tick both, apply, restart Fiji |

The heavy packages are imported only by the stages that use them. So you can, for example, run pillar detection without torch installed.

---

## 2. Starting the program

```
python Myelination.py
```

This opens the **settings window**. Choose what to run, then press **Run**. The window closes and progress prints in the console. When the run ends, the console waits for Enter before closing.

Settings are kept in **`wrap_pipeline_settings.json`**, next to the script (or next to `Myelination.exe` in the [packaged app](#12-running-it-as-an-application)). They are saved whenever you press **Run** or **Save settings**, and loaded the next time you start. Settings missing from that file take their default values. Unknown entries are ignored.

To run without the window, see [Command line](#8-command-line).

---

## 3. How the pipeline fits together

```
 .lif files
     │
     ▼
 A  LIF extraction ──► Root/<lif>/<series>/{nuclei, mbp, pillar}/*.png  (+ calibration.json)
     │
     ▼
 0  Pillar detection ─► Results/pillar_detection/<lif>/<series>/<stack>_coords.json
     │
     ├── TRAIN mode ──► 1 Crops ─► 2 Label ─► 3 Train  or  3B Continue training ─► (4 Score)
     │
     ├── SCORE mode ──► 4 Score ─► Results/per_series_counts.csv, directory_summary.csv, …
     │
     └── NEITHER   ──► (no myelin stages)

 N  Nuclei count (optional, any mode, always last) ─► Results/nuclei/nuclei_summary.csv
```

- Stages always run in the order above. Ticked stages run; unticked ones are skipped.
- Because each stage reads the previous stage's files from disk, a run can **resume part-way**. For example, untick extraction and detection once they have been done, or untick cropping to go straight back into labelling.
- If a stage fails, the run stops there. The console prints the reason and lists the stages that did not run.
- The nuclei count always runs last, so a Fiji problem never blocks the myelin results.

---

## 4. Folder layout and file naming

### What the program expects under the Root folder

LIF extraction produces this layout. If you supply PNGs yourself, follow the same layout:

```
Root/
  <lif name>/
    <series name>/
      calibration.json          (pixel size, written by LIF extraction)
      nuclei/  <series>_c0_cyan_z0_t0.png, …_z1_t0.png, …
      mbp/     <series>_c1_cyan_z1_t0.png, …
      pillar/  <series>_c2_yellow_z0_t0.png, …
```

- **Folders are found by name, at any depth** under Root. Every folder named `pillar` (or your *Pillar folder name*) is used for detection. Every `mbp` folder is used for crops and scoring. Every `nuclei` folder is used for the nuclei count. Name matching ignores case.
- **The series name** is the path of the folder that holds `pillar/`, `mbp/` and `nuclei/`, relative to Root. For example: `MyExperiment/Region 3`.
- **Each slice filename must contain a z token**: `_z<number>` followed by `_` or `.`. Files without one are ignored, and the console reports how many.
- **Removing the z token gives the stack id.** For example, `Region 3_c2_yellow_z7_t0.png` belongs to stack `Region 3_c2_yellow_t0`. One folder can hold several stacks, such as several timepoints.
- **Pillar and MBP stacks are paired** by deleting the channel token `_c<N>_<colour>` from both stack ids. For example, `…_c2_yellow_t0` and `…_c1_cyan_t0` both become `…_t0`, so they pair. A pillar stack with no matching MBP stack is skipped, with a message.

---

## 5. The settings window

The window has a **Run** tab, six settings tabs, and a bar of buttons along the bottom.

### Kinds of field

| Field | How it behaves |
|---|---|
| Text box with **…** button | A folder (or, on the Nuclei tab, a file). Type the path, or press **…** to browse. |
| Tick box | On/off. |
| Drop-down (fixed) | Pick one of the listed values. |
| Drop-down (editable) | Pick from the list or type your own value (only *Backbone*). |
| Short text box | A number or short text. A number field holding something else shows an **Invalid value** message when you press Run or Save, naming the field. |

The grey text to the right of a field is a short hint.

### 5.1 Bottom bar buttons

| Button | What it does |
|---|---|
| **Reset to defaults** | Asks for confirmation, then puts every field on every tab, including the Run-tab ticks and the mode, back to its default. The LIF file list is kept. Nothing is saved until you press Save settings or Run. |
| **Save settings** | Checks every field is a valid value, writes them to `wrap_pipeline_settings.json`, and confirms with a message. Does not run anything. |
| **Run ▶** | Checks every field, then runs the [pre-run checks](#7-checks-before-a-run). **Errors** are listed and stop the run. **Warnings** are listed with *Run anyway?*. If you go ahead, the settings are saved, the window closes, and the pipeline runs in the console. |
| **Quit** | Closes the window without saving or running. |

### 5.2 Run tab

This tab decides **what** runs. Each numbered tick box is one stage.

#### Mode (radio buttons at the top)

| Option | Effect |
|---|---|
| **Train a model** | Enables the *Train* box below. The run does crops, labelling and training, whichever are ticked. |
| **Score with a trained model** | Enables the *Score* box. The run scores every series with the model in the Model folder. |
| **Neither (only the stages below)** | No myelin stages. Only A, 0 and N run, whichever are ticked. Use it for extraction-only, detection-only or nuclei-only runs. |

The box that doesn't match the mode is greyed out, and its ticks are ignored.

#### "Every mode" box

| Control | What it does |
|---|---|
| **A. Extract LIF files into the Root folder** | Runs [LIF extraction](#a-lif-extraction) on the files in the list below it. The list is greyed out when this is unticked. |
| LIF file list | The `.lif` files to extract. Click to select; Ctrl- or Shift-click to select several. |
| **Add .lif…** | Opens a file browser. The chosen files are added to the list, skipping any already in it, and box A is ticked automatically. |
| **Remove** | Removes the selected files from the list. |
| **Clear** | Empties the list. |
| **0. Detect pillars (writes Results/pillar_detection)** | Runs [pillar detection](#0-pillar-detection) on every pillar folder under Root. |
| **N. Count nuclei: MIP of every nuclei stack, then StarDist in Fiji** | Runs the [nuclei count](#n-nuclei-count) as the last stage. Its settings are on the Nuclei tab. |

#### "Train" box (Train mode only)

| Control | What it does |
|---|---|
| **1. Make crops for labelling** | Runs [Crops](#1-crops): writes labelling crops plus `manifest.csv` into the Crops folder. |
| ↳ **every site (sample still sets the order)** | Crop every pillar site, not just the sample of *Sites sampled per series*. The sample is still listed first in the manifest, so it is labelled first. |
| ↳ **every z-slice** | Crop every z-slice in the Z range, not *Slices per site* evenly spread slices. |
| **2. Label crops (resumes at the first unlabelled crop)** | Opens the [labelling window](#2-labelling-window). Training (if ticked) starts after you close it. |
| **3. Train:** | Runs training, of the kind chosen with the two radio buttons beside it: |
| ↳ **fresh from the backbone** | [Training (fresh)](#3-training-fresh): starts from the pretrained *Backbone* (Training tab) and saves to the Model folder. |
| ↳ **continue from the parent model (Paths tab)** | [Continue training](#3b-continue-training): starts from the *Parent model* and saves to the Model folder. |
| **4. Then score every series with the new model** | After training, runs [Scoring](#4-scoring) with the model just saved. Skipped when *Continue: evaluate only* is ticked, since no new model is saved then. |

#### "Score" box (Score mode only)

There are no controls in this box; it just confirms that stage 4 will score every series under Root with the model in the Model folder. The scoring options are on the Scoring tab.

The grey line at the bottom of the tab gives the name of the settings file and reminds you that progress shows in the console.

### 5.3 Paths tab

| Field | Default | What it is |
|---|---|---|
| **Root folder** | `G:\Training Datasets\Pup` | Where LIF extraction writes. Detection, crops, scoring and the nuclei count read from here. |
| **Results folder** | `G:\Training Datasets\Pup\Results` | Pillar-detection output, scoring CSVs and QC images, and nuclei results. |
| **Crops folder** | `G:\Training Datasets\Pup\Crops` | Train mode: crops, `manifest.csv`, `labels.csv`, `preproc.json`. For continue training this is the **new** crop set. |
| **Model folder** | `G:\Training Datasets\Modelv4D` | Train mode: where the new model is **saved** (anything there is overwritten). Score mode: the model that is **used**. |
| **Parent model** | empty | Continue training only: the trained model to start from. It must differ from the Model folder. |
| **Old crops (replay)** | empty | Continue training only, optional: the crop set (with its `labels.csv`) that the parent model was trained on. It is mixed back in so the model doesn't forget it. |
| **Put Crops and Results inside Root** (button) | – | Sets Crops to `<Root>\Crops` and Results to `<Root>\Results`. Asks for Root first if it is empty. |

### 5.4 Pillar detection tab

These settings control [stage 0](#0-pillar-detection). Distances are in pixels of the pillar images.

| Field | Default | What it does |
|---|---|---|
| **Pillar folder name** | `pillar` | The folder name that holds pillar-channel slices. |
| **MBP folder name** | `mbp` | The folder name that holds MBP-channel slices (read by crops and scoring). |
| **Min pillar diameter (px)** | 35 | Smallest blob counted as a pillar (diameter of a circle of equal area). Also the radius used to merge detections of the same pillar across slices. |
| **Max pillar diameter (px)** | 50 | Largest blob counted as a pillar. |
| **Max eccentricity** | 0.5 | How elongated a blob may be: 0 is a perfect circle, values near 1 are line-like. Blobs above this are rejected. |
| **Background percentile** | 50 | In the normalised stack, this percentile of each slice becomes 0. |
| **Bright reference percentile** | 99.9 | This percentile becomes 1. |
| **Min normalisation scale** | 1.0 | Floor on the reference-minus-background gap, so near-empty slices aren't blown up. The normalised stack is used for the inferred-site evidence vote, the QC overlay and the focus stack. Blob finding uses Otsu thresholding on each slice instead. |
| **Debris seed percentile** | 99.0 | Pixels above this percentile seed a search for very bright debris (cells, dust). |
| **Debris min area (frame frac)** | 0.00005 | A connected bright region at least this fraction of the frame is treated as debris and excluded. 1.0 turns debris exclusion off. |
| **Debris dilation (px)** | 15 | Grows each debris region by this much to cover its halo. |
| **Usable slice fraction** | 0.3 | A slice is used only if it finds at least this fraction × the pillar count of the best slice… |
| **Usable slice min blobs** | 5 | …and at least this many pillars. Noisy slices (top/bottom of the stack) fall out this way. If no slice passes, all are used. |
| **Inferred-site vote threshold** | 0.3 | For each grid site the detector missed: a slice "votes" for it if its mean normalised intensity around the site is above this. |
| **Inferred-site min vote fraction** | 0.25 | An inferred site has local evidence if at least this fraction of usable slices vote for it. This sets only the `has_local_evidence` flag in `coords.json` (and a smaller circle in the QC overlay); inferred sites are kept either way. |
| **Off-lattice residual (x pitch)** | 0.35 | A detection further than this × the pitch from its nearest lattice node is "off-lattice". The same distance sets the lattice-fit inliers and the radius of the evidence vote. |
| **Edge sites: max fraction cut off** | 0.5 | A grid site whose pillar disc is cut off by the image edge by more than this fraction is dropped. Sites cut less than this are kept and marked `clipped`. |
| **Grid extent** | `detections` | `detections`: fill the grid only over the rows and columns where pillars were found. `image`: fill the whole frame. |
| **Grid padding (cells)** | 0 | Extra rows and columns added around the detections (*detections* mode). |
| **Extend grid to clipped edges** | on | In *detections* mode, keep adding edge rows and columns while their pillars would be partly cut off by the frame. This reaches the image edge without adding sites in empty areas. |
| **Keep off-lattice blobs as sites** | off | Keep off-lattice detections as extra sites (type `off_lattice`) rather than discarding them. |
| **Lattice refine iterations** | 3 | Rounds of least-squares re-fitting of the lattice pitch and offset, on inliers only. |
| **Max pitch drift per step** | 0.25 | A refinement round that changes the pitch by more than this fraction is rejected, and refinement stops. |
| **Write QC overlay** | on | Write `<stack>_qc.png` (see [outputs](#0-pillar-detection)). |
| **Write focus stack** | on | Write `<stack>_focus_stack.png`. |

*Pillar radius (x pitch)* on the Sites & crops tab also affects detection, because it sets the disc size used for edge clipping and the grid mask.

### 5.5 Sites & crops tab

These settings control which sites are used and how each crop is cut. Crops and scoring both use them.

| Field | Default | What it does |
|---|---|---|
| **Include inferred sites** | on | Use grid sites the detector missed but the lattice filled in. |
| **Include off-lattice sites** | on | Use off-lattice sites. These exist only if *Keep off-lattice blobs as sites* was on during detection. |
| **Headline denominator** | `all` | Which sites the headline scoring counts and percentages cover: `all` sites, or only `detected` ones. Both count sets are always written. |
| **Crop size (pitches)** | 2.5 | Side of the square crop window, in multiples of the lattice pitch. |
| **Model input size (px)** | 224 | Crops are resized to this square size. It must suit the backbone. |
| **Pillar radius (x pitch)** | 0.15 | Radius of the pillar discs drawn into each crop (and used in detection), × pitch. |
| **Slide window in at edges** | on | Near the image border the crop window slides inward so it stays full-sized, rather than being cut short. |
| **Z smoothing (+-N slices)** | 1 | Each MBP slice is averaged with up to N slices either side before cropping or scoring. 0 = raw slices. |
| **Z range** | blank | Which slices to use, by the z number in the filename, inclusive: blank = all, or `lo:hi`, e.g. `5:45`. Note that MBP stacks lose their first and last slice at extraction. |
| **Sites sampled per series** | 40 | Crops only: how many sites to crop per series. The sample is stratified by site type (detected / inferred / off-lattice). 0 = every site. |
| **Slices per site** | 6 | Crops only: how many slices per site, spread evenly across the Z range so the ambiguous ends are included. 0 = every slice. |
| **Sampling seed** | 0 | Random seed for the site sample, so the same settings give the same crops. |

**Crop geometry travels with the model.** *Crop size*, *Model input size*, *Pillar radius*, *Slide window in at edges* and *Z smoothing* are saved in `preproc.json` with the crops, and copied into the model folder at training. **Scoring uses the model's values, not this tab's**, so a model is always scored with the geometry it was trained on. This tab's values are used only if the model has no `preproc.json`, and the console warns when that happens.

### 5.6 Training tab

| Field | Default | What it does |
|---|---|---|
| **Backbone** | `google/vit-base-patch16-224-in21k` | The pretrained image model fresh training starts from. The list offers ViT-base, ConvNeXt-tiny, ResNet-18 and DINOv2-small; you can type any Hugging Face image-classification model id. Downloaded on first use. |
| **Epochs** | 8 | Passes over the training crops (fresh training). |
| **Batch size** | 16 | Crops per training step (both kinds of training). Lower it if you run out of GPU memory. |
| **Learning rate** | 3e-5 | Fresh training. |
| **Hold-out fraction** | 0.2 | Fraction of groups held out for evaluation. |
| **Split by** | `series` | What a group is. `series`: whole series go to train or eval, which tests on unseen fields of view (recommended). `site_id`: split by pillar. If only one series is labelled, the split falls back to `site_id`, with a warning that eval scores will look better than they are. |
| **Class-weighted loss** | on | Weight the loss by inverse class frequency, so rare classes aren't ignored. |
| **Training seed** | 42 | Seed for the split, augmentation and training. |
| **Continue: epochs** | 3 | Epochs for continue training. |
| **Continue: learning rate** | 1e-5 | Learning rate for continue training, deliberately lower than for a fresh fit. |
| **Continue: freeze first N blocks** | 0 | Freeze the embeddings and the first N transformer blocks, so only later blocks change. It reduces forgetting. Works on ViT-style backbones; others print a note and skip it. |
| **Continue: early stopping** | on | Stop when the eval score hasn't improved for 2 epochs. |
| **Continue: evaluate only** | off | Score the parent model on the new (and old) crop sets and stop, without training or saving anything. Use it to see whether retraining is needed at all. |
| **Continue: allow crop-geometry mismatch** | off | Continue even if a crop set was cut with different geometry from the parent model. Normally this stops the run. |

### 5.7 Scoring tab

| Field | Default | What it does |
|---|---|---|
| **Aggregation rule** | `maxmip` | How a pillar's per-slice scores become one class (see the table below). It is used for the headline counts and the QC overlay. Every rule is still written to `per_pillar/*.csv`, and the other rules' counts go in `per_series_counts.csv`. |
| **Inference batch size** | 32 | Crops scored at once. Lower it if you run out of memory. |
| **Device** | `auto` | `auto` uses the GPU (CUDA) if available, otherwise the CPU. `cuda` or `cpu` forces it. Applies to training too. |
| **Write QC overlays** | on | Write `qc/<series>.png`: the image with a coloured ring per pillar. |
| **MIP pass** | on | After per-slice scoring, score one projected crop per pillar (see [MIP pass](#the-mip-pass)). |
| **MIP projection** | `max` | `max`: brightest value at each pixel across the slices. `topk`: mean of the *k* brightest values, which is less sensitive to a single bright slice. |
| **MIP top-k** | 2 | The *k* for `topk`. |
| **MIP dulling** | `match` | How the projection's brightness is brought back to single-slice levels, since the model was trained on single slices. `match`: histogram-matched to the average single slice. `gamma`: darkened with *MIP gamma*. `scale`: multiplied by *MIP scale*. `none`: left as is. |
| **MIP gamma** | 1.6 | Used by `gamma` dulling; above 1 darkens. |
| **MIP scale** | 0.7 | Used by `scale` dulling. |
| **MIP shared normalisation** | off | Normalise all slices of a series against shared percentiles before projecting, instead of each slice separately. Keeps a second copy of the stack in memory. |
| **MIP QC images per series** | 12 | Save up to this many side-by-side images per series where the MIP raised a pillar's class: the best single slice on the left, the projection on the right. 0 = none. |

**Aggregation rules**

| Rule | A pillar's class is… |
|---|---|
| `max` | the highest class predicted on any single slice. |
| `consec2` | the highest class held (or exceeded) on at least 2 consecutive slices. It ignores one-slice misfires. |
| `consec3` | the same, over 3 consecutive slices. |
| `meanprob` | the class with the highest probability averaged over all slices. |
| `q90` | the 90th percentile of the per-slice classes, rounded down. |
| `maxmip` | the higher of `max` and the MIP pass's class. With the MIP pass off this equals `max`. |

### 5.8 Nuclei tab

| Field | Default | What it does |
|---|---|---|
| **Fiji** | empty | The Fiji launcher (`ImageJ-win64.exe` on Windows), or the `Fiji.app` folder; the launcher is found inside it. **…** opens a file browser. |
| **Nuclei folder name** | `nuclei` | The folder name that holds nuclei-channel slices. |
| **Remake existing MIPs** | off | Off: a MIP already on disk is reused if it is newer than all its slices. On: every MIP is rebuilt. |
| **StarDist probability threshold** | 0.7 | Detection confidence StarDist needs (0–1). Lower finds more, including more false objects. |
| **Min equivalent diameter (um)** | 5.0 | Detections smaller than this (diameter of a circle of equal area) are rejected. 0 = no size filter. |
| **Filter dark-centred objects** | on | Reject detections whose centre is dimmer than their average. These are usually hollow rings rather than nuclei. |
| **Min centre / object brightness** | 0.9 | For that filter: the mean of the central ellipse (30% of the object's bounding box) ÷ the object's mean must be at least this. |
| **Fallback pixel size (um)** | 0 | Pixel size for series that have no `calibration.json`, e.g. extracted before it was added. 0 = none: such series are skipped when a size filter is set. |

---

## 6. Stage by stage

### A. LIF extraction

**Reads:** the `.lif` files on the Run tab. **Writes:** `Root/<lif name>/<series>/<role>/*.png` and `Root/<lif name>/<series>/calibration.json`.

1. Each LIF is opened with readlif and every image (series) in it is processed.
2. **Channel assignment dialog.** The first time a channel setup is seen, a window opens titled *Confirm channel assignment – <series>*. For each channel it shows:
   - a thumbnail of the middle z-slice (auto-contrasted)
   - the channel number, and the LUT/dye name from the Leica metadata
   - a drop-down to assign the channel's role: **nuclei**, **mbp**, **pillar** or **skip**. The suggested roles are channel 0 = nuclei, 1 = mbp, 2 = pillar; check them every time.

   Press **Confirm mapping**. Closing the window also confirms whatever is selected. The answer is reused for every later series with the same channel metadata during this run. Different channel setups get their own dialog.
3. Every slice (all z, all timepoints) of each assigned channel is saved as a PNG named `<series>_c<channel>_<colour>_z<z>_t<t>.png`:
   - **Each slice is contrast-stretched on its own** (its minimum → black, its maximum → full).
   - The **pillar** role is saved in a yellow colour map and every other role in cyan.
   - **MBP stacks drop their first and last z-slice** (when there are more than 2 slices).
4. The pixel size from the LIF metadata is saved to `calibration.json` for the nuclei count. A series without it prints a warning.

If one LIF fails, its error is printed and the others continue. The stage fails only if none could be extracted. Characters not allowed in filenames (`<>:"/\|?*`) are replaced with `_` in LIF and series names.

### 0. Pillar detection

**Reads:** every pillar folder under Root. **Writes:** `Results/pillar_detection/<series>/`.

For each pillar stack:

1. **Loads every slice.** Slices of a different size from the first are skipped. Each pixel's intensity is the brightest of its R, G and B values, so the colour map doesn't dilute it.
2. **Masks debris**: large, very bright regions, using the *Debris* settings.
3. **Finds round blobs of pillar size** in each slice with an Otsu threshold. Each slice is also tried with its contrast **inverted**; the version that finds more pillars is kept, and the inverted slices are reported.
4. **Drops noisy slices**, using *Usable slice fraction / min blobs*.
5. **Merges** detections of the same pillar across slices.
6. **Fits an axis-aligned lattice** (pitch and offset in x and y), then refines it. At least 6 merged detections are needed. With fewer, or with an implausible fit, the stack keeps its raw detections only, with no grid.
7. **Fills the grid.** Every lattice node inside the grid extent becomes a site:
   - **detected**: a pillar was found there
   - **inferred**: no pillar was found, so the position comes from the lattice
   - **off-lattice**: a detection away from every node, discarded unless *Keep off-lattice blobs as sites* is on
8. **Writes the outputs** and prints a per-stack summary with warnings. It warns when over 10% of detections are off-lattice, or over half the grid is inferred; check the QC image in either case.

**Outputs per stack**

| File | Contents |
|---|---|
| `<stack>_coords.json` | Every site: position, lattice row/column, type, radius, best z, whether it is clipped by the edge, and local evidence. Also the lattice fit, the slices used, disregarded and flipped, and all settings used. Crops and scoring read this file. |
| `<stack>_mask.png` | White discs where pillars were detected in the usable slices. |
| `<stack>_grid_mask.png` | White discs at every grid site (only when a lattice was fitted). |
| `<stack>_qc.png` | Max projection with circles: **red** = detected, **yellow** = inferred (a smaller circle means no local evidence), **blue** = off-lattice kept as a site. **Grey crosses** = off-lattice detections that were discarded. |
| `<stack>_focus_stack.png` | Max projection with each pillar shown from the slice where it was sharpest. |

### 1. Crops

**Reads:** pillar-detection output and the matching MBP stacks. **Writes:** into the Crops folder.

1. Each pillar stack is paired with its MBP stack. The series name becomes `<series>/<position key>`.
2. Sites are chosen. By default this is a stratified random sample of *Sites sampled per series*, honouring the *Include …* site options. With **every site**, all sites are chosen, and the sample is ranked first.
3. Slices are chosen. By default *Slices per site* evenly spread over the Z range; with **every z-slice**, all slices in range.
4. For each chosen slice, the MBP image (z-smoothed, then stretched between its 1st and 99.7th percentile) is cut around each site into an RGB crop:
   - **red** = the MBP signal
   - **green** = discs at every pillar position in the window
   - **blue** = a small marker on the pillar being scored

   The window is *Crop size* × pitch, resized to *Model input size*.

**Outputs**

| File | Contents |
|---|---|
| `<series>/s<site>_z<n>.png` | One crop. `<n>` is the slice's position in the stack; the true z is in the manifest's `slice_key`. |
| `manifest.csv` | One row per crop: `filepath, series, site_id, source, priority, slice_idx, slice_key, x, y`. It is sorted in labelling order, with the stratified sample (priority 0) first. |
| `sites/<series>.csv` | Every site of the series: id, position, type, grid row/column, size. |
| `preproc.json` | The crop geometry; copied into the model at training. |

Crops overwrite crops of the same name. `labels.csv` is never touched by this stage. The console warns when more than 5,000 crops are written; label in manifest order and stop when the classes stop improving.

### 2. Labelling window

**Reads:** `Crops/manifest.csv` (and an existing `labels.csv`). **Writes:** `Crops/labels.csv`.

The window shows one crop at a time, enlarged to 720 × 720, in manifest order. It opens at the **first crop not yet labelled**, so you can stop and resume at any time. Every label is saved to `labels.csv` the moment you press the key.

**The four classes**

| Key | Class | Meaning |
|---|---|---|
| **0** | none | not wrapped |
| **1** | partial_lt50 | somewhat wrapped (< 50%) |
| **2** | mostly | ≥ 50%, with a gap visible |
| **3** | full | a complete ring |

**Keys**

| Key | Action |
|---|---|
| **0 1 2 3** | Label the current crop and move to the next one. |
| **→ / ←** | Next / previous crop, without labelling. Use ← to go back and re-label. |
| **u** | Undo the last label and jump back to that crop. Repeats back through the session. |
| **v** | Cycle the view: *cyan* → *mbp* → *raw*. |
| **+** (or **=**, keypad +) / **−** (or keypad −) | Brighten / darken faint MBP (display gamma, 0.4–4.0, starts at 1.6). |
| **] / [** | More / less yellow fill on the pillar discs (0–0.9, starts at 0.10). |
| **. / ,** | Stronger / weaker yellow pillar outlines (0–1, starts at 0.55). |
| **q**, or closing the window | Save and quit. Training starts next, if ticked. |

The display settings change only what you see; the crop files are never modified.

**Views**

| View | Shows |
|---|---|
| **cyan** | MBP in cyan, pillar discs in translucent yellow, and a white outline round the pillar being scored. |
| **mbp** | MBP alone in cyan, with a white circle round the pillar being scored. Use it to judge the wrap without the overlays. |
| **raw** | The crop exactly as the model sees it: red = MBP, green = pillars, blue = target marker. |

**Status line**: position in the list, total labelled, series, site number, site type (when not *detected*), z, the current label, and the display settings. The two lines underneath list the classes and the keys. When you pass the last crop, the window shows *Done – press q to quit*. The console then prints the label count per class.

`labels.csv` columns: `filepath, label, series, site_id, source, slice_key`.

### 3. Training (fresh)

**Reads:** `Crops/labels.csv` and the crops. **Writes:** the Model folder.

1. Rows whose crop file is missing are dropped. The class counts are printed, with a warning for any class with fewer than 20 examples.
2. Labelled crops are split into train and eval by *Split by* / *Hold-out fraction*. Both sides must be non-empty.
3. The *Backbone* is loaded with a new 4-class head and fine-tuned:
   - training crops are randomly rotated and flipped (all 8 symmetries of the square lattice)
   - optional class-weighted loss
   - 10% learning-rate warm-up, and mixed precision on GPU
4. After every epoch the model is evaluated, and a checkpoint is saved (the 2 most recent are kept). At the end, the **epoch with the best quadratic weighted kappa (QWK)** is restored.
5. The final metrics and the confusion matrix are printed and saved.

**Metrics**

| Metric | Meaning |
|---|---|
| `accuracy` | Exact-class agreement. |
| `adjacent_acc` | Agreement within one class. |
| `mae` | Mean absolute class error. |
| `qwk` | Quadratic weighted kappa: agreement on an ordered scale, where large errors cost more. Used to pick the best epoch. |

**Outputs:** the model files (`config.json`, weights, image-processor settings), `preproc.json`, `confusion_matrix.csv` (rows = true class, columns = predicted), and `checkpoint-*` folders.

### 3B. Continue training

**Reads:** the Parent model, `Crops/labels.csv` (the **new** set), and optionally `Old crops/labels.csv`. **Writes:** the Model folder.

1. Loads both crop sets. Series names are tagged `new:` / `old:` so they never collide. Without an old set, a warning explains that nothing then guards against forgetting.
2. **Checks crop geometry** against the parent's `preproc.json`: crop size, input size, pillar radius and edge sliding. A mismatch stops the run unless *Continue: allow crop-geometry mismatch* is on.
3. Splits **each** set into train and eval on its own series, and trains on the union of both training parts.
4. Applies *Continue: freeze first N blocks*, if set.
5. **Evaluates the parent on each eval set ("Before")**. With *Continue: evaluate only*, it stops here.
6. Trains with the *Continue:* learning rate and epochs (and early stopping, if on), then **evaluates again ("After")**, per set.
7. Prints QWK before → after for each set. If the old set dropped by more than 0.02, it warns that this is forgetting, and suggests a lower learning rate, fewer epochs, or freezing 6 blocks.

**Outputs:** the model files, `preproc.json` (from the new crops, else the parent's), `confusion_matrix_new.csv`, `confusion_matrix_old.csv`, and `train_manifest.json` (parent, crop sets, sizes, class counts, settings, and before/after metrics).

After continue training, re-score **every** series with the new model before comparing conditions.

### 4. Scoring

**Reads:** the Model folder, pillar-detection output, the MBP stacks. **Writes:** into the Results folder.

For each series:

1. Builds the site list with the *Include …* options, and selects slices in the Z range.
2. For every slice: z-smooths and stretches it, crops every site with the **model's** crop geometry, and gets the 4 class probabilities for each site.
3. Takes the most likely class for each site on each slice, then combines the slices with **every** aggregation rule.
4. Runs the MIP pass, if on.
5. Writes the CSVs and the QC overlay, and prints the headline counts.

It **warns** when:
- the MIP raised more than 10% of pillars (check `qc/mips/<series>/`)
- `max`/`maxmip` and `consec2` disagree on more than 10% of pillars (likely single-slice misfires)

#### The MIP pass

A pillar wrapped across several slices may never look fully wrapped on any one slice. For each site, the MIP pass:

1. takes the slices where it scored above class 0
2. projects their crops into one image (*MIP projection*)
3. pulls the brightness back to single-slice levels (*MIP dulling*)
4. scores that projected crop once

Sites that scored 0 on every slice are not projected. The `maxmip` rule takes the higher of the per-slice maximum and the MIP class.

**Outputs:** see [Results folder](#results-folder).

### N. Nuclei count

**Reads:** every nuclei folder under Root, plus each series' `calibration.json`. **Writes:** `Results/nuclei/`.

1. **One maximum-intensity projection (MIP) per nuclei stack** is saved as an 8-bit TIFF, `Results/nuclei/<series>/<stack>_MIP.tif`. An existing MIP that is newer than all its slices is reused, unless *Remake existing MIPs* is on.
2. **The pixel size** comes from `calibration.json`, otherwise from *Fallback pixel size*. If there is neither and a size filter is set, that stack is skipped and marked in the summary. With the size filter at 0, sizes are left in pixels.
3. **Fiji is started once for all MIPs** with a generated macro, `_count_nuclei.ijm`. For each MIP, the macro:
   - opens it and sets its pixel size
   - runs StarDist (*Versatile (fluorescent nuclei)*: normalisation 1–99.8%, NMS 0.4, 2-pixel boundary exclusion) at the chosen probability threshold
   - saves the unfiltered ROIs and outlines
   - rejects detections that are too small or dark-centred, recording each detection's measurements
   - saves the filtered ROIs, outlines and a count

   Fiji closes itself when done. These are the same StarDist settings and filters as the `LIF_Nuclei_Count_ExcludeHollow_5um` macros, applied to the pipeline's MIPs instead of the LIF.
4. **The counts are collected** into `nuclei_summary.csv`, and a table is printed.

Close any open Fiji window before running. If the macro hits an error, Fiji stays open on the error dialog; read it, then close Fiji, and the pipeline marks the remaining stacks *not counted*. Rerunning retries them, and the MIPs are reused.

The counts can differ slightly from running the old macros on the `.lif` directly. This is because extraction contrast-stretches each slice separately.

---

## 7. Checks before a run

When you press **Run ▶** (or run headless), the plan is checked first.

**Errors** (the run does not start):

| Message | Cause |
|---|---|
| Root folder is empty / Results folder is empty | Set them on the Paths tab. |
| Z range '…' must be blank or lo:hi | A badly formatted Z range. |
| LIF extraction is ticked but no .lif files were added | Add files, or untick A. |
| Root folder does not exist | Only checked when extraction is off. |
| Crops folder is empty / Model folder (output) is empty | Train mode. |
| Continue training needs a parent model folder containing a trained model | The Parent model has no `config.json`. |
| Parent model and output model folders must differ | The parent would be overwritten. |
| Old crops folder has no labels.csv | The replay set is not labelled. |
| Scoring needs a trained model in the Model folder | Score mode, and the Model folder has no `config.json`. |
| Nuclei count: Fiji path is empty / Fiji not found / no Fiji launcher in … | Fix the Fiji path on the Nuclei tab. |
| Nuclei StarDist probability must be between 0 and 1 / … zero or greater | A bad value on the Nuclei tab. |

**Warnings** (you are asked *Run anyway?*):

- Pillar detection is off and `Results/pillar_detection` doesn't exist yet.
- Cropping is off and there is no `manifest.csv` in the Crops folder.
- A model already exists in the Model folder and will be overwritten.
- Rule `maxmip` with the MIP pass off gives the same counts as `max`.
- No StarDist plugin was found in the chosen Fiji.

---

## 8. Command line

```
python Myelination.py [--headless] [--mode train|score|none] [--nuclei] [--settings FILE]
```

| Option | Effect |
|---|---|
| *(none)* | Open the settings window. |
| `--headless` | No window: run straight away with the saved settings. Warnings are printed; errors stop the run. |
| `--mode train\|score\|none` | Override the saved mode for this run. |
| `--nuclei` | Also run the nuclei count this time. It can only turn N on; to turn it off, untick it and save. |
| `--settings FILE` | Use another settings file instead of `wrap_pipeline_settings.json`. Useful for keeping several configurations. |

Examples:

```
python Myelination.py --headless --mode score                  score with saved settings
python Myelination.py --headless --mode none --nuclei          nuclei count only
python Myelination.py --settings cohort2.json                   window, using another settings file
```

**Exit codes:** 0 = finished, 1 = a stage failed, 2 = the pre-run check found errors.

**Headless runs still open windows in two stages.** LIF extraction asks for channel roles, and labelling is interactive. Untick them for unattended runs. If tkinter isn't installed at all, the program runs headless automatically, and those two stages stop with an error.

---

## 9. Output reference

### Root folder

```
<lif>/<series>/calibration.json
<lif>/<series>/{nuclei,mbp,pillar}/<series>_c<ch>_<colour>_z<z>_t<t>.png
```

### Crops folder

```
manifest.csv             crops in labelling order
labels.csv               your labels
preproc.json             crop geometry
sites/<series>.csv       all sites per series
<series>/s####_z###.png  the crops
```

### Model folder

```
config.json, model weights, preprocessor_config.json   the model
preproc.json                                           crop geometry it was trained with
confusion_matrix.csv                                   fresh training
confusion_matrix_new.csv, confusion_matrix_old.csv     continue training
train_manifest.json                                    continue training
checkpoint-*/                                          per-epoch checkpoints (can be deleted)
```

### Results folder

```
pillar_detection/<series>/<stack>_coords.json, _mask.png, _grid_mask.png, _qc.png, _focus_stack.png

sites/<series>.csv             every site used for scoring
per_slice/<series>.csv         one row per site per slice
per_pillar/<series>.csv        one row per site
per_stack_counts.csv           class counts per slice
per_series_counts.csv          everything per series
directory_summary.csv          the headline table, with a TOTAL row
qc/<series>.png                scoring overlay
qc/mips/<series>/site####_slice<a>_mip<b>.png   MIP raised a site from class a to b

nuclei/nuclei_summary.csv
nuclei/<series>/<stack>_MIP.tif
nuclei/<series>/<stack>_unfiltered_ROIs.zip, _unfiltered_outlines.png
nuclei/<series>/<stack>_filtered_ROIs.zip,   _filtered_outlines.png
nuclei/<series>/<stack>_detection_measurements.csv
nuclei/<series>/<stack>_count.csv
nuclei/_count_nuclei.ijm, _fiji_jobs.tsv, _fiji_done.txt   (working files for the Fiji run)
```

**Scoring CSV columns**

| File | Columns |
|---|---|
| `per_slice/<series>.csv` | `site_id, source, slice_idx, slice_key, pred` (class on that slice), `p0–p3` (probabilities) |
| `per_pillar/<series>.csv` | `site_id, x, y, source, grid_i, grid_j, n_slices`, `agg_<rule>` for every rule, `mip_class`, `n_mip_slices` (slices projected), `mip_conf`, `mip_upgraded` (MIP class > per-slice max), `mean_conf` (mean top probability across slices) |
| `per_stack_counts.csv` | per series and slice: `det_class_0–3` (detected sites), `all_class_0–3` (all sites) |
| `per_series_counts.csv` | site counts by type, `lattice_ok`, `pitch_px`, slices, rule, denominator, MIP settings, `n_mip_scored`, `n_mip_upgraded`, headline `class_<k>_<name>` and `pct_class_<k>`, `det_class_*`, `all_class_*`, and `alt_<rule>_class_*` for every other rule |
| `directory_summary.csv` | The headline subset of `per_series_counts.csv`, one row per series plus `TOTAL` |

**Scoring QC overlay** (`qc/<series>.png`): the last slice in the Z range, with a ring per pillar coloured by class under the chosen rule. **Grey** = 0, **blue** = 1, **amber** = 2, **green** = 3. Thin rings are detected sites; thick rings are inferred or off-lattice.

**`nuclei_summary.csv` columns:** `series, stack, status` (counted / skipped: no pixel size / not counted / unreadable count file), `n_slices_in_mip, initial_detections, below_min_diameter, dark_centre, total_rejected, retained_nuclei, pixel_width_um, pixel_height_um, calibration` (where the pixel size came from), the four filter settings, and `mip`. The two reject columns can overlap, since one object can be both too small and dark-centred. **`retained_nuclei` is the count.**

**`<stack>_detection_measurements.csv` columns** (one row per StarDist detection): `Detection, X_pixels, Y_pixels, Area_um2, Equivalent_diameter_um, Object_mean, Centre_mean, Centre_ratio, Too_small, Dark_centre, Rejected`.

---

## 10. Typical workflows

**First model, from raw LIFs**
1. Paths tab: set Root. Press *Put Crops and Results inside Root*, then set the Model folder.
2. Run tab: choose **Train a model**. Tick A (add the LIFs), 0, 1, 2 and 3 (fresh). Press **Run**.
3. Confirm the channel roles when asked. Check a few `pillar_detection/…_qc.png` images, then label in the window that opens. Close it to start training.

**Label more and retrain from scratch:** Train mode, with A and 0 off. Tick 1 with *every site* on, then 2 and 3. Labelling resumes where you stopped. Cropping again does not touch your labels.

**Score new data:** Score mode. Tick A and 0 for new LIFs, or neither if they are already extracted and detected. Set the Model folder to the trained model and press **Run**. Read `directory_summary.csv`.

**Adapt a model to a new batch:** Train mode with 3 set to *continue*. Set the Parent model, a **new** Model folder, the new batch's Crops folder, and the parent's crops as *Old crops (replay)*. Tick *Continue: evaluate only* first. Train only if the "new" score is poor.

**Nuclei only:** mode **Neither**. Tick N (plus A if not yet extracted), set the Fiji path on the Nuclei tab, and press **Run**. From the command line: `python Myelination.py --headless --mode none --nuclei`.

---

## 11. Troubleshooting

| Problem | What to do |
|---|---|
| *no 'pillar' folders found* / *no 'nuclei' folders found* | Check the Root folder and the folder-name settings. Folders are found at any depth, but the names must match. |
| *no pillar-detection output for series …* | Run pillar detection, or check that the Results folder is the one detection wrote to. |
| *no matching mbp stack for …* | The pillar and MBP filenames don't share a position key once `_c<N>_<colour>` and `_z<n>` are removed. Check the names, e.g. a different `_t` timepoint or a renamed series. |
| *(ignored N file(s) with no recognisable z-index token)* | These files lack `_z<number>` before `_` or `.`. |
| *too few raw candidates to fit a lattice* | Fewer than 6 pillars were found. Check the pillar diameter range against `<stack>_mask.png`, and the debris settings. |
| Many inferred or off-lattice sites | Open `<stack>_qc.png`. Adjust the diameters, *Max eccentricity* or *Off-lattice residual*. |
| *… crops were cut with different geometry* | Recut the new crops with the parent model's crop settings, or tick *Continue: allow crop-geometry mismatch*. |
| *Classes [...] have <20 examples* | Label more of those classes, e.g. with *every site* on, so later manifest rows add variety. |
| Out of memory during training or scoring | Lower *Batch size* / *Inference batch size*, or turn off *MIP shared normalisation*. |
| *training/scoring needs torch and transformers* | `pip install torch transformers accelerate`. |
| *LIF extraction needs readlif* | `pip install readlif`. |
| Fiji opens and stops on an error | Usually StarDist or CSBDeep is missing: enable both update sites and restart Fiji. Close Fiji and rerun; MIPs are reused. |
| *skipped: no pixel size* in the nuclei summary | Re-extract that LIF (this writes `calibration.json`), or set *Fallback pixel size (um)*. |
| The labelling window doesn't respond to keys | Click on it once to give it keyboard focus. |

---

## 12. Running it as an application

`build_app.bat` sets everything up for you. It finds or installs Python, installs every library into a private environment (`python_env`), and builds `Myelination.exe`. You don't need to install anything by hand first.

### Building `Myelination.exe`

1. Put `build_app.bat` next to `Myelination.py` and **double-click it**. You need an internet connection and about 10 GB of free disk space.
2. It works through five steps, printing each:

   | Step | What happens |
   |---|---|
   | 1. Python | Looks for Python 3.12, 3.13, 3.11, 3.10 or 3.14 (with tkinter), in that order. If there is none, it offers to install Python 3.12 with *winget*; otherwise it points you to python.org. |
   | 2. Environment | Creates `python_env` next to the script. It is private to this app, so your other Python setups, Anaconda included, are not touched. On later runs it is reused. |
   | 3. PyTorch | Asks for a **GPU** or **CPU-only** build. The default is GPU if this PC has an NVIDIA card (picked after 30 s). A GPU build also runs on PCs without a GPU, just larger. A CPU build is smaller but never uses a GPU. |
   | 4. Libraries | Installs numpy, Pillow, scipy, scikit-image, readlif, transformers, accelerate and PyInstaller, then checks they all import and prints the versions. |
   | 5. Build | Builds `dist\Myelination\Myelination.exe` and copies your `wrap_pipeline_settings.json` in, if there is one. |

3. The first run takes 15–40 minutes, mostly downloading PyTorch; later runs only rebuild the app. Keep the **whole `dist\Myelination` folder together**, because the `.exe` needs the `_internal` folder beside it. Copy it anywhere, and make a desktop shortcut to `Myelination.exe`.

| Command | Does |
|---|---|
| `build_app.bat` | Install whatever is missing, then build. |
| `build_app.bat setup` | Install the libraries only, without building the `.exe`. `Run_Myelination.bat` then uses them. |
| `build_app.bat clean` | Delete `python_env`, reinstall everything from scratch (newest versions), then build. |

**Settings at the top of `build_app.bat`**
- `TORCH_CUDA`: which CUDA build of PyTorch a GPU build uses: `cu126` for older NVIDIA drivers, `cu128` (default), or `cu130` for the newest.
- `PACKAGES`: the other libraries. To keep a library at a version that worked, write it with the version, e.g. `transformers==X.Y.Z`, then run `build_app.bat clean`.

### Double-click launcher (no `.exe`)

`Run_Myelination.bat`, next to `Myelination.py`, opens the settings window as `python Myelination.py` would. It uses `python_env` if `build_app.bat` has made one (`build_app.bat setup` is enough); otherwise it uses the Python on PATH. For a desktop icon, right-click it, then choose *Send to* › *Desktop (create shortcut)*. Unlike the `.exe`, it always runs your current `Myelination.py`.

### How the packaged app behaves

- It opens the same settings window, and the console window alongside it shows the progress.
- Settings are saved in `wrap_pipeline_settings.json` **next to `Myelination.exe`**.
- The command-line options work the same: `Myelination.exe --headless --mode none --nuclei`.
- Backbone models are still downloaded from Hugging Face the first time a new backbone is used, and cached in your user folder. Trained models in your Model folder need no download.
- Fiji is not bundled; the nuclei count uses the Fiji set on the Nuclei tab, as before.
- To use a backbone family other than ViT, ConvNeXt, ResNet or DINOv2, add `--collect-submodules transformers.models.<family>` to `build_app.bat` and rebuild.
- A GPU build uses the GPU only on PCs with a recent enough NVIDIA driver; otherwise it runs on the CPU. You can also set *Device* to `cpu`.
- Rebuild after every change to `Myelination.py`; the `.exe` does not read the `.py` file.
- If the `.exe` closes immediately, open a command prompt in its folder and run `Myelination.exe`, to keep the error on screen.
- `python_env` is only needed for building. Keep it so rebuilds are fast, or delete it to free several GB.
