# 4DAnyone → 4DGaussians

Dynamic (4D) reconstruction from a 4DAnyone multi-view generation, plus a static
3DGS control for comparison. Everything lives outside the working 4DAnyone and
Rerun environments: the upstream checkout is `work/4DGaussians` pinned to
`843d5ac636c37e4b611242287754f3d4ed150144`, and the interpreter is `work/4dgs-env`.

## Current state (2026-09-06)

Best model: **`outputs/4dgs/IMG_7164_quality3`** at iteration 30,000 —
323,546 Gaussians, 69 minutes on an RTX 6000 Ada, from the 47-view
full-resolution export `outputs/4dgs/IMG_7164_full48`.

Held-out camera 40 (yaw 180°, pitch 35°), all 121 frames:

| Model | Gaussians | Full PSNR | Foreground PSNR | Detail vs GT | Colour outliers |
|---|---|---|---|---|---|
| `IMG_7164_full48_model` (20k) | 60,564 | 29.16 dB | 22.42 dB | 63.1% | 0.03% |
| `IMG_7164_quality` (30k) | 238,267 | 28.16 dB | 21.00 dB | 67.1% | **7.08%** (broken) |
| **`IMG_7164_quality3` (30k)** | **323,546** | **29.24 dB** | **22.73 dB** | **74.6%** | 0.16% |
| `IMG_7164_static_model` (static, frame 42 only) | 116,914 | — | 25.83 dB¹ | 134.9%¹ | — |

¹ single frame, camera 40 only. Detail above 100% is ghosting, not real detail.

## Reproduce the best model

The export already exists; this is the 69-minute step.

```bash
cd work/4DGaussians
../4dgs-env/bin/python train.py -s ../../outputs/4dgs/IMG_7164_full48 \
  --model_path ../../outputs/4dgs/IMG_7164_quality3 \
  --configs ../../integrations/4dgaussians/quality3.py \
  --expname IMG_7164_quality3 --port 6025 \
  --save_iterations 7000 14000 30000 --test_iterations 7000 14000 30000 \
  --checkpoint_iterations 30000
```

## Environment setup

Shell snippets below use `$FDA_ROOT` for this repository's checkout and
`$GS_ROOT` for a vanilla 3DGS checkout (only needed for the SIBR viewer and
the static control):

```bash
export FDA_ROOT=$(pwd)
export GS_ROOT=$HOME/gaussian-splatting
```

From the 4DAnyone root, with Python 3.11 and the CUDA 12.8 toolkit:

```bash
git clone --recursive https://github.com/hustvl/4DGaussians work/4DGaussians
git -C work/4DGaussians checkout 843d5ac636c37e4b611242287754f3d4ed150144
git -C work/4DGaussians submodule update --init --recursive
python -m venv work/4dgs-env
work/4dgs-env/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
work/4dgs-env/bin/python -m pip install -r integrations/4dgaussians/requirements.txt
python integrations/4dgaussians/patch_checkout.py work/4DGaussians
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 \
  work/4dgs-env/bin/python -m pip install --no-build-isolation \
  ./work/4DGaussians/submodules/depth-diff-gaussian-rasterization \
  ./work/4DGaussians/submodules/simple-knn
```

Architecture 8.9 targets an RTX 6000 Ada; change it for other hardware.
`environment-freeze.txt` records the resolved dependencies. Re-run
`patch_checkout.py` after any fresh clone — it is idempotent.

`patch_checkout.py` applies: the `fdanyone` scene loader, MMEngine in place of
legacy MMCV, `<cfloat>` for simple-knn under CUDA 12.8, a data-loader epoch-reset
fix, `weights_only=False` for locally produced checkpoints, **8 data-loader
workers** (full-resolution PNG decoding otherwise leaves the GPU ~90% idle), and
**a densification cap of 800,000** (upstream's 360,000 blocks a detailed human).

## Pipeline

### 1. Export (~22 min, one-off per clip)

Run in the 4DAnyone conda environment, from the repository root:

```bash
conda run --no-capture-output -n 4danyone python scripts/export_4dgaussians.py \
  data/fdanyone/IMG_7164_linux_compatible --output outputs/4dgs/IMG_7164_full48 \
  --scale 1.0 --train-views 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 41 42 43 44 45 46 47 \
  --test-views 40
```

48 cameras × 121 frames of RGBA PNG at 704×1280, ~4.1 GB, ~28 s per camera
(BiRefNet segmentation dominates). The manifest `fdanyone.json` is written last,
so an incomplete export cannot be trained on. Choose a fresh output path for
another export; `--scale 0.5` gives the older 352×640 datasets.

At least one camera must be held out: the exporter rejects overlapping splits and
the loader takes its evaluation and video camera from `test_views[0]`.

### 2. Denser seed (optional, ~1 min)

Densification grows from the seed, and 40,000 points is a weak start:

```bash
conda run --no-capture-output -n 4danyone python integrations/4dgaussians/reseed.py \
  data/fdanyone/IMG_7164_linux_compatible outputs/4dgs/IMG_7164_full48 --points 200000
```

Rebuilds only `seed.npz`, leaving the segmented images alone. Note the first
prune drops most of it (200k → ~22k) and densification regrows from there; the
benefit is distribution, not a head start on count.

### 3. Validate (~3 min)

```bash
work/4dgs-env/bin/python integrations/4dgaussians/validate_export.py \
  outputs/4dgs/IMG_7164_full48
```

Checks every frame's size and mode, non-degenerate masks, orthonormal poses,
centered intrinsics and a disjoint split before any GPU time is spent.

### 4. Train

See "Reproduce the best model" above. Configs, in order of quality:

| Config | Notes |
|---|---|
| `pilot.py` | Original 2,000-iteration smoke test. **Downgrades upstream defaults — do not build on it.** |
| `train_14000.py` | Pilot extended to 14,000 iterations. Historical. |
| `full48.py` | 47 views, upstream grid still downgraded. Historical baseline. |
| `quality.py` | Restores upstream grid, 4× primitives, DSSIM. **Produces the magenta-arm artifact.** |
| `quality2.py` | `quality.py` + opacity reset. Does *not* fix the artifact; kept as a record. |
| `quality3.py` | **Current best.** `quality.py` + `no_dshs=False, no_do=False`. |

### 5. Render and evaluate

```bash
cd work/4DGaussians
../4dgs-env/bin/python render.py --model_path ../../outputs/4dgs/IMG_7164_quality3 \
  --iteration 30000 --configs ../../integrations/4dgaussians/quality3.py --skip_train
cd ../..
work/4dgs-env/bin/python integrations/4dgaussians/evaluate.py \
  --export outputs/4dgs/IMG_7164_full48 \
  --renders outputs/4dgs/IMG_7164_quality3/test/ours_30000/renders --camera 40
```

To render an arbitrary training camera at one timestamp (for front-view
comparisons, which `render.py` will not do without rendering all 5,687 images):

```bash
work/4dgs-env/bin/python integrations/4dgaussians/render_camera.py \
  -s outputs/4dgs/IMG_7164_full48 --model_path outputs/4dgs/IMG_7164_quality3 \
  --configs integrations/4dgaussians/quality3.py --iteration 30000 \
  --camera 0 --frame 42 --out /tmp/front.png
```

### 6. Interactive viewer

`SIBR_remoteGaussian_app` is a *client*: the Python server holds the deformation
network and renders each requested camera. Both run on the same Linux host; no
Windows machine or SSH forwarding is needed.

Build it once (needs `sudo` for the system packages):

```bash
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
  libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev \
  libxxf86vm-dev libembree-dev
cd $GS_ROOT/SIBR_viewers
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .
cmake --build build -j24 --target install
```

Then, once per model, write the `cameras.json` that SIBR needs and start both
processes:

```bash
work/4dgs-env/bin/python integrations/4dgaussians/write_sibr_cameras.py \
  outputs/4dgs/IMG_7164_full48 outputs/4dgs/IMG_7164_quality3

work/4dgs-env/bin/python integrations/4dgaussians/serve_viewer.py \
  -s outputs/4dgs/IMG_7164_full48 --model_path outputs/4dgs/IMG_7164_quality3 \
  --configs integrations/4dgaussians/quality3.py --iteration 30000 --port 6019

# in a second terminal
cd $GS_ROOT/SIBR_viewers
./install/bin/SIBR_remoteGaussian_app --port 6019 \
  --path $FDA_ROOT/outputs/4dgs/IMG_7164_quality3
```

Controls (trackball is the startup mode): **left-drag rotates**, right-drag pans,
scroll zooms; starting a drag near the window border rolls or dollies instead.
`Y` toggles trackball/FPS, `B` orbit/FPS, `Space` toggles snapping, `P` snaps to a
camera. Time advances on the server's wall clock; `--time 0.35` freezes an
instant instead. There is no time slider in the client.

### 7. Static control (optional, ~5 min)

Reconstructs a single timestamp with vanilla 3DGS — same images, masks and
calibration, no deformation network. Establishes what the data supports.

```bash
work/4dgs-env/bin/python integrations/4dgaussians/export_static_colmap.py \
  outputs/4dgs/IMG_7164_full48 outputs/4dgs/IMG_7164_static_f42 --frame 42
cd $GS_ROOT
$HOME/miniconda3/envs/gaussian_splatting/bin/python train.py \
  -s $FDA_ROOT/outputs/4dgs/IMG_7164_static_f42 \
  -m $FDA_ROOT/outputs/4dgs/IMG_7164_static_model \
  --eval --iterations 30000 --test_iterations 7000 30000 --save_iterations 30000
```

Image order matches camera id, so 3DGS's `--eval` holdout (every 8th index)
leaves cameras 00, 08, 16, 24, 32 and 40 unseen — camera 40 being the same view
that scores the dynamic models.

## Findings

### What actually improved sharpness

Camera count was **not** the lever. Going from 12 to 47 training views improved
geometric agreement but left detail at 63% of ground truth. What mattered:

1. **Primitive count.** The pilot's densification settings throttled growth to
   1.5× from seed (40,000 → 60,564), and upstream's hard cap of 360,000 would
   have blocked a detailed human regardless. Now 323,546.
2. **Deformation grid capacity.** `pilot.py` downgraded upstream's own defaults
   to a `32⁴` grid with `multires=[1,2]` and 16 features. Restoring
   `64×64×64×60`, `multires=[1,2,4,8]`, 32 features was the single largest
   quality change. The temporal axis should be about half the frame count
   (60 for 121 frames), per upstream's comment in `arguments/__init__.py`.
3. **Per-timestamp colour and opacity** (`no_dshs=False, no_do=False`). Required
   once the grid is large — see the trap below.
4. Supporting changes: L1 + 0.2·DSSIM (upstream leaves `lambda_dssim` at 0),
   8 data-loader workers, densify threshold 0.0001, 200k seed points.

### What did not matter

- **Opacity reset.** Restoring it (`quality2.py`) changed nothing about the
  artifact it was meant to fix. Upstream disables it for real captures; either
  setting works here.
- **More cameras**, beyond the geometric agreement already gained by 47.

### Measurement

**Full-image PSNR is misleading on this data** — the subject covers ~14% of the
frame and the black background inflates every score by ~7 dB. Worse, the two are
not even rank-consistent: `quality3` scores 29.24 dB full-image against the older
model's 29.16 dB, a difference that says nothing, while carrying 18% more detail.

Use `evaluate.py`, which reports foreground PSNR, a detail ratio (mean gradient
magnitude inside an eroded silhouette, relative to ground truth) and a colour
outlier percentage. **PSNR alone will not tell you whether a run got sharper**;
a sharper render that is slightly misaligned scores worse than a blurry one.

The colour-outlier figure is a regression guard for the failure below: it reads
0.03% on ground truth, 0.16% on a healthy model, and 7.08% on the broken one.

### The ceiling: generated views disagree with each other

The static control fits its **training** views to 35.55 dB but **held-out** views
to 25.83 dB, at a single instant with no deformation network. That ~10 dB gap is
pure multi-view inconsistency in the generated videos, and no reconstruction
tuning removes it.

Consequences worth remembering when comparing against a static 3DGS result:

- On a **training** view, static 3DGS essentially reproduces the input
  (95.6% detail, 35.55 dB). Much of that is memorising one image.
- On a **held-out** view, static ghosts badly — 134.9% "detail" is doubled limbs
  and streaks, not information — and the 4D model is *more accurate*
  (27.44 dB vs 25.83 dB at camera 40, frame 42).
- So a sharp screenshot from a static viewer is not evidence the 4D pipeline is
  misconfigured. Compare at matched view types, and prefer views midway between
  rig cameras, which is the honest worst case (22.5° spacing here).

The dynamic model's remaining softness — ~75% of ground-truth detail regardless
of viewpoint — is the intrinsic cost of 121 timestamps sharing one primitive set
through a deformation field, not a configuration error.

## Traps

Each of these cost real time.

- **`no_dshs` / `no_do` default to `True`**, giving every Gaussian one colour for
  all 121 frames. Harmless with a small deformation grid; with a large one,
  primitives travel far enough that a single Gaussian must serve skin at one
  instant and denim at another. The compromise drove the forearm's green channel
  to 0.004 (ground truth 0.357) — a saturated magenta arm — and it is a local
  minimum the optimiser never escapes. Fully formed by iteration 7,000 and stable
  thereafter, so a short run is enough to test for it.
- **CMake 4 rejects the vendored `xatlas`** (`cmake_minimum_required` below 3.5).
  Configure SIBR with `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`; do not edit the
  vendored source. The flag is needed on every reconfigure.
- **SIBR cannot identify the dataset** without `<path>/cameras.json`, which
  4DGaussians never writes. Without it the client connects, then aborts with
  "Cannot determine type of dataset". `write_sibr_cameras.py` generates it plus
  the `input.ply` proxy mesh from the export manifest.
- **The static COLMAP export must be premultiplied and alpha-free.** Vanilla 3DGS
  keeps raw RGB as ground truth and masks only the render, so unpremultiplied
  images give ~5 dB nonsense; keeping an alpha channel leaves the background
  unsupervised and it fills with floaters (PSNR *falls* during training,
  21.1 → 19.8 dB). `export_static_colmap.py` handles both.
- **`pgrep -x SIBR_remoteGaussian_app` always fails** — Linux truncates the
  process name to 15 characters (`SIBR_remoteGaus`). Use `pgrep -f` or `ps`.
- **Stale viewer servers hold their port.** A server from an earlier session was
  still bound to 6017 hours later. Check with `ss -ltnp | grep 601` before
  assuming a port is free.
- **Training is CPU-bound without data-loader workers.** With `num_workers=0` at
  full resolution the GPU sits at ~10%; with 8 workers it saturates at ~97%.

## Run history

Runs marked *(deleted)* were removed on 2026-09-07 to reclaim ~5.7 GB. Their
configs remain, so any of them can be reproduced; the numbers below are the
record of what they showed.

| Model | Config | Views / res | Gaussians | Result |
|---|---|---|---|---|
| `IMG_7164_model` *(deleted)* | `pilot.py` | 12 / 352×640 | 41,719 | Pipeline smoke test, 2k iterations |
| `IMG_7164_model_14000` *(deleted)* | `train_14000.py` | 12 / 352×640 | 58,712 | 27.18 dB upstream test PSNR |
| `IMG_7164_full48_model` | `full48.py` | 47 / 704×1280 | 60,564 | 63.1% detail; grid still downgraded |
| `IMG_7164_quality` *(deleted)* | `quality.py` | 47 / 704×1280 | 238,267 | Sharper but magenta arm (7.08% outliers) |
| `IMG_7164_quality2` *(deleted)* | `quality2.py` | 47 / 704×1280 | ~195k @20k | Stopped early; opacity reset did not help |
| `IMG_7164_quality3` | `quality3.py` | 47 / 704×1280 | 323,546 | **Best.** 74.6% detail, no artifact, 69 min |
| `IMG_7164_static_model` | vanilla 3DGS | 42 / frame 42 | 116,914 | Static control |

The `IMG_7164_pilot` dataset (12 views at 352×640) was deleted with the two
models that used it. `pilot.py` and `train_14000.py` are kept because
`full48.py` inherits from `pilot.py`, but reproducing those two runs now
requires re-exporting at `--scale 0.5` first.

## Files

| File | Purpose |
|---|---|
| `fdanyone_loader.py` | Lazy calibrated loader injected into the upstream `Scene` |
| `patch_checkout.py` | All modifications to the pinned checkout; idempotent |
| `quality3.py` | Current best training configuration |
| `quality.py`, `quality2.py`, `full48.py`, `train_14000.py`, `pilot.py` | Historical configs |
| `reseed.py` | Regenerate `seed.npz` at a different point count |
| `validate_export.py` | Pre-training dataset validation |
| `evaluate.py` | Foreground PSNR, detail ratio, colour-outlier guard |
| `render_camera.py` | Render one camera at one timestamp |
| `write_sibr_cameras.py` | `cameras.json` + `input.ply` for the SIBR client |
| `serve_viewer.py` | SIBR wire-protocol render server |
| `export_static_colmap.py` | One timestamp as a COLMAP model for static 3DGS |
| `test_viewer_protocol.py` | Single-frame check of the SIBR protocol |

`scripts/export_4dgaussians.py` (repository root) produces the datasets.

## Open questions

- Densification saturated at 323,546 against the 800,000 cap. A lower gradient
  threshold and more iterations may add detail; expect diminishing returns.
- **4DGaussians is a 2023 method** with vanilla-3DGS-era optimisation. A newer
  dynamic method (Deformable-3DGS, SpaceTime Gaussians, or a 4DGS variant with
  MCMC-style densification) is the largest untested lever, and has not been
  evaluated here.
- Per-view or per-time appearance modelling would let the model absorb the
  generated-view inconsistency instead of averaging it into blur. Untested.
- Only camera 40 has ever been held out by a dynamic run. A rotating holdout
  would give a more trustworthy generalisation estimate.
- Test-view targets are generated images, not independent real footage. Every
  metric here is measured against 4DAnyone's own output.
