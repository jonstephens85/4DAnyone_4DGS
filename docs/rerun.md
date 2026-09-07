# Rerun results viewer

Inspect an existing 4DAnyone run with a 3D camera rig and synchronized video panes.
The viewer reads `cameras.json`, `metadata.json`, and the selected MP4 files. It
runs independently of inference and does not load model weights.

## Install

Use a separate environment to keep viewer dependencies independent of Torch:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-rerun.txt
```

Native video playback also requires **FFmpeg >= 5.1**. The launcher checks its
version and puts the selected executable on the viewer's PATH. It prefers
`.venv/bin/ffmpeg`, then the system PATH, or accepts `--ffmpeg /path/to/ffmpeg`.
In this checkout, `.venv/bin/ffmpeg` links to the existing FFmpeg 7.1 in the
`lerobot` Conda environment. Keep that environment available or replace the
link with another compatible FFmpeg. System FFmpeg 4.4 is incompatible.
Use `scripts/open_rerun.py` to reopen recordings; invoking `rerun` directly may
pick up the old system decoder. No recording regeneration is needed for this fix.

Rerun SDK and native viewer are pinned together at 0.36.3. Run commands from the
repository root. The local `.venv` has already been installed for this checkout.

## Open results

```bash
.venv/bin/python scripts/view_rerun.py data/fdanyone/IMG_7164_linux_compatible
```

This creates `outputs/rerun/IMG_7164_linux_compatible.rrd` and launches the native
viewer. All camera frustums appear in the upper 3D pane; six generated videos
sampled across yaw and pitch layers appear below. Drag in the 3D pane to orbit
and use the shared `time` timeline to play, pause, or scrub. The `frame` timeline
is also available for frame-index inspection.

Recordings embed selected videos, so they can be reopened without the original
result folder. Existing recordings are never overwritten: reopen one directly
or choose a new `--save` path when changing the selection.

Two recordings have been prepared for the personal clip:

```bash
.venv/bin/python scripts/open_rerun.py outputs/rerun/IMG_7164_overview.rrd
.venv/bin/python scripts/open_rerun.py outputs/rerun/IMG_7164_pose_comparison.rrd
```

The overview selects six yaw directions across all three rings. The comparison
shows the front camera from each ring alongside its pose-conditioning video.

## Choose cameras

```bash
# Six views sampled from pitch layer 1 (zero-based).
.venv/bin/python scripts/view_rerun.py data/fdanyone/IMG_7164_linux_compatible \
  --layer 1 --save outputs/rerun/middle-ring.rrd

# Explicit views, with synchronized skeleton panes.
.venv/bin/python scripts/view_rerun.py data/fdanyone/IMG_7164_linux_compatible \
  --views 0 16 32 --skeletons --save outputs/rerun/front-poses.rrd

# Export without a graphical display.
.venv/bin/python scripts/view_rerun.py data/fdanyone/IMG_7164_linux_compatible \
  --max-views 8 --no-open --save outputs/rerun/eight-views.rrd
```

Camera IDs, yaw, pitch, and layer membership come from `cameras.json`. `--views`
overrides `--max-views`; combining it with `--layer` validates IDs against that
layer. Selection is made at launch. Only selected videos are embedded, while
all camera transforms remain visible. More simultaneous video panes increase
decoding and memory costs; start with six.

Timing uses output FPS from metadata and each MP4's actual frame timestamps.
Every selected video must contain the metadata's frame count. Camera transforms
retain the recorded Y-up world and OpenCV right/down/forward camera convention.

## Scope

This first milestone uses native Rerun playback and original-resolution MP4s.
Browser embedding, preview transcodes, source-video alignment, recovered SMPL-X
geometry, and live inference previews are follow-up work. The rig view contains
camera geometry and image planes, not a reconstructed photorealistic subject.
No additional novel viewpoints are synthesized by this viewer.

The visualization approach follows the [Rerun Hugging Face example](https://huggingface.co/spaces/rerun/4danyone-rerun/blob/main/fdanyone/viz.py).
This adapter uses standalone `AssetVideo` recordings for completed outputs and
has no dependency on the Space's modified inference fork or Gradio environment.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/rerun rrd verify outputs/rerun/IMG_7164_overview.rrd
```

Automated checks cover camera selection, malformed camera geometry, frame
counts, and recording validity. Visual playback and decoder performance should
be checked on the display where you will use the viewer.

## Browse all views in one recording

```bash
.venv/bin/python scripts/view_rerun.py data/fdanyone/IMG_7164_linux_compatible \
  --browse-all --no-open --save outputs/rerun/IMG_7164_all_views.rrd
.venv/bin/python scripts/open_rerun.py outputs/rerun/IMG_7164_all_views.rrd
```

The prepared all-views recording is approximately 100 MB. Below the camera rig,
choose **Overview** for the six-view comparison, or a **Layer** tab for that
pitch ring. Within each layer, select a **Views** tab to show four neighboring
yaw angles in a 2×2 grid. The timeline is shared across all tabs.

`--browse-all` embeds every generated video regardless of `--views` or `--layer`;
those options still control the overview selection. Without it, only the
selected videos are embedded. Videos are stored once even when used in multiple
panes. The rig pane includes camera entities only, excluding their video image
children. Hidden tab panes are not displayed; decoder caching and resource
usage still depend on Rerun. With `--skeletons`, each page also includes matching
pose panes, increasing the number of visible videos.

## Animated 3D skeleton

Export the cached motion using the existing **4danyone** environment (it needs
Torch, SMPL-X and the installed body-model assets). This reuses the conditioning
pipeline's canonical-world conversion on CPU; it does not rerun generation.

```bash
conda run -n 4danyone python scripts/export_rerun_motion.py \
  data/gvhmr/results/IMG_7164_linux_compatible \
  --out outputs/rerun/IMG_7164_motion.npz
.venv/bin/python scripts/view_rerun.py data/fdanyone/IMG_7164_linux_compatible \
  --browse-all --motion outputs/rerun/IMG_7164_motion.npz --no-open \
  --save outputs/rerun/IMG_7164_3d_skeleton.rrd
.venv/bin/python scripts/open_rerun.py outputs/rerun/IMG_7164_3d_skeleton.rrd
```

These artifacts are already prepared for your personal clip. The colored
conditioning keypoints and links appear under `world/body/joints` and
`world/body/bones` in the rig pane. Toggle their visibility independently using
the blueprint tree. Orbit the rig to inspect the motion from any direction;
the same time/frame timelines drive the skeleton and videos. Joint labels are
available but hidden by default to reduce clutter. This shows estimated source
motion, which may differ slightly from generated poses. The NPZ is a compact
geometry interchange file so the viewer environment stays independent of Torch.
The loader checks clip name, coordinate convention, FPS, frame count and geometry.
