# View results in Rerun

Inspect the generated camera rig and synchronized videos. Optionally add an
animated 3D skeleton from cached motion. Start by
[generating the example](example.md); run all commands below from the repository root.

## Install once

Use Python 3.11 in a separate environment:

```bash
conda activate 4danyone
python -m venv .venv
.venv/bin/python -m pip install -r requirements-rerun.txt
```

Native playback requires **FFmpeg 5.1 or newer** on PATH. Check `ffmpeg -version`.
If your system version is older, install a separate compatible FFmpeg and pass
`--ffmpeg /path/to/ffmpeg` to the viewer commands below. The launcher also checks
`.venv/bin/ffmpeg`. It does not install FFmpeg automatically.

## Open all views

```bash
.venv/bin/python scripts/view_rerun.py data/fdanyone/action \
  --browse-all --save outputs/rerun/action.rrd
```

The upper pane shows the camera rig. Below it, **Overview** shows six views;
**Layer** tabs contain groups of four neighboring views. All share the playback
timeline. Drag in the rig pane to orbit; play, pause or scrub with the timeline.
The rig excludes video planes to avoid displaying all videos simultaneously.

The `.rrd` embeds the videos and can be reopened without the result folder:

```bash
.venv/bin/python scripts/open_rerun.py outputs/rerun/action.rrd
```

Existing recordings are never overwritten. Reopen one, or use a different
`--save` filename when changing its contents. Add `--no-open` during export on a
machine without a graphical display.

## Add the 3D skeleton

Export motion once in the **4danyone** environment, then log it with the viewer:

```bash
conda run -n 4danyone python scripts/export_rerun_motion.py \
  data/gvhmr/results/action --out outputs/rerun/action-motion.npz
.venv/bin/python scripts/view_rerun.py data/fdanyone/action \
  --browse-all --motion outputs/rerun/action-motion.npz \
  --save outputs/rerun/action-with-motion.rrd
```

Toggle `world/body/joints` and `world/body/bones` in the blueprint tree. Motion
uses the same timeline and canonical coordinate frame as the cameras. It is the
estimated source pose; the generated video poses can differ. This step needs the
installed body-model assets but does not rerun video generation.

## Optional selections

```bash
# Only selected videos, with matching 2D pose-conditioning videos.
.venv/bin/python scripts/view_rerun.py data/fdanyone/action \
  --views 0 16 32 --skeletons --save outputs/rerun/action-front.rrd
```

Use `--layer 1` to select a pitch ring, or `--max-views 8` to sample more views.
Without `--browse-all`, only selected videos are embedded. With it, selections
control the overview and all videos remain available in tabs. More visible
panes increase decoding costs. `--skeletons` means 2D pose videos; `--motion`
adds true 3D joints and bones.

## Troubleshooting

- **FFmpeg decode error:** check the version and pass `--ffmpeg /path/to/ffmpeg`.
- **Recording already exists:** use `open_rerun.py`, or choose another output name.
- **Missing views or frame-count mismatch:** ensure inference completed for this clip.
- **No graphical display:** export with `--no-open`, then open the recording on a desktop.

This viewer is for completed results. It does not reconstruct a photorealistic
person or stream live diffusion previews. Visualization was inspired by the
[Rerun example](https://huggingface.co/spaces/rerun/4danyone-rerun/blob/main/fdanyone/viz.py).
