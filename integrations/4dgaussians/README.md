# 4DGaussians integration

Follow the [4DGS workflow](../../docs/4dgs.md) for installation, export, training,
rendering and viewing. It starts from a fresh checkout and uses all cameras by
default. The [included example](../../docs/example.md) supplies an input video.

## Entry points

Run from the repository root; each command supports `--help`:

| Command | Purpose |
|---|---|
| `scripts/export_4dgs.py` | Export calibrated, masked videos in the inference environment |
| `scripts/train_4dgs.py` | Train in the separate environment with configurable iterations |
| `scripts/render_4dgs.py` | Render the saved playback camera |
| `scripts/view_4dgs.py` | Prepare and serve the model to SIBR |

`default.py` contains the default training settings. New runs save their
settings and command in the model folder. Earlier configuration names remain
compatibility aliases into `experiments/`; new runs do not need them.
`scripts/export_4dgaussians.py` also remains compatible.

## Adapter internals

| File | Purpose |
|---|---|
| `patch_checkout.py` | Idempotent changes to the pinned upstream checkout |
| `fdanyone_loader.py` | Calibrated loader with independent playback camera |
| `validate_export.py` | Validate frames, masks and geometry before training |
| `reseed.py` | Replace initialization points in an existing export |
| `render_camera.py` | Render a training camera at one timestamp |
| `evaluate.py` | Optional metrics for a held-out view |
| `export_static_colmap.py` | Export one timestamp for a static trainer |
| `write_sibr_cameras.py`, `serve_viewer.py` | SIBR metadata and rendering server |
| `test_viewer_protocol.py` | Single-frame SIBR protocol check |

The checkout is pinned to `843d5ac636c37e4b611242287754f3d4ed150144` in
`work/4DGaussians`; dependencies are isolated in `work/4dgs-env`.
The patch adds this loader, replaces MMCV with MMEngine, fixes data-loader epoch
reset and checkpoint loading, enables eight loader workers, raises the
primitive cap to 800,000, preserves exported video FPS, and adds a CUDA compiler
compatibility include.

Recorded results and limitations are in [experiment notes](experiments.md).
