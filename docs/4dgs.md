# Reconstruct the action as 4DGS

Train a dynamic Gaussian model from synchronized 4DAnyone videos. Start with
[the included example](example.md), then run these steps from the repository
root unless a command explicitly changes directory.

The default uses **all cameras for training**, with no evaluation split. The
playback camera is independent of training. Defaults come from the best-tested
configuration on the example clip: 30,000 fine iterations after 3,000 coarse
iterations. This is an experimental reconstruction workflow; generated-view
inconsistencies can produce blur and artifacts.

## Install once

Tested on Linux with an RTX 6000 Ada (48 GB), Python 3.11, PyTorch 2.8 and the
CUDA 12.8 toolkit. This is a tested configuration, not a minimum-VRAM claim.
The toolkit/compiler is needed to build extensions, not just the NVIDIA driver.

```bash
conda activate 4danyone
python -m venv work/4dgs-env
work/4dgs-env/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
work/4dgs-env/bin/python -m pip install -r integrations/4dgaussians/requirements.txt

git clone --recursive https://github.com/hustvl/4DGaussians work/4DGaussians
git -C work/4DGaussians checkout 843d5ac636c37e4b611242287754f3d4ed150144
git -C work/4DGaussians submodule update --init --recursive
python integrations/4dgaussians/patch_checkout.py work/4DGaussians

CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 \
  work/4dgs-env/bin/python -m pip install --no-build-isolation \
  ./work/4DGaussians/submodules/depth-diff-gaussian-rasterization \
  ./work/4DGaussians/submodules/simple-knn
```

`8.9` targets the Ada GPU used here. Change the architecture and toolkit path
for other hardware. Installation does not replace the 4DAnyone or Rerun
environments. Skip this section if this checkout is already set up.

## Export the sequence

Use the **4danyone** environment for segmentation and body-model initialization:

```bash
conda run --no-capture-output -n 4danyone python scripts/export_4dgs.py \
  data/fdanyone/action --output outputs/4dgs/action-data
```

The exporter uses all available cameras and all 121 timestamps, at original
resolution. It writes per-camera RGBA PNGs, `seed.npz` with up to 200,000 sampled body
points, and `fdanyone.json` with camera calibration and timing. The loader
composites the foreground onto black. No COLMAP estimation or Nerfstudio is needed.
Choose a fresh output folder: an existing export is not overwritten.

For a smaller experiment, use `--scale 0.5` and optionally
`--train-views 0 4 8 12`. To choose the rendered camera, add `--video-camera 0`;
the default is the first training camera. Its view is a training view, not a
held-out quality estimate. Export time and storage depend on the clip and hardware;
the previous full-resolution 48-camera export took about 22 minutes and 4.1 GB.

Validate the export:

```bash
work/4dgs-env/bin/python integrations/4dgaussians/validate_export.py outputs/4dgs/action-data
```

## Train

```bash
python scripts/train_4dgs.py \
  --data outputs/4dgs/action-data --output outputs/4dgs/action-model
```

The launcher automatically uses `work/4dgs-env`. To change the schedule:

```bash
python scripts/train_4dgs.py \
  --data outputs/4dgs/action-data --output outputs/4dgs/action-model-20k \
  --iterations 20000 --coarse-iterations 3000 --batch-size 2
```

`--iterations` is the fine-stage count, in addition to the coarse stage. The
launcher adjusts the position-learning-rate horizon and caps densification at
the lesser of 20,000 and the penultimate fine iteration. Add `--dry-run` to
inspect resolved settings without writing files or starting training.
Use a new model directory for each experiment. Defaults live in
`integrations/4dgaussians/default.py`; advanced changes can be made there before
starting a new run.

Each model saves `config.py` with the selected overrides, `run.json` with paths,
revision and command, and upstream `cfg_args`. Keep these alongside the model:
rendering must use the same deformation-network architecture. The historical
best run took about 69 minutes for 47 training cameras on the tested machine;
that is an observation, not a runtime guarantee for your data.

## Render a video

```bash
python scripts/render_4dgs.py --model outputs/4dgs/action-model
```

This renders the saved playback camera at the exported FPS through the full action, without rendering
every training camera. Output:
`outputs/4dgs/action-model/video/ours_30000/video_rgb.mp4`.
For a 20,000-iteration run the directory is `ours_20000`.

## View interactively

The SIBR remote client sends camera requests to a Python server that renders the
animated model. Build/install the client once using the
[viewer setup guide](4dgs-viewer.md).

Terminal 1, from this repository root:

```bash
python scripts/view_4dgs.py --model outputs/4dgs/action-model --port 6019
```

The launcher prepares SIBR camera files and starts the server. In Terminal 2,
start `SIBR_remoteGaussian_app` with port 6019 and the model directory as `--path`.
The client/server must both remain running. Add `--time 0.35` to the server
command to freeze the example at 1.4 seconds; otherwise it loops the action.

## What to keep

The dynamic model is the **entire** `point_cloud/iteration_30000/` directory,
including `point_cloud.ply` and deformation weights. A PLY alone is not an
animated model. Keep the configuration files and dataset needed by this loader
too. `chkpnt_fine_30000.pth` is a training checkpoint, not a portable viewer file.
The rendered MP4 is a conventional video from one viewpoint.

## Optional evaluation

Reserve cameras during export with `--test-views 40` (for a 48-camera rig).
All remaining cameras train by default. Pass `--evaluate` to `train_4dgs.py`
to evaluate at the final iteration. Evaluation is off otherwise. Earlier
47-train/1-test exports remain compatible.

## Troubleshooting

- **Missing extension or environment:** finish installation, including both CUDA builds.
- **CUDA build mismatch:** check `nvcc --version`, toolkit path and GPU architecture.
- **Out of memory:** try `--batch-size 1`; export at `--scale 0.5` for a smaller dataset.
- **Output already exists:** use a new dataset/model path; reopening/rendering is separate.
- **Port in use:** choose another `--port` for both server and client.
- **Old model lacks saved settings:** render/view accept `--data`, `--config` and
  `--iteration` explicitly, e.g. `--config integrations/4dgaussians/quality3.py`.

Historical measurements and tuning notes are in [experiments.md](../integrations/4dgaussians/experiments.md).
