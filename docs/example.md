# Run the included example

This fork adds viewers and reconstruction tools to the original 4DAnyone pipeline.
The original installation and inference instructions in the root README are
preserved. For these additions, clone **this fork**:

```bash
git clone https://github.com/jonstephens85/4DAnyone_4DGS.git
cd 4DAnyone_4DGS
git submodule update --init third_party/GVHMR
```

Continue with the environment/dependency commands in the original
[Installation section](../README.md#installation), starting at `conda create`.
If you already cloned this fork and installed 4DAnyone, skip installation.

## Generate the views

From the repository root in the `4danyone` environment:

```bash
conda activate 4danyone
python inference.py --video_path examples/action.mp4 \
  --views_per_layer 16 --layer_pitches '[-10,15,35]'
```

This produces 48 synchronized videos in `data/fdanyone/action/videos/dense/`,
camera calibration in `data/fdanyone/action/cameras.json`, and cached motion in
`data/gvhmr/results/action/`. Required assets download on first use.
These are generated camera views, not yet a Gaussian model.

## Choose what to do next

| Goal | Guide | Result |
|---|---|---|
| Inspect videos and motion | [Rerun](rerun.md) | Interactive cameras, videos and optional 3D skeleton |
| Reconstruct one moment | [3DGS](3dgs.md) | Static Gaussian model after training |
| Reconstruct the full action | [4DGS](4dgs.md) | Gaussians and a time-dependent deformation model |

The viewers are different: Rerun inspects generation outputs; SIBR displays the
trained 4DGS model. Neither viewer is required to train a reconstruction.

For your own video, use `--video_path /path/to/my_clip.mp4` and replace `action`
with `my_clip` in subsequent paths. See upstream's
[input guidance](../README.md#custom-data) and [camera layouts](../README.md#inference).
