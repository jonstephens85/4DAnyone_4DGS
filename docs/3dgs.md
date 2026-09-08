# Reconstruct a single moment as 3DGS

A static reconstruction uses one synchronized frame from every generated view.
Start with [the example](example.md). The original
[Nerfstudio guide](nerfstudio.md) remains the reference for training with Splatfacto.

## Choose a timestamp

Frame indices start at zero. Use `round(seconds × output FPS)` for the nearest
frame, reading FPS from `data/fdanyone/<clip>/metadata.json`. For 30 FPS,
1.4 seconds is frame **42**; for 25 FPS it is frame **35**. Time is relative to
the start of the generated clip. Valid frame indices are 0–120.

## Export training inputs

From the repository root in the 4danyone environment:

```bash
conda activate 4danyone
python scripts/export_nerfstudio.py \
  --result_dir data/fdanyone/action --frame_index 42
```

This writes `data/nerfstudio/action/frame_042/` with images, masks,
`transforms.json`, and a visual-hull initialization point cloud. The script
**does not require Nerfstudio installed**. It uses the inference environment
for foreground segmentation and the existing known camera calibration.

These files are a training dataset. The initialization PLY is not a trained
Gaussian splat. To produce the model, follow the original guide's
[training instructions](nerfstudio.md#train), using `frame_042` as the dataset.
The guide also describes viewing the trained model.

## Other reconstruction software

Use `--output_dir outputs/static/action-frame42` to choose another export path.
A tool that supports the exported transforms format may import this dataset;
confirm camera and mask support in your installed version.

If you already have a [4DGS dataset export](4dgs.md#export-the-sequence), you can
also convert one timestamp to a COLMAP dataset without running camera estimation:

```bash
work/4dgs-env/bin/python integrations/4dgaussians/export_static_colmap.py \
  outputs/4dgs/action-data outputs/static/action-colmap --frame 42
```

This route expects `fdanyone.json`, per-camera RGBA images and `seed.npz` from
the dynamic exporter. It writes RGB images on black and COLMAP camera/point
files. Import that dataset into your chosen COLMAP-compatible trainer; this
command alone does not train or export a Gaussian model. A static result has
one pose. For animation, use the [4DGS workflow](4dgs.md).
