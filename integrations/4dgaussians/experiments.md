# 4DGS experiment notes

Historical observations from the contributed example clip, recorded September
2026. Start with the [4DGS guide](../../docs/4dgs.md) to run your own experiment.
Datasets and models mentioned here are local artifacts, not included in this
repository. These measurements were recorded during earlier runs and have not
been rerun as part of the documentation cleanup.

## Recorded results

The best tested configuration was `quality3.py`: 30,000 fine iterations,
323,546 Gaussians, about 69 minutes on an RTX 6000 Ada. It used **47 training
cameras and held out camera 40**, unlike the new all-camera default.
The export contained 121 frames per camera at 704×1280.

| Model | Gaussians | Full PSNR | Foreground PSNR | Detail vs GT | Colour outliers |
|---|---|---|---|---|---|
| `IMG_7164_full48_model` (20k) | 60,564 | 29.16 dB | 22.42 dB | 63.1% | 0.03% |
| `IMG_7164_quality` (30k) | 238,267 | 28.16 dB | 21.00 dB | 67.1% | **7.08%** (broken) |
| **`IMG_7164_quality3` (30k)** | **323,546** | **29.24 dB** | **22.73 dB** | **74.6%** | 0.16% |
| `IMG_7164_static_model` (static, frame 42 only) | 116,914 | — | 25.83 dB¹ | 134.9%¹ | — |

¹ single frame, camera 40 only. Detail above 100% is ghosting, not real detail.

## Observations and limitations

- A larger grid and more permissive densification improved detail. Several
  settings changed together, so the individual effects are not isolated.
- Time-dependent color and opacity (`no_dshs=False`, `no_do=False`) removed
  the observed magenta-arm artifact. These settings are now in `default.py`.
- An opacity-reset experiment did not resolve that artifact.
- Full-image PSNR favored the large black background. Foreground PSNR and
  matched-frame inspection were more useful for judging the person.
- The static control showed ghosting on an unseen camera. Generated-view
  inconsistency may contribute, but these tests do not prove a quality ceiling
  or rule out further reconstruction improvements.
- Eight data-loader workers improved GPU utilization on this machine.

`evaluate.py` measures full/foreground PSNR across the sequence. Its detail
ratio and color-outlier statistic use one selected frame (default 42). The
color check targets the particular magenta artifact, not general quality.
Detail above 100% can reflect ghosting. Targets are generated 4DAnyone images,
not independent real footage.

## Run history

| Model | Config | Views / res | Gaussians | Result |
|---|---|---|---|---|
| `IMG_7164_model` *(deleted)* | `pilot.py` | 12 / 352×640 | 41,719 | Pipeline smoke test, 2k iterations |
| `IMG_7164_model_14000` *(deleted)* | `train_14000.py` | 12 / 352×640 | 58,712 | 27.18 dB upstream test PSNR |
| `IMG_7164_full48_model` | `full48.py` | 47 / 704×1280 | 60,564 | 63.1% detail; grid still downgraded |
| `IMG_7164_quality` *(deleted)* | `quality.py` | 47 / 704×1280 | 238,267 | Sharper but magenta arm (7.08% outliers) |
| `IMG_7164_quality2` *(deleted)* | `quality2.py` | 47 / 704×1280 | ~195k @20k | Stopped early; opacity reset did not help |
| `IMG_7164_quality3` | `quality3.py` | 47 / 704×1280 | 323,546 | **Best.** 74.6% detail, no artifact, 69 min |
| `IMG_7164_static_model` | vanilla 3DGS | 42 / frame 42 | 116,914 | Static control |


Historical configs live in `experiments/`. Original filenames remain as
compatibility aliases for existing commands. New runs use `scripts/train_4dgs.py`
and save their configuration in the model directory. Reproducing an old run
also requires its dataset split, resolution, seed and schedule.

## Future experiments

- Test multiple held-out camera positions instead of only camera 40.
- Compare grid capacity, densification and seed density independently.
- Investigate appearance modeling and other dynamic reconstruction methods.
- Inspect interpolated viewpoints and temporal stability.
