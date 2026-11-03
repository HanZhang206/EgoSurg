# Evaluation

## Near-field metrics

Training holds out every camera whose name contains `_R` (the right eye of each
stereo pair). PSNR, SSIM, and LPIPS are computed on those held-out views at the
final step. There is no extra crop.

## Egocentric metrics

`eval/pov_metrics.py` compares a rendered handheld view with `pov/pov.jpg`.

- PSNR: MSE over pixels where `visible_mask.png` is nonzero, data range 255.
- SSIM: full-image SSIM map, then the mean of the map inside the mask.
  `--gaussian-weights` matches the setting used for the paper's egocentric table.
- LPIPS (Alex): spatial LPIPS weighted by the downsampled mask when the backend
  returns a map; otherwise masked-out pixels are replaced with the reference
  before the scalar LPIPS call.

The mask comes from the manual pose alignment. This repository does not
re-run that alignment. Supply the mask with the capture.

## Reference table

`eval/reference/near_field.csv` holds per-reconstruction near-field scores
(phase, method, psnr, ssim, lpips). Case ids and site labels are not in the
file. Row order is shuffled.

Method names in the file:

| CSV method | Paper name |
| --- | --- |
| pointmap | Depth reprojection |
| 3DGS_random | Image-only 3DGS |
| 3DGS_rgpd | EgoSurg without diffusion |
| 3DGS_diffusion | EgoSurg |

On the four-case subset that the paper uses for Table 3A and for the phase rows
of Table 2, the unweighted means match the reported PSNR / SSIM / LPIPS to the
printed precision (for example EgoSurg 26.8 dB, SSIM 0.895, LPIPS 0.143, and
phase means 26.9, 26.8, 26.4, 26.8, 27.2). That subset is not marked in the
CSV, so those site- and phase-conditioned rows cannot be recomputed from the
released file alone.

## Temporal stability (Table 3B)

`eval/temporal_evaluation.py` scores consecutive rendered frames. For each
surgery and phase it loads frames `000000` through `000009`, crops the right
half of the side-by-side validation render, and averages PSNR, SSIM, LPIPS,
and mean-absolute flicker over the nine adjacent pairs and the four held-out
cameras. One clip writes `temporal_evaluation/summary_mean.csv`.

`eval/collect_temporal.py` takes the mean and sample standard deviation
(ddof=1) of those clip rows. The default is surgeries `1` through `6` and
phases `01` through `05` (30 clips). That is Table 3B. Dropping surgeries 5
and 6 changes EgoSurg t-PSNR from 31.7 to about 31.3, so the script exits if
any requested CSV is missing.

| CSV method | Paper name | Table 3B |
| --- | --- | --- |
| pointmap | Depth reprojection | 16.8 ± 0.8, .341 ± .031, .272 ± .008, .073 ± .009 |
| 3DGS_random | Image-only 3DGS | 27.9 ± 0.9, .883 ± .020, .134 ± .009, .022 ± .003 |
| 3DGS_rgpd | EgoSurg w/o diffusion | 28.7 ± 0.7, .897 ± .014, .085 ± .009, .018 ± .002 |
| 3DGS_diffusion | EgoSurg | 31.7 ± 0.7, .941 ± .009, .050 ± .005, .013 ± .001 |

Columns are t-PSNR, t-SSIM, t-LPIPS, and flicker. The renders are not in this
repository. `pointmap` reads `val_step0_000{i}.png`. The other methods read
`val_step1999_000{i}.png`.

## Commands

```bash
python eval/pov_metrics.py \
  --eval-dir outputs/t001/pov_render \
  --ref assets/egosurg_demo/t001/pov/pov.jpg \
  --mask assets/egosurg_demo/t001/pov/visible_mask.png \
  --gaussian-weights \
  --save-csv outputs/t001_pov_metrics.csv

python eval/cross_view.py \
  --reference-dir assets/egosurg_demo/t001/images \
  --baseline-dir outputs/t001

python eval/temporal_evaluation.py --dataset-dir /path/to/dataset
python eval/collect_temporal.py --dataset-dir /path/to/dataset
```

`cross_view.py` expects the any-point render folders produced by the trainer's
trajectory evaluation. It is the trajectory / CLIP script used for the
supplementary comparisons, not Table 3B. Table 3B is the pair of commands
above.
