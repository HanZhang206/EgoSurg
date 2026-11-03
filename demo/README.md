# Demo data layout

One timestamp ships with the repository:

```
assets/egosurg_demo/
  t001/
    images/
      cam0_L.jpg
      cam0_R.jpg
      cam1_L.jpg
      cam1_R.jpg
      cam2_L.jpg
      cam2_R.jpg
      cam3_L.jpg
      cam3_R.jpg
    depths/
      cam0_L.npy
      cam1_L.npy
      cam2_L.npy
      cam3_L.npy
    transforms.json
    pov/
      extrinsics.txt      # one 4x4 camera-to-world matrix, 16 floats, row-major
      intrinsics.txt       # fx fy cx cy
      pov.jpg              # handheld ground truth
      visible_mask.png     # pixels used for egocentric metrics
```

`transforms.json` is a pinhole frame list. Each frame has `file_path`, `depth_path`
(`"None"` when that view has no depth), `width`, `height`, flattened 3x3 `K`,
and a 4x4 `c2w` matrix. Images are rectified stereo, OpenCV camera axes, metric
scale. Right-eye names end in `_R` and are held out of training.

Camera ids in this clip are `cam0` ... `cam3`. They are not hardware serial numbers.

The root README trains `3DGS_diffusion` on `assets/egosurg_demo/t001` for 2000 steps with `--opacity-reg 0.3`. Diffusion refinement still runs at step 500, with 15 views inside 0.4 m. The handheld render is then scored with `eval/pov_metrics.py --gaussian-weights` against `pov/pov.jpg` and `pov/visible_mask.png`. Table 4 in the paper is 17.8 / 0.766 / 0.406 over ten egocentric scenes. This directory is one timestamp, not that aggregate.
