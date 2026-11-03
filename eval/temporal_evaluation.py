#!/usr/bin/env python3
# Copyright 2025 The EgoSurg Authors.
# SPDX-License-Identifier: Apache-2.0
"""Temporal stability metrics for Table 3B.

For each 10-frame clip, score consecutive rendered frames (nine pairs) on each
held-out camera, then average cameras. The per-clip means are
``summary_mean.csv``. ``eval/collect_temporal.py`` averages those files over
surgeries 1-6 and phases 01-05, which is Table 3B.

Expected renders, written by the trainer:

    {dataset}/{surgery}/{phase}_01/{frame}/gsplat_results/{method}/renders/

``pointmap`` uses ``val_step0_000{cam}.png``. The 3DGS methods use
``val_step1999_000{cam}.png``. Images are side-by-side; the right half is the
prediction. The captures themselves are not in this repository.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

try:
    import lpips
except ImportError as exc:  # pragma: no cover
    raise ImportError("temporal evaluation needs the lpips package") from exc

_LOSS_FN = None

METHODS = ("pointmap", "3DGS_random", "3DGS_rgpd", "3DGS_diffusion")
DEFAULT_SURGERIES = ["1", "2", "3", "4", "5", "6"]
DEFAULT_PHASES = ["01", "02", "03", "04", "05"]
DEFAULT_FRAMES = [f"{i:06d}" for i in range(10)]
DEFAULT_CAMERAS = ["cam0", "cam1", "cam2", "cam3"]


def get_lpips():
    global _LOSS_FN
    if _LOSS_FN is None:
        with contextlib.redirect_stdout(io.StringIO()):
            loss_fn = lpips.LPIPS(net="alex")
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()
        _LOSS_FN = loss_fn
    return _LOSS_FN


def crop_pred(img: np.ndarray) -> np.ndarray:
    """Right half of a side-by-side render (3840x1080 -> 1920x1080)."""
    height, width, _ = img.shape
    return img[:, width // 2 :, :]


def compute_metrics(imgs: list[np.ndarray]) -> dict[str, float]:
    """Mean PSNR, SSIM, LPIPS, and mean-absolute flicker over consecutive pairs."""
    temporal_psnr, temporal_ssim, temporal_lpips, flicker_vals = [], [], [], []
    loss_fn = get_lpips()
    for i in range(len(imgs) - 1):
        img1 = imgs[i].astype(np.float32) / 255.0
        img2 = imgs[i + 1].astype(np.float32) / 255.0
        temporal_psnr.append(peak_signal_noise_ratio(img1, img2, data_range=1.0))
        temporal_ssim.append(
            structural_similarity(img1, img2, channel_axis=2, data_range=1.0)
        )
        t1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        t2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        if torch.cuda.is_available():
            t1, t2 = t1.cuda(), t2.cuda()
        temporal_lpips.append(loss_fn(t1, t2).item())
        flicker_vals.append(float(np.mean(np.abs(img1 - img2))))
    return {
        "PSNR": float(np.mean(temporal_psnr)),
        "SSIM": float(np.mean(temporal_ssim)),
        "LPIPS": float(np.mean(temporal_lpips)),
        "Flicker": float(np.mean(flicker_vals)),
    }


def render_name(method: str, camera_index: int) -> str:
    step = "0" if method == "pointmap" else "1999"
    return f"val_step{step}_{camera_index:04d}.png"


def load_rgb(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def evaluate_method(
    cam_paths: dict[str, list[Path]],
    output_root: Path,
    method: str,
    save_gifs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_out = output_root / method
    method_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cam, paths in cam_paths.items():
        imgs = []
        for path in paths:
            img = load_rgb(path)
            if img is None:
                continue
            imgs.append(crop_pred(img))
        if len(imgs) < 2:
            continue
        metrics = compute_metrics(imgs)
        metrics["camera"] = cam
        metrics["method"] = method
        rows.append(metrics)
        if save_gifs and len(imgs) >= 2:
            imageio.mimsave(method_out / f"{method}_{cam}.gif", imgs, fps=5)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()
    df.to_csv(method_out / f"{method}_metrics.csv", index=False)
    mean_df = (
        df.drop(columns=["camera"]).groupby("method").mean(numeric_only=True).reset_index()
    )
    mean_df.to_csv(method_out / f"{method}_metrics_mean.csv", index=False)
    return df, mean_df


def evaluate_clip(
    dataset_dir: Path,
    surgery: str,
    phase: str,
    frames: list[str],
    cameras: list[str],
    save_gifs: bool,
) -> None:
    clip = dataset_dir / surgery / f"{phase}_01"
    output_root = clip / "temporal_evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    per_camera, per_method = [], []
    for method in METHODS:
        cam_paths = {cam: [] for cam in cameras}
        for frame in frames:
            render_dir = clip / frame / "gsplat_results" / method / "renders"
            for index, cam in enumerate(cameras):
                cam_paths[cam].append(render_dir / render_name(method, index))
        df, mean_df = evaluate_method(cam_paths, output_root, method, save_gifs)
        if not df.empty:
            per_camera.append(df)
        if not mean_df.empty:
            per_method.append(mean_df)
    summary = pd.concat(per_camera, ignore_index=True) if per_camera else pd.DataFrame()
    summary_mean = pd.concat(per_method, ignore_index=True) if per_method else pd.DataFrame()
    summary.to_csv(output_root / "summary.csv", index=False)
    summary_mean.to_csv(output_root / "summary_mean.csv", index=False)
    print(f"{surgery}/{phase}_01 -> {output_root / 'summary_mean.csv'}")
    if not summary_mean.empty:
        print(summary_mean.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--surgeries", nargs="+", default=DEFAULT_SURGERIES)
    parser.add_argument("--phases", nargs="+", default=DEFAULT_PHASES)
    parser.add_argument("--frames", nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument(
        "--save-gifs",
        action="store_true",
        help="Also write a GIF per camera. The Table 3B CSVs do not need this.",
    )
    args = parser.parse_args()
    for surgery in tqdm(args.surgeries, desc="Surgeries"):
        for phase in args.phases:
            evaluate_clip(
                args.dataset_dir,
                surgery,
                phase,
                args.frames,
                args.cameras,
                args.save_gifs,
            )


if __name__ == "__main__":
    main()
