# Copyright 2025 The EgoSurg Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Paper metrics (arXiv 2510.04802):
# Near-field PSNR/SSIM/LPIPS compare a rendered right-eye view with the
# held-out right-eye photograph. There is no additional crop.
# Egocentric PSNR/SSIM/LPIPS use the handheld frame and a binary visible
# mask from the manual pose alignment. Pixels outside the mask are excluded
# from PSNR, averaged out of the SSIM map, and either spatially weighted
# or replaced with the reference before LPIPS. Pass --gaussian-weights to
# match the SSIM setting used for the egocentric table.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP 参考集（anchor）相似度：仅计算图像-图像 CLIP 分数 + 可视化每个 baseline 的最相关配对
并新增：为每个 baseline 计算**时域一致性**指标：tLPIPS↓、tSSIM↑、tPSNR↑、Flicker↓。

输出：
- 每个 baseline：
  - `<baseline>_clipref_perframe.csv` 逐帧与 anchor 的相似度（max / mean / topK_mean）、对应 best_anchor 名
  - `<baseline>_temporal.csv` 连续帧的时域指标（tLPIPS / tSSIM / tPSNR / Flicker）
  - `<baseline>_bestmatch.jpg` 最相关配对可视化（baseline 最像的一帧 vs 对应 anchor）
- `summary_clipref_all.csv`：跨 baseline 的汇总（含 CLIP & 时域指标的 mean/median/p10/p90 与 best 配对信息）

依赖：
    pip install open_clip_torch pillow tqdm pandas torch pyiqa scikit-image opencv-contrib-python
    # 可选：若你想用 torchvision RAFT 光流（更准）： pip install torchvision --upgrade

说明：
- CLIP：open_clip ViT-B/32（laion2b_s34b_b79k）为默认模型；图像特征 L2 归一化，余弦相似度=点积。
- 光流：优先 torchvision RAFT；否则回退 OpenCV TV-L1 / Farneback；再否则零流（仅用于兜底）。
- 时域指标：将 t+1 帧**光流对齐到 t 帧坐标**后再算 LPIPS / SSIM / PSNR；Flicker 用对齐后高频差（更能感受闪烁）。
"""
import argparse
# from __future__ import annotations
from dataclasses import field
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
import torch.nn.functional as F

# ----------------------- 依赖可用性 -----------------------
try:
    import open_clip
    _HAS_OPENCLIP = True
except Exception:
    _HAS_OPENCLIP = False

try:
    import pyiqa
    _HAS_PYIQA = True
except Exception:
    _HAS_PYIQA = False

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.filters import gaussian
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import cv2
    _HAS_OPENCV = True
except Exception:
    _HAS_OPENCV = False

try:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    _HAS_TORCHVISION_RAFT = True
except Exception:
    _HAS_TORCHVISION_RAFT = False

# ----------------------- I/O -----------------------
from collections import defaultdict

def _strip_r_suffix(baseline_name: str) -> str:
    """
    Convert 'baseline_pointmap_r1.5' -> 'baseline_pointmap'
    by removing the trailing `_r{r}`.
    """
    if '_r' in baseline_name:
        return baseline_name.rsplit('_r', 1)[0]
    return baseline_name

def summarize_across_r(result_dirs, out_csv: Path = Path('summary_all_r.csv'),
                       clip_col_choice: str = 'CLIPRef_max_mean'):
    """
    Aggregate metrics across multiple radii r.

    Parameters
    ----------
    result_dirs : List[Path] or List[str]
        List like [Path('eval_results1'), Path('eval_results1.5'), Path('eval_results2')].
    out_csv : Path
        Where to write the final summary CSV.
    clip_col_choice : str
        Which CLIP summary column to use from `summary_clipref_all.csv`.
        Common choices produced by evaluate_clip_only():
            - 'CLIPRef_max_mean'
            - 'CLIPRef_mean_mean'
            - 'CLIPRef_top2_mean_mean'   # since your topk=2
    """
    rows = []

    for d in map(Path, result_dirs):
        clip_csv = d / 'summary_clipref_all.csv'
        tmp_csv = d / 'summary_temporal_all.csv'
        if not clip_csv.exists() and not tmp_csv.exists():
            warnings.warn(f"Skip {d}: no summary CSVs found.")
            continue

        # Load with safe fallbacks
        clip_df = pd.read_csv(clip_csv) if clip_csv.exists() else pd.DataFrame()
        tmp_df  = pd.read_csv(tmp_csv)  if tmp_csv.exists()  else pd.DataFrame()

        # Normalize baseline names to collapse _r{r}
        if len(clip_df):
            clip_df = clip_df.rename(columns={'baseline': 'baseline_raw'})
            clip_df['baseline'] = clip_df['baseline_raw'].apply(_strip_r_suffix)
        if len(tmp_df):
            tmp_df = tmp_df.rename(columns={'baseline': 'baseline_raw'})
            tmp_df['baseline'] = tmp_df['baseline_raw'].apply(_strip_r_suffix)

        # Select & rename temporal columns we need
        # (These come from temporal_metrics_for_folder() as ..._mean)
        # tLPIPS_mean, tSSIM_mean, tPSNR_mean, Flicker_mean
        keep_tmp_cols = ['baseline', 'tLPIPS_mean', 'tSSIM_mean', 'tPSNR_mean', 'Flicker_mean']
        tmp_keep = tmp_df[keep_tmp_cols].copy() if len(tmp_df) else pd.DataFrame(columns=keep_tmp_cols)

        # Select CLIP column to use
        # Options likely present (check your generated CSV):
        #   CLIPRef_max_mean, CLIPRef_mean_mean, CLIPRef_top2_mean
        clip_cols = ['baseline', clip_col_choice]
        if len(clip_df) and clip_col_choice not in clip_df.columns:
            raise KeyError(
                f"Requested CLIP column '{clip_col_choice}' not found in {clip_csv}.\n"
                f"Available: {', '.join([c for c in clip_df.columns if c.startswith('CLIPRef_')])}"
            )
        clip_keep = clip_df[clip_cols].copy() if len(clip_df) else pd.DataFrame(columns=clip_cols)

        # Merge per dir (inner join on 'baseline' so we only keep those present)
        merged = None
        if len(tmp_keep) and len(clip_keep):
            merged = pd.merge(tmp_keep, clip_keep, on='baseline', how='outer')
        elif len(tmp_keep):
            merged = tmp_keep.copy()
        elif len(clip_keep):
            merged = clip_keep.copy()

        if merged is not None and len(merged):
            merged['source_dir'] = str(d)
            rows.append(merged)

    if not rows:
        print("No rows collected from result directories; nothing to summarize.")
        return pd.DataFrame()

    all_df = pd.concat(rows, ignore_index=True)

    # Group by baseline (collapsed across r) and take the mean across dirs (r values)
    grouped = (
        all_df
        .groupby('baseline', as_index=False)
        .agg({
            'tLPIPS_mean': 'mean',
            'tSSIM_mean': 'mean',
            'tPSNR_mean': 'mean',
            'Flicker_mean': 'mean',
            clip_col_choice: 'mean'
        })
    )

    # Final pretty column names matching your requested header
    # "baseline_name,mean_LPIPS...,mean_SSIM...,mean_PSNR...,mean_Flicker...,mean_CLIP..."
    final = grouped.rename(columns={
        'baseline': 'baseline_name',
        'tLPIPS_mean': 'mean_LPIPS_all_r',
        'tSSIM_mean': 'mean_SSIM_all_r',
        'tPSNR_mean': 'mean_PSNR_all_r',
        'Flicker_mean': 'mean_Flicker_all_r',
        clip_col_choice: 'mean_CLIP_all_r'
    })

    final.to_csv(out_csv, index=False)
    print(f"Saved overall summary: {out_csv}")
    print(final)
    return final

def list_images(folder: Path) -> List[Path]:
    # breakpoint()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
def load_tensor_image(path: Path, device: torch.device, max_side: Optional[int] = None) -> torch.Tensor:
    """读图为 [1,3,H,W]，0..1 float32；可选限制最大边。"""
    img = Image.open(path).convert("RGB")
    if max_side is not None:
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return ten

def to_numpy_image(t: torch.Tensor) -> np.ndarray:
    return t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
# ----------------------- CLIP 提取 -----------------------
def build_optical_flow(device: torch.device):
    """返回 (flow_fn, flow_name)。flow_fn(img0,img1)-> flow01(H,W,2)，将 img1 反向对齐到 img0。"""
    # 1) Torchvision RAFT（优先）
    if _HAS_TORCHVISION_RAFT:
        try:
            weights = Raft_Large_Weights.C_T_V2
            raft = raft_large(weights=weights).to(device).eval()

            @torch.inference_mode()
            def flow_fn(img0: torch.Tensor, img1: torch.Tensor) -> np.ndarray:
                def _prep(x):
                    return (x * 2.0 - 1.0)
                flow12 = raft(_prep(img0), _prep(img1))[0]  # [1,2,H,W] pixels from 0->1
                return flow12[0].permute(1, 2, 0).detach().cpu().numpy()

            return flow_fn, "RAFT(torchvision)"
        except Exception:
            pass

    # 2) OpenCV TV-L1（需 opencv-contrib-python）
    if _HAS_OPENCV and hasattr(cv2, 'optflow') and hasattr(cv2.optflow, 'DualTVL1OpticalFlow_create'):
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        def flow_fn(img0: torch.Tensor, img1: torch.Tensor) -> np.ndarray:
            a = (to_numpy_image(img0) * 255.0).astype(np.uint8)
            b = (to_numpy_image(img1) * 255.0).astype(np.uint8)
            a_g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            b_g = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
            flow = tvl1.calc(a_g, b_g, None)
            return flow.astype(np.float32)
        return flow_fn, "TV-L1(OpenCV)"

    # 3) Farneback（兜底）
    if _HAS_OPENCV:
        def flow_fn(img0: torch.Tensor, img1: torch.Tensor) -> np.ndarray:
            a = (to_numpy_image(img0) * 255.0).astype(np.uint8)
            b = (to_numpy_image(img1) * 255.0).astype(np.uint8)
            a_g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            b_g = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(a_g, b_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            return flow.astype(np.float32)
        return flow_fn, "Farneback(OpenCV)"

    # 4) 零流（不对齐，警告）
    def _noop_flow(img0: torch.Tensor, img1: torch.Tensor) -> np.ndarray:
        h, w = img0.shape[-2:]
        return np.zeros((h, w, 2), dtype=np.float32)
    warnings.warn("未找到可用光流实现，将使用零流（不对齐）——时域指标可能不可靠。")
    return _noop_flow, "None(zero-flow)"

def warp_img1_to_img0(img0: torch.Tensor, img1: torch.Tensor, flow01: np.ndarray) -> torch.Tensor:
    h, w = img0.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=img0.device),
                             torch.arange(w, device=img0.device), indexing='ij')
    grid_x = (xx + torch.from_numpy(flow01[..., 0]).to(img0.device))
    grid_y = (yy + torch.from_numpy(flow01[..., 1]).to(img0.device))
    grid_x = 2.0 * (grid_x / max(w - 1, 1)) - 1.0
    grid_y = 2.0 * (grid_y / max(h - 1, 1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    warped = F.grid_sample(img1, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped

def temporal_metrics_for_folder(folder: Path, device: torch.device, max_side: Optional[int],
                                flow_impl=None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """对 folder 连续帧计算：tLPIPS（对齐后）、tSSIM、tPSNR、Flicker。返回 per-pair DF 与摘要。"""
    paths = list_images(folder)
    if len(paths) < 2:
        warnings.warn(f"{folder} 少于2张，跳过时域指标。")
        return pd.DataFrame(), {}

    # 准备光流与 LPIPS
    if flow_impl is None:
        flow_fn, flow_name = build_optical_flow(device)
    else:
        flow_fn, flow_name = flow_impl

    if _HAS_PYIQA:
        try:
            lpips_metric = pyiqa.create_metric('lpips', device=device)
        except Exception:
            lpips_metric = None
            warnings.warn("创建 LPIPS 失败，tLPIPS 将为 NaN。")
    else:
        lpips_metric = None

    rows = []
    # 逐对计算
    imgs = [load_tensor_image(p, device, max_side=max_side) for p in paths]

    for i in tqdm(range(len(paths)-1), desc=f"Temporal | {folder.name}"):
        p0, p1 = paths[i], paths[i+1]
        img0, img1 = imgs[i], imgs[i+1]
        flow01 = flow_fn(img0, img1)
        img1_w = warp_img1_to_img0(img0, img1, flow01)

        # tLPIPS
        if lpips_metric is not None:
            with torch.inference_mode():
                try:
                    t_lpips = float(lpips_metric(img0, img1_w).mean().detach().cpu().item())
                except Exception:
                    t_lpips = float('nan')
        else:
            t_lpips = float('nan')

        # tSSIM / tPSNR
        if _HAS_SKIMAGE:
            a = to_numpy_image(img0)
            b = to_numpy_image(img1_w)
            try:
                t_ssim = float(ssim(a, b, channel_axis=2, data_range=1.0))
            except TypeError:
                t_ssim = float(ssim(a, b, multichannel=True, data_range=1.0))
            t_psnr = float(psnr(a, b, data_range=1.0))
            # Flicker：高频差
            a_hf = a - gaussian(a, sigma=1, channel_axis=2)
            b_hf = b - gaussian(b, sigma=1, channel_axis=2)
            flicker = float(np.mean(np.abs(a_hf - b_hf)))
        else:
            # 退路：未安装 skimage
            a = to_numpy_image(img0)
            b = to_numpy_image(img1_w)
            t_ssim = float('nan')
            t_psnr = float('nan')
            flicker = float(np.mean(np.abs(a - b)))

        rows.append({
            'pair': f"{p0.name} -> {p1.name}",
            'tLPIPS': t_lpips,
            'tSSIM': t_ssim,
            'tPSNR': t_psnr,
            'Flicker': flicker,
        })

    df = pd.DataFrame(rows)

    def _agg(col):
        vals = df[col].astype(float).values
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return {'mean': float('nan'), 'median': float('nan'), 'p10': float('nan'), 'p90': float('nan')}
        return {
            'mean': float(np.mean(vals)),
            # 'median': float(np.median(vals)),
            # 'p10': float(np.percentile(vals, 10)),
            # 'p90': float(np.percentile(vals, 90)),
        }

    summary = {}
    for k in ['tLPIPS', 'tSSIM', 'tPSNR', 'Flicker']:
        stats = _agg(k)
        for kk, vv in stats.items():
            summary[f'{k}_{kk}'] = vv

    return df, summary
class CLIPEncoder:
    def __init__(self, device: Optional[str] = None,
                 model_name: str = 'ViT-B-32', pretrained: str = 'laion2b_s34b_b79k'):
        if not _HAS_OPENCLIP:
            raise ImportError("open_clip_torch 未安装，请先 pip install open_clip_torch")
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def encode_images(self, paths: List[Path], batch_size: int = 32) -> torch.Tensor:
        feats = []
        for i in tqdm(range(0, len(paths), batch_size), desc='Encode (images)'):
            batch_paths = paths[i:i+batch_size]
            ims = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                ims.append(self.preprocess(img))
            if len(ims) == 0:
                continue
            x = torch.stack(ims, 0).to(self.device)
            z = self.model.encode_image(x)
            # 归一化
            z = z / z.norm(dim=-1, keepdim=True)
            feats.append(z)
        if len(feats) == 0:
            return torch.empty(0, 512, device=self.device)
        return torch.cat(feats, dim=0)
# ----------------------- 评估 -----------------------
def clip_ref_scores_for_folder(folder: Path,
                               anchor_feats: torch.Tensor,
                               encoder: CLIPEncoder,
                               topk: int = 5,
                               batch_size: int = 32) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    对 folder 内所有图片，与 anchor_feats 做相似度：返回逐帧 DataFrame 与 summary。
    """
    frames = list_images(folder)
    if not frames:
        warnings.warn(f"无图片：{folder}")
        return pd.DataFrame(), {"count": 0}

    # 提取本目录所有帧的特征
    with torch.inference_mode():
        feats = encoder.encode_images(frames, batch_size=batch_size)  # [N, D], 已归一化
    if feats.numel() == 0 or anchor_feats.numel() == 0:
        warnings.warn("特征为空，检查输入路径或依赖")
        return pd.DataFrame(), {"count": 0}

    # 相似度：余弦 -> 点积
    sims = feats @ anchor_feats.T  # [N, M]
    topk = int(min(topk, anchor_feats.shape[0])) if anchor_feats.shape[0] > 0 else 0

    per_rows = []
    for i, p in enumerate(frames):
        s = sims[i]  # [M]
        s_np = s.detach().float().cpu().numpy()
        max_v = float(s_np.max())
        mean_v = float(s_np.mean())
        if topk > 0:
            topk_mean_v = float(np.sort(s_np)[-topk:].mean())
        else:
            topk_mean_v = float('nan')
        per_rows.append({
            'frame': p.name,
            'CLIPRef_max': max_v,
            'CLIPRef_mean': mean_v,
            f'CLIPRef_top{topk}_mean': topk_mean_v,
        })

    per_df = pd.DataFrame(per_rows)

    # 汇总
    def agg(col: str) -> Dict[str, float]:
        if col not in per_df:
            return {f'{col}_mean': float('nan'), f'{col}_median': float('nan')}
        vals = per_df[col].astype(float).values
        return {
            f'{col}_mean': float(np.mean(vals)),
            f'{col}_median': float(np.median(vals)),
        }

    summary = {'count': len(frames)}
    for col in ['CLIPRef_max', 'CLIPRef_mean', f'CLIPRef_top{topk}_mean']:
        summary.update(agg(col))

    return per_df, summary
# ----------------------- 主流程 -----------------------
def evaluate_clip_only(baselines: Dict[str, Path], anchor_dir: Path,
                       out_dir: Path = Path('./_clipref_outputs'),
                       topk: int = 5, batch_size: int = 32, device: Optional[str] = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 载入 CLIP 编码器
    enc = CLIPEncoder(device=device)

    # 2) 预计算 anchor 特征
    anchor_imgs = list_images(anchor_dir)
    if not anchor_imgs:
        raise FileNotFoundError(f"anchor 目录无图片：{anchor_dir}")
    print(f"Anchor images: {len(anchor_imgs)} @ {anchor_dir}")
    with torch.inference_mode():
        anchor_feats = enc.encode_images(anchor_imgs, batch_size=batch_size)  # [M, D], 已归一化

    # 3) 各 baseline
    all_sum = []
    for name, folder in baselines.items():
        print(f"\n==> Baseline: {name} | {folder}")
        per_df, summary = clip_ref_scores_for_folder(folder, anchor_feats, enc, topk=topk, batch_size=batch_size)
        if len(per_df) == 0:
            continue
        per_csv = out_dir / f"{name}_clipref_perframe.csv"
        per_df.to_csv(per_csv, index=False)
        print(f"Saved per-frame: {per_csv}")

        row = {'baseline': name, 'frames_dir': str(folder)}
        row.update(summary)
        all_sum.append(row)

    if all_sum:
        sum_df = pd.DataFrame(all_sum)
        sum_csv = out_dir / 'summary_clipref_all.csv'
        sum_df.to_csv(sum_csv, index=False)
        print(f"Saved summary: {sum_csv}\n")
        print(sum_df)
    else:
        print("No valid baselines.")
def evaluate_temporal_only(baselines: Dict[str, Path],
                           out_dir: Path = Path('./_clipref_outputs'),
                           device: Optional[str] = None,
                           max_side: Optional[int] = None) -> pd.DataFrame:
    """仅计算时域指标（tLPIPS/tSSIM/tPSNR/Flicker），返回汇总 DataFrame。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    flow_impl = build_optical_flow(device)

    all_sum = []
    for name, folder in baselines.items():
        print(f"Temporal | Baseline: {name} | {folder}")
        t_df, t_sum = temporal_metrics_for_folder(folder, device, max_side=max_side, flow_impl=flow_impl)
        if len(t_df) == 0:
            continue
        t_csv = out_dir / f"{name}_temporal.csv"
        t_df.to_csv(t_csv, index=False)
        print(f"Saved temporal: {t_csv}")
        row = {'baseline': name, 'frames_dir': str(folder)}
        row.update(t_sum)
        all_sum.append(row)

    if not all_sum:
        print("No valid baselines for temporal metrics.")
        return pd.DataFrame()

    sum_df = pd.DataFrame(all_sum)
    sum_csv = out_dir / 'summary_temporal_all.csv'
    sum_df.to_csv(sum_csv, index=False)
    print(f"Saved Temporal summary: {sum_csv}")
    print(sum_df)
    return sum_df

# ----------------------- 入口 -----------------------

def main(args):
    anchor = Path(args.reference_dir)
    for r in [1, 1.5, 2]:
        baselines = {}
        baselines[f"baseline_pointmap_r{r}"] = Path(args.baseline_dir + f"/pointmap/Anypoint_View/anypov_circle_h1m_r{r}/frames/")
        baselines[f"baseline_3DGS_random_r{r}"] = Path(args.baseline_dir + f"/3DGS_random/Anypoint_View/anypov_circle_h1m_r{r}/frames/")
        baselines[f"baseline_3DGS_rgbd_r{r}"] = Path(args.baseline_dir + f"/3DGS_rgpd/Anypoint_View/anypov_circle_h1m_r{r}/frames/")
        baselines[f"baseline_3DGS_diffusion_r{r}"] = Path(args.baseline_dir + f"/3DGS_diffusion/Anypoint_View/anypov_circle_h1m_r{r}/frames/")
        # breakpoint()
        out_dir = Path(args.baseline_dir + f"/eval_results{r}")

        # out_dir = Path('./eval_results' + str(r))
        # breakpoint()
        evaluate_clip_only(baselines, anchor, out_dir=out_dir, topk=1, batch_size=50, device="cuda")
        evaluate_temporal_only(baselines, out_dir=out_dir, device="cuda", max_side=None)   
    
    # Summary across all r values
    summarize_across_r(
        result_dirs=[Path(args.baseline_dir + f"/eval_results1"), Path(args.baseline_dir + f"/eval_results1.5"), Path(args.baseline_dir + f"/eval_results2")],
        out_csv=Path(args.baseline_dir + f"/summary_all_r.csv"),
        # choose which CLIP aggregate to average across r:
        clip_col_choice='CLIPRef_top1_mean_mean'  # or 'CLIPRef_mean_mean' / 'CLIPRef_max_mean'
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", type=str, default="assets/egosurg_demo/t001/images", help="path to the reference directory")
    parser.add_argument("--baseline_dir", type=str, default="results/", help="path to the baseline directory")
    args = parser.parse_args()
    main(args)