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
"""
Compute full-frame and masked PSNR/SSIM/LPIPS between a folder of eval images
and a single reference image, with a visible-mask. Now with argparse.

Example usage:

python masked-image-metrics-cli.py \
  --eval-dir /path/to/pov_rendered \
  --ref /path/to/pov.jpg \
  --mask /path/to/visible_mask.png \
  --lpips-net alex \
  --device auto \
  --summary-only \
  --save-csv /tmp/metrics.csv

Notes:
- "summary-only" prints only the mean row; omit it to print per-image rows too.
- LPIPS spatial weighting is used when the model exposes spatial output. Otherwise,
  a masked replacement trick is used.
"""

import os
import argparse
from PIL import Image
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

import torch
import lpips  # pip install lpips

# ---------------------
# Helpers
# ---------------------

def pick_device(arg_device: str) -> torch.device:
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_rgb(path):
    return Image.open(path).convert("RGB")


def ensure_same_size(img, ref_img):
    """Resize evaluated image to reference image size if needed."""
    if img.size != ref_img.size:
        img = img.resize(ref_img.size, Image.BICUBIC)
    return img


def pil_to_np_uint8(img):
    return np.array(img, dtype=np.uint8)


def pil_to_lpips_tensor(img, device):
    """LPIPS expects [-1,1], shape [1,3,H,W]."""
    arr = np.array(img).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    ten = ten * 2.0 - 1.0
    return ten.to(device)


def load_mask(mask_path, target_size):
    """
    读取可见掩膜并缩放到参考图大小。返回 bool mask (H, W), True=有效像素
    """
    m = Image.open(mask_path).convert("L")  # 灰度
    if m.size != target_size:
        # 用最近邻避免软化边缘
        m = m.resize(target_size, Image.NEAREST)
    m_np = np.array(m)
    mask = m_np > 0
    return mask


def masked_psnr(ref_np, eval_np, mask):
    """
    ref_np, eval_np: uint8, shape (H,W,3)
    mask: bool, shape (H,W)
    """
    if mask.sum() == 0:
        return float('nan')
    diff = (ref_np.astype(np.float32) - eval_np.astype(np.float32))  # (H,W,3)
    # 只在 mask 上计算 MSE（对三个通道一起平均）
    mse = (diff[mask].reshape(-1, 3) ** 2).mean()
    if mse == 0:
        return float('inf')
    data_range = 255.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def masked_ssim(ref_np, eval_np, mask,gaussian_weights=False):
    """
    用 skimage 的 full=True 返回 SSIM map，再对 mask 求均值
    """
    if mask.sum() == 0:
        return float('nan')
    ssim_val, ssim_map = ssim_fn(
        ref_np, eval_np, data_range=255, channel_axis=2, full=True,gaussian_weights=gaussian_weights
    )
    # ssim_map: (H,W) 浮点
    m = mask.astype(bool)
    return float(ssim_map[m].mean())


@torch.no_grad()
def masked_lpips(ref_img, eval_img, mask, lpips_model, device):
    """
    - 若 LPIPS 支持空间图：下采样 mask 做加权平均（严谨）。
    - 否则：把 mask 外的 eval 像素替换为 ref（近似，使其对 LPIPS 贡献≈0），
      然后计算一次标量 LPIPS。
    """
    if mask.sum() == 0:
        return float('nan')

    LPIPS_SPATIAL = getattr(lpips_model, 'spatial', False)

    if LPIPS_SPATIAL:
        ref_t = pil_to_lpips_tensor(ref_img, device)
        eval_t = pil_to_lpips_tensor(eval_img, device)

        # 空间图: [1,1,h,w]
        lpips_spatial = lpips_model(ref_t, eval_t).detach().cpu().numpy()[0, 0]  # (h,w)

        # 将 mask 下采样到 (h,w)，双线性->权重
        h, w = lpips_spatial.shape
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_small = mask_img.resize((w, h), Image.BILINEAR)
        mask_small = np.array(mask_small, dtype=np.float32) / 255.0

        if mask_small.sum() <= 1e-6:
            return float('nan')

        val = float((lpips_spatial * mask_small).sum() / (mask_small.sum()))
        return val

    else:
        # 近似方案：把掩膜外像素设成与 GT 相同，使其贡献≈0
        ref_arr  = np.array(ref_img, dtype=np.uint8)
        eval_arr = np.array(eval_img, dtype=np.uint8)

        # Create 3D mask for RGB channels
        m3 = np.broadcast_to(mask[..., None], ref_arr.shape)  # (H,W,3)
        eval_mod = eval_arr.copy()
        eval_mod[~m3] = ref_arr[~m3]

        ref_t  = pil_to_lpips_tensor(Image.fromarray(ref_arr), device)
        eval_t = pil_to_lpips_tensor(Image.fromarray(eval_mod), device)

        return float(lpips_model(ref_t, eval_t).item())


# ---------------------
# Main
# ---------------------

def main():
    parser = argparse.ArgumentParser(description="Compute full-frame and masked PSNR/SSIM/LPIPS.")
    parser.add_argument("--eval-dir", required=True, help="Directory of evaluated images (png/jpg/jpeg)")
    parser.add_argument("--ref", required=True, help="Path to reference RGB image")
    parser.add_argument("--mask", required=True, help="Path to visible mask (L/PNG)")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"], help="Backbone for LPIPS")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    parser.add_argument("--summary-only", action="store_true", help="Print only the [MEAN] lines")
    parser.add_argument("--save-csv", default=None, help="Optional: path to save a CSV of per-image and mean metrics")
    parser.add_argument("--gaussian-weights", action="store_true", help="Use Gaussian weights for SSIM")

    args = parser.parse_args()

    device = pick_device(args.device)

    # Load LPIPS model
    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    # Load reference & mask
    ref_img = open_rgb(args.ref)
    ref_np  = pil_to_np_uint8(ref_img)
    mask    = load_mask(args.mask, ref_img.size)  # bool (H,W)
    coverage = float(mask.mean())

    results = []
    masked_results = []

    # Iterate eval images
    for fname in sorted(os.listdir(args.eval_dir)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        eval_path = os.path.join(args.eval_dir, fname)
        if not os.path.isfile(eval_path):
            continue

        eval_img = open_rgb(eval_path)
        eval_img = ensure_same_size(eval_img, ref_img)
        eval_np  = pil_to_np_uint8(eval_img)
        if args.gaussian_weights:
            gaussian_weights = True
        else:
            gaussian_weights = False
        # --- Full-frame ---
        psnr_val = psnr_fn(ref_np, eval_np, data_range=255)
        ssim_val = ssim_fn(ref_np, eval_np, data_range=255, channel_axis=2,gaussian_weights=gaussian_weights)
        with torch.no_grad():
            ref_t = pil_to_lpips_tensor(ref_img, device)
            eval_t = pil_to_lpips_tensor(eval_img, device)
            lpips_val = lpips_model(ref_t, eval_t).item()

        results.append({"image": fname, "PSNR": float(psnr_val), "SSIM": float(ssim_val), "LPIPS": float(lpips_val)})

        # --- Masked ---
        m_psnr  = masked_psnr(ref_np, eval_np, mask)
        m_ssim  = masked_ssim(ref_np, eval_np, mask,gaussian_weights=gaussian_weights)
        m_lpips = masked_lpips(ref_img, eval_img, mask, lpips_model, device)

        masked_results.append({"image": fname, "mPSNR": m_psnr, "mSSIM": m_ssim, "mLPIPS": m_lpips, "coverage": coverage})

    # Output helpers
    def safe_mean(arr, key):
        vals = [x[key] for x in arr if np.isfinite(x[key])]
        if not vals:
            return float('nan')
        return float(sum(vals) / len(vals))

    # Printing
    if not args.summary_only:
        if results:
            print("=== Full-frame metrics ===")
            for r in results:
                print(f"{r['image']}: PSNR={r['PSNR']:.4f}, SSIM={r['SSIM']:.5f}, LPIPS={r['LPIPS']:.6f}")
        if masked_results:
            print("=== Masked metrics (on visible_mask) ===")
            for r in masked_results:
                print(f"{r['image']}: mPSNR={r['mPSNR']:.4f}, mSSIM={r['mSSIM']:.5f}, mLPIPS={r['mLPIPS']:.6f}, coverage={r['coverage']:.4f}")
    # Optional CSV save
    if args.save_csv is not None:
        import csv
        # merge per-image dicts by key
        by_name = {}
        for r in results:
            by_name.setdefault(r['image'], {}).update(r)
        for r in masked_results:
            by_name.setdefault(r['image'], {}).update(r)
        fieldnames = [
            'image', 'PSNR', 'SSIM', 'LPIPS', 'mPSNR', 'mSSIM', 'mLPIPS', 'coverage'
        ]
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, row in sorted(by_name.items()):
                row_out = {k: row.get(k, '') for k in fieldnames}
                row_out['image'] = name
                writer.writerow(row_out)
        print(f"Saved CSV to {args.save_csv}")


if __name__ == "__main__":
    main()


# python pov_evaluation.py \
#   --summary-only \
