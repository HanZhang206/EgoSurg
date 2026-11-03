#!/usr/bin/env python3
"""Run upstream FoundationStereo on one rectified pair and save a depth map.

This is a thin wrapper. The model code stays in the FoundationStereo checkout
pointed at by FOUNDATION_STEREO_ROOT (commit 41390d52).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def timestamp_dir(left: Path) -> Path:
    """{timestamp}/images/cam{i}_L.jpg -> {timestamp}."""
    left = left.resolve()
    if left.parent.name != "images" or not left.stem.endswith("_L"):
        sys.exit(
            "The left image must be {timestamp}/images/cam{i}_L.jpg "
            f"(got {left})."
        )
    return left.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--intrinsics", required=True, help="FoundationStereo K.txt (3x3 row, then baseline)")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    root = os.environ.get("FOUNDATION_STEREO_ROOT", "")
    script = Path(root) / "scripts" / "run_demo.py"
    if not script.is_file():
        sys.exit("Set FOUNDATION_STEREO_ROOT to the pinned FoundationStereo checkout.")
    ckpt = args.ckpt or str(Path(root) / "pretrained_models" / "23-51-11" / "model_best_bp2.pth")
    left = Path(args.left)
    scene = timestamp_dir(left)
    depth_path = scene / "depths" / f"{left.stem}.npy"
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    # run_demo.py writes depth_meter.npy only on the point-cloud path, and that
    # path then blocks in an Open3D window until ESC. Load the same script with
    # the viewer replaced so a batch of cameras can finish unattended.
    launcher = r"""
import sys
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
import open3d as o3d

class _QuietVisualizer:
    def create_window(self, *args, **kwargs):
        return None
    def add_geometry(self, *args, **kwargs):
        return None
    def get_render_option(self):
        class _Opt:
            point_size = 1.0
            background_color = None
        return _Opt()
    def run(self):
        return None
    def destroy_window(self):
        return None

o3d.visualization.Visualizer = _QuietVisualizer
import runpy
runpy.run_path(script, run_name="__main__")
"""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable,
            "-c",
            launcher,
            str(script),
            "--left_file",
            str(left),
            "--right_file",
            args.right,
            "--intrinsic_file",
            args.intrinsics,
            "--ckpt_dir",
            ckpt,
            "--out_dir",
            tmp,
            "--scale",
            str(args.scale),
            "--get_pc",
            "1",
        ]
        env = os.environ.copy()
        # PyTorch 2.6+ defaults torch.load to weights_only=True. The upstream
        # demo loads a checkpoint that contains numpy scalars.
        env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.check_call(cmd, env=env, cwd=root)
        produced = Path(tmp) / "depth_meter.npy"
        if not produced.is_file():
            sys.exit("FoundationStereo did not write depth_meter.npy.")
        shutil.copyfile(produced, depth_path)
    for name in ("pts_cleaned.ply", "depth_vals.npy"):
        stale = scene / name
        if stale.is_file():
            stale.unlink()
    print(f"Wrote {depth_path}")


if __name__ == "__main__":
    main()
