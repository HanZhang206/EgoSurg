#!/usr/bin/env python3
"""Download weights EgoSurg does not redistribute.

Difix weights come from Hugging Face (Stability AI Community License and
the NVIDIA non-commercial license). FoundationStereo weights come from the
upstream Google Drive release and are research-only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--foundation-dir",
        default=os.environ.get("FOUNDATION_STEREO_ROOT", "third_party/FoundationStereo"),
    )
    parser.add_argument("--skip-difix", action="store_true")
    parser.add_argument("--skip-stereo", action="store_true")
    args = parser.parse_args()

    if not args.skip_difix:
        from huggingface_hub import snapshot_download

        for repo in ("nvidia/difix_ref", "nvidia/difix"):
            path = snapshot_download(repo)
            print(f"{repo}: {path}")

    if not args.skip_stereo:
        dest = os.path.join(args.foundation_dir, "pretrained_models")
        os.makedirs(dest, exist_ok=True)
        # Folder published by the FoundationStereo README.
        folder = "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf"
        cmd = ["gdown", "--folder", folder, "-O", dest]
        print(" ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except FileNotFoundError:
            sys.exit("Install gdown (pip install gdown) or download the folder by hand into pretrained_models/.")
        except subprocess.CalledProcessError as exc:
            sys.exit(f"gdown failed ({exc.returncode}). Download the folder linked in the FoundationStereo README.")


if __name__ == "__main__":
    main()
