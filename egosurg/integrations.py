"""Load upstream Difix3D without vendoring it into this repository."""

from __future__ import annotations

import os
import sys

DIFIX_COMMIT = "c76edc595586e16732c91ddee82f3a6d83a8a9cc"
DIFIX_URL = "https://github.com/nv-tlabs/Difix3D"


def difix_root() -> str:
    root = os.environ.get("DIFIX3D_ROOT", "")
    if not root or not os.path.isdir(root):
        raise ImportError(
            "Set DIFIX3D_ROOT to a checkout of "
            f"{DIFIX_URL} at commit {DIFIX_COMMIT}. "
            "scripts/setup_third_party.sh clones that commit."
        )
    return root


def load_difix():
    """Return (DifixPipeline, pose interpolator class).

    The interpolator is the upstream CameraPoseInterpolator plus the
    novel-view samplers used by EgoSurg. Those samplers live in this
    repository; the diffusion code stays in the upstream checkout.
    """
    root = difix_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from examples.utils import CameraPoseInterpolator
    from src.pipeline_difix import DifixPipeline

    from egosurg.pose_sampling import interpolator_cls

    return DifixPipeline, interpolator_cls(CameraPoseInterpolator)
