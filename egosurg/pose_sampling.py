# Copyright 2025 The EgoSurg Authors.
# SPDX-License-Identifier: Apache-2.0
"""Novel-view pose samplers used by the EgoSurg diffusion step.

These functions are EgoSurg code. They are attached to the upstream
Difix CameraPoseInterpolator at runtime so that the NVIDIA/Stability
implementation is not vendored here.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


def pose_distance(a, b, w_t=1.0, w_r=1.0):
        dt = np.linalg.norm(a[:3,3]-b[:3,3])
        def so3_distance_rad(R1, R2):
            q1 = Rotation.from_matrix(R1).as_quat()
            q2 = Rotation.from_matrix(R2).as_quat()
            if np.dot(q1,q2) < 0: q2 = -q2
            dot = float(np.clip(np.dot(q1,q2), -1.0, 1.0))
            return 2.0 * np.arccos(dot)
        dr = so3_distance_rad(a[:3,:3], b[:3,:3])
        return w_t*dt + w_r*dr
def fibonacci_hemisphere_dirs(m: int, front_axis: int = -2):
    """
    在相机坐标的前向半球上生成 m 个近似等面积的方向（Fibonacci spiral）。
    front_axis: -2 表示前向是 -Z（NeRF 常见）；+2 表示 +Z（COLMAP 默认）。
    返回: [m,3] 单位向量
    """
    phi = (1 + 5**0.5) / 2.0
    i = np.arange(m)
    # 方位角沿黄金角递进
    theta = 2.0 * np.pi * i / phi
    # z 用等面积参数化（[0,1]）→ 半球 z∈[0,1]
    z = 1.0 - (i + 0.5) / m
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    x, y = r * np.cos(theta), r * np.sin(theta)
    dirs = np.stack([x, y, z], axis=1)  # 以 +Z 为前
    if front_axis == -2:
        dirs[:, 2] *= -1.0              # 切换到 -Z 前向
    return dirs

def candidates_around_camera(c2w, radius, n_dirs, rot_dev_deg=30.0,
                            front_axis=-2, surface=True, rng=None):
    if rng is None: rng = np.random.default_rng()
    def fibonacci_hemisphere_dirs(m: int, front_axis: int = 2):
        """返回相机坐标系前半球( z>=0 )的 m 个单位方向（均匀）。
        front_axis=2 表示相机前方是 +z；若你的约定是 -z，最后把 z 取负即可。
        """
        # 在半球表面均匀取样
        g = (1 + 5**0.5) / 2
        idx = np.arange(m)
        theta = 2 * np.pi * idx / g
        z = 1 - (idx + 0.5) / m             # [0,1] 线性 → 顶部密度均匀
        r = np.sqrt(np.maximum(0.0, 1 - z*z))
        x, y = r * np.cos(theta), r * np.sin(theta)
        dirs = np.stack([x, y, z], axis=1)  # 半球: z>=0

        if front_axis == 2:  # 相机前方是 +z
            return dirs
        elif front_axis == -2:  # 相机前方是 -z（常见NeRF约定）
            dirs[:, 2] *= -1
            return dirs
        else:
            raise ValueError("front_axis must be 2 or -2")
    Rw, tw = c2w[:3, :3], c2w[:3, 3]
    dirs_cam = fibonacci_hemisphere_dirs(n_dirs, front_axis=front_axis)  # 相机坐标
    cand = []
    for d_cam in dirs_cam:
        if surface:
            r = radius
        else:
            r = radius * (rng.random() ** (1/3))  # 均匀体采样
        offset_world = Rw @ (d_cam * r)
        t_new = tw + offset_world

        # 小角度扰动：不超过 rot_dev_deg
        if rot_dev_deg > 0:
            axis_cam = rng.normal(size=3); axis_cam /= (np.linalg.norm(axis_cam) + 1e-9)
            ang = np.deg2rad(rng.random() * rot_dev_deg)
            dR_cam = Rotation.from_rotvec(axis_cam * ang).as_matrix()
            R_new = Rw @ dR_cam
        else:
            R_new = Rw

        pose = np.eye(4); pose[:3,:3]=R_new; pose[:3,3]=t_new
        cand.append(pose)
    return np.stack(cand, axis=0)  # [n_dirs,4,4]



def kcenter_select(candidates, K, w_t=1.0, w_r=1.0, seed=0):
    rng = np.random.default_rng(seed)
    M = len(candidates)
    chosen = [rng.integers(M)]
    # 维护每个候选到“已选集合”的最小距离
    mind = np.array([pose_distance(candidates[i], candidates[chosen[0]], w_t, w_r)
                    for i in range(M)], dtype=np.float64)
    for _ in range(1, K):
        idx = int(np.argmax(mind))
        chosen.append(idx)
        # 更新 mind
        di = np.array([pose_distance(candidates[j], candidates[idx], w_t, w_r)
                    for j in range(M)], dtype=np.float64)
        mind = np.minimum(mind, di)
    return np.array(chosen, dtype=int)


def interpolator_cls(base_cls):
    """Return a subclass of the upstream interpolator with EgoSurg samplers."""

    class EgoPoseInterpolator(base_cls):

        def sample_novel_poses(self, training_poses, radius=0.1, per_cam=16,
                               rot_dev_deg=25.0, front_axis=-2,
                               target_total=None, w_t=1.0, w_r=0.5, seed=42):
            rng = np.random.default_rng(seed)
            all_cands = []
            for c2w in training_poses:
                cand = candidates_around_camera(c2w, radius, per_cam,
                                                rot_dev_deg, front_axis, surface=True, rng=rng)
                all_cands.append(cand)
            C = np.concatenate(all_cands, axis=0)

            keep = []
            for i in range(C.shape[0]):
                dmin = np.min([pose_distance(C[i], t, w_t, w_r) for t in training_poses])
                if dmin > 0.02 * radius:
                    keep.append(i)
            C = C[keep]

            if target_total is None:
                target_total = training_poses.shape[0]
            idx = kcenter_select(C, min(target_total, len(C)),
                                 w_t=w_t, w_r=w_r, seed=seed)
            return C[idx]
        def sample_uniform_poses_per_camera(
            self,
            training_poses,
            radius=0.1,
            n_per_cam=8,
            rot_dev_deg=20.0,
            front_axis=-2,
            seed=42,
        ):
            """
            对于每一个输入相机，在其前向半球上均匀生成 n_per_cam 个新视角。
            Args:
                training_poses: [N,4,4] 相机 cam2world
                radius: 半球半径
                n_per_cam: 每台相机生成的个数
                rot_dev_deg: 最大朝向扰动角度
                front_axis: 相机前方方向 (+2=+Z, -2=-Z)
                seed: 随机种子
            Returns:
                novel_poses: [N*n_per_cam,4,4] 新位姿数组
            """
            rng = np.random.default_rng(seed)
            dev_rad = np.deg2rad(rot_dev_deg)
            novel_list = []

            # 半球均匀方向
            dirs_cam = fibonacci_hemisphere_dirs(n_per_cam, front_axis=front_axis)

            for c2w in training_poses:
                Rw, tw = c2w[:3, :3], c2w[:3, 3]
                for d_cam in dirs_cam:
                    offset_world = Rw @ (d_cam * radius)
                    t_new = tw + offset_world

                    # 小角度扰动（随机轴随机角度，不超过 rot_dev_deg）
                    if rot_dev_deg > 0:
                        axis_cam = rng.normal(size=3)
                        axis_cam /= (np.linalg.norm(axis_cam) + 1e-9)
                        ang = rng.random() * dev_rad
                        dR_cam = Rotation.from_rotvec(axis_cam * ang).as_matrix()
                        R_new = Rw @ dR_cam
                    else:
                        R_new = Rw

                    pose = np.eye(4)
                    pose[:3, :3] = R_new
                    pose[:3, 3] = t_new
                    novel_list.append(pose)

            return np.stack(novel_list, axis=0)
        def sample_sample_spline(        self,
            key_poses,
            n_samples: int = 120,
            closed: bool = False,
            boundary: str = "natural",  # "natural" | "clamped" | "periodic"
            seed: int = 42,
        ):
            """
            Creates a smooth spline path between input keyframe camera poses.

            根据给定的关键帧相机位姿（cam2world, [K,4,4]），
            生成一条平滑的样条轨迹，并采样出 n_samples 个新位姿。

            Args:
                key_poses: [K,4,4] cam2world 矩阵序列（关键帧）
                n_samples: 采样的新视角数量
                closed: 是否闭环（首尾相接形成循环路径）
                boundary: 平移样条的边界条件：
                        - "natural": 二阶导为0的自然边界（非闭环默认）
                        - "clamped": 一阶导固定（使用端点的邻近导数估计）
                        - "periodic": 周期边界（通常与 closed=True 搭配）
                seed: 随机种子（当前函数中未用到，但保留接口一致性）

            Returns:
                poses: [n_samples,4,4] 样条路径上的新位姿（cam2world）
            """
            key_poses = np.asarray(key_poses)
            assert key_poses.ndim == 3 and key_poses.shape[1:] == (4, 4), \
                "key_poses must have shape [K,4,4]"
            K = key_poses.shape[0]
            if K < 2:
                # 只有1个关键帧时，重复它
                out = np.repeat(key_poses[:1], n_samples, axis=0)
                return out

            # ---------- 拆分旋转与平移 ----------
            R_list = []
            t_list = []
            for k in range(K):
                c2w = key_poses[k]
                R_list.append(c2w[:3, :3])
                t_list.append(c2w[:3, 3])
            R_list = np.stack(R_list, axis=0)          # [K,3,3]
            t_list = np.stack(t_list, axis=0)          # [K,3]

            # ---------- 闭环处理（可选） ----------
            # 若闭环或周期边界，对平移与旋转都在末尾追加首帧，保证首尾一致
            use_periodic = (boundary == "periodic") or closed
            if use_periodic:
                # 如果首尾不一致，显式补一个首帧到末尾
                if not np.allclose(t_list[0], t_list[-1]) or \
                not np.allclose(R_list[0], R_list[-1]):
                    t_list = np.concatenate([t_list, t_list[:1]], axis=0)
                    R_list = np.concatenate([R_list, R_list[:1]], axis=0)

            # ---------- 以弦长作为参数（避免非均匀采样导致的速度抖动） ----------
            # 仅按平移的弦长来定义参数轴 s
            d = np.linalg.norm(np.diff(t_list, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(d)], axis=0)     # [K'] or [K'+1] if closed-extended
            if s[-1] < 1e-8:
                # 所有点重合，退化为原地旋转插值
                s = np.linspace(0.0, 1.0, num=t_list.shape[0])

            # 规范到 [0,1]
            s = (s - s[0]) / (s[-1] - s[0] + 1e-12)

            # ---------- 平移样条 ----------
            # 边界条件
            bc = boundary
            if use_periodic:
                bc = "periodic"

            if bc == "clamped" and t_list.shape[0] >= 3:
                # 用端点处的前向/后向差分近似速度，构造成 (bc_type, value) 形式
                v0 = (t_list[1] - t_list[0]) / (s[1] - s[0] + 1e-12)
                v1 = (t_list[-1] - t_list[-2]) / (s[-1] - s[-2] + 1e-12)
                # CubicSpline 支持为每个轴分别给边界导数，传 shape 为 (2,3)
                bc_type = ((1, v0), (1, v1))
            else:
                bc_type = bc  # "natural" or "periodic" 或者 "clamped" 但点数不足时退化

            # 对 x,y,z 三个轴分别拟合样条
            tx = CubicSpline(s, t_list[:, 0], bc_type=bc_type)
            ty = CubicSpline(s, t_list[:, 1], bc_type=bc_type)
            tz = CubicSpline(s, t_list[:, 2], bc_type=bc_type)

            # ---------- 旋转 SLERP ----------
            # 用关键帧的 s 作为时间戳，做四元数插值
            key_rots = Rotation.from_matrix(R_list)      # 长度与 t_list 相同（可能 +1）
            # 若 periodic，且最后一个重复首帧，Slerp 也能处理（需要递增的 times）
            # 避免 Slerp 在 times 重复时出错，做一个极小扰动
            s_slerp = s.copy()
            for i in range(1, len(s_slerp)):
                if abs(s_slerp[i] - s_slerp[i-1]) < 1e-10:
                    s_slerp[i] = s_slerp[i-1] + 1e-10
            slerp = Slerp(s_slerp, key_rots)

            # ---------- 采样 ----------
            s_min = float(np.min(s_slerp))
            s_max = float(np.max(s_slerp))

            if (boundary == "periodic") or closed:
                # 闭环：避免与首帧重复，用 endpoint=False
                s_samples = np.linspace(s_min, s_max, n_samples, endpoint=False)
            else:
                # 非闭环：把上界向下取一个“前驱”浮点，确保落在区间内
                s_max_adj = np.nextafter(s_max, -np.inf)
                s_samples = np.linspace(s_min, s_max_adj, n_samples, endpoint=True)

            t_samples = np.stack([tx(s_samples), ty(s_samples), tz(s_samples)], axis=-1)  # [N,3]
            R_samples = slerp(s_samples).as_matrix()  
            # s_samples = np.linspace(0.0, 1.0, n_samples)
            # t_samples = np.stack([tx(s_samples), ty(s_samples), tz(s_samples)], axis=-1)  # [N,3]
            # R_samples = slerp(s_samples).as_matrix()                                      # [N,3,3]

            # ---------- 组装 cam2world ----------
            poses = np.repeat(np.eye(4)[None, ...], n_samples, axis=0)  # [N,4,4]
            poses[:, :3, :3] = R_samples
            poses[:, :3, 3] = t_samples
            return poses

    EgoPoseInterpolator.__name__ = "EgoPoseInterpolator"
    return EgoPoseInterpolator
