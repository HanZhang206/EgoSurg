# Copyright 2025 The EgoSurg Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the gsplat examples, which are licensed under Apache-2.0:
# https://github.com/nerfstudio-project/gsplat

# datasets/rgbd.py
import json, os, numpy as np, imageio.v3 as iio
import torch
from pathlib import Path
import numpy as np

import os, json
from pathlib import Path
import numpy as np
import imageio.v3 as iio   # ← 需要这个
# 其他 import 省略...
def remove_sparse_voxels(pts, cols, voxel_size=0.02, min_points=5):
    """
    voxel_size: 体素边长（单位=当前坐标单位；normalize=True 后就是“归一化单位”）
    min_points: 每个体素保留的最少点数，少于此值的体素整体丢弃
    """
    # 体素坐标(整数)
    keys = np.floor(pts / voxel_size).astype(np.int64)
    # 为了 hash，平移到非负
    base = keys - keys.min(axis=0, keepdims=True)
    # 简单 hash（避免按行 unique 的开销）
    key_hash = (base[:, 0] * 73856093) ^ (base[:, 1] * 19349663) ^ (base[:, 2] * 83492791)

    uniq, inv, counts = np.unique(key_hash, return_inverse=True, return_counts=True)
    keep = counts[inv] >= min_points
    return pts[keep], cols[keep]

def voxel_downsample(pts, cols, target_n=None, voxel_size=None, max_iter=10, agg="mean"):
    """
    pts: (N,3) float32
    cols: (N,3) float32 in [0,1] or [0,255]
    target_n: 期望点数（可选）；如果给定，就自适应 voxel_size
    voxel_size: 体素边长（米，或你normalize之后的单位）；二选一
    agg: "mean" / "rand" / "first" 体素内如何聚合
    """
    assert (target_n is not None) ^ (voxel_size is not None), \
        "必须二选一：给 target_n 或给 voxel_size"

    # 初始体素大小：如果用户给了 target_n，就用自适应
    if voxel_size is None:
        # 用场景尺度估个起点（90分位半径/100，经验值）
        center = np.median(pts, axis=0, keepdims=True)
        radius = np.percentile(np.linalg.norm(pts - center, axis=1), 90)
        voxel_size = float(radius / 100.0) if radius > 0 else 1.0

    for it in range(max_iter):
        # 体素键：向下取整
        keys = np.floor(pts / voxel_size).astype(np.int64)

        # 将 (i,j,k) 映射到唯一整数（哈希），避免 np.unique 按行慢
        # 用 Morton code/自实现都行，这里简单拼装（注意溢出就转 int64）
        # 也可：view 成结构化 dtype 再 unique
        base = keys - keys.min(axis=0, keepdims=True)
        key_hash = (base[:,0] * 73856093) ^ (base[:,1] * 19349663) ^ (base[:,2] * 83492791)
        uniq, inv = np.unique(key_hash, return_inverse=True)

        if (target_n is None) or (len(uniq) <= target_n) or (it == max_iter - 1):
            # 聚合
            out_pts = np.zeros((len(uniq), 3), dtype=pts.dtype)
            out_cols = np.zeros((len(uniq), 3), dtype=cols.dtype)
            if agg == "mean":
                # 体素内算均值（质心 & 平均颜色）
                np.add.at(out_pts, inv, pts)
                np.add.at(out_cols, inv, cols)
                counts = np.bincount(inv)
                out_pts /= counts[:, None]
                out_cols /= counts[:, None]
            elif agg == "rand":
                # 每个体素随机取一个点
                # 找到每个体素的第一个索引，然后替换为随机索引
                first_idx = np.zeros(len(uniq), dtype=np.int64) - 1
                # 记录体素内的所有索引
                buckets = [[] for _ in range(len(uniq))]
                for idx, b in enumerate(inv):
                    buckets[b].append(idx)
                for b, arr in enumerate(buckets):
                    j = np.random.randint(0, len(arr))
                    first_idx[b] = arr[j]
                out_pts = pts[first_idx]
                out_cols = cols[first_idx]
            else:  # "first"
                first_idx = np.zeros(len(uniq), dtype=np.int64) - 1
                for i_idx, b in enumerate(inv):
                    if first_idx[b] < 0:
                        first_idx[b] = i_idx
                out_pts = pts[first_idx]
                out_cols = cols[first_idx]
            return out_pts, out_cols, voxel_size

        # 体素数太多 → 增大 voxel_size（体素体积 ~ voxel_size^3）
        ratio = (len(uniq) / target_n) if target_n else 2.0
        voxel_size *= (ratio ** (1.0/3.0)) * 1.05  # 稍微加一点松弛
class Parser:
    def __init__(self, data_dir: str, factor: int = 1, normalize: bool = False, test_every: int = 8, skip_loading_ply: bool = False):
        self.data_dir = Path(data_dir)
        meta = json.loads((self.data_dir / "transforms.json").read_text())

        self.frames = meta["frames"]
        self.factor = factor
        self.test_every = test_every
        self.camera_model = meta.get("camera_model", "pinhole")
        self.skip_loading_ply = skip_loading_ply
        # ------- 收集相机与内参 -------
        camtoworlds = []
        Ks, imsizes = {}, {}
        for i, f in enumerate(self.frames):
            w, h = int(f["width"] // factor), int(f["height"] // factor)
            K = np.array(f["K"], dtype=np.float32).reshape(3, 3).copy()
            K[0, :] /= factor; K[1, :] /= factor
            Ks[i] = K
            imsizes[i] = (w, h)
            c2w = np.array(f["c2w"], dtype=np.float32)
            assert c2w.shape == (4, 4) and np.isfinite(c2w).all()
            camtoworlds.append(c2w)

        self.camtoworlds = np.stack(camtoworlds, axis=0)  # (M, 4, 4)
        self.Ks_dict = Ks
        self.imsize_dict = imsizes

        # ------- 用 RGB-D 反投影生成点云 -------
        depth_vals = [] 
        if not os.path.exists(self.data_dir / "pts_cleaned.ply") and not os.path.exists(self.data_dir / "cols_cleaned.ply") and not self.skip_loading_ply:
            pts_list, cols_list = [], []
            print("Generating pts_cleaned.ply and cols_cleaned.ply")
            for i, f in enumerate(self.frames):
                img = iio.imread(self.data_dir / f["file_path"])
                img = img[..., :3]  # 去 alpha
                if img.dtype != np.float32:
                    img = img.astype(np.float32) / 255.0

                depth_path = self.data_dir / f["depth_path"]
                dp = f.get("depth_path", None)
                if dp == 'None':
                    continue
                if depth_path.suffix == ".npy":
                    D = np.load(depth_path).astype(np.float32)



                # 如果是毫米，解除注释
                # if meta.get("depth_unit", "m") == "mm":
                #     D *= 1e-3

                img = img[::factor, ::factor, :]
                D   = D[::factor, ::factor]

                h, w = D.shape
                ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                                    np.arange(w, dtype=np.float32), indexing="ij")
                pix = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3)

                K = self.Ks_dict[i]
                rays_c = (np.linalg.inv(K) @ pix.T).T  # (HW,3)
                Z = D.reshape(-1, 1)
                valid = np.isfinite(Z[:, 0]) & (Z[:, 0] > 0)
                if not np.any(valid):
                    continue

                rays_c = rays_c[valid]
                Z = Z[valid]
                xyz_c = rays_c * Z  # (N,3)

                c2w = self.camtoworlds[i]
                R = c2w[:3, :3]; t = c2w[:3, 3:4]
                xyz_w = (R @ xyz_c.T + t).T  # (N,3)

                col = img.reshape(-1, 3)[valid]
                pts_list.append(xyz_w); cols_list.append(col)
                depth_vals.append(Z[:, 0])  # for bounds
            assert len(pts_list) > 0, "No valid depth points found."
            pts  = np.concatenate(pts_list, 0)
            cols = np.concatenate(cols_list, 0)
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            total_points = len(pts)
            print('total_points', total_points)
            voxel = 0.01  # 先略大一点，减少点数
            nb_neighbors = 100
            std_ratio = 1.5
            pcd = pcd.voxel_down_sample(voxel_size=voxel)
            print('after voxel_down_sample', len(pcd.points))
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            print('after remove_statistical_outlier', len(pcd.points))
            # o3d.visualization.draw_geometries([pcd])
            # breakpoint()
            pts = np.asarray(pcd.points)
            cols = np.asarray(pcd.colors)
            # save it in the data_dir as pts.ply and cols.ply
            np.save(self.data_dir / "depth_vals.npy", np.concatenate(depth_vals, axis=0))
            o3d.io.write_point_cloud(self.data_dir / "pts_cleaned.ply", pcd)
            # o3d.io.write_point_cloud(self.data_dir / "cols_cleaned.ply", pcd)
        
        elif self.skip_loading_ply:
            # random crate some points
            pts = np.random.rand(10000, 3)
            cols = np.random.rand(10000, 3)
            depth_vals = np.random.rand(10000)
            depth_vals = []
        
        else:
            print("Loading pts_cleaned.ply and cols_cleaned.ply")
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(self.data_dir / "pts_cleaned.ply")
            dv_path = self.data_dir / "depth_vals.npy"
            pts = np.asarray(pcd.points)
            cols = np.asarray(pcd.colors)
            if dv_path.exists():
                depth_vals = [np.load(dv_path)]  # keep type compatible with len(...) and possible concatenate later
            else:
                depth_vals = []
        


        # ------- 归一化：同时作用于 points 和 camtoworlds -------
        # 用 90 分位半径定义尺度，更鲁棒
        if normalize:
            center = np.median(pts, axis=0, keepdims=True)  # (1,3)
            pts -= center
            # 平移相机
            self.camtoworlds[:, :3, 3] -= center.reshape(3)

            scale = np.percentile(np.linalg.norm(pts, axis=1), 90)
            scale = float(max(scale, 1e-6))
            pts /= scale
            # 相机做同样的缩放
            self.camtoworlds[:, :3, 3] /= scale

            # 记录整体相似变换（S @ T）
            T = np.eye(4, dtype=np.float32); T[:3, 3] = -center.reshape(3)
            S = np.eye(4, dtype=np.float32); S[:3, :3] *= (1.0 / scale)
            self.transform = (S @ T).astype(np.float32)

            self.scene_scale = 1.0
        else:
            # breakpoint()
            # 非归一化：用相机轨迹的尺度估计 scene_scale
            cam_locs = self.camtoworlds[:, :3, 3]
            scene_center = cam_locs.mean(axis=0)
            self.scene_scale = float(np.linalg.norm(cam_locs - scene_center, axis=1).max())
            self.transform = np.eye(4, dtype=np.float32)

        # ------- 其余字段 -------
        self.points = pts.astype(np.float32)
        self.points_rgb = (cols * 255.0).astype(np.float32)

        # bounds: 用深度统计，归一化后要除以 scale
        if len(depth_vals) > 0:
            depths = np.concatenate(depth_vals, 0)
            depths = depths[np.isfinite(depths) & (depths > 0)]
            if normalize:
                depths = depths / scale
            near = float(np.percentile(depths, 1))
            far  = float(np.percentile(depths, 99))
            if near <= 0: near = 0.01
            if far <= near: far = near * 2.0
            self.bounds = np.array([near, far], dtype=np.float32)
        else:
            self.bounds = np.array([0.01, 1.0], dtype=np.float32)

        self.extconf = {"spiral_radius_scale": 1.0}

        # ------- 兼容 ColmapDataset 的“按名字”接口 -------
        self.image_names, self.name_to_id, self.id_to_name = [], {}, {}
        Ks_by_name, sizes_by_name = {}, {}
        image_path_dict, depths_path_dict = {}, {}

        for i, f in enumerate(self.frames):
            name = Path(f["file_path"]).stem
            self.image_names.append(name)
            self.name_to_id[name] = i
            self.id_to_name[i] = name

            Ks_by_name[name] = np.array(self.Ks_dict[i], dtype=np.float32)
            sizes_by_name[name] = tuple(self.imsize_dict[i])  # (W,H)
            image_path_dict[name] = str(f["file_path"]).replace("\\", "/")
            # depths_path_dict[name] = str(f["depth_path"]).replace("\\", "/")
            if f["depth_path"] == 'None':
                depths_path_dict[name] = None
            else:
                depths_path_dict[name] = str(f["depth_path"]).replace("\\", "/")

        # 最终按 camera_id 重新映射
        self.camera_ids = list(range(len(self.image_names)))
        self.Ks_dict = {i: Ks_by_name[self.image_names[i]] for i in self.camera_ids}
        self.imsize_dict = {i: sizes_by_name[self.image_names[i]] for i in self.camera_ids}
        self.params_dict = {i: np.empty(0, dtype=np.float32) for i in self.camera_ids}
        self.mask_dict = {i: None for i in self.camera_ids}
        self.mapx_dict, self.mapy_dict, self.roi_undist_dict = {}, {}, {}

        self.image_path_dict = image_path_dict
        self.depths_path_dict = depths_path_dict
        self.point_indices = {name: np.array([], dtype=np.int32) for name in self.image_names}

        # ColmapDataset 需要 image_paths(list)
        self.image_paths = [str(self.data_dir / self.image_path_dict[name]) for name in self.image_names]
