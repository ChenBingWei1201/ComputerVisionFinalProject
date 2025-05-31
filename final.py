import os
import os.path as osp
import numpy as np
import cv2
import open3d as o3d
from typing import Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 7-Scenes 內參 (fx, fy, cx, cy)
INTRINSIC = (525, 525, 320, 240)


class SevenScenes3DReconstructor:
    """
    3D reconstruction pipeline for 7-Scenes using
    key-frame RGB-D fusion + PnP + ICP refinement.
    """

    def __init__(
        self,
        intrinsics=INTRINSIC,
    ):
        self.fx, self.fy, self.cx, self.cy = intrinsics

        # intrinsic matrix (3×3)
        self.K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float32,
        )

        # SIFT + BF-L2 matcher
        self.detector = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def load_frame_data(self, frame_path: str) -> Dict:
        """Load RGB, depth and (optional) pose for a single frame."""
        frame_dir = osp.dirname(frame_path)
        frame_name = osp.basename(frame_path).replace(".color.png", "")

        # RGB
        rgb_path = osp.join(frame_dir, f"{frame_name}.color.png")
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise ValueError(f"Cannot load RGB image: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # depth: prefer depth.proj.png
        depth_path = osp.join(frame_dir, f"{frame_name}.depth.proj.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            depth_path = osp.join(frame_dir, f"{frame_name}.depth.png")
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        depth = depth.astype(np.float32) / 1000.0  # mm → m
        depth[depth == 65.535] = 0
        depth[(depth > 10.0) | (depth < 0.1)] = 0

        # pose (only available for first test frame)
        pose_path = osp.join(frame_dir, f"{frame_name}.pose.txt")
        pose = np.loadtxt(pose_path, dtype=np.float32) if osp.exists(pose_path) else None

        return {"rgb": rgb, "depth": depth, "pose": pose}

    # ------------------------------------------------------------------
    # Geometric helpers
    # ------------------------------------------------------------------
    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        pose: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Back-project depth to 3-D points (camera or world)."""
        h, w = depth.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        mask = depth > 0
        z = depth[mask]
        x = (xx[mask] - self.cx) * z / self.fx
        y = (yy[mask] - self.cy) * z / self.fy
        pts_cam = np.stack([x, y, z], axis=-1)
        cols = rgb[mask] / 255.0

        if pose is not None:
            pts_world = (pose[:3, :3] @ pts_cam.T).T + pose[:3, 3]
            return pts_world, cols
        return pts_cam, cols

    # ------------------------------------------------------------------
    # ICP refinement
    # ------------------------------------------------------------------
    def refine_pose_with_icp(
        self, source_pts: np.ndarray, target_pts: np.ndarray, init_T: np.ndarray
    ) -> np.ndarray:
        """
        Point-to-plane ICP using Open3D.
        `init_T` is the PnP-estimated transform from frame2 → frame1.
        """
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pts))
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pts))
        src = src.voxel_down_sample(0.02)
        tgt = tgt.voxel_down_sample(0.02)
        src.estimate_normals()
        tgt.estimate_normals()

        reg = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            max_correspondence_distance=0.05,
            init=init_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=20
            ),
        )
        return reg.transformation

    # ------------------------------------------------------------------
    # Relative pose (PnP → ICP)
    # ------------------------------------------------------------------
    def estimate_relative_pose(
        self, frame1: Dict, frame2: Dict
    ) -> np.ndarray:
        """
        Compute T_21 (transform points in frame2 to frame1)
        via feature PnP + optional ICP refinement.
        """
        gray1 = cv2.cvtColor(frame1["rgb"], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2["rgb"], cv2.COLOR_RGB2GRAY)
        kp1, desc1 = self.detector.detectAndCompute(gray1, None)
        kp2, desc2 = self.detector.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None:
            return np.eye(4, dtype=np.float32)

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        if len(good) < 20:
            return np.eye(4, dtype=np.float32)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        depth1 = frame1["depth"]
        pts3d, pts2d = [], []
        for (u, v), p2 in zip(pts1, pts2):
            x, y = int(u), int(v)
            if 0 <= x < depth1.shape[1] and 0 <= y < depth1.shape[0]:
                z = depth1[y, x]
                if z > 0:
                    pts3d.append(
                        [(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z]
                    )
                    pts2d.append(p2)
        if len(pts3d) < 10:
            return np.eye(4, dtype=np.float32)

        pts3d = np.array(pts3d, dtype=np.float32)
        pts2d = np.array(pts2d, dtype=np.float32)

        success, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            self.K,
            None,
            iterationsCount=1000,
            reprojectionError=3.0,
            confidence=0.99,
        )
        if not success:
            return np.eye(4, dtype=np.float32)

        R, _ = cv2.Rodrigues(rvec)
        T_rel = np.eye(4, dtype=np.float32)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = tvec.flatten()

        # ------------- ICP refinement -------------
        src_pts, _ = self.depth_to_pointcloud(frame1["depth"], frame1["rgb"])
        tgt_pts, _ = self.depth_to_pointcloud(frame2["depth"], frame2["rgb"])
        if len(src_pts) > 2000 and len(tgt_pts) > 2000:
            T_rel = self.refine_pose_with_icp(src_pts, tgt_pts, T_rel)

        return T_rel

    # ------------------------------------------------------------------
    # Depth filtering
    # ------------------------------------------------------------------
    def refine_depth_bilateral(self, depth: np.ndarray) -> np.ndarray:
        depth_mm = (depth * 1000).astype(np.float32)
        depth_f = cv2.bilateralFilter(depth_mm, 5, 10, 5) / 1000.0
        depth_f[depth == 0] = 0
        return depth_f

    # ------------------------------------------------------------------
    # Main sequence reconstruction
    # ------------------------------------------------------------------
    def reconstruct_sequence(
        self,
        sequence_path: str,
        output_path: str,
        kf_every: int = 20,
        voxel_size: float = 7.5e-3,
        max_frames: Optional[int] = None,
    ):
        print(f"=== Reconstructing: {sequence_path}")
        frame_files = sorted(f for f in os.listdir(sequence_path) if f.endswith(".color.png"))
        if max_frames:
            frame_files = frame_files[: max_frames]

        key_idx = list(range(0, len(frame_files), kf_every))

        poses, comb_pts, comb_cols = [], [], []

        for i, idx in enumerate(tqdm(key_idx, desc="Keyframes")):
            frame_path = osp.join(sequence_path, frame_files[idx])
            frame = self.load_frame_data(frame_path)
            frame["depth"] = self.refine_depth_bilateral(frame["depth"])

            if i == 0:
                pose = frame["pose"] if frame["pose"] is not None else np.eye(
                    4, dtype=np.float32
                )
            else:
                prev_path = osp.join(sequence_path, frame_files[key_idx[i - 1]])
                prev_frame = self.load_frame_data(prev_path)
                T_rel = self.estimate_relative_pose(prev_frame, frame)
                pose = poses[-1] @ np.linalg.inv(T_rel)

            poses.append(pose)

            pts, cols = self.depth_to_pointcloud(frame["depth"], frame["rgb"], pose)
            if len(pts) > 100:
                pcd_tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
                pcd_tmp, ind = pcd_tmp.remove_statistical_outlier(20, 2.0)
                pts, cols = np.asarray(pcd_tmp.points), cols[ind]

            comb_pts.append(pts)
            comb_cols.append(cols)

        all_pts = np.vstack(comb_pts)
        all_cols = np.vstack(comb_cols)
        print(f"Points before filter: {len(all_pts)}")

        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(all_pts)
        )
        pcd.colors = o3d.utility.Vector3dVector(all_cols)
        pcd, _ = pcd.remove_statistical_outlier(30, 1.0)
        pcd = pcd.voxel_down_sample(voxel_size)

        print(f"Points after filter : {len(pcd.points)}")
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved: {output_path}")
        return pcd


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("7-Scenes reconstruction")
    parser.add_argument("--mode", choices=["single", "all"], default="all")
    parser.add_argument("--sequence", help="sequence path (for single)")
    parser.add_argument("--output", help="output .ply (for single)")
    parser.add_argument("--kf_every", type=int, default=20)
    parser.add_argument("--voxel_size", type=float, default=7.5e-3)
    parser.add_argument("--max_frames", type=int)
    args = parser.parse_args()

    recon = SevenScenes3DReconstructor()

    if args.mode == "single":
        if not args.sequence or not args.output:
            parser.error("--sequence and --output are required for single mode")
        recon.reconstruct_sequence(
            args.sequence,
            args.output,
            kf_every=args.kf_every,
            voxel_size=args.voxel_size,
            max_frames=args.max_frames,
        )
    else:
        test_seqs = [
            ("chess", "seq-03"),
            ("fire", "seq-03"),
            ("heads", "seq-01"),
            ("office", "seq-02"),
            ("office", "seq-06"),
            ("office", "seq-07"),
            ("office", "seq-09"),
            ("pumpkin", "seq-01"),
            ("redkitchen", "seq-03"),
            ("redkitchen", "seq-04"),
            ("redkitchen", "seq-06"),
            ("redkitchen", "seq-12"),
            ("redkitchen", "seq-14"),
            ("stairs", "seq-01"),
        ]
        os.makedirs("test", exist_ok=True)
        for scene, seq in test_seqs:
            seq_path = f"7SCENES/{scene}/test/{seq}"
            out_path = f"test/{scene}-{seq}.ply"
            try:
                recon.reconstruct_sequence(
                    seq_path,
                    out_path,
                    kf_every=args.kf_every,
                    voxel_size=args.voxel_size,
                    max_frames=args.max_frames,
                )
            except Exception as e:
                print(f"[ERR] {scene}-{seq}: {e}")


if __name__ == "__main__":
    main()
