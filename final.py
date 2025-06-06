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

# Camera intrinsics for 7-Scenes dataset
INTRINSIC = (525, 525, 320, 240)  # fx, fy, cx, cy

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
        depth[(depth > 4) | (depth < 0.2)] = 0

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
        """Back-project depth image to 3D point cloud."""
        h, w = depth.shape

        # Create pixel coordinate grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))

        # Valid depth mask
        mask = depth > 0

        # Back-project to 3D camera coordinates
        z = depth[mask]
        x = (xx[mask] - self.cx) * z / self.fx
        y = (yy[mask] - self.cy) * z / self.fy

        # Stack to get 3D points in camera frame
        pts_cam = np.stack([x, y, z], axis=-1)

        # Get corresponding colors
        colors = rgb[mask] / 255.0

        # Transform to world coordinates if pose is provided
        if pose is not None:
            pts_world = (pose[:3, :3] @ pts_cam.T).T + pose[:3, 3]
            return pts_world, colors
        
        return pts_cam, colors

    # ------------------------------------------------------------------
    # ICP refinement
    # ------------------------------------------------------------------
    def refine_pose_with_icp(
        self, source_points: np.ndarray, target_points: np.ndarray, init_T: np.ndarray
    ) -> np.ndarray:
        """
        Refine initial pose using ICP.
        """
        # Create Open3D point clouds from numpy arrays
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))

        # Initialize transformation with PnP result
        transformation = init_T.copy()

        # Multi-scale ICP: coarse (5cm) -> medium (2cm) -> fine (1cm)
        # Coarse scales avoid local minima, fine scales achieve precision
        for voxel, max_iter in zip([0.05, 0.02, 0.01], [20, 20, 20]):
            # Downsample point clouds for current scale
            # Reduces computational cost and provides multi-resolution alignment
            src_down = src.voxel_down_sample(voxel)
            tgt_down = tgt.voxel_down_sample(voxel)

            # Estimate surface normals for point-to-plane ICP
            # Uses 2x voxel radius to capture local surface geometry
            src_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30)
            )

            # Estimate normals for target point cloud as well
            tgt_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30)
            )

            # Perform point-to-plane ICP registration
            # Point-to-plane is more accurate than point-to-point for planar surfaces
            reg = o3d.pipelines.registration.registration_icp(
                src_down,  # Source point cloud (frame1)
                tgt_down,  # Target point cloud (frame2)
                max_correspondence_distance=voxel * 2,  # Adaptive threshold based on current scale
                init=transformation,  # Initialize with previous scale result
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
            )

            # Update transformation for next scale iteration
            transformation = reg.transformation

        return transformation

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
        # Convert RGB images to grayscale for feature detection
        gray1 = cv2.cvtColor(frame1["rgb"], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2["rgb"], cv2.COLOR_RGB2GRAY)
        
        # Extract SIFT features and descriptors from both frames
        kp1, desc1 = self.detector.detectAndCompute(gray1, None)
        kp2, desc2 = self.detector.detectAndCompute(gray2, None)

        # Return identity if no features detected
        if desc1 is None or desc2 is None:
            return np.eye(4, dtype=np.float32)

        # Find feature correspondences using k-nearest neighbors (k=2)
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        # Ratio test: distance to closest / distance to second closest < 0.75
        good = []
        for m_n in matches:
            if len(m_n) == 2:  # Ensure we have 2 nearest neighbors
                m, n = m_n
                if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                    good.append(m)
        
        # Need minimum 20 matches for reliable PnP estimation
        if len(good) < 20:
            return np.eye(4, dtype=np.float32)

        # Extract 2D pixel coordinates from matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])  # Points in frame1
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])  # Corresponding points in frame2

        # Build 3D-2D correspondences for PnP
        # Use depth from frame1 to get 3D points, match with 2D points in frame2
        depth1 = frame1["depth"]
        pts3d, pts2d = [], []
        for (u, v), p2 in zip(pts1, pts2):
            x, y = int(u), int(v)
            # Check if pixel coordinates are within image bounds
            if 0 <= x < depth1.shape[1] and 0 <= y < depth1.shape[0]:
                z = depth1[y, x]  # Get depth value
                if z > 0:  # Valid depth measurement
                    # Back-project 2D pixel to 3D point using camera intrinsics
                    pts3d.append(
                        [(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z]
                    )
                    pts2d.append(p2)  # Corresponding 2D point in frame2
        
        # Need minimum 10 3D-2D correspondences for PnP
        if len(pts3d) < 10:
            return np.eye(4, dtype=np.float32)

        # Convert to numpy arrays for PnP solver
        pts3d = np.array(pts3d, dtype=np.float32)
        pts2d = np.array(pts2d, dtype=np.float32)

        # Solve PnP with RANSAC to estimate camera pose
        # Given 3D points in frame1 coordinate system and their 2D projections in frame2
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d,  # 3D points in frame1 coordinate system
            pts2d,  # 2D points in frame2 image
            self.K,  # Camera intrinsic matrix
            None,   # No distortion coefficients
            iterationsCount=1000,      # RANSAC iterations
            reprojectionError=3.0,     # Inlier threshold in pixels
            confidence=0.99,           # Desired confidence level
        )
        if not success:
            return np.eye(4, dtype=np.float32)

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Build 4x4 transformation matrix T_21 (frame2 to frame1)
        T_rel = np.eye(4, dtype=np.float32)
        T_rel[:3, :3] = R         # Rotation part
        T_rel[:3, 3] = tvec.flatten()  # Translation part

        # ------------- ICP refinement -------------
        # Generate dense point clouds from depth images for ICP alignment
        src_points, _ = self.depth_to_pointcloud(frame1["depth"], frame1["rgb"])
        tgt_points, _ = self.depth_to_pointcloud(frame2["depth"], frame2["rgb"])
        
        # Only perform ICP if we have sufficient points (>2000 each)
        # ICP works better with dense point clouds
        if len(src_points) > 2000 and len(tgt_points) > 2000:
            # Refine PnP result using multi-scale ICP for better accuracy
            T_rel = self.refine_pose_with_icp(src_points, tgt_points, T_rel)

        return T_rel

    # ------------------------------------------------------------------
    # Depth filtering
    # ------------------------------------------------------------------
    def refine_depth_bilateral(self, depth: np.ndarray) -> np.ndarray:
        """
        Refine depth map using bilateral filter.
        """
        # Convert depth to mm
        depth_mm = (depth * 1000).astype(np.float32)

        # Apply bilateral filter
        depth_f = cv2.bilateralFilter(depth_mm, 5, 10, 5) / 1000.0

        # Set invalid depths to 0
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
        voxel_size: float = 5e-3,
        max_frames: Optional[int] = None,
    ):
        """
        Reconstruct a sequence of frames.
        """
        # Get frame files
        print(f"=== Reconstructing: {sequence_path}")

        # Find all color frames
        frame_files = sorted(f for f in os.listdir(sequence_path) if f.endswith(".color.png"))

        if max_frames:
            frame_files = frame_files[: max_frames]

        # Select keyframes
        key_idx = list(range(0, len(frame_files), kf_every))

        # Initialize point cloud
        comb_points, comb_colors = [], []
        
        # Keep track of camera poses
        poses = [] 

        # Process frames
        for i, idx in enumerate(tqdm(key_idx, desc="Keyframes")):
            frame_path = osp.join(sequence_path, frame_files[idx])
            
            # Load frame data
            frame = self.load_frame_data(frame_path)

            # Refine depth
            frame["depth"] = self.refine_depth_bilateral(frame["depth"])

            # Estimate pose
            if i == 0:
                # First frame: use provided pose or identity
                pose = frame["pose"] if frame["pose"] is not None else np.eye(
                    4, dtype=np.float32
                )
            else:
                # Estimate relative pose from previous keyframe
                prev_path = osp.join(sequence_path, frame_files[key_idx[i - 1]])
                prev_frame = self.load_frame_data(prev_path)
                
                # Get relative transformation
                T_rel = self.estimate_relative_pose(prev_frame, frame)

                # Chain with previous pose
                pose = poses[-1] @ np.linalg.inv(T_rel)

            poses.append(pose)

            # Convert to point cloud
            points, colors = self.depth_to_pointcloud(frame["depth"], frame["rgb"], pose)

            # Filter outliers locally
            if len(points) > 100:
                pcd_tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                pcd_tmp, ind = pcd_tmp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                points, colors = np.asarray(pcd_tmp.points), colors[ind]

            comb_points.append(points)
            comb_colors.append(colors)

        # Combine all points and colors
        all_points = np.vstack(comb_points)
        all_colors = np.vstack(comb_colors)

        print(f"Points before filter: {len(all_points)}")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1) # 517改的

        # Voxel downsampling
        pcd = pcd.voxel_down_sample(voxel_size)

        print(f"Points after filter : {len(pcd.points)}")

        # Save point cloud
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved: {output_path}")

        return pcd

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="3D Reconstruction for 7-Scenes Dataset")
    parser.add_argument("--mode", type=str, choices=["single", "all"], default="all", help="Reconstruction mode: single sequence or all test sequences")
    parser.add_argument("--sequence", type=str, default=None, help="Path to single sequence (for single)")
    parser.add_argument("--output", type=str, default=None, help="Path to output .ply (for single)")
    parser.add_argument("--kf_every", type=int, default=20, help="Keyframe selection interval")
    parser.add_argument("--voxel_size", type=float, default=5e-3, help="Voxel size for downsampling")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process (for debugging)")
    parser.add_argument("--set", type=str, default="test", choices=["test", "bonus"], help="Set to process: 'test' or 'bonus' sequences")
    
    args = parser.parse_args()

    # Initialize reconstructor
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
    else: # all mode
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

        bonus_seqs = [
            ("chess", "sparse-seq-05"),
            ("fire", "sparse-seq-04"),
            ("pumpkin", "sparse-seq-07"),
            ("stairs", "sparse-seq-04")
        ]

        # Create output directory
        os.makedirs("test", exist_ok=True)
        os.makedirs("bonus", exist_ok=True)
        # Process each test sequence
        if args.set == "test":
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

        if args.set == "bonus":
            for scene, seq in bonus_seqs:
                seq_path = f"7SCENES/{scene}/test/{seq}"
                out_path = f"bonus/{scene}-{seq}.ply"

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
