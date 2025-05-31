import os
import os.path as osp
import numpy as np
import cv2
import open3d as o3d
from typing import Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Camera intrinsics for 7-Scenes dataset
INTRINSIC = (525, 525, 320, 240)  # fx, fy, cx, cy

class SevenScenes3DReconstructor:
    """
    Advanced 3D reconstruction pipeline for 7-Scenes dataset.
    Combines RGB-D fusion, feature matching, and point cloud optimization.
    """
    
    def __init__(self, intrinsics=INTRINSIC):
        self.fx, self.fy, self.cx, self.cy = intrinsics
        
        # Build intrinsic matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Feature detector for pose estimation
        self.detector = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
    def load_frame_data(self, frame_path: str) -> Dict:
        """Load RGB, depth, and pose data for a frame."""
        # Extract frame directory and name
        frame_dir = osp.dirname(frame_path)
        frame_name = osp.basename(frame_path).replace('.color.png', '')
        
        # Load RGB image
        rgb_path = osp.join(frame_dir, f"{frame_name}.color.png")
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise ValueError(f"Could not load RGB image: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load depth image (using depth.proj.png for better alignment)
        depth_path = osp.join(frame_dir, f"{frame_name}.depth.proj.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            # Fallback to regular depth if projected depth not available
            depth_path = osp.join(frame_dir, f"{frame_name}.depth.png")
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # Convert depth to meters
        depth = depth.astype(np.float32) / 1000.0
        depth[depth == 65.535] = 0  # Invalid depth
        depth[depth > 10.0] = 0     # Remove far points
        depth[depth < 0.1] = 0      # Remove too close points
        
        # Load pose if available (only for first frame in test)
        pose_path = osp.join(frame_dir, f"{frame_name}.pose.txt")
        pose = None
        if osp.exists(pose_path):
            pose = np.loadtxt(pose_path, dtype=np.float32)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'pose': pose,
            'frame_name': frame_name
        }
    
    def depth_to_pointcloud(self, depth: np.ndarray, rgb: np.ndarray, 
                           pose: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Convert depth image to 3D point cloud."""
        h, w = depth.shape
        
        # Create pixel coordinate grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Valid depth mask
        valid_mask = depth > 0
        
        # Back-project to 3D camera coordinates
        z = depth[valid_mask]
        x = (xx[valid_mask] - self.cx) * z / self.fx
        y = (yy[valid_mask] - self.cy) * z / self.fy
        
        # Stack to get 3D points in camera frame
        points_cam = np.stack([x, y, z], axis=-1)
        
        # Get corresponding colors
        colors = rgb[valid_mask] / 255.0
        
        # Transform to world coordinates if pose is provided
        if pose is not None:
            points_world = (pose[:3, :3] @ points_cam.T).T + pose[:3, 3]
            return points_world, colors
        
        return points_cam, colors
    
    def estimate_relative_pose(self, frame1_data: Dict, frame2_data: Dict) -> np.ndarray:
        """
        Estimate relative pose between two frames using feature matching and PnP.
        """
        # Extract features from both images
        gray1 = cv2.cvtColor(frame1_data['rgb'], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2_data['rgb'], cv2.COLOR_RGB2GRAY)
        
        kp1, desc1 = self.detector.detectAndCompute(gray1, None)
        kp2, desc2 = self.detector.detectAndCompute(gray2, None)
        
        if desc1 is None or desc2 is None:
            return np.eye(4, dtype=np.float32)
        
        # Match features
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 20:
            return np.eye(4, dtype=np.float32)
        
        # Get matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # Get 3D points for first frame
        depth1 = frame1_data['depth']
        pts3d = []
        pts2d = []
        
        for i, (x, y) in enumerate(pts1):
            x, y = int(x), int(y)
            if 0 <= x < depth1.shape[1] and 0 <= y < depth1.shape[0]:
                z = depth1[y, x]
                if z > 0:
                    # Back-project to 3D
                    x3d = (x - self.cx) * z / self.fx
                    y3d = (y - self.cy) * z / self.fy
                    pts3d.append([x3d, y3d, z])
                    pts2d.append(pts2[i])
        
        if len(pts3d) < 10:
            return np.eye(4, dtype=np.float32)
        
        pts3d = np.array(pts3d, dtype=np.float32)
        pts2d = np.array(pts2d, dtype=np.float32)
        
        # Solve PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, self.K, None,
            iterationsCount=1000,
            reprojectionError=3.0,
            confidence=0.99
        )
        
        if not success:
            return np.eye(4, dtype=np.float32)
        
        # Convert to transformation matrix
        R_rel, _ = cv2.Rodrigues(rvec)
        T_rel = np.eye(4, dtype=np.float32)
        T_rel[:3, :3] = R_rel
        T_rel[:3, 3] = tvec.flatten()
        
        return T_rel
    
    def refine_depth_bilateral(self, depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering to refine depth maps."""
        # Convert depth to uint16 for bilateral filter
        depth_mm = (depth * 1000).astype(np.uint16)
        
        # Apply bilateral filter
        depth_filtered = cv2.bilateralFilter(
            depth_mm.astype(np.float32),
            d=5,
            sigmaColor=10,
            sigmaSpace=5
        )
        
        # Convert back to meters
        depth_filtered = depth_filtered / 1000.0
        
        # Preserve original valid/invalid mask
        mask = depth > 0
        depth_filtered[~mask] = 0
        
        return depth_filtered
    
    def reconstruct_sequence(self, sequence_path: str, output_path: str,
                           kf_every: int = 20, voxel_size: float = 7.5e-3,
                           max_frames: int = None):
        """
        Main reconstruction pipeline for a sequence.
        """
        print(f"Reconstructing sequence: {sequence_path}")
        
        # Find all color frames
        frame_files = sorted([f for f in os.listdir(sequence_path) 
                            if f.endswith('.color.png')])
        
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        # Select keyframes
        keyframe_indices = list(range(0, len(frame_files), kf_every))
        print(f"Using {len(keyframe_indices)} keyframes out of {len(frame_files)} total frames")
        
        # Initialize point cloud
        combined_points = []
        combined_colors = []
        
        # Keep track of camera poses
        poses = []
        
        # Process frames
        for i, kf_idx in enumerate(tqdm(keyframe_indices, desc="Processing keyframes")):
            frame_file = frame_files[kf_idx]
            frame_path = osp.join(sequence_path, frame_file)
            
            # Load frame data
            frame_data = self.load_frame_data(frame_path)
            
            # Refine depth
            frame_data['depth'] = self.refine_depth_bilateral(
                frame_data['depth'], frame_data['rgb']
            )
            
            # Estimate pose
            if i == 0:
                # First frame - use provided pose or identity
                if frame_data['pose'] is not None:
                    current_pose = frame_data['pose']
                else:
                    current_pose = np.eye(4, dtype=np.float32)
            else:
                # Estimate relative pose from previous keyframe
                prev_frame_file = frame_files[keyframe_indices[i-1]]
                prev_frame_path = osp.join(sequence_path, prev_frame_file)
                prev_frame_data = self.load_frame_data(prev_frame_path)
                
                # Get relative transformation
                T_rel = self.estimate_relative_pose(prev_frame_data, frame_data)
                
                # Chain with previous pose
                current_pose = poses[-1] @ np.linalg.inv(T_rel)
            
            poses.append(current_pose)
            
            # Convert to point cloud
            points, colors = self.depth_to_pointcloud(
                frame_data['depth'], frame_data['rgb'], current_pose
            )
            
            # Filter outliers locally
            if len(points) > 100:
                pcd_temp = o3d.geometry.PointCloud()
                pcd_temp.points = o3d.utility.Vector3dVector(points)
                pcd_temp, ind = pcd_temp.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0
                )
                points = np.asarray(pcd_temp.points)
                colors = colors[ind]
            
            combined_points.append(points)
            combined_colors.append(colors)
        
        # Combine all points
        all_points = np.vstack(combined_points)
        all_colors = np.vstack(combined_colors)
        
        print(f"Total points before filtering: {len(all_points)}")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
        
        # Voxel downsampling
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        print(f"Final points after filtering: {len(pcd.points)}")
        
        # Save point cloud
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved point cloud to: {output_path}")
        
        return pcd


def main():
    parser = argparse.ArgumentParser(description='3D Reconstruction for 7-Scenes Dataset')
    parser.add_argument('--mode', type=str, default='all', 
                      choices=['single', 'all'],
                      help='Reconstruction mode: single sequence or all test sequences')
    parser.add_argument('--sequence', type=str, default=None,
                      help='Path to single sequence (for single mode)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output path for single sequence (for single mode)')
    parser.add_argument('--kf_every', type=int, default=20,
                      help='Keyframe selection interval')
    parser.add_argument('--voxel_size', type=float, default=7.5e-3,
                      help='Voxel size for downsampling')
    parser.add_argument('--max_frames', type=int, default=None,
                      help='Maximum number of frames to process (for debugging)')
    
    args = parser.parse_args()
    
    # Initialize reconstructor
    reconstructor = SevenScenes3DReconstructor()
    
    if args.mode == 'single':
        if not args.sequence or not args.output:
            parser.error("--sequence and --output required for single mode")
        
        reconstructor.reconstruct_sequence(
            args.sequence, args.output,
            kf_every=args.kf_every,
            voxel_size=args.voxel_size,
            max_frames=args.max_frames
        )
    
    else:  # all mode
        # Define all test sequences
        test_sequences = [
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
        
        # Create output directory
        os.makedirs("test", exist_ok=True)
        
        # Process each sequence
        for scene, seq in test_sequences:
            sequence_path = f"7SCENES/{scene}/test/{seq}"
            output_path = f"test/{scene}-{seq}.ply"
            
            try:
                reconstructor.reconstruct_sequence(
                    sequence_path, output_path,
                    kf_every=args.kf_every,
                    voxel_size=args.voxel_size,
                    max_frames=args.max_frames
                )
            except Exception as e:
                print(f"Error processing {scene}-{seq}: {e}")
                continue

if __name__ == "__main__":
    main()
