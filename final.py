import os
import cv2
import numpy as np
from glob import glob
import open3d as o3d

# Camera intrinsics for 7-Scenes Kinect
fx = 525
fy = 525
cx = 320
cy = 240
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

SCENES_ROOT = "./7SCENES"
OUTPUT_DIR = "test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENE_LIST = [d for d in os.listdir(SCENES_ROOT) if os.path.isdir(os.path.join(SCENES_ROOT, d))]
FRAME_THRESHOLD = 20  # skip test sequences with <= 20 frames (sparse)

def find_sequences(scene_path):
    test_seqs = sorted([os.path.join(scene_path, 'test', d) for d in os.listdir(os.path.join(scene_path, 'test')) if os.path.isdir(os.path.join(scene_path, 'test', d))])
    return test_seqs

def load_frame_files(seq_path):
    color_files = sorted(glob(os.path.join(seq_path, '*.color.png')))
    depth_files = sorted(glob(os.path.join(seq_path, '*.depth.png')))
    pose_files = sorted(glob(os.path.join(seq_path, '*.pose.txt')))
    return list(zip(color_files, depth_files, pose_files))

def load_pose(pose_file):
    with open(pose_file, 'r') as f:
        lines = f.readlines()
        mat = np.array([[float(num) for num in line.strip().split()] for line in lines])
    return mat  # 4x4 camera-to-world

def backproject_depth(depth_img, rgb_img):
    h, w = depth_img.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_img / 1000.0  # mm to meters
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    valid = (z > 0) & (z < 10)
    xyz = np.stack((x, y, z), axis=-1)[valid]
    rgb = rgb_img[valid]
    return xyz, rgb

def process_test_sequence(scene, seq_path):
    frames = load_frame_files(seq_path)
    if not frames:
        return
    # Use first frame's pose for calibration
    _, _, first_pose_file = frames[0]
    T0 = load_pose(first_pose_file)
    all_points = []
    all_colors = []
    for color_file, depth_file, _ in frames:
        rgb = cv2.imread(color_file, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        pts_cam, colors = backproject_depth(depth, rgb)
        # Transform to world using T0
        pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
        pts_world = (T0 @ pts_cam_h.T).T[:, :3]
        all_points.append(pts_world)
        all_colors.append(colors)

    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        # Save as .ply
        seq_id = os.path.basename(seq_path)
        ply_path = os.path.join(OUTPUT_DIR, f"{scene}-{seq_id}.ply")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors / 255.0)
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"Saved: {ply_path} ({all_points.shape[0]} points)")

def is_dense_sequence(seq_path):
    seq_name = os.path.basename(seq_path).lower()
    if 'sparse' in seq_name:
        return False
    # Count number of color frames
    color_files = glob(os.path.join(seq_path, '*.color.png'))
    if len(color_files) <= FRAME_THRESHOLD:
        return False
    return True

if __name__ == "__main__":
    for scene in sorted(SCENE_LIST):
        scene_path = os.path.join(SCENES_ROOT, scene)
        test_seqs = find_sequences(scene_path)
        for seq_path in test_seqs:
            if not is_dense_sequence(seq_path):
                print(f"Skipping sparse sequence: {seq_path}")
                continue
            process_test_sequence(scene, seq_path)
    print("\nAll dense test sequences processed. Results are in the 'test/' folder, ready for submission.")
