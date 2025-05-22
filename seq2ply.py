import os 
import os.path as osp
from typing import List,Dict,Tuple

from PIL import Image
import cv2
import numpy as np
import open3d as o3d


INTRINSINC = (525, 525, 320, 240)  # fx, fy, cx, cy


def imread_cv2(path:str, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def depthmap_to_world_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Projects a depth map into 3D world coordinates using camera intrinsics and optional pose.
    
    Args:
        depthmap (H x W): Depth values (in camera space).
        intrinsics (3 x 3): Camera intrinsic matrix.
        pose (optional, 4 x 4 or 4 x 3): Camera-to-world transformation.
        pseudo_focal (optional, H x W): Per-pixel focal length override.
        
    Returns:
        pts_world (H x W x 3): 3D point cloud in world coordinates.
        valid_mask (H x W): Boolean mask indicating valid (non-zero) depth pixels.
    """

    H, W = depthmap.shape
    camera_intrinsics = np.float32(camera_intrinsics)
    
    # Extract intrinsic parameters
    assert camera_intrinsics[0, 1] == 0.0 and camera_intrinsics[1, 0] == 0.0
    fu,fv = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cu, cv = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # Generate pixel coordinate grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # u: cols, v: rows

    # Backproject depth to 3D camera coordinates
    z = depthmap
    x = (u - cu) * z / fu
    y = (v - cv) * z / fv
    pts_cam = np.stack((x, y, z), axis=-1).astype(np.float32)

    # Mark valid points (depth > 0)
    valid_mask = z > 0.0

    # Transform to world coordinates if pose is given
    if camera_pose is not None:
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        pts_world = np.einsum("ik, vuk -> vui", R, pts_cam) + t
    else:
        pts_world = pts_cam

    return pts_world, valid_mask

class SevenSceneSequence:
    def __init__(
            self,
            seq_dir_path,
        ):
        self.seq_dir_path = seq_dir_path    
        # Find all the filenames end with ".color.png" 
        # and check if corresponding ".proj.png" and ".pose.txt" exists

        _color_files = [f for f in os.listdir(seq_dir_path) if f.endswith(".color.png")]
        frame_names = [f.rstrip(".color.png") for f in _color_files]

        self.valid_frame_names = []
        for name in frame_names:
            proj_path = osp.join(seq_dir_path, f"{name}.depth.proj.png")
            pose_path = osp.join(seq_dir_path, f"{name}.pose.txt")
            if osp.isfile(proj_path) and osp.isfile(pose_path):
                self.valid_frame_names.append(name)
        self.valid_frame_names = sorted(self.valid_frame_names)

        print(f"{len(self.valid_frame_names)} frames collected in {self.seq_dir_path}!!")
        print(f"{len(_color_files) - len(self.valid_frame_names)} rgb frames miss .proj.png or .pose.txt!!")
        
        
    def get_views(self,kf_every = 200)->List[Dict]:
        
        names = self.valid_frame_names[::kf_every] # select 1 out of every kf_every frames for reconstruction
        
        views = [] 
        """
        For each view(key frame), we compute the following metric
        """
        for idx,name in enumerate(names):
            view = dict()

            impath = osp.join(self.seq_dir_path, f"{name}.color.png")
            depthpath = osp.join(self.seq_dir_path, f"{name}.depth.proj.png")
            posepath = osp.join(self.seq_dir_path, f"{name}.pose.txt")
            view["name"] = f'{self.seq_dir_path}/{name}'

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))
            
            width, height = Image.fromarray(rgb_image).size
            assert (width,height) == (640,480)
            view['img'] = (rgb_image / 255.0 ).astype(np.float32)# Normalize to 0 to 1 for open3d format
            view["true_shape"] = np.int32((height, width))

            depthmap[depthmap == 65535] = 0
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0
            assert np.isfinite(depthmap).all(), \
                f"NaN in depthmap for view {view['name']}"
            view['depthmap'] = depthmap

            camera_pose = np.loadtxt(posepath).astype(np.float32)
            fx, fy, cx, cy = INTRINSINC ### NOTE: This intrinsic does not match with that on internet
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            assert np.isfinite(camera_pose).all(), \
                f"NaN in camera pose for view {view['name']}"
            
            view['camera_pose'] = camera_pose
            view['camera_intrinsics'] = intrinsics

            # encode the image
            pts3d, valid_mask = depthmap_to_world_coordinates(**view)
            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)
            view["img_mask"] = True
            
            # check all datatypes
            for key, val in view.items():
                res, err_msg = self._is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view['name']}"
            
            views.append(view)
            
        for view in views:
            height, width = view['true_shape']
            assert width >= height, ValueError("Width > Height")
        
        return views
    
    def _is_good_type(self,key, v):
        """returns (is_good, err_msg)"""
        if isinstance(v, (str, int, tuple)):
            return True, None
        if v.dtype not in (np.float32, bool, np.int32, np.int64, np.uint8):
            return False, f"bad {v.dtype=}"
        return True, None

def seq2ply(seq_dir_path, ply_path, kf_every = 1, crop_size = None, voxel_grid_size = None):
    """
    Converts a sequence of frames into a single 3D point cloud and saves it as a .ply file.

    Parameters:
        seq_dir_path (str): Path to the sequence directory. This directory should contain multiple
                            frame subdirectories or files, each including:
                                - .color.png: RGB image
                                - .proj.png: Projected depth or coordinate image
                                - .pose.txt: Camera pose matrix (usually 4x4)

        ply_path (str): Destination path for the output .ply point cloud file.
        kf_every (int): Selec key frame every "kf_every" frames for building points cloud

    Description:
        This function reads all frames in the given sequence directory, reconstructs 3D points using the color,
        projection, and pose data, merges them into a single point cloud, and writes the result to
        a .ply file.
    """
    # Step 1: Collect the necessary information of frames for reconstruction
    seq = SevenSceneSequence(seq_dir_path = seq_dir_path )  
    views = seq.get_views(kf_every = kf_every)
    pts_gt_all, images_all,  masks_all = [], [], []

    # Step 2: Only believe the central information of the camera
    assert crop_size is None \
        or isinstance(crop_size, int), \
        "crop_size must be None or an integer"
    
    for _, view in enumerate(views):
        image = view["img"]  # W,H,3
        mask = view["valid_mask"]    # W,H
        pts_gt = view['pts3d'] # W,H,3
        
        # Center on the given window size
        if crop_size is not None:
            H, W = image.shape[:2]
            if crop_size > H or crop_size > W:
                print(f"Warning: Adjust crop_size({crop_size}) since it exceeds H({H}) or W({W})")
                crop_size = min(W,H)
            _shift = crop_size//2
            cx,cy = W // 2,H // 2
            l, t = cx - _shift, cy - _shift # left, top
            r, b = cx + _shift, cy + _shift # right, bottom
            
            image = image[t:b, l:r]
            mask = mask[t:b, l:r]
            pts_gt = pts_gt[t:b, l:r]

        #### Align predicted 3D points to the ground truth
        images_all.append( image[None, ...] )
        pts_gt_all.append( pts_gt[None, ...] )
        masks_all.append( mask[None, ...] )


    # Step 3: Build the 3D points map
    images_all = np.concatenate(images_all, axis=0)
    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)
    pts_gt_all_masked = pts_gt_all[masks_all > 0]
    images_all_masked = images_all[masks_all > 0]

    #save_params = {}
    #save_params["images_all"] = images_all
    #save_params["pts_gt_all"] = pts_gt_all
    #save_params["masks_all"] = masks_all
    #np.save(_path_,save_params,)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(
        pts_gt_all_masked.reshape(-1, 3)
    )
    pcd_gt.colors = o3d.utility.Vector3dVector(
        images_all_masked.reshape(-1, 3)
    )
    print(f'Points Cloud has {len(pcd_gt.points)} points')
    if voxel_grid_size is not None:
        pcd_gt = pcd_gt.voxel_down_sample(voxel_size=voxel_grid_size)
        print(f'After downsample, Points Cloud has {len(pcd_gt.points)} points')

    o3d.io.write_point_cloud(ply_path, pcd_gt, )

def visualize_ply_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)

    if not pcd.has_points():
        print("Points Cloud file has no results")
        return

    print(f'Points Cloud has {len(pcd.points)} points')
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)
    
if __name__ == '__main__':
    sequence_path = "7SCENES/chess/test/seq-03"
    ply_path = "test/chess-seq-03.ply"
    seq2ply(sequence_path, ply_path, kf_every =20, voxel_grid_size = 7.5e-3) 
    