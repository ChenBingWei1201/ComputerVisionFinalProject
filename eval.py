import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from collections import defaultdict


def load_point_cloud(ply_path):
    """
    Load a .ply file and return an (N×3) NumPy float32 array of its vertices.
    Requires: pip install plyfile
    """
    ply = PlyData.read(ply_path)
    xyz = np.vstack([
        ply['vertex']['x'],
        ply['vertex']['y'],
        ply['vertex']['z']
    ]).T
    return xyz.astype(np.float32)


def compute_accuracy_completeness(pc_pred, pc_gt):
    """
    Given two point clouds:
      pc_pred: (N_pred × 3) NumPy array of predicted points P
      pc_gt:   (N_gt   × 3) NumPy array of ground-truth points G

    Returns:
      accuracy     = median_{p ∈ P} [ min_{g ∈ G} ‖p − g‖₂ ]
      completeness = median_{g ∈ G} [ min_{p ∈ P} ‖g − p‖₂ ]
    """
    tree_gt = cKDTree(pc_gt)     # for queries “predicted → ground-truth”
    tree_pred = cKDTree(pc_pred)  # for queries “ground-truth → predicted”

    # 1) For each predicted point p ∈ P, find distance to nearest ground-truth g ∈ G
    dists_pred_to_gt, _ = tree_gt.query(pc_pred)    # shape = (N_pred,)
    accuracy = float(np.median(dists_pred_to_gt))

    # 2) For each ground-truth point g ∈ G, find distance to nearest predicted p ∈ P
    dists_gt_to_pred, _ = tree_pred.query(pc_gt)    # shape = (N_gt,)
    completeness = float(np.median(dists_gt_to_pred))

    return accuracy, completeness


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, choices=["test", "bonus"], default="test",
                        help="Which dataset split to evaluate (default: test)")
    args = parser.parse_args()
    eval_set = args.set
    # 1. Directory containing your predicted “.ply” files (test split):
    pred_dir = eval_set

    # 2. Directory containing ground-truth “.ply” files for test split:
    gt_dir = os.path.join("ground_truth_data", eval_set)

    # 3. Gather all predicted .ply files
    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "*.ply")))
    if len(pred_paths) == 0:
        raise RuntimeError(
            f"No .ply files found in '{pred_dir}'. Make sure your path is correct."
        )

    # We will store per‐scene accuracy/completeness lists
    acc_per_scene = defaultdict(list)
    comp_per_scene = defaultdict(list)

    for pred_path in tqdm(pred_paths, desc="Evaluating sequences"):
        filename = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, filename)

        if not os.path.exists(gt_path):
            print(f"{filename}: ground-truth file not found in '{gt_dir}' → SKIPPING")
            continue

        # Load both point clouds
        pc_pred = load_point_cloud(pred_path)
        pc_gt = load_point_cloud(gt_path)

        # Compute Accuracy & Completeness for this sequence
        acc, comp = compute_accuracy_completeness(pc_pred, pc_gt)

        # Print per‐sequence result
        print(f"{filename}: {acc} {comp}")

        # Extract scene name from filename (everything before "-seq-")
        if "-seq-" in filename:
            scene_name = filename.split("-seq-", 1)[0]
        else:
            # If format is unexpected, skip adding to any scene
            continue

        # Accumulate into per‐scene lists
        acc_per_scene[scene_name].append(acc)
        comp_per_scene[scene_name].append(comp)

    # After all sequences have been processed, compute per‐scene means:
    scene_acc_means = []
    scene_comp_means = []

    for scene in sorted(acc_per_scene.keys()):
        acc_list = acc_per_scene[scene]
        comp_list = comp_per_scene[scene]

        mean_acc_scene = sum(acc_list) / len(acc_list)
        mean_comp_scene = sum(comp_list) / len(comp_list)

        scene_acc_means.append(mean_acc_scene)
        scene_comp_means.append(mean_comp_scene)

    # Now compute the final average across all scenes:
    if len(scene_acc_means) == 0:
        raise RuntimeError("No valid scenes found to average.")

    final_acc = sum(scene_acc_means) / len(scene_acc_means)
    final_comp = sum(scene_comp_means) / len(scene_comp_means)

    # Print the final score (average of every scene's average)
    print(f"\nFinal (scene‐averaged) Accuracy & Completeness:")
    print(f"acc: {final_acc}    comp: {final_comp}")
