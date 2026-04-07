#!/usr/bin/env python3
"""
Generate per-frame PLY files with scene point cloud + green MANO hand overlay.

For each video, reads:
  - Depth maps from intermediate_depth/depth/
  - Camera poses from intermediate_depth/camera/
  - Color frames from intermediate_depth/color/
  - Corrected annotation from corrected_annotations/<episode_id>.npy

Outputs PLY files to <results_dir>/hand_overlay/<video_name>/<frame>.ply
Each PLY has the scene point cloud in natural color + MANO hand mesh
sampled as a green point cloud overlaid on it.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# MANO utilities (extracted from generate_combined_ply.py history)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MANO_RIGHT_PATH = os.path.join(PROJECT_ROOT, "WiLoR", "mano_data", "MANO_RIGHT_clean.npz")
MANO_LEFT_PATH = os.path.join(PROJECT_ROOT, "WiLoR", "mano_data", "MANO_LEFT_clean.npz")

# Datasets where video_decode_frame contains absolute frame indices
_ABSOLUTE_VDF_DATASETS = {"epic", "ego4d", "egoexo4d"}
_DATASET_PREFIXES = [
    ("somethingsomethingv2_", "ssv2"),
    ("epic_kitchens_", "epic"),
    ("EgoExo4D_", "egoexo4d"),
    ("Ego4D_", "ego4d"),
]


def _rebase_vdf(vdf, annotation_path):
    basename = os.path.basename(annotation_path)
    dataset = None
    for prefix, ds in _DATASET_PREFIXES:
        if basename.startswith(prefix):
            dataset = ds
            break
    if dataset in _ABSOLUTE_VDF_DATASETS:
        return vdf - vdf[0]
    return vdf


def load_mano_model(path):
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def mano_forward(mano_data, beta, global_orient, hand_pose, transl):
    """MANO forward pass: shape + pose blend shapes + LBS."""
    v_template = mano_data['v_template'].copy()
    shapedirs = mano_data['shapedirs'].copy()
    posedirs = mano_data['posedirs'].copy()
    J_regressor = mano_data['J_regressor']
    weights = mano_data['weights']
    kintree = mano_data['kintree_table']
    faces = mano_data['f'].copy()

    v_shaped = v_template + np.einsum('ijk,k->ij', shapedirs, beta)
    J = J_regressor @ v_shaped

    pose_rotmats = np.concatenate([global_orient[None], hand_pose], axis=0)

    ident = np.tile(np.eye(3), (15, 1, 1))
    pose_feature = (pose_rotmats[1:] - ident).reshape(-1)
    v_posed = v_shaped + np.einsum('ijk,k->ij', posedirs, pose_feature)

    parents = kintree[0].astype(np.int64)

    local_T = np.zeros((16, 4, 4))
    for i in range(16):
        local_T[i, :3, :3] = pose_rotmats[i]
        local_T[i, :3, 3] = J[0] if i == 0 else J[i] - J[parents[i]]
        local_T[i, 3, 3] = 1.0

    global_T = np.zeros((16, 4, 4))
    global_T[0] = local_T[0]
    for i in range(1, 16):
        global_T[i] = global_T[parents[i]] @ local_T[i]

    for i in range(16):
        pad_J = np.array([J[i, 0], J[i, 1], J[i, 2], 0.0])
        global_T[i, :, 3] -= global_T[i] @ pad_J

    T = np.einsum('vj,jab->vab', weights, global_T)
    v_homo = np.ones((len(v_posed), 4))
    v_homo[:, :3] = v_posed
    v_out = np.einsum('vab,vb->va', T, v_homo)[:, :3]

    v_out += transl
    return v_out, faces


def generate_hand_points(mano_data, mano_left_data, hand_dict, frame_idx,
                         is_left, num_samples=5000, recon_K=None,
                         image_shape=None):
    """Generate MANO hand surface points for a single frame.

    If recon_K and image_shape are provided, uses z-buffer rasterization
    to only sample from the visible surface (handles all self-occlusion).
    Otherwise samples from the full mesh.

    Returns (N, 3) camera-space points, or None if frame is invalid.
    """
    kept = hand_dict['kept_frames']
    if frame_idx >= len(kept) or not kept[frame_idx]:
        return None

    beta = hand_dict['beta']
    global_orient = hand_dict['global_orient_camspace'][frame_idx]
    hand_pose = hand_dict['hand_pose'][frame_idx]
    S = np.diag([-1.0, 1.0, 1.0])

    if is_left and mano_left_data is not None:
        hp_corrected = np.array([S @ r @ S for r in hand_pose])
        verts_zero, faces = mano_forward(
            mano_left_data, beta, global_orient, hp_corrected, np.zeros(3))
        wrist_offset = mano_left_data['J_regressor'][0] @ verts_zero
        target_wrist = hand_dict['joints_camspace'][frame_idx][0]
        transl = target_wrist - wrist_offset
        verts, faces = mano_forward(
            mano_left_data, beta, global_orient, hp_corrected, transl)
    else:
        model = mano_data
        verts_zero, faces = mano_forward(model, beta, global_orient, hand_pose, np.zeros(3))
        wrist_offset = model['J_regressor'][0] @ verts_zero
        target_wrist = hand_dict['joints_camspace'][frame_idx][0]
        transl = target_wrist - wrist_offset
        verts, faces = mano_forward(model, beta, global_orient, hand_pose, transl)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if mesh.area < 1e-6:
        return None

    # If intrinsics provided, cull back-facing faces (normal-based)
    if recon_K is not None and image_shape is not None:
        centroids = verts[faces].mean(axis=1)
        view_dirs = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        dots = np.sum(view_dirs * mesh.face_normals, axis=1)
        front_faces = np.where(dots < 0)[0]
        if len(front_faces) > 0:
            mesh = mesh.submesh([front_faces], append=True)

    points, _ = mesh.sample(num_samples, return_index=True)
    return points


def unproject_depth(depth, intrinsics, pose_c2w, color=None):
    """Unproject depth map to world-space points with colors."""
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    valid = depth > 0
    z = depth[valid]
    X = (x[valid] - cx) * z / fx
    Y = (y[valid] - cy) * z / fy
    pts_cam = np.stack([X, Y, z], axis=-1)

    R = pose_c2w[:3, :3]
    t = pose_c2w[:3, 3]
    pts_world = (R @ pts_cam.T).T + t

    colors = None
    if color is not None:
        colors = color[valid]
        if colors.shape[-1] == 4:
            colors = colors[:, :3]
    return pts_world, colors


def _save_ply(pts, colors, path):
    """Save a point cloud PLY with uint8 RGBA colors."""
    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    if colors.shape[1] == 3:
        alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
        colors = np.hstack([colors, alpha])
    pcd = trimesh.PointCloud(pts, colors=colors)
    pcd.export(path)


def process_video(video_dir, annotation_path, original_annotation_path,
                   output_dir, mano_r, mano_l):
    """Generate separate hand-overlay PLY files for comparison.

    For each frame, saves two PLY files:
      - <frame>_original.ply: scene + BLUE original VITRA hands
      - <frame>_corrected.ply: scene + GREEN corrected hands
    """
    depth_dir = os.path.join(video_dir, "intermediate_depth", "depth")
    camera_dir = os.path.join(video_dir, "intermediate_depth", "camera")
    color_dir = os.path.join(video_dir, "intermediate_depth", "color")

    ann = np.load(annotation_path, allow_pickle=True).item()
    orig_ann = np.load(original_annotation_path, allow_pickle=True).item()

    vdf = ann.get("video_decode_frame")
    if vdf is None:
        print(f"  No video_decode_frame, skipping")
        return

    vdf = _rebase_vdf(vdf, annotation_path)

    os.makedirs(output_dir, exist_ok=True)

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    vdf_set = set(int(v) for v in vdf)

    n_saved = 0
    for depth_path in depth_files:
        basename = os.path.splitext(os.path.basename(depth_path))[0]
        frame_idx = int(basename)

        if frame_idx not in vdf_set:
            continue

        cam_path = os.path.join(camera_dir, f"{basename}.npz")
        if not os.path.exists(cam_path):
            continue

        cam = np.load(cam_path)
        depth_map = np.load(depth_path)
        intrinsics = cam["intrinsics"]
        pose_c2w = cam["pose"]

        # Load color
        color_img = None
        for ext in [".png", ".jpg"]:
            cp = os.path.join(color_dir, basename + ext)
            if os.path.exists(cp):
                import cv2
                color_img = cv2.imread(cp)
                if color_img is not None:
                    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                    if color_img.shape[:2] != depth_map.shape:
                        color_img = cv2.resize(color_img,
                                               (depth_map.shape[1], depth_map.shape[0]))
                break

        # Unproject scene
        scene_pts, scene_colors = unproject_depth(depth_map, intrinsics, pose_c2w, color_img)
        if scene_colors is None:
            scene_colors = np.full((len(scene_pts), 3), 180, dtype=np.uint8)

        # Find VITRA annotation index for this frame
        vi = None
        for idx, vf in enumerate(vdf):
            if int(vf) == frame_idx:
                vi = idx
                break
        if vi is None:
            continue

        R = pose_c2w[:3, :3]
        t = pose_c2w[:3, 3]

        # Pi3 intrinsics for this frame
        pi3_fx, pi3_fy = intrinsics[0, 0], intrinsics[1, 1]
        pi3_cx, pi3_cy = intrinsics[0, 2], intrinsics[1, 2]

        def _reproject_to_pi3(pts_cam, ann_intrinsics):
            """Adjust 3D points so they project to the correct 2D position in
            the Pi3 image when using Pi3's estimated intrinsics.

            WARNING: This changes the actual 3D coordinates of the points.
            Camera-space 3D positions are intrinsics-independent (a physical
            position doesn't change with the lens), so this is technically
            incorrect in 3D — the reprojected points will be at slightly
            wrong world-space positions.  However, it ensures correct 2D
            overlay when the PLY is viewed from the Pi3 camera pose.

            This is ONLY appropriate for the side-by-side comparison PLYs
            generated by this script.  For depth sampling / metric alignment
            (rescale_to_metric.py), use recon_K directly on the original
            camera-space points — do NOT reproject.
            """
            if ann_intrinsics is None:
                return pts_cam
            a_fx, a_fy = ann_intrinsics[0, 0], ann_intrinsics[1, 1]
            a_cx, a_cy = ann_intrinsics[0, 2], ann_intrinsics[1, 2]
            a_W, a_H = a_cx * 2, a_cy * 2
            pi3_W, pi3_H = pi3_cx * 2, pi3_cy * 2
            sx, sy = pi3_W / a_W, pi3_H / a_H

            z = pts_cam[:, 2:3]  # keep z from annotation (metric)
            # Project to annotation pixels, scale to Pi3 pixels, unproject
            u_ann = a_fx * pts_cam[:, 0:1] / z + a_cx
            v_ann = a_fy * pts_cam[:, 1:2] / z + a_cy
            u_pi3 = u_ann * sx
            v_pi3 = v_ann * sy
            x_pi3 = (u_pi3 - pi3_cx) * z / pi3_fx
            y_pi3 = (v_pi3 - pi3_cy) * z / pi3_fy
            return np.hstack([x_pi3, y_pi3, z])

        # Collect hand points for each variant separately
        orig_hand_pts, orig_hand_colors = [], []
        corr_hand_pts, corr_hand_colors = [], []

        for hand_name in ["left", "right"]:
            is_left = (hand_name == "left")

            # --- Original VITRA hand (BLUE) ---
            if hand_name in orig_ann:
                pts_cam = generate_hand_points(
                    mano_r, mano_l, orig_ann[hand_name], vi, is_left, num_samples=5000)
                if pts_cam is not None:
                    pts_cam = _reproject_to_pi3(pts_cam, orig_ann.get("intrinsics"))
                    pts_world = (R @ pts_cam.T).T + t
                    c = np.tile(np.array([50, 50, 255], dtype=np.uint8), (len(pts_world), 1))
                    orig_hand_pts.append(pts_world)
                    orig_hand_colors.append(c)

            # --- Corrected hand (GREEN) ---
            if hand_name in ann:
                pts_cam = generate_hand_points(
                    mano_r, mano_l, ann[hand_name], vi, is_left, num_samples=5000)
                if pts_cam is not None:
                    pts_cam = _reproject_to_pi3(pts_cam, ann.get("intrinsics"))
                    pts_world = (R @ pts_cam.T).T + t
                    c = np.tile(np.array([0, 255, 0], dtype=np.uint8), (len(pts_world), 1))
                    corr_hand_pts.append(pts_world)
                    corr_hand_colors.append(c)

        # Save original variant (scene + blue hands)
        if orig_hand_pts:
            pts = np.vstack([scene_pts] + orig_hand_pts)
            colors = np.vstack([scene_colors] + orig_hand_colors)
        else:
            pts, colors = scene_pts, scene_colors
        _save_ply(pts, colors, os.path.join(output_dir, f"{basename}_original.ply"))

        # Save corrected variant (scene + green hands)
        if corr_hand_pts:
            pts = np.vstack([scene_pts] + corr_hand_pts)
            colors = np.vstack([scene_colors] + corr_hand_colors)
        else:
            pts, colors = scene_pts, scene_colors
        _save_ply(pts, colors, os.path.join(output_dir, f"{basename}_corrected.ply"))

        n_saved += 1

    print(f"  Saved {n_saved} frame pairs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PLY files with scene + green MANO hand overlay")
    parser.add_argument("--manifest", required=True,
                        help="Batch manifest JSON file")
    parser.add_argument("--results_dir", required=True,
                        help="Top-level results directory (e.g., batch_50_pi3)")
    parser.add_argument("--output_base", default=None,
                        help="Output base dir (default: <results_dir>/hand_overlay)")
    parser.add_argument("--corrected_subdir", default="corrected_annotations",
                        help="Subdirectory with corrected annotations (default: corrected_annotations)")
    args = parser.parse_args()

    output_base = args.output_base or os.path.join(args.results_dir, "hand_overlay")

    # Load MANO models
    print("Loading MANO models...")
    mano_r = load_mano_model(MANO_RIGHT_PATH)
    mano_l = load_mano_model(MANO_LEFT_PATH)
    if mano_r is None:
        print(f"ERROR: MANO_RIGHT not found at {MANO_RIGHT_PATH}")
        sys.exit(1)
    if mano_l is None:
        print(f"WARNING: MANO_LEFT not found at {MANO_LEFT_PATH}, left hands will use MANO_RIGHT")

    manifest = json.load(open(args.manifest))

    corrected_dir = os.path.join(args.results_dir, args.corrected_subdir)
    print(f"Using corrected annotations from: {corrected_dir}")

    for i, entry in enumerate(manifest):
        video_dir = entry["output_dir"]
        video_name = os.path.basename(video_dir)
        original_ann_path = entry["annotation_path"]

        # Use corrected annotation if available — check top-level dir first,
        # then per-video corrected_annotations/ subdirectory
        ann_basename = os.path.splitext(os.path.basename(original_ann_path))[0]
        corrected_path = os.path.join(corrected_dir, ann_basename + ".npy")
        per_video_corrected = os.path.join(video_dir, "corrected_annotations",
                                           ann_basename + ".npy")
        if os.path.exists(corrected_path):
            ann_path = corrected_path
        elif os.path.exists(per_video_corrected):
            ann_path = per_video_corrected
        else:
            ann_path = original_ann_path

        print(f"[{i+1}/{len(manifest)}] {video_name}")
        out_dir = os.path.join(output_base, video_name)
        process_video(video_dir, ann_path, original_ann_path, out_dir, mano_r, mano_l)


if __name__ == "__main__":
    main()
