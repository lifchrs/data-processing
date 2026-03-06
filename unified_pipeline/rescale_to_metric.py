#!/usr/bin/env python3
"""
Rescale TTT3R depth maps and camera poses to metric scale using VITRA hand joints.

TTT3R outputs depth in an arbitrary, globally-consistent scale. VITRA provides
metric 3D hand joints in camera space. By comparing TTT3R depth at joint
locations to the known metric depth (joint z-coordinate), we recover a single
scale factor and apply it to all depth maps and camera translations.

Usage (standalone):
    python rescale_to_metric.py --ttt3r_out path/to/ttt3r_output --annotation path/to/annot.npy

Or as a library:
    from rescale_to_metric import rescale_ttt3r_to_metric
    scale = rescale_ttt3r_to_metric(ttt3r_out_dir, annotation_path)
"""

import glob
import os

import numpy as np

# Datasets where video_decode_frame contains absolute frame indices
_ABSOLUTE_VDF_DATASETS = {"epic", "ego4d", "egoexo4d"}

_DATASET_PREFIXES = [
    ("somethingsomethingv2_", "ssv2"),
    ("epic_kitchens_", "epic"),
    ("EgoExo4D_", "egoexo4d"),
    ("Ego4D_", "ego4d"),
]


def _rebase_vdf(vdf, annotation_path):
    """Rebase video_decode_frame to clip-relative indices if needed."""
    basename = os.path.basename(annotation_path)
    dataset = None
    for prefix, ds in _DATASET_PREFIXES:
        if basename.startswith(prefix):
            dataset = ds
            break
    if dataset in _ABSOLUTE_VDF_DATASETS:
        return vdf - vdf[0]
    return vdf


def compute_metric_scale(ttt3r_out, annotation_path):
    """Compute the scale factor to convert TTT3R depth to metric.

    For each valid hand frame, projects all 21 VITRA joints to 2D, reads
    TTT3R depth at those pixels, and computes the ratio ttt3r_z / vitra_z.
    Returns the robust median of all samples.

    Returns:
        scale: float, where metric_depth = ttt3r_depth / scale.
               None if insufficient data.
    """
    annotation = np.load(annotation_path, allow_pickle=True).item()
    vitra_intrinsics = annotation.get("intrinsics")
    vdf = annotation.get("video_decode_frame")

    if vitra_intrinsics is None or vdf is None:
        return None

    vdf = _rebase_vdf(vdf, annotation_path)

    depth_dir = os.path.join(ttt3r_out, "depth")
    camera_dir = os.path.join(ttt3r_out, "camera")

    samples = []

    for hand_name in ["right", "left"]:
        if hand_name not in annotation:
            continue
        hand_data = annotation[hand_name]
        joints_cam = hand_data.get("joints_camspace")
        kept_frames = hand_data.get("kept_frames")
        if joints_cam is None or kept_frames is None:
            continue

        for vi, vf in enumerate(vdf):
            if vi >= len(kept_frames) or not kept_frames[vi]:
                continue
            if vi >= len(joints_cam):
                continue

            depth_path = os.path.join(depth_dir, f"{int(vf):06d}.npy")
            cam_path = os.path.join(camera_dir, f"{int(vf):06d}.npz")
            if not os.path.exists(depth_path) or not os.path.exists(cam_path):
                continue

            depth_map = np.load(depth_path)
            H_d, W_d = depth_map.shape

            # Scale VITRA intrinsics to depth map resolution
            vW = int(round(2 * vitra_intrinsics[0, 2]))
            vH = int(round(2 * vitra_intrinsics[1, 2]))
            sx = W_d / vW
            sy = H_d / vH
            fx = vitra_intrinsics[0, 0] * sx
            fy = vitra_intrinsics[1, 1] * sy
            cx = vitra_intrinsics[0, 2] * sx
            cy = vitra_intrinsics[1, 2] * sy

            joints = joints_cam[vi]  # (21, 3) metric camera-space

            for j in range(21):
                pt = joints[j]
                if pt[2] <= 0.05:  # skip joints behind camera or too close
                    continue

                # Project to 2D
                u = fx * pt[0] / pt[2] + cx
                v = fy * pt[1] / pt[2] + cy

                ui, vi_ = int(round(u)), int(round(v))
                if ui < 2 or ui >= W_d - 2 or vi_ < 2 or vi_ >= H_d - 2:
                    continue

                # Read TTT3R depth (average small patch for robustness)
                patch = depth_map[max(0, vi_-1):vi_+2, max(0, ui-1):ui+2]
                valid = patch[patch > 0]
                if len(valid) == 0:
                    continue
                d_ttt3r = float(np.median(valid))

                samples.append(d_ttt3r / pt[2])

    if len(samples) < 10:
        return None

    arr = np.array(samples)

    # IQR outlier filtering
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = arr[(arr >= lo) & (arr <= hi)]
    else:
        filtered = arr

    if len(filtered) == 0:
        filtered = arr

    scale = float(np.median(filtered))
    print(f"  Metric scale: {scale:.6f} "
          f"(median of {len(filtered)}/{len(samples)} samples, "
          f"range [{filtered.min():.6f}, {filtered.max():.6f}])")
    return scale


def rescale_ttt3r_to_metric(ttt3r_out, annotation_path):
    """Rescale TTT3R depth maps and camera poses to metric in-place.

    Computes the scale factor from VITRA hand joints, then divides all depth
    maps and camera pose translations by that factor.

    Returns:
        scale: the computed scale factor, or None if rescaling was skipped.
    """
    marker_path = os.path.join(ttt3r_out, ".metric_scale")
    if os.path.exists(marker_path):
        scale = float(open(marker_path).read().strip())
        print(f"  Already rescaled to metric (scale={scale:.6f})")
        return scale

    scale = compute_metric_scale(ttt3r_out, annotation_path)
    if scale is None or scale <= 0:
        print("  WARNING: Could not compute metric scale, skipping rescale")
        return None

    # Rescale depth maps
    depth_dir = os.path.join(ttt3r_out, "depth")
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    for dp in depth_files:
        d = np.load(dp)
        d = d / scale
        np.save(dp, d)

    # Rescale camera pose translations
    camera_dir = os.path.join(ttt3r_out, "camera")
    camera_files = sorted(glob.glob(os.path.join(camera_dir, "*.npz")))
    for cp in camera_files:
        cam = dict(np.load(cp))
        pose = cam["pose"].copy()
        pose[:3, 3] /= scale
        np.savez(cp, pose=pose, intrinsics=cam["intrinsics"])

    # Write marker so we don't rescale twice
    with open(marker_path, "w") as f:
        f.write(f"{scale}")

    print(f"  Rescaled {len(depth_files)} depth maps and "
          f"{len(camera_files)} camera poses to metric")
    return scale


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rescale TTT3R output to metric using VITRA hand joints")
    parser.add_argument("--ttt3r_out", required=True,
                        help="TTT3R output directory (depth/, camera/)")
    parser.add_argument("--annotation", required=True,
                        help="VITRA annotation .npy file")
    args = parser.parse_args()

    scale = rescale_ttt3r_to_metric(args.ttt3r_out, args.annotation)
    if scale:
        print(f"Done. Scale factor: {scale:.6f}")
    else:
        print("Rescaling skipped (insufficient hand data).")
