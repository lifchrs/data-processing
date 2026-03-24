#!/usr/bin/env python3
"""
Rescale reconstructed depth maps and camera poses to metric scale using VITRA hand joints.

The reconstruction (Pi3/TTT3R) outputs depth in an arbitrary, globally-consistent
scale. VITRA provides metric 3D hand joints in camera space. We align depth to
the hand joints in two steps:

  1. Global scale: median ratio of recon_depth / joint_z across all joints and
     frames, applied uniformly to all depth maps and camera translations.
     This puts everything in approximate metric (meters).
  2. Per-frame hand correction: for each frame with valid hand joints, compute a
     z-offset that aligns the hand's wrist to the depth surface, then save the
     corrected joint positions. This adjusts hand *position* only — hand
     geometry (shape/size) is never modified, preserving MANO as a consistent
     physical ruler across all videos.

Usage (standalone):
    python rescale_to_metric.py --ttt3r_out path/to/intermediate_depth --annotation path/to/annot.npy

Or as a library:
    from rescale_to_metric import rescale_ttt3r_to_metric
    scale = rescale_ttt3r_to_metric(ttt3r_out_dir, annotation_path)
"""

import glob
import os

import numpy as np

# MANO model paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_MANO_RIGHT_PATH = os.path.join(_PROJECT_ROOT, "WiLoR", "mano_data", "MANO_RIGHT_clean.npz")
_MANO_LEFT_PATH = os.path.join(_PROJECT_ROOT, "WiLoR", "mano_data", "MANO_LEFT_clean.npz")
_MANO_CACHE = {}


def _load_mano(path):
    """Load MANO model, cached."""
    if path not in _MANO_CACHE:
        if os.path.exists(path):
            _MANO_CACHE[path] = dict(np.load(path, allow_pickle=True))
        else:
            _MANO_CACHE[path] = None
    return _MANO_CACHE[path]


def _mano_forward(mano_data, beta, global_orient, hand_pose, transl):
    """Numpy MANO forward pass → (vertices, faces)."""
    v_template = mano_data['v_template'].copy()
    shapedirs = mano_data['shapedirs'].copy()
    posedirs = mano_data['posedirs'].copy()
    J_regressor = mano_data['J_regressor']
    weights = mano_data['weights']
    kintree = mano_data['kintree_table']

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
    return v_out


def _get_hand_vertices(hand_dict, vi, is_left):
    """Generate MANO mesh vertices for one hand at one frame.

    Returns (778, 3) camera-space vertices, or None if frame is invalid.
    """
    kept = hand_dict['kept_frames']
    if vi >= len(kept) or not kept[vi]:
        return None

    mano_data = _load_mano(_MANO_LEFT_PATH if is_left else _MANO_RIGHT_PATH)
    if mano_data is None:
        mano_data = _load_mano(_MANO_RIGHT_PATH)
    if mano_data is None:
        return None

    beta = hand_dict['beta']
    global_orient = hand_dict['global_orient_camspace'][vi]
    hand_pose = hand_dict['hand_pose'][vi]
    S = np.diag([-1.0, 1.0, 1.0])

    if is_left:
        hp = np.array([S @ r @ S for r in hand_pose])
        verts = _mano_forward(mano_data, beta, global_orient, hp, np.zeros(3))
        wrist_offset = mano_data['J_regressor'][0] @ verts
    else:
        verts = _mano_forward(mano_data, beta, global_orient, hand_pose, np.zeros(3))
        wrist_offset = mano_data['J_regressor'][0] @ verts

    target_wrist = hand_dict['joints_camspace'][vi][0]
    transl = target_wrist - wrist_offset

    if is_left:
        verts = _mano_forward(mano_data, beta, global_orient, hp, transl)
    else:
        verts = _mano_forward(mano_data, beta, global_orient, hand_pose, transl)

    return verts

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


def _collect_surface_samples(depth_map, recon_K, points, half_patch=2):
    """Project 3D points into depth map and collect depth/point_z ratios.

    Works with any set of camera-space 3D points — joints (21) or mesh
    vertices (778).  Uses a small patch (5x5) per point for robustness.

    Args:
        depth_map: (H, W) depth array
        recon_K: 3x3 reconstruction intrinsics
        points: (N, 3) camera-space 3D points
        half_patch: half-size of sampling patch (default 2 → 5x5)

    Returns:
        list of (recon_depth / point_z) ratios for valid points
    """
    H_d, W_d = depth_map.shape
    fx, fy = recon_K[0, 0], recon_K[1, 1]
    cx, cy = recon_K[0, 2], recon_K[1, 2]

    ratios = []
    for i in range(len(points)):
        pt = points[i]
        if pt[2] <= 0.05:
            continue

        u = fx * pt[0] / pt[2] + cx
        v = fy * pt[1] / pt[2] + cy
        ui, vi_px = int(round(u)), int(round(v))

        if ui < 2 or ui >= W_d - 2 or vi_px < 2 or vi_px >= H_d - 2:
            continue

        r0 = max(vi_px - half_patch, 0)
        r1 = min(vi_px + half_patch + 1, H_d)
        c0 = max(ui - half_patch, 0)
        c1 = min(ui + half_patch + 1, W_d)
        patch = depth_map[r0:r1, c0:c1]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if len(valid) == 0:
            continue

        d_recon = float(np.median(valid))

        ratio = d_recon / pt[2]
        if ratio < 0.1 or ratio > 20.0:
            continue

        ratios.append(ratio)

    return ratios


def _robust_median(arr):
    """Compute IQR-filtered median."""
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = arr[(arr >= lo) & (arr <= hi)]
    else:
        filtered = arr
    if len(filtered) == 0:
        filtered = arr
    return float(np.median(filtered)), filtered


def _collect_joint_offsets(depth_map, recon_K, joints, half_patch=5):
    """Project joints into depth map and collect z-offsets (depth - joint_z).

    Uses only the wrist (joint 0) and proximal joints (1, 5, 9, 13, 17) which
    are least likely to be occluded by a held object.  Fingertips and middle
    phalanges are often behind the object surface, biasing the offset.

    Args:
        depth_map: (H, W) depth array (already in metric)
        recon_K: 3x3 reconstruction intrinsics
        joints: (21, 3) metric camera-space joints
        half_patch: half-size of sampling patch

    Returns:
        list of (depth_surface_z - joint_z) offsets for valid joints
    """
    H_d, W_d = depth_map.shape
    fx, fy = recon_K[0, 0], recon_K[1, 1]
    cx, cy = recon_K[0, 2], recon_K[1, 2]

    # Wrist + MCP (knuckle) joints — least occluded by held objects
    _PROXY_JOINTS = [0, 1, 5, 9, 13, 17]

    offsets = []
    for j in _PROXY_JOINTS:
        pt = joints[j]
        if pt[2] <= 0.05:
            continue

        u = fx * pt[0] / pt[2] + cx
        v = fy * pt[1] / pt[2] + cy
        ui, vi = int(round(u)), int(round(v))

        if ui < 2 or ui >= W_d - 2 or vi < 2 or vi >= H_d - 2:
            continue

        r0 = max(vi - half_patch, 0)
        r1 = min(vi + half_patch + 1, H_d)
        c0 = max(ui - half_patch, 0)
        c1 = min(ui + half_patch + 1, W_d)
        patch = depth_map[r0:r1, c0:c1]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if len(valid) == 0:
            continue

        d_surface = float(np.median(valid))

        # Tighter ratio filter: reject if surface depth differs >30% from
        # joint depth.  A held object in front of the hand typically causes
        # depth < joint_z (ratio < 1), so this catches most occlusions.
        ratio = d_surface / pt[2]
        if ratio < 0.7 or ratio > 1.3:
            continue

        offsets.append(d_surface - pt[2])

    return offsets


def _save_original_as_fallback(annotation_path, results_dir):
    """Copy the original VITRA annotation to corrected_annotations/ unchanged.

    Called when rescaling/correction cannot be performed (e.g. insufficient
    hand data).  Ensures training always has a complete set of annotations
    in ``corrected_annotations/``.
    """
    import shutil
    episode_id = os.path.splitext(os.path.basename(annotation_path))[0]
    corrected_dir = os.path.join(results_dir, "corrected_annotations")
    os.makedirs(corrected_dir, exist_ok=True)
    corrected_path = os.path.join(corrected_dir, episode_id + ".npy")
    if not os.path.exists(corrected_path):
        shutil.copy2(annotation_path, corrected_path)
        print(f"  Saved original annotation as fallback: {corrected_path}")


def _fix_metric_scale_if_needed(ttt3r_out, annotation_path, results_dir,
                                marker_path):
    """Check if a 'metric' depth is actually consistent with VITRA hand joints.

    Some backends (Pi3) claim metric scale but the depth can be off by 2-6x on
    certain videos.  This function computes the depth/joint_z ratio across all
    frames, and if the median ratio is outside [0.7, 1.3], applies a global
    scale correction to depth maps and camera translations — just like Step 1
    of the TTT3R rescaling — before per-frame hand correction runs.

    Uses a wide ratio filter (0.1-10.0) since the whole point is to catch large
    scale mismatches that the tight filter in _collect_joint_offsets would miss.
    """
    annotation = np.load(annotation_path, allow_pickle=True).item()
    vdf = annotation.get("video_decode_frame")
    if vdf is None:
        return

    vdf = _rebase_vdf(vdf, annotation_path)
    frame_hands = _collect_frame_hands(annotation, vdf)
    if not frame_hands:
        return

    depth_dir = os.path.join(ttt3r_out, "depth")
    camera_dir = os.path.join(ttt3r_out, "camera")

    # Collect depth/joint_z ratios with WIDE filter (0.1-20.0) to catch
    # large mismatches that the standard filter would reject
    all_ratios = []
    for vf, hands_info in sorted(frame_hands.items()):
        depth_path = os.path.join(depth_dir, f"{vf:06d}.npy")
        cam_path = os.path.join(camera_dir, f"{vf:06d}.npz")
        if not os.path.exists(depth_path) or not os.path.exists(cam_path):
            continue
        depth_map = np.load(depth_path)
        recon_K = np.load(cam_path)["intrinsics"]
        H_d, W_d = depth_map.shape
        fx, fy = recon_K[0, 0], recon_K[1, 1]
        cx, cy = recon_K[0, 2], recon_K[1, 2]
        for _, _, joints in hands_info:
            for j in range(21):
                pt = joints[j]
                if pt[2] <= 0.05:
                    continue
                u = int(round(fx * pt[0] / pt[2] + cx))
                v = int(round(fy * pt[1] / pt[2] + cy))
                if u < 2 or u >= W_d - 2 or v < 2 or v >= H_d - 2:
                    continue
                patch = depth_map[max(v-5,0):v+6, max(u-5,0):u+6]
                valid = patch[(patch > 0) & np.isfinite(patch)]
                if len(valid) == 0:
                    continue
                ratio = float(np.median(valid)) / pt[2]
                if 0.1 <= ratio <= 20.0:
                    all_ratios.append(ratio)

    if len(all_ratios) < 5:
        return

    median_ratio = float(np.median(all_ratios))
    if 0.7 <= median_ratio <= 1.3:
        # Scale is already consistent, no global correction needed
        return

    # Apply global scale correction
    print(f"  Metric scale correction: depth/VITRA ratio = {median_ratio:.3f}, "
          f"rescaling depth maps by 1/{median_ratio:.3f}")

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    for dp in depth_files:
        d = np.load(dp)
        d = d / median_ratio
        np.save(dp, d)

    camera_files = sorted(glob.glob(os.path.join(camera_dir, "*.npz")))
    for cp in camera_files:
        cam = dict(np.load(cp))
        pose = cam["pose"].copy()
        pose[:3, 3] /= median_ratio
        np.savez(cp, pose=pose, intrinsics=cam["intrinsics"])

    print(f"  Rescaled {len(depth_files)} depth maps and "
          f"{len(camera_files)} camera poses")

    # Update marker to record the additional correction
    with open(marker_path, "w") as f:
        f.write(f"{median_ratio}")


def rescale_ttt3r_to_metric(ttt3r_out, annotation_path, results_dir=None):
    """Rescale depth maps and camera poses to metric in-place, and correct
    hand joint positions to sit on the depth surface.

    Two-step approach:
      Step 1 — Global scale: single median ratio across all frames/joints,
               applied to every depth map and camera translation.  Puts
               everything in approximate metric (meters).
      Step 2 — Per-frame hand correction: for frames with hand data, compute
               a z-offset so the hand's joints align with the depth surface.
               Offset is applied in **world space** to ``transl_worldspace``
               and ``joints_worldspace`` (what training reads), plus
               ``joints_camspace`` for consistency.  Corrected episode .npy
               is saved to ``<results_dir>/corrected_annotations/<episode_id>.npy``.

    Args:
        ttt3r_out: Path to intermediate_depth directory (depth/, camera/).
        annotation_path: Path to the VITRA episode .npy file.
        results_dir: Top-level results directory for this video.  If None,
                     defaults to the parent of ttt3r_out.

    Returns:
        global_scale: the global scale factor, or None if rescaling was skipped.
    """
    if results_dir is None:
        results_dir = os.path.dirname(ttt3r_out)

    marker_path = os.path.join(ttt3r_out, ".metric_scale")
    if os.path.exists(marker_path):
        info = open(marker_path).read().strip()
        print(f"  Already rescaled to metric ({info})")

        episode_id = os.path.splitext(os.path.basename(annotation_path))[0]
        corrected_dir = os.path.join(results_dir, "corrected_annotations")
        corrected_path = os.path.join(corrected_dir, episode_id + ".npy")

        if not os.path.exists(corrected_path):
            # Check if metric scale is actually consistent with VITRA hands.
            # Some backends (Pi3) claim metric scale but can be off by 2-6x
            # on certain videos.  If the depth/joint ratio is far from 1.0,
            # apply a global correction before per-frame hand correction.
            _fix_metric_scale_if_needed(
                ttt3r_out, annotation_path, results_dir, marker_path)
            _correct_hand_positions(ttt3r_out, annotation_path, results_dir)

        return float(info.split(",")[0])

    annotation = np.load(annotation_path, allow_pickle=True).item()
    vdf = annotation.get("video_decode_frame")
    if vdf is None:
        print("  WARNING: No video_decode_frame in annotation, skipping rescale")
        _save_original_as_fallback(annotation_path, results_dir)
        return None

    vdf = _rebase_vdf(vdf, annotation_path)

    depth_dir = os.path.join(ttt3r_out, "depth")
    camera_dir = os.path.join(ttt3r_out, "camera")

    # ── Collect per-frame hand joint data ──────────────────────────────
    frame_hands = _collect_frame_hands(annotation, vdf)

    if not frame_hands:
        print("  WARNING: No valid hand frames, skipping rescale")
        _save_original_as_fallback(annotation_path, results_dir)
        return None

    # ── Step 1: Global scale ──────────────────────────────────────────
    all_samples = []
    for vf, hands_info in sorted(frame_hands.items()):
        depth_path = os.path.join(depth_dir, f"{vf:06d}.npy")
        cam_path = os.path.join(camera_dir, f"{vf:06d}.npz")
        if not os.path.exists(depth_path) or not os.path.exists(cam_path):
            continue
        depth_map = np.load(depth_path)
        recon_K = np.load(cam_path)["intrinsics"]
        for _, _, joints in hands_info:
            all_samples.extend(_collect_joint_samples(depth_map, recon_K, joints))

    if len(all_samples) < 10:
        print(f"  WARNING: Only {len(all_samples)} joint samples, skipping rescale")
        _save_original_as_fallback(annotation_path, results_dir)
        return None

    global_scale, filtered = _robust_median(np.array(all_samples))
    print(f"  Global scale: {global_scale:.6f} "
          f"(median of {len(filtered)}/{len(all_samples)} samples, "
          f"range [{filtered.min():.6f}, {filtered.max():.6f}])")

    # Apply global scale to ALL depth maps and camera poses
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    for dp in depth_files:
        d = np.load(dp)
        d = d / global_scale
        np.save(dp, d)

    camera_files = sorted(glob.glob(os.path.join(camera_dir, "*.npz")))
    for cp in camera_files:
        cam = dict(np.load(cp))
        pose = cam["pose"].copy()
        pose[:3, 3] /= global_scale
        np.savez(cp, pose=pose, intrinsics=cam["intrinsics"])

    print(f"  Rescaled {len(depth_files)} depth maps and "
          f"{len(camera_files)} camera poses to metric")

    # Write marker
    with open(marker_path, "w") as f:
        f.write(f"{global_scale}")

    # ── Step 2: Per-frame hand position correction ────────────────────
    _correct_hand_positions(ttt3r_out, annotation_path, results_dir)

    return global_scale


def _collect_frame_hands(annotation, vdf):
    """Build mapping: video_frame_index -> list of (hand_name, vi, joints_camspace).

    Args:
        annotation: VITRA annotation dict
        vdf: rebased video_decode_frame array

    Returns:
        dict: vf_int -> [(hand_name, vi, (21, 3) joints), ...]
              vi is the VITRA annotation index (row in joints_camspace)
    """
    frame_hands = {}
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
            vf_int = int(vf)
            if vf_int not in frame_hands:
                frame_hands[vf_int] = []
            frame_hands[vf_int].append((hand_name, vi, joints_cam[vi]))
    return frame_hands


def _correct_hand_positions(ttt3r_out, annotation_path, results_dir):
    """Compute per-frame z-offsets to align hand joints with the depth surface,
    and save a corrected episode .npy that training can load directly.

    For each frame with valid hand data, projects the VITRA wrist/proximal
    joints into the (already metric-scaled) depth map, computes the median
    z-offset, then transforms that offset to **world space** and applies it
    to ``transl_worldspace`` and ``joints_worldspace`` (the fields read by
    training).

    **Occlusion robustness**: Only wrist + MCP joints are used (held objects
    occlude fingertips first).  A temporal outlier pass rejects per-frame
    offsets that deviate >2 sigma from the per-hand median, preventing
    sporadic occlusion frames from corrupting the correction.

    Saves the corrected annotation to
    ``<results_dir>/corrected_annotations/<episode_id>.npy`` as a drop-in
    replacement for the original episode file.
    """
    import copy

    annotation = np.load(annotation_path, allow_pickle=True).item()
    vdf = annotation.get("video_decode_frame")
    if vdf is None:
        return

    vdf = _rebase_vdf(vdf, annotation_path)
    frame_hands = _collect_frame_hands(annotation, vdf)
    if not frame_hands:
        return

    depth_dir = os.path.join(ttt3r_out, "depth")
    camera_dir = os.path.join(ttt3r_out, "camera")

    # Deep-copy the annotation so we can modify fields in-place
    corrected_ann = copy.deepcopy(annotation)

    # Pre-extract extrinsics (world-to-cam, (T, 4, 4)) for world-space offset
    extrinsics = annotation.get("extrinsics")  # (T, 4, 4) w2c

    # ── Pass 1: collect raw z-offsets per (hand, frame) ──────────────
    raw_offsets = {}  # (hand_name, vi) -> z_offset or None
    for vf, hands_info in sorted(frame_hands.items()):
        depth_path = os.path.join(depth_dir, f"{vf:06d}.npy")
        cam_path = os.path.join(camera_dir, f"{vf:06d}.npz")
        if not os.path.exists(depth_path) or not os.path.exists(cam_path):
            continue

        depth_map = np.load(depth_path)
        recon_K = np.load(cam_path)["intrinsics"]

        for hand_name, vi, joints in hands_info:
            offsets = _collect_joint_offsets(depth_map, recon_K, joints)
            if len(offsets) >= 2:
                raw_offsets[(hand_name, vi)] = float(np.median(offsets))
            else:
                raw_offsets[(hand_name, vi)] = None

    # ── Pass 2: temporal outlier rejection per hand ──────────────────
    # Compute per-hand median & MAD, then reject frames > 2*MAD away.
    # This catches sporadic occlusion spikes (hand behind held object).
    hand_offsets = {}  # hand_name -> list of z_offsets (non-None)
    for (hn, vi), z in raw_offsets.items():
        if z is not None:
            hand_offsets.setdefault(hn, []).append(z)

    hand_median = {}
    hand_mad = {}
    for hn, vals in hand_offsets.items():
        arr = np.array(vals)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        hand_median[hn] = med
        hand_mad[hn] = max(mad, 0.005)  # floor at 5mm

    # ── Pass 3: apply filtered offsets ───────────────────────────────
    n_corrected = 0
    n_rejected = 0
    all_offsets = []

    for (hand_name, vi), z_raw in raw_offsets.items():
        if z_raw is None:
            z_offset = 0.0
        elif hand_name in hand_median:
            # Reject if > 2*MAD from per-hand median (likely occlusion)
            if abs(z_raw - hand_median[hand_name]) > 2.0 * hand_mad[hand_name]:
                z_offset = 0.0
                n_rejected += 1
            else:
                z_offset = z_raw
        else:
            z_offset = 0.0

        # --- Camera-space correction (z-only) ---
        corrected_ann[hand_name]["joints_camspace"][vi, :, 2] += z_offset
        if "transl_camspace" in corrected_ann[hand_name]:
            corrected_ann[hand_name]["transl_camspace"][vi, 2] += z_offset

        # --- World-space correction ---
        # The camera-space offset is [0, 0, z_offset].
        # Transform to world space: offset_world = R_c2w @ [0, 0, z_offset]
        # where R_c2w = R_w2c^T (rotation part of extrinsics).
        if extrinsics is not None and vi < len(extrinsics):
            R_w2c = extrinsics[vi, :3, :3]  # (3, 3)
            R_c2w = R_w2c.T
            offset_cam = np.array([0.0, 0.0, z_offset])
            offset_world = R_c2w @ offset_cam  # (3,)
        else:
            offset_world = np.array([0.0, 0.0, z_offset])

        corrected_ann[hand_name]["transl_worldspace"][vi] += offset_world
        corrected_ann[hand_name]["joints_worldspace"][vi, :, :] += offset_world

        if abs(z_offset) > 1e-4:
            n_corrected += 1
        all_offsets.append(z_offset)

    if all_offsets:
        offs = np.array(all_offsets)
        print(f"  Hand position correction: {n_corrected}/{len(all_offsets)} hands adjusted, "
              f"{n_rejected} rejected as outliers "
              f"(z-offsets: mean={offs.mean():.4f}m, std={offs.std():.4f}m, "
              f"range=[{offs.min():.4f}, {offs.max():.4f}]m)")

    # Save corrected episode .npy to corrected_annotations/<episode_id>.npy
    episode_id = os.path.splitext(os.path.basename(annotation_path))[0]
    corrected_dir = os.path.join(results_dir, "corrected_annotations")
    os.makedirs(corrected_dir, exist_ok=True)
    corrected_path = os.path.join(corrected_dir, episode_id + ".npy")
    np.save(corrected_path, corrected_ann)
    print(f"  Saved corrected annotation: {corrected_path}")


def rescale_depth_maps_only(ttt3r_out, annotation_path):
    """Compute global scale from hand joints and rescale depth maps + camera
    poses in-place.  Does NOT modify the annotation.

    Returns the global_scale factor, or None if rescaling was skipped.
    """
    annotation = np.load(annotation_path, allow_pickle=True).item()
    vdf = annotation.get("video_decode_frame")
    if vdf is None:
        return None

    vdf = _rebase_vdf(vdf, annotation_path)
    depth_dir = os.path.join(ttt3r_out, "depth")
    camera_dir = os.path.join(ttt3r_out, "camera")
    frame_hands = _collect_frame_hands(annotation, vdf)
    if not frame_hands:
        return None

    all_samples = []
    for vf, hands_info in sorted(frame_hands.items()):
        depth_path = os.path.join(depth_dir, f"{vf:06d}.npy")
        cam_path = os.path.join(camera_dir, f"{vf:06d}.npz")
        if not os.path.exists(depth_path) or not os.path.exists(cam_path):
            continue
        depth_map = np.load(depth_path)
        recon_K = np.load(cam_path)["intrinsics"]
        for hand_name, vi, joints in hands_info:
            # Use full MANO mesh surface (778 vertices) for dense sampling
            is_left = (hand_name == "left")
            verts = _get_hand_vertices(annotation[hand_name], vi, is_left)
            if verts is not None:
                all_samples.extend(
                    _collect_surface_samples(depth_map, recon_K, verts))
            else:
                # Fall back to 21 joints if MANO model not available
                all_samples.extend(
                    _collect_surface_samples(depth_map, recon_K, joints))

    if len(all_samples) < 5:
        print(f"  WARNING: Only {len(all_samples)} surface samples, skipping rescale")
        return 1.0

    global_scale, filtered = _robust_median(np.array(all_samples))
    print(f"  Global scale: {global_scale:.6f} "
          f"(median of {len(filtered)}/{len(all_samples)} samples)")

    # Rescale depth maps
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    for dp in depth_files:
        d = np.load(dp)
        d = d / global_scale
        np.save(dp, d)

    # Rescale camera translations
    camera_files = sorted(glob.glob(os.path.join(camera_dir, "*.npz")))
    for cp in camera_files:
        cam = dict(np.load(cp))
        pose = cam["pose"].copy()
        pose[:3, 3] /= global_scale
        np.savez(cp, pose=pose, intrinsics=cam["intrinsics"])

    print(f"  Rescaled {len(depth_files)} depth maps and "
          f"{len(camera_files)} camera poses")

    return global_scale


def rescale_annotation_camspace(annotation_path, global_scale, results_dir):
    """Rescale an annotation's camera-space fields by 1/global_scale to match
    rescaled depth maps.  Recomputes world-space fields from the rescaled
    camera-space values.

    Saves to <results_dir>/corrected_annotations/<episode_id>.npy.
    """
    import copy
    annotation = np.load(annotation_path, allow_pickle=True).item()
    corrected = copy.deepcopy(annotation)
    extrinsics = annotation.get("extrinsics")  # (T, 4, 4) w2c

    for hand_name in ["left", "right"]:
        if hand_name not in corrected:
            continue
        hd = corrected[hand_name]
        kept = hd["kept_frames"]

        for vi in range(len(kept)):
            if not kept[vi]:
                continue

            # Rescale camera-space (all 3 dims)
            hd["joints_camspace"][vi] /= global_scale
            hd["transl_camspace"][vi] /= global_scale

            # Recompute world-space from rescaled camera-space
            if extrinsics is not None and vi < len(extrinsics):
                w2c = extrinsics[vi]
                R_c2w = w2c[:3, :3].T
                t_c2w = -R_c2w @ w2c[:3, 3]

                j_cam = hd["joints_camspace"][vi]
                hd["joints_worldspace"][vi] = (R_c2w @ j_cam.T + t_c2w[:, None]).T
                hd["transl_worldspace"][vi] = R_c2w @ hd["transl_camspace"][vi] + t_c2w

    episode_id = os.path.splitext(os.path.basename(annotation_path))[0]
    corrected_dir = os.path.join(results_dir, "corrected_annotations")
    os.makedirs(corrected_dir, exist_ok=True)
    corrected_path = os.path.join(corrected_dir, episode_id + ".npy")
    np.save(corrected_path, corrected)
    print(f"  Rescaled annotation saved: {corrected_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rescale TTT3R output to metric using VITRA hand joints")
    parser.add_argument("--ttt3r_out", required=True,
                        help="TTT3R output directory (depth/, camera/)")
    parser.add_argument("--annotation", required=True,
                        help="VITRA annotation .npy file")
    parser.add_argument("--results_dir", default=None,
                        help="Top-level results directory (default: parent of ttt3r_out)")
    args = parser.parse_args()

    scale = rescale_ttt3r_to_metric(args.ttt3r_out, args.annotation,
                                    results_dir=args.results_dir)
    if scale:
        print(f"Done. Global scale factor: {scale:.6f}")
    else:
        print("Rescaling skipped (insufficient hand data).")
