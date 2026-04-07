#!/usr/bin/env python3
"""
Compute metric scale factor for reconstructed depth maps using VITRA hand joints.

The reconstruction (Pi3) outputs depth in an arbitrary, globally-consistent
scale. VITRA provides metric 3D hand joints in camera space. We align depth to
the hand joints in two steps:

  1. Global scale: median ratio of recon_depth / joint_z across all MANO mesh
     vertices and frames. This single scalar is saved to ``metric_scale.txt``
     and applied on-the-fly by readers (non-destructive — depth files are
     never overwritten).
  2. Per-frame hand correction: for each frame with valid hand joints, compute a
     z-offset that aligns the hand's wrist to the depth surface, then save the
     corrected joint positions. This adjusts hand *position* only — hand
     geometry (shape/size) is never modified, preserving MANO as a consistent
     physical ruler across all videos.

Usage (standalone):
    python rescale_to_metric.py --recon_dir path/to/reconstruction --annotation path/to/annot.npy

Or as a library:
    from rescale_to_metric import rescale_to_metric_scale, load_metric_scale
    scale = rescale_to_metric_scale(recon_dir, annotation_path)
    # Later, when reading depth:
    metric_scale = load_metric_scale(recon_dir)
    depth_metric = raw_depth / metric_scale
"""

import glob
import os

import numpy as np


def load_metric_scale(recon_dir, frame_idx=None):
    """Load the metric scale factor.

    If per-frame scales exist (metric_scale_per_frame.json), returns the
    scale for the given frame_idx.  Otherwise falls back to the global
    scale from metric_scale.txt.  Returns 1.0 if no scale file exists.

    Args:
        recon_dir: Path to reconstruction directory.
        frame_idx: Frame index (int) for per-frame lookup.  If None,
                   returns the global scale.
    """
    import json as _json

    # Per-frame scale (preferred for debugging / tight alignment)
    pf_path = os.path.join(recon_dir, "metric_scale_per_frame.json")
    if os.path.exists(pf_path):
        with open(pf_path) as f:
            per_frame = _json.load(f)
        if frame_idx is not None:
            key = str(frame_idx)
            if key in per_frame:
                return float(per_frame[key])
        # Fall back to global if frame not found
        if "global" in per_frame:
            return float(per_frame["global"])

    # Global scale
    p = os.path.join(recon_dir, "metric_scale.txt")
    if os.path.exists(p):
        return float(open(p).read().strip())
    return 1.0


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


def _rasterize_hand_zbuffer(verts, faces, recon_K, image_shape,
                            n_samples=200000, supersample=8):
    """Build a z-buffer of the MANO mesh via dense surface sampling.

    Densely samples points on the mesh surface, projects them to a
    supersampled 2D grid, and keeps the minimum depth per pixel.  The
    supersampled z-buffer is then downsampled (min-pool) to the target
    resolution.  Fully vectorized (no Python loops).

    Supersampling ensures that thin structures (fingers) and silhouette
    edges are properly represented without gaps that would let occluded
    vertices leak through.

    Args:
        verts: (V, 3) camera-space vertices
        faces: (F, 3) triangle indices
        recon_K: 3x3 reconstruction intrinsics
        image_shape: (H, W) for the final z-buffer resolution
        n_samples: number of surface samples (default 50K)
        supersample: resolution multiplier (default 4x)

    Returns:
        zbuf: (H, W) array, inf where no surface covers the pixel
    """
    import trimesh

    H, W = image_shape
    Hs, Ws = H * supersample, W * supersample
    fx, fy = recon_K[0, 0] * supersample, recon_K[1, 1] * supersample
    cx, cy = recon_K[0, 2] * supersample, recon_K[1, 2] * supersample

    # Densely sample surface points (vectorized in trimesh)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pts, _ = mesh.sample(n_samples, return_index=True)

    # Include original vertices
    pts = np.vstack([pts, verts])

    # Project to supersampled 2D (vectorized)
    z = pts[:, 2]
    valid = z > 0.01
    u = np.full(len(pts), -1, dtype=np.int32)
    v = np.full(len(pts), -1, dtype=np.int32)
    u[valid] = np.round(fx * pts[valid, 0] / z[valid] + cx).astype(np.int32)
    v[valid] = np.round(fy * pts[valid, 1] / z[valid] + cy).astype(np.int32)

    in_bounds = valid & (u >= 0) & (u < Ws) & (v >= 0) & (v < Hs)
    idx = np.where(in_bounds)[0]

    # Build supersampled z-buffer
    zbuf_hi = np.full((Hs, Ws), np.inf, dtype=np.float64)
    np.minimum.at(zbuf_hi, (v[idx], u[idx]), z[idx])

    # Downsample to target resolution (min-pool: take minimum depth in
    # each supersample x supersample block)
    zbuf = zbuf_hi.reshape(H, supersample, W, supersample).min(axis=(1, 3))

    return zbuf


def _get_visible_hand_vertices(verts, faces, recon_K, image_shape):
    """Return only vertices on camera-facing faces (normal-based culling).

    For each face, computes the dot product of the face normal with the
    view direction (camera origin to face centroid).  Faces with negative
    dot product have normals pointing toward the camera and are visible.
    Vertices belonging to at least one visible face are kept.

    This is fast (~1ms) and handles most visibility cases.  It does not
    handle self-occlusion from curled fingers, but those outliers are
    filtered by the IQR-based robust median in the scale computation.

    Args:
        verts: (V, 3) camera-space vertices
        faces: (F, 3) triangle indices
        recon_K: unused (kept for API compatibility)
        image_shape: unused (kept for API compatibility)

    Returns:
        visible_verts: (N, 3) subset of verts on front-facing faces
    """
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # View direction = camera origin (0,0,0) to face centroid (= centroid itself)
    centroids = verts[faces].mean(axis=1)
    view_dirs = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    dots = np.sum(view_dirs * mesh.face_normals, axis=1)

    # Front-facing: normal points toward camera (dot < 0)
    front_vert_ids = np.unique(faces[dots < 0])
    return verts[front_vert_ids]


def _get_hand_vertices(hand_dict, vi, is_left, recon_K=None, image_shape=None):
    """Generate MANO mesh vertices for one hand at one frame.

    If recon_K and image_shape are provided, performs proper visibility
    testing via z-buffer rasterization — only vertices on the frontmost
    visible surface are returned.  This handles all self-occlusion cases
    (e.g., curled fingers in front of palm).

    Without recon_K/image_shape, returns all vertices (no culling).

    Returns camera-space vertices, or None if frame is invalid or MANO
    model is not available.
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
    faces = mano_data['f'].astype(np.int64)
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

    if recon_K is not None and image_shape is not None:
        return _get_visible_hand_vertices(verts, faces, recon_K, image_shape)
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

    IMPORTANT — intrinsics and camera space:
      ``points`` are VITRA joints/vertices in camera space.  Camera-space 3D
      coordinates are *intrinsics-independent* — they are physical metric
      positions relative to the camera center.  A point at (0.3, 0.05, 0.5)
      is at that position regardless of lens or image resolution.

      To project a camera-space point to pixel coordinates, use the intrinsics
      of the IMAGE you are projecting onto (here, the reconstruction depth map).
      Do NOT reproject through annotation intrinsics — that would change the 3D
      position and produce wrong pixel lookups.  Both VITRA and the
      reconstruction backend (Pi3/TTT3R) view the same physical scene from the
      same camera, so they share the same camera space.

    Args:
        depth_map: (H, W) depth array
        recon_K: 3x3 intrinsics of the reconstruction (matches depth_map pixels)
        points: (N, 3) camera-space 3D points (from VITRA joints or MANO verts)
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


def _collect_joint_offsets(depth_map, recon_K, joints, half_patch=5,
                          metric_scale=1.0):
    """Project joints into depth map and collect z-offsets (depth - joint_z).

    Uses only the wrist (joint 0) and proximal joints (1, 5, 9, 13, 17) which
    are least likely to be occluded by a held object.  Fingertips and middle
    phalanges are often behind the object surface, biasing the offset.

    Same intrinsics rule as _collect_surface_samples: project camera-space
    joints using recon_K directly.  See that function's docstring for why
    annotation intrinsics should NOT be used here.

    Args:
        depth_map: (H, W) RAW depth array (not yet metric-scaled)
        recon_K: 3x3 intrinsics of the reconstruction (matches depth_map pixels)
        joints: (21, 3) camera-space joints from VITRA (metric)
        half_patch: half-size of sampling patch
        metric_scale: global scale factor (divide raw depth by this to get metric)

    Returns:
        list of (metric_depth_surface_z - joint_z) offsets for valid joints
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

        d_surface = float(np.median(valid)) / metric_scale  # raw → metric

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


def rescale_to_metric_scale(recon_dir, annotation_path, results_dir=None,
                            correct_hand_positions=False,
                            per_frame_scale=False,
                            apply_scale=False):
    """Compute the metric scale factor and correct hand positions.

    Non-destructive: depth and camera files are NOT overwritten.  The global
    scale factor is saved to ``metric_scale.txt`` and applied on the fly by
    readers via ``load_metric_scale()``.

    Two-step approach:
      Step 1 — Global scale: median ratio of raw_depth / MANO_vertex_z across
               all frames, using back-face-culled MANO mesh vertices for dense
               sampling.  Saved to ``<recon_dir>/metric_scale.txt``.
      Step 2 — Per-frame hand correction: for each frame with hand data,
               compute a z-offset that aligns the hand joints to the (metric-
               scaled) depth surface.  The offset is applied in world space
               to ``transl_worldspace`` and ``joints_worldspace``, plus
               ``joints_camspace`` for consistency.  Corrected annotation is
               saved to ``<results_dir>/corrected_annotations/<episode_id>.npy``.

    Args:
        recon_dir: Path to reconstruction directory (depth/, camera/).
        annotation_path: Path to the VITRA episode .npy file.
        results_dir: Top-level results directory for this video.  If None,
                     defaults to the parent of recon_dir.

    Returns:
        global_scale: the global scale factor, or None if rescaling was skipped.
    """
    if results_dir is None:
        results_dir = os.path.dirname(recon_dir)

    scale_path = os.path.join(recon_dir, "metric_scale.txt")

    # Skip if already computed
    if os.path.exists(scale_path):
        global_scale = load_metric_scale(recon_dir)
        print(f"  Metric scale already computed: {global_scale:.6f}")

        # Still need to run hand correction if not done yet
        episode_id = os.path.splitext(os.path.basename(annotation_path))[0]
        corrected_dir = os.path.join(results_dir, "corrected_annotations")
        corrected_path = os.path.join(corrected_dir, episode_id + ".npy")
        if correct_hand_positions and not os.path.exists(corrected_path):
            _correct_hand_positions(recon_dir, annotation_path, results_dir,
                                    metric_scale=global_scale)

        return global_scale

    annotation = np.load(annotation_path, allow_pickle=True).item()
    vdf = annotation.get("video_decode_frame")
    if vdf is None:
        print("  WARNING: No video_decode_frame in annotation, skipping rescale")
        return None

    # Use raw (absolute) vdf for filenames — depth files are named with
    # absolute source-video frame indices for both SSv2 and Epic.
    vdf = np.asarray(vdf)

    depth_dir = os.path.join(recon_dir, "depth")
    camera_dir = os.path.join(recon_dir, "camera")

    # ── Collect per-frame hand data ───────────────────────────────────
    frame_hands = _collect_frame_hands(annotation, vdf)

    if not frame_hands:
        print("  WARNING: No valid hand frames, skipping rescale")
        return None

    # ── Step 1: Global scale using MANO mesh vertices ─────────────────
    # Uses back-face-culled MANO vertices (~450 per hand) for dense
    # sampling.  Falls back to 21 joints if MANO model unavailable.
    all_samples = []
    for vf, hands_info in sorted(frame_hands.items()):
        depth_path = os.path.join(depth_dir, f"{vf:06d}.npy")
        cam_path = os.path.join(camera_dir, f"{vf:06d}.npz")
        if not os.path.exists(depth_path) or not os.path.exists(cam_path):
            continue
        depth_map = np.load(depth_path).astype(np.float32)
        recon_K = np.load(cam_path)["intrinsics"]
        for hand_name, vi, joints in hands_info:
            is_left = (hand_name == "left")
            verts = _get_hand_vertices(
                annotation[hand_name], vi, is_left,
                recon_K=recon_K, image_shape=depth_map.shape)
            if verts is None:
                raise RuntimeError(
                    f"MANO model not available for {'left' if is_left else 'right'} hand. "
                    f"Expected at: {_MANO_LEFT_PATH if is_left else _MANO_RIGHT_PATH}")
            all_samples.extend(
                _collect_surface_samples(depth_map, recon_K, verts))

    if len(all_samples) < 5:
        print(f"  WARNING: Only {len(all_samples)} surface samples, skipping rescale")
        return None

    global_scale, filtered = _robust_median(np.array(all_samples))
    print(f"  Global scale: {global_scale:.6f} "
          f"(median of {len(filtered)}/{len(all_samples)} samples)")

    # Save global scale
    with open(scale_path, "w") as f:
        f.write(f"{global_scale}")

    # Optionally apply the scale to depth and camera files in-place
    if apply_scale:
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
        for dp in depth_files:
            d = np.load(dp)
            np.save(dp, (d / global_scale).astype(np.float16))

        camera_files = sorted(glob.glob(os.path.join(camera_dir, "*.npz")))
        for cp in camera_files:
            cam = dict(np.load(cp))
            pose = cam["pose"].copy()
            pose[:3, 3] /= global_scale
            np.savez(cp, pose=pose, intrinsics=cam["intrinsics"])

        # Scale is now baked in — set metric_scale.txt to 1.0 so readers
        # don't double-apply
        with open(scale_path, "w") as f:
            f.write("1.0")

        print(f"  Applied scale to {len(depth_files)} depth maps and "
              f"{len(camera_files)} camera poses")

    # Optionally compute per-frame scales for tighter alignment
    if per_frame_scale:
        import json as _json
        per_frame_scales = {"global": global_scale}
        for vf, hands_info in sorted(frame_hands.items()):
            depth_path = os.path.join(depth_dir, f"{vf:06d}.npy")
            cam_path = os.path.join(camera_dir, f"{vf:06d}.npz")
            if not os.path.exists(depth_path) or not os.path.exists(cam_path):
                continue
            depth_map = np.load(depth_path).astype(np.float32)
            recon_K = np.load(cam_path)["intrinsics"]
            frame_samples = []
            for hand_name, vi, joints in hands_info:
                is_left = (hand_name == "left")
                verts = _get_hand_vertices(
                    annotation[hand_name], vi, is_left,
                    recon_K=recon_K, image_shape=depth_map.shape)
                if verts is None:
                    raise RuntimeError(
                        f"MANO model not available for {'left' if is_left else 'right'} hand. "
                        f"Expected at: {_MANO_LEFT_PATH if is_left else _MANO_RIGHT_PATH}")
                frame_samples.extend(
                    _collect_surface_samples(depth_map, recon_K, verts))
            if len(frame_samples) >= 3:
                frame_scale, _ = _robust_median(np.array(frame_samples))
                per_frame_scales[str(vf)] = round(frame_scale, 8)
        pf_path = os.path.join(recon_dir, "metric_scale_per_frame.json")
        with open(pf_path, "w") as f:
            _json.dump(per_frame_scales, f, indent=2)
        print(f"  Per-frame scales: {len(per_frame_scales)-1} frames "
              f"(saved to metric_scale_per_frame.json)")

    # ── Step 2: Per-frame hand position correction (optional) ──────────
    if correct_hand_positions:
        _correct_hand_positions(recon_dir, annotation_path, results_dir,
                                metric_scale=global_scale)

    return global_scale


def _collect_frame_hands(annotation, vdf):
    """Build mapping: video_frame_index -> list of (hand_name, vi, joints_camspace).

    Frame indexing convention:
      - ``vi`` = VITRA annotation index (0-based row into joints_camspace,
        kept_frames, etc.).  This is the temporal index within the episode.
      - ``vf`` = video frame index from video_decode_frame (absolute source-
        video frame index).  Depth/camera files are named ``{vf:06d}.npy``.
      - Both SSv2 and Epic use absolute frame indices in output filenames.

    Args:
        annotation: VITRA annotation dict
        vdf: rebased video_decode_frame array (clip-relative frame indices)

    Returns:
        dict: vf_int -> [(hand_name, vi, (21, 3) joints_camspace), ...]
              vi is the VITRA annotation index (row in joints_camspace)
              vf_int is the clip-relative frame index (matches depth filenames)
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


def _correct_hand_positions(recon_dir, annotation_path, results_dir,
                            metric_scale=1.0):
    """Compute per-frame z-offsets to align hand joints with the depth surface,
    and save a corrected episode .npy that training can load directly.

    For each frame with valid hand data, projects the VITRA wrist/proximal
    joints into the depth map (applying metric_scale on-the-fly), computes the
    median z-offset, then transforms that offset to **world space** and applies
    it to ``transl_worldspace`` and ``joints_worldspace`` (the fields read by
    training).

    **Occlusion robustness**: Only wrist + MCP joints are used (held objects
    occlude fingertips first).  A temporal outlier pass rejects per-frame
    offsets that deviate >2 sigma from the per-hand median, preventing
    sporadic occlusion frames from corrupting the correction.

    Saves the corrected annotation to
    ``<results_dir>/corrected_annotations/<episode_id>.npy`` as a drop-in
    replacement for the original episode file.

    Args:
        recon_dir: Path to reconstruction directory (depth/, camera/).
        annotation_path: Path to the VITRA episode .npy file.
        results_dir: Top-level results directory for this video.
        metric_scale: Global scale factor (raw depth / metric_scale = metric depth).
    """
    import copy

    annotation = np.load(annotation_path, allow_pickle=True).item()
    vdf = annotation.get("video_decode_frame")
    if vdf is None:
        return

    vdf = np.asarray(vdf)  # absolute frame indices (matches depth filenames)
    frame_hands = _collect_frame_hands(annotation, vdf)
    if not frame_hands:
        return

    depth_dir = os.path.join(recon_dir, "depth")
    camera_dir = os.path.join(recon_dir, "camera")

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

        depth_map = np.load(depth_path).astype(np.float32)
        recon_K = np.load(cam_path)["intrinsics"]

        for hand_name, vi, joints in hands_info:
            offsets = _collect_joint_offsets(depth_map, recon_K, joints,
                                            metric_scale=metric_scale)
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



# rescale_depth_maps_only was removed — use rescale_to_metric_scale instead,
# which computes the same scale but saves it to metric_scale.txt without
# overwriting depth/camera files (non-destructive).


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
