#!/usr/bin/env python3
"""
Shared HaWoR hand estimation utilities.

Functions for running HaWoR with known intrinsics, computing MANO joints,
building VITRA-format hand annotations with depth correction, and loading
Pi3 reconstruction outputs.

Used by:
  - batch_worker.py (integrated pipeline)
  - rerun_hawor_pi3.py (standalone re-processing)
"""

import copy
import glob
import os
import sys

import cv2
import numpy as np
import torch

# Add VITRA3D to path for HaWoR/MANO imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VITRA3D_ROOT = os.path.join(PROJECT_ROOT, 'VITRA3D')
if VITRA3D_ROOT not in sys.path:
    sys.path.insert(0, VITRA3D_ROOT)

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


def load_hawor_models(device="cuda"):
    """Load HaWoR pipeline and MANO model.

    Must be called from VITRA3D root or with correct sys.path.
    Returns (hawor_pipeline, mano_model).
    """
    from data.tools.utils_hawor import HaworPipeline
    from libs.models.mano_wrapper import MANO

    original_cwd = os.getcwd()
    os.chdir(VITRA3D_ROOT)

    mano_model = MANO(
        model_path=os.path.join(VITRA3D_ROOT, 'weights', 'mano')
    ).to(device)
    mano_model.eval()

    hawor_pipeline = HaworPipeline(
        model_path=os.path.join(VITRA3D_ROOT, 'weights', 'hawor',
                                'checkpoints', 'hawor.ckpt'),
        detector_path=os.path.join(VITRA3D_ROOT, 'weights', 'hawor',
                                   'external', 'detector.pt'),
        device=device,
    )

    os.chdir(original_cwd)
    return hawor_pipeline, mano_model


def load_video_frames(video_path, frames_dir=None):
    """Load video frames as list of BGR numpy arrays.

    If frames_dir is available (from Pi3 output), loads from there.
    Otherwise loads from video file.
    """
    if frames_dir and os.path.isdir(frames_dir):
        color_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")) +
                             glob.glob(os.path.join(frames_dir, "*.jpg")))
        if color_files:
            return [cv2.imread(f) for f in color_files if cv2.imread(f) is not None]

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_pi3_intrinsics(camera_dir):
    """Load intrinsics from Pi3 camera files.

    Returns (K, focal_length) or (None, None) if no camera files found.
    """
    cam_files = sorted(glob.glob(os.path.join(camera_dir, "*.npz")))
    if not cam_files:
        return None, None
    cam = np.load(cam_files[0])
    K = cam["intrinsics"].astype(np.float64)
    return K, float(K[0, 0])


def build_frame_num_to_list_idx(color_dir):
    """Build mapping from video frame number to loaded-list index.

    Pi3 color files are named by frame number (e.g., 000009.png) but
    HaWoR processes them as a list indexed 0..N-1.
    """
    color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")) +
                         glob.glob(os.path.join(color_dir, "*.jpg")))
    mapping = {}
    for list_idx, f in enumerate(color_files):
        frame_num = int(os.path.splitext(os.path.basename(f))[0])
        mapping[frame_num] = list_idx
    return mapping


def run_hawor(hawor_pipeline, mano_model, images, focal_length, device):
    """Run HaWoR hand estimation and MANO wrist alignment.

    Args:
        hawor_pipeline: HaworPipeline instance
        mano_model: MANO model instance
        images: list of BGR numpy arrays
        focal_length: focal length in pixels
        device: torch device

    Returns:
        dict with 'left' and 'right' keys, each mapping
        list_index -> {beta, hand_pose, global_orient, transl}
    """
    recon_results = hawor_pipeline.recon(
        images, img_focal=focal_length, single_image=(len(images) == 1)
    )

    aligned = {'left': {}, 'right': {}}
    for frame_idx in range(len(images)):
        for hand_type in ['left', 'right']:
            if frame_idx not in recon_results[hand_type]:
                continue
            result = recon_results[hand_type][frame_idx]

            betas = torch.from_numpy(result['beta']).unsqueeze(0).float().to(device)
            hand_pose = torch.from_numpy(result['hand_pose']).unsqueeze(0).float().to(device)
            transl = torch.from_numpy(result['transl']).unsqueeze(0).float().to(device)

            model_output = mano_model(betas=betas, hand_pose=hand_pose)
            joints_m = model_output.joints[0]

            if hand_type == 'left':
                joints_m[:, 0] = -1 * joints_m[:, 0]

            wrist = joints_m[0]
            transl_new = wrist + transl

            result_aligned = copy.deepcopy(result)
            result_aligned['transl'] = transl_new[0].cpu().numpy()
            aligned[hand_type][frame_idx] = result_aligned

    return aligned


def compute_joints_camspace(mano_model, result, hand_type, device):
    """Compute 21 joints in camera space using MANO forward pass.

    Uses the formula: joints_cam = R_cam @ (J_local - J0) + T_cam
    where J_local is from MANO with identity global_orient.
    """
    betas = torch.from_numpy(result['beta']).unsqueeze(0).float().to(device)
    hand_pose = torch.from_numpy(result['hand_pose']).unsqueeze(0).float().to(device)

    identity_go = torch.eye(3).float().unsqueeze(0).unsqueeze(0).to(device)
    output = mano_model(
        betas=betas,
        hand_pose=hand_pose,
        global_orient=identity_go,
    )
    joints_local = output.joints[0].cpu().numpy()  # (21, 3)

    if hand_type == 'left':
        joints_local[:, 0] *= -1

    j0 = joints_local[0:1]
    joints_centered = joints_local - j0

    go = result['global_orient']
    if go.ndim == 3:
        go = go[0]
    transl = result['transl']

    joints_cam = (go @ joints_centered.T).T + transl
    return joints_cam.astype(np.float32)


def build_hand_dict(hawor_results, mano_model, T, extrinsics_w2c, hand_type,
                    device, vdf=None, frame_num_to_list_idx=None):
    """Build per-hand annotation dict in VITRA format from raw HaWoR output.

    Uses HaWoR's camera-space predictions directly — no depth correction.
    The hand serves as a metric ruler; depth maps are rescaled to match
    via rescale_to_metric afterwards.

    Args:
        hawor_results: dict mapping list_index -> result dict
        mano_model: MANO model instance
        T: number of annotation frames (len(vdf))
        extrinsics_w2c: (T, 4, 4) world-to-camera matrices
        hand_type: 'left' or 'right'
        device: torch device
        vdf: video_decode_frame array
        frame_num_to_list_idx: dict mapping video frame number -> list index
    """
    beta = np.zeros(10, dtype=np.float64)
    global_orient_cam = np.zeros((T, 3, 3), dtype=np.float64)
    global_orient_world = np.zeros((T, 3, 3), dtype=np.float64)
    hand_pose = np.zeros((T, 15, 3, 3), dtype=np.float64)
    transl_cam = np.zeros((T, 3), dtype=np.float64)
    transl_world = np.zeros((T, 3), dtype=np.float64)
    joints_cam = np.zeros((T, 21, 3), dtype=np.float32)
    joints_world = np.zeros((T, 21, 3), dtype=np.float64)
    kept_frames = np.zeros(T, dtype=np.int64)

    betas_list = [r['beta'] for r in hawor_results.values()]
    if betas_list:
        beta = np.median(np.array(betas_list), axis=0).astype(np.float64)

    for vi in range(T):
        if vdf is not None:
            video_frame_idx = int(vdf[vi])
        else:
            video_frame_idx = vi

        if frame_num_to_list_idx is not None:
            list_idx = frame_num_to_list_idx.get(video_frame_idx)
            if list_idx is None or list_idx not in hawor_results:
                continue
        else:
            list_idx = video_frame_idx
            if list_idx not in hawor_results:
                continue

        result = hawor_results[list_idx]
        kept_frames[vi] = 1

        go = result['global_orient']
        if go.ndim == 3:
            go = go[0]
        global_orient_cam[vi] = go
        hand_pose[vi] = result['hand_pose']

        j_cam = compute_joints_camspace(mano_model, result, hand_type, device)
        result_transl = result['transl']

        transl_cam[vi] = result_transl
        joints_cam[vi] = j_cam.astype(np.float32)

        # Convert to world space using extrinsics
        w2c = extrinsics_w2c[vi]
        R_w2c = w2c[:3, :3]
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ w2c[:3, 3]

        global_orient_world[vi] = R_c2w @ go
        transl_world[vi] = R_c2w @ result_transl + t_c2w
        joints_world[vi] = (R_c2w @ j_cam.T + t_c2w[:, None]).T

    return {
        'beta': beta,
        'global_orient_camspace': global_orient_cam,
        'global_orient_worldspace': global_orient_world,
        'hand_pose': hand_pose,
        'transl_camspace': transl_cam,
        'transl_worldspace': transl_world,
        'kept_frames': kept_frames,
        'joints_camspace': joints_cam,
        'joints_worldspace': joints_world,
        'wrist': None,
        'max_finger_joint_angle_movement': None,
        'max_translation_movement': None,
        'max_wrist_rotation_movement': None,
    }


def process_video_hawor(video_name, video_path, video_dir, annotation_path,
                        hawor_pipeline, mano_model, device, output_dir):
    """Run HaWoR on one video using Pi3's intrinsics + depth correction.

    HaWoR runs on Pi3's color frames (at Pi3's resolution) with Pi3's
    focal length.  This keeps everything in Pi3's camera model — HaWoR
    joints, depth maps, and camera poses are all consistent.

    Args:
        video_name: video identifier (e.g., "129605")
        video_path: path to source video file
        video_dir: Pi3 output directory for this video
        annotation_path: path to original VITRA annotation .npy
        hawor_pipeline: HaworPipeline instance
        mano_model: MANO model instance
        device: torch device
        output_dir: top-level results directory

    Returns:
        path to saved annotation, or None on failure
    """
    intermediate_dir = os.path.join(video_dir, "intermediate_depth")
    camera_dir = os.path.join(intermediate_dir, "camera")
    depth_dir = os.path.join(intermediate_dir, "depth")
    color_dir = os.path.join(intermediate_dir, "color")

    K, focal_length = get_pi3_intrinsics(camera_dir)
    if K is None:
        print(f"  No camera files found, skipping HaWoR")
        return None
    print(f"  Pi3 focal length: {focal_length:.1f}")

    orig_ann = np.load(annotation_path, allow_pickle=True).item()
    vdf = orig_ann.get("video_decode_frame")
    if vdf is None:
        print(f"  No video_decode_frame, skipping HaWoR")
        return None
    vdf = _rebase_vdf(vdf, annotation_path)
    T = len(vdf)

    # Load Pi3's color frames (at Pi3's resolution)
    frames = load_video_frames(video_path, frames_dir=color_dir)
    if not frames:
        print(f"  No frames loaded, skipping HaWoR")
        return None
    print(f"  Loaded {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]})")

    frame_num_to_list_idx = build_frame_num_to_list_idx(color_dir)

    print(f"  Running HaWoR...")
    hawor_results = run_hawor(hawor_pipeline, mano_model, frames, focal_length, device)
    n_left = len(hawor_results['left'])
    n_right = len(hawor_results['right'])
    print(f"  HaWoR detected: left={n_left}, right={n_right} frames")

    extrinsics_w2c = orig_ann.get("extrinsics")
    if extrinsics_w2c is None:
        extrinsics_w2c = np.tile(np.eye(4), (T, 1, 1))

    left_dict = build_hand_dict(
        hawor_results['left'], mano_model, T, extrinsics_w2c, 'left', device,
        vdf=vdf, frame_num_to_list_idx=frame_num_to_list_idx,
    )
    right_dict = build_hand_dict(
        hawor_results['right'], mano_model, T, extrinsics_w2c, 'right', device,
        vdf=vdf, frame_num_to_list_idx=frame_num_to_list_idx,
    )

    corrected_ann = copy.deepcopy(orig_ann)
    corrected_ann['left'] = left_dict
    corrected_ann['right'] = right_dict
    corrected_ann['intrinsics'] = K

    left_count = left_dict['kept_frames'].sum()
    right_count = right_dict['kept_frames'].sum()
    corrected_ann['anno_type'] = 'right' if right_count >= left_count else 'left'

    episode_id = os.path.splitext(os.path.basename(annotation_path))[0]
    corrected_dir = os.path.join(output_dir, "hawor_annotations")
    os.makedirs(corrected_dir, exist_ok=True)
    corrected_path = os.path.join(corrected_dir, episode_id + ".npy")
    np.save(corrected_path, corrected_ann)

    print(f"  Saved: {corrected_path}")
    print(f"  Left: {left_count}/{T} kept, Right: {right_count}/{T} kept")

    return corrected_path
