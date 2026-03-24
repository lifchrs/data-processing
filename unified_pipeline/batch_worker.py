#!/usr/bin/env python3
"""
Persistent batch worker for the unified pipeline.

Processes multiple videos from a manifest file, keeping models loaded between
videos to avoid repeated model initialization overhead (~30-60s per video).

Two modes:
  - TTT3R mode:  Loads TTT3R model once, processes all videos for Step 1.
                 Runs undistortion (Ego4D/EgoExo4D) before each video.
  - SAM3D mode:  Loads SAM3 + Inference models once, processes Steps 3-4 for all videos.
                 Auto-derives frames_dir/camera_dir from intermediate_depth if not in manifest.
                 Runs quality flags check after Step 3.

Usage (run from the appropriate conda env):

    # TTT3R batch worker (run in ttt3r env):
    python batch_worker.py --mode ttt3r --manifest videos.json

    # SAM3D batch worker (run in sam3d-objects env):
    python batch_worker.py --mode sam3d --manifest videos.json

Manifest format: JSON list of dicts:
    [
        {
            "video_path": "/path/to/video.mp4",
            "output_dir": "/path/to/output",
            "annotation_path": "/path/to/annotation.npy",  // optional
            "frames_dir": "/path/to/frames",                // optional, auto-derived from intermediate_depth
            "camera_dir": "/path/to/cameras",               // optional, auto-derived from intermediate_depth
            "prompt": "object held in hand"                  // optional
        },
        ...
    ]

The worker processes videos sequentially on a single GPU. For multi-GPU
parallelism, launch one worker per GPU with CUDA_VISIBLE_DEVICES set, each
with a different slice of the manifest.
"""

import argparse
import json
import os
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Datasets where video_decode_frame contains absolute frame indices from the
# full source video, NOT relative to the clip we actually process.  For these
# datasets the pipeline works on a short extracted clip, so vdf must be rebased
# to clip-relative indices (subtract vdf[0]).
_ABSOLUTE_VDF_DATASETS = {"epic", "ego4d", "egoexo4d"}


def _detect_dataset(annotation_path):
    """Determine the source dataset from an annotation filename prefix."""
    if not annotation_path:
        return None
    basename = os.path.basename(annotation_path)
    for prefix, dataset in [
        ("somethingsomethingv2_", "ssv2"),
        ("epic_kitchens_", "epic"),
        ("EgoExo4D_", "egoexo4d"),
        ("Ego4D_", "ego4d"),
    ]:
        if basename.startswith(prefix):
            return dataset
    return None


def _rebase_vdf(vdf, annotation_path):
    """Rebase video_decode_frame to clip-relative indices if needed.

    For datasets like Epic-Kitchens, vdf contains absolute frame indices from
    the full source video (e.g., 84849).  When we process a pre-extracted clip,
    these must be rebased to start from 0.
    """
    dataset = _detect_dataset(annotation_path)
    if dataset in _ABSOLUTE_VDF_DATASETS:
        return vdf - vdf[0]
    return vdf


def _rename_intermediate_depths(ttt3r_out, offset, n_frames):
    """Rename sequential TTT3R output files to original video frame indices.

    prepare_output saves files as 000000, 000001, ... but when we clipped
    the input to a subrange starting at `offset`, file 000000 actually
    corresponds to video frame `offset`.  Rename in reverse order to avoid
    collisions (target indices are always >= source indices).
    """
    for subdir in ["depth", "conf", "color", "camera"]:
        d = os.path.join(ttt3r_out, subdir)
        if not os.path.isdir(d):
            continue
        # Rename in reverse to avoid collisions
        for seq_idx in range(n_frames - 1, -1, -1):
            orig_idx = seq_idx + offset
            if seq_idx == orig_idx:
                continue
            ext = ".png" if subdir == "color" else (".npz" if subdir == "camera" else ".npy")
            src = os.path.join(d, f"{seq_idx:06d}{ext}")
            dst = os.path.join(d, f"{orig_idx:06d}{ext}")
            if os.path.exists(src):
                os.rename(src, dst)


def _run_undistortion(video_path, annotation_path, output_dir, intrinsics_root=None):
    """Run undistortion as subprocess if needed. Returns effective video path."""
    if not annotation_path or not os.path.exists(annotation_path):
        return video_path

    undistorted_path = os.path.join(output_dir, "undistorted_video.mp4")
    if os.path.exists(undistorted_path):
        return undistorted_path  # already undistorted

    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "undistort_clip.py"),
        "--video", os.path.abspath(video_path),
        "--annotation", os.path.abspath(annotation_path),
        "--output", os.path.abspath(undistorted_path),
    ]
    if intrinsics_root:
        cmd.extend(["--intrinsics_root", os.path.abspath(intrinsics_root)])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  WARNING: Undistortion failed: {result.stderr[:200]}")
        return video_path

    for line in result.stdout.splitlines():
        if line.startswith("USE_VIDEO="):
            return line.split("=", 1)[1].strip()

    return video_path


# ---------------------------------------------------------------------------
# TTT3R Batch Worker
# ---------------------------------------------------------------------------

def run_ttt3r_batch(manifest, model_path, img_size, device, frame_interval,
                    intrinsics_root=None):
    """Load TTT3R model once and process all videos."""
    # Add TTT3R to path
    ttt3r_dir = os.path.join(PROJECT_ROOT, "TTT3R")
    sys.path.insert(0, ttt3r_dir)
    os.chdir(ttt3r_dir)

    import torch
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(model_path)
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    # Import demo helpers
    from demo import prepare_input, prepare_output, parse_seq_path

    # Load model once
    print(f"Loading TTT3R model from {model_path}...")
    t0 = time.time()
    model = ARCroco3DStereo.from_pretrained(model_path).to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    results = []
    for i, entry in enumerate(manifest):
        video_path = entry["video_path"]
        output_dir = entry["output_dir"]
        annotation_path = entry.get("annotation_path")
        ttt3r_out = os.path.join(output_dir, "intermediate_depth")
        os.makedirs(ttt3r_out, exist_ok=True)

        # Skip if already processed
        depth_out = os.path.join(ttt3r_out, "depth")
        if os.path.isdir(depth_out) and os.listdir(depth_out):
            print(f"\n[{i+1}/{len(manifest)}] Skipping (already done): {video_path}")
            results.append({"video": video_path, "success": True, "skipped": True})
            continue

        print(f"\n[{i+1}/{len(manifest)}] Processing: {video_path}")
        t_start = time.time()

        try:
            import shutil

            # Clean up partial output from previous runs
            if os.path.isdir(ttt3r_out):
                for sub in os.listdir(ttt3r_out):
                    sub_path = os.path.join(ttt3r_out, sub)
                    if os.path.isdir(sub_path):
                        shutil.rmtree(sub_path)

            # Undistort if needed (Ego4D/EgoExo4D)
            effective_video = _run_undistortion(
                video_path, annotation_path, output_dir, intrinsics_root)
            if effective_video != video_path:
                print(f"  Using undistorted video: {effective_video}")

            # Prepare input frames
            img_paths, tmpdirname = parse_seq_path(effective_video, frame_interval)
            if not img_paths:
                print(f"  No images found, skipping")
                results.append({"video": video_path, "success": False, "error": "no_frames"})
                continue

            # Clip to VITRA annotation range + margin to avoid processing
            # unnecessary frames (VITRA episodes are often subclips)
            clip_offset = 0
            if annotation_path and os.path.exists(annotation_path):
                import numpy as np
                annot = np.load(annotation_path, allow_pickle=True).item()
                vdf = annot.get('video_decode_frame')
                if vdf is not None and len(vdf) > 0:
                    margin = 3
                    vdf = _rebase_vdf(vdf, annotation_path)
                    first_idx = vdf[0] // frame_interval
                    last_idx = vdf[-1] // frame_interval
                    clip_start = max(0, first_idx - margin)
                    clip_end = min(len(img_paths) - 1, last_idx + margin)
                    n_before = len(img_paths)
                    img_paths = img_paths[clip_start:clip_end + 1]
                    clip_offset = clip_start
                    print(f" - Clipping to VITRA range: frames {clip_start}-{clip_end} "
                          f"({len(img_paths)}/{n_before} frames)")

            img_mask = [True] * len(img_paths)
            views = prepare_input(
                img_paths=img_paths, img_mask=img_mask, size=img_size,
                revisit=1, update=True, reset_interval=1000000,
            )
            if tmpdirname is not None:
                shutil.rmtree(tmpdirname)

            # Reset model state for new video
            model.config.model_update_type = "ttt3r"

            # Run inference
            outputs, state_args = inference_recurrent_lighter(views, model, device)

            # Save outputs
            prepare_output(outputs, ttt3r_out, 1, True)

            # Rename sequential outputs to original video frame indices
            if clip_offset > 0:
                _rename_intermediate_depths(ttt3r_out, clip_offset, len(img_paths))

            elapsed = time.time() - t_start
            print(f"  Done in {elapsed:.1f}s")
            results.append({"video": video_path, "success": True, "elapsed_s": round(elapsed, 1)})

        except Exception as e:
            elapsed = time.time() - t_start
            print(f"  FAILED: {e}")
            results.append({"video": video_path, "success": False, "error": str(e),
                            "elapsed_s": round(elapsed, 1)})

    return results


# ---------------------------------------------------------------------------
# Pi3 Batch Worker
# ---------------------------------------------------------------------------

def run_pi3_batch(manifest, device, frame_interval, pi3_ckpt=None,
                  intrinsics_root=None):
    """Load Pi3 model once and process all videos.

    For each video:
      1. Pi3 reconstruction → depth, color, camera
      2. Metric rescaling using VITRA hand annotations as ruler
    """
    import torch

    pi3_dir = os.path.join(PROJECT_ROOT, "Pi3")
    sys.path.insert(0, pi3_dir)

    from pi3.utils.basic import load_images_as_tensor
    from pi3.models.pi3x import Pi3X

    # Load Pi3 model
    print(f"Loading Pi3X model...")
    t0 = time.time()
    if pi3_ckpt is not None:
        model = Pi3X(use_multimodal=False).eval()
        if pi3_ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(pi3_ckpt)
        else:
            weight = torch.load(pi3_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        model.disable_multimodal()
    model = model.to(device)
    print(f"Pi3 model loaded in {time.time() - t0:.1f}s")

    dtype = torch.bfloat16 if (device != "cpu" and
            torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    # Top-level results dir (parent of per-video output_dirs)
    results_dir = os.path.dirname(manifest[0]["output_dir"]) if manifest else "."

    results = []
    for i, entry in enumerate(manifest):
        video_path = entry["video_path"]
        output_dir = entry["output_dir"]
        annotation_path = entry.get("annotation_path")
        pi3_out = os.path.join(output_dir, "intermediate_depth")
        os.makedirs(pi3_out, exist_ok=True)

        print(f"\n[{i+1}/{len(manifest)}] Processing: {os.path.basename(output_dir)}")
        t_start = time.time()

        try:
            import shutil
            import cv2
            import numpy as np

            # Clean up any previous output
            if os.path.isdir(pi3_out):
                shutil.rmtree(pi3_out)

            # Undistort if needed
            effective_video = _run_undistortion(
                video_path, annotation_path, output_dir, intrinsics_root)
            if effective_video != video_path:
                print(f"  Using undistorted video: {effective_video}")

            # Create output dirs
            depth_dir = os.path.join(pi3_out, "depth")
            color_dir = os.path.join(pi3_out, "color")
            camera_dir = os.path.join(pi3_out, "camera")
            for d in [depth_dir, color_dir, camera_dir]:
                os.makedirs(d, exist_ok=True)

            # Load frames
            imgs = load_images_as_tensor(effective_video, interval=frame_interval).to(device)
            N, C, H, W = imgs.shape

            # Load original-resolution frames for color output
            orig_frames = []
            cap = cv2.VideoCapture(effective_video)
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    orig_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_idx += 1
            cap.release()
            orig_frames = orig_frames[:N]

            # Clip to VITRA range if available
            clip_offset = 0
            if annotation_path and os.path.exists(annotation_path):
                annot = np.load(annotation_path, allow_pickle=True).item()
                vdf = annot.get('video_decode_frame')
                if vdf is not None and len(vdf) > 0:
                    margin = 3
                    vdf = _rebase_vdf(vdf, annotation_path)
                    first_idx = vdf[0] // frame_interval
                    last_idx = vdf[-1] // frame_interval
                    clip_start = max(0, first_idx - margin)
                    clip_end = min(N - 1, last_idx + margin)
                    n_before = N
                    imgs = imgs[clip_start:clip_end + 1]
                    orig_frames = orig_frames[clip_start:clip_end + 1]
                    clip_offset = clip_start
                    N = imgs.shape[0]
                    print(f"  Clipping to VITRA range: frames {clip_start}-{clip_end} "
                          f"({N}/{n_before} frames)")

            # Run Pi3 inference
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=dtype):
                    res = model(imgs[None])

            local_points = res['local_points'][0].cpu().numpy()
            camera_poses = res['camera_poses'][0].cpu().numpy()
            conf = res['conf'][0].cpu().numpy()

            # Derive intrinsics
            from run_pi3 import derive_intrinsics
            K = derive_intrinsics(local_points, conf)

            # Save per-frame outputs
            for fi in range(N):
                frame_num = (fi + clip_offset) * frame_interval
                basename = f"{frame_num:06d}"

                depth = local_points[fi, :, :, 2].copy()
                depth[depth < 0] = 0
                np.save(os.path.join(depth_dir, f"{basename}.npy"),
                        depth.astype(np.float32))

                np.savez(os.path.join(camera_dir, f"{basename}.npz"),
                         pose=camera_poses[fi].astype(np.float64),
                         intrinsics=K.astype(np.float64))

                if fi < len(orig_frames):
                    color = orig_frames[fi]
                    if color.shape[0] != H or color.shape[1] != W:
                        color = cv2.resize(color, (W, H))
                    cv2.imwrite(os.path.join(color_dir, f"{basename}.png"),
                                cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

            print(f"  Pi3 done in {time.time() - t_start:.1f}s ({N} frames)")

            # --- Step 2: Metric rescaling ---
            # Pi3's depth scale can be off by 2-7x. Use VITRA's MANO hand
            # annotations as a metric ruler to rescale depth maps + camera
            # translations to true metric.
            if annotation_path and os.path.exists(annotation_path):
                from rescale_to_metric import rescale_depth_maps_only
                rescale_depth_maps_only(pi3_out, annotation_path)

            elapsed = time.time() - t_start
            print(f"  Total: {elapsed:.1f}s")
            results.append({"video": video_path, "success": True,
                            "elapsed_s": round(elapsed, 1)})

        except Exception as e:
            import traceback
            elapsed = time.time() - t_start
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append({"video": video_path, "success": False,
                            "error": str(e), "elapsed_s": round(elapsed, 1)})

    return results


# ---------------------------------------------------------------------------
# SAM3D Batch Worker (Steps 2-4: Scene Seg + Object Detection + Combine)
# ---------------------------------------------------------------------------

def run_sam3d_batch(manifest, device, frame_interval, skip_scene_seg=False,
                    skip_sam3d=False, skip_combine=False, skip_visualizations=False,
                    intrinsics_root=None):
    """Load SAM3 models once and process all videos for steps 3-4."""
    import torch
    import numpy as np
    from PIL import Image

    # Import quality flags checker
    sys.path.insert(0, SCRIPT_DIR)
    from run_pipeline import check_quality_flags

    # Set up SAM3D imports
    sam3d_root = os.path.join(PROJECT_ROOT, "sam-3d-objects")
    sam3d_objects_dir = os.path.join(sam3d_root, "sam3d_objects")
    notebook_dir = os.path.join(sam3d_root, "notebook")
    sys.path.insert(0, sam3d_objects_dir)
    sys.path.insert(0, notebook_dir)

    from transformers import Sam3VideoModel, Sam3VideoProcessor
    from accelerate import Accelerator
    from inference import Inference

    # Load models once
    print("Loading SAM3 models (one-time initialization)...")
    t0 = time.time()

    accelerator = Accelerator()
    dev = accelerator.device

    # SAM3 video model
    sam3_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(dev, dtype=torch.bfloat16)
    sam3_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    # Inference model (for 3D reconstruction)
    config_path = os.path.join(sam3d_root, "checkpoints", "hf", "pipeline.yaml")
    inference_model = Inference(config_path, compile=False)

    print(f"Models loaded in {time.time() - t0:.1f}s")

    results = []
    for i, entry in enumerate(manifest):
        video_path = entry["video_path"]
        output_dir = entry["output_dir"]
        annotation_path = entry.get("annotation_path")
        ttt3r_out = os.path.join(output_dir, "intermediate_depth")
        frames_dir = entry.get("frames_dir") or os.path.join(ttt3r_out, "color")
        camera_dir = entry.get("camera_dir") or os.path.join(ttt3r_out, "camera")
        prompt = entry.get("prompt", "object held in either hand")

        # Skip if final output already exists
        ply_out = os.path.join(output_dir, "depth")
        if os.path.isdir(ply_out) and any(
            f.endswith(".npy") for f in os.listdir(ply_out)
        ):
            print(f"\n[{i+1}/{len(manifest)}] Skipping (already done): {os.path.basename(output_dir)}")
            results.append({"video": video_path, "success": True, "skipped": True})
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(manifest)}] Processing: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        t_start = time.time()

        try:
            # --- Metric rescaling ---
            if annotation_path and os.path.exists(annotation_path):
                from rescale_to_metric import rescale_ttt3r_to_metric
                rescale_ttt3r_to_metric(ttt3r_out, annotation_path,
                                        results_dir=output_dir)

            # --- Step 3: SAM3D Object Detection + Reconstruction ---
            if not skip_sam3d:
                sam3d_out = os.path.join(output_dir, "sam3d_output")
                os.makedirs(sam3d_out, exist_ok=True)

                # Run track_objects logic with pre-loaded models
                _run_sam3d_single(
                    sam3_model, sam3_processor, inference_model,
                    video_path=video_path,
                    frames_dir=frames_dir,
                    output_dir=sam3d_out,
                    camera_dir=camera_dir,
                    prompt=prompt,
                    stride=frame_interval,
                    annotation_path=annotation_path,
                    device=dev,
                )
            else:
                sam3d_out = os.path.join(output_dir, "sam3d_output")
                if not os.path.exists(sam3d_out):
                    sam3d_out = None

            # --- Quality flags ---
            action_text = prompt
            if annotation_path and os.path.exists(annotation_path):
                try:
                    annotation = np.load(annotation_path, allow_pickle=True).item()
                    text_data = annotation.get('text', {})
                    for hand in ['left', 'right']:
                        actions = text_data.get(hand, [])
                        if actions:
                            action_text = actions[0][0]
                            break
                except Exception:
                    pass

            episode_id = None
            if annotation_path:
                episode_id = os.path.splitext(os.path.basename(annotation_path))[0]
            check_quality_flags(
                output_dir, sam3d_out, action_text,
                episode_id=episode_id, sam3d_skipped=skip_sam3d,
            )

            # --- Step 4: Combine outputs ---
            if not skip_combine:
                _run_combine_single(
                    ttt3r_out=ttt3r_out,
                    sam3d_out=sam3d_out,
                    output_dir=output_dir,
                    frame_interval=frame_interval,
                    annotation_path=annotation_path,
                    skip_visualizations=skip_visualizations,
                )

            elapsed = time.time() - t_start
            print(f"  Done in {elapsed:.1f}s")
            results.append({"video": video_path, "success": True, "elapsed_s": round(elapsed, 1)})

        except Exception as e:
            import traceback
            elapsed = time.time() - t_start
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append({"video": video_path, "success": False, "error": str(e),
                            "elapsed_s": round(elapsed, 1)})

    return results


def _run_sam3d_single(model, processor, inference_model,
                      video_path, frames_dir, output_dir, camera_dir,
                      prompt, stride, annotation_path, device):
    """Run SAM3D detection + reconstruction for a single video using pre-loaded models."""
    import torch
    import numpy as np
    from PIL import Image

    from detection_strategies import detect_object
    from text_utils import extract_object_noun

    # Load video frames
    # frame_index_map: list position → original video frame number
    frame_index_map = None
    if frames_dir is not None:
        import glob as _glob
        frame_paths = sorted(_glob.glob(os.path.join(frames_dir, "*.png"))
                             + _glob.glob(os.path.join(frames_dir, "*.jpg")))
        video_frames = [Image.open(p).convert("RGB") for p in frame_paths]
        # Extract original frame indices from filenames (e.g., 000007.png → 7)
        frame_index_map = [
            int(os.path.splitext(os.path.basename(p))[0]) for p in frame_paths
        ]
    else:
        from transformers.video_utils import load_video
        try:
            video_frames, _ = load_video(video_path)
        except Exception:
            import cv2
            cap = cv2.VideoCapture(video_path)
            video_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()

    # Parse VITRA annotation if available
    hand_joints = None
    intrinsics = None
    kept_frames = None
    detection_strategy = "simple"
    detection_kwargs = {}

    if annotation_path and os.path.exists(annotation_path):
        annotation = np.load(annotation_path, allow_pickle=True).item()
        text_data = annotation.get('text', {})
        action_text = None
        for hand in ['left', 'right']:
            actions = text_data.get(hand, [])
            if actions:
                action_text = actions[0][0]
                break
        if action_text:
            prompt = extract_object_noun(action_text)

        intrinsics = annotation.get('intrinsics')
        for hand in ['left', 'right']:
            if hand in annotation:
                hand_data = annotation[hand]
                hand_joints = hand_data.get('joints_camspace')
                kept_frames = hand_data.get('kept_frames')
                if hand_joints is not None:
                    break

        if hand_joints is not None and intrinsics is not None:
            # Remap VITRA-indexed hand data to video_frames list positions.
            # VITRA index i corresponds to video frame video_decode_frame[i].
            # video_frames list position j corresponds to video frame frame_index_map[j].
            # We need hand_joints/kept_frames indexed by list position j.
            vdf = annotation.get('video_decode_frame')
            if vdf is not None and frame_index_map is not None:
                vdf = _rebase_vdf(vdf, annotation_path)
                frame_num_to_pos = {fn: pos for pos, fn in enumerate(frame_index_map)}
                n_list = len(video_frames)
                remapped_joints = np.zeros((n_list, hand_joints.shape[1], hand_joints.shape[2]))
                remapped_kept = np.zeros(n_list, dtype=int)
                for vi, vf in enumerate(vdf):
                    if vi < len(hand_joints) and vf in frame_num_to_pos:
                        pos = frame_num_to_pos[vf]
                        remapped_joints[pos] = hand_joints[vi]
                        if kept_frames is not None and vi < len(kept_frames):
                            remapped_kept[pos] = kept_frames[vi]
                        else:
                            remapped_kept[pos] = 1
                hand_joints = remapped_joints
                kept_frames = remapped_kept

            detection_strategy = "hand_point"
            detection_kwargs.update({
                "hand_joints": hand_joints,
                "intrinsics": intrinsics,
                "kept_frames": kept_frames,
            })

    print(f"  Detection strategy: {detection_strategy}, prompt: '{prompt}'")

    # Run detection
    detection_result = detect_object(
        model=model, processor=processor, video_frames=video_frames,
        prompt=prompt, strategy=detection_strategy,
        device=device, dtype=torch.bfloat16,
        **detection_kwargs,
    )

    print(f"  Detection: {len(detection_result.masks)} frames, "
          f"best frame: {detection_result.best_frame_idx}")

    # Convert to outputs_per_frame format
    outputs_per_frame = {}
    for frame_idx, mask in detection_result.masks.items():
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        outputs_per_frame[frame_idx] = {
            'masks': mask_tensor,
            'object_ids': torch.tensor(detection_result.object_ids if detection_result.object_ids else [0]),
            'scores': torch.tensor([detection_result.scores.get(frame_idx, 1.0)]),
        }

    # Select best frame for reconstruction
    best_frame = detection_result.best_frame_idx
    if best_frame is not None and best_frame in outputs_per_frame:
        frames_to_process = [best_frame]
    else:
        frames_to_process = sorted(outputs_per_frame.keys())[::stride][:1]

    # Import reconstruction helpers
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "sam-3d-objects", "sam3d_objects"))
    from track_objects import load_camera_pose, apply_pose_to_gaussian, transform_gaussian_to_world

    # Reconstruct objects
    for frame_idx in frames_to_process:
        results = outputs_per_frame[frame_idx]
        masks = results['masks']
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        image_pil = video_frames[frame_idx]
        image_np = np.array(image_pil)

        for obj_idx, mask_tensor in enumerate(masks):
            if mask_tensor.sum() > 0:
                mask_np = mask_tensor.cpu().numpy().astype(bool)

                # Map list-position frame_idx to original video frame number
                orig_frame = frame_index_map[frame_idx] if frame_index_map else frame_idx
                print(f"  Reconstructing Object {obj_idx} frame {orig_frame}")
                output = inference_model(image_np, mask_np, seed=42)

                ttt3r_idx = orig_frame // stride
                camera_to_world, cam_intrinsics = load_camera_pose(ttt3r_idx, camera_dir)

                obj_dir = os.path.join(output_dir, f"object_{obj_idx}")
                os.makedirs(obj_dir, exist_ok=True)

                gs_camera = apply_pose_to_gaussian(
                    output['gs'], output['rotation'], output['translation'],
                    output['scale'], flip_facing=False,
                )
                gs_world = transform_gaussian_to_world(gs_camera, camera_to_world)
                ply_path = os.path.join(obj_dir, f"frame_{orig_frame:04d}.ply")
                gs_world.save_ply(ply_path)

                mask_path = os.path.join(obj_dir, f"frame_{orig_frame:04d}_mask.npy")
                np.save(mask_path, mask_np)

                print(f"  Saved: {ply_path}")


def _run_combine_single(ttt3r_out, sam3d_out, output_dir, frame_interval,
                        annotation_path, skip_visualizations):
    """Run the combine step as a subprocess (it's fast, not worth deep integration)."""

    ply_out = os.path.join(output_dir, "depth")
    os.makedirs(ply_out, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "generate_combined_ply.py"),
        "--ttt3r_out", ttt3r_out,
        "--output_dir", ply_out,
        "--stride", str(frame_interval),
    ]

    if sam3d_out is not None:
        cmd.extend(["--object_dir", sam3d_out])
    if annotation_path:
        cmd.extend(["--vitra_annotation", os.path.abspath(annotation_path)])
    if skip_visualizations:
        cmd.append("--skip_visualizations")

    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _prepare_for_training(manifest, manifest_path):
    """Post-pipeline step: make outputs training-ready.

    1. Create depth symlinks: <video_dir>/depth -> intermediate_depth/depth
    2. Build filtered episode_frame_index.npz for these videos

    Annotations are VITRA originals (already in episodic_annotations/).
    Only depth maps need to be made accessible to training.
    """
    import numpy as np

    results_dir = os.path.dirname(os.path.abspath(manifest_path))
    vitra3d_root = os.path.join(PROJECT_ROOT, "VITRA3D", "data_root")

    print(f"\n{'='*60}")
    print("Preparing outputs for training...")
    print(f"{'='*60}")

    # 1. Depth symlinks
    n_links = 0
    for entry in manifest:
        video_dir = entry["output_dir"]
        link = os.path.join(video_dir, "depth")
        target = os.path.join("intermediate_depth", "depth")
        if not os.path.exists(link):
            os.symlink(target, link)
            n_links += 1
    print(f"  Depth symlinks: {n_links} created")

    # 2. Build filtered episode_frame_index
    # Determine dataset and collect annotation basenames
    ann_basenames = set()
    dataset = None
    for entry in manifest:
        ann_path = entry.get("annotation_path")
        if not ann_path:
            continue
        bn = os.path.splitext(os.path.basename(ann_path))[0]
        ann_basenames.add(bn)
        if dataset is None:
            fname = os.path.basename(ann_path)
            for prefix, ds in [
                ("somethingsomethingv2_", "ssv2"),
                ("epic_kitchens_", "epic"),
                ("EgoExo4D_", "egoexo4d"),
                ("Ego4D_", "ego4d"),
                ("adhitya_collected_data_", "adhitya_collected_data"),
                ("dustpan_collected_data_", "dustpan_collected_data"),
            ]:
                if fname.startswith(prefix):
                    dataset = ds
                    break
            if dataset is None:
                dataset = "ssv2"

    if not ann_basenames:
        print("  WARNING: No annotations in manifest, skipping index build")
        return

    full_index_path = os.path.join(vitra3d_root, "Annotation", dataset,
                                   "episode_frame_index.npz")
    if not os.path.exists(full_index_path):
        print(f"  WARNING: {full_index_path} not found, skipping index build")
        return

    idx = np.load(full_index_path, allow_pickle=True)
    all_eps = list(idx['index_to_episode_id'])
    all_pairs = idx['index_frame_pair']

    # Only include episodes that match ep_000000 for the videos we processed
    # (secondary episodes like ep_000001 won't have depth coverage)
    keep_idx = [i for i, eid in enumerate(all_eps) if eid in ann_basenames]
    keep_set = set(keep_idx)

    mask = np.array([int(p[0]) in keep_set for p in all_pairs])
    old_to_new = {old: new for new, old in enumerate(keep_idx)}
    new_pairs = np.array([[old_to_new[int(p[0])], p[1]]
                          for p in all_pairs[mask]], dtype=np.uint32)
    new_eps = np.array([all_eps[i] for i in keep_idx])

    batch_name = os.path.basename(results_dir)
    index_path = os.path.join(vitra3d_root, "Annotation", dataset,
                              f"episode_frame_index_{batch_name}.npz")
    np.savez(index_path, index_frame_pair=new_pairs,
             index_to_episode_id=new_eps)
    print(f"  Episode index: {len(new_eps)} episodes, {len(new_pairs)} frames")
    print(f"  Saved: {index_path}")

    print(f"\n  To train, set in your config:")
    print(f'    "precomputed_depth_root": "{results_dir}"')
    print(f'    "annotation_override": "Annotation/{dataset}/'
          f'episode_frame_index_{batch_name}.npz"')


def main():
    parser = argparse.ArgumentParser(
        description="Persistent batch worker: load models once, process multiple videos",
    )
    parser.add_argument("--mode", choices=["ttt3r", "pi3", "sam3d"], required=True,
                        help="Worker mode: 'ttt3r'/'pi3' for Step 1, 'sam3d' for Steps 2-4")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to JSON manifest file listing videos to process")

    # TTT3R options
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "TTT3R", "src", "cut3r_512_dpt_4_64.pth"),
                        help="TTT3R model checkpoint path")
    parser.add_argument("--img_size", type=int, default=512)

    # Pi3 options
    parser.add_argument("--pi3_ckpt", type=str, default=None,
                        help="Pi3X checkpoint path (default: auto-download)")
    # Common options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--skip_scene_seg", action="store_true")
    parser.add_argument("--skip_sam3d", action="store_true")
    parser.add_argument("--skip_combine", action="store_true")
    parser.add_argument("--skip_visualizations", action="store_true")
    parser.add_argument("--intrinsics_root", type=str, default=None,
                        help="Root dir for intrinsics (ego4d/*.npy, egoexo4d/*.json)")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to write results JSON (default: {manifest_dir}/batch_{mode}_results.json)")

    # Multi-GPU
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use. Splits manifest and launches "
                             "one worker per GPU as subprocesses.")
    parser.add_argument("--make_training_ready", action="store_true",
                        help="After processing, create depth symlinks and build "
                             "filtered episode_frame_index for training.")

    args = parser.parse_args()

    # Multi-GPU: split manifest and fork
    if args.num_gpus > 1:
        _run_multi_gpu(args)
    else:
        _run_single(args)


def _run_single(args):
    """Run a single worker on the current GPU."""
    import subprocess as _sp

    with open(args.manifest) as f:
        manifest = json.load(f)

    print(f"Batch worker mode: {args.mode}")
    print(f"Videos to process: {len(manifest)}")
    print(f"Device: {args.device}")

    t0 = time.time()

    if args.mode == "ttt3r":
        results = run_ttt3r_batch(
            manifest, args.model_path, args.img_size, args.device, args.frame_interval,
            intrinsics_root=args.intrinsics_root,
        )
    elif args.mode == "pi3":
        results = run_pi3_batch(
            manifest, args.device, args.frame_interval,
            pi3_ckpt=args.pi3_ckpt,
            intrinsics_root=args.intrinsics_root,
        )
    else:
        results = run_sam3d_batch(
            manifest, args.device, args.frame_interval,
            skip_scene_seg=args.skip_scene_seg,
            skip_sam3d=args.skip_sam3d,
            skip_combine=args.skip_combine,
            skip_visualizations=args.skip_visualizations,
            intrinsics_root=args.intrinsics_root,
        )

    total = time.time() - t0
    n_ok = sum(1 for r in results if r.get("success"))

    print(f"\n{'='*60}")
    print(f"Batch worker complete: {n_ok}/{len(results)} succeeded in {total:.1f}s")
    print(f"{'='*60}")

    # Save results
    if args.results_file:
        results_path = args.results_file
    else:
        results_path = os.path.join(os.path.dirname(args.manifest), f"batch_{args.mode}_results.json")
    with open(results_path, "w") as f:
        json.dump({"results": results, "total_s": round(total, 1), "n_ok": n_ok}, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Post-processing: prepare outputs for training
    if args.make_training_ready and args.mode == "pi3":
        _prepare_for_training(manifest, args.manifest)


def _run_multi_gpu(args):
    """Split manifest across GPUs and launch one subprocess per GPU."""
    import subprocess as _sp
    import tempfile

    with open(args.manifest) as f:
        manifest = json.load(f)

    num_gpus = args.num_gpus
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))

    print(f"Multi-GPU mode: {num_gpus} GPUs, {len(manifest)} videos")
    print(f"  ~{len(manifest) // num_gpus} videos per GPU\n")

    # Write per-GPU shard manifests
    shard_files = []
    for gpu_id in range(num_gpus):
        shard = manifest[gpu_id::num_gpus]
        if not shard:
            continue
        shard_path = os.path.join(manifest_dir,
                                  f".shard_gpu{gpu_id}_{os.path.basename(args.manifest)}")
        with open(shard_path, "w") as f:
            json.dump(shard, f)
        shard_files.append((gpu_id, shard_path, len(shard)))

    # Build base command (pass through all args except --num_gpus and --manifest)
    base_cmd = [sys.executable, os.path.abspath(__file__)]
    base_cmd += ["--mode", args.mode, "--num_gpus", "1"]
    if args.pi3_ckpt:
        base_cmd += ["--pi3_ckpt", args.pi3_ckpt]
    if args.frame_interval != 1:
        base_cmd += ["--frame_interval", str(args.frame_interval)]
    if args.intrinsics_root:
        base_cmd += ["--intrinsics_root", args.intrinsics_root]
    if args.skip_scene_seg:
        base_cmd.append("--skip_scene_seg")
    if args.skip_sam3d:
        base_cmd.append("--skip_sam3d")
    if args.skip_combine:
        base_cmd.append("--skip_combine")
    if args.skip_visualizations:
        base_cmd.append("--skip_visualizations")
    if args.mode == "ttt3r":
        base_cmd += ["--model_path", args.model_path, "--img_size", str(args.img_size)]

    # Launch subprocesses
    procs = []
    for gpu_id, shard_path, n_videos in shard_files:
        results_file = os.path.join(
            manifest_dir, f"batch_{args.mode}_gpu{gpu_id}_results.json")
        cmd = base_cmd + [
            "--manifest", shard_path,
            "--device", "cuda",
            "--results_file", results_file,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_path = os.path.join(manifest_dir, f"gpu{gpu_id}.log")
        log_file = open(log_path, "w")

        print(f"  GPU {gpu_id}: {n_videos} videos, log: {log_path}")
        proc = _sp.Popen(cmd, env=env, stdout=log_file, stderr=_sp.STDOUT)
        procs.append((gpu_id, proc, log_file, shard_path, results_file))

    print(f"\nAll {len(procs)} workers launched. Waiting...\n")

    # Wait for all
    all_ok = True
    for gpu_id, proc, log_file, shard_path, results_file in procs:
        proc.wait()
        log_file.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"  GPU {gpu_id}: {status}")
        if proc.returncode != 0:
            all_ok = False
        # Clean up shard file
        os.unlink(shard_path)

    # Merge results
    all_results = []
    for gpu_id, _, _, _, results_file in procs:
        if os.path.exists(results_file):
            with open(results_file) as f:
                data = json.load(f)
            all_results.extend(data.get("results", []))

    n_ok = sum(1 for r in all_results if r.get("success"))
    print(f"\n{'='*60}")
    print(f"Multi-GPU batch complete: {n_ok}/{len(all_results)} succeeded")
    print(f"{'='*60}")

    # Save merged results
    if args.results_file:
        merged_path = args.results_file
    else:
        merged_path = os.path.join(manifest_dir, f"batch_{args.mode}_results.json")
    with open(merged_path, "w") as f:
        json.dump({"results": all_results, "n_ok": n_ok,
                    "num_gpus": num_gpus}, f, indent=2)
    print(f"Merged results: {merged_path}")

    # Post-processing: prepare outputs for training
    if args.make_training_ready and args.mode == "pi3":
        with open(args.manifest) as f:
            full_manifest = json.load(f)
        _prepare_for_training(full_manifest, args.manifest)


if __name__ == "__main__":
    main()
