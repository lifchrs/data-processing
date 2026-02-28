#!/usr/bin/env python3
"""
Persistent batch worker for the unified pipeline.

Processes multiple videos from a manifest file, keeping models loaded between
videos to avoid repeated model initialization overhead (~30-60s per video).

Two modes:
  - TTT3R mode:  Loads TTT3R model once, processes all videos for Step 1.
  - SAM3D mode:  Loads SAM3 + Inference models once, processes Steps 2-4 for all videos.

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
            "frames_dir": "/path/to/frames",                // optional, for SAM3D
            "camera_dir": "/path/to/cameras",               // required for SAM3D
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
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# ---------------------------------------------------------------------------
# TTT3R Batch Worker
# ---------------------------------------------------------------------------

def run_ttt3r_batch(manifest, model_path, img_size, device, frame_interval):
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
        ttt3r_out = os.path.join(output_dir, "ttt3r_output")
        os.makedirs(ttt3r_out, exist_ok=True)

        print(f"\n[{i+1}/{len(manifest)}] Processing: {video_path}")
        t_start = time.time()

        try:
            import shutil

            # Prepare input frames
            img_paths, tmpdirname = parse_seq_path(video_path, frame_interval)
            if not img_paths:
                print(f"  No images found, skipping")
                results.append({"video": video_path, "success": False, "error": "no_frames"})
                continue

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
# SAM3D Batch Worker (Steps 2-4: Scene Seg + Object Detection + Combine)
# ---------------------------------------------------------------------------

def run_sam3d_batch(manifest, device, frame_interval, skip_scene_seg=False,
                    skip_sam3d=False, skip_combine=False, skip_visualizations=False):
    """Load SAM3 models once and process all videos for steps 2-4."""
    import torch
    import numpy as np
    from PIL import Image

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
        frames_dir = entry.get("frames_dir")
        camera_dir = entry.get("camera_dir")
        prompt = entry.get("prompt", "object held in either hand")

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(manifest)}] Processing: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        t_start = time.time()

        try:
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

            # --- Step 4: Combine outputs ---
            if not skip_combine:
                ttt3r_out = os.path.join(output_dir, "ttt3r_output")
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
    if frames_dir is not None:
        import glob as _glob
        frame_paths = sorted(_glob.glob(os.path.join(frames_dir, "*.png"))
                             + _glob.glob(os.path.join(frames_dir, "*.jpg")))
        video_frames = [Image.open(p).convert("RGB") for p in frame_paths]
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
    from track_objects import load_camera_pose, apply_pose_to_gaussian, gaussian_to_world

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

                print(f"  Reconstructing Object {obj_idx} frame {frame_idx}")
                output = inference_model(image_np, mask_np, seed=42)

                ttt3r_idx = frame_idx // stride
                camera_to_world, cam_intrinsics = load_camera_pose(ttt3r_idx, camera_dir)

                obj_dir = os.path.join(output_dir, f"object_{obj_idx}")
                os.makedirs(obj_dir, exist_ok=True)

                from track_objects import apply_pose_to_gaussian, gaussian_to_world, save_gaussian_splat
                gs_camera = apply_pose_to_gaussian(
                    output['gs'], output['rotation'], output['translation'],
                    output['scale'], flip_facing=False,
                )
                gs_world = gaussian_to_world(gs_camera, camera_to_world)
                ply_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}.ply")
                save_gaussian_splat(gs_world, ply_path)

                mask_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}_mask.npy")
                np.save(mask_path, mask_np)

                print(f"  Saved: {ply_path}")


def _run_combine_single(ttt3r_out, sam3d_out, output_dir, frame_interval,
                        annotation_path, skip_visualizations):
    """Run the combine step as a subprocess (it's fast, not worth deep integration)."""
    import subprocess

    ply_out = os.path.join(output_dir, "ply_output")
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

def main():
    parser = argparse.ArgumentParser(
        description="Persistent batch worker: load models once, process multiple videos",
    )
    parser.add_argument("--mode", choices=["ttt3r", "sam3d"], required=True,
                        help="Worker mode: 'ttt3r' for Step 1, 'sam3d' for Steps 2-4")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to JSON manifest file listing videos to process")

    # TTT3R options
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "TTT3R", "src", "cut3r_512_dpt_4_64.pth"),
                        help="TTT3R model checkpoint path")
    parser.add_argument("--img_size", type=int, default=512)

    # Common options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--skip_scene_seg", action="store_true")
    parser.add_argument("--skip_sam3d", action="store_true")
    parser.add_argument("--skip_combine", action="store_true")
    parser.add_argument("--skip_visualizations", action="store_true")

    args = parser.parse_args()

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    print(f"Batch worker mode: {args.mode}")
    print(f"Videos to process: {len(manifest)}")
    print(f"Device: {args.device}")

    t0 = time.time()

    if args.mode == "ttt3r":
        results = run_ttt3r_batch(
            manifest, args.model_path, args.img_size, args.device, args.frame_interval,
        )
    else:
        results = run_sam3d_batch(
            manifest, args.device, args.frame_interval,
            skip_scene_seg=args.skip_scene_seg,
            skip_sam3d=args.skip_sam3d,
            skip_combine=args.skip_combine,
            skip_visualizations=args.skip_visualizations,
        )

    total = time.time() - t0
    n_ok = sum(1 for r in results if r.get("success"))

    print(f"\n{'='*60}")
    print(f"Batch worker complete: {n_ok}/{len(results)} succeeded in {total:.1f}s")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(os.path.dirname(args.manifest), f"batch_{args.mode}_results.json")
    with open(results_path, "w") as f:
        json.dump({"results": results, "total_s": round(total, 1), "n_ok": n_ok}, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
