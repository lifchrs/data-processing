#!/usr/bin/env python3
"""
MegaSAM Pipeline Wrapper

This script runs the MegaSAM pipeline and converts its output to TTT3R format
for compatibility with the downstream pipeline stages.

MegaSAM Pipeline Stages:
1. Frame extraction from video
2. Mono-depth estimation (DepthAnything)
3. Metric-depth estimation (UniDepth)
4. Camera tracking (DROID-SLAM based)
5. CVD optimization (optional, for temporal consistency)
6. Format conversion to TTT3R structure

Usage:
    python run_megasam.py --video input.mp4 --output_dir output/ --device cuda
"""

import argparse
import os
import sys
import glob
import subprocess
import shutil
import cv2
import numpy as np
from pathlib import Path


def extract_frames(video_path, output_dir, frame_interval=1):
    """
    Extract frames from video to a directory.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame

    Returns:
        Path to directory containing extracted frames
    """
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_path = os.path.join(frames_dir, f"{saved_idx:06d}.png")
            cv2.imwrite(frame_path, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved_idx} frames to {frames_dir}")

    return frames_dir


def run_mono_depth(frames_dir, output_dir, megasam_dir):
    """
    Run DepthAnything mono-depth estimation.

    Args:
        frames_dir: Directory containing input frames
        output_dir: Base output directory
        megasam_dir: Path to mega-sam repository
    """
    print("\n--- Stage 1: Running DepthAnything (mono-depth) ---")

    scene_name = "scene"
    mono_out_dir = os.path.join(megasam_dir, "Depth-Anything", "video_visualization", scene_name)
    # Clear stale outputs from previous runs
    if os.path.exists(mono_out_dir):
        import shutil
        shutil.rmtree(mono_out_dir)
    os.makedirs(mono_out_dir, exist_ok=True)

    checkpoint_path = os.path.join(megasam_dir, "Depth-Anything", "checkpoints", "depth_anything_vitl14.pth")

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            f"DepthAnything checkpoint not found at {checkpoint_path}. "
            "Please download it from: https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"
        )

    cmd = [
        "python", os.path.join(megasam_dir, "Depth-Anything", "run_videos.py"),
        "--encoder", "vitl",
        "--load-from", checkpoint_path,
        "--img-path", frames_dir,
        "--outdir", mono_out_dir
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=megasam_dir, check=True)

    print(f"Mono-depth output saved to: {mono_out_dir}")
    return mono_out_dir


def run_unidepth(frames_dir, output_dir, megasam_dir):
    """
    Run UniDepth metric-depth estimation.

    Args:
        frames_dir: Directory containing input frames
        output_dir: Base output directory
        megasam_dir: Path to mega-sam repository
    """
    print("\n--- Stage 2: Running UniDepth (metric-depth) ---")

    scene_name = "scene"
    unidepth_out_dir = os.path.join(megasam_dir, "UniDepth", "outputs")
    # Clear stale outputs from previous runs
    scene_out = os.path.join(unidepth_out_dir, scene_name)
    if os.path.exists(scene_out):
        import shutil
        shutil.rmtree(scene_out)
    os.makedirs(unidepth_out_dir, exist_ok=True)

    # Set PYTHONPATH to include UniDepth
    env = os.environ.copy()
    unidepth_path = os.path.join(megasam_dir, "UniDepth")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{unidepth_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = unidepth_path

    cmd = [
        "python", os.path.join(megasam_dir, "UniDepth", "scripts", "demo_mega-sam.py"),
        "--scene-name", scene_name,
        "--img-path", frames_dir,
        "--outdir", unidepth_out_dir
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=megasam_dir, env=env, check=True)

    print(f"Metric-depth output saved to: {unidepth_out_dir}")
    return unidepth_out_dir


def run_camera_tracking(frames_dir, output_dir, megasam_dir, device="cuda"):
    """
    Run DROID-SLAM based camera tracking.

    Args:
        frames_dir: Directory containing input frames
        output_dir: Base output directory
        megasam_dir: Path to mega-sam repository
        device: CUDA device to use
    """
    print("\n--- Stage 3: Running Camera Tracking (DROID-SLAM) ---")

    scene_name = "scene"

    # Check for checkpoint
    checkpoint_path = os.path.join(megasam_dir, "checkpoints", "megasam_final.pth")
    if not os.path.exists(checkpoint_path):
        # Try alternative checkpoint name
        checkpoint_path = os.path.join(megasam_dir, "checkpoints", "droid.pth")

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            f"MegaSAM checkpoint not found in {os.path.join(megasam_dir, 'checkpoints')}. "
            "Please download it according to MegaSAM documentation."
        )

    mono_depth_path = os.path.join(megasam_dir, "Depth-Anything", "video_visualization")
    metric_depth_path = os.path.join(megasam_dir, "UniDepth", "outputs")

    cmd = [
        "python", os.path.join(megasam_dir, "camera_tracking_scripts", "test_demo.py"),
        f"--datapath={frames_dir}",
        f"--weights={checkpoint_path}",
        "--scene_name", scene_name,
        "--mono_depth_path", mono_depth_path,
        "--metric_depth_path", metric_depth_path,
        "--disable_vis"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=megasam_dir, check=True)

    # Output is saved to mega-sam/outputs/scene_droid.npz
    droid_out = os.path.join(megasam_dir, "outputs", f"{scene_name}_droid.npz")
    print(f"Camera tracking output saved to: {droid_out}")

    return droid_out


def precompute_optical_flow(megasam_dir, frames_dir):
    """
    Precompute optical flow for CVD optimization.

    Args:
        megasam_dir: Path to mega-sam repository
        frames_dir: Directory containing input frames
    """
    print("\n--- Stage 4a: Precomputing Optical Flow ---")

    scene_name = "scene"
    raft_checkpoint = os.path.join(megasam_dir, "cvd_opt", "raft-things.pth")

    if not os.path.exists(raft_checkpoint):
        print(f"WARNING: RAFT checkpoint not found at {raft_checkpoint}. Skipping CVD optimization.")
        return False

    cmd = [
        "python", os.path.join(megasam_dir, "cvd_opt", "preprocess_flow.py"),
        "--model", raft_checkpoint,
        "--scene_name", scene_name,
        "--datapath", frames_dir
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=megasam_dir, check=False)

    if result.returncode != 0:
        print("WARNING: Optical flow preprocessing failed. Skipping CVD optimization.")
        return False

    return True


def run_cvd_optimization(megasam_dir):
    """
    Run Consistent Video Depth (CVD) optimization.

    Args:
        megasam_dir: Path to mega-sam repository
    """
    print("\n--- Stage 4b: Running CVD Optimization ---")

    scene_name = "scene"
    output_dir = os.path.join(megasam_dir, "outputs_cvd")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", os.path.join(megasam_dir, "cvd_opt", "cvd_opt.py"),
        "--scene_name", scene_name,
        "--output_dir", output_dir
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=megasam_dir, check=False)

    if result.returncode != 0:
        print("WARNING: CVD optimization failed. Using camera tracking output instead.")
        return None

    cvd_out = os.path.join(output_dir, f"{scene_name}_sgd_cvd_hr.npz")
    print(f"CVD optimization output saved to: {cvd_out}")

    return cvd_out


def convert_to_ttt3r_format(megasam_dir, output_dir):
    """
    Convert MegaSAM bundled npz output to TTT3R per-frame format.

    Args:
        megasam_dir: Path to mega-sam repository
        output_dir: Final output directory (TTT3R format)
    """
    print("\n--- Stage 5: Converting to TTT3R Format ---")

    scene_name = "scene"

    # Prefer CVD-optimized output if available
    cvd_path = os.path.join(megasam_dir, "outputs_cvd", f"{scene_name}_sgd_cvd_hr.npz")
    droid_path = os.path.join(megasam_dir, "outputs", f"{scene_name}_droid.npz")

    if os.path.exists(cvd_path):
        npz_path = cvd_path
        print(f"Using CVD-optimized output: {npz_path}")
    elif os.path.exists(droid_path):
        npz_path = droid_path
        print(f"Using camera tracking output: {npz_path}")
    else:
        raise RuntimeError(f"No MegaSAM output found. Expected {cvd_path} or {droid_path}")

    # Load the bundled npz
    data = np.load(npz_path)

    # Create output directories (TTT3R format)
    depth_dir = os.path.join(output_dir, "depth")
    color_dir = os.path.join(output_dir, "color")
    camera_dir = os.path.join(output_dir, "camera")

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(camera_dir, exist_ok=True)

    # Extract data
    images = data['images']  # Shape: (N, H, W, 3), RGB, uint8
    depths = data['depths']  # Shape: (N, H, W), float
    intrinsic = data['intrinsic']  # Shape: (3, 3)
    cam_c2w = data['cam_c2w']  # Shape: (N, 4, 4)

    num_frames = len(images)
    print(f"Converting {num_frames} frames...")

    # Keep intrinsic as 3x3 matrix (TTT3R format)
    intrinsics_mat = intrinsic.astype(np.float32)

    for i in range(num_frames):
        frame_name = f"{i:06d}"

        # Save depth (as float32 npy)
        depth = depths[i].astype(np.float32)
        np.save(os.path.join(depth_dir, f"{frame_name}.npy"), depth)

        # Save color image (RGB -> BGR for cv2)
        image = images[i]
        cv2.imwrite(
            os.path.join(color_dir, f"{frame_name}.png"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

        # Save camera (pose + intrinsics)
        # cam_c2w is camera-to-world transformation
        pose = cam_c2w[i].astype(np.float32)
        np.savez(
            os.path.join(camera_dir, f"{frame_name}.npz"),
            pose=pose,
            intrinsics=intrinsics_mat
        )

    print(f"Converted {num_frames} frames to TTT3R format:")
    print(f"  Depth:  {depth_dir}")
    print(f"  Color:  {color_dir}")
    print(f"  Camera: {camera_dir}")

    return output_dir


def cleanup_intermediate_files(megasam_dir):
    """
    Clean up intermediate files created during MegaSAM processing.

    Args:
        megasam_dir: Path to mega-sam repository
    """
    # Clean up scene-specific files
    scene_name = "scene"

    dirs_to_clean = [
        os.path.join(megasam_dir, "Depth-Anything", "video_visualization", scene_name),
        os.path.join(megasam_dir, "UniDepth", "outputs", scene_name),
        os.path.join(megasam_dir, "reconstructions", scene_name),
        os.path.join(megasam_dir, "cache_flow", scene_name),
    ]

    files_to_clean = [
        os.path.join(megasam_dir, "outputs", f"{scene_name}_droid.npz"),
        os.path.join(megasam_dir, "outputs_cvd", f"{scene_name}_sgd_cvd_hr.npz"),
    ]

    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)

    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f)


def run_megasam(video_path, output_dir, frame_interval=1, device="cuda",
                skip_cvd=False, cleanup=False):
    """
    Run the full MegaSAM pipeline.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save final output
        frame_interval: Process every Nth frame
        device: CUDA device to use
        skip_cvd: Skip CVD optimization step
        cleanup: Clean up intermediate files after processing

    Returns:
        Path to output directory in TTT3R format
    """
    print("="*80)
    print("MegaSAM Pipeline")
    print("="*80)
    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Frame interval: {frame_interval}")
    print(f"Device: {device}")
    print()

    # Get absolute paths
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)

    # Find mega-sam directory (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    megasam_dir = os.path.join(project_root, "mega-sam")

    if not os.path.exists(megasam_dir):
        raise RuntimeError(f"mega-sam directory not found at {megasam_dir}")

    megasam_dir = os.path.abspath(megasam_dir)

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Stage 1: Extract frames
        print("\n--- Stage 0: Extracting Frames ---")
        frames_dir = extract_frames(video_path, output_dir, frame_interval)

        # Stage 2: Run DepthAnything (mono-depth)
        run_mono_depth(frames_dir, output_dir, megasam_dir)

        # Stage 3: Run UniDepth (metric-depth)
        run_unidepth(frames_dir, output_dir, megasam_dir)

        # Stage 4: Run camera tracking
        run_camera_tracking(frames_dir, output_dir, megasam_dir, device)

        # Stage 5: CVD optimization (optional)
        if not skip_cvd:
            flow_ok = precompute_optical_flow(megasam_dir, frames_dir)
            if flow_ok:
                run_cvd_optimization(megasam_dir)

        # Stage 6: Convert to TTT3R format
        convert_to_ttt3r_format(megasam_dir, output_dir)

        # Cleanup if requested
        if cleanup:
            cleanup_intermediate_files(megasam_dir)
            # Also remove extracted frames
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)

        print("\n" + "="*80)
        print("MegaSAM Pipeline Complete!")
        print("="*80)
        print(f"Output saved to: {output_dir}")

        return output_dir

    except Exception as e:
        print(f"\nERROR: MegaSAM pipeline failed: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MegaSAM pipeline and convert output to TTT3R format"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output"
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--skip_cvd",
        action="store_true",
        help="Skip CVD optimization step"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up intermediate files after processing"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_megasam(
        video_path=args.video,
        output_dir=args.output_dir,
        frame_interval=args.frame_interval,
        device=args.device,
        skip_cvd=args.skip_cvd,
        cleanup=args.cleanup
    )
