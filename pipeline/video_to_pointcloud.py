#!/usr/bin/env python3
"""
Unified Video to Augmented Point Cloud Pipeline

This script takes a video as input and produces augmented point clouds combining:
- TTT3R scene reconstruction (depth, camera poses)
- WiLoR hand pose estimation

Since TTT3R and WiLoR require separate conda environments, this script orchestrates
the pipeline using subprocess calls to each environment.

Usage:
    python video_to_pointcloud.py --video path/to/video.mp4 --output_dir results \\
        --ttt3r_env ttt3r --wilor_env wilor

Example:
    python video_to_pointcloud.py --video my_hand_video.mp4 --output_dir ./output \\
        --ttt3r_env ttt3r --wilor_env wilor --frame_interval 2
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path
import tempfile


def check_conda_env(env_name):
    """Check if a conda environment exists."""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        # Check if env_name appears in the output
        for line in result.stdout.split('\n'):
            if line.strip().startswith(env_name + ' ') or line.strip().startswith(env_name + '\t') or ('/' + env_name) in line:
                return True
        return False
    except subprocess.CalledProcessError:
        return False


def run_ttt3r(video_path, ttt3r_env, output_dir, frame_interval, model_path, img_size, device):
    """
    Run TTT3R inference on the video.

    Returns the path to TTT3R output directory.
    """
    print("\n" + "="*80)
    print("STEP 1/3: Running TTT3R Scene Reconstruction")
    print("="*80)

    ttt3r_out = os.path.join(output_dir, "ttt3r_output")
    os.makedirs(ttt3r_out, exist_ok=True)

    # Build TTT3R command
    cmd = [
        "conda", "run", "-n", ttt3r_env, "--no-capture-output",
        "python", "TTT3R/demo.py",
        "--seq_path", video_path,
        "--output_dir", ttt3r_out,
        "--model_path", model_path,
        "--size", str(img_size),
        "--device", device,
        "--frame_interval", str(frame_interval),
        "--model_update_type", "ttt3r"
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nERROR: TTT3R failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\nTTT3R completed successfully!")
    print(f"Output saved to: {ttt3r_out}")

    return ttt3r_out


def run_wilor(video_path, wilor_env, ttt3r_out, output_dir, rescale_factor, frame_interval, device):
    """
    Run WiLoR hand pose estimation.

    Returns the path to WiLoR output directory.
    """
    print("\n" + "="*80)
    print("STEP 2/3: Running WiLoR Hand Pose Estimation")
    print("="*80)

    wilor_out = os.path.join(output_dir, "wilor_output")
    os.makedirs(wilor_out, exist_ok=True)

    # Extract frames from video to temporary directory if needed
    # WiLoR expects an image folder, so we'll extract frames with proper interval
    color_dir = os.path.join(ttt3r_out, "color")

    if not os.path.exists(color_dir) or len(os.listdir(color_dir)) == 0:
        print(f"ERROR: TTT3R color output not found at {color_dir}")
        sys.exit(1)

    # Get absolute paths since we'll be running from WiLoR directory
    abs_color_dir = os.path.abspath(color_dir)
    abs_wilor_out = os.path.abspath(wilor_out)
    abs_camera_path = os.path.abspath(os.path.join(ttt3r_out, "camera"))

    # Build WiLoR command - use the color images from TTT3R
    # Run from WiLoR directory so relative paths to pretrained_models work
    cmd = [
        "conda", "run", "-n", wilor_env, "--no-capture-output",
        "python", "demo.py",
        "--img_folder", abs_color_dir,
        "--out_folder", abs_wilor_out,
        "--save_mesh",
        "--rescale_factor", str(rescale_factor),
        "--ttt3r_path", abs_camera_path,
        "--file_type", "*.png", "*.jpg"
    ]

    print(f"Running from WiLoR directory: {' '.join(cmd)}")
    print()

    # Run from WiLoR directory
    result = subprocess.run(cmd, check=False, cwd="WiLoR")

    if result.returncode != 0:
        print(f"\nWARNING: WiLoR exited with return code {result.returncode}")
        print("This may be normal if no hands were detected in some frames.")

    print(f"\nWiLoR completed!")
    print(f"Output saved to: {wilor_out}")

    return wilor_out


def run_sam3d(video_path, sam3d_env, ttt3r_out, output_dir, prompt, stride, device):
    """
    Run SAM3D object tracking and reconstruction.

    Returns the path to SAM3D output directory.
    """
    print("\n" + "="*80)
    print("STEP 2.5/3: Running SAM3D Object Tracking")
    print("="*80)

    sam3d_out = os.path.join(output_dir, "sam3d_output/object_0")
    os.makedirs(sam3d_out, exist_ok=True)
    
    camera_dir = os.path.join(ttt3r_out, "camera")
    if not os.path.exists(camera_dir):
        print(f"ERROR: TTT3R camera output not found at {camera_dir}")
        sys.exit(1)

    # Use absolute paths since we'll change CWD
    abs_video_path = os.path.abspath(video_path)
    abs_sam3d_out = os.path.abspath(sam3d_out)
    abs_camera_dir = os.path.abspath(camera_dir)

    # Build SAM3D command
    # calls sam3d_objects/track_objects.py (relative to sam-3d-objects root)
    script_path = "sam3d_objects/track_objects.py"
    
    cmd = [
        "conda", "run", "-n", sam3d_env, "--no-capture-output",
        "python", script_path,
        "--video", abs_video_path,
        "--output_dir", abs_sam3d_out,
        "--camera_dir", abs_camera_dir,
        "--prompt", prompt,
        "--stride", str(stride)
    ]

    print(f"Running from sam-3d-objects directory: {' '.join(cmd)}")
    print()

    # Run from sam-3d-objects directory so it can find 'notebook' module and 'checkpoints'
    result = subprocess.run(cmd, check=False, cwd="sam-3d-objects")

    if result.returncode != 0:
        print(f"\nWARNING: SAM3D exited with return code {result.returncode}")
        print("This might happen if no objects matching the prompt were found.")
    
    print(f"\nSAM3D completed!")
    print(f"Output saved to: {sam3d_out}")

    return sam3d_out


def combine_outputs(ttt3r_out, wilor_out, output_dir, combine_env, object_dir=None, contrast=False, stride=1, no_object_transform=False, render_replaced_depth=False):
    """
    Combine TTT3R and WiLoR outputs into PLY files.
    """
    print("\n" + "="*80)
    print("STEP 3/3: Combining Outputs into Point Cloud")
    print("="*80)

    ply_out = os.path.join(output_dir, "ply_output")
    os.makedirs(ply_out, exist_ok=True)

    # Build generation command
    cmd = [
        "python", "pipeline/generate_combined_ply.py",
        "--ttt3r_out", ttt3r_out,
        "--wilor_out", wilor_out,
        "--output_dir", ply_out,
        "--stride", str(stride)
    ]

    if object_dir:
        cmd.extend(["--object_dir", object_dir])
    
    if contrast:
        cmd.append("--contrast")

    if no_object_transform:
        cmd.append("--no_object_transform")

    if render_replaced_depth:
        cmd.append("--render_replaced_depth")

    # If specific environment is requested
    if combine_env:
        cmd = ["conda", "run", "-n", combine_env, "--no-capture-output"] + cmd
    else:
        cmd = cmd

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nERROR: Combining outputs failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\nCombining completed successfully!")
    print(f"Final point clouds saved to: {ply_out}")

    return ply_out


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified pipeline: Video → TTT3R → WiLoR → Augmented Point Cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python video_to_pointcloud.py --video input.mp4 --output_dir results

  # With custom environments and frame interval
  python video_to_pointcloud.py --video input.mp4 --output_dir results \\
      --ttt3r_env my_ttt3r --wilor_env my_wilor --frame_interval 5

  # Process every 10th frame with custom model
  python video_to_pointcloud.py --video long_video.mp4 --output_dir results \\
      --frame_interval 10 --ttt3r_model TTT3R/src/custom_model.pth
        """
    )

    # Required arguments
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pipeline_output",
        help="Directory to save all outputs (default: ./pipeline_output)"
    )

    # Conda environment arguments
    parser.add_argument(
        "--ttt3r_env",
        type=str,
        default="ttt3r",
        help="Name of conda environment for TTT3R (default: ttt3r)"
    )
    parser.add_argument(
        "--wilor_env",
        type=str,
        default="wilor",
        help="Name of conda environment for WiLoR (default: wilor)"
    )
    parser.add_argument(
        "--combine_env",
        type=str,
        default=None,
        help="Conda environment for combining outputs (default: None, uses current environment)"
    )
    parser.add_argument(
        "--sam3d_env",
        type=str,
        default="sam3d-objects",
        help="Name of conda environment for SAM3D (default: sam3d-objects)"
    )

    # TTT3R-specific arguments
    parser.add_argument(
        "--ttt3r_model",
        type=str,
        default="TTT3R/src/cut3r_512_dpt_4_64.pth",
        help="Path to TTT3R model checkpoint (default: TTT3R/src/cut3r_512_dpt_4_64.pth)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Input image size for TTT3R (default: 512)"
    )

    # WiLoR-specific arguments
    parser.add_argument(
        "--rescale_factor",
        type=float,
        default=2.0,
        help="Bounding box rescale factor for WiLoR (default: 2.0)"
    )

    # SAM3D-specific arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="object held in either hand",
        help="Text prompt for SAM3D object tracking"
    )
    # Stride is now controlled by frame_interval effectively, 
    # but we keep this for specific override if needed, though simpler to unify.
    # To simplify usage as requested by user, we will just use frame_interval.
    # But to avoid breaking existing calls or separate control, we can keep it 
    # but strictly prefer frame_interval in logic if we want enforcement.
    # However, user asked for enforcement. Let's remove separate stride or make it alias.
    # We will use frame_interval for everything.
    
    # Common arguments
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Process every Nth frame (applies to TTT3R, WiLoR, and SAM3D) (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu' (default: cuda)"
    )

    # Pipeline control
    parser.add_argument(
        "--skip_ttt3r",
        action="store_true",
        help="Skip TTT3R step (use existing output)"
    )
    parser.add_argument(
        "--skip_wilor",
        action="store_true",
        help="Skip WiLoR step (use existing output)"
    )
    parser.add_argument(
        "--skip_combine",
        action="store_true",
        help="Skip combining step (only run TTT3R and WiLoR)"
    )
    parser.add_argument(
        "--skip_sam3d",
        action="store_true",
        help="Skip SAM3D object tracking step"
    )

    parser.add_argument(
        "--contrast",
        action="store_true",
        help="Use bright purple color for objects instead of natural colors"
    )
    parser.add_argument(
        "--no_object_transform",
        action="store_true",
        help="Skip object transformation in combine step - use raw SAM3D world coordinates"
    )
    parser.add_argument(
        "--render_replaced_depth",
        action="store_true",
        help="Excise original object from scene, insert reconstruction, and render depth image"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("Video to Augmented Point Cloud Pipeline")
    print("="*80)
    print(f"Input video: {args.video}")
    print(f"Output directory: {args.output_dir}")
    print(f"Frame interval: {args.frame_interval}")
    print(f"Device: {args.device}")
    print()

    # Validate input video
    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check conda environments
    if not args.skip_ttt3r:
        print(f"Checking TTT3R conda environment: {args.ttt3r_env}...", end=" ")
        if not check_conda_env(args.ttt3r_env):
            print(f"NOT FOUND")
            print(f"\nERROR: Conda environment '{args.ttt3r_env}' does not exist.")
            print(f"Please create it first or specify the correct name with --ttt3r_env")
            sys.exit(1)
        print("OK")

    if not args.skip_wilor:
        print(f"Checking WiLoR conda environment: {args.wilor_env}...", end=" ")
        if not check_conda_env(args.wilor_env):
            print(f"NOT FOUND")
            print(f"\nERROR: Conda environment '{args.wilor_env}' does not exist.")
            print(f"Please create it first or specify the correct name with --wilor_env")
            sys.exit(1)
        print("OK")

    if args.combine_env:
        print(f"Checking combine conda environment: {args.combine_env}...", end=" ")
        if not check_conda_env(args.combine_env):
            print(f"NOT FOUND")
            print(f"\nERROR: Conda environment '{args.combine_env}' does not exist.")
            sys.exit(1)
        print("OK")
    
    if not args.skip_sam3d:
        print(f"Checking SAM3D conda environment: {args.sam3d_env}...", end=" ")
        if not check_conda_env(args.sam3d_env):
            print(f"NOT FOUND")
            print(f"\nERROR: Conda environment '{args.sam3d_env}' does not exist.")
            print(f"Please create it first or specify the correct name with --sam3d_env")
            sys.exit(1)
        print("OK")

    print()

    # Step 1: Run TTT3R
    if args.skip_ttt3r:
        ttt3r_out = os.path.join(args.output_dir, "ttt3r_output")
        print(f"Skipping TTT3R (using existing output: {ttt3r_out})")
        if not os.path.exists(ttt3r_out):
            print(f"ERROR: TTT3R output directory not found: {ttt3r_out}")
            sys.exit(1)
    else:
        ttt3r_out = run_ttt3r(
            args.video,
            args.ttt3r_env,
            args.output_dir,
            args.frame_interval,
            args.ttt3r_model,
            args.img_size,
            args.device
        )

    # Step 2: Run WiLoR
    if args.skip_wilor:
        wilor_out = os.path.join(args.output_dir, "wilor_output")
        print(f"Skipping WiLoR (using existing output: {wilor_out})")
        if not os.path.exists(wilor_out):
            print(f"ERROR: WiLoR output directory not found: {wilor_out}")
            sys.exit(1)
    else:
        wilor_out = run_wilor(
            args.video,
            args.wilor_env,
            ttt3r_out,
            args.output_dir,
            args.rescale_factor,
            args.frame_interval,
            args.device
        )

    # Step 2.5: Run SAM3D
    if args.skip_sam3d:
        sam3d_out = os.path.join(args.output_dir, "sam3d_output/object_0")
        print(f"Skipping SAM3D (using existing output if available: {sam3d_out})")
        if not os.path.exists(sam3d_out) and not args.skip_combine:
             print("Note: SAM3D output not found, combined output will lack objects.")
             sam3d_out = None
    else:
        sam3d_out = run_sam3d(
            args.video,
            args.sam3d_env,
            ttt3r_out,
            args.output_dir,
            args.prompt,
            args.frame_interval, # Use frame_interval as stride for SAM3D
            args.device
        )

    # Step 3: Combine outputs
    if args.skip_combine:
        print("\nSkipping combine step.")
        ply_out = None
    else:
        ply_out = combine_outputs(
            ttt3r_out,
            wilor_out,
            args.output_dir,
            args.combine_env,
            object_dir=sam3d_out,
            contrast=args.contrast,
            stride=args.frame_interval,
            no_object_transform=args.no_object_transform,
            render_replaced_depth=args.render_replaced_depth
        )

    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"TTT3R output: {ttt3r_out}")
    print(f"WiLoR output: {wilor_out}")
    if sam3d_out:
        print(f"SAM3D output: {sam3d_out}")
    if ply_out:
        print(f"Combined point clouds: {ply_out}")
        print(f"\nYou can view the PLY files in any point cloud viewer:")
        print(f"  - MeshLab: https://www.meshlab.net/")
        print(f"  - CloudCompare: https://www.cloudcompare.org/")
        print(f"  - Open3D: pip install open3d && python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('{ply_out}/000000.ply')])\"")
    print()


if __name__ == "__main__":
    main()
