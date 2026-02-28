#!/usr/bin/env python3
"""
Unified Video Pipeline

End-to-end pipeline: Video + VITRA annotation → Scene reconstruction,
scene segmentation, object detection/reconstruction, and combined point clouds.

Steps:
  1. Scene Reconstruction (TTT3R or MegaSAM) → depth, camera poses
  2. Scene Segmentation (SAM3 auto-mask + tracking) → per-frame label maps
  3. Object Detection + 3D Reconstruction (SAM3D) → per-object PLY + masks
  4. Combine into Point Cloud → final PLY files

Usage:
    python run_pipeline.py --video input.mp4 --output_dir results

    # With VITRA annotation for hand-aware detection:
    python run_pipeline.py --video input.mp4 --output_dir results \\
        --vitra_annotation annotation.npy

    # Using MegaSAM instead of TTT3R:
    python run_pipeline.py --video input.mp4 --output_dir results --use_megasam
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class PipelineLog:
    """Simple structured log accumulator for pipeline diagnostics."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.entries = []

    def log(self, phase: str, message: str, **data):
        """Append a structured entry. Print if verbose."""
        entry = {"phase": phase, "message": message,
                 "timestamp": time.time(), **data}
        self.entries.append(entry)
        if self.verbose:
            print(message)

    def to_dict(self) -> list:
        return self.entries


def resolve_conda_python(env_name):
    """Resolve conda environment name to its Python binary path.

    Returns the absolute path to the env's Python, or None if not found.
    This avoids the ~18s overhead of `conda run` per subprocess call.
    """
    # Try conda info to get the envs directory
    try:
        result = subprocess.run(
            ["conda", "info", "--json"],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(result.stdout)
        envs_dirs = info.get("envs_dirs", [])
        for envs_dir in envs_dirs:
            python_path = os.path.join(envs_dir, env_name, "bin", "python")
            if os.path.isfile(python_path):
                return python_path
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        pass

    # Fallback: check common locations
    for base in [os.path.expanduser("~/anaconda3/envs"),
                 os.path.expanduser("~/miniconda3/envs"),
                 "/opt/conda/envs"]:
        python_path = os.path.join(base, env_name, "bin", "python")
        if os.path.isfile(python_path):
            return python_path

    return None


def check_conda_env(env_name):
    """Check if a conda environment exists."""
    return resolve_conda_python(env_name) is not None


# Cache of resolved Python paths to avoid repeated lookups
_python_cache = {}


def get_python_cmd(env_name):
    """Get the command prefix for running Python in a conda env.

    Returns a list like ['/path/to/envs/env/bin/python'] for direct execution,
    falling back to ['conda', 'run', '-n', env_name, '--no-capture-output', 'python']
    if the direct path can't be resolved.
    """
    if env_name not in _python_cache:
        python_path = resolve_conda_python(env_name)
        if python_path:
            _python_cache[env_name] = [python_path]
        else:
            _python_cache[env_name] = [
                "conda", "run", "-n", env_name, "--no-capture-output", "python"
            ]
    return list(_python_cache[env_name])


def get_env_vars(env_name):
    """Get environment variables needed when bypassing conda run.

    Sets CONDA_PREFIX, CONDA_DEFAULT_ENV, and prepends env bin/ to PATH
    so that packages expecting conda environment variables work correctly.
    """
    python_path = resolve_conda_python(env_name)
    if python_path is None:
        return None  # using conda run, no extra env needed

    env_dir = os.path.dirname(os.path.dirname(python_path))  # .../envs/name
    env = os.environ.copy()
    env["CONDA_PREFIX"] = env_dir
    env["CONDA_DEFAULT_ENV"] = env_name
    # Prepend env's bin to PATH so executables (nvcc etc.) are found
    env["PATH"] = os.path.join(env_dir, "bin") + os.pathsep + env.get("PATH", "")
    # Set CUDA_HOME if the env has nvcc
    nvcc_path = os.path.join(env_dir, "bin", "nvcc")
    if os.path.isfile(nvcc_path):
        env["CUDA_HOME"] = env_dir
    return env


# ---------------------------------------------------------------------------
# Step 1: Scene Reconstruction
# ---------------------------------------------------------------------------

def run_ttt3r(video_path, output_dir, frame_interval, device, ttt3r_env,
              model_path, img_size, log=None):
    """Run TTT3R scene reconstruction."""
    log.log("step1_reconstruction",
            "\n" + "=" * 80 + "\nSTEP 1/4: Running TTT3R Scene Reconstruction\n" + "=" * 80)

    ttt3r_out = os.path.join(output_dir, "ttt3r_output")
    os.makedirs(ttt3r_out, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    cmd = get_python_cmd(ttt3r_env) + [
        os.path.join(project_root, "TTT3R", "demo.py"),
        "--seq_path", video_path,
        "--output_dir", ttt3r_out,
        "--model_path", model_path,
        "--size", str(img_size),
        "--device", device,
        "--frame_interval", str(frame_interval),
        "--model_update_type", "ttt3r",
    ]

    log.log("step1_reconstruction", f"Running: {' '.join(cmd)}\n",
            command=cmd)

    # Use Popen instead of run: TTT3R's demo.py launches a blocking point
    # cloud viewer after saving outputs.  We poll for the expected output
    # files and terminate the process once they appear.
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=get_env_vars(ttt3r_env))

    # Wait for depth outputs to appear (written before the viewer launches)
    depth_dir = os.path.join(ttt3r_out, "depth")
    camera_dir = os.path.join(ttt3r_out, "camera")
    while proc.poll() is None:
        time.sleep(2)
        if (os.path.isdir(depth_dir) and os.listdir(depth_dir)
                and os.path.isdir(camera_dir) and os.listdir(camera_dir)):
            # Give a moment for file writes to flush
            time.sleep(3)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            break

    elapsed = time.time() - t0

    # Check that outputs actually exist (process may have crashed before writing)
    if not os.path.isdir(depth_dir) or not os.listdir(depth_dir):
        log.log("step1_reconstruction",
                f"\nERROR: TTT3R produced no output in {depth_dir}",
                returncode=proc.returncode, elapsed_s=elapsed, success=False)
        sys.exit(1)

    log.log("step1_reconstruction",
            f"\nTTT3R completed successfully!\nOutput saved to: {ttt3r_out}",
            output_dir=ttt3r_out, elapsed_s=elapsed, success=True)
    return ttt3r_out


def run_megasam(video_path, output_dir, frame_interval, device, megasam_env,
                log=None):
    """Run MegaSAM scene reconstruction."""
    log.log("step1_reconstruction",
            "\n" + "=" * 80 + "\nSTEP 1/4: Running MegaSAM Scene Reconstruction\n" + "=" * 80)

    megasam_out = os.path.join(output_dir, "ttt3r_output")
    os.makedirs(megasam_out, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = get_python_cmd(megasam_env) + [
        os.path.join(script_dir, "run_megasam.py"),
        "--video", video_path,
        "--output_dir", megasam_out,
        "--frame_interval", str(frame_interval),
        "--device", device,
    ]

    log.log("step1_reconstruction", f"Running: {' '.join(cmd)}\n",
            command=cmd)

    t0 = time.time()
    result = subprocess.run(cmd, check=False, env=get_env_vars(megasam_env))
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.log("step1_reconstruction",
                f"\nERROR: MegaSAM failed with return code {result.returncode}",
                returncode=result.returncode, elapsed_s=elapsed, success=False)
        sys.exit(1)

    log.log("step1_reconstruction",
            f"\nMegaSAM completed successfully!\nOutput saved to: {megasam_out}",
            output_dir=megasam_out, elapsed_s=elapsed, success=True)
    return megasam_out


# ---------------------------------------------------------------------------
# Step 2: Scene Segmentation
# ---------------------------------------------------------------------------

def run_scene_segmentation(video_path, output_dir, device, sam3d_env,
                           log=None):
    """Run scene segmentation (all visible objects)."""
    log.log("step2_scene_seg",
            "\n" + "=" * 80 + "\nSTEP 2/4: Running Scene Segmentation\n" + "=" * 80)

    seg_out = os.path.join(output_dir, "scene_segmentation")
    os.makedirs(seg_out, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = get_python_cmd(sam3d_env) + [
        os.path.join(script_dir, "run_scene_segmentation.py"),
        "--video", video_path,
        "--output_dir", seg_out,
        "--device", device,
    ]

    log.log("step2_scene_seg", f"Running: {' '.join(cmd)}\n",
            command=cmd)

    t0 = time.time()
    result = subprocess.run(cmd, check=False, env=get_env_vars(sam3d_env))
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.log("step2_scene_seg",
                f"\nERROR: Scene segmentation failed with return code {result.returncode}",
                returncode=result.returncode, elapsed_s=elapsed, success=False)
        sys.exit(1)

    log.log("step2_scene_seg",
            f"\nScene segmentation completed successfully!\nOutput saved to: {seg_out}",
            output_dir=seg_out, elapsed_s=elapsed, success=True)
    return seg_out


# ---------------------------------------------------------------------------
# Step 3: Object Detection + 3D Reconstruction (SAM3D)
# ---------------------------------------------------------------------------

def run_sam3d(video_path, ttt3r_out, output_dir, prompt, frame_interval,
              device, sam3d_env, vitra_annotation=None,
              inference_steps=None, early_exit_score=0.8, log=None):
    """Run SAM3D object tracking and reconstruction."""
    log.log("step3_sam3d",
            "\n" + "=" * 80 + "\nSTEP 3/4: Running SAM3D Object Detection + Reconstruction\n" + "=" * 80)

    sam3d_out = os.path.join(output_dir, "sam3d_output")
    os.makedirs(sam3d_out, exist_ok=True)

    camera_dir = os.path.join(ttt3r_out, "camera")
    if not os.path.exists(camera_dir):
        log.log("step3_sam3d",
                f"ERROR: Camera poses not found at {camera_dir}",
                error="camera_dir_missing")
        sys.exit(1)

    abs_video_path = os.path.abspath(video_path)
    abs_sam3d_out = os.path.abspath(sam3d_out)
    abs_camera_dir = os.path.abspath(camera_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sam3d_root = os.path.join(project_root, "sam-3d-objects")

    # Use pre-extracted TTT3R color frames so SAM3D doesn't need to decode
    # the source video (which may be .webm or other formats load_video can't handle)
    frames_dir = os.path.join(ttt3r_out, "color")
    abs_frames_dir = os.path.abspath(frames_dir)

    cmd = get_python_cmd(sam3d_env) + [
        "sam3d_objects/track_objects.py",
        "--video", abs_video_path,
        "--frames_dir", abs_frames_dir,
        "--output_dir", abs_sam3d_out,
        "--camera_dir", abs_camera_dir,
        "--prompt", prompt,
        "--stride", str(frame_interval),
    ]

    # Add VITRA annotation if provided — enables hand_point detection
    if vitra_annotation is not None:
        abs_vitra = os.path.abspath(vitra_annotation)
        cmd.extend([
            "--vitra_annotation", abs_vitra,
            "--detection_strategy", "hand_point",
        ])

    # Pass optimization flags
    if inference_steps is not None:
        cmd.extend(["--inference_steps", str(inference_steps)])
    if early_exit_score is not None:
        cmd.extend(["--early_exit_score", str(early_exit_score)])

    log.log("step3_sam3d",
            f"Running from sam-3d-objects directory: {' '.join(cmd)}\n",
            command=cmd, cwd=sam3d_root, prompt=prompt,
            vitra_annotation=vitra_annotation)

    t0 = time.time()
    result = subprocess.run(cmd, check=False, cwd=sam3d_root, env=get_env_vars(sam3d_env))
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.log("step3_sam3d",
                f"\nERROR: SAM3D failed with return code {result.returncode}",
                returncode=result.returncode, elapsed_s=elapsed, success=False)
        sys.exit(1)

    log.log("step3_sam3d",
            f"\nSAM3D completed successfully!\nOutput saved to: {sam3d_out}",
            output_dir=sam3d_out, elapsed_s=elapsed, success=True)
    return sam3d_out


# ---------------------------------------------------------------------------
# Step 4: Combine into Point Cloud
# ---------------------------------------------------------------------------

def combine_outputs(ttt3r_out, sam3d_out, output_dir, frame_interval, sam3d_env,
                    vitra_annotation=None, skip_visualizations=False,
                    export_obj=False, log=None):
    """Combine scene reconstruction + object reconstructions into PLY files."""
    log.log("step4_combine",
            "\n" + "=" * 80 + "\nSTEP 4/4: Combining Outputs into Point Cloud\n" + "=" * 80)

    ply_out = os.path.join(output_dir, "ply_output")
    os.makedirs(ply_out, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = get_python_cmd(sam3d_env) + [
        os.path.join(script_dir, "generate_combined_ply.py"),
        "--ttt3r_out", ttt3r_out,
        "--output_dir", ply_out,
        "--stride", str(frame_interval),
    ]

    if sam3d_out is not None:
        cmd.extend(["--object_dir", sam3d_out])

    if vitra_annotation is not None:
        cmd.extend(["--vitra_annotation", os.path.abspath(vitra_annotation)])

    if skip_visualizations:
        cmd.append("--skip_visualizations")

    if export_obj:
        cmd.append("--export_obj")

    log.log("step4_combine", f"Running: {' '.join(cmd)}\n",
            command=cmd)

    t0 = time.time()
    result = subprocess.run(cmd, check=False, env=get_env_vars(sam3d_env))
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.log("step4_combine",
                f"\nERROR: Combining outputs failed with return code {result.returncode}",
                returncode=result.returncode, elapsed_s=elapsed, success=False)
        sys.exit(1)

    log.log("step4_combine",
            f"\nCombining completed successfully!\nFinal point clouds saved to: {ply_out}",
            output_dir=ply_out, elapsed_s=elapsed, success=True)
    return ply_out


# ---------------------------------------------------------------------------
# Quality Filters
# ---------------------------------------------------------------------------

GENERIC_LABELS = {"object held in hand", "object held in either hand", "object", ""}


def check_quality_flags(output_dir, sam3d_out, action_text, episode_id=None,
                        check_deformable=False, sam3d_skipped=False, log=None):
    """Run quality filters and write quality_flags.json.

    Returns the flags dict (also written to disk).
    """
    filters = {}
    metadata = {"action_text": action_text}

    # --- Filter 1: missing SAM detection ---
    if sam3d_skipped:
        filters["missing_sam_detection"] = False  # not applicable
    else:
        has_objects = False
        if sam3d_out and os.path.isdir(sam3d_out):
            for entry in os.listdir(sam3d_out):
                obj_dir = os.path.join(sam3d_out, entry)
                if os.path.isdir(obj_dir) and entry.startswith("object_"):
                    # Check it actually has mask files
                    masks = glob.glob(os.path.join(obj_dir, "**", "*mask*"), recursive=True)
                    if masks:
                        has_objects = True
                        break
        filters["missing_sam_detection"] = not has_objects

    # --- Filter 2: missing / generic label ---
    label_text = (action_text or "").strip().lower()
    filters["missing_label"] = label_text in GENERIC_LABELS or not label_text

    # --- Filter 3: deformable object (LLM) ---
    if check_deformable and not filters["missing_label"]:
        try:
            # Import from sam3d_objects (lives in sam3d conda env, but text_utils
            # has no heavy deps beyond transformers which the env has)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            sys.path.insert(0, os.path.join(project_root, "sam-3d-objects"))
            from sam3d_objects.text_utils import extract_object_noun, check_deformable_llm

            object_noun = extract_object_noun(action_text)
            metadata["object_noun"] = object_noun
            is_deformable = check_deformable_llm(object_noun)
            filters["deformable_object"] = is_deformable
            metadata["deformable_response"] = is_deformable
        except Exception as e:
            if log:
                log.log("quality_check", f"  Deformable check failed: {e}")
            filters["deformable_object"] = False
            metadata["deformable_error"] = str(e)
    else:
        filters["deformable_object"] = False

    # --- Aggregate ---
    skip_reasons = [k for k, v in filters.items() if v]
    skip = len(skip_reasons) > 0

    flags = {
        "episode_id": episode_id,
        "skip": skip,
        "skip_reasons": skip_reasons,
        "filters": filters,
        "metadata": metadata,
    }

    # Write to output dir
    flags_path = os.path.join(output_dir, "quality_flags.json")
    with open(flags_path, "w") as f:
        json.dump(flags, f, indent=2)

    if log:
        status = f"SKIP ({', '.join(skip_reasons)})" if skip else "PASS"
        log.log("quality_check",
                f"\nQuality check: {status}\n  Saved to: {flags_path}",
                **flags)

    return flags


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: Video → Scene Reconstruction + "
                    "Segmentation + Object Detection → Point Cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (TTT3R, no VITRA)
  python run_pipeline.py --video input.mp4 --output_dir results

  # With VITRA annotation for hand-aware object detection
  python run_pipeline.py --video input.mp4 --output_dir results \\
      --vitra_annotation annotation.npy

  # Using MegaSAM instead of TTT3R
  python run_pipeline.py --video input.mp4 --output_dir results --use_megasam

  # Skip steps to reuse existing outputs
  python run_pipeline.py --video input.mp4 --output_dir results \\
      --skip_reconstruction --skip_scene_seg
        """,
    )

    # Required
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save all outputs")

    # Optional
    parser.add_argument("--vitra_annotation", type=str, default=None,
                        help="Path to VITRA .npy annotation (enables hand_point detection)")
    parser.add_argument("--use_megasam", action="store_true",
                        help="Use MegaSAM instead of TTT3R (default: TTT3R)")
    parser.add_argument("--skip_reconstruction", action="store_true",
                        help="Skip step 1 (reuse existing ttt3r_output)")
    parser.add_argument("--skip_scene_seg", action="store_true",
                        help="Skip step 2 (scene segmentation)")
    parser.add_argument("--skip_sam3d", action="store_true",
                        help="Skip step 3 (object detection + reconstruction)")
    parser.add_argument("--skip_combine", action="store_true",
                        help="Skip step 4 (combine into point cloud)")
    parser.add_argument("--skip_visualizations", action="store_true",
                        help="Skip rendering PNG visualizations in combine step")
    parser.add_argument("--export_obj", action="store_true", default=True,
                        help="Export object reconstructions as Wavefront OBJ (default: True)")
    parser.add_argument("--no_export_obj", action="store_false", dest="export_obj",
                        help="Disable OBJ export of object reconstructions")
    parser.add_argument("--inference_steps", type=int, default=None,
                        help="Diffusion steps for 3D reconstruction (default: 25). "
                             "Lower values (e.g. 12) trade quality for speed.")
    parser.add_argument("--early_exit_score", type=float, default=0.8,
                        help="Stop sampling detection frames once score exceeds this (default: 0.8)")
    parser.add_argument("--parallel_steps", action="store_true", default=True,
                        help="Run steps 1 and 2 in parallel (default: True)")
    parser.add_argument("--no_parallel_steps", action="store_false", dest="parallel_steps",
                        help="Run steps 1 and 2 sequentially")
    parser.add_argument("--frame_interval", type=int, default=1,
                        help="Process every Nth frame (default: 1)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use: 'cuda' or 'cpu' (default: cuda)")
    parser.add_argument("--prompt", type=str,
                        default="object held in either hand",
                        help="Object text prompt for SAM3D (default: auto from VITRA or "
                             "'object held in either hand')")

    # Conda environments
    parser.add_argument("--ttt3r_env", type=str, default="ttt3r",
                        help="Conda environment for TTT3R (default: ttt3r)")
    parser.add_argument("--megasam_env", type=str, default="mega_sam",
                        help="Conda environment for MegaSAM (default: mega_sam)")
    parser.add_argument("--sam3d_env", type=str, default="sam3d-objects",
                        help="Conda environment for SAM3D and scene seg (default: sam3d-objects)")

    # TTT3R-specific (internal defaults)
    parser.add_argument("--ttt3r_model", type=str,
                        default="TTT3R/src/cut3r_512_dpt_4_64.pth",
                        help="Path to TTT3R model checkpoint")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Input image size for TTT3R (default: 512)")

    # Undistortion
    parser.add_argument("--intrinsics_root", type=str, default=None,
                        help="Root dir for intrinsics (ego4d/*.npy, egoexo4d/*.json). "
                             "Default: {project_root}/vitra_data/intrinsics")
    parser.add_argument("--skip_undistortion", action="store_true",
                        help="Skip step 0 (video undistortion for Ego4D/EgoExo4D)")

    # Quality filters
    parser.add_argument("--check_deformable", action="store_true",
                        help="Use LLM to flag deformable objects (loads Qwen2.5-7B)")
    parser.add_argument("--episode_id", type=str, default=None,
                        help="Episode identifier for quality_flags.json "
                             "(default: inferred from VITRA annotation filename)")

    # Logging
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Enable verbose print output (default: True). "
                             "Use --no-verbose to suppress.")

    return parser.parse_args()


def run_undistortion(video_path, vitra_annotation, output_dir, sam3d_env,
                     intrinsics_root=None, log=None):
    """Step 0: Undistort the input video if Ego4D or EgoExo4D.

    Returns the path to the video to use for downstream steps
    (either the undistorted output or the original path if no
    undistortion was needed).
    """
    log.log("step0_undistort",
            "\n" + "=" * 80 + "\nSTEP 0: Undistorting Video\n" + "=" * 80)

    undistorted_path = os.path.join(output_dir, "undistorted_video.mp4")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = get_python_cmd(sam3d_env) + [
        os.path.join(script_dir, "undistort_clip.py"),
        "--video", os.path.abspath(video_path),
        "--annotation", os.path.abspath(vitra_annotation),
        "--output", os.path.abspath(undistorted_path),
    ]
    if intrinsics_root is not None:
        cmd.extend(["--intrinsics_root", os.path.abspath(intrinsics_root)])

    log.log("step0_undistort", f"Running: {' '.join(cmd)}\n", command=cmd)

    t0 = time.time()
    result = subprocess.run(
        cmd, check=False, capture_output=True, text=True,
        env=get_env_vars(sam3d_env),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.log("step0_undistort",
                f"\nERROR: Undistortion failed (rc={result.returncode})\n"
                f"{result.stderr}",
                returncode=result.returncode, elapsed_s=elapsed, success=False)
        sys.exit(1)

    # Parse the USE_VIDEO= line from stdout to get the actual path
    use_video = video_path
    for line in result.stdout.splitlines():
        if line.startswith("USE_VIDEO="):
            use_video = line.split("=", 1)[1].strip()
            break

    if log.verbose and result.stdout:
        # Print non-USE_VIDEO lines
        for line in result.stdout.splitlines():
            if not line.startswith("USE_VIDEO="):
                print(line)

    if use_video != os.path.abspath(video_path):
        log.log("step0_undistort",
                f"\nUndistortion completed in {elapsed:.1f}s → {use_video}",
                output=use_video, elapsed_s=elapsed, success=True)
    else:
        log.log("step0_undistort",
                f"\nNo undistortion needed, using original video.",
                output=use_video, elapsed_s=elapsed, success=True)

    return use_video


def main():
    args = parse_args()
    log = PipelineLog(verbose=args.verbose)
    pipeline_start = time.time()

    backend = "MegaSAM" if args.use_megasam else "TTT3R"
    log.log("config",
            "=" * 80 + "\nUnified Video Pipeline\n" + "=" * 80
            + f"\nInput video:       {args.video}"
            + f"\nOutput directory:  {args.output_dir}"
            + f"\nFrame interval:    {args.frame_interval}"
            + f"\nDevice:            {args.device}"
            + f"\nDepth/Pose:        {backend}"
            + (f"\nVITRA annotation:  {args.vitra_annotation}" if args.vitra_annotation else "")
            + "\n",
            video=args.video, output_dir=args.output_dir,
            frame_interval=args.frame_interval, device=args.device,
            backend=backend, vitra_annotation=args.vitra_annotation)

    # Validate input video
    if not os.path.exists(args.video):
        log.log("config", f"ERROR: Video file not found: {args.video}",
                error="video_not_found")
        sys.exit(1)

    if args.vitra_annotation and not os.path.exists(args.vitra_annotation):
        log.log("config", f"ERROR: VITRA annotation not found: {args.vitra_annotation}",
                error="vitra_not_found")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Check required conda environments
    envs_to_check = []
    if not args.skip_reconstruction:
        if args.use_megasam:
            envs_to_check.append(("MegaSAM", args.megasam_env))
        else:
            envs_to_check.append(("TTT3R", args.ttt3r_env))
    if not args.skip_scene_seg or not args.skip_sam3d or not args.skip_combine:
        envs_to_check.append(("SAM3D/Scene-Seg", args.sam3d_env))

    for label, env_name in envs_to_check:
        if log.verbose:
            print(f"Checking {label} conda environment: {env_name}...", end=" ")
        if not check_conda_env(env_name):
            if log.verbose:
                print("NOT FOUND")
            log.log("env_check",
                     f"ERROR: Conda environment '{env_name}' does not exist.",
                     env_name=env_name, label=label, found=False)
            sys.exit(1)
        if log.verbose:
            print("OK")
        log.log("env_check", f"Conda env '{env_name}' ({label}): OK",
                env_name=env_name, label=label, found=True)
    if log.verbose:
        print()

    # -----------------------------------------------------------------------
    # Step 0: Undistort video (Ego4D / EgoExo4D only)
    # -----------------------------------------------------------------------
    if args.vitra_annotation and not args.skip_undistortion:
        effective_video = run_undistortion(
            args.video, args.vitra_annotation, args.output_dir,
            args.sam3d_env, intrinsics_root=args.intrinsics_root, log=log,
        )
    else:
        effective_video = args.video

    # -----------------------------------------------------------------------
    # Steps 1 & 2: Scene Reconstruction + Scene Segmentation
    # These are independent and can run in parallel when --parallel_steps
    # is enabled (default). Both fit in ~14GB VRAM on a 48GB A6000.
    # -----------------------------------------------------------------------
    run_step1 = not args.skip_reconstruction
    run_step2 = not args.skip_scene_seg
    can_parallel = args.parallel_steps and run_step1 and run_step2

    if can_parallel:
        log.log("parallel", "\nRunning Steps 1 & 2 in parallel...")

        def _do_step1():
            if args.use_megasam:
                return run_megasam(
                    effective_video, args.output_dir, args.frame_interval,
                    args.device, args.megasam_env, log=log,
                )
            else:
                return run_ttt3r(
                    effective_video, args.output_dir, args.frame_interval,
                    args.device, args.ttt3r_env, args.ttt3r_model, args.img_size,
                    log=log,
                )

        def _do_step2():
            return run_scene_segmentation(
                effective_video, args.output_dir, args.device, args.sam3d_env,
                log=log,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut1 = pool.submit(_do_step1)
            fut2 = pool.submit(_do_step2)
            ttt3r_out = fut1.result()
            seg_out = fut2.result()
    else:
        # Sequential execution
        if args.skip_reconstruction:
            ttt3r_out = os.path.join(args.output_dir, "ttt3r_output")
            log.log("step1_reconstruction",
                    f"Skipping reconstruction (using existing output: {ttt3r_out})",
                    skipped=True, output_dir=ttt3r_out)
            if not os.path.exists(ttt3r_out):
                log.log("step1_reconstruction",
                        f"ERROR: Reconstruction output not found: {ttt3r_out}",
                        error="output_not_found")
                sys.exit(1)
        elif args.use_megasam:
            ttt3r_out = run_megasam(
                effective_video, args.output_dir, args.frame_interval,
                args.device, args.megasam_env, log=log,
            )
        else:
            ttt3r_out = run_ttt3r(
                effective_video, args.output_dir, args.frame_interval,
                args.device, args.ttt3r_env, args.ttt3r_model, args.img_size,
                log=log,
            )

        if args.skip_scene_seg:
            seg_out = os.path.join(args.output_dir, "scene_segmentation")
            log.log("step2_scene_seg", "\nSkipping scene segmentation.",
                    skipped=True, output_dir=seg_out)
        else:
            seg_out = run_scene_segmentation(
                effective_video, args.output_dir, args.device, args.sam3d_env,
                log=log,
            )

    # -----------------------------------------------------------------------
    # Step 3: Object Detection + 3D Reconstruction
    # -----------------------------------------------------------------------
    if args.skip_sam3d:
        sam3d_out = os.path.join(args.output_dir, "sam3d_output")
        log.log("step3_sam3d", "\nSkipping SAM3D.",
                skipped=True)
        if not os.path.exists(sam3d_out):
            sam3d_out = None
    else:
        sam3d_out = run_sam3d(
            effective_video, ttt3r_out, args.output_dir, args.prompt,
            args.frame_interval, args.device, args.sam3d_env,
            vitra_annotation=args.vitra_annotation,
            inference_steps=args.inference_steps,
            early_exit_score=args.early_exit_score, log=log,
        )

    # -----------------------------------------------------------------------
    # Quality Filters
    # -----------------------------------------------------------------------
    # Resolve action text from VITRA annotation (if available)
    action_text = args.prompt
    if args.vitra_annotation:
        try:
            import numpy as np
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            sys.path.insert(0, os.path.join(project_root, "vitra_data", "pipeline"))
            from vitra_utils import load_vitra_annotation, get_text_prompt
            annotation = load_vitra_annotation(args.vitra_annotation)
            action_text = get_text_prompt(annotation)
        except Exception:
            pass  # fall back to args.prompt

    # Resolve episode_id
    episode_id = args.episode_id
    if episode_id is None and args.vitra_annotation:
        episode_id = os.path.splitext(os.path.basename(args.vitra_annotation))[0]

    quality_flags = check_quality_flags(
        args.output_dir, sam3d_out, action_text,
        episode_id=episode_id,
        check_deformable=args.check_deformable,
        sam3d_skipped=args.skip_sam3d,
        log=log,
    )

    # -----------------------------------------------------------------------
    # Step 4: Combine into Point Cloud
    # -----------------------------------------------------------------------
    if args.skip_combine:
        log.log("step4_combine", "\nSkipping combine step.",
                skipped=True)
        ply_out = None
    else:
        ply_out = combine_outputs(
            ttt3r_out, sam3d_out, args.output_dir,
            args.frame_interval, args.sam3d_env,
            vitra_annotation=args.vitra_annotation,
            skip_visualizations=args.skip_visualizations,
            export_obj=args.export_obj, log=log,
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    summary_lines = (
        "\n" + "=" * 80
        + "\nPIPELINE COMPLETE!"
        + "\n" + "=" * 80
        + f"\nReconstruction: {ttt3r_out}"
        + f"\nScene seg:      {seg_out}"
        + (f"\nSAM3D output:   {sam3d_out}" if sam3d_out else "")
        + (f"\nPoint clouds:   {ply_out}" if ply_out else "")
        + f"\nTotal time:     {total_elapsed:.1f}s"
        + "\n"
    )
    log.log("summary", summary_lines,
            reconstruction_dir=ttt3r_out, scene_seg_dir=seg_out,
            sam3d_dir=sam3d_out, ply_dir=ply_out,
            total_elapsed_s=total_elapsed)

    # Save pipeline log (always, regardless of verbose)
    log_path = os.path.join(args.output_dir, "pipeline_log.json")
    with open(log_path, "w") as f:
        json.dump(log.to_dict(), f, indent=2, default=str)
    if args.verbose:
        print(f"Pipeline log saved to {log_path}")


if __name__ == "__main__":
    main()
