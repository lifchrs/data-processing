#!/usr/bin/env python3
"""
Multi-GPU parallel batch processor for the unified pipeline.

Distributes N videos across M GPUs, running one video per GPU simultaneously.
Achieves near-linear throughput scaling: 8 GPUs → ~8× throughput.

Usage:
    # Process specific annotation files (auto-finds SSv2 videos):
    python batch_parallel.py --annotations path/to/*.npy --ssv2_video_dir /data/ssv2/videos

    # Use specific GPUs:
    python batch_parallel.py --annotations *.npy --ssv2_video_dir /data/ssv2 --gpus 0,1,2,3

    # Process a manifest file (JSON list of {video_path, annotation_path, ...}):
    python batch_parallel.py --manifest videos.json

    # With pipeline flags:
    python batch_parallel.py --manifest videos.json --skip_scene_seg --frame_interval 2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def detect_gpus():
    """Detect available NVIDIA GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return [int(idx.strip()) for idx in result.stdout.strip().split("\n") if idx.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [0]


def load_manifest(manifest_path):
    """Load a JSON manifest of videos to process.

    Expected format: list of dicts with at least 'video_path'.
    Optional keys: 'annotation_path', 'video_name', 'action_text'.
    """
    with open(manifest_path) as f:
        entries = json.load(f)

    episodes = []
    for entry in entries:
        ep = {
            "video_path": entry["video_path"],
            "annotation_path": entry.get("annotation_path"),
            "video_name": entry.get("video_name",
                                    os.path.splitext(os.path.basename(entry["video_path"]))[0]),
            "action_text": entry.get("action_text", ""),
        }
        episodes.append(ep)
    return episodes


def _detect_dataset(annotation_path):
    """Determine the source dataset from an annotation filename prefix."""
    basename = os.path.basename(annotation_path)
    for prefix, dataset in [
        ("somethingsomethingv2_", "ssv2"),
        ("epic_kitchens_", "epic"),
        ("EgoExo4D_", "egoexo4d"),
        ("Ego4D_", "ego4d"),
    ]:
        if basename.startswith(prefix):
            return dataset, prefix
    return None, ""


def _extract_video_name(annotation_path):
    """Extract the video ID from an annotation filename."""
    import re
    basename = os.path.splitext(os.path.basename(annotation_path))[0]
    m = re.match(r"^(.+)_ep_\d+$", basename)
    body = m.group(1) if m else basename
    for prefix, _ in [
        ("somethingsomethingv2_", "ssv2"),
        ("epic_kitchens_", "epic"),
        ("EgoExo4D_", "egoexo4d"),
        ("Ego4D_", "ego4d"),
    ]:
        if body.startswith(prefix):
            return body[len(prefix):]
    return body


def _get_action_text(annotation):
    """Extract action text from a loaded VITRA annotation dict."""
    text_data = annotation.get("text", {})
    for hand in ["left", "right"]:
        entries = text_data.get(hand, [])
        if entries:
            return entries[0][0]
    return "object held in hand"


def episodes_from_annotations(annotation_paths, ssv2_video_dir):
    """Build episode list from .npy annotation files, auto-resolving video paths.

    For SSv2 annotations, the video is looked up automatically in
    ssv2_video_dir.  Other datasets require a --manifest with explicit video_path.
    """
    import numpy as np

    episodes = []
    for annot_path in annotation_paths:
        annot_path = os.path.abspath(annot_path)
        dataset, _ = _detect_dataset(annot_path)
        video_name = _extract_video_name(annot_path)

        # Resolve video path based on dataset
        video_path = None
        if dataset == "ssv2":
            for ext in [".webm", ".mp4"]:
                p = os.path.join(ssv2_video_dir, f"{video_name}{ext}")
                if os.path.exists(p):
                    video_path = p
                    break
            if video_path is None:
                print(f"WARNING: SSv2 video not found for {video_name} "
                      f"in {ssv2_video_dir}, skipping")
                continue
        else:
            print(f"WARNING: Auto video lookup not supported for dataset "
                  f"'{dataset}' ({os.path.basename(annot_path)}). "
                  f"Use --manifest with explicit video_path instead.")
            continue

        # Load annotation for action text
        annotation = np.load(annot_path, allow_pickle=True).item()
        action_text = _get_action_text(annotation)

        episodes.append({
            "annotation_path": annot_path,
            "video_path": video_path,
            "video_name": video_name,
            "action_text": action_text,
        })

    return episodes


def run_single_video(episode, output_dir, gpu_id, pipeline_args):
    """Run the unified pipeline on a single video, pinned to a specific GPU.

    This function runs in a subprocess worker. It sets CUDA_VISIBLE_DEVICES
    so the pipeline only sees the assigned GPU.
    """
    ep_output = os.path.join(output_dir, episode["video_name"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Build pipeline command
    python_bin = sys.executable  # use same Python that launched this script
    cmd = [
        python_bin, os.path.join(SCRIPT_DIR, "run_pipeline.py"),
        "--video", episode["video_path"],
        "--output_dir", ep_output,
        "--device", "cuda",  # always cuda (mapped to the assigned GPU via CUDA_VISIBLE_DEVICES)
    ]

    if episode.get("annotation_path"):
        cmd.extend(["--vitra_annotation", episode["annotation_path"]])

    # Pass through pipeline flags
    for key, val in pipeline_args.items():
        if val is True:
            cmd.append(f"--{key}")
        elif val is not False and val is not None:
            cmd.extend([f"--{key}", str(val)])

    t0 = time.time()
    result = subprocess.run(cmd, check=False, env=env)
    elapsed = time.time() - t0

    return {
        "video_name": episode["video_name"],
        "action_text": episode.get("action_text", ""),
        "gpu_id": gpu_id,
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": ep_output,
    }


def _run_with_batch_workers(episodes, output_dir, gpu_ids, max_parallel, pipeline_args):
    """Partition videos across GPUs and run persistent batch workers.

    Each GPU gets a batch_worker.py process that loads models once and
    processes its assigned videos sequentially. Multiple GPUs run in parallel.
    """
    import tempfile

    # Partition episodes across GPUs (round-robin)
    gpu_episodes = {gpu_id: [] for gpu_id in gpu_ids}
    for i, ep in enumerate(episodes):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        gpu_episodes[gpu_id].append(ep)

    # Create per-GPU manifest files
    manifests = {}
    for gpu_id, eps in gpu_episodes.items():
        if not eps:
            continue
        # Build manifest entries with output_dir per video
        manifest_entries = []
        for ep in eps:
            entry = {
                "video_path": ep["video_path"],
                "output_dir": os.path.join(output_dir, ep["video_name"]),
                "annotation_path": ep.get("annotation_path"),
                "video_name": ep["video_name"],
            }
            # Check if TTT3R frames exist (from a prior Step 1 run)
            ttt3r_frames = os.path.join(entry["output_dir"], "ttt3r_output", "color")
            if os.path.isdir(ttt3r_frames):
                entry["frames_dir"] = ttt3r_frames
            ttt3r_camera = os.path.join(entry["output_dir"], "ttt3r_output", "camera")
            if os.path.isdir(ttt3r_camera):
                entry["camera_dir"] = ttt3r_camera
            manifest_entries.append(entry)

        manifest_path = os.path.join(output_dir, f"manifest_gpu{gpu_id}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest_entries, f, indent=2)
        manifests[gpu_id] = manifest_path

    # Phase 1: Run TTT3R batch worker on each GPU (if not skipped)
    if not pipeline_args.get("skip_reconstruction"):
        print("Phase 1: Running TTT3R batch workers...")
        _run_worker_phase(manifests, "ttt3r", gpu_ids, max_parallel, pipeline_args)

    # Phase 2: Run SAM3D batch worker on each GPU
    print("\nPhase 2: Running SAM3D batch workers (Steps 2-4)...")
    # Update manifests with frames_dir/camera_dir now that TTT3R has run
    for gpu_id, manifest_path in manifests.items():
        with open(manifest_path) as f:
            entries = json.load(f)
        for entry in entries:
            ttt3r_out = os.path.join(entry["output_dir"], "ttt3r_output")
            entry["frames_dir"] = os.path.join(ttt3r_out, "color")
            entry["camera_dir"] = os.path.join(ttt3r_out, "camera")
        with open(manifest_path, "w") as f:
            json.dump(entries, f, indent=2)

    _run_worker_phase(manifests, "sam3d", gpu_ids, max_parallel, pipeline_args)

    # Collect results
    results = []
    for ep in episodes:
        ep_dir = os.path.join(output_dir, ep["video_name"])
        success = os.path.isdir(os.path.join(ep_dir, "ply_output")) or \
                  os.path.isdir(os.path.join(ep_dir, "sam3d_output"))
        results.append({
            "video_name": ep["video_name"],
            "action_text": ep.get("action_text", ""),
            "gpu_id": gpu_ids[episodes.index(ep) % len(gpu_ids)],
            "returncode": 0 if success else 1,
            "elapsed_seconds": 0,  # individual timing not available in batch mode
            "output_dir": ep_dir,
        })

    return results


def _run_worker_phase(manifests, mode, gpu_ids, max_parallel, pipeline_args):
    """Launch batch_worker.py for each GPU in parallel."""
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for gpu_id, manifest_path in manifests.items():
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            cmd = [
                sys.executable, os.path.join(SCRIPT_DIR, "batch_worker.py"),
                "--mode", mode,
                "--manifest", manifest_path,
                "--device", "cuda",
                "--frame_interval", str(pipeline_args.get("frame_interval", 1)),
            ]

            if pipeline_args.get("skip_scene_seg"):
                cmd.append("--skip_scene_seg")
            if pipeline_args.get("skip_sam3d"):
                cmd.append("--skip_sam3d")
            if pipeline_args.get("skip_combine"):
                cmd.append("--skip_combine")
            if pipeline_args.get("skip_visualizations"):
                cmd.append("--skip_visualizations")

            future = executor.submit(
                subprocess.run, cmd, check=False, env=env,
            )
            futures[future] = gpu_id

        for future in as_completed(futures):
            gpu_id = futures[future]
            result = future.result()
            status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
            print(f"  GPU {gpu_id} {mode} worker: {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU parallel batch processor for unified pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Video source (one of these required)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=str,
                        help="Path to JSON manifest file with video list")
    source.add_argument("--annotations", nargs="+", metavar="NPY",
                        help="VITRA .npy annotation files. For SSv2 annotations the "
                             "video is found automatically.")

    # GPU config
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs to use (default: auto-detect all)")
    parser.add_argument("--max_parallel", type=int, default=None,
                        help="Max videos to run in parallel (default: number of GPUs)")
    parser.add_argument("--use_batch_worker", action="store_true",
                        help="Use persistent batch workers (loads models once per GPU, "
                             "processes multiple videos). Saves ~30-60s per video in model "
                             "load time. Requires running in the correct conda env.")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Batch output directory (default: unified_pipeline/batch_parallel_output)")
    parser.add_argument("--ssv2_video_dir", type=str, default=None,
                        help="Directory containing SSv2 video files. Required with "
                             "--annotations for SSv2 annotations.")

    # Pipeline flags (passed through to run_pipeline.py)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--use_megasam", action="store_true")
    parser.add_argument("--skip_reconstruction", action="store_true")
    parser.add_argument("--skip_scene_seg", action="store_true")
    parser.add_argument("--skip_sam3d", action="store_true")
    parser.add_argument("--skip_combine", action="store_true")
    parser.add_argument("--skip_visualizations", action="store_true")
    parser.add_argument("--inference_steps", type=int, default=None,
                        help="Diffusion steps for 3D reconstruction (default: 25). "
                             "Lower values (e.g. 12) trade quality for speed.")
    parser.add_argument("--early_exit_score", type=float, default=0.8,
                        help="Stop sampling detection frames once score exceeds this (default: 0.8)")

    # Conda envs (passed through)
    parser.add_argument("--ttt3r_env", type=str, default="ttt3r")
    parser.add_argument("--megasam_env", type=str, default="mega_sam")
    parser.add_argument("--sam3d_env", type=str, default="sam3d-objects")

    args = parser.parse_args()

    # Resolve GPUs
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        gpu_ids = detect_gpus()

    max_parallel = args.max_parallel or len(gpu_ids)

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "batch_parallel_output")

    # Load episodes
    if args.manifest:
        episodes = load_manifest(args.manifest)
    else:
        if args.ssv2_video_dir is None:
            # Default to the old location if it exists, otherwise error
            default_dir = os.path.join(PROJECT_ROOT, "VITRA", "20bn-something-something-v2")
            if os.path.isdir(default_dir):
                args.ssv2_video_dir = default_dir
            else:
                print("ERROR: --ssv2_video_dir is required with --annotations "
                      "(directory containing SSv2 .webm/.mp4 files)")
                sys.exit(1)
        episodes = episodes_from_annotations(args.annotations, args.ssv2_video_dir)

    if not episodes:
        print("ERROR: No videos to process.")
        sys.exit(1)

    # Collect pipeline args to pass through
    pipeline_args = {
        "frame_interval": args.frame_interval,
        "ttt3r_env": args.ttt3r_env,
        "megasam_env": args.megasam_env,
        "sam3d_env": args.sam3d_env,
        "use_megasam": args.use_megasam,
        "skip_reconstruction": args.skip_reconstruction,
        "skip_scene_seg": args.skip_scene_seg,
        "skip_sam3d": args.skip_sam3d,
        "skip_combine": args.skip_combine,
        "skip_visualizations": args.skip_visualizations,
        "inference_steps": args.inference_steps,
        "early_exit_score": args.early_exit_score,
    }

    print("=" * 80)
    print("Multi-GPU Parallel Batch Processor")
    print("=" * 80)
    print(f"Videos:        {len(episodes)}")
    print(f"GPUs:          {gpu_ids} ({len(gpu_ids)} total)")
    print(f"Max parallel:  {max_parallel}")
    print(f"Output:        {args.output_dir}")
    print()

    # Print episode list
    print(f"{'#':>3}  {'Video':>10}  Action")
    print("-" * 60)
    for i, ep in enumerate(episodes):
        print(f"{i+1:>3}  {ep['video_name']:>10}  {ep.get('action_text', '')}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(episodes, f, indent=2, default=str)

    # Process videos in parallel across GPUs
    results = []
    batch_start = time.time()

    if args.use_batch_worker:
        # Batch worker mode: partition videos across GPUs, each GPU runs a
        # persistent worker that loads models once
        print("Using persistent batch workers (models loaded once per GPU)\n")
        results = _run_with_batch_workers(
            episodes, args.output_dir, gpu_ids, max_parallel, pipeline_args,
        )
    else:
        # Standard mode: one subprocess per video
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all videos, cycling through GPUs
            futures = {}
            for i, episode in enumerate(episodes):
                gpu_id = gpu_ids[i % len(gpu_ids)]
                future = executor.submit(
                    run_single_video, episode, args.output_dir, gpu_id, pipeline_args,
                )
                futures[future] = (i, episode)

            # Collect results as they complete
            for future in as_completed(futures):
                idx, episode = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "video_name": episode["video_name"],
                        "action_text": episode.get("action_text", ""),
                        "gpu_id": -1,
                        "returncode": -1,
                        "elapsed_seconds": 0,
                        "output_dir": "",
                        "error": str(e),
                    }

                results.append(result)
                status = "OK" if result["returncode"] == 0 else f"FAIL (rc={result['returncode']})"
                n_done = len(results)
                print(f"[{n_done}/{len(episodes)}] GPU {result['gpu_id']}: "
                      f"{result['video_name']} -> {status} ({result['elapsed_seconds']}s)")

    batch_elapsed = time.time() - batch_start

    # Sort results by video name for consistent display
    results.sort(key=lambda r: r["video_name"])

    # Summary
    n_ok = sum(1 for r in results if r["returncode"] == 0)
    total_video_time = sum(r["elapsed_seconds"] for r in results)

    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"\n{'#':>3}  {'Video':>10}  {'GPU':>3}  {'Status':>6}  {'Time':>7}  Action")
    print("-" * 80)
    for i, r in enumerate(results):
        status = "OK" if r["returncode"] == 0 else "FAIL"
        print(f"{i+1:>3}  {r['video_name']:>10}  {r['gpu_id']:>3}  {status:>6}  "
              f"{r['elapsed_seconds']:>6.1f}s  {r.get('action_text', '')}")

    print(f"\n{n_ok}/{len(results)} succeeded")
    print(f"Wall time:      {batch_elapsed:.1f}s ({batch_elapsed/60:.1f} min)")
    print(f"Total CPU time: {total_video_time:.1f}s ({total_video_time/60:.1f} min)")
    print(f"Speedup:        {total_video_time/max(batch_elapsed, 1):.1f}×")
    print(f"Output:         {args.output_dir}")

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    summary = {
        "n_videos": len(results),
        "n_ok": n_ok,
        "gpus": gpu_ids,
        "max_parallel": max_parallel,
        "wall_time_s": round(batch_elapsed, 1),
        "total_video_time_s": round(total_video_time, 1),
        "speedup": round(total_video_time / max(batch_elapsed, 1), 1),
        "results": results,
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results:        {results_path}")

    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
