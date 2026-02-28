#!/usr/bin/env python3
"""
Batch test the unified pipeline on SSv2 videos with VITRA annotations.

Selects a sample of SSv2 episodes that have valid hand data and runs the full
unified pipeline on each. Results are saved to a batch output directory.

Usage:
    python batch_test_ssv2.py --num_videos 5
    python batch_test_ssv2.py --num_videos 10 --skip_reconstruction --skip_scene_seg
"""

import argparse
import json
import os
import sys
import time
import subprocess
import random

import numpy as np

# Import vitra_utils from sibling directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "vitra_data", "pipeline"))
from vitra_utils import load_vitra_annotation, get_video_info, get_text_prompt

# Paths
ANNOT_DIR = os.path.join(PROJECT_ROOT, "vitra_data", "ssv2", "episodic_annotations")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "VITRA", "20bn-something-something-v2")


def find_ssv2_video(video_name):
    """Find an SSv2 video file by its ID."""
    for ext in [".webm", ".mp4"]:
        path = os.path.join(VIDEO_DIR, f"{video_name}{ext}")
        if os.path.exists(path):
            return path
    return None


def select_annotations(num_videos, seed=42):
    """Select SSv2 annotations with valid hand data and matching videos."""
    if not os.path.isdir(ANNOT_DIR):
        print(f"ERROR: Annotation directory not found: {ANNOT_DIR}")
        print("Extract ssv2.tar.gz first:")
        print(f"  tar xzf {PROJECT_ROOT}/vitra_data/ssv2.tar.gz -C {PROJECT_ROOT}/vitra_data/")
        sys.exit(1)

    annots = sorted(os.listdir(ANNOT_DIR))
    random.seed(seed)
    random.shuffle(annots)

    selected = []
    for fname in annots:
        if len(selected) >= num_videos:
            break

        annot_path = os.path.join(ANNOT_DIR, fname)
        annotation = load_vitra_annotation(annot_path)
        info = get_video_info(annotation)

        # Check video exists
        video_path = find_ssv2_video(info["video_name"])
        if video_path is None:
            continue

        # Check has valid hand action text
        action_text = get_text_prompt(annotation)
        if action_text == "object held in hand":
            continue  # no real action text

        # Check hand joints are available for the active hand
        text_data = annotation.get("text", {})
        active_hand = None
        for hand in ["left", "right"]:
            if text_data.get(hand, []):
                active_hand = hand
                break

        if active_hand is None:
            continue

        hand_data = annotation.get(active_hand, {})
        kept_frames = hand_data.get("kept_frames")
        if kept_frames is None or np.sum(kept_frames) < 5:
            continue

        selected.append({
            "annotation_file": fname,
            "annotation_path": annot_path,
            "video_path": video_path,
            "video_name": info["video_name"],
            "num_frames": info["num_frames"],
            "action_text": action_text,
            "active_hand": active_hand,
            "valid_hand_frames": int(np.sum(kept_frames)),
        })

    return selected


def run_pipeline_on_episode(episode, output_dir, args):
    """Run the unified pipeline on a single episode."""
    ep_output = os.path.join(output_dir, episode["video_name"])

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "run_pipeline.py"),
        "--video", episode["video_path"],
        "--output_dir", ep_output,
        "--vitra_annotation", episode["annotation_path"],
        "--frame_interval", str(args.frame_interval),
        "--device", args.device,
        "--sam3d_env", args.sam3d_env,
    ]

    if args.use_megasam:
        cmd.extend(["--use_megasam", "--megasam_env", args.megasam_env])
    else:
        cmd.extend(["--ttt3r_env", args.ttt3r_env])

    if args.skip_reconstruction:
        cmd.append("--skip_reconstruction")
    if args.skip_scene_seg:
        cmd.append("--skip_scene_seg")
    if args.skip_sam3d:
        cmd.append("--skip_sam3d")
    if args.skip_combine:
        cmd.append("--skip_combine")
    if getattr(args, 'skip_visualizations', False):
        cmd.append("--skip_visualizations")

    print(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    return {
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": ep_output,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch test unified pipeline on SSv2 + VITRA",
    )
    parser.add_argument("--num_videos", type=int, default=5,
                        help="Number of SSv2 videos to test (default: 5)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Batch output directory (default: unified_pipeline/batch_ssv2_output)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for annotation selection (default: 42)")

    # Pipeline flags (pass through to run_pipeline.py)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_megasam", action="store_true")
    parser.add_argument("--skip_reconstruction", action="store_true")
    parser.add_argument("--skip_scene_seg", action="store_true")
    parser.add_argument("--skip_sam3d", action="store_true")
    parser.add_argument("--skip_combine", action="store_true")
    parser.add_argument("--skip_visualizations", action="store_true")

    # Conda envs
    parser.add_argument("--ttt3r_env", type=str, default="ttt3r")
    parser.add_argument("--megasam_env", type=str, default="mega_sam")
    parser.add_argument("--sam3d_env", type=str, default="sam3d-objects")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "batch_ssv2_output")

    print("=" * 80)
    print("Batch Test: Unified Pipeline on SSv2 + VITRA")
    print("=" * 80)
    print(f"Selecting {args.num_videos} videos...")
    print()

    episodes = select_annotations(args.num_videos, seed=args.seed)

    if not episodes:
        print("ERROR: No valid episodes found.")
        sys.exit(1)

    print(f"Selected {len(episodes)} episodes:\n")
    print(f"{'#':>3}  {'Video':>8}  {'Hand':>5}  {'Frames':>6}  {'HandF':>5}  Action")
    print("-" * 80)
    for i, ep in enumerate(episodes):
        print(f"{i+1:>3}  {ep['video_name']:>8}  {ep['active_hand']:>5}  "
              f"{ep['num_frames']:>6}  {ep['valid_hand_frames']:>5}  {ep['action_text']}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save selection manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(episodes, f, indent=2)

    # Run pipeline on each episode
    results = []
    for i, ep in enumerate(episodes):
        print("\n" + "=" * 80)
        print(f"[{i+1}/{len(episodes)}] Video {ep['video_name']}: {ep['action_text']}")
        print("=" * 80)

        result = run_pipeline_on_episode(ep, args.output_dir, args)
        result["video_name"] = ep["video_name"]
        result["action_text"] = ep["action_text"]
        results.append(result)

        status = "OK" if result["returncode"] == 0 else f"FAIL (rc={result['returncode']})"
        print(f"\n  -> {status} in {result['elapsed_seconds']}s")

    # Summary
    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"\n{'#':>3}  {'Video':>8}  {'Status':>8}  {'Time':>7}  Action")
    print("-" * 80)
    n_ok = 0
    for i, r in enumerate(results):
        status = "OK" if r["returncode"] == 0 else "FAIL"
        if r["returncode"] == 0:
            n_ok += 1
        print(f"{i+1:>3}  {r['video_name']:>8}  {status:>8}  {r['elapsed_seconds']:>6.1f}s  {r['action_text']}")

    print(f"\n{n_ok}/{len(results)} succeeded")
    print(f"Output: {args.output_dir}")

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {results_path}")

    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
