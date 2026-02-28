#!/usr/bin/env python3
"""
Batch run the unified pipeline on 25 SSv2 + 25 Epic-Kitchens videos.

Selects episodes with valid hand data from both datasets and runs the full
unified pipeline on each sequentially.

Usage:
    python batch_run_50.py
    python batch_run_50.py --num_ssv2 25 --num_epic 25 --seed 123
    python batch_run_50.py --skip_reconstruction --skip_scene_seg  # rerun later steps
"""

import argparse
import json
import os
import subprocess
import sys
import time
import random

import numpy as np

# Import vitra_utils from sibling directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "vitra_data", "pipeline"))
from vitra_utils import load_vitra_annotation, get_video_info, get_text_prompt, frames_to_timestamp

# ── Paths ──────────────────────────────────────────────────────────────────
SSV2_ANNOT_DIR = os.path.join(PROJECT_ROOT, "vitra_data", "ssv2", "episodic_annotations")
SSV2_VIDEO_DIR = os.path.join(PROJECT_ROOT, "VITRA", "20bn-something-something-v2")

EPIC_ANNOT_DIR = os.path.join(
    PROJECT_ROOT, "vitra_data", "epic_annotations", "epic", "episodic_annotations"
)
EPIC_VIDEO_BASE = os.path.join(PROJECT_ROOT, "vitra_data", "epic_videos", "EPIC-KITCHENS")

# The 4 downloaded EPIC-Kitchens source videos
EPIC_DOWNLOADED_VIDEOS = {"P01_01", "P04_05", "P22_07", "P30_05"}

# Already-processed SSv2 videos (bench_15_phase_b)
ALREADY_PROCESSED_SSV2 = {
    "101888", "112397", "116184", "137544", "195066", "199381", "202971",
    "37086", "51359", "5321", "58029", "60910", "77349", "79484", "84911", "87626",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _validate_annotation(annotation):
    """Check annotation has valid hand data, action text, kept_frames >= 5."""
    action_text = get_text_prompt(annotation)
    if action_text == "object held in hand":
        return None

    text_data = annotation.get("text", {})
    active_hand = None
    for hand in ["left", "right"]:
        if text_data.get(hand, []):
            active_hand = hand
            break
    if active_hand is None:
        return None

    hand_data = annotation.get(active_hand, {})
    kept_frames = hand_data.get("kept_frames")
    if kept_frames is None or np.sum(kept_frames) < 5:
        return None

    return {
        "action_text": action_text,
        "active_hand": active_hand,
        "valid_hand_frames": int(np.sum(kept_frames)),
    }


def find_ssv2_video(video_name):
    """Find an SSv2 video file by its ID."""
    for ext in [".webm", ".mp4"]:
        path = os.path.join(SSV2_VIDEO_DIR, f"{video_name}{ext}")
        if os.path.exists(path):
            return path
    return None


def find_epic_source_video(video_name):
    """Find an EPIC-Kitchens source .MP4 by participant/video name (e.g. P01_01)."""
    participant = video_name.split("_")[0]  # P01
    path = os.path.join(EPIC_VIDEO_BASE, participant, "videos", f"{video_name}.MP4")
    if os.path.exists(path):
        return path
    return None


def extract_epic_clip(annotation, source_video, output_path):
    """Extract clip from EPIC-Kitchens source video using ffmpeg.

    EPIC-Kitchens is 60fps. Uses video_decode_frame from annotation.
    """
    if os.path.exists(output_path):
        return True  # already extracted

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    info = get_video_info(annotation)
    fps = 60.0  # EPIC-Kitchens
    start_time = frames_to_timestamp(info["start_frame"], fps)
    num_frames = info["num_frames"]

    cmd = [
        "ffmpeg", "-y",
        "-ss", start_time,
        "-i", source_video,
        "-frames:v", str(num_frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: ffmpeg clip extraction failed: {result.stderr[:200]}")
        return False
    return True


# ── Selection ──────────────────────────────────────────────────────────────

def select_ssv2(num_videos, seed):
    """Select SSv2 annotations, skipping already-processed videos."""
    if not os.path.isdir(SSV2_ANNOT_DIR):
        print(f"ERROR: SSv2 annotation dir not found: {SSV2_ANNOT_DIR}")
        sys.exit(1)

    annots = sorted(os.listdir(SSV2_ANNOT_DIR))
    rng = random.Random(seed)
    rng.shuffle(annots)

    selected = []
    for fname in annots:
        if len(selected) >= num_videos:
            break

        annot_path = os.path.join(SSV2_ANNOT_DIR, fname)
        annotation = load_vitra_annotation(annot_path)
        info = get_video_info(annotation)

        # Skip already processed
        if info["video_name"] in ALREADY_PROCESSED_SSV2:
            continue

        # Check video exists
        video_path = find_ssv2_video(info["video_name"])
        if video_path is None:
            continue

        valid = _validate_annotation(annotation)
        if valid is None:
            continue

        selected.append({
            "dataset": "ssv2",
            "video_id": info["video_name"],
            "annotation_file": fname,
            "annotation_path": annot_path,
            "video_path": video_path,
            "num_frames": info["num_frames"],
            **valid,
        })

    return selected


def select_epic(num_videos, seed, output_dir):
    """Select EPIC-Kitchens annotations from the 4 downloaded source videos."""
    if not os.path.isdir(EPIC_ANNOT_DIR):
        print(f"ERROR: EPIC annotation dir not found: {EPIC_ANNOT_DIR}")
        sys.exit(1)

    # Gather annotations matching the 4 downloaded videos
    candidates = []
    for fname in sorted(os.listdir(EPIC_ANNOT_DIR)):
        # epic_kitchens_P01_01_ep_000123.npy → video_name = P01_01
        parts = fname.replace("epic_kitchens_", "").split("_ep_")
        if len(parts) != 2:
            continue
        video_name = parts[0]
        if video_name not in EPIC_DOWNLOADED_VIDEOS:
            continue
        candidates.append(fname)

    rng = random.Random(seed + 1)  # different seed from SSv2
    rng.shuffle(candidates)

    selected = []
    for fname in candidates:
        if len(selected) >= num_videos:
            break

        annot_path = os.path.join(EPIC_ANNOT_DIR, fname)
        annotation = load_vitra_annotation(annot_path)
        info = get_video_info(annotation)

        valid = _validate_annotation(annotation)
        if valid is None:
            continue

        # Find source video
        source_video = find_epic_source_video(info["video_name"])
        if source_video is None:
            continue

        # Clip will be extracted to output_dir/clips/<fname>.mp4
        clip_name = fname.replace(".npy", ".mp4")
        clip_path = os.path.join(output_dir, "clips", clip_name)

        # Video ID for directory naming: e.g. P01_01_ep_000123
        ep_id = fname.replace("epic_kitchens_", "").replace(".npy", "")

        selected.append({
            "dataset": "epic",
            "video_id": ep_id,
            "annotation_file": fname,
            "annotation_path": annot_path,
            "video_path": clip_path,  # will be extracted
            "source_video": source_video,
            "num_frames": info["num_frames"],
            **valid,
        })

    return selected


# ── Pipeline runner ────────────────────────────────────────────────────────

def run_pipeline_on_episode(episode, output_dir, args):
    """Run the unified pipeline on a single episode."""
    ep_output = os.path.join(output_dir, episode["video_id"])

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
    if args.skip_visualizations:
        cmd.append("--skip_visualizations")

    print(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    return {
        "video_id": episode["video_id"],
        "dataset": episode["dataset"],
        "action_text": episode["action_text"],
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": ep_output,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch run unified pipeline on 25 SSv2 + 25 Epic-Kitchens videos",
    )
    parser.add_argument("--num_ssv2", type=int, default=25)
    parser.add_argument("--num_epic", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Batch output dir (default: unified_pipeline/batch_50_output)")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for selection (default: 123)")

    # Pipeline pass-through flags
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

    # Resume support
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing manifest, skip already-completed videos")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "batch_50_output")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Batch Run: 25 SSv2 + 25 Epic-Kitchens")
    print("=" * 80)

    # ── Resume or fresh selection ──────────────────────────────────────────
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    results_path = os.path.join(args.output_dir, "results.json")

    if args.resume and os.path.exists(manifest_path):
        print("Resuming from existing manifest...")
        with open(manifest_path) as f:
            all_episodes = json.load(f)

        # Load existing results to skip completed
        completed_ids = set()
        if os.path.exists(results_path):
            with open(results_path) as f:
                prev_results = json.load(f)
            completed_ids = {
                r["video_id"] for r in prev_results if r["returncode"] == 0
            }
        print(f"  {len(completed_ids)} already completed, skipping those.")
    else:
        # Fresh selection
        print(f"\nSelecting {args.num_ssv2} SSv2 videos (seed={args.seed})...")
        ssv2_episodes = select_ssv2(args.num_ssv2, args.seed)
        print(f"  Selected {len(ssv2_episodes)} SSv2 episodes")

        print(f"\nSelecting {args.num_epic} Epic-Kitchens episodes (seed={args.seed + 1})...")
        epic_episodes = select_epic(args.num_epic, args.seed, args.output_dir)
        print(f"  Selected {len(epic_episodes)} Epic-Kitchens episodes")

        all_episodes = ssv2_episodes + epic_episodes
        completed_ids = set()

        # Save manifest
        with open(manifest_path, "w") as f:
            json.dump(all_episodes, f, indent=2)
        print(f"\nManifest saved: {manifest_path}")

    # ── Print selection table ──────────────────────────────────────────────
    print(f"\n{'#':>3}  {'Dataset':>5}  {'Video ID':>25}  {'Hand':>5}  {'Frm':>4}  {'HF':>4}  Action")
    print("-" * 90)
    for i, ep in enumerate(all_episodes):
        skip = " [done]" if ep["video_id"] in completed_ids else ""
        print(
            f"{i+1:>3}  {ep['dataset']:>5}  {ep['video_id']:>25}  "
            f"{ep['active_hand']:>5}  {ep['num_frames']:>4}  "
            f"{ep['valid_hand_frames']:>4}  {ep['action_text']}{skip}"
        )
    print()

    # ── Extract EPIC-Kitchens clips ────────────────────────────────────────
    epic_eps = [e for e in all_episodes if e["dataset"] == "epic"
                and e["video_id"] not in completed_ids]
    if epic_eps:
        print(f"Extracting {len(epic_eps)} EPIC-Kitchens clips...")
        for ep in epic_eps:
            if os.path.exists(ep["video_path"]):
                continue
            annotation = load_vitra_annotation(ep["annotation_path"])
            ok = extract_epic_clip(annotation, ep["source_video"], ep["video_path"])
            if not ok:
                print(f"  FAILED to extract clip for {ep['video_id']}")
        print()

    # ── Run pipeline ───────────────────────────────────────────────────────
    # Load previous results (if resuming) to append to
    results = []
    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    total = len(all_episodes)
    to_run = [ep for ep in all_episodes if ep["video_id"] not in completed_ids]
    batch_start = time.time()

    for i, ep in enumerate(to_run):
        idx = all_episodes.index(ep) + 1
        print("\n" + "=" * 80)
        print(f"[{idx}/{total}] {ep['dataset'].upper()} — {ep['video_id']}: {ep['action_text']}")
        print("=" * 80)

        result = run_pipeline_on_episode(ep, args.output_dir, args)
        results.append(result)

        status = "OK" if result["returncode"] == 0 else f"FAIL (rc={result['returncode']})"
        print(f"\n  -> {status} in {result['elapsed_seconds']}s")

        # Save results incrementally so we can resume
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────
    total_elapsed = time.time() - batch_start
    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"\n{'#':>3}  {'Dataset':>5}  {'Video ID':>25}  {'Status':>6}  {'Time':>7}  Action")
    print("-" * 90)
    n_ok = sum(1 for r in results if r["returncode"] == 0)
    for i, r in enumerate(results):
        status = "OK" if r["returncode"] == 0 else "FAIL"
        print(
            f"{i+1:>3}  {r['dataset']:>5}  {r['video_id']:>25}  "
            f"{status:>6}  {r['elapsed_seconds']:>6.1f}s  {r['action_text']}"
        )

    print(f"\n{n_ok}/{len(results)} succeeded in {total_elapsed:.0f}s total")
    print(f"Output: {args.output_dir}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
