#!/usr/bin/env python3
"""
Multi-GPU parallel batch processor for the unified pipeline.

Distributes N videos across G GPUs using persistent batch workers — each GPU
loads models once and processes its N/G videos sequentially.

Input: JSON file with video directories and episode list (see example_input.json):

    {
        "ssv2_video_dir": "VITRA/20bn-something-something-v2",
        "epic_video_dir": "vitra_data/epic_videos/EPIC-KITCHENS",
        "episodes": [
            {"annotation_path": "vitra_data/ssv2/episodic_annotations/somethingsomethingv2_100000_ep_000000.npy"},
            {"annotation_path": "vitra_data/epic_annotations/epic/episodic_annotations/epic_kitchens_P01_01_ep_000229.npy"}
        ]
    }

Videos are auto-resolved from the annotation filename:
  - SSv2:  looks in ssv2_video_dir for {video_id}.webm or .mp4
  - Epic:  extracts clip from source video in epic_video_dir/{participant}/videos/{video}.MP4
  - Other: provide "video_path" explicitly in the episode entry

Usage:
    python batch_parallel.py --input example_input.json
    python batch_parallel.py --input episodes.json --output_dir ./output --gpus 0,1,2,3
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Dataset detection + video resolution
# ---------------------------------------------------------------------------

# Dataset config: (annotation prefix, default video dir relative to PROJECT_ROOT)
_DATASET_CONFIG = [
    ("somethingsomethingv2_", "ssv2",    None),
    ("epic_kitchens_",       "epic",     "vitra_data/epic_videos/EPIC-KITCHENS"),
    ("EgoExo4D_",            "egoexo4d", None),
    ("Ego4D_",               "ego4d",    None),
]


def _detect_dataset(annotation_path):
    """Determine the source dataset from an annotation filename prefix."""
    basename = os.path.basename(annotation_path)
    for prefix, dataset, _ in _DATASET_CONFIG:
        if basename.startswith(prefix):
            return dataset, prefix
    return None, ""


def _extract_video_name(annotation_path):
    """Extract the video/episode ID from an annotation filename.

    Returns (video_name, episode_id) where:
      - video_name: the source video identifier (e.g. "12345" for SSv2, "P01_01" for Epic)
      - episode_id: unique episode identifier used for output directory naming
    """
    import re
    basename = os.path.splitext(os.path.basename(annotation_path))[0]
    dataset, prefix = _detect_dataset(annotation_path)

    # Strip dataset prefix
    body = basename[len(prefix):] if prefix else basename

    # Strip _ep_NNNNNN suffix to get video name
    m = re.match(r"^(.+)_ep_\d+$", body)
    video_name = m.group(1) if m else body

    # Episode ID is the full body (without dataset prefix)
    episode_id = body

    return video_name, episode_id


def _get_action_text(annotation):
    """Extract action text from a loaded VITRA annotation dict."""
    text_data = annotation.get("text", {})
    for hand in ["left", "right"]:
        entries = text_data.get(hand, [])
        if entries:
            return entries[0][0]
    return "object held in hand"


def _resolve_ssv2_video(video_name, ssv2_video_dir):
    """Find an SSv2 video file by ID."""
    if not ssv2_video_dir:
        return None
    for ext in [".webm", ".mp4"]:
        p = os.path.join(ssv2_video_dir, f"{video_name}{ext}")
        if os.path.exists(p):
            return p
    return None


def _resolve_epic_video(video_name, annotation, epic_video_dir, output_dir):
    """Find Epic-Kitchens source video and extract clip.

    Epic-Kitchens annotations reference a subclip of a long source video.
    We extract the clip to output_dir/clips/ using ffmpeg.
    """
    if not epic_video_dir:
        return None

    # Find source video: {epic_video_dir}/{participant}/videos/{video_name}.MP4
    participant = video_name.split("_")[0]  # P01_01 -> P01
    source_path = os.path.join(epic_video_dir, participant, "videos", f"{video_name}.MP4")
    if not os.path.exists(source_path):
        return None

    # Extract clip
    annot_basename = os.path.splitext(os.path.basename(
        annotation["_annotation_path"]))[0]
    clip_path = os.path.join(output_dir, "clips", f"{annot_basename}.mp4")

    if os.path.exists(clip_path):
        return clip_path

    os.makedirs(os.path.dirname(clip_path), exist_ok=True)

    vdf = annotation.get("video_decode_frame")
    if vdf is None:
        return None

    fps = 60.0  # Epic-Kitchens
    start_frame = int(vdf[0])
    num_frames = len(vdf)
    total_seconds = start_frame / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    start_time = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    cmd = [
        "ffmpeg", "-y",
        "-ss", start_time,
        "-i", source_path,
        "-frames:v", str(num_frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        clip_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: ffmpeg clip extraction failed: {result.stderr[:200]}")
        return None

    return clip_path


def resolve_episodes(input_entries, output_dir, ssv2_video_dir=None,
                     epic_video_dir=None, base_dir=None):
    """Resolve input entries to full episodes with video paths.

    Each input entry needs at minimum 'annotation_path'. If 'video_path' is
    not provided, it's auto-resolved based on the dataset detected from the
    annotation filename.

    Relative paths in entries are resolved against base_dir (defaults to cwd).
    """
    if base_dir is None:
        base_dir = os.getcwd()
    # Default video directories
    if ssv2_video_dir is None:
        default = os.path.join(PROJECT_ROOT, "VITRA", "20bn-something-something-v2")
        if os.path.isdir(default):
            ssv2_video_dir = default

    if epic_video_dir is None:
        for _, dataset, rel_path in _DATASET_CONFIG:
            if dataset == "epic" and rel_path:
                default = os.path.join(PROJECT_ROOT, rel_path)
                if os.path.isdir(default):
                    epic_video_dir = default
                break

    episodes = []
    for entry in input_entries:
        annot_path = entry["annotation_path"]
        if not os.path.isabs(annot_path):
            annot_path = os.path.join(base_dir, annot_path)
        annot_path = os.path.abspath(annot_path)

        if not os.path.exists(annot_path):
            print(f"WARNING: Annotation not found: {annot_path}, skipping")
            continue

        dataset, _ = _detect_dataset(annot_path)
        video_name, episode_id = _extract_video_name(annot_path)

        # Load annotation for action text and clip extraction
        annotation = np.load(annot_path, allow_pickle=True).item()
        annotation["_annotation_path"] = annot_path
        action_text = _get_action_text(annotation)

        # Resolve video path
        video_path = entry.get("video_path")
        if video_path:
            if not os.path.isabs(video_path):
                video_path = os.path.join(base_dir, video_path)
            video_path = os.path.abspath(video_path)
        if video_path and os.path.exists(video_path):
            pass  # use as-is
        elif dataset == "ssv2":
            video_path = _resolve_ssv2_video(video_name, ssv2_video_dir)
            if video_path is None:
                print(f"WARNING: SSv2 video not found for {video_name}, skipping")
                continue
        elif dataset == "epic":
            video_path = _resolve_epic_video(
                video_name, annotation, epic_video_dir, output_dir)
            if video_path is None:
                print(f"WARNING: Epic video not found/extracted for {video_name}, skipping")
                continue
        else:
            if video_path is None:
                print(f"WARNING: No video_path for {dataset} annotation "
                      f"{os.path.basename(annot_path)}, skipping")
                continue
            video_path = os.path.abspath(video_path)

        episodes.append({
            "video_path": video_path,
            "annotation_path": annot_path,
            "video_name": episode_id,
            "action_text": action_text,
        })

    return episodes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU parallel batch processor for unified pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", type=str, required=True,
                        help="JSON file listing episodes to process")

    # GPU config
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs to use (default: auto-detect all)")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Batch output directory (default: unified_pipeline/batch_output)")

    # Pipeline flags
    parser.add_argument("--recon_method", type=str, default=None,
                        choices=["ttt3r", "megasam", "pi3"],
                        help="Scene reconstruction method (default: from JSON or ttt3r)")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--skip_reconstruction", action="store_true")
    parser.add_argument("--skip_scene_seg", action="store_true")
    parser.add_argument("--skip_sam3d", action="store_true")
    parser.add_argument("--skip_combine", action="store_true")
    parser.add_argument("--skip_visualizations", action="store_true")

    # Conda envs
    parser.add_argument("--ttt3r_env", type=str, default="ttt3r")
    parser.add_argument("--pi3_env", type=str, default="sam3d-objects")
    parser.add_argument("--sam3d_env", type=str, default="sam3d-objects")

    args = parser.parse_args()

    # Resolve GPUs
    from batch_utils import detect_gpu_ids
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        gpu_ids = detect_gpu_ids()

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "batch_output")
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load input JSON
    with open(args.input) as f:
        input_data = json.load(f)

    # Support both dict (with config) and bare list formats
    if isinstance(input_data, list):
        input_entries = input_data
        ssv2_video_dir = None
        epic_video_dir = None
        json_recon_method = None
    else:
        input_entries = input_data["episodes"]
        ssv2_video_dir = input_data.get("ssv2_video_dir")
        epic_video_dir = input_data.get("epic_video_dir")
        json_recon_method = input_data.get("recon_method")

    # CLI flag overrides JSON; JSON overrides default
    recon_method = args.recon_method or json_recon_method or "ttt3r"

    # Resolve relative video dirs against the input JSON's directory
    input_dir = os.path.dirname(os.path.abspath(args.input))
    if ssv2_video_dir and not os.path.isabs(ssv2_video_dir):
        ssv2_video_dir = os.path.join(input_dir, ssv2_video_dir)
    if epic_video_dir and not os.path.isabs(epic_video_dir):
        epic_video_dir = os.path.join(input_dir, epic_video_dir)

    # Resolve episodes (auto-find videos, extract Epic clips)
    print("Resolving episodes...")
    episodes = resolve_episodes(
        input_entries, args.output_dir,
        ssv2_video_dir=ssv2_video_dir,
        epic_video_dir=epic_video_dir,
        base_dir=input_dir,
    )

    if not episodes:
        print("ERROR: No videos to process.")
        sys.exit(1)

    print("=" * 80)
    print("Multi-GPU Parallel Batch Processor")
    print("=" * 80)
    print(f"Videos:  {len(episodes)}")
    print(f"Recon:   {recon_method}")
    print(f"GPUs:    {gpu_ids} ({len(gpu_ids)} total)")
    print(f"Output:  {args.output_dir}")
    print()

    # Print episode list
    print(f"{'#':>3}  {'Video':>25}  Action")
    print("-" * 70)
    for i, ep in enumerate(episodes):
        print(f"{i+1:>3}  {ep['video_name']:>25}  {ep.get('action_text', '')}")
    print()

    # Save episode manifest (human-readable)
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(episodes, f, indent=2, default=str)

    # Build batch worker manifest
    batch_manifest = []
    for ep in episodes:
        batch_manifest.append({
            "video_path": ep["video_path"],
            "output_dir": os.path.join(args.output_dir, ep["video_name"]),
            "annotation_path": ep.get("annotation_path"),
        })
    batch_manifest_path = os.path.join(args.output_dir, "batch_manifest.json")
    with open(batch_manifest_path, "w") as f:
        json.dump(batch_manifest, f, indent=2)

    # Run two-phase batch workers across GPUs
    from batch_utils import run_batch_phases, collect_batch_results_raw

    batch_start = time.time()
    run_batch_phases(
        batch_manifest_path, args.output_dir, gpu_ids=gpu_ids,
        ttt3r_env=args.ttt3r_env, sam3d_env=args.sam3d_env,
        pi3_env=args.pi3_env, recon_method=recon_method,
        frame_interval=args.frame_interval,
        skip_reconstruction=args.skip_reconstruction,
        skip_scene_seg=args.skip_scene_seg,
        skip_sam3d=args.skip_sam3d, skip_combine=args.skip_combine,
        skip_visualizations=args.skip_visualizations,
    )
    batch_elapsed = time.time() - batch_start

    # Collect results
    ttt3r_results, sam3d_results = collect_batch_results_raw(args.output_dir)
    results = []
    for ep in episodes:
        video_path = ep["video_path"]
        ttt3r_r = ttt3r_results.get(video_path, {})
        sam3d_r = sam3d_results.get(video_path, {})
        success = sam3d_r.get("success", ttt3r_r.get("success", False))
        elapsed = ttt3r_r.get("elapsed_s", 0) + sam3d_r.get("elapsed_s", 0)
        results.append({
            "video_name": ep["video_name"],
            "action_text": ep.get("action_text", ""),
            "returncode": 0 if success else 1,
            "elapsed_seconds": round(elapsed, 1),
            "output_dir": os.path.join(args.output_dir, ep["video_name"]),
        })

    # Summary
    n_ok = sum(1 for r in results if r["returncode"] == 0)
    total_video_time = sum(r["elapsed_seconds"] for r in results)

    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"\n{'#':>3}  {'Video':>25}  {'Status':>6}  {'Time':>7}  Action")
    print("-" * 80)
    for i, r in enumerate(results):
        status = "OK" if r["returncode"] == 0 else "FAIL"
        print(f"{i+1:>3}  {r['video_name']:>25}  {status:>6}  "
              f"{r['elapsed_seconds']:>6.1f}s  {r.get('action_text', '')}")

    print(f"\n{n_ok}/{len(results)} succeeded")
    print(f"Wall time:      {batch_elapsed:.1f}s ({batch_elapsed/60:.1f} min)")
    print(f"Total GPU time: {total_video_time:.1f}s ({total_video_time/60:.1f} min)")
    if batch_elapsed > 0:
        print(f"Speedup:        {total_video_time/batch_elapsed:.1f}x")
    print(f"Output:         {args.output_dir}")

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    summary = {
        "n_videos": len(results),
        "n_ok": n_ok,
        "gpus": gpu_ids,
        "wall_time_s": round(batch_elapsed, 1),
        "total_video_time_s": round(total_video_time, 1),
        "results": results,
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results:        {results_path}")

    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    import argparse
    main()
