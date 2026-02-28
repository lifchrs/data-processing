#!/usr/bin/env python3
"""
Scene Segmentation CLI Wrapper

Segments all visible objects in a video using VideoSceneSegmenter (SAM3-based
automatic mask generation + multi-object tracking).

Output:
    scene_segmentation/
    ├── masks/000000.npy     # (H,W) int32 label maps (0=bg, N=obj_id)
    ├── tracked.mp4          # Visualization video
    └── metadata.json        # {num_frames, num_objects, object_ids}

Usage:
    python run_scene_segmentation.py --video input.mp4 --output_dir output/scene_segmentation
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image


def load_video(path):
    """Load video frames as list of PIL Images."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"ERROR: Failed to open video: {path}")
        sys.exit(1)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment all visible objects in a video using SAM3"
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for segmentation results")
    parser.add_argument("--sam3d_root", type=str, default=None,
                        help="Path to sam-3d-objects/ directory (default: ../sam-3d-objects relative to this script)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve sam3d_root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if args.sam3d_root is None:
        sam3d_root = os.path.join(project_root, "sam-3d-objects")
    else:
        sam3d_root = os.path.abspath(args.sam3d_root)

    sam3d_module_dir = os.path.join(sam3d_root, "sam3d_objects")
    if not os.path.isdir(sam3d_module_dir):
        print(f"ERROR: sam3d_objects module not found at {sam3d_module_dir}")
        sys.exit(1)

    # Add sam3d_objects to sys.path so we can import VideoSceneSegmenter
    sys.path.insert(0, sam3d_module_dir)
    from video_scene_segmenter import VideoSceneSegmenter

    # Validate input
    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)

    # Create output directories
    masks_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    print("=" * 60)
    print("Scene Segmentation")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print()

    # Load video
    print("Loading video...")
    frames = load_video(args.video)
    print(f"  Loaded {len(frames)} frames, size {frames[0].size}")

    # Load segmenter
    print("Loading SAM3 models...")
    segmenter = VideoSceneSegmenter(device=args.device)

    # Run segmentation with internal defaults
    t0 = time.time()
    video_seg = segmenter.segment_video(
        frames,
        keyframe_idx=0,
        min_mask_area=500,
        max_objects=30,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.85,
    )
    elapsed = time.time() - t0

    n_objects = len(video_seg.object_ids)
    n_tracked = len(video_seg.frames)
    print(f"  {n_objects} objects tracked across {n_tracked} frames in {elapsed:.1f}s")

    # Save masks as .npy per frame
    for fidx, frame_seg in video_seg.frames.items():
        label_map = frame_seg.label_map()
        np.save(os.path.join(masks_dir, f"{fidx:06d}.npy"), label_map)

    # Save tracked video visualization
    vid_out = os.path.join(args.output_dir, "tracked.mp4")
    segmenter.visualize_video(frames, video_seg, vid_out, fps=12)

    # Save metadata
    metadata = {
        "num_frames": len(frames),
        "num_objects": n_objects,
        "object_ids": video_seg.object_ids,
        "tracked_frames": n_tracked,
        "time_seconds": round(elapsed, 1),
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nScene segmentation complete!")
    print(f"  Masks: {masks_dir}")
    print(f"  Video: {vid_out}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
