#!/usr/bin/env python3
"""
Render detection visualization videos.

Takes SAM3D detection masks + original video and produces overlay MP4s showing
what objects were detected for reconstruction.

Also renders scene segmentation overlays if available.

Usage:
    python render_detection_video.py --video input.mp4 --results_dir output/
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image


# Colors for detected objects
COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
]


def load_video_frames(video_path):
    """Load all frames from a video as numpy arrays (RGB)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 12
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def render_mask_overlay(frame_rgb, mask, color, alpha=0.45, outline=True):
    """Overlay a colored mask on a frame."""
    vis = frame_rgb.copy()
    overlay = np.zeros_like(vis)
    overlay[mask] = color
    vis = cv2.addWeighted(vis, 1.0, overlay, alpha, 0)

    if outline:
        # Draw contour
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.drawContours(vis_bgr, contours, -1, color[::-1], 2)
        vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    return vis


def render_detection_video(video_path, sam3d_dir, output_path, fps=12):
    """
    Render a video showing SAM3D detected object masks overlaid on original frames.

    Looks for mask files in sam3d_dir/object_*/frame_XXXX_mask.npy
    """
    frames, video_fps = load_video_frames(video_path)
    fps = video_fps if video_fps > 0 else fps
    H, W = frames[0].shape[:2]

    # Find all object subdirectories
    obj_dirs = sorted(glob.glob(os.path.join(sam3d_dir, "object_*")))
    if not obj_dirs:
        # Flat structure — masks directly in sam3d_dir
        obj_dirs = [sam3d_dir]

    # Collect masks: {frame_idx: [(obj_id, mask), ...]}
    masks_by_frame = {}
    for obj_idx, obj_dir in enumerate(obj_dirs):
        mask_files = sorted(glob.glob(os.path.join(obj_dir, "frame_*_mask.npy")))
        for mf in mask_files:
            # Parse frame index from filename: frame_XXXX_mask.npy
            basename = os.path.basename(mf)
            frame_idx = int(basename.split("_")[1])
            mask = np.load(mf)
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask = mask.astype(bool)
            if frame_idx not in masks_by_frame:
                masks_by_frame[frame_idx] = []
            masks_by_frame[frame_idx].append((obj_idx, mask))

    if not masks_by_frame:
        print(f"  No detection masks found in {sam3d_dir}")
        return False

    detected_frames = sorted(masks_by_frame.keys())
    print(f"  Found masks for {len(detected_frames)} frames, "
          f"{len(obj_dirs)} object(s)")

    # Render video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for fidx, frame in enumerate(frames):
        vis = frame.copy()

        if fidx in masks_by_frame:
            for obj_id, mask in masks_by_frame[fidx]:
                color = COLORS[obj_id % len(COLORS)]
                vis = render_mask_overlay(vis, mask, color)

            # Draw "DETECTED" label
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.putText(vis_bgr, f"Frame {fidx}: {len(masks_by_frame[fidx])} object(s)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Dim undetected frames slightly
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.putText(vis_bgr, f"Frame {fidx}: no detection",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"  Saved: {output_path}")
    return True


def render_scene_seg_from_masks(video_path, seg_dir, output_path, fps=12):
    """
    Render scene segmentation overlay from saved mask .npy files.

    Uses masks from seg_dir/masks/XXXXXX.npy (int32 label maps).
    """
    frames, video_fps = load_video_frames(video_path)
    fps = video_fps if video_fps > 0 else fps
    H, W = frames[0].shape[:2]

    masks_dir = os.path.join(seg_dir, "masks")
    if not os.path.isdir(masks_dir):
        print(f"  No scene seg masks directory: {masks_dir}")
        return False

    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.npy")))
    if not mask_files:
        print(f"  No mask files found in {masks_dir}")
        return False

    # Load metadata for object count
    metadata_path = os.path.join(seg_dir, "metadata.json")
    num_objects = 0
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        num_objects = meta.get("num_objects", 0)

    # Build frame_idx -> label_map lookup
    label_maps = {}
    for mf in mask_files:
        fidx = int(os.path.splitext(os.path.basename(mf))[0])
        label_maps[fidx] = np.load(mf)

    print(f"  Scene seg: {len(label_maps)} frames, {num_objects} objects")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for fidx, frame in enumerate(frames):
        vis = frame.copy()

        if fidx in label_maps:
            lmap = label_maps[fidx]
            if lmap.shape != (H, W):
                lmap = cv2.resize(lmap.astype(np.int32), (W, H),
                                  interpolation=cv2.INTER_NEAREST)

            overlay = np.zeros_like(vis)
            unique_ids = np.unique(lmap)
            for oid in unique_ids:
                if oid == 0:
                    continue  # background
                mask = lmap == oid
                color = COLORS[(oid - 1) % len(COLORS)]
                overlay[mask] = color

            vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

            n_objs = len(unique_ids) - (1 if 0 in unique_ids else 0)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.putText(vis_bgr, f"Frame {fidx}: {n_objs} segments",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"  Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Render detection and segmentation visualization videos"
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to original input video")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Pipeline output directory (contains sam3d_output/, scene_segmentation/)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for visualization videos (default: results_dir/visualizations)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "visualizations")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Video: {args.video}")
    print(f"Results: {args.results_dir}")
    print()

    # 1. SAM3D detection overlay
    sam3d_dir = os.path.join(args.results_dir, "sam3d_output")
    if os.path.isdir(sam3d_dir):
        print("Rendering SAM3D detection overlay...")
        det_path = os.path.join(args.output_dir, "detection_overlay.mp4")
        render_detection_video(args.video, sam3d_dir, det_path)
    else:
        print("No SAM3D output found, skipping detection overlay.")

    # 2. Scene segmentation overlay
    seg_dir = os.path.join(args.results_dir, "scene_segmentation")
    if os.path.isdir(seg_dir):
        print("\nRendering scene segmentation overlay...")
        seg_path = os.path.join(args.output_dir, "scene_seg_overlay.mp4")
        render_scene_seg_from_masks(args.video, seg_dir, seg_path)
    else:
        print("No scene segmentation output found, skipping.")

    print(f"\nDone. Visualizations in: {args.output_dir}")


if __name__ == "__main__":
    main()
