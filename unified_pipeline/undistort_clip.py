#!/usr/bin/env python3
"""
Undistort a single video clip for Ego4D or EgoExo4D datasets.

VITRA annotations were computed on undistorted frames. This script applies
the same undistortion so that our pipeline's 3D reconstruction and hand
mesh alignment are correct.

SSv2 and Epic-Kitchens use pinhole cameras and need no undistortion.

Usage:
    python undistort_clip.py \
        --video input.mp4 \
        --annotation Ego4D_0031d268_ep_000346.npy \
        --output undistorted.mp4

    # Custom intrinsics root:
    python undistort_clip.py \
        --video input.mp4 \
        --annotation EgoExo4D_indiana_cooking_21_4_ep_000224.npy \
        --output undistorted.mp4 \
        --intrinsics_root /path/to/intrinsics
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Dataset detection from annotation filename
# ---------------------------------------------------------------------------

# Ordered so longer prefixes match first
_DATASET_PREFIXES = [
    ("somethingsomethingv2_", "ssv2"),
    ("epic_kitchens_", "epic"),
    ("EgoExo4D_", "egoexo4d"),
    ("Ego4D_", "ego4d"),
]


def detect_dataset(annotation_path):
    """Determine the source dataset from an annotation filename.

    Returns one of: "ego4d", "egoexo4d", "epic", "ssv2", or None.
    """
    basename = os.path.basename(annotation_path)
    for prefix, dataset in _DATASET_PREFIXES:
        if basename.startswith(prefix):
            return dataset
    return None


def get_video_name_from_annotation(annotation_path):
    """Extract the video UID / take-name from an annotation filename.

    Expected patterns:
        Ego4D_{uuid}_ep_NNNNNN.npy        -> {uuid}
        EgoExo4D_{take_name}_ep_NNNNNN.npy -> {take_name}
    """
    basename = os.path.splitext(os.path.basename(annotation_path))[0]

    # Strip trailing _ep_NNNNNN
    m = re.match(r"^(.+)_ep_\d+$", basename)
    if m:
        body = m.group(1)
    else:
        body = basename

    # Strip dataset prefix
    for prefix, _ in _DATASET_PREFIXES:
        if body.startswith(prefix):
            return body[len(prefix):]

    return body


# ---------------------------------------------------------------------------
# Ego4D undistortion (omnidirectional → pinhole)
# ---------------------------------------------------------------------------

def undistort_ego4d(video_path, intrinsics_path, output_path,
                    frame_range=None):
    """Undistort an Ego4D clip using omnidirectional intrinsics.

    The intrinsics .npy contains ``intrinsics_ori`` (K, D, xi) describing
    the original fisheye model and ``intrinsics_new`` (K) for the target
    pinhole model. When xi == 0 the video is already pinhole and we skip.

    If ``frame_range=(start, end)`` is given, only process frames in
    [start, end) (0-based, end-exclusive). Ego4D sources are multi-hour
    full-scale videos, so restricting to the VITRA annotation range avoids
    undistorting the entire video. Frame-accurate seeking is used via
    ``cap.set(CAP_PROP_POS_FRAMES, start)``.

    The output clip starts at the requested ``start`` frame — downstream
    code must offset frame indices accordingly. This matches Epic clip
    extraction.
    """
    intrinsics_info = np.load(intrinsics_path, allow_pickle=True).item()
    intrinsics_ori = intrinsics_info["intrinsics_ori"]
    intrinsics_new = intrinsics_info["intrinsics_new"]

    K = intrinsics_ori["K"].astype(np.float32)
    D = intrinsics_ori["D"].astype(np.float32)
    xi = np.array(intrinsics_ori["xi"]).astype(np.float32)
    new_K = intrinsics_new["K"].astype(np.float32)

    already_pinhole = (xi == 0)
    if already_pinhole:
        print(f"  xi == 0 → already pinhole; clipping to frame range without remap.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_range is not None:
        start, end = frame_range
        start = max(0, int(start))
        end = min(total, int(end))
        num_to_write = max(0, end - start)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    else:
        if already_pinhole:
            # Nothing to do — full pinhole video is fine as-is.
            cap.release()
            return None
        start, num_to_write = 0, total

    # Pre-compute remap tables (once) only if we actually need to undistort.
    if not already_pinhole:
        map1, map2 = cv2.omnidir.initUndistortRectifyMap(
            K, D, xi, np.eye(3), new_K, (width, height),
            cv2.CV_16SC2, cv2.omnidir.RECTIFY_PERSPECTIVE,
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for idx in range(num_to_write):
        ret, frame = cap.read()
        if not ret:
            break
        if already_pinhole:
            undistorted = frame
        else:
            undistorted = cv2.remap(
                frame, map1, map2,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
            )
        writer.write(undistorted)
        if (idx + 1) % 100 == 0 or idx + 1 == num_to_write:
            print(f"  Ego4D undistort: {idx + 1}/{num_to_write} frames "
                  f"(source range {start}-{start + num_to_write})", end="\r")

    cap.release()
    writer.release()
    print(f"\n  Wrote undistorted video to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# EgoExo4D undistortion (Aria fisheye → pinhole via projectaria_tools)
# ---------------------------------------------------------------------------

def undistort_egoexo4d(video_path, intrinsics_path, output_path):
    """Undistort an EgoExo4D clip using ProjectAria calibration.

    Requires ``projectaria_tools`` to be installed in the active environment.
    """
    try:
        from projectaria_tools.core import calibration
    except ImportError:
        print(
            "ERROR: projectaria_tools is required for EgoExo4D undistortion.\n"
            "  Install with: pip install projectaria-tools",
            file=sys.stderr,
        )
        sys.exit(1)

    device_calib = calibration.device_calibration_from_json(intrinsics_path)
    src_calib = device_calib.get_camera_calib("camera-rgb")
    pinhole = calibration.get_linear_camera_calibration(1408, 1408, 412.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output resolution matches pinhole target (1408x1408)
    out_w, out_h = 1408, 1408

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        undistorted = calibration.distort_by_calibration(frame, pinhole, src_calib)
        writer.write(undistorted)
        idx += 1
        if idx % 100 == 0 or idx == total:
            print(f"  EgoExo4D undistort: {idx}/{total} frames", end="\r")

    cap.release()
    writer.release()
    print(f"\n  Wrote undistorted video to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def undistort_clip(video_path, annotation_path, output_path, intrinsics_root):
    """Undistort a video clip if required by its dataset.

    Returns the path to the video that downstream steps should use:
    - The undistorted output path for Ego4D/EgoExo4D
    - The original video_path if no undistortion is needed
    """
    dataset = detect_dataset(annotation_path)

    if dataset not in ("ego4d", "egoexo4d"):
        print(f"Dataset '{dataset}' does not require undistortion.")
        return video_path

    video_name = get_video_name_from_annotation(annotation_path)

    if dataset == "ego4d":
        intrinsics_path = os.path.join(intrinsics_root, "ego4d", f"{video_name}.npy")
        if not os.path.exists(intrinsics_path):
            print(f"WARNING: Intrinsics not found at {intrinsics_path}, skipping undistortion.",
                  file=sys.stderr)
            return video_path
        # Ego4D source videos are multi-hour full-scale recordings; only
        # undistort the VITRA annotation's frame range.  The output clip
        # starts at vdf[0] — batch_worker derives the absolute frame offset
        # from the raw vdf values so filenames stay absolute.
        frame_range = None
        try:
            annot = np.load(annotation_path, allow_pickle=True).item()
            vdf = annot.get("video_decode_frame")
            if vdf is not None and len(vdf) > 0:
                vdf = np.asarray(vdf)
                frame_range = (int(vdf[0]), int(vdf[-1]) + 1)
        except Exception as e:
            print(f"  WARNING: Could not read vdf from {annotation_path}: {e}",
                  file=sys.stderr)
        result = undistort_ego4d(video_path, intrinsics_path, output_path,
                                 frame_range=frame_range)
        return result if result is not None else video_path

    elif dataset == "egoexo4d":
        intrinsics_path = os.path.join(intrinsics_root, "egoexo4d", f"{video_name}.json")
        if not os.path.exists(intrinsics_path):
            print(f"WARNING: Intrinsics not found at {intrinsics_path}, skipping undistortion.",
                  file=sys.stderr)
            return video_path
        return undistort_egoexo4d(video_path, intrinsics_path, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Undistort a video clip based on its dataset (Ego4D / EgoExo4D).",
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--annotation", required=True,
                        help="Path to VITRA annotation .npy (used to detect dataset)")
    parser.add_argument("--output", required=True,
                        help="Path for undistorted output video")
    parser.add_argument("--intrinsics_root", default=None,
                        help="Root dir containing ego4d/ and egoexo4d/ intrinsics "
                             "(default: vitra_data/intrinsics relative to project root)")
    args = parser.parse_args()

    if args.intrinsics_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.intrinsics_root = os.path.join(project_root, "vitra_data", "intrinsics")

    result_path = undistort_clip(
        args.video, args.annotation, args.output, args.intrinsics_root,
    )
    # Print the path to use — caller can capture this
    print(f"USE_VIDEO={result_path}")


if __name__ == "__main__":
    main()
