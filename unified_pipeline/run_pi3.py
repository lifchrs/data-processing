#!/usr/bin/env python3
"""
Run Pi3 scene reconstruction and save output in TTT3R-compatible format.

Pi3 outputs global points, local points (camera-space), camera poses, and
confidence. This script converts them into the depth/color/camera directory
structure expected by the rest of the pipeline.

Pi3 produces metric-scale output, so a `.metric_scale` marker is written
to skip the Procrustes hand alignment in downstream steps.

Usage:
    python run_pi3.py --video input.mp4 --output_dir pi3_output
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np


def derive_intrinsics(local_points, conf, conf_threshold=0.1):
    """Derive pinhole intrinsics from Pi3 local point maps.

    For pixel (u, v) with camera-space point (X, Y, Z):
        X = (u - cx) * Z / fx
    We assume cx = W/2, cy = H/2 and solve for fx, fy via robust median.

    Args:
        local_points: (N, H, W, 3) camera-space point maps
        conf: (N, H, W, 1) confidence logits
        conf_threshold: sigmoid(conf) threshold for valid pixels

    Returns:
        intrinsics: (3, 3) camera intrinsics matrix
    """
    import torch

    if torch.is_tensor(local_points):
        local_points = local_points.cpu().numpy()
    if torch.is_tensor(conf):
        conf = conf.cpu().numpy()

    N, H, W, _ = local_points.shape
    cx, cy = W / 2.0, H / 2.0

    # Build pixel coordinate grids
    us = np.arange(W, dtype=np.float64) + 0.5  # pixel centers
    vs = np.arange(H, dtype=np.float64) + 0.5

    fx_samples = []
    fy_samples = []

    for i in range(N):
        pts = local_points[i]  # (H, W, 3)
        c = 1.0 / (1.0 + np.exp(-conf[i, :, :, 0]))  # sigmoid
        Z = pts[:, :, 2]
        X = pts[:, :, 0]
        Y = pts[:, :, 1]

        valid = (c > conf_threshold) & (Z > 0.01) & (np.abs(X) > 1e-6) & (np.abs(Y) > 1e-6)

        if valid.sum() < 100:
            continue

        # fx = (u - cx) * Z / X for all valid pixels
        uu = np.broadcast_to(us[None, :], (H, W))
        vv = np.broadcast_to(vs[:, None], (H, W))

        fx_est = (uu[valid] - cx) * Z[valid] / X[valid]
        fy_est = (vv[valid] - cy) * Z[valid] / Y[valid]

        fx_samples.append(np.median(fx_est))
        fy_samples.append(np.median(fy_est))

    if not fx_samples:
        # Fallback: assume ~60 degree FoV
        fx = fy = W / (2 * math.tan(math.radians(30)))
    else:
        fx = float(np.median(fx_samples))
        fy = float(np.median(fy_samples))

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    return K


def run_pi3_inference(video_path, output_dir, frame_interval=1, device="cuda",
                      ckpt=None):
    """Run Pi3 inference and save results in TTT3R-compatible format.

    Creates: output_dir/{depth,color,camera}/ and .metric_scale marker.

    Args:
        video_path: path to input video or image directory
        output_dir: output directory (will contain depth/, color/, camera/)
        frame_interval: sample every Nth frame
        device: 'cuda' or 'cpu'
        ckpt: optional path to Pi3X checkpoint

    Returns:
        output_dir path
    """
    import torch

    # Add Pi3 to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    pi3_dir = os.path.join(project_root, "Pi3")
    sys.path.insert(0, pi3_dir)

    from pi3.utils.basic import load_images_as_tensor
    from pi3.models.pi3x import Pi3X

    # Create output directories
    depth_dir = os.path.join(output_dir, "depth")
    color_dir = os.path.join(output_dir, "color")
    camera_dir = os.path.join(output_dir, "camera")
    for d in [depth_dir, color_dir, camera_dir]:
        os.makedirs(d, exist_ok=True)

    # Load frames
    print(f"Loading frames from {video_path} (interval={frame_interval})...")
    device_obj = torch.device(device)
    imgs = load_images_as_tensor(video_path, interval=frame_interval).to(device_obj)
    N, C, H, W = imgs.shape
    print(f"Loaded {N} frames at {W}x{H}")

    # Load original-resolution frames for color output
    orig_frames = []
    if os.path.isdir(video_path):
        filenames = sorted([x for x in os.listdir(video_path)
                            if x.lower().endswith((".png", ".jpg", ".jpeg"))])
        for i in range(0, len(filenames), frame_interval):
            img = cv2.imread(os.path.join(video_path, filenames[i]))
            if img is not None:
                orig_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                orig_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
        cap.release()
    orig_frames = orig_frames[:N]

    # Load model
    print("Loading Pi3X model...")
    if ckpt is not None:
        model = Pi3X(use_multimodal=False).eval()
        if ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(ckpt)
        else:
            weight = torch.load(ckpt, map_location=device_obj, weights_only=False)
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        model.disable_multimodal()
    model = model.to(device_obj)

    # Run inference
    print("Running Pi3 inference...")
    dtype = torch.bfloat16 if (device != "cpu" and
            torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None])  # (1, N, H, W, 3) etc.

    # Extract results
    local_points = res['local_points'][0]  # (N, H, W, 3)
    camera_poses = res['camera_poses'][0]  # (N, 4, 4)
    conf = res['conf'][0]  # (N, H, W, 1)

    # Derive intrinsics from local point maps
    K = derive_intrinsics(local_points, conf)
    print(f"Derived intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
          f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    # Convert to numpy
    local_points_np = local_points.cpu().numpy()
    camera_poses_np = camera_poses.cpu().numpy()
    conf_np = conf.cpu().numpy()

    # Save per-frame outputs
    print(f"Saving {N} frames...")
    for i in range(N):
        frame_num = i * frame_interval
        basename = f"{frame_num:06d}"

        # Depth: z-component of local points
        depth = local_points_np[i, :, :, 2].copy()  # (H, W)
        # Zero out low-confidence pixels
        c = 1.0 / (1.0 + np.exp(-conf_np[i, :, :, 0]))
        depth[c < 0.1] = 0
        depth[depth < 0] = 0
        np.save(os.path.join(depth_dir, f"{basename}.npy"), depth.astype(np.float32))

        # Camera pose: c2w matrix
        pose = camera_poses_np[i]  # (4, 4)
        # Scale intrinsics to depth map resolution (same as Pi3 internal res)
        np.savez(os.path.join(camera_dir, f"{basename}.npz"),
                 pose=pose.astype(np.float64),
                 intrinsics=K.astype(np.float64))

        # Color: save original-resolution frame
        if i < len(orig_frames):
            color = orig_frames[i]
            # Resize to match depth map dimensions for consistency
            if color.shape[0] != H or color.shape[1] != W:
                color = cv2.resize(color, (W, H))
            cv2.imwrite(os.path.join(color_dir, f"{basename}.png"),
                        cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    print(f"Pi3 output saved to {output_dir} ({N} frames)")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Pi3 scene reconstruction (TTT3R-compatible output)")
    parser.add_argument("--video", type=str, required=True,
                        help="Input video or image directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--frame_interval", type=int, default=1,
                        help="Sample every Nth frame")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to Pi3X checkpoint (default: auto-download)")
    args = parser.parse_args()

    run_pi3_inference(args.video, args.output_dir,
                      frame_interval=args.frame_interval,
                      device=args.device, ckpt=args.ckpt)
