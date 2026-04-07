#!/usr/bin/env python3
"""
Interactive 4D Reconstruction Viewer

Launches a local web server with a Three.js-based viewer showing:
- Left panel: original video frame
- Right panel: interactive 3D point cloud (rotate/pan/zoom)
- Time slider to scrub through frames
- Video selector dropdown

Expected directory layout per video:
    <results_dir>/<video_id>/depth/          — .npy depth maps (required)
    <results_dir>/<video_id>/intermediate_depth/ — color frames + camera poses

Usage:
    python visualize_results.py --results_dir batch_ssv2_output
    python visualize_results.py --results_dir batch_ssv2_output --port 8888
"""

import argparse
import http.server
import io
import json
import os
import socketserver
import struct
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import trimesh

from rescale_to_metric import load_metric_scale, _get_hand_vertices
from visualize_hand_overlay import (
    load_mano_model, generate_hand_points,
    MANO_RIGHT_PATH, MANO_LEFT_PATH, _rebase_vdf,
)


def _find_episode_dirs(results_dir):
    """Find all episode directories, handling both flat and nested layouts.

    Supports:
      - Flat: results_dir/episode_id/reconstruction/...
      - Nested: results_dir/dataset/episode_id/reconstruction/...
      - Legacy: results_dir/episode_id/intermediate_depth/...

    Returns list of (display_name, video_dir, recon_dir) tuples.
    """
    found = []
    results_dir = os.path.abspath(results_dir)

    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # Check if this entry is an episode dir (has reconstruction/ or intermediate_depth/)
        recon_dir = os.path.join(entry_path, "reconstruction")
        if not os.path.isdir(recon_dir):
            recon_dir = os.path.join(entry_path, "intermediate_depth")
        if os.path.isdir(recon_dir):
            found.append((entry, entry_path, recon_dir))
            continue

        # Otherwise check if it's a dataset subdir (e.g. epic/, ssv2/)
        for sub in sorted(os.listdir(entry_path)):
            sub_path = os.path.join(entry_path, sub)
            if not os.path.isdir(sub_path):
                continue
            recon_dir = os.path.join(sub_path, "reconstruction")
            if not os.path.isdir(recon_dir):
                recon_dir = os.path.join(sub_path, "intermediate_depth")
            if os.path.isdir(recon_dir):
                # Display as dataset/episode for clarity
                display = f"{entry}/{sub}"
                found.append((display, sub_path, recon_dir))

    return found


def scan_results(results_dir):
    """Scan results directory and build a manifest of available data."""
    results_dir = os.path.abspath(results_dir)
    videos = {}

    for entry, video_dir, recon_dir in _find_episode_dirs(results_dir):
        depth_dir = os.path.join(recon_dir, "depth")
        color_dir = os.path.join(recon_dir, "color")
        camera_dir = os.path.join(recon_dir, "camera")
        seg_dir = os.path.join(video_dir, "scene_segmentation")

        if not os.path.isdir(depth_dir) or not os.path.isdir(color_dir):
            continue

        depth_npy_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")])
        color_files = sorted([f for f in os.listdir(color_dir) if f.endswith((".png", ".jpg"))])

        if not depth_npy_files:
            continue

        # Get action text from manifest if available.
        # entry may be "epic/kitchens_P01_01_ep_000000" (nested) or just
        # "kitchens_P01_01_ep_000000" (flat). Match against both.
        action_text = entry
        episode_basename = os.path.basename(entry)
        manifest_path = os.path.join(results_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            for item in manifest:
                vn = item.get("video_name", "")
                if vn == entry or vn == episode_basename:
                    action_text = item.get("action_text", entry)
                    break

        # Frame basenames (without extension)
        depth_basenames = [os.path.splitext(f)[0] for f in depth_npy_files]

        # Scan SAM3D output for frames with object reconstructions
        sam3d_dir = os.path.join(video_dir, "sam3d_output")
        recon_frame_indices = set()
        if os.path.isdir(sam3d_dir):
            import glob as _glob
            for obj_ply in _glob.glob(os.path.join(sam3d_dir, "object_*", "frame_????.ply")):
                fname = os.path.basename(obj_ply)
                if "camera_coords" not in fname:
                    idx = int(fname.replace("frame_", "").replace(".ply", ""))
                    recon_frame_indices.add(idx)

        # Scan for VITRA annotations — prefer corrected_annotations/ in the
        # output dir (dataset-agnostic), fall back to original annotation from
        # the manifest.  The old code hardcoded SSv2 annotation paths, which
        # meant Epic Kitchen / Ego4D clips never got has_hand=True.
        #
        # hand_frame_indices uses annotation index ``vi`` (0-based), which
        # matches depth frame basenames (000000, 000001, ...) after vdf
        # rebasing.  We do NOT use raw vdf values here — those are absolute
        # source-video frame indices for Epic/Ego4D.
        hand_frame_indices = set()
        annotation_path = None
        corrected_dir = os.path.join(video_dir, "corrected_annotations")
        if os.path.isdir(corrected_dir):
            npy_files = [f for f in os.listdir(corrected_dir) if f.endswith(".npy")]
            if npy_files:
                annotation_path = os.path.join(corrected_dir, npy_files[0])
        if annotation_path is None and os.path.exists(manifest_path):
            # Try annotation_path from manifest
            for item in manifest:
                if item.get("video_name") == entry:
                    ap = item.get("annotation_path")
                    if ap and os.path.exists(ap):
                        annotation_path = ap
                    break
        if annotation_path is not None:
            try:
                ann = np.load(annotation_path, allow_pickle=True).item()
                n_frames = len(ann.get("video_decode_frame", []))
                for hand_key in ["left", "right"]:
                    if hand_key in ann:
                        kept = ann[hand_key].get("kept_frames", np.array([]))
                        for vi in range(min(n_frames, len(kept))):
                            if kept[vi]:
                                hand_frame_indices.add(vi)
            except Exception:
                pass

        # Depth .npy frames are the limiting factor
        frames = []
        for bn in depth_basenames:
            color_file = None
            for cf in color_files:
                if os.path.splitext(cf)[0] == bn:
                    color_file = cf
                    break
            # Load camera pose (c2w 4x4 matrix) and apply metric scale
            # to the translation so it matches the scaled point cloud
            pose = None
            camera_path = os.path.join(camera_dir, bn + ".npz")
            if os.path.exists(camera_path):
                cam_data = np.load(camera_path)
                if "pose" in cam_data:
                    p = cam_data["pose"].astype(float).copy()
                    frame_scale = load_metric_scale(recon_dir,
                                                    frame_idx=int(bn))
                    p[:3, 3] /= frame_scale
                    pose = p.tolist()
            frames.append({
                "basename": bn,
                "ply": bn + ".ply",  # virtual — generated on-the-fly from .npy
                "color": color_file,
                "pose_c2w": pose,
                "has_reconstruction": int(bn) in recon_frame_indices,
                "has_hand": int(bn) in hand_frame_indices,
            })

        # Reconstruction subdir name (reconstruction/ or intermediate_depth/)
        recon_subdir = os.path.basename(recon_dir)

        videos[entry] = {
            "video_id": entry,
            "action_text": action_text,
            "recon_dir": recon_subdir,
            "num_frames": len(frames),
            "num_color_frames": len(color_files),
            "frames": frames,
            "has_scene_seg": os.path.exists(os.path.join(seg_dir, "tracked.mp4")),
        }

    return videos


# ── MANO models (loaded once on first use) ──
_mano_r = None
_mano_l = None
_mano_loaded = False

def _ensure_mano():
    global _mano_r, _mano_l, _mano_loaded
    if _mano_loaded:
        return
    _mano_r = load_mano_model(MANO_RIGHT_PATH)
    _mano_l = load_mano_model(MANO_LEFT_PATH)
    _mano_loaded = True

# Cache: video_name -> (annotation_dict, annotation_path)
_annotation_cache = {}

def _get_annotation(results_dir, video_name):
    """Load and cache the corrected (or original) annotation for a video.

    video_name may be nested (e.g. "epic/kitchens_P01_01_ep_000000") or
    flat (e.g. "kitchens_P01_01_ep_000000").
    """
    if video_name in _annotation_cache:
        return _annotation_cache[video_name]

    video_dir = os.path.join(results_dir, video_name)
    episode_basename = os.path.basename(video_name)
    ann_path = None

    # Check corrected_annotations/ in the video dir
    corrected_dir = os.path.join(video_dir, "corrected_annotations")
    if os.path.isdir(corrected_dir):
        npy_files = [f for f in os.listdir(corrected_dir) if f.endswith(".npy")]
        if npy_files:
            ann_path = os.path.join(corrected_dir, npy_files[0])

    if ann_path is None:
        # Try manifest (match on full name or basename)
        manifest_path = os.path.join(results_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                for item in json.load(f):
                    vn = item.get("video_name", "")
                    if vn == video_name or vn == episode_basename:
                        ap = item.get("annotation_path")
                        if ap and os.path.exists(ap):
                            ann_path = ap
                        break

    if ann_path is None:
        _annotation_cache[video_name] = (None, None)
        return None, None

    ann = np.load(ann_path, allow_pickle=True).item()
    _annotation_cache[video_name] = (ann, ann_path)
    return ann, ann_path


class ViewerHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that serves the viewer and data files."""

    results_dir = None
    videos_manifest = None
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_viewer()
        elif path == "/api/manifest":
            self._serve_json(self.videos_manifest)
        elif path.startswith("/data/"):
            self._serve_data_file(path[6:])  # strip /data/
        else:
            self.send_error(404)

    def _send(self, content, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(content))
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _serve_viewer(self):
        viewer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")
        if not os.path.exists(viewer_path):
            self.send_error(500, "viewer.html not found")
            return
        with open(viewer_path, "rb") as f:
            content = f.read()
        self._send(content, "text/html")

    def _serve_json(self, data):
        content = json.dumps(data).encode()
        self._send(content, "application/json")

    def _serve_data_file(self, rel_path):
        # Prevent directory traversal
        full_path = os.path.normpath(os.path.join(self.results_dir, rel_path))
        if not full_path.startswith(os.path.normpath(self.results_dir)):
            self.send_error(403)
            return

        # On-the-fly PLY generation from .npy depth maps
        if full_path.endswith(".ply") and not os.path.isfile(full_path):
            content = self._generate_ply_from_npy(full_path)
            if content is not None:
                self._send(content, "application/octet-stream")
                return
            self.send_error(404)
            return

        if not os.path.isfile(full_path):
            self.send_error(404)
            return

        ext = os.path.splitext(full_path)[1].lower()
        content_types = {
            ".ply": "application/octet-stream",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".mp4": "video/mp4",
            ".json": "application/json",
            ".npy": "application/octet-stream",
        }
        ct = content_types.get(ext, "application/octet-stream")

        with open(full_path, "rb") as f:
            content = f.read()
        self._send(content, ct)

    def _generate_ply_from_npy(self, ply_path):
        """Generate a binary PLY point cloud from .npy depth + camera + color.

        Given a request for e.g. <video>/depth/000001.ply, loads:
          - <video>/depth/000001.npy          (depth map)
          - <video>/intermediate_depth/camera/000001.npz  (intrinsics + pose)
          - <video>/intermediate_depth/color/000001.png    (color image)
        and returns binary PLY bytes.

        If a VITRA annotation is available, MANO hand surface points are
        generated live and appended as colored points (green) to the scene.
        """
        basename = os.path.splitext(os.path.basename(ply_path))[0]
        depth_dir = os.path.dirname(ply_path)
        # depth_dir is <video>/reconstruction/depth/ (or <video>/intermediate_depth/depth/)
        # recon_dir is one level up from depth_dir
        recon_dir = os.path.dirname(depth_dir)
        video_dir = os.path.dirname(recon_dir)

        npy_path = os.path.join(depth_dir, basename + ".npy")
        if not os.path.isfile(npy_path):
            return None

        cam_path = os.path.join(recon_dir, "camera", basename + ".npz")
        if not os.path.isfile(cam_path):
            return None

        depth = np.load(npy_path)
        cam = np.load(cam_path)
        intrinsics = cam["intrinsics"]
        pose_c2w = cam["pose"].copy()

        # Apply metric scale (non-destructive — raw depth on disk).
        # Uses per-frame scale if available, else global scale.
        frame_idx = int(basename)
        metric_scale = load_metric_scale(recon_dir, frame_idx=frame_idx)
        depth = depth / metric_scale
        pose_c2w[:3, 3] = pose_c2w[:3, 3] / metric_scale

        # Load color image
        color = None
        for ext in (".png", ".jpg"):
            color_path = os.path.join(recon_dir, "color", basename + ext)
            if os.path.isfile(color_path):
                color = cv2.imread(color_path)
                if color is not None:
                    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    if color.shape[:2] != depth.shape[:2]:
                        color = cv2.resize(color, (depth.shape[1], depth.shape[0]))
                break

        # Unproject depth to 3D world points
        H, W = depth.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        valid = depth > 0

        # Subsample for performance (limit to ~200k points)
        valid_count = valid.sum()
        stride = max(1, int(np.sqrt(valid_count / 200000)))
        if stride > 1:
            subsample = np.zeros_like(valid)
            subsample[::stride, ::stride] = True
            valid = valid & subsample

        z = depth[valid]
        x = x_grid[valid]
        y = y_grid[valid]

        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        Z = z

        pts_cam = np.stack([X, Y, Z], axis=-1)
        R = pose_c2w[:3, :3]
        t = pose_c2w[:3, 3]
        pts_world = (R @ pts_cam.T).T + t

        if color is not None:
            colors = color[valid]
        else:
            colors = np.full((len(pts_world), 3), 180, dtype=np.uint8)

        # ── Add MANO hand points from VITRA annotation ──
        #
        # generate_hand_points() returns camera-space 3D points (from the MANO
        # mesh surface).  These are in the same camera space as the depth map's
        # unprojected points above — no intrinsics reprojection is needed.
        # Camera-space coordinates are intrinsics-independent: they represent
        # physical metric positions relative to the camera center.
        #
        # Frame indexing: depth files are named 000000..N (clip-relative).
        # For Epic/Ego4D, vdf contains absolute source-video frame indices
        # which must be rebased (subtract vdf[0]) to match clip-relative
        # depth filenames.  The annotation index ``vi`` (row in joints_camspace)
        # maps to rebased vdf[vi].
        # Use relative path from results_dir to handle nested layouts
        # (e.g. epic/kitchens_P01_01_ep_000000)
        video_rel = os.path.relpath(video_dir, self.results_dir)
        ann, ann_path = _get_annotation(self.results_dir, video_rel)
        if ann is not None:
            _ensure_mano()
            frame_idx = int(basename)

            # Map depth frame index → annotation index (vi)
            vdf = ann.get("video_decode_frame")
            if vdf is not None:
                rebased = _rebase_vdf(np.asarray(vdf), ann_path)
                vi = None
                for i, vf in enumerate(rebased):
                    if int(vf) == frame_idx:
                        vi = i
                        break
            else:
                vi = frame_idx

            if vi is not None:
                for hand_name in ["left", "right"]:
                    if hand_name not in ann:
                        continue
                    is_left = (hand_name == "left")

                    # Visible surface (red) — sampled from visibility-tested faces
                    vis_pts = generate_hand_points(
                        _mano_r, _mano_l, ann[hand_name], vi,
                        is_left=is_left, num_samples=5000,
                        recon_K=intrinsics, image_shape=depth.shape)

                    # Full surface (green) — includes occluded parts
                    all_pts = generate_hand_points(
                        _mano_r, _mano_l, ann[hand_name], vi,
                        is_left=is_left, num_samples=5000)

                    if all_pts is not None:
                        all_world = (R @ all_pts.T).T + t
                        all_colors = np.tile(
                            np.array([0, 255, 0], dtype=np.uint8),
                            (len(all_world), 1))
                        pts_world = np.vstack([pts_world, all_world])
                        colors = np.vstack([colors, all_colors])

                    if vis_pts is not None:
                        vis_world = (R @ vis_pts.T).T + t
                        vis_colors = np.tile(
                            np.array([255, 0, 0], dtype=np.uint8),
                            (len(vis_world), 1))
                        pts_world = np.vstack([pts_world, vis_world])
                        colors = np.vstack([colors, vis_colors])

        # Build binary PLY
        n = len(pts_world)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        buf = io.BytesIO()
        buf.write(header.encode("ascii"))
        pts_f32 = pts_world.astype(np.float32)
        colors_u8 = colors.astype(np.uint8)
        vertex_data = np.empty(n, dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
        ])
        vertex_data['x'] = pts_f32[:, 0]
        vertex_data['y'] = pts_f32[:, 1]
        vertex_data['z'] = pts_f32[:, 2]
        vertex_data['r'] = colors_u8[:, 0]
        vertex_data['g'] = colors_u8[:, 1]
        vertex_data['b'] = colors_u8[:, 2]
        buf.write(vertex_data.tobytes())
        return buf.getvalue()

    def log_message(self, format, *args):
        super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Interactive 4D Reconstruction Viewer")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to batch results directory (default: batch_ssv2_output)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Server port (default: 8080)")
    parser.add_argument("--no_browser", action="store_true",
                        help="Don't auto-open browser")
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_ssv2_output")

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)

    print("Scanning results...")
    videos = scan_results(args.results_dir)

    if not videos:
        print(f"ERROR: No valid results found in {args.results_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} videos:")
    for vid, info in videos.items():
        print(f"  {vid}: {info['num_frames']} frames — {info['action_text']}")

    ViewerHandler.results_dir = os.path.abspath(args.results_dir)
    ViewerHandler.videos_manifest = videos

    class ThreadedServer(http.server.ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    with ThreadedServer(("0.0.0.0", args.port), ViewerHandler) as httpd:
        url = f"http://localhost:{args.port}"
        print(f"\nViewer running at: {url}")
        print(f"(Listening on 0.0.0.0:{args.port} for port forwarding)")
        print("Press Ctrl+C to stop.\n")

        if not args.no_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
