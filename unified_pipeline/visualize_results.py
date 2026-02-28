#!/usr/bin/env python3
"""
Interactive 4D Reconstruction Viewer

Launches a local web server with a Three.js-based viewer showing:
- Left panel: original video frame
- Right panel: interactive 3D point cloud (rotate/pan/zoom)
- Time slider to scrub through frames
- Video selector dropdown

Expected directory layout per video:
    <results_dir>/<video_id>/ply_output/   — PLY point clouds (required)
    <results_dir>/<video_id>/ttt3r_output/ — color frames + camera poses

Usage:
    python visualize_results.py --results_dir batch_ssv2_output
    python visualize_results.py --results_dir batch_ssv2_output --port 8888
"""

import argparse
import http.server
import json
import os
import socketserver
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np


def scan_results(results_dir):
    """Scan results directory and build a manifest of available data."""
    results_dir = os.path.abspath(results_dir)
    videos = {}

    for entry in sorted(os.listdir(results_dir)):
        video_dir = os.path.join(results_dir, entry)
        if not os.path.isdir(video_dir):
            continue

        ply_dir = os.path.join(video_dir, "ply_output")
        color_dir = os.path.join(video_dir, "ttt3r_output", "color")
        camera_dir = os.path.join(video_dir, "ttt3r_output", "camera")
        seg_dir = os.path.join(video_dir, "scene_segmentation")

        if not os.path.isdir(ply_dir) or not os.path.isdir(color_dir):
            continue

        ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith(".ply") and "_" not in f])
        color_files = sorted([f for f in os.listdir(color_dir) if f.endswith((".png", ".jpg"))])

        if not ply_files:
            continue

        # Get action text from manifest if available
        action_text = entry
        manifest_path = os.path.join(results_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            for item in manifest:
                if item.get("video_name") == entry:
                    action_text = item.get("action_text", entry)
                    break

        # Frame basenames (without extension) — intersection of PLY and color
        ply_basenames = [os.path.splitext(f)[0] for f in ply_files]
        color_basenames = [os.path.splitext(f)[0] for f in color_files]

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

        # Scan VITRA annotations for frames with hand models
        hand_frame_indices = set()
        vitra_candidates = [
            os.path.join(os.path.dirname(os.path.dirname(results_dir)),
                         "vitra_data", "ssv2", "episodic_annotations"),
            os.path.join(os.path.dirname(results_dir),
                         "vitra_data", "ssv2", "episodic_annotations"),
        ]
        vitra_pattern = None
        for vd in vitra_candidates:
            vp = os.path.join(vd, f"somethingsomethingv2_{entry}_ep_000000.npy")
            if os.path.exists(vp):
                vitra_pattern = vp
                break
        if vitra_pattern is not None:
            try:
                ann = np.load(vitra_pattern, allow_pickle=True).item()
                vdf = ann.get("video_decode_frame", np.array([]))
                for hand_key in ["left", "right"]:
                    if hand_key in ann:
                        kept = ann[hand_key].get("kept_frames", np.array([]))
                        for vi, vf in enumerate(vdf):
                            if vi < len(kept) and kept[vi]:
                                hand_frame_indices.add(int(vf))
            except Exception:
                pass

        # PLY frames are the limiting factor
        frames = []
        for bn in ply_basenames:
            color_file = None
            for cf in color_files:
                if os.path.splitext(cf)[0] == bn:
                    color_file = cf
                    break
            # Load camera pose (c2w 4x4 matrix) if available
            pose = None
            camera_path = os.path.join(camera_dir, bn + ".npz")
            if os.path.exists(camera_path):
                cam_data = np.load(camera_path)
                if "pose" in cam_data:
                    pose = cam_data["pose"].astype(float).tolist()
            frames.append({
                "basename": bn,
                "ply": ply_files[ply_basenames.index(bn)],
                "color": color_file,
                "pose_c2w": pose,
                "has_reconstruction": int(bn) in recon_frame_indices,
                "has_hand": int(bn) in hand_frame_indices,
            })

        videos[entry] = {
            "video_id": entry,
            "action_text": action_text,
            "num_frames": len(frames),
            "num_color_frames": len(color_files),
            "frames": frames,
            "has_scene_seg": os.path.exists(os.path.join(seg_dir, "tracked.mp4")),
        }

    return videos


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
