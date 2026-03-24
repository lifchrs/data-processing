#!/usr/bin/env python3
"""
Side-by-side hand correction comparison viewer.

Serves PLY files from <results_dir>/hand_overlay/<video_name>/ with two
synchronized 3D panels: left = original VITRA (blue), right = corrected (green).

Usage:
    python visualize_hand_comparison.py --results_dir batch_50_pi3
    python visualize_hand_comparison.py --results_dir batch_50_pi3 --port 8888
"""

import argparse
import glob
import http.server
import json
import os
import sys
import webbrowser

import numpy as np


def scan_hand_overlay(results_dir):
    """Scan hand_overlay directory and build manifest."""
    overlay_dir = os.path.join(results_dir, "hand_overlay")
    if not os.path.isdir(overlay_dir):
        return {}

    # Load manifest for action text
    manifest_path = os.path.join(results_dir, "manifest.json")
    manifest_data = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest_data = json.load(f)

    videos = {}

    for video_name in sorted(os.listdir(overlay_dir)):
        video_overlay_dir = os.path.join(overlay_dir, video_name)
        if not os.path.isdir(video_overlay_dir):
            continue

        # Find frame basenames from _original.ply files
        orig_plys = sorted(glob.glob(os.path.join(video_overlay_dir, "*_original.ply")))
        if not orig_plys:
            continue

        basenames = []
        for p in orig_plys:
            fn = os.path.basename(p)
            bn = fn.replace("_original.ply", "")
            corr = os.path.join(video_overlay_dir, f"{bn}_corrected.ply")
            if os.path.exists(corr):
                basenames.append(bn)

        if not basenames:
            continue

        # Get action text from VITRA annotation
        action_text = video_name
        for item in manifest_data:
            if item.get("video_name") == video_name or \
               os.path.basename(item.get("output_dir", "")) == video_name:
                ann_path = item.get("annotation_path")
                if ann_path and os.path.exists(ann_path):
                    try:
                        ann = np.load(ann_path, allow_pickle=True).item()
                        text_dict = ann.get("text", {})
                        for hand in ["right", "left"]:
                            if hand in text_dict and text_dict[hand]:
                                action_text = text_dict[hand][0][0]
                                break
                    except Exception:
                        pass
                if action_text == video_name:
                    action_text = item.get("action_text", item.get("prompt", video_name))
                break

        # Load camera poses for follow-cam
        video_dir = os.path.join(results_dir, video_name)
        camera_dir = os.path.join(video_dir, "intermediate_depth", "camera")

        frames = []
        for bn in basenames:
            pose = None
            cam_path = os.path.join(camera_dir, f"{bn}.npz")
            if os.path.exists(cam_path):
                try:
                    cam = np.load(cam_path)
                    if "pose" in cam:
                        pose = cam["pose"].astype(float).tolist()
                except Exception:
                    pass
            frames.append({
                "basename": bn,
                "pose_c2w": pose,
            })

        videos[video_name] = {
            "video_id": video_name,
            "action_text": action_text,
            "num_frames": len(frames),
            "frames": frames,
        }

    return videos


class ComparisonHandler(http.server.BaseHTTPRequestHandler):
    """Serves the comparison viewer and PLY files."""

    results_dir = None
    videos_manifest = None
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/" or path == "/index.html":
            self._serve_viewer()
        elif path == "/api/manifest":
            self._serve_json(self.videos_manifest)
        elif path.startswith("/data/"):
            self._serve_data_file(path[6:])
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
        viewer_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "hand_comparison_viewer.html"
        )
        if not os.path.exists(viewer_path):
            self.send_error(500, "hand_comparison_viewer.html not found")
            return
        with open(viewer_path, "rb") as f:
            content = f.read()
        self._send(content, "text/html")

    def _serve_json(self, data):
        content = json.dumps(data).encode()
        self._send(content, "application/json")

    def _serve_data_file(self, rel_path):
        # Serve from hand_overlay/<video>/<file>.ply
        overlay_dir = os.path.join(self.results_dir, "hand_overlay")
        full_path = os.path.normpath(os.path.join(overlay_dir, rel_path))
        if not full_path.startswith(os.path.normpath(overlay_dir)):
            self.send_error(403)
            return

        if not os.path.isfile(full_path):
            self.send_error(404)
            return

        with open(full_path, "rb") as f:
            content = f.read()
        self._send(content, "application/octet-stream")

    def log_message(self, format, *args):
        super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side hand correction comparison viewer")
    parser.add_argument("--results_dir", required=True,
                        help="Results directory (e.g., batch_50_pi3)")
    parser.add_argument("--port", type=int, default=8081,
                        help="Server port (default: 8081)")
    parser.add_argument("--no_browser", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: {args.results_dir} not found")
        sys.exit(1)

    print("Scanning hand overlay results...")
    videos = scan_hand_overlay(args.results_dir)

    if not videos:
        print(f"ERROR: No hand overlay data in {args.results_dir}/hand_overlay/")
        print("Run visualize_hand_overlay.py first to generate PLY files.")
        sys.exit(1)

    print(f"Found {len(videos)} videos:")
    for vid, info in videos.items():
        print(f"  {vid}: {info['num_frames']} frames — {info['action_text']}")

    ComparisonHandler.results_dir = os.path.abspath(args.results_dir)
    ComparisonHandler.videos_manifest = videos

    class ThreadedServer(http.server.ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    with ThreadedServer(("0.0.0.0", args.port), ComparisonHandler) as httpd:
        url = f"http://localhost:{args.port}"
        print(f"\nComparison viewer at: {url}")
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
