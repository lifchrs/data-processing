#!/usr/bin/env python3
"""
Export batch results as a self-contained HTML viewer.

Embeds all point cloud data and images directly into a single HTML file
that can be opened locally in any browser — no server needed.

Expected directory layout per video:
    <results_dir>/<video_id>/ply_output/   — PLY point clouds (required)
    <results_dir>/<video_id>/ttt3r_output/ — color frames + camera poses

Usage:
    python export_viewer.py --results_dir batch_ssv2_output --output viewer_export.html
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np


def scan_results(results_dir):
    """Scan results directory and return video data."""
    results_dir = os.path.abspath(results_dir)
    videos = {}

    manifest_path = os.path.join(results_dir, "manifest.json")
    manifest_data = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest_data = json.load(f)

    for entry in sorted(os.listdir(results_dir)):
        video_dir = os.path.join(results_dir, entry)
        if not os.path.isdir(video_dir):
            continue

        ply_dir = os.path.join(video_dir, "ply_output")
        color_dir = os.path.join(video_dir, "ttt3r_output", "color")
        camera_dir = os.path.join(video_dir, "ttt3r_output", "camera")

        if not os.path.isdir(ply_dir) or not os.path.isdir(color_dir):
            continue

        ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith(".ply") and "_" not in f])
        color_files = sorted([f for f in os.listdir(color_dir) if f.endswith((".png", ".jpg"))])

        if not ply_files:
            continue

        action_text = entry
        for item in manifest_data:
            if item.get("video_name") == entry:
                action_text = item.get("action_text", entry)
                break

        ply_basenames = [os.path.splitext(f)[0] for f in ply_files]

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

        frames = []
        for i, bn in enumerate(ply_basenames):
            color_file = None
            for cf in color_files:
                if os.path.splitext(cf)[0] == bn:
                    color_file = cf
                    break
            # Load camera pose (c2w 4x4 matrix) if available
            camera_path = os.path.join(camera_dir, bn + ".npz")
            pose = None
            if os.path.exists(camera_path):
                cam_data = np.load(camera_path)
                if "pose" in cam_data:
                    pose = cam_data["pose"].astype(float).tolist()  # 4x4 list
            frames.append({
                "basename": bn,
                "ply_path": os.path.join(ply_dir, ply_files[i]),
                "color_path": os.path.join(color_dir, color_file) if color_file else None,
                "pose_c2w": pose,
                "has_reconstruction": int(bn) in recon_frame_indices,
            })

        videos[entry] = {
            "video_id": entry,
            "action_text": action_text,
            "frames": frames,
        }

    return videos


def encode_file_b64(path):
    """Read file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description="Export results as self-contained HTML viewer")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to batch results (default: batch_ssv2_output)")
    parser.add_argument("--output", type=str, default="viewer_export.html",
                        help="Output HTML file (default: viewer_export.html)")
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_ssv2_output")

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)

    print("Scanning results...")
    videos = scan_results(args.results_dir)

    if not videos:
        print("ERROR: No valid results found.")
        sys.exit(1)

    print(f"Found {len(videos)} videos")

    # Build embedded data: base64-encode all PLY and image files
    print("Encoding data (this may take a moment)...")
    embedded = {}
    total_size = 0

    for vid_id, info in videos.items():
        print(f"  {vid_id}: {info['action_text']}")
        vid_data = {
            "video_id": vid_id,
            "action_text": info["action_text"],
            "num_frames": len(info["frames"]),
            "frames": [],
        }
        for frame in info["frames"]:
            fd = {"basename": frame["basename"]}

            # Encode PLY
            ply_b64 = encode_file_b64(frame["ply_path"])
            fd["ply_b64"] = ply_b64
            total_size += len(ply_b64)

            # Encode color image
            if frame["color_path"] and os.path.exists(frame["color_path"]):
                ext = os.path.splitext(frame["color_path"])[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                img_b64 = encode_file_b64(frame["color_path"])
                fd["img_data_url"] = f"data:{mime};base64,{img_b64}"
                total_size += len(img_b64)
            else:
                fd["img_data_url"] = None

            # Include camera pose and reconstruction flag
            fd["pose_c2w"] = frame.get("pose_c2w")
            fd["has_reconstruction"] = frame.get("has_reconstruction", False)

            vid_data["frames"].append(fd)

        embedded[vid_id] = vid_data

    print(f"Total embedded data: {total_size / 1e6:.1f} MB")

    # Generate HTML
    html = generate_html(embedded)

    with open(args.output, "w") as f:
        f.write(html)

    file_size = os.path.getsize(args.output)
    print(f"\nExported to: {args.output} ({file_size / 1e6:.1f} MB)")
    print("Open this file directly in your browser — no server needed.")


def generate_html(embedded_data):
    """Generate self-contained HTML with all data embedded."""
    data_json = json.dumps(embedded_data)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>4D Reconstruction Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; overflow: hidden; height: 100vh; }}

  #header {{
    display: flex; align-items: center; gap: 16px; padding: 8px 16px;
    background: #16213e; border-bottom: 1px solid #0f3460; height: 48px;
  }}
  #header h1 {{ font-size: 16px; font-weight: 600; white-space: nowrap; color: #e94560; }}
  #video-select {{
    padding: 4px 8px; background: #0f3460; color: #e0e0e0; border: 1px solid #533483;
    border-radius: 4px; font-size: 13px; max-width: 500px;
  }}
  #action-label {{ font-size: 13px; color: #a0a0c0; font-style: italic; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  #loading-indicator {{
    display: none; padding: 2px 10px; background: #e94560; border-radius: 10px;
    font-size: 12px; animation: pulse 1s infinite;
  }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}

  #main {{ display: flex; height: calc(100vh - 48px - 56px); }}

  #left-panel {{ flex: 1; display: flex; flex-direction: column; border-right: 1px solid #0f3460; position: relative; background: #111; }}
  .panel-label {{
    position: absolute; top: 8px; left: 8px; z-index: 10;
    background: rgba(15,52,96,0.85); padding: 3px 10px; border-radius: 4px; font-size: 12px;
  }}
  #frame-image {{ width: 100%; height: 100%; object-fit: contain; display: block; background: #000; }}

  #right-panel {{ flex: 1; position: relative; background: #0a0a1a; }}
  #three-canvas {{ width: 100%; height: 100%; display: block; }}

  #controls {{
    display: flex; align-items: center; gap: 12px; padding: 8px 16px;
    background: #16213e; border-top: 1px solid #0f3460; height: 56px;
  }}
  #play-btn {{
    width: 36px; height: 36px; background: #e94560; border: none; border-radius: 50%;
    color: white; font-size: 16px; cursor: pointer; display: flex; align-items: center; justify-content: center;
  }}
  #play-btn:hover {{ background: #ff6b81; }}
  #time-slider {{ flex: 1; height: 6px; -webkit-appearance: none; background: #0f3460; border-radius: 3px; outline: none; }}
  #time-slider::-webkit-slider-thumb {{ -webkit-appearance: none; width: 18px; height: 18px; background: #e94560; border-radius: 50%; cursor: pointer; }}
  #time-slider::-moz-range-thumb {{ width: 18px; height: 18px; background: #e94560; border-radius: 50%; cursor: pointer; border: none; }}
  #frame-label {{ font-size: 13px; min-width: 90px; text-align: center; font-variant-numeric: tabular-nums; }}
  #point-size-group {{ display: flex; align-items: center; gap: 6px; }}
  #point-size-group label {{ font-size: 12px; color: #a0a0c0; }}
  #point-size-slider {{ width: 80px; }}
  #follow-cam-group {{ display: flex; align-items: center; gap: 4px; }}
  #follow-cam-group label {{ font-size: 12px; color: #a0a0c0; cursor: pointer; user-select: none; }}
  #follow-cam-cb {{ cursor: pointer; }}
  #recon-label {{
    display: none; position: absolute; top: 8px; right: 8px; z-index: 10;
    background: rgba(46, 204, 113, 0.9); color: #fff; padding: 4px 10px;
    border-radius: 4px; font-size: 12px; font-weight: 600;
  }}
  .help-hint {{
    position: absolute; bottom: 8px; right: 8px; z-index: 10;
    background: rgba(15,52,96,0.7); padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #888;
  }}
</style>
</head>
<body>

<div id="header">
  <h1>4D Viewer</h1>
  <select id="video-select"></select>
  <span id="action-label"></span>
  <span id="loading-indicator">Loading...</span>
</div>

<div id="main">
  <div id="left-panel">
    <div class="panel-label">Original Frame</div>
    <img id="frame-image" src="" alt="Video frame">
  </div>
  <div id="right-panel">
    <div class="panel-label">3D Reconstruction</div>
    <canvas id="three-canvas"></canvas>
    <div id="recon-label">Reconstruction present</div>
    <div class="help-hint">Drag: rotate | Scroll: zoom | Right-drag: pan | C: toggle camera follow</div>
  </div>
</div>

<div id="controls">
  <button id="play-btn">&#9654;</button>
  <input type="range" id="time-slider" min="0" max="0" value="0" step="1">
  <span id="frame-label">0 / 0</span>
  <div id="point-size-group">
    <label>Pt size</label>
    <input type="range" id="point-size-slider" min="0.5" max="8" value="2" step="0.5">
  </div>
  <div id="follow-cam-group">
    <input type="checkbox" id="follow-cam-cb" checked>
    <label for="follow-cam-cb">Follow camera</label>
  </div>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.163.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.163.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ PLYLoader }} from 'three/addons/loaders/PLYLoader.js';

// ── Embedded data ──
const DATA = {data_json};

// ── State ──
let currentVideo = null;
let currentFrameIdx = 0;
let playing = false;
let playInterval = null;
let pointCloud = null;
let pointSize = 2;
let followCamera = true;
let sceneRadius = 1;  // updated on first geometry load per video
const geometryCache = new Map();

// ── Three.js setup ──
const canvas = document.getElementById('three-canvas');
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x0a0a1a);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
camera.position.set(0, 0, 3);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.screenSpacePanning = true;

const plyLoader = new PLYLoader();

function resizeRenderer() {{
  const panel = document.getElementById('right-panel');
  const w = panel.clientWidth;
  const h = panel.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}}
window.addEventListener('resize', resizeRenderer);

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

// ── Camera pose helpers ──
// TTT3R outputs camera-to-world (c2w) poses in OpenCV convention:
//   X=right, Y=down, Z=forward
// Three.js camera convention:
//   X=right, Y=up, Z=backward
// So we extract position, forward, and up from the c2w matrix
// and set the Three.js camera accordingly.
function applyCameraPose(pose_c2w) {{
  if (!pose_c2w) return;
  // pose_c2w is a 4x4 row-major array [[r00,r01,r02,tx], ...]
  // Camera position in world = translation column
  const px = pose_c2w[0][3];
  const py = pose_c2w[1][3];
  const pz = pose_c2w[2][3];

  // Forward direction = 3rd column of rotation (camera Z = into scene in OpenCV)
  const fwdX = pose_c2w[0][2];
  const fwdY = pose_c2w[1][2];
  const fwdZ = pose_c2w[2][2];

  // Up direction = negated 2nd column (OpenCV Y is down, Three.js Y is up)
  const upX = -pose_c2w[0][1];
  const upY = -pose_c2w[1][1];
  const upZ = -pose_c2w[2][1];

  // Place camera at the estimated position
  camera.position.set(px, py, pz);
  camera.up.set(upX, upY, upZ);

  // Look target = position + forward * distance
  // Use scene radius so the orbit center is at a natural depth
  const d = sceneRadius * 0.5;
  controls.target.set(px + fwdX * d, py + fwdY * d, pz + fwdZ * d);
  controls.update();
}}

// ── UI ──
const videoSelect = document.getElementById('video-select');
const actionLabel = document.getElementById('action-label');
const frameImage = document.getElementById('frame-image');
const timeSlider = document.getElementById('time-slider');
const frameLabel = document.getElementById('frame-label');
const playBtn = document.getElementById('play-btn');
const pointSizeSlider = document.getElementById('point-size-slider');
const loadingIndicator = document.getElementById('loading-indicator');
const followCamCb = document.getElementById('follow-cam-cb');
const reconLabel = document.getElementById('recon-label');

followCamCb.addEventListener('change', () => {{
  followCamera = followCamCb.checked;
  // If just enabled, snap to current frame's pose
  if (followCamera && currentVideo) {{
    const frame = currentVideo.frames[currentFrameIdx];
    if (frame && frame.pose_c2w) applyCameraPose(frame.pose_c2w);
  }}
}});

// ── Init ──
function init() {{
  const videoIds = Object.keys(DATA);
  videoSelect.innerHTML = '';
  for (const vid of videoIds) {{
    const opt = document.createElement('option');
    opt.value = vid;
    opt.textContent = vid + ' — ' + DATA[vid].action_text;
    videoSelect.appendChild(opt);
  }}
  videoSelect.addEventListener('change', () => selectVideo(videoSelect.value));
  selectVideo(videoIds[0]);
  resizeRenderer();
}}

function selectVideo(videoId) {{
  stopPlayback();
  currentVideo = DATA[videoId];
  currentVideo._sceneComputed = false;
  currentFrameIdx = 0;
  actionLabel.textContent = currentVideo.action_text;
  timeSlider.max = currentVideo.num_frames - 1;
  timeSlider.value = 0;
  loadFrame(0);
}}

function loadFrame(idx) {{
  if (!currentVideo || idx >= currentVideo.num_frames) return;
  currentFrameIdx = idx;
  timeSlider.value = idx;
  frameLabel.textContent = (idx + 1) + ' / ' + currentVideo.num_frames;

  const frame = currentVideo.frames[idx];

  // Show image
  if (frame.img_data_url) {{
    frameImage.src = frame.img_data_url;
  }}

  // Show/hide reconstruction label
  reconLabel.style.display = frame.has_reconstruction ? 'block' : 'none';

  // Load/show point cloud
  const cacheKey = currentVideo.video_id + '/' + frame.basename;
  if (geometryCache.has(cacheKey)) {{
    showPointCloud(geometryCache.get(cacheKey), frame);
  }} else {{
    loadingIndicator.style.display = 'inline';
    // Decode base64 PLY
    const raw = atob(frame.ply_b64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const geometry = plyLoader.parse(bytes.buffer);
    geometryCache.set(cacheKey, geometry);
    showPointCloud(geometry, frame);
    loadingIndicator.style.display = 'none';
  }}
}}

function showPointCloud(geometry, frame) {{
  if (pointCloud) {{
    scene.remove(pointCloud);
  }}
  const material = new THREE.PointsMaterial({{
    size: pointSize * 0.01,
    vertexColors: true,
    sizeAttenuation: true,
  }});
  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

  // Compute scene scale on first frame of each video
  if (!currentVideo._sceneComputed) {{
    geometry.computeBoundingSphere();
    sceneRadius = geometry.boundingSphere.radius;
    currentVideo._sceneComputed = true;
  }}

  // Apply camera pose from estimated viewpoint
  if (followCamera && frame && frame.pose_c2w) {{
    applyCameraPose(frame.pose_c2w);
  }} else if (!currentVideo._fallbackCentered) {{
    // Fallback: center on bounding sphere if no pose data
    geometry.computeBoundingSphere();
    const center = geometry.boundingSphere.center;
    const radius = geometry.boundingSphere.radius;
    controls.target.copy(center);
    camera.position.set(center.x, center.y, center.z + radius * 2.5);
    controls.update();
    currentVideo._fallbackCentered = true;
  }}
}}

// ── Controls ──
timeSlider.addEventListener('input', () => loadFrame(parseInt(timeSlider.value)));

playBtn.addEventListener('click', () => {{
  if (playing) stopPlayback(); else startPlayback();
}});

function startPlayback() {{
  playing = true;
  playBtn.innerHTML = '&#9646;&#9646;';
  playInterval = setInterval(() => {{
    let next = currentFrameIdx + 1;
    if (next >= currentVideo.num_frames) next = 0;
    loadFrame(next);
  }}, 200);
}}

function stopPlayback() {{
  playing = false;
  playBtn.innerHTML = '&#9654;';
  if (playInterval) {{ clearInterval(playInterval); playInterval = null; }}
}}

pointSizeSlider.addEventListener('input', () => {{
  pointSize = parseFloat(pointSizeSlider.value);
  if (pointCloud) pointCloud.material.size = pointSize * 0.01;
}});

document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight' || e.key === 'd') {{
    stopPlayback(); loadFrame(Math.min(currentFrameIdx + 1, currentVideo.num_frames - 1));
  }} else if (e.key === 'ArrowLeft' || e.key === 'a') {{
    stopPlayback(); loadFrame(Math.max(currentFrameIdx - 1, 0));
  }} else if (e.key === ' ') {{
    e.preventDefault(); if (playing) stopPlayback(); else startPlayback();
  }} else if (e.key === 'Home') {{
    stopPlayback(); loadFrame(0);
  }} else if (e.key === 'End') {{
    stopPlayback(); loadFrame(currentVideo.num_frames - 1);
  }} else if (e.key === 'c') {{
    followCamCb.checked = !followCamCb.checked;
    followCamera = followCamCb.checked;
    if (followCamera && currentVideo) {{
      const frame = currentVideo.frames[currentFrameIdx];
      if (frame && frame.pose_c2w) applyCameraPose(frame.pose_c2w);
    }}
  }}
}});

init();
</script>
</body>
</html>'''


if __name__ == "__main__":
    main()
