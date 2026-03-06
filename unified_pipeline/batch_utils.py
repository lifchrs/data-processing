#!/usr/bin/env python3
"""
Shared utilities for multi-GPU batch execution.

Used by batch_parallel.py to shard work across GPUs
and launch parallel batch workers with persistent model loading.
"""

import glob as _glob
import json
import os
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_gpu_count():
    """Detect number of available CUDA GPUs via nvidia-smi."""
    return len(detect_gpu_ids())


def detect_gpu_ids():
    """Detect available CUDA GPU indices via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return [int(idx.strip()) for idx in result.stdout.strip().splitlines() if idx.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [0]


def run_batch_phases(manifest_path, output_dir, num_gpus=1, gpu_ids=None,
                     ttt3r_env="ttt3r", sam3d_env="sam3d-objects",
                     pi3_env="pi3", recon_method="ttt3r",
                     frame_interval=1,
                     skip_reconstruction=False, skip_scene_seg=False,
                     skip_sam3d=False, skip_combine=False,
                     skip_visualizations=False):
    """Run reconstruction and SAM3D batch phases, optionally across multiple GPUs.

    Each GPU gets a shard of the manifest and loads models independently.
    Within each GPU, models are loaded once and reused across all videos.

    Args:
        gpu_ids: Explicit list of GPU indices (e.g. [0, 2, 5]).
                 If None, uses GPUs 0..num_gpus-1.
        recon_method: 'ttt3r', 'megasam', or 'pi3'.
    """
    from run_pipeline import get_python_cmd, get_env_vars

    # Resolve GPU IDs
    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))

    # Load manifest for sharding
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Shard manifest across GPUs (round-robin)
    if len(gpu_ids) > 1 and len(manifest) > 1:
        active_ids = gpu_ids[:min(len(gpu_ids), len(manifest))]
        shards = [[] for _ in range(len(active_ids))]
        for i, entry in enumerate(manifest):
            shards[i % len(active_ids)].append(entry)
        shard_info = []
        for idx, (gid, shard) in enumerate(zip(active_ids, shards)):
            if not shard:
                continue
            shard_path = os.path.join(output_dir, f"batch_manifest_gpu{gid}.json")
            with open(shard_path, "w") as f:
                json.dump(shard, f, indent=2)
            shard_info.append((gid, shard_path, len(shard)))
    else:
        shard_info = [(gpu_ids[0], manifest_path, len(manifest))]

    # Phase 1: Scene Reconstruction (TTT3R or Pi3)
    if not skip_reconstruction:
        if recon_method == "pi3":
            _launch_phase("pi3", shard_info, output_dir,
                          env_name=pi3_env, frame_interval=frame_interval,
                          get_python_cmd=get_python_cmd, get_env_vars=get_env_vars)
        else:
            _launch_phase("ttt3r", shard_info, output_dir,
                          env_name=ttt3r_env, frame_interval=frame_interval,
                          get_python_cmd=get_python_cmd, get_env_vars=get_env_vars)

    # Phase 2: SAM3D (steps 3-4)
    if not (skip_sam3d and skip_combine):
        extra_flags = []
        if skip_scene_seg:
            extra_flags.append("--skip_scene_seg")
        if skip_sam3d:
            extra_flags.append("--skip_sam3d")
        if skip_combine:
            extra_flags.append("--skip_combine")
        if skip_visualizations:
            extra_flags.append("--skip_visualizations")
        _launch_phase("sam3d", shard_info, output_dir,
                      env_name=sam3d_env, frame_interval=frame_interval,
                      get_python_cmd=get_python_cmd, get_env_vars=get_env_vars,
                      extra_flags=extra_flags)


def _launch_phase(mode, shard_info, output_dir, env_name, frame_interval,
                  get_python_cmd, get_env_vars, extra_flags=None):
    """Launch batch workers for one phase, one per GPU, and wait for all."""
    phase_labels = {"ttt3r": "1 (TTT3R)", "pi3": "1 (Pi3)", "sam3d": "2 (SAM3D)"}
    phase_label = phase_labels.get(mode, mode)
    n_gpus = len(shard_info)

    print(f"\n{'='*80}")
    print(f"PHASE {phase_label}: {n_gpus} GPU{'s' if n_gpus > 1 else ''}, "
          f"model loaded once per GPU")
    print(f"{'='*80}")

    procs = []
    for gpu_id, manifest_path, n_videos in shard_info:
        results_file = os.path.join(
            output_dir, f"batch_{mode}_results_gpu{gpu_id}.json")

        cmd = get_python_cmd(env_name) + [
            os.path.join(SCRIPT_DIR, "batch_worker.py"),
            "--mode", mode,
            "--manifest", manifest_path,
            "--frame_interval", str(frame_interval),
            "--device", "cuda",
            "--results_file", results_file,
        ]
        if extra_flags:
            cmd.extend(extra_flags)

        env = get_env_vars(env_name) or os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"  GPU {gpu_id}: {n_videos} videos")
        proc = subprocess.Popen(cmd, env=env)
        procs.append((gpu_id, proc))

    t0 = time.time()
    for gpu_id, proc in procs:
        rc = proc.wait()
        if rc != 0:
            print(f"  WARNING: GPU {gpu_id} {mode} worker failed (rc={rc})")

    elapsed = time.time() - t0
    print(f"\n{mode.upper()} phase completed in {elapsed:.1f}s")


def collect_batch_results_raw(output_dir):
    """Read all batch worker result files and return merged per-video dicts.

    Returns (ttt3r_by_video, sam3d_by_video) where each maps video_path -> result dict.
    """
    ttt3r_results = {}
    sam3d_results = {}

    # Collect reconstruction results (ttt3r or pi3)
    for mode in ["ttt3r", "pi3"]:
        for rpath in sorted(_glob.glob(
            os.path.join(output_dir, f"batch_{mode}_results*.json")
        )):
            with open(rpath) as f:
                data = json.load(f)
            for r in data.get("results", []):
                ttt3r_results[r["video"]] = r

    for rpath in sorted(_glob.glob(
        os.path.join(output_dir, "batch_sam3d_results*.json")
    )):
        with open(rpath) as f:
            data = json.load(f)
        for r in data.get("results", []):
            sam3d_results[r["video"]] = r

    return ttt3r_results, sam3d_results
