# Unified Video-to-3D Pipeline

End-to-end pipeline that takes a video and produces per-frame 3D point clouds
with reconstructed objects and (optionally) MANO hand meshes.

```
Video + (optional) VITRA annotation
  → Step 1  Scene Reconstruction   (depth maps, camera poses)
  → Step 2  Scene Segmentation     (per-frame instance labels)
  → Step 3  Object Detection + 3D  (per-object mesh & Gaussian splat)
  → Step 4  Combine Point Cloud    (final PLY + OBJ files)
```

## Quick start

```bash
python run_pipeline.py \
    --video input.mp4 \
    --output_dir results
```

With VITRA annotation for hand-aware object detection:

```bash
python run_pipeline.py \
    --video input.mp4 \
    --output_dir results \
    --vitra_annotation annotation.npy
```

## Requirements

### Directory layout

The pipeline expects the following sibling directories relative to its parent:

```
project_root/
├── unified_pipeline/     ← this directory
├── sam-3d-objects/        ← SAM3D object tracking & reconstruction
├── TTT3R/                 ← scene depth & camera poses (default)
├── mega-sam/              ← alternative scene reconstruction (optional)
└── mano_data/              ← MANO hand models (for --vitra_annotation)
```

### Conda environments

The pipeline orchestrates work across multiple conda environments. Each step
runs as a subprocess in the appropriate env.

| Environment | Default name | Used by | Setup |
|---|---|---|---|
| SAM3D | `sam3d-objects` | Steps 2, 3, 4 | `pip install -r ../sam-3d-objects/requirements.txt` |
| TTT3R | `ttt3r` | Step 1 (default) | `pip install -r ../TTT3R/requirements.txt` |
| MegaSAM | `mega_sam` | Step 1 (alt) | `conda env create -f ../mega-sam/environment.yml` |

Override env names with `--ttt3r_env`, `--megasam_env`, `--sam3d_env`.

### Python packages (within the sam3d-objects env)

```bash
pip install -r requirements.txt
```

### Model checkpoints

| Model | Location | Download |
|---|---|---|
| TTT3R | `TTT3R/src/cut3r_512_dpt_4_64.pth` | See TTT3R repo |
| SAM3D | `sam-3d-objects/checkpoints/` | See sam-3d-objects repo |
| MANO (optional) | `mano_data/MANO_{RIGHT,LEFT}_clean.npz` | See WiLoR repo |

## Output structure

```
results/
├── ttt3r_output/           ← Step 1: depth/, color/, camera/
├── scene_segmentation/     ← Step 2: masks/, tracked.mp4, metadata.json
├── sam3d_output/           ← Step 3: per-object reconstructions
│   └── object_0/
│       ├── frame_XXXX.ply          ← Gaussian splat (world coords)
│       ├── frame_XXXX.glb          ← triangle mesh (world coords)
│       ├── frame_XXXX.obj          ← OBJ mesh export (with .mtl + texture)
│       ├── frame_XXXX_mask.npy     ← binary segmentation mask
│       └── frame_XXXX_mask.jpg     ← mask visualisation
├── ply_output/             ← Step 4: combined point clouds
│   ├── XXXX.ply                            ← main combined PLY
│   ├── XXXX_replaced_by_intersection.ply   ← scene with object swapped
│   ├── XXXX_*_depth.png                    ← depth visualisations
│   └── XXXX_*_color.png                    ← colour renderings
├── quality_flags.json      ← automated quality assessment
└── pipeline_log.json       ← execution log with timing
```

## CLI reference

### run_pipeline.py

| Flag | Description | Default |
|---|---|---|
| `--video` | Input video path | required |
| `--output_dir` | Output directory | required |
| `--vitra_annotation` | VITRA `.npy` annotation path | None |
| `--use_megasam` | Use MegaSAM instead of TTT3R | off |
| `--prompt` | Object text prompt for SAM3D | `"object held in either hand"` |
| `--frame_interval` | Process every Nth frame | 1 |
| `--device` | `cuda` or `cpu` | `cuda` |
| `--inference_steps` | Diffusion steps for 3D reconstruction | 25 |
| `--early_exit_score` | Score threshold for early detection exit | 0.8 |
| `--export_obj` / `--no_export_obj` | OBJ mesh export | on |
| `--skip_reconstruction` | Skip step 1 (reuse existing) | off |
| `--skip_scene_seg` | Skip step 2 | off |
| `--skip_sam3d` | Skip step 3 | off |
| `--skip_combine` | Skip step 4 | off |
| `--skip_visualizations` | Skip PNG rendering in step 4 | off |
| `--parallel_steps` / `--no_parallel_steps` | Run steps 1 & 2 in parallel | on |
| `--check_deformable` | LLM-based deformable object flag | off |

### generate_combined_ply.py (standalone)

Can be run independently on existing TTT3R + SAM3D outputs:

```bash
python generate_combined_ply.py \
    --ttt3r_out results/ttt3r_output \
    --object_dir results/sam3d_output \
    --output_dir results/ply_output
```

## Batch processing

For processing multiple videos in parallel:

```bash
python batch_parallel.py \
    --input_list videos.txt \
    --output_root batch_output/ \
    --num_workers 4
```

## Scripts

| Script | Purpose |
|---|---|
| `run_pipeline.py` | Main 4-step orchestrator |
| `generate_combined_ply.py` | Step 4: combine scene + hands + objects into PLY/OBJ |
| `run_megasam.py` | Step 1 alternative: MegaSAM depth + camera tracking |
| `run_scene_segmentation.py` | Step 2: SAM3-based instance segmentation |
| `undistort_clip.py` | Step 0: video undistortion for Ego4D/EgoExo4D |
| `batch_parallel.py` | Parallel batch orchestrator |
| `batch_worker.py` | Worker process for batch jobs |
| `batch_test_ssv2.py` | Batch testing on Something-Something v2 |
| `visualize_results.py` | Result visualization |
| `export_viewer.py` | Export to interactive 3D viewer |
| `render_detection_video.py` | Render detection overlay videos |
