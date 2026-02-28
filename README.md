# Data Preprocessing Pipeline

End-to-end pipeline for reconstructing 3D objects from egocentric video.
Given a video and a VITRA annotation, the pipeline produces per-frame 3D point
clouds with reconstructed object meshes (OBJ) and MANO hand meshes, all in a
shared world coordinate frame.

```
Video + VITRA annotation
  → Step 0  Undistort video         (Ego4D / EgoExo4D lens correction)
  → Step 1  Scene Reconstruction    (TTT3R → depth maps + camera poses)
  → Step 2  Scene Segmentation      (SAM3 auto-mask tracking)
  → Step 3  Object Detection + 3D   (SAM 3D Objects → mesh + Gaussian splat)
  → Step 4  Combine Point Cloud     (scene + hands + objects → PLY / OBJ)
```

## Setup

### Prerequisites

- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU with >= 32 GB VRAM (A6000, A100, etc.)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/)
- Git LFS (for large files in submodules)

### One-command setup

```bash
git clone --recurse-submodules git@github.com:lifchrs/data-processing.git
cd data-processing
./setup.sh
```

This will:
1. Initialize git submodules (sam-3d-objects, TTT3R)
2. Create the `sam3d-objects` conda environment and install dependencies
3. Create the `ttt3r` conda environment and install dependencies
4. Download model checkpoints (SAM3D from HuggingFace, TTT3R from Google Drive)
5. Verify everything is in place

### Manual setup

If you prefer to set things up step by step:

#### 1. Submodules

```bash
git submodule update --init --recursive
```

#### 2. SAM3D environment

```bash
mamba env create -f sam-3d-objects/environments/default.yml
mamba activate sam3d-objects

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e 'sam-3d-objects/.[dev]'
pip install -e 'sam-3d-objects/.[p3d]'

export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e 'sam-3d-objects/.[inference]'

sam-3d-objects/patching/hydra

pip install transformers accelerate
```

#### 3. TTT3R environment

```bash
conda create -n ttt3r python=3.10 -y
conda activate ttt3r
pip install -r TTT3R/requirements.txt
```

#### 4. SAM3D checkpoints

Requires [HuggingFace access](https://huggingface.co/facebook/sam-3d-objects) —
request access, then authenticate with `huggingface-cli login`.

```bash
huggingface-cli download \
    --repo-type model \
    --local-dir sam-3d-objects/checkpoints/hf-download \
    --max-workers 1 \
    facebook/sam-3d-objects
mv sam-3d-objects/checkpoints/hf-download/checkpoints sam-3d-objects/checkpoints/hf
rm -rf sam-3d-objects/checkpoints/hf-download
```

#### 5. TTT3R checkpoint

```bash
pip install gdown
cd TTT3R/src
gdown --fuzzy "https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link"
cd ../..
```

#### 6. MANO hand models (for hand reconstruction)

Place `MANO_RIGHT_clean.npz` and `MANO_LEFT_clean.npz` in `mano_data/`.

## Usage

### Basic (no hand reconstruction)

```bash
python unified_pipeline/run_pipeline.py \
    --video input.mp4 \
    --output_dir results
```

### With VITRA annotation (hand-aware detection + hand meshes)

```bash
python unified_pipeline/run_pipeline.py \
    --video input.mp4 \
    --output_dir results \
    --vitra_annotation annotation.npy
```

### Re-run specific steps

```bash
# Skip steps 1-2, re-run object detection + reconstruction + combine
python unified_pipeline/run_pipeline.py \
    --video input.mp4 \
    --output_dir results \
    --vitra_annotation annotation.npy \
    --skip_reconstruction --skip_scene_seg
```

### Batch processing

```bash
python unified_pipeline/batch_parallel.py \
    --input_list videos.txt \
    --output_root batch_output/ \
    --num_workers 4
```

## Output structure

```
results/
├── ttt3r_output/                ← depth maps, RGB frames, camera poses
│   ├── depth/                       frame_XXXX.npy
│   ├── color/                       frame_XXXX.png
│   └── camera/                      frame_XXXX.npz (intrinsics + c2w pose)
├── scene_segmentation/          ← per-frame instance label maps
├── sam3d_output/                ← per-object 3D reconstructions
│   └── object_0/
│       ├── frame_XXXX.ply           Gaussian splat (world coords)
│       ├── frame_XXXX.glb           triangle mesh (world coords)
│       ├── frame_XXXX.obj           OBJ mesh + .mtl + texture
│       ├── frame_XXXX_mask.npy      binary segmentation mask
│       └── frame_XXXX_mask.jpg      mask visualization
├── ply_output/                  ← combined point clouds
│   ├── XXXX.ply                     scene + hands + objects
│   ├── XXXX_*_depth.png             depth visualizations
│   └── XXXX_*_color.png             color renderings
├── quality_flags.json           ← automated quality assessment
└── pipeline_log.json            ← execution log with timing
```

## Repository structure

```
data-processing/
├── README.md                    ← this file
├── setup.sh                     ← one-command setup
├── unified_pipeline/            ← pipeline orchestration code
│   ├── run_pipeline.py              main 4-step orchestrator
│   ├── generate_combined_ply.py     step 4: combine scene + hands + objects
│   ├── run_megasam.py               step 1 alt: MegaSAM wrapper
│   ├── run_scene_segmentation.py    step 2: scene segmentation
│   ├── undistort_clip.py            step 0: lens undistortion
│   ├── batch_parallel.py            parallel batch orchestrator
│   ├── batch_worker.py              batch worker process
│   └── ...
├── sam-3d-objects/               ← [submodule] SAM 3D Objects (object reconstruction)
├── TTT3R/                        ← [submodule] scene depth + camera poses
└── mano_data/                    ← MANO hand model files
```

## CLI reference

See [unified_pipeline/README.md](unified_pipeline/README.md) for full CLI
documentation, including all flags and standalone usage of individual scripts.

## Key flags

| Flag | Description | Default |
|---|---|---|
| `--video` | Input video path | required |
| `--output_dir` | Output directory | required |
| `--vitra_annotation` | VITRA `.npy` annotation (enables hand detection) | None |
| `--prompt` | Object text prompt | `"object held in either hand"` |
| `--frame_interval` | Process every Nth frame | 1 |
| `--inference_steps` | Diffusion steps (lower = faster) | 25 |
| `--export_obj` / `--no_export_obj` | OBJ mesh export | on |
| `--skip_reconstruction` | Reuse existing depth/poses | off |
| `--skip_sam3d` | Reuse existing object reconstruction | off |
| `--skip_combine` | Skip point cloud combination | off |
