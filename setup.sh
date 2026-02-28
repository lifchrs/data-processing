#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh — one-command setup for the Object Tracking Pipeline
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "============================================================"
echo " Data Preprocessing Pipeline — Setup"
echo "============================================================"
echo ""

# ── 1. Clone submodules ──────────────────────────────────────────────────────
echo "[1/5] Initializing git submodules..."
git submodule update --init --recursive
echo "  Done."
echo ""

# ── 2. SAM3D environment ─────────────────────────────────────────────────────
echo "[2/5] Setting up sam3d-objects conda environment..."
if conda env list 2>/dev/null | grep -q "sam3d-objects"; then
    echo "  Environment 'sam3d-objects' already exists, skipping."
else
    echo "  Creating environment from sam-3d-objects/environments/default.yml..."
    mamba env create -f sam-3d-objects/environments/default.yml || \
        conda env create -f sam-3d-objects/environments/default.yml

    echo "  Installing sam3d-objects packages..."
    eval "$(conda shell.bash hook)"
    conda activate sam3d-objects

    export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
    pip install -e 'sam-3d-objects/.[dev]'
    pip install -e 'sam-3d-objects/.[p3d]'

    export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
    pip install -e 'sam-3d-objects/.[inference]'

    # Patch hydra
    if [ -x sam-3d-objects/patching/hydra ]; then
        sam-3d-objects/patching/hydra
    fi

    # Extra deps used by the pipeline
    pip install transformers accelerate

    conda deactivate
fi
echo ""

# ── 3. TTT3R environment ─────────────────────────────────────────────────────
echo "[3/5] Setting up TTT3R conda environment..."
if conda env list 2>/dev/null | grep -q "ttt3r"; then
    echo "  Environment 'ttt3r' already exists, skipping."
else
    echo "  Creating environment 'ttt3r'..."
    conda create -n ttt3r python=3.10 -y
    eval "$(conda shell.bash hook)"
    conda activate ttt3r
    pip install -r TTT3R/requirements.txt
    conda deactivate
fi
echo ""

# ── 4. Download model checkpoints ────────────────────────────────────────────
echo "[4/5] Downloading model checkpoints..."

# SAM3D checkpoints (requires HuggingFace authentication)
if [ -f sam-3d-objects/checkpoints/hf/pipeline.yaml ]; then
    echo "  SAM3D checkpoints already present, skipping."
else
    echo "  Downloading SAM3D checkpoints from HuggingFace..."
    echo "  NOTE: You must have access to facebook/sam-3d-objects on HuggingFace."
    echo "  Run 'huggingface-cli login' first if not already authenticated."
    pip install 'huggingface-hub[cli]<1.0' 2>/dev/null || true
    huggingface-cli download \
        --repo-type model \
        --local-dir sam-3d-objects/checkpoints/hf-download \
        --max-workers 1 \
        facebook/sam-3d-objects
    mv sam-3d-objects/checkpoints/hf-download/checkpoints sam-3d-objects/checkpoints/hf
    rm -rf sam-3d-objects/checkpoints/hf-download
fi

# TTT3R checkpoint
if [ -f TTT3R/src/cut3r_512_dpt_4_64.pth ]; then
    echo "  TTT3R checkpoint already present, skipping."
else
    echo "  Downloading TTT3R checkpoint..."
    pip install gdown 2>/dev/null || true
    cd TTT3R/src
    gdown --fuzzy "https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link"
    cd "$ROOT_DIR"
fi

# MANO hand models
if [ -f mano_data/MANO_RIGHT_clean.npz ]; then
    echo "  MANO models already present, skipping."
else
    echo "  WARNING: MANO hand models not found."
    echo "  Please place MANO_RIGHT_clean.npz and MANO_LEFT_clean.npz in mano_data/"
    echo "  These are required for hand-aware reconstruction with --vitra_annotation."
fi
echo ""

# ── 5. Verify ─────────────────────────────────────────────────────────────────
echo "[5/5] Verifying setup..."
OK=true

for env in sam3d-objects ttt3r; do
    if conda env list 2>/dev/null | grep -q "$env"; then
        echo "  [OK] conda env: $env"
    else
        echo "  [MISSING] conda env: $env"
        OK=false
    fi
done

for f in \
    sam-3d-objects/checkpoints/hf/pipeline.yaml \
    TTT3R/src/cut3r_512_dpt_4_64.pth \
; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
        OK=false
    fi
done

for f in \
    mano_data/MANO_RIGHT_clean.npz \
    mano_data/MANO_LEFT_clean.npz \
; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [OPTIONAL] $f (needed for --vitra_annotation)"
    fi
done

echo ""
if [ "$OK" = true ]; then
    echo "============================================================"
    echo " Setup complete! Run the pipeline with:"
    echo ""
    echo "   python unified_pipeline/run_pipeline.py \\"
    echo "       --video input.mp4 \\"
    echo "       --output_dir results"
    echo ""
    echo " With VITRA annotations:"
    echo ""
    echo "   python unified_pipeline/run_pipeline.py \\"
    echo "       --video input.mp4 \\"
    echo "       --output_dir results \\"
    echo "       --vitra_annotation annotation.npy"
    echo "============================================================"
else
    echo "============================================================"
    echo " Setup incomplete — see [MISSING] items above."
    echo "============================================================"
    exit 1
fi
