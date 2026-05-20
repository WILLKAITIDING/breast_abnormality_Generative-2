#!/bin/bash
# Finetune the MAISI Diffusion UNet (rflow-ct) on ODELIA breast MRI data.
#
# Modes:
#   encode    - Encode raw NIfTIs -> VAE latent embeddings (ROI=[320,320,160])
#   train     - Train diffusion UNet on latent embeddings
#   visualize - Generate samples from a checkpoint
#
# Conditioning: sequence ID + spacing (mri_breast_pre/post/sub/t2)
# Training data: non-lesion patients only (Lesion==0), RSH excluded
#
# Usage:
#   bash run_finetune_diff_unet_odelia.sh

set -euo pipefail

# ---- Mode: "encode", "train", or "visualize" ---- #
MODE="train"

# ---- GPU selection ---- #
export CUDA_VISIBLE_DEVICES=1,2,6,7
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR=/home/zl445/tmp
mkdir -p "$TMPDIR"
NUM_GPUS=4

# ---- Conda environment ---- #
eval "$(conda shell.bash hook)"
conda activate diffusion

# ---- Run identity ---- #
RUN_NAME="v1_rflow"
DATASET_NAME="odelia_breast"
CHECKPOINT_ROOT="./checkpoints"

# ---- Data ---- #
DATA_DIR="/home/zl445/abnormality_detection/dataset/ODELIA"
MODALITY="mri"
VAL_SPLIT=0.15

# ---- Embedding directory (pre-computed) ---- #
EMBEDDING_DIR="/home/zl445/abnormality_detection/dataset/ODELIA/embeddings_unilateral"

# ---- Preprocessed volumes (for future ControlNet conditioning) ---- #
PREPROCESSED_DIR="/home/zl445/abnormality_detection/dataset/ODELIA/preprocessed_unilateral"

# ---- VAE checkpoint (for validation visualization / decoding) ---- #
VAE_CHECKPOINT="./models/autoencoder_v1.pt"

# ---- Training hyperparameters ---- #
N_EPOCHS=500
BATCH_SIZE=8
LR=1e-5
NUM_WORKERS=4
CACHE_RATE=0

# ---- Validation / visualization ---- #
VAL_INTERVAL=10          # Generate sample every N epochs
NUM_INFERENCE_STEPS=30      # Denoising steps for sample generation
VAL_LATENT_SHAPE="64 64 32" # Latent spatial dims (256/4=64, 256/4=64, 128/4=32)
VIS_MODALITY="mri_breast_post"  # Modality to visualize (post-contrast most informative)
VIS_SPACING="0.7 0.7 0.75"     # Effective spacing after resize (matches JSON metadata)
VIS_DISPLAY_SPACING="0.7 0.7 0.75"  # Same as VIS_SPACING (isotropic-ish after resize)
N_VIS_SAMPLES=5              # Number of samples for standalone visualization

# ---- Checkpoint for standalone visualization ---- #
VIS_CKPT="./checkpoints/maisi_diff_unet/odelia_breast/v1_rflow/version_3/checkpoints/latest_checkpoint.pt"

# ---- Working directory ---- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Download pretrained models if missing ---- #
if [ ! -f "models/diff_unet_3d_rflow-ct.pt" ]; then
    echo "Pretrained diffusion UNet not found. Downloading..."
    python -m scripts.download_model_data --version rflow-ct --root_dir ./ --model_only
fi

# ---- Common args ---- #
COMMON_ARGS="--data_dir $DATA_DIR \
    --modality $MODALITY \
    --run_name $RUN_NAME \
    --checkpoint_root $CHECKPOINT_ROOT \
    --dataset_name $DATASET_NAME \
    --val_split $VAL_SPLIT \
    --num_workers $NUM_WORKERS \
    --embedding_dir $EMBEDDING_DIR \
    --vae_checkpoint $VAE_CHECKPOINT"

# ---- Encoding settings ---- #
ENCODE_GPUS="0 1 2 3"  # Logical GPU IDs (after CUDA_VISIBLE_DEVICES remapping)

# ================================================================== #
# Run
# ================================================================== #
if [ "$MODE" = "encode" ]; then
    echo "=================================================================="
    echo "Encoding ODELIA breast MRI -> VAE latent embeddings"
    echo "  Data root:     $DATA_DIR"
    echo "  Embedding dir: $EMBEDDING_DIR"
    echo "  VAE:           $VAE_CHECKPOINT"
    echo "  GPUs:          $ENCODE_GPUS"
    echo "  ROI size:      [320, 320, 160]"
    echo "=================================================================="

    python data/script/breast_mri/encode_odelia_unilateral.py \
        --data_root "$DATA_DIR" \
        --embedding_dir "$EMBEDDING_DIR" \
        --preprocessed_dir "$PREPROCESSED_DIR" \
        --config_dir ./configs \
        --vae_checkpoint "$VAE_CHECKPOINT" \
        --target_size 256 256 128 \
        --sw_batch_size 1 \
        --sw_overlap 0.4 \
        --gpus $ENCODE_GPUS \
        --val_split "$VAL_SPLIT"

elif [ "$MODE" = "train" ]; then
    echo "=================================================================="
    echo "Training diffusion UNet on ODELIA breast MRI embeddings..."
    echo "  Embedding dir: $EMBEDDING_DIR"
    echo "  GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS)"
    echo "  Epochs: $N_EPOCHS, Batch: $BATCH_SIZE, LR: $LR"
    echo "  Val interval: every $VAL_INTERVAL epochs"
    echo "  Vis modality: $VIS_MODALITY"
    echo "=================================================================="

    TRAIN_ARGS="$COMMON_ARGS \
        --train \
        --n_epochs $N_EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --cache_rate $CACHE_RATE \
        --val_interval $VAL_INTERVAL \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --val_latent_shape $VAL_LATENT_SHAPE \
        --vis_modality $VIS_MODALITY \
        --vis_spacing $VIS_SPACING \
        --vis_display_spacing $VIS_DISPLAY_SPACING"

    torchrun --nproc_per_node=$NUM_GPUS finetune_diff_unet.py $TRAIN_ARGS

elif [ "$MODE" = "visualize" ]; then
    echo "=================================================================="
    echo "Standalone visualization from checkpoint"
    echo "  Checkpoint: $VIS_CKPT"
    echo "  Samples:    $N_VIS_SAMPLES"
    echo "  Modality:   $VIS_MODALITY"
    echo "  Conditioning spacing: $VIS_SPACING"
    echo "  Display spacing:      $VIS_DISPLAY_SPACING"
    echo "=================================================================="

    VIS_ARGS="$COMMON_ARGS \
        --visualize \
        --ckpt $VIS_CKPT \
        --n_samples $N_VIS_SAMPLES \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --val_latent_shape $VAL_LATENT_SHAPE \
        --vis_modality $VIS_MODALITY \
        --vis_spacing $VIS_SPACING \
        --vis_display_spacing $VIS_DISPLAY_SPACING"

    python finetune_diff_unet.py $VIS_ARGS

else
    echo "ERROR: Unknown MODE='$MODE'. Use 'encode', 'train', or 'visualize'."
    exit 1
fi
