#!/bin/bash
# Train ControlNet with pixel-level image conditioning for ODELIA breast MRI.
#
# Modes:
#   train                - Generate datalist + train ControlNet (multi-GPU via torchrun)
#   visualize_mask       - Sample random preprocessed images, apply random_mask_3d,
#                          and save comparison PNGs so you can tune mask hyperparameters.
#   preprocess_lesion    - Preprocess N random lesion cases (intensity-norm + resize)
#   visualize_standalone - Load a trained checkpoint and generate masked->reconstructed samples
#   inference            - Load a trained checkpoint and run lesion inference (ODELIA)
#   inference_MAMA-MIA   - Run inference on MAMA-MIA dataset with GT segmentation overlay
#
# Prerequisites:
#   - VAE embeddings created in dataset/ODELIA/embeddings_unilateral/
#   - Preprocessed images created in dataset/ODELIA/preprocessed_unilateral/
#   - Finetuned diffusion UNet checkpoint available
#
# Usage:
#   bash run_train_controlnet_odelia.sh

set -euo pipefail

# ---- Mode ---- #
MODE="inference_MAMA-MIA"

# ---- GPU selection ---- #
export CUDA_VISIBLE_DEVICES=1,2
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR=/home/zl445/tmp
mkdir -p "$TMPDIR"
NUM_GPUS=2

# ---- Conda environment ---- #
eval "$(conda shell.bash hook)"
conda activate diffusion

# ---- Working directory ---- #
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---- Data paths ---- #
DATA_DIR="/home/zl445/abnormality_detection/dataset/ODELIA"
EMB_DIR="/home/zl445/abnormality_detection/dataset/ODELIA/embeddings_unilateral"
PREP_DIR="/home/zl445/abnormality_detection/dataset/ODELIA/preprocessed_unilateral"
DATALIST_FILE="./datalists/controlnet_odelia_datalist.json"

# ---- Config files ---- #
ENV_CONFIG="./configs/environment_controlnet_odelia.json"
TRAIN_CONFIG="./configs/config_controlnet_odelia.json"
MODEL_DEF="./configs/config_network_rflow_odelia_controlnet.json"

# ---- Model paths ---- #
VAE_CKPT="./models/autoencoder_v1.pt"
DIFFUSION_UNET_CKPT="./checkpoints/maisi_diff_unet/odelia_breast/v1_rflow/version_3/checkpoints/best_diff_unet_epoch344.pt"

# ---- Training hyperparameters ---- #
RUN_NAME="v1_image_cond"
CHECKPOINT_ROOT="./checkpoints/controlnet_odelia"
N_EPOCHS=500
BATCH_SIZE=4
LR=1e-4
NUM_WORKERS=4
CACHE_RATE=0

# ---- Masking hyperparameters ---- #
MASK_MAX_MASKS=3
MASK_MIN_FRAC=0.1
MASK_MAX_FRAC=0.4
MASK_P_NOMASK=0.1

# ---- Lesion inference settings ---- #
LESION_PREP_DIR="/home/zl445/abnormality_detection/dataset/ODELIA/preprocessed_unilateral_lesion"
LESION_N_SAMPLES=200
INFER_INTERVAL=50
INFER_WINDOW_FRAC=0.25
INFER_STRATEGY="sliding_window"   # "sliding_window" (64 passes, slow) or "single_pass" (1 pass, fast)
INFER_N_SAMPLES=3

# ---- Standalone visualization / inference ---- #
CONTROLNET_CKPT="./checkpoints/controlnet_odelia/v1_image_cond/version_3/checkpoints/controlnet_best_epoch43.pt"  # Set to a checkpoint path for visualize_standalone / inference modes
VIS_N_SAMPLES=5
VIS_OUTPUT_DIR="./outputs/controlnet_odelia/mask_visualization"
STANDALONE_VIS_DIR="./outputs/controlnet_odelia/visualize_standalone"
STANDALONE_INFER_DIR="./outputs/controlnet_odelia/inference_standalone"

# ---- MAMA-MIA inference ---- #
MAMA_MIA_DIR="./dataset/MAMA-MIA/ras_cropped_img_seg_inference"
MAMA_MIA_N_SAMPLES=30
MAMA_MIA_INFER_DIR="./outputs/controlnet_odelia/inference_mama_mia"
MAMA_MIA_WINDOW_FRAC=0.25  # sliding window size as fraction of each spatial dim (e.g. 0.25 → 4×4×4 = 64 passes)
MAMA_MIA_MODALITY="mri_breast_pre"  # conditioning: mri_breast_pre (20), mri_breast_post (21), mri_breast_sub (22), mri_breast_t2 (23)

# ---- Create output directories ---- #
mkdir -p "$(dirname "$DATALIST_FILE")"

# ---- Download pretrained models if needed ---- #
if [ ! -f "models/autoencoder_v1.pt" ]; then
    echo "VAE model not found. Downloading..."
    python -m scripts.download_model_data --version rflow-ct --root_dir ./ --model_only
fi

# ================================================================== #
# Run
# ================================================================== #
if [ "$MODE" = "train" ]; then
    # ---- Phase 1: Generate ControlNet datalist ---- #
    echo "=================================================================="
    echo "Phase 1: Generating ControlNet datalist..."
    echo "=================================================================="
    python prepare_odelia_controlnet_datalist.py \
        --embedding_dir "$EMB_DIR" \
        --preprocessed_dir "$PREP_DIR" \
        --output "$DATALIST_FILE"

    # ---- Phase 2: Train ControlNet with image conditioning ---- #
    echo "=================================================================="
    echo "Phase 2: Training ControlNet with image conditioning..."
    echo "  Mask: max_masks=$MASK_MAX_MASKS, frac=[$MASK_MIN_FRAC, $MASK_MAX_FRAC], p_nomask=$MASK_P_NOMASK"
    echo "=================================================================="
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29503 train_controlnet_odelia.py \
        -e "$ENV_CONFIG" \
        -c "$TRAIN_CONFIG" \
        -t "$MODEL_DEF" \
        -g "$NUM_GPUS" \
        --run_name "$RUN_NAME" \
        --checkpoint_root "$CHECKPOINT_ROOT" \
        --vae_checkpoint "$VAE_CKPT" \
        --diffusion_ckpt "$DIFFUSION_UNET_CKPT" \
        --n_epochs "$N_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --num_workers "$NUM_WORKERS" \
        --cache_rate "$CACHE_RATE" \
        --mask_max_masks "$MASK_MAX_MASKS" \
        --mask_min_frac "$MASK_MIN_FRAC" \
        --mask_max_frac "$MASK_MAX_FRAC" \
        --mask_p_nomask "$MASK_P_NOMASK" \
        --lesion_prep_dir "$LESION_PREP_DIR" \
        --infer_interval "$INFER_INTERVAL" \
        --infer_window_frac "$INFER_WINDOW_FRAC" \
        --infer_n_samples "$INFER_N_SAMPLES"

elif [ "$MODE" = "visualize_mask" ]; then
    echo "=================================================================="
    echo "Visualizing random_mask_3d on preprocessed images"
    echo "  Source:     $PREP_DIR"
    echo "  Samples:    $VIS_N_SAMPLES"
    echo "  Mask: max_masks=$MASK_MAX_MASKS, frac=[$MASK_MIN_FRAC, $MASK_MAX_FRAC]"
    echo "  Output:     $VIS_OUTPUT_DIR"
    echo "=================================================================="
    mkdir -p "$VIS_OUTPUT_DIR"

    python -c "
import sys, os, glob, random, torch, numpy as np, nibabel as nib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '.')
from train_controlnet_odelia import random_mask_3d

prep_dir = '${PREP_DIR}'
emb_dir  = '${EMB_DIR}'
out_dir  = '${VIS_OUTPUT_DIR}'
n_samples = ${VIS_N_SAMPLES}
max_masks = ${MASK_MAX_MASKS}
min_frac  = ${MASK_MIN_FRAC}
max_frac  = ${MASK_MAX_FRAC}

files = sorted(glob.glob(os.path.join(prep_dir, '*.nii.gz')))
if not files:
    print('ERROR: No .nii.gz files in', prep_dir); sys.exit(1)

random.seed(0)
chosen = random.sample(files, min(n_samples, len(files)))

for idx, fpath in enumerate(chosen):
    vol = nib.load(fpath).get_fdata().astype(np.float32)
    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # [1,1,H,W,D]

    masked, mask, info = random_mask_3d(x, max_masks=max_masks, min_frac=min_frac,
                                        max_frac=max_frac, p_nomask=0.0, return_info=True)

    orig = vol
    msk  = masked[0, 0].numpy()
    diff = np.abs(orig - msk)

    name = os.path.basename(fpath).replace('.nii.gz', '')
    emb_path = os.path.join(emb_dir, name + '_emb.nii.gz')
    has_emb = os.path.isfile(emb_path)
    if has_emb:
        emb = nib.load(emb_path).get_fdata().astype(np.float32)  # [C,H,W,D] or [H,W,D,C]
        if emb.ndim == 4 and emb.shape[-1] == 4:
            emb = np.transpose(emb, (3, 0, 1, 2))  # -> [C,H,W,D]
        elif emb.ndim == 4 and emb.shape[0] == 4:
            pass
        else:
            print(f'  Warning: unexpected embedding shape {emb.shape}, skipping latent display')
            has_emb = False

    n_rows = 4 if has_emb else 3
    h, w, d = orig.shape
    fig = plt.figure(figsize=(18, 6 * n_rows))
    gs = GridSpec(n_rows, 3, figure=fig)

    views_pixel = [
        ('Sagittal', orig[h//2,:,:], msk[h//2,:,:], diff[h//2,:,:]),
        ('Coronal',  orig[:,w//2,:], msk[:,w//2,:], diff[:,w//2,:]),
        ('Axial',    orig[:,:,d//2], msk[:,:,d//2], diff[:,:,d//2]),
    ]
    for col, (view, so, sm, sd_) in enumerate(views_pixel):
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(so.T, cmap='gray', origin='lower')
        ax0.set_title(f'Original [{h},{w},{d}] - {view}'); ax0.axis('off')
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(sm.T, cmap='gray', origin='lower')
        ax1.set_title(f'Masked - {view}'); ax1.axis('off')
        ax2 = fig.add_subplot(gs[2, col])
        ax2.imshow(sd_.T, cmap='hot', origin='lower')
        ax2.set_title(f'Mask region - {view}'); ax2.axis('off')

    if has_emb:
        C, lh, lw, ld = emb.shape
        views_latent = [
            ('Sagittal', [emb[c, lh//2, :, :] for c in range(C)]),
            ('Coronal',  [emb[c, :, lw//2, :] for c in range(C)]),
            ('Axial',    [emb[c, :, :, ld//2] for c in range(C)]),
        ]
        for col, (view, ch_slices) in enumerate(views_latent):
            ax3 = fig.add_subplot(gs[3, col])
            top = np.concatenate([ch_slices[0].T, ch_slices[1].T], axis=1)
            bot = np.concatenate([ch_slices[2].T, ch_slices[3].T], axis=1)
            grid = np.concatenate([top, bot], axis=0)
            ax3.imshow(grid, cmap='gray', origin='lower')
            ax3.set_title(f'Latent [{C},{lh},{lw},{ld}] ch0-3 - {view}')
            ax3.axis('off')

    cube_lines = []
    for ci, c in enumerate(info):
        o = c['origin']; s = c['size']; f = c['frac']
        cube_lines.append(f'  Cube {ci+1}: origin=({o[0]},{o[1]},{o[2]}), size=({s[0]},{s[1]},{s[2]}), frac=({f[0]:.2f},{f[1]:.2f},{f[2]:.2f})')
    title = f'{name}  |  n_masks={len(info)}, pixel=[{h},{w},{d}]'
    if has_emb:
        title += f', latent=[{C},{lh},{lw},{ld}]'
    title += '\n' + '\n'.join(cube_lines)
    fig.suptitle(title, fontsize=12, family='monospace', ha='center')
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.02 * (len(info) + 1)])
    save_path = os.path.join(out_dir, f'mask_vis_{idx:02d}_{name}.png')
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{idx+1}/{n_samples}] Saved: {save_path}  (latent: {\"yes\" if has_emb else \"no\"})')

print(f'Done. {n_samples} visualizations saved to {out_dir}')
"

elif [ "$MODE" = "preprocess_lesion" ]; then
    echo "=================================================================="
    echo "Preprocessing lesion cases for inference"
    echo "  Data root:  $DATA_DIR"
    echo "  Output:     $LESION_PREP_DIR"
    echo "  N samples:  $LESION_N_SAMPLES"
    echo "=================================================================="
    mkdir -p "$LESION_PREP_DIR"

    python -c "
import sys, os, random, torch, numpy as np, nibabel as nib
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '.')
sys.path.insert(0, 'data/script/breast_mri')
from encode_odelia_unilateral import collect_samples, intensity_norm

data_root = '${DATA_DIR}'
out_dir   = '${LESION_PREP_DIR}'
n_samples = ${LESION_N_SAMPLES}
target_size = (256, 256, 128)

print('Collecting all samples...')
all_samples = collect_samples(data_root)
lesion_samples = [s for s in all_samples if s['lesion'] != 0]
print(f'Total lesion samples: {len(lesion_samples)}')

random.seed(42)
chosen = random.sample(lesion_samples, min(n_samples, len(lesion_samples)))
print(f'Selected {len(chosen)} for preprocessing')

done, skipped = 0, 0
for s in tqdm(chosen, desc='Preprocessing'):
    out_name = f\"{s['uid']}_{s['sequence']}.nii.gz\"
    out_path = os.path.join(out_dir, out_name)
    if os.path.isfile(out_path):
        skipped += 1
        continue
    try:
        img = nib.load(s['nii_path'])
        data = np.asarray(img.dataobj, dtype=np.float32)
        affine = img.affine
        data = intensity_norm(data)
        orig_shape = data.shape
        scale_factors = [o / n for o, n in zip(orig_shape, target_size)]
        new_affine = affine.copy()
        for ax in range(3):
            new_affine[:3, ax] *= scale_factors[ax]
        if orig_shape != target_size:
            t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
            t = torch.nn.functional.interpolate(t, size=target_size, mode='trilinear', align_corners=False)
            data = t.squeeze().numpy()
        nib.save(nib.Nifti1Image(data, affine=new_affine), out_path)
        done += 1
    except Exception as e:
        print(f'  Error processing {s[\"uid\"]}: {e}')

print(f'Done. Processed: {done}, Skipped (exist): {skipped}, Total in dir: {done + skipped}')
"

elif [ "$MODE" = "visualize_standalone" ]; then
    echo "=================================================================="
    echo "Standalone visualization from trained ControlNet"
    echo "  Checkpoint: $CONTROLNET_CKPT"
    echo "  Source:     $PREP_DIR"
    echo "  Samples:    $VIS_N_SAMPLES"
    echo "  Mask: max_masks=$MASK_MAX_MASKS, frac=[$MASK_MIN_FRAC, $MASK_MAX_FRAC]"
    echo "  Output:     $STANDALONE_VIS_DIR"
    echo "=================================================================="
    if [ -z "$CONTROLNET_CKPT" ]; then
        echo "ERROR: Set CONTROLNET_CKPT to a trained checkpoint path."
        exit 1
    fi
    python train_controlnet_odelia.py \
        --mode visualize \
        -e "$ENV_CONFIG" \
        -c "$TRAIN_CONFIG" \
        -t "$MODEL_DEF" \
        --controlnet_ckpt "$CONTROLNET_CKPT" \
        --vae_checkpoint "$VAE_CKPT" \
        --diffusion_ckpt "$DIFFUSION_UNET_CKPT" \
        --prep_dir "$PREP_DIR" \
        --datalist_file "$DATALIST_FILE" \
        --output_dir "$STANDALONE_VIS_DIR" \
        --infer_n_samples "$VIS_N_SAMPLES" \
        --mask_max_masks "$MASK_MAX_MASKS" \
        --mask_min_frac "$MASK_MIN_FRAC" \
        --mask_max_frac "$MASK_MAX_FRAC"

elif [ "$MODE" = "inference" ]; then
    echo "=================================================================="
    echo "Standalone lesion inference from trained ControlNet"
    echo "  Checkpoint:  $CONTROLNET_CKPT"
    echo "  Lesion dir:  $LESION_PREP_DIR"
    echo "  Samples:     $INFER_N_SAMPLES"
    echo "  Strategy:    $INFER_STRATEGY"
    echo "  Window frac: $INFER_WINDOW_FRAC (only for sliding_window)"
    echo "  Output:      $STANDALONE_INFER_DIR"
    echo "=================================================================="
    if [ -z "$CONTROLNET_CKPT" ]; then
        echo "ERROR: Set CONTROLNET_CKPT to a trained checkpoint path."
        exit 1
    fi
    python train_controlnet_odelia.py \
        --mode infer \
        -e "$ENV_CONFIG" \
        -c "$TRAIN_CONFIG" \
        -t "$MODEL_DEF" \
        --controlnet_ckpt "$CONTROLNET_CKPT" \
        --vae_checkpoint "$VAE_CKPT" \
        --diffusion_ckpt "$DIFFUSION_UNET_CKPT" \
        --lesion_prep_dir "$LESION_PREP_DIR" \
        --output_dir "$STANDALONE_INFER_DIR" \
        --infer_n_samples "$INFER_N_SAMPLES" \
        --infer_window_frac "$INFER_WINDOW_FRAC" \
        --infer_strategy "$INFER_STRATEGY"

elif [ "$MODE" = "inference_MAMA-MIA" ]; then
    echo "=================================================================="
    echo "MAMA-MIA inference with ground-truth segmentation"
    echo "  Checkpoint:  $CONTROLNET_CKPT"
    echo "  Data dir:    $MAMA_MIA_DIR"
    echo "  Samples:     $MAMA_MIA_N_SAMPLES"
    echo "  Strategy:    $INFER_STRATEGY"
    echo "  Window frac: $MAMA_MIA_WINDOW_FRAC (only for sliding_window)"
    echo "  Modality:    $MAMA_MIA_MODALITY"
    echo "  Output:      $MAMA_MIA_INFER_DIR"
    echo "=================================================================="
    if [ -z "$CONTROLNET_CKPT" ]; then
        echo "ERROR: Set CONTROLNET_CKPT to a trained checkpoint path."
        exit 1
    fi
    python train_controlnet_odelia.py \
        --mode infer_mama_mia \
        -e "$ENV_CONFIG" \
        -c "$TRAIN_CONFIG" \
        -t "$MODEL_DEF" \
        --controlnet_ckpt "$CONTROLNET_CKPT" \
        --vae_checkpoint "$VAE_CKPT" \
        --diffusion_ckpt "$DIFFUSION_UNET_CKPT" \
        --mama_mia_dir "$MAMA_MIA_DIR" \
        --output_dir "$MAMA_MIA_INFER_DIR" \
        --infer_n_samples "$MAMA_MIA_N_SAMPLES" \
        --infer_strategy "$INFER_STRATEGY" \
        --infer_window_frac "$MAMA_MIA_WINDOW_FRAC" \
        --infer_modality "$MAMA_MIA_MODALITY"

else
    echo "ERROR: Unknown MODE='$MODE'. Use 'train', 'visualize_mask', 'preprocess_lesion', 'visualize_standalone', 'inference', or 'inference_MAMA-MIA'."
    exit 1
fi
