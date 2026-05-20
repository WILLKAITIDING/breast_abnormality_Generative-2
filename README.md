# Breast Abnormality Detection via Generative Models

Generative model pipeline for breast MRI abnormality detection using MAISI-based diffusion models with ControlNet conditioning. The system reconstructs masked breast MRI volumes and computes MSE-based anomaly scores to detect abnormalities.

## Project Structure

```
breast_abnormality_Generative/
├── run_train_controlnet_odelia.sh        # ControlNet: train, visualize, inference
├── run_finetune_diff_unet_odelia.sh      # Diffusion UNet: encode, train, visualize
├── train_controlnet_odelia.py            # ControlNet training & inference logic
├── finetune_diff_unet.py                 # Diffusion UNet finetuning logic
├── prepare_odelia_controlnet_datalist.py  # Datalist generator for ControlNet
├── configs/
│   ├── environment_controlnet_odelia.json
│   ├── config_controlnet_odelia.json
│   ├── config_network_rflow_odelia_controlnet.json
│   └── modality_mapping.json
├── scripts/                              # Shared Python utilities
│   ├── __init__.py
│   ├── utils.py, sample.py, transforms.py, ...
│   └── (16 files total)
├── data/script/breast_mri/
│   └── encode_odelia_unilateral.py       # ODELIA encoding/preprocessing
├── models/                               # Pretrained model weights
│   ├── autoencoder_v1.pt                 # VAE (~80 MB)
│   ├── diff_unet_3d_rflow-ct.pt         # Pretrained diffusion UNet (~2.1 GB)
│   ├── controlnet_3d_rflow-ct.pt         # Pretrained ControlNet (~275 MB)
│   ├── mask_generation_autoencoder.pt    # Mask generation VAE (~21 MB)
│   └── mask_generation_diffusion_unet.pt # Mask generation UNet (~753 MB)
├── checkpoints/                          # Finetuned checkpoints
│   ├── maisi_diff_unet/odelia_breast/v1_rflow/version_3/checkpoints/
│   │   ├── best_diff_unet_epoch344.pt    # Best diffusion UNet
│   │   └── latest_checkpoint.pt
│   └── controlnet_odelia/v1_image_cond/version_3/checkpoints/
│       └── controlnet_best_epoch43.pt    # Best ControlNet
├── datalists/
│   └── controlnet_odelia_datalist.json
├── dataset/
│   └── MAMA-MIA/
│       ├── ras_cropped_img_seg_inference/      # Preprocessed MAMA-MIA with segmentation
│       └── ras_cropped_img_rib_seg_inference/  # Preprocessed MAMA-MIA with rib segmentation
└── outputs/                              # Inference results (generated at runtime)
```

## Environment Setup

### Prerequisites

- Linux (tested on Ubuntu 20.04, kernel 5.4)
- NVIDIA GPU(s) with CUDA 12.x support
- Conda (Miniconda or Anaconda)

### Create Conda Environment

```bash
conda create -n diffusion python=3.14 -y
conda activate diffusion
```

### Install PyTorch (CUDA 12.x)

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

### Install Dependencies

```bash
pip install monai==1.5.1 nibabel==5.3.3 numpy==2.4.0 matplotlib==3.10.8 \
    scipy==1.17.0 tqdm==4.67.1 imageio==2.37.3 pandas==2.3.3 \
    torchio==1.0.0 torchmetrics==1.8.2
```

## Before Running

Edit the following variables in the `.sh` scripts to match your system:

| Variable | Description | Where |
|----------|-------------|-------|
| `CUDA_VISIBLE_DEVICES` | GPU IDs to use | Both scripts |
| `NUM_GPUS` | Number of GPUs for distributed training | Both scripts |
| `TMPDIR` | Temporary directory path | Both scripts |
| `conda activate diffusion` | Your conda environment name | Both scripts |

The ODELIA dataset paths (`DATA_DIR`, `EMB_DIR`, `PREP_DIR`, etc.) reference the original data location at `/home/zl445/abnormality_detection/dataset/ODELIA/`. Update these if running on a different machine or if the ODELIA data is stored elsewhere. These paths are only needed for training and ODELIA-specific inference modes.

## Usage

### ControlNet Script (`run_train_controlnet_odelia.sh`)

Set the `MODE` variable at the top of the script to select a mode:

| Mode | Description |
|------|-------------|
| `train` | Generate datalist + train ControlNet (requires ODELIA data + embeddings) |
| `visualize_mask` | Visualize random masking on preprocessed images |
| `preprocess_lesion` | Preprocess lesion cases for inference |
| `visualize_standalone` | Generate masked-to-reconstructed samples from a checkpoint |
| `inference` | Run lesion inference on ODELIA data |
| `inference_MAMA-MIA` | Run inference on MAMA-MIA dataset with GT segmentation overlay |

**For MAMA-MIA inference** (most common use case):

```bash
# 1. Set MODE at top of script:
MODE="inference_MAMA-MIA"

# 2. Adjust GPU and sample settings as needed, then run:
bash run_train_controlnet_odelia.sh
```

Key MAMA-MIA inference parameters:
- `MAMA_MIA_N_SAMPLES`: Number of subjects to process (default: 30)
- `MAMA_MIA_WINDOW_FRAC`: Sliding window fraction (default: 0.25, i.e. 4x4x4 = 64 passes)
- `MAMA_MIA_MODALITY`: Conditioning modality (`mri_breast_pre`, `mri_breast_post`, `mri_breast_sub`, `mri_breast_t2`)
- `INFER_STRATEGY`: `"sliding_window"` (64 passes, thorough) or `"single_pass"` (1 pass, fast)

### Diffusion UNet Script (`run_finetune_diff_unet_odelia.sh`)

| Mode | Description |
|------|-------------|
| `encode` | Encode raw NIfTIs to VAE latent embeddings (requires ODELIA data) |
| `train` | Train diffusion UNet on latent embeddings (requires ODELIA embeddings) |
| `visualize` | Generate samples from a trained checkpoint |

```bash
# Set MODE and run:
MODE="visualize"
bash run_finetune_diff_unet_odelia.sh
```

## Model Architecture and Pipeline

### Pipeline Overview

```
Input breast MRI volume (variable size, e.g. 207x221x235)
    │
    ├─── Intensity normalization (clip at 99.5th percentile, scale to [0, 1])
    │
    ├─── Trilinear resize ──→ 256 x 256 x 128 (pixel space)
    │
    ├─── VAE encoder (4x downsampling) ──→ latent embedding: 4 x 64 x 64 x 32
    │
    ├─── Random 3D cubic masking (training) or sliding window (inference)
    │
    ├─── Diffusion UNet + ControlNet ──→ denoised latent
    │        conditioned on: modality ID + pixel spacing + image conditioning
    │
    ├─── VAE decoder (sliding window, roi=[80,80,80], overlap=0.4) ──→ reconstructed volume 256x256x128
    │
    └─── MSE(original, reconstructed) ──→ per-voxel anomaly score map
```

### Model Conditioning

The diffusion UNet and ControlNet receive three types of conditioning at inference time:

**1. Modality Embedding (class label)**

An integer ID passed as `class_labels` to both the UNet and ControlNet. This tells the model which MRI sequence the volume comes from. The mapping is defined in `configs/modality_mapping.json`:

| Modality Key | ID | Description |
|---|---|---|
| `mri_breast_pre` | 20 | Pre-contrast breast MRI |
| `mri_breast_post` | 21 | Post-contrast (DCE) breast MRI |
| `mri_breast_sub` | 22 | Subtraction image (post - pre) |
| `mri_breast_t2` | 23 | T2-weighted breast MRI |

The model was trained on ODELIA data covering all 4 sequences. For inference, set `MAMA_MIA_MODALITY` in the shell script (default: `mri_breast_pre`). The UNet has `num_class_embeds=128` slots, so the modality ID indexes into a learned embedding table.

**2. Pixel Spacing (continuous)**

A 3-element float tensor `[sx, sy, sz]` representing the voxel spacing in mm, multiplied by 100 before being fed to the model (i.e., the model sees values around 70-75, not 0.7-0.75). This is passed as `spacing_tensor` to the UNet.

For MAMA-MIA inference, the spacing is hardcoded to `[0.7, 0.7, 0.75] * 100 = [70, 70, 75]`, matching the resampled dataset spacing. The UNet has `include_spacing_input=true` so it learns to condition generation on the physical scale of the volume.

**3. Image Conditioning (ControlNet)**

The ControlNet receives a single-channel 3D volume as pixel-level conditioning (`conditioning_embedding_in_channels=1`). During training, this is the original image with random cubic masks zeroed out; the ControlNet learns to reconstruct the masked regions. During inference, the full (unmasked) image is used as conditioning, so if a region is abnormal the reconstruction will differ from the input, producing high MSE.

### Latent Space

| Property | Value |
|---|---|
| VAE architecture | AutoencoderKlMaisi (MONAI) |
| Spatial downsampling | 4x per axis |
| Latent channels | 4 |
| Pixel-space input | `1 x 256 x 256 x 128` |
| Latent shape | `4 x 64 x 64 x 32` |
| Scale factor | ~1.03 (stored in diffusion UNet checkpoint as `scale_factor`) |
| VAE decode ROI | `[80, 80, 80]` with sliding window (overlap 0.4, gaussian weighting) |

### Noise Scheduler

| Property | Value |
|---|---|
| Type | Rectified Flow (`RFlowScheduler`) |
| Train timesteps | 1000 |
| Discrete timesteps | No (continuous) |
| Timestep transform | Yes |
| Scale | 1.4 |
| Inference steps | 30 (configurable via `num_inference_steps`) |

## Dataset

### MAMA-MIA (included)

Source: Duke Breast Cancer MRI dataset, preprocessed and cropped to breast region.

#### `ras_cropped_img_seg_inference/` (30 pairs, ~170 MB)

Contains 30 image + 30 segmentation label NIfTI files (60 files total):
- **Images**: `duke_XXX_0001*.nii.gz` — cropped breast MRI volumes
- **Labels**: `duke_XXX*-Segment_1-label*.nii.gz` — binary tumor segmentation masks (values: 0 = background, 1 = tumor)

| Property | Value |
|---|---|
| File format | NIfTI (`.nii.gz`), float64 |
| Orientation | **RAS** (Right-Anterior-Superior) — all volumes are reoriented to standard radiological axes |
| Pixel spacing | `[0.7, 0.7, 0.75]` mm (resampled to near-isotropic) |
| Spatial dimensions | **Variable** per subject (e.g., 207x221x235, 121x109x229, 216x184x285) |
| Intensity range | Raw (unnormalized), varies per subject (e.g., 0-455, 0-3086, 0-9197) |
| Normalization | Applied at inference time: clip at 99.5th percentile, scale to [0, 1] |
| Resize at inference | Trilinear interpolation to **256 x 256 x 128** before encoding |

#### `ras_cropped_img_rib_seg_inference/` (30 pairs, ~432 MB)

Same subjects but with a larger crop that includes the rib cage region:
- **Images**: `duke_XXX_0001*.nii.gz` — breast + rib cage MRI volumes
- **Labels**: `duke_XXX*-Segment_1-label*.nii.gz` — binary segmentation masks
- Spatial dimensions are larger than the seg-only crop (e.g., 231x344x239)
- Same RAS orientation and `[0.7, 0.7, 0.75]` mm spacing

### What "RAS" Means

All NIfTI volumes in this project follow the **RAS** (Right-Anterior-Superior) convention:
- **R** (x-axis): patient's Right → Left
- **A** (y-axis): patient's Anterior (front) → Posterior (back)
- **S** (z-axis): patient's feet (Inferior) → head (Superior)

This is the standard neuroimaging/radiological orientation. The affine matrix in each NIfTI header encodes the mapping from voxel indices to RAS physical coordinates (mm). All preprocessing reorients volumes to RAS before cropping and resampling, ensuring consistent orientation across subjects.

### ODELIA (not included)

The ODELIA breast MRI dataset is referenced in training and encoding modes but is not included in this package. The model was trained on non-lesion ODELIA cases (Lesion==0) with 4 MRI sequences (pre-contrast, post-contrast, subtraction, T2). If needed for training, update `DATA_DIR`, `EMB_DIR`, and `PREP_DIR` in the shell scripts.

## Inference Strategies

### Sliding Window (default, recommended)

`INFER_STRATEGY="sliding_window"` with `INFER_WINDOW_FRAC=0.25`

Divides the latent volume (64x64x32) into non-overlapping windows (16x16x8 each = 4x4x4 = 64 positions). For each window position:
1. Zeros out that window region in the conditioning image
2. Runs the full diffusion + ControlNet pipeline
3. Keeps only the reconstructed voxels within that window

This forces the model to reconstruct each region from surrounding context. Normal tissue is reconstructed accurately (low MSE); abnormal tissue deviates from the learned normal distribution (high MSE).

### Single Pass

`INFER_STRATEGY="single_pass"`

Feeds the entire unmasked image as conditioning and reconstructs the full volume in one forward pass. Faster (1 pass vs 64) but less sensitive to subtle abnormalities, since the model sees the abnormal region in its own conditioning.

## Output

Inference results are saved to `outputs/controlnet_odelia/inference_mama_mia/` and include:

- `images/`: PNG visualizations showing original, reconstructed, MSE heatmap, and GT segmentation overlay (sagittal/coronal/axial views)
- `videos/`: MP4 slice-by-slice comparisons with MSE overlay
