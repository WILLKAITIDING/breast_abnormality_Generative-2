#!/usr/bin/env python
"""
Finetune the MAISI Diffusion UNet (rflow-ct) on a custom CT dataset.

Two-phase pipeline adapted from train_diff_unet_tutorial.ipynb:
  Phase 1 (--create_embeddings):
      Load raw CT NIfTIs -> preprocess -> encode with VAE -> save embeddings + metadata
  Phase 2 (--train, default):
      Train diffusion UNet on the latent embeddings with RFlowScheduler

Usage:
    # Phase 1: create embeddings (single GPU)
    python finetune_diff_unet.py --create_embeddings --data_dir /path/to/nifti

    # Phase 2: train diffusion UNet (multi-GPU via torchrun)
    torchrun --nproc_per_node=2 finetune_diff_unet.py --train --data_dir /path/to/nifti
"""

import argparse
import glob
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from monai.config import print_config
from monai.data import DataLoader, partition_dataset
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.transforms import Compose
from monai.utils import first, set_determinism
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from scripts.diff_model_create_training_data import create_transforms, round_number
from scripts.utils import define_instance, dynamic_infer

import re


# ------------------------------------------------------------------ #
# Laterality helpers
# ------------------------------------------------------------------ #
def extract_ct_id(filename: str) -> str | None:
    """Extract the CT ID number from a filename like 'CT003_bone.nii.gz' -> '3'."""
    m = re.match(r"CT(\d+)", os.path.basename(filename))
    if m:
        return str(int(m.group(1)))  # "003" -> "3"
    return None


def extract_mr_id(filename: str) -> str | None:
    """Extract the MR study ID from a path like '.../MR48/...nii.gz' -> 'MR48'."""
    m = re.search(r"MR(\d+)", filename)
    if m:
        return f"MR{int(m.group(1))}"
    return None


def load_ct_metadata(ct_metadata_json: str) -> dict[str, dict]:
    """
    Load CT metadata from ct_dataset_index.json.
    Returns dict mapping CT ID (str) -> {"laterality": "left"/"right", "contrast": "with_contrast"/"without_contrast"}.
    """
    with open(ct_metadata_json, "r") as f:
        index = json.load(f)
    return index.get("ct_metadata", {})


def load_mri_metadata(mri_metadata_json: str) -> dict[str, dict]:
    """
    Load MRI metadata from MRI_vibe_laterailty.json.

    Expected format:
      {
        "MR48": {"study_id": "MR48", "laterality": "right"},
        ...
      }
    """
    with open(mri_metadata_json, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def is_contrast_ct(filename: str, ct_metadata: dict[str, dict]) -> bool:
    """Check if a CT scan is a contrast-enhanced scan."""
    ct_id = extract_ct_id(filename)
    if ct_id and ct_id in ct_metadata:
        return ct_metadata[ct_id].get("contrast", "") == "with_contrast"
    return False


def get_modality_with_laterality(filename: str, ct_metadata: dict[str, dict],
                                 base_modality: str = "ct") -> str:
    """
    Determine the modality class label with laterality encoding.
    E.g. 'ct_left_shoulder' or 'ct_right_shoulder'.
    Falls back to base_modality (e.g. 'ct') if laterality is unknown.
    """
    ct_id = extract_ct_id(filename)
    if ct_id and ct_id in ct_metadata:
        side = ct_metadata[ct_id].get("laterality")
        if side:
            return f"{base_modality}_{side}_shoulder"
    return base_modality


def get_mri_modality_with_laterality(filename: str, mri_metadata: dict[str, dict],
                                     base_modality: str = "mri_vibe") -> str:
    """
    Determine MRI modality class label with laterality encoding.
    E.g. 'mri_vibe_left_shoulder' or 'mri_vibe_right_shoulder'.
    Falls back to base_modality if laterality is unknown.
    """
    mr_id = extract_mr_id(filename)
    if mr_id and mr_id in mri_metadata:
        side = mri_metadata[mr_id].get("laterality")
        if isinstance(side, str) and side.lower() in {"left", "right"}:
            return f"{base_modality}_{side.lower()}_shoulder"
    return base_modality


# ------------------------------------------------------------------ #
# Visualization helpers (matching finetune_vae.py style)
# ------------------------------------------------------------------ #
def _resample_isotropic(vol, spacing):
    """Resample a 3D volume to isotropic spacing using the smallest spacing as target."""
    from scipy.ndimage import zoom
    spacing = np.array(spacing, dtype=np.float64)
    target_sp = spacing.min()
    zoom_factors = spacing / target_sp
    return zoom(vol, zoom_factors, order=1)


def save_slices_comparison(vol_a, vol_b, save_path, title_a="A", title_b="B",
                           suptitle="", spacing=None, subtitle_a="", subtitle_b=""):
    """Save axial/coronal/sagittal center slices comparing two 3D volumes [H, W, D].

    If spacing is provided (list of 3 floats), volumes are resampled to
    isotropic spacing so that each view shows correct physical proportions.
    subtitle_a/subtitle_b: extra info lines shown below each row's titles.
    """
    if spacing and len(spacing) >= 3:
        vol_a = _resample_isotropic(vol_a, spacing)
        vol_b = _resample_isotropic(vol_b, spacing)

    h, w, d = vol_a.shape
    slices = [
        (vol_a[h // 2, :, :], vol_b[h // 2, :, :], "Sagittal"),
        (vol_a[:, w // 2, :], vol_b[:, w // 2, :], "Coronal"),
        (vol_a[:, :, d // 2], vol_b[:, :, d // 2], "Axial"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for col, (s_a, s_b, view) in enumerate(slices):
        axes[0, col].imshow(s_a.T, cmap="gray", origin="lower")
        t_a = f"{title_a} - {view}"
        if subtitle_a:
            t_a += f"\n{subtitle_a}"
        axes[0, col].set_title(t_a, fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(s_b.T, cmap="gray", origin="lower")
        t_b = f"{title_b} - {view}"
        if subtitle_b:
            t_b += f"\n{subtitle_b}"
        axes[1, col].set_title(t_b, fontsize=10)
        axes[1, col].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_video_comparison(vol_a, vol_b, save_path, fps=10,
                          title_a="", title_b="", suptitle=""):
    """Save side-by-side video sweeping through depth slices of two 3D volumes [H, W, D].

    If titles are provided, each frame is rendered via matplotlib with labels.
    """
    def to_uint8(arr):
        vmin, vmax = np.percentile(arr, [1, 99])
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return (arr * 255).astype(np.uint8)

    use_titles = bool(title_a or title_b or suptitle)
    frames = []
    n_slices = vol_a.shape[2]
    for s in range(n_slices):
        sa = vol_a[:, :, s].T
        sb = vol_b[:, :, s].T

        if use_titles:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.imshow(sa, cmap="gray", origin="lower", vmin=np.percentile(vol_a, 1), vmax=np.percentile(vol_a, 99))
            ax1.set_title(f"{title_a}\nslice {s}/{n_slices}", fontsize=9)
            ax1.axis("off")
            ax2.imshow(sb, cmap="gray", origin="lower", vmin=np.percentile(vol_b, 1), vmax=np.percentile(vol_b, 99))
            ax2.set_title(f"{title_b}\nslice {s}/{n_slices}", fontsize=9)
            ax2.axis("off")
            if suptitle:
                fig.suptitle(suptitle, fontsize=10)
            fig.tight_layout()
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3].copy()
            plt.close(fig)
            frames.append(frame)
        else:
            sa_u8 = to_uint8(sa)
            sb_u8 = to_uint8(sb)
            divider = np.full((sa_u8.shape[0], 4), 128, dtype=np.uint8)
            frames.append(np.concatenate([sa_u8, divider, sb_u8], axis=1))

    imageio.mimwrite(save_path, frames, fps=fps)


def save_denoising_progress_video(step_snapshots, save_path, fps=5):
    """
    Save a video showing the middle axial slice evolving through denoising steps.

    Args:
        step_snapshots: list of (step_idx, timestep_value, latent_ch0_numpy) tuples
        save_path: output video path
    """
    def to_uint8(arr):
        vmin, vmax = np.percentile(arr, [1, 99])
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return (arr * 255).astype(np.uint8)

    frames = []
    for step_idx, t_val, latent_ch0 in step_snapshots:
        # Middle axial slice of channel 0
        mid_z = latent_ch0.shape[2] // 2
        slc = to_uint8(latent_ch0[:, :, mid_z].T)

        # Add text overlay using matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(slc, cmap="gray", origin="lower")
        ax.set_title(f"Step {step_idx}/{len(step_snapshots)-1}  t={t_val:.3f}", fontsize=12)
        ax.axis("off")
        fig.tight_layout()

        # Render to numpy array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    imageio.mimwrite(save_path, frames, fps=fps)


@torch.inference_mode()
def generate_validation_sample(
    raw_unet, noise_scheduler_cfg, scale_factor, device,
    latent_shape, spacing, modality_label, num_inference_steps=30,
    autoencoder=None,
):
    """
    Generate one sample via full denoising loop.

    Returns:
        denoised_latent: [4, H, W, D] numpy array (the denoised embedding)
        decoded_image: [H', W', D'] numpy array (VAE-decoded image) or None if no autoencoder
        step_snapshots: list of (step_idx, timestep_value, latent_ch0_numpy) for denoising progress video
    """
    raw_unet.eval()
    noise_scheduler = define_instance(argparse.Namespace(**noise_scheduler_cfg), "noise_scheduler")

    # Start from random noise
    noise = torch.randn((1, 4, *latent_shape), device=device)
    image = noise

    # Set timesteps
    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    spacing_tensor = torch.tensor([spacing], dtype=torch.float16, device=device) * 1e2
    modality_tensor = torch.tensor([modality_label], dtype=torch.long, device=device)

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))

    include_modality = raw_unet.num_class_embeds is not None

    # Collect snapshots at each denoising step
    step_snapshots = []
    # Snapshot the initial noise (step 0)
    step_snapshots.append((0, float(all_timesteps[0]), image[0, 0].cpu().float().numpy()))

    with torch.amp.autocast("cuda"):
        for step_idx, (t, next_t) in enumerate(zip(all_timesteps, all_next_timesteps)):
            unet_inputs = {
                "x": image,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
            }
            if include_modality:
                unet_inputs["class_labels"] = modality_tensor

            model_output = raw_unet(**unet_inputs)

            if not isinstance(noise_scheduler, RFlowScheduler):
                image, _ = noise_scheduler.step(model_output, t, image)
            else:
                image, _ = noise_scheduler.step(model_output, t, image, next_t)

            # Snapshot after this denoising step (channel 0 of latent)
            step_snapshots.append((step_idx + 1, float(next_t), image[0, 0].cpu().float().numpy()))

    # Denoised latent
    denoised_latent = image[0].cpu().float().numpy()  # [4, H, W, D]

    # Decode with VAE using SlidingWindowInferer (matches author's inference pipeline)
    decoded_image = None
    if autoencoder is not None:
        from scripts.sample import ReconModel
        autoencoder.eval()
        torch.cuda.empty_cache()
        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
        inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80],
            sw_batch_size=1,
            mode="gaussian",
            overlap=0.4,
            sw_device=device,
            device=device,
        )
        with torch.amp.autocast("cuda"):
            recon = dynamic_infer(inferer, recon_model, image)
        decoded_image = recon[0, 0].cpu().float().numpy()  # [H', W', D']
        del recon, recon_model

    raw_unet.train()
    return denoised_latent, decoded_image, step_snapshots


# ------------------------------------------------------------------ #
# Modality label augmentation (from scripts/diff_model_train.py)
# ------------------------------------------------------------------ #
def augment_modality_label(modality_tensor, prob=0.1):
    """Randomly augment modality labels for classifier-free guidance training."""
    mask_ct = (modality_tensor < 8) & (modality_tensor >= 2)
    prob_ct = torch.rand(modality_tensor.size(), device=modality_tensor.device) < prob
    modality_tensor[mask_ct & prob_ct] = 1

    mask_mri = modality_tensor >= 9
    prob_mri = torch.rand(modality_tensor.size(), device=modality_tensor.device) < prob
    modality_tensor[mask_mri & prob_mri] = 8

    mask_zero = torch.rand(modality_tensor.size(), device=modality_tensor.device) > prob
    modality_tensor = modality_tensor * mask_zero.long()
    return modality_tensor


# ------------------------------------------------------------------ #
# CLI argument parsing
# ------------------------------------------------------------------ #
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune MAISI Diffusion UNet on custom CT data")

    # Mode selection
    parser.add_argument("--create_embeddings", action="store_true",
                        help="Phase 1: create VAE embeddings from raw NIfTI files")
    parser.add_argument("--train", action="store_true",
                        help="Phase 2: train the diffusion UNet (default if no mode specified)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization samples from a checkpoint (no training)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path for --visualize mode")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of samples to generate in --visualize mode")
    parser.add_argument("--vis_spacing", type=float, nargs=3, default=None,
                        help="Original pixel spacing [x y z] for UNet conditioning (default: median from data)")
    parser.add_argument("--vis_display_spacing", type=float, nargs=3, default=None,
                        help="Resampled spacing [x y z] for visualization aspect ratio "
                             "(default: derived from vis_spacing accounting for z interpolation)")
    parser.add_argument("--vis_modality", type=str, default=None,
                        help="Modality label for generation (e.g. mri_vibe_left_shoulder, ct_left_shoulder)")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing CT*_bone.nii.gz files")
    parser.add_argument("--modality", type=str, default="ct",
                        help="Imaging modality: ct, mri_t1, etc.")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data for validation")

    # Run identity
    parser.add_argument("--run_name", type=str, default="v1",
                        help="Name for this training run (used for output subfolder)")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints",
                        help="Root directory for all checkpoints")
    parser.add_argument("--dataset_name", type=str, default="levin_ct",
                        help="Dataset name (used in folder hierarchy)")

    # Config directories
    parser.add_argument("--config_dir", type=str, default="./configs",
                        help="Directory containing JSON config files")

    # Phase 1 options
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to finetuned VAE checkpoint (default: models/autoencoder_v1.pt)")
    parser.add_argument("--embedding_dir", type=str, default=None,
                        help="Directory for embeddings (default: auto from checkpoint_root)")
    parser.add_argument("--ct_metadata_json", type=str,
                        default="/home/zl445/latent_diffusion_mri_ct/data/metadata/ct_dataset_index.json",
                        help="Path to CT dataset index JSON with laterality metadata")
    parser.add_argument("--mri_metadata_json", type=str,
                        default="/home/zl445/latent_diffusion_mri_ct/data/metadata/MRI_vibe_laterailty.json",
                        help="Path to MRI VIBE laterality JSON metadata")
    parser.add_argument("--sw_batch_size", type=int, default=1,
                        help="SlidingWindowInferer batch size for VAE encoding (default=1, matching tutorial)")
    parser.add_argument("--sw_overlap", type=float, default=0.4,
                        help="SlidingWindowInferer overlap (default=0.4, matching tutorial)")

    # Phase 2 overrides
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--cache_rate", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", dest="amp", action="store_false",
                        help="Disable automatic mixed precision")
    parser.add_argument("--no_finetune", action="store_true",
                        help="Train from scratch (don't load pretrained UNet)")
    parser.add_argument("--reset_scheduler", action="store_true",
                        help="Reset LR scheduler on resume (use when changing n_epochs)")

    # Validation / visualization
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Generate validation sample every N epochs (0 to disable)")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of denoising steps for validation sample generation")
    parser.add_argument("--val_latent_shape", type=int, nargs=3, default=[128, 128, 32],
                        help="Spatial shape of latent to generate for validation (e.g., 128 128 32)")

    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


# ================================================================== #
#  PHASE 1: Create Embeddings
# ================================================================== #
@torch.inference_mode()
def create_embeddings(cli_args):
    """
    Encode raw CT NIfTIs into VAE latent embeddings + metadata JSON.
    Mirrors the pipeline from scripts/diff_model_create_training_data.py.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 1] Creating embeddings on {device}")

    # ---- Load configs ---- #
    net_file = os.path.join(cli_args.config_dir, "config_network_rflow.json")
    with open(net_file, "r") as f:
        net_dict = json.load(f)

    env_file = os.path.join(cli_args.config_dir, "environment_maisi_diff_model_rflow-ct.json")
    with open(env_file, "r") as f:
        env_dict = json.load(f)

    args = argparse.Namespace()
    for k, v in net_dict.items():
        setattr(args, k, v)
    for k, v in env_dict.items():
        setattr(args, k, v)

    # ---- Load VAE ---- #
    args.autoencoder_def["num_splits"] = 4  # tensor splitting for full-volume encode
    autoencoder = define_instance(args, "autoencoder_def").to(device)

    vae_path = cli_args.vae_checkpoint or args.trained_autoencoder_path
    print(f"[Phase 1] Loading VAE from: {vae_path}")
    ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    if "unet_state_dict" in ckpt:
        ckpt = ckpt["unet_state_dict"]
    elif "autoencoder_state_dict" in ckpt:
        ckpt = ckpt["autoencoder_state_dict"]
    autoencoder.load_state_dict(ckpt)
    autoencoder.eval()
    del ckpt

    # ---- Discover images ---- #
    is_mri = "mri" in cli_args.modality.lower()
    if is_mri:
        all_images_raw = sorted(glob.glob(os.path.join(cli_args.data_dir, "**", "*.nii.gz"), recursive=True))
        if not all_images_raw:
            raise FileNotFoundError(f"No *.nii.gz files in {cli_args.data_dir}")
        print(f"[Phase 1] Found {len(all_images_raw)} MRI volumes total")
        all_images = all_images_raw
        ct_metadata = {}
        mri_metadata = {}
        if "vibe" in cli_args.modality.lower():
            mri_metadata = load_mri_metadata(cli_args.mri_metadata_json)
            print(f"[Phase 1] Loaded metadata for {len(mri_metadata)} MRs from {cli_args.mri_metadata_json}")
    else:
        ct_patterns = [
            "CT*_bone.nii.gz",
            "CT*_bone_mri_fov.nii.gz",
        ]
        all_images_raw = []
        for pattern in ct_patterns:
            all_images_raw.extend(glob.glob(os.path.join(cli_args.data_dir, pattern)))
        all_images_raw = sorted(set(all_images_raw))
        all_images_raw = [p for p in all_images_raw if not p.endswith("_bone_p.nii.gz")]
        if not all_images_raw:
            raise FileNotFoundError(
                f"No CT*_bone.nii.gz or CT*_bone_mri_fov.nii.gz files in {cli_args.data_dir}"
            )
        print(f"[Phase 1] Found {len(all_images_raw)} CT scans total")

        ct_metadata = load_ct_metadata(cli_args.ct_metadata_json)
        print(f"[Phase 1] Loaded metadata for {len(ct_metadata)} CTs from {cli_args.ct_metadata_json}")

        all_images = [p for p in all_images_raw if not is_contrast_ct(p, ct_metadata)]
        n_excluded = len(all_images_raw) - len(all_images)
        print(f"[Phase 1] Excluded {n_excluded} contrast CTs, keeping {len(all_images)} non-contrast CTs")
        mri_metadata = {}

    # ---- Setup output dirs ---- #
    embedding_dir = cli_args.embedding_dir
    if embedding_dir is None:
        embedding_dir = os.path.join(cli_args.checkpoint_root, "maisi_diff_unet",
                                     cli_args.dataset_name, "embeddings")
    Path(embedding_dir).mkdir(parents=True, exist_ok=True)

    # ---- Build dataset.json ---- #
    data_dicts = []
    for img in all_images:
        if is_mri:
            if "vibe" in cli_args.modality.lower():
                modality_str = get_mri_modality_with_laterality(
                    img, mri_metadata, cli_args.modality
                )
            else:
                modality_str = cli_args.modality
        else:
            modality_str = get_modality_with_laterality(img, ct_metadata, cli_args.modality)
        data_dicts.append({"image": os.path.basename(img), "modality": modality_str})
    n_val = max(1, int(len(data_dicts) * cli_args.val_split))
    n_train = len(data_dicts) - n_val
    train_files = data_dicts[:n_train]
    val_files = data_dicts[n_train:]

    dataset_json_path = os.path.join(embedding_dir, "dataset.json")
    dataset_list = {"training": train_files, "testing": val_files}
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_list, f, indent=4)
    print(f"[Phase 1] Saved dataset.json: {n_train} train, {n_val} val -> {dataset_json_path}")

    # ---- Baseline transforms (no resize) to probe metadata ---- #
    plain_transforms = create_transforms(dim=None, modality=cli_args.modality)

    # ---- Encode each image ---- #
    n_created = 0
    n_skipped = 0
    for img_path in tqdm(all_images, desc="Creating embeddings"):
        basename = os.path.basename(img_path)
        stem = basename.replace(".nii.gz", "").replace(".gz", "").replace(".nii", "")
        out_emb_path = os.path.join(embedding_dir, f"{stem}_emb.nii.gz")
        out_json_path = out_emb_path + ".json"

        # Skip if already exists
        if os.path.isfile(out_emb_path) and os.path.isfile(out_json_path):
            n_skipped += 1
            continue

        try:
            # A) Probe original metadata (dim, spacing)
            test_data = {"image": img_path}
            transformed = plain_transforms(test_data)
            nda = transformed["image"]
            orig_dim = [int(nda.meta["dim"][i]) for i in range(1, 4)]

            # B) Compute rounded target dims (multiples of 128)
            new_dim = tuple(round_number(d) for d in orig_dim)

            # C) Full preprocessing: orient + intensity + resize
            full_transforms = create_transforms(dim=new_dim, modality=cli_args.modality)
            new_data = full_transforms(test_data)
            nda_image = new_data["image"]
            new_affine = nda_image.meta["affine"].numpy()
            nda_image = nda_image.numpy().squeeze()  # [C, H, W, D] -> [H, W, D]

            # D) Encode with VAE
            with torch.amp.autocast("cuda"):
                pt_nda = torch.from_numpy(nda_image).float().to(device).unsqueeze(0).unsqueeze(0)
                inferer = SlidingWindowInferer(
                    roi_size=[320, 320, 160],
                    sw_batch_size=cli_args.sw_batch_size,
                    progress=False,
                    mode="gaussian",
                    overlap=cli_args.sw_overlap,
                    sw_device=device,
                    device=device,
                )
                z = dynamic_infer(inferer, autoencoder.encode_stage_2_inputs, pt_nda)

                # Save embedding NIfTI with resized image's affine
                out_nda = z.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)  # [C,H,W,D]->[H,W,D,C]
                out_img = nib.Nifti1Image(np.float32(out_nda), affine=new_affine)
                nib.save(out_img, out_emb_path)

            # E) Create metadata JSON from embedding NIfTI header
            emb_img = nib.load(out_emb_path)
            emb_dim = emb_img.shape[:3]
            emb_spacing = [float(s) for s in emb_img.header.get_zooms()[:3]]
            if is_mri:
                if "vibe" in cli_args.modality.lower():
                    sample_modality = get_mri_modality_with_laterality(
                        img_path, mri_metadata, cli_args.modality
                    )
                else:
                    sample_modality = cli_args.modality
            else:
                sample_modality = get_modality_with_laterality(
                    img_path, ct_metadata, cli_args.modality
                )
            meta = {
                "dim": list(emb_dim),
                "spacing": emb_spacing,
                "modality": sample_modality,
            }
            with open(out_json_path, "w") as f:
                json.dump(meta, f, indent=4)

            n_created += 1

        except Exception as e:
            print(f"  [ERROR] Failed to process {basename}: {e}")
            continue

    print(f"[Phase 1] Done: {n_created} created, {n_skipped} skipped (already existed)")
    print(f"[Phase 1] Embeddings dir: {embedding_dir}")
    print(f"[Phase 1] Dataset JSON:   {dataset_json_path}")
    return embedding_dir, dataset_json_path


# ================================================================== #
#  PHASE 2: Train Diffusion UNet
# ================================================================== #
# ------------------------------------------------------------------ #
# Picklable helper functions for data transforms (multiprocessing compat)
# ------------------------------------------------------------------ #
def _load_spacing_from_json(file_path):
    with open(file_path) as f:
        return torch.FloatTensor(json.load(f)["spacing"])


def _scale_spacing(x):
    return x * 1e2


class _LoadModalityFromJson:
    """Picklable callable that loads modality from JSON and maps to int."""
    def __init__(self, modality_mapping):
        self.modality_mapping = modality_mapping

    def __call__(self, file_path):
        with open(file_path) as f:
            modality_str = json.load(f)["modality"]
        return self.modality_mapping[modality_str]


def train_diffusion_unet(cli_args):
    """Train the diffusion UNet on VAE latent embeddings."""

    # ---- DDP setup ---- #
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=120))
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        world_size = 1
    is_main = local_rank == 0

    if not is_main:
        import builtins
        builtins.print = lambda *a, **kw: None

    print_config()
    set_determinism(seed=cli_args.seed + local_rank)
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")

    # ---- Load configs ---- #
    net_file = os.path.join(cli_args.config_dir, "config_network_rflow.json")
    with open(net_file, "r") as f:
        net_dict = json.load(f)

    model_config_file = os.path.join(cli_args.config_dir, "config_maisi_diff_model_rflow-ct.json")
    with open(model_config_file, "r") as f:
        model_config = json.load(f)

    env_file = os.path.join(cli_args.config_dir, "environment_maisi_diff_model_rflow-ct.json")
    with open(env_file, "r") as f:
        env_dict = json.load(f)

    modality_mapping_path = os.path.join(cli_args.config_dir, "modality_mapping.json")
    with open(modality_mapping_path, "r") as f:
        modality_mapping = json.load(f)

    args = argparse.Namespace()
    for k, v in net_dict.items():
        setattr(args, k, v)
    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in model_config.items():
        setattr(args, k, v)

    # CLI overrides
    train_cfg = args.diffusion_unet_train
    if cli_args.n_epochs is not None:
        train_cfg["n_epochs"] = cli_args.n_epochs
    if cli_args.batch_size is not None:
        train_cfg["batch_size"] = cli_args.batch_size
    if cli_args.lr is not None:
        train_cfg["lr"] = cli_args.lr
    if cli_args.cache_rate is not None:
        train_cfg["cache_rate"] = cli_args.cache_rate
    if cli_args.no_finetune:
        args.existing_ckpt_filepath = None

    # ---- Embedding & dataset paths ---- #
    embedding_dir = cli_args.embedding_dir
    if embedding_dir is None:
        embedding_dir = os.path.join(cli_args.checkpoint_root, "maisi_diff_unet",
                                     cli_args.dataset_name, "embeddings")
    dataset_json_path = os.path.join(embedding_dir, "dataset.json")

    if not os.path.isfile(dataset_json_path):
        raise FileNotFoundError(
            f"dataset.json not found at {dataset_json_path}. "
            "Run with --create_embeddings first."
        )

    # ---- Auto-versioning output directory ---- #
    base_run_dir = os.path.join(cli_args.checkpoint_root, "maisi_diff_unet",
                                cli_args.dataset_name, cli_args.run_name)

    if is_main:
        # Find latest checkpoint from previous versions
        resume_checkpoint_path = None
        if os.path.exists(base_run_dir):
            best_version = -1
            for folder in sorted(os.listdir(base_run_dir)):
                if not folder.startswith("version_"):
                    continue
                try:
                    vid = int(folder.split("_")[1])
                except (IndexError, ValueError):
                    continue
                ckpt_file = os.path.join(base_run_dir, folder, "checkpoints",
                                         "latest_checkpoint.pt")
                if os.path.isfile(ckpt_file) and vid > best_version:
                    best_version = vid
                    resume_checkpoint_path = ckpt_file

        # Next version number
        next_version = 0
        if os.path.exists(base_run_dir):
            existing = [d for d in os.listdir(base_run_dir) if d.startswith("version_")]
            if existing:
                vids = []
                for d in existing:
                    try:
                        vids.append(int(d.split("_")[1]))
                    except (IndexError, ValueError):
                        pass
                if vids:
                    next_version = max(vids) + 1

        run_dir = os.path.join(base_run_dir, f"version_{next_version}")

        if resume_checkpoint_path:
            print(f"[Resume] Found checkpoint: {resume_checkpoint_path}")
        else:
            print("[Resume] No previous checkpoint found.")

    # Broadcast to all ranks
    if distributed:
        if is_main:
            info = {"run_dir": run_dir, "resume_checkpoint_path": resume_checkpoint_path}
        else:
            info = None
        info_list = [info]
        dist.broadcast_object_list(info_list, src=0)
        info = info_list[0]
        run_dir = info["run_dir"]
        resume_checkpoint_path = info["resume_checkpoint_path"]

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    img_dir = os.path.join(run_dir, "images")
    vid_dir = os.path.join(run_dir, "videos")
    for d in [ckpt_dir, log_dir, img_dir, vid_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Save run config
    run_config = {
        "data_dir": cli_args.data_dir,
        "embedding_dir": embedding_dir,
        "run_name": cli_args.run_name,
        "n_epochs": train_cfg["n_epochs"],
        "batch_size": train_cfg["batch_size"],
        "lr": train_cfg["lr"],
        "cache_rate": train_cfg["cache_rate"],
        "modality": cli_args.modality,
        "amp": cli_args.amp,
        "seed": cli_args.seed,
        "world_size": world_size,
        "num_train_timesteps": args.noise_scheduler["num_train_timesteps"],
    }
    if is_main:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(run_config, f, indent=4)

    print("=" * 60)
    print(f"Run: {cli_args.run_name}")
    print(f"Output directory: {run_dir}")
    for k, v in run_config.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    tensorboard_writer = SummaryWriter(log_dir) if is_main else None

    # ---- Load dataset ---- #
    with open(dataset_json_path, "r") as f:
        dataset_list = json.load(f)

    # Build training file list (image -> emb NIfTI, spacing/modality -> JSON sidecar)
    filenames_train = dataset_list["training"]
    train_files = []
    for item in filenames_train:
        stem = item["image"].replace(".nii.gz", "").replace(".gz", "").replace(".nii", "")
        emb_path = os.path.join(embedding_dir, f"{stem}_emb.nii.gz")
        json_path = emb_path + ".json"
        if not os.path.exists(emb_path):
            print(f"  [WARN] Embedding not found, skipping: {emb_path}")
            continue
        if not os.path.exists(json_path):
            print(f"  [WARN] Metadata JSON not found, skipping: {json_path}")
            continue
        train_files.append({
            "image": emb_path,
            "spacing": json_path,
            "modality": json_path,
        })

    # Build validation file list
    filenames_val = dataset_list.get("testing", [])
    val_files = []
    for item in filenames_val:
        stem = item["image"].replace(".nii.gz", "").replace(".gz", "").replace(".nii", "")
        emb_path = os.path.join(embedding_dir, f"{stem}_emb.nii.gz")
        json_path = emb_path + ".json"
        if not os.path.exists(emb_path) or not os.path.exists(json_path):
            continue
        val_files.append({
            "image": emb_path,
            "spacing": json_path,
            "modality": json_path,
        })

    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

    if distributed:
        train_files = partition_dataset(
            data=train_files, shuffle=True,
            num_partitions=dist.get_world_size(), even_divisible=True
        )[local_rank]
        if val_files:
            val_files = partition_dataset(
                data=val_files, shuffle=False,
                num_partitions=dist.get_world_size(), even_divisible=False
            )[local_rank]

    # ---- Data transforms for embedding loading ---- #
    # NOTE: Use top-level picklable functions instead of lambdas for multiprocessing compat
    train_transforms_list = [
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Lambdad(keys="spacing",
                                 func=_load_spacing_from_json),
        monai.transforms.Lambdad(keys="spacing", func=_scale_spacing),
        monai.transforms.Lambdad(
            keys="modality",
            func=_LoadModalityFromJson(modality_mapping),
        ),
        monai.transforms.EnsureTyped(keys=["modality"], dtype=torch.long),
    ]
    train_transforms = Compose(train_transforms_list)

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=train_cfg["cache_rate"], num_workers=cli_args.num_workers,
    )
    train_loader = DataLoader(
        train_ds, num_workers=cli_args.num_workers,
        batch_size=train_cfg["batch_size"], shuffle=True,
    )

    val_loader = None
    if val_files:
        val_ds = monai.data.CacheDataset(
            data=val_files, transform=train_transforms,
            cache_rate=train_cfg["cache_rate"], num_workers=cli_args.num_workers,
        )
        val_loader = DataLoader(
            val_ds, num_workers=max(1, cli_args.num_workers // 4),
            batch_size=train_cfg["batch_size"], shuffle=False,
        )

    # ---- Model ---- #
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    if distributed:
        unet = DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)

    raw_unet = unet.module if distributed else unet

    # Load pretrained weights (only if not resuming)
    if resume_checkpoint_path is None:
        if args.existing_ckpt_filepath is not None:
            ckpt = torch.load(args.existing_ckpt_filepath, map_location=device, weights_only=False)
            raw_unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
            print(f"Loaded pretrained UNet from {args.existing_ckpt_filepath}")
            del ckpt
        else:
            print("Training from scratch.")

        # Initialize custom class embeddings from base modality embeddings.
        # Only on first load — skipped on resume so fine-tuned weights are kept.
        if raw_unet.num_class_embeds is not None and hasattr(raw_unet, "class_embedding"):
            with torch.no_grad():
                emb_w = raw_unet.class_embedding.weight
                init_pairs = [
                    (4, 1),   # ct_left_shoulder  <- ct
                    (5, 1),   # ct_right_shoulder <- ct
                    (18, 8),  # mri_vibe_left_shoulder  <- mri
                    (19, 8),  # mri_vibe_right_shoulder <- mri
                    (20, 8),  # mri_breast_pre  <- mri
                    (21, 8),  # mri_breast_post <- mri
                    (22, 8),  # mri_breast_sub  <- mri
                    (23, 8),  # mri_breast_t2   <- mri
                ]
                for dst_idx, src_idx in init_pairs:
                    if dst_idx < emb_w.shape[0] and src_idx < emb_w.shape[0]:
                        emb_w[dst_idx] = emb_w[src_idx].clone()
                print(f"Initialized class embeddings: "
                      + ", ".join(f"{d}<-{s}" for d, s in init_pairs))

    noise_scheduler = define_instance(args, "noise_scheduler")
    include_body_region = raw_unet.include_top_region_index_input
    include_modality = raw_unet.num_class_embeds is not None

    # Load VAE for validation visualization (rank 0 only)
    val_autoencoder = None
    if is_main and cli_args.val_interval > 0:
        args.autoencoder_def["num_splits"] = 4
        val_autoencoder = define_instance(args, "autoencoder_def").to(device)
        vae_path = cli_args.vae_checkpoint or args.trained_autoencoder_path
        vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
        if "unet_state_dict" in vae_ckpt:
            vae_ckpt = vae_ckpt["unet_state_dict"]
        elif "autoencoder_state_dict" in vae_ckpt:
            vae_ckpt = vae_ckpt["autoencoder_state_dict"]
        val_autoencoder.load_state_dict(vae_ckpt)
        val_autoencoder.eval()
        del vae_ckpt
        print(f"Loaded VAE for validation from: {vae_path}")

    # ---- Scale factor ---- #
    scale_factor = None
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    if distributed:
        dist.barrier()
        dist.all_reduce(scale_factor, op=dist.ReduceOp.AVG)
    print(f"Scale factor: {scale_factor.item():.6f}")

    # ---- Optimizer / Scheduler ---- #
    optimizer = torch.optim.Adam(params=unet.parameters(), lr=train_cfg["lr"])
    total_steps = (train_cfg["n_epochs"] * len(train_loader.dataset)) / train_cfg["batch_size"]
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=int(total_steps), power=2.0)
    loss_fn = torch.nn.L1Loss()
    scaler = GradScaler("cuda") if cli_args.amp else None

    # ---- Resume from checkpoint ---- #
    start_epoch = 0
    resume_total_step = 0
    best_loss = float("inf")
    if resume_checkpoint_path is not None:
        ckpt = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        if "unet_state_dict" in ckpt:
            raw_unet.load_state_dict(ckpt["unet_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in ckpt and not cli_args.reset_scheduler:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
            print(f"[Resume] Loaded LR scheduler state (total_iters={lr_scheduler.total_iters})")
        elif cli_args.reset_scheduler:
            # Optimizer state dict restores lr=0 from exhausted schedule.
            # PolynomialLR.get_lr() multiplies current group["lr"] by a decay
            # factor, so 0 * anything = 0 forever.  Fix: restore base lr and
            # recreate the scheduler so its internal state is consistent.
            for pg in optimizer.param_groups:
                pg["lr"] = train_cfg["lr"]
                pg["initial_lr"] = train_cfg["lr"]
            lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=int(total_steps), power=2.0
            )
            print(f"[Resume] Reset LR scheduler: lr={train_cfg['lr']}, "
                  f"new total_iters={int(total_steps)} "
                  f"(n_epochs={train_cfg['n_epochs']}, steps/epoch={len(train_loader.dataset)})")
        if cli_args.amp and "scaler_state_dict" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        resume_total_step = ckpt.get("total_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        if "scale_factor" in ckpt:
            scale_factor = ckpt["scale_factor"]
            if isinstance(scale_factor, (int, float)):
                scale_factor = torch.tensor(scale_factor, device=device)
        print(f"[Resume] epoch {start_epoch}, step {resume_total_step}, "
              f"best_loss {best_loss:.6f}, scale_factor {scale_factor.item():.6f}")
        del ckpt

    torch.set_float32_matmul_precision("highest")

    # ---- Training loop ---- #
    total_step = resume_total_step
    num_train_timesteps = args.noise_scheduler["num_train_timesteps"]

    for epoch in range(start_epoch, train_cfg["n_epochs"]):
        unet.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True, disable=not is_main)
        for batch_idx, train_data in enumerate(pbar):
            images = train_data["image"].to(device)
            images = images * scale_factor
            spacing_tensor = train_data["spacing"].to(device)

            if include_modality:
                modality_tensor = train_data["modality"].to(device)
                modality_tensor = augment_modality_label(modality_tensor).to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=cli_args.amp):
                noise = torch.randn_like(images)

                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(images)
                else:
                    timesteps = torch.randint(
                        0, num_train_timesteps, (images.shape[0],),
                        device=images.device
                    ).long()

                noisy_latent = noise_scheduler.add_noise(
                    original_samples=images, noise=noise, timesteps=timesteps
                )

                unet_inputs = {
                    "x": noisy_latent,
                    "timesteps": timesteps,
                    "spacing_tensor": spacing_tensor,
                }
                if include_body_region:
                    unet_inputs["top_region_index_tensor"] = train_data["top_region_index"].to(device)
                    unet_inputs["bottom_region_index_tensor"] = train_data["bottom_region_index"].to(device)
                if include_modality:
                    unet_inputs["class_labels"] = modality_tensor

                model_output = unet(**unet_inputs)

                # Determine ground truth based on prediction type
                if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                    model_gt = noise
                elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                    model_gt = images
                elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                    model_gt = images - noise
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.prediction_type}")

                loss = loss_fn(model_output.float(), model_gt.float())

            if cli_args.amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            total_step += 1
            epoch_loss_sum += loss.item()
            epoch_loss_count += 1

            if is_main:
                tensorboard_writer.add_scalar("train_loss_iter", loss.item(), total_step)
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                                 step=total_step)

        pbar.close()

        # Aggregate epoch loss across ranks
        if distributed:
            loss_tensor = torch.tensor([epoch_loss_sum, epoch_loss_count],
                                       dtype=torch.float, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss_sum = loss_tensor[0].item()
            epoch_loss_count = loss_tensor[1].item()

        epoch_loss = epoch_loss_sum / max(epoch_loss_count, 1)

        # --- Validation loss ---
        val_loss = float("inf")
        if val_loader is not None:
            unet.eval()
            val_loss_sum = 0.0
            val_loss_count = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_images = val_data["image"].to(device) * scale_factor
                    val_spacing = val_data["spacing"].to(device)

                    with autocast("cuda", enabled=cli_args.amp):
                        val_noise = torch.randn_like(val_images)
                        if isinstance(noise_scheduler, RFlowScheduler):
                            val_ts = noise_scheduler.sample_timesteps(val_images)
                        else:
                            val_ts = torch.randint(0, num_train_timesteps, (val_images.shape[0],), device=device).long()

                        val_noisy = noise_scheduler.add_noise(original_samples=val_images, noise=val_noise, timesteps=val_ts)
                        val_unet_in = {"x": val_noisy, "timesteps": val_ts, "spacing_tensor": val_spacing}
                        if include_body_region:
                            val_unet_in["top_region_index_tensor"] = val_data["top_region_index"].to(device)
                            val_unet_in["bottom_region_index_tensor"] = val_data["bottom_region_index"].to(device)
                        if include_modality:
                            val_unet_in["class_labels"] = val_data["modality"].to(device)
                        val_output = unet(**val_unet_in)

                        if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                            val_gt = val_noise
                        elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                            val_gt = val_images
                        elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                            val_gt = val_images - val_noise
                        else:
                            val_gt = val_noise

                        val_loss_sum += loss_fn(val_output.float(), val_gt.float()).item()
                        val_loss_count += 1

            if distributed and val_loss_count > 0:
                vl_tensor = torch.tensor([val_loss_sum, val_loss_count], dtype=torch.float, device=device)
                dist.all_reduce(vl_tensor, op=dist.ReduceOp.SUM)
                val_loss_sum, val_loss_count = vl_tensor[0].item(), vl_tensor[1].item()

            val_loss = val_loss_sum / max(val_loss_count, 1)
            unet.train()

        print(f"  Epoch {epoch} | train_loss: {epoch_loss:.6f} | val_loss: {val_loss:.6f}")

        if is_main:
            tensorboard_writer.add_scalar("train_loss_epoch", epoch_loss, epoch)
            if val_loader is not None:
                tensorboard_writer.add_scalar("val_loss_epoch", val_loss, epoch)
            tensorboard_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            tensorboard_writer.flush()

            # Save full checkpoint for resume
            ckpt_state = {
                "epoch": epoch + 1,
                "total_step": total_step,
                "best_loss": best_loss,
                "scale_factor": scale_factor,
                "num_train_timesteps": num_train_timesteps,
                "unet_state_dict": raw_unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            if cli_args.amp and scaler is not None:
                ckpt_state["scaler_state_dict"] = scaler.state_dict()
            torch.save(ckpt_state, os.path.join(ckpt_dir, "latest_checkpoint.pt"))

            # Also save model-only checkpoint (compatible with inference scripts)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "num_train_timesteps": num_train_timesteps,
                    "scale_factor": scale_factor,
                    "unet_state_dict": raw_unet.state_dict(),
                },
                os.path.join(ckpt_dir, "diff_unet_3d_rflow-ct.pt"),
            )

            select_loss = val_loss if val_loader is not None else epoch_loss
            if select_loss < best_loss:
                best_loss = select_loss
                best_path = os.path.join(ckpt_dir, f"best_diff_unet_epoch{epoch}.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "train_loss": epoch_loss,
                        "val_loss": val_loss,
                        "num_train_timesteps": num_train_timesteps,
                        "scale_factor": scale_factor,
                        "unet_state_dict": raw_unet.state_dict(),
                    },
                    best_path,
                )
                print(f"  ** New best {'val' if val_loader else 'train'}_loss! Saved to {best_path}")

            # ---- Validation: generate sample & visualize ---- #
            if cli_args.val_interval > 0 and (epoch + 1) % cli_args.val_interval == 0:
                print(f"  Generating validation sample (epoch {epoch})...")
                torch.cuda.empty_cache()
                try:
                    val_spacing = cli_args.vis_spacing or [0.5, 0.5, 0.6]
                    display_spacing = cli_args.vis_display_spacing or val_spacing
                    if cli_args.vis_modality and cli_args.vis_modality in modality_mapping:
                        val_modality = modality_mapping[cli_args.vis_modality]
                    else:
                        val_modality = 4

                    noise_sched_cfg = {"noise_scheduler": args.noise_scheduler}
                    denoised_latent, decoded_image, step_snapshots = generate_validation_sample(
                        raw_unet=raw_unet,
                        noise_scheduler_cfg=noise_sched_cfg,
                        scale_factor=scale_factor,
                        device=device,
                        latent_shape=tuple(cli_args.val_latent_shape),
                        spacing=val_spacing,
                        modality_label=val_modality,
                        num_inference_steps=cli_args.num_inference_steps,
                        autoencoder=val_autoencoder,
                    )

                    base_name = f"val_epoch{epoch:04d}_step{total_step:06d}"

                    # --- Visualize denoised latent (channel 0) ---
                    latent_ch0 = denoised_latent[0]  # [H, W, D] first latent channel
                    # Compare with random noise for reference
                    noise_ref = np.random.randn(*latent_ch0.shape).astype(np.float32)
                    save_slices_comparison(
                        noise_ref, latent_ch0,
                        os.path.join(img_dir, f"{base_name}_latent.png"),
                        title_a="Random Noise", title_b="Denoised Latent (ch0)",
                        suptitle=f"Epoch {epoch}  Loss {epoch_loss:.4f}",
                        spacing=display_spacing,
                    )
                    save_video_comparison(
                        noise_ref, latent_ch0,
                        os.path.join(vid_dir, f"{base_name}_latent.mp4"),
                    )

                    # --- Denoising progress video (middle slice through all steps) ---
                    save_denoising_progress_video(
                        step_snapshots,
                        os.path.join(vid_dir, f"{base_name}_denoise_progress.mp4"),
                        fps=3,
                    )

                    # --- Visualize decoded image vs real MRI ---
                    if decoded_image is not None:
                        real_image = None
                        real_label = ""
                        try:
                            _KNOWN_SEQ = ["Post_1", "Post_2", "Post_3", "Post_4", "Post_5",
                                          "Post_6", "Post_7", "Pre", "Sub_1", "T2"]
                            ds_json = os.path.join(embedding_dir, "dataset.json")
                            if os.path.isfile(ds_json):
                                with open(ds_json) as _f:
                                    _ds = json.load(_f)
                                vis_mod = cli_args.vis_modality
                                _candidates = [f for f in _ds.get("testing", [])
                                               if (not vis_mod) or f.get("modality") == vis_mod]
                                if _candidates:
                                    _pick = _candidates[np.random.randint(len(_candidates))]
                                    _stem = _pick["image"].replace(".nii.gz", "")
                                    _seq = None
                                    for s in sorted(_KNOWN_SEQ, key=len, reverse=True):
                                        if _stem.endswith(f"_{s}"):
                                            _uid = _stem[:-(len(s) + 1)]
                                            _seq = s
                                            break
                                    if _seq:
                                        for _inst in os.listdir(cli_args.data_dir):
                                            _nii = os.path.join(cli_args.data_dir, _inst,
                                                                "data_unilateral", _uid, f"{_seq}.nii.gz")
                                            if os.path.isfile(_nii):
                                                _data = np.asarray(nib.load(_nii).dataobj, dtype=np.float32)
                                                _lo, _hi = np.percentile(_data, [0, 99.5])
                                                _data = np.clip((_data - _lo) / max(_hi - _lo, 1e-8), 0, 1)
                                                target_sz = decoded_image.shape
                                                if _data.shape != target_sz:
                                                    from scipy.ndimage import zoom as _zoom
                                                    _data = _zoom(_data, [t / s for t, s in
                                                                          zip(target_sz, _data.shape)], order=1)
                                                real_image = _data
                                                real_label = _stem
                                                break
                        except Exception as e:
                            print(f"  [WARN] Failed to load real sample: {e}")

                        if real_image is not None:
                            save_slices_comparison(
                                real_image, decoded_image,
                                os.path.join(img_dir, f"{base_name}_decoded.png"),
                                title_a="Real", title_b="Generated",
                                suptitle=f"Epoch {epoch}  Loss {epoch_loss:.4f}  |  Real vs Generated",
                                spacing=display_spacing,
                                subtitle_a=real_label,
                                subtitle_b=f"seq: {cli_args.vis_modality}, cond_sp={val_spacing}",
                            )
                            save_video_comparison(
                                real_image, decoded_image,
                                os.path.join(vid_dir, f"{base_name}_decoded.mp4"),
                                title_a=f"Real ({real_label[:20]})",
                                title_b=f"Generated | {cli_args.vis_modality}",
                                suptitle=f"Epoch {epoch}  Loss {epoch_loss:.4f}",
                            )
                        else:
                            save_slices_comparison(
                                decoded_image, decoded_image,
                                os.path.join(img_dir, f"{base_name}_decoded.png"),
                                title_a="Generated (overview)", title_b="Generated (detail)",
                                suptitle=f"Epoch {epoch}  Generated from noise",
                                spacing=display_spacing,
                                subtitle_b=f"seq: {cli_args.vis_modality}, cond_sp={val_spacing}",
                            )
                        print(f"  Saved validation images/videos to {img_dir}")

                except Exception as e:
                    print(f"  [WARN] Validation sample generation failed: {e}")

        # Sync all ranks
        if distributed:
            dist.barrier()

    if is_main:
        tensorboard_writer.close()
    if distributed:
        dist.destroy_process_group()

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Run directory:     {run_dir}")
    print(f"Latest checkpoint: {os.path.join(ckpt_dir, 'latest_checkpoint.pt')}")
    print(f"Model checkpoint:  {os.path.join(ckpt_dir, 'diff_unet_3d_rflow-ct.pt')}")
    print(f"TensorBoard logs:  {log_dir}")
    print("=" * 60)


# ================================================================== #
#  Standalone visualization
# ================================================================== #
def visualize_from_checkpoint(cli_args):
    """Load a checkpoint and generate visualization samples (no training)."""
    import argparse as _argparse

    device = torch.device("cuda:0")

    ckpt_path = cli_args.ckpt
    if ckpt_path is None:
        raise ValueError("--ckpt is required for --visualize mode")

    # Load configs (same as training)
    config_dir = cli_args.config_dir or "./configs"
    env_config = os.path.join(config_dir, "environment_maisi_diff_model_rflow-ct.json")
    train_config = os.path.join(config_dir, "config_maisi_diff_model_rflow-ct.json")
    net_config = os.path.join(config_dir, "config_network_rflow.json")

    with open(env_config) as f:
        env_dict = json.load(f)
    with open(train_config) as f:
        train_dict = json.load(f)
    with open(net_config) as f:
        net_dict = json.load(f)

    args = _argparse.Namespace()
    for d in [env_dict, train_dict, net_dict]:
        for k, v in d.items():
            setattr(args, k, v)

    # Build and load UNet
    print(f"Loading UNet from {ckpt_path}...")
    unet = define_instance(args, "diffusion_unet_def").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    unet.load_state_dict(ckpt["unet_state_dict"])

    scale_factor = ckpt.get("scale_factor", 1.0)
    if isinstance(scale_factor, (int, float)):
        scale_factor = torch.tensor(scale_factor, device=device)
    epoch = ckpt.get("epoch", 0)
    loss = ckpt.get("loss", 0)
    print(f"  Epoch: {epoch}, Loss: {loss:.6f}, Scale factor: {scale_factor.item():.4f}")
    del ckpt

    # Load VAE
    args.autoencoder_def["num_splits"] = 4
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    vae_path = cli_args.vae_checkpoint or args.trained_autoencoder_path
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    if "unet_state_dict" in vae_ckpt:
        vae_ckpt = vae_ckpt["unet_state_dict"]
    elif "autoencoder_state_dict" in vae_ckpt:
        vae_ckpt = vae_ckpt["autoencoder_state_dict"]
    autoencoder.load_state_dict(vae_ckpt)
    autoencoder.eval()
    for m in autoencoder.modules():
        if hasattr(m, "norm_float16"):
            m.norm_float16 = False
    del vae_ckpt
    print("  VAE loaded")

    # Output directory
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)),
                           "vis_standalone", ckpt_name)
    img_dir = os.path.join(out_dir, "images")
    vid_dir = os.path.join(out_dir, "videos")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    latent_shape = tuple(cli_args.val_latent_shape)
    n_steps = cli_args.num_inference_steps
    noise_sched_cfg = {"noise_scheduler": args.noise_scheduler}

    # Load modality mapping
    modality_mapping_path = os.path.join(config_dir, "modality_mapping.json")
    with open(modality_mapping_path) as f:
        modality_mapping = json.load(f)

    # Spacing: val_spacing for UNet conditioning, display_spacing for visualization
    val_spacing = cli_args.vis_spacing or [0.3125, 0.3125, 0.63]
    display_spacing = cli_args.vis_display_spacing or val_spacing

    # Modality conditioning
    if cli_args.vis_modality:
        if cli_args.vis_modality not in modality_mapping:
            print(f"  Available modalities: {list(modality_mapping.keys())}")
            raise ValueError(f"Unknown modality: {cli_args.vis_modality}")
        val_modality = modality_mapping[cli_args.vis_modality]
        modality_name = cli_args.vis_modality
    else:
        val_modality = modality_mapping.get("mri_vibe_left_shoulder", 18)
        modality_name = "mri_vibe_left_shoulder"

    # ---- Load real validation images for side-by-side comparison ---- #
    # Load raw NIfTIs (no VAE decode needed) and resize to match generated output
    embedding_dir = cli_args.embedding_dir
    if embedding_dir is None:
        embedding_dir = os.path.join(cli_args.checkpoint_root, "maisi_diff_unet",
                                     cli_args.dataset_name, "embeddings")
    dataset_json_path = os.path.join(embedding_dir, "dataset.json")
    data_dir = cli_args.data_dir
    decoded_target_size = (256, 256, 128)

    _KNOWN_SEQUENCES = ["Post_1", "Post_2", "Post_3", "Post_4", "Post_5",
                        "Post_6", "Post_7", "Pre", "Sub_1", "T2"]

    def _parse_uid_sequence(image_name):
        """Parse 'RUMC_007_right_Post_1.nii.gz' -> ('RUMC_007_right', 'Post_1')."""
        stem = image_name.replace(".nii.gz", "")
        for seq in sorted(_KNOWN_SEQUENCES, key=len, reverse=True):
            if stem.endswith(f"_{seq}"):
                uid = stem[: -(len(seq) + 1)]
                return uid, seq
        return stem, None

    def _find_raw_nifti(uid, sequence):
        """Search institution folders for the raw NIfTI file."""
        for inst in os.listdir(data_dir):
            nii_path = os.path.join(data_dir, inst, "data_unilateral", uid, f"{sequence}.nii.gz")
            if os.path.isfile(nii_path):
                return nii_path
        return None

    def _load_and_resize_nifti(nii_path, target_size):
        """Load a NIfTI, normalize intensity, and resize to target_size."""
        data = np.asarray(nib.load(nii_path).dataobj, dtype=np.float32)
        lo, hi = np.percentile(data, [0, 99.5])
        data = np.clip((data - lo) / max(hi - lo, 1e-8), 0, 1)
        if data.shape != target_size:
            from scipy.ndimage import zoom as _zoom
            data = _zoom(data, [t / s for t, s in zip(target_size, data.shape)], order=1)
        return data

    real_images = []
    if os.path.isfile(dataset_json_path):
        with open(dataset_json_path) as f:
            dataset_list = json.load(f)
        val_files = dataset_list.get("testing", dataset_list.get("training", []))

        # Filter to same modality as vis_modality for fair comparison
        vis_mod = cli_args.vis_modality or modality_name
        val_files_filtered = [f for f in val_files if f.get("modality") == vis_mod]
        if not val_files_filtered:
            val_files_filtered = val_files
            print(f"  [WARN] No val samples with modality '{vis_mod}', using all")
        else:
            print(f"  Filtered to {len(val_files_filtered)} val samples with modality '{vis_mod}'")

        np.random.seed(42)
        indices = np.random.permutation(len(val_files_filtered))
        for idx in indices:
            if len(real_images) >= cli_args.n_samples:
                break
            item = val_files_filtered[idx]
            uid, seq = _parse_uid_sequence(item["image"])
            if seq is None:
                continue
            nii_path = _find_raw_nifti(uid, seq)
            if nii_path is None:
                continue
            try:
                real_np = _load_and_resize_nifti(nii_path, decoded_target_size)
                real_images.append((f"{uid}_{seq}", real_np))
            except Exception as e:
                print(f"  [WARN] Failed to load {uid}/{seq}: {e}")
        print(f"  Loaded {len(real_images)} real validation images for comparison")
    else:
        print(f"  [WARN] dataset.json not found at {dataset_json_path}, no real samples for comparison")

    print(f"\nGenerating {cli_args.n_samples} samples...")
    print(f"  Latent shape:    {latent_shape}")
    print(f"  Spacing:         {val_spacing}")
    print(f"  Modality:        {modality_name} (id={val_modality})")
    print(f"  Inference steps: {n_steps}")
    print(f"  Output:          {out_dir}")

    for i in range(cli_args.n_samples):
        print(f"\n  Sample {i+1}/{cli_args.n_samples}...")
        denoised_latent, decoded_image, step_snapshots = generate_validation_sample(
            raw_unet=unet,
            noise_scheduler_cfg=noise_sched_cfg,
            scale_factor=scale_factor,
            device=device,
            latent_shape=latent_shape,
            spacing=val_spacing,
            modality_label=val_modality,
            num_inference_steps=n_steps,
            autoencoder=autoencoder,
        )

        base = f"sample_{i:03d}_epoch{epoch}"

        # Latent visualization
        latent_ch0 = denoised_latent[0]
        noise_ref = np.random.randn(*latent_ch0.shape).astype(np.float32)
        save_slices_comparison(
            noise_ref, latent_ch0,
            os.path.join(img_dir, f"{base}_latent.png"),
            title_a="Random Noise", title_b="Denoised Latent (ch0)",
            suptitle=f"Epoch {epoch}  Sample {i+1}  |  cond: {modality_name}, cond_sp={val_spacing}",
            spacing=display_spacing,
        )
        save_video_comparison(
            noise_ref, latent_ch0,
            os.path.join(vid_dir, f"{base}_latent.mp4"),
            title_a="Random Noise", title_b="Denoised Latent (ch0)",
            suptitle=f"Epoch {epoch}  Sample {i+1}  |  {modality_name}, cond_sp={val_spacing}",
        )

        # Denoising progress
        save_denoising_progress_video(
            step_snapshots,
            os.path.join(vid_dir, f"{base}_denoise_progress.mp4"),
            fps=3,
        )

        gen_sub = (f"latent: {list(latent_shape)}  |  seq: {modality_name} (id={val_modality})"
                   f"  |  spacing: {val_spacing}  |  steps: {n_steps}")

        # Decoded image: real (left) vs generated (right)
        if decoded_image is not None:
            if i < len(real_images):
                real_name, real_np = real_images[i]
                if real_np.shape != decoded_image.shape:
                    from scipy.ndimage import zoom as _zoom
                    real_np = _zoom(real_np,
                        [g / r for g, r in zip(decoded_image.shape, real_np.shape)], order=1)
                title_a = "Real"
                sub_a = real_name
                title_b = "Generated"
                suptitle = f"Epoch {epoch}  Sample {i+1}  |  Real vs Generated"
            else:
                real_np = decoded_image
                title_a = "Generated (copy)"
                sub_a = ""
                title_b = "Generated"
                suptitle = f"Epoch {epoch}  Sample {i+1}  (no real sample)"

            save_slices_comparison(
                real_np, decoded_image,
                os.path.join(img_dir, f"{base}_decoded.png"),
                title_a=title_a, title_b=title_b,
                suptitle=suptitle,
                spacing=display_spacing,
                subtitle_a=sub_a, subtitle_b=gen_sub,
            )
            vid_title_a = f"Real ({sub_a[:20]})" if sub_a else title_a
            vid_title_b = f"Generated | {modality_name}, cond_sp={val_spacing}"
            save_video_comparison(
                real_np, decoded_image,
                os.path.join(vid_dir, f"{base}_decoded.mp4"),
                title_a=vid_title_a, title_b=vid_title_b,
                suptitle=f"Epoch {epoch}  Sample {i+1}",
            )
            print(f"    Saved: {base}_decoded.png/mp4")

        torch.cuda.empty_cache()

    print(f"\nDone! Output at: {out_dir}")


# ================================================================== #
#  Main
# ================================================================== #
def main():
    cli_args = parse_args()

    # Default: if neither flag given, run training
    if not cli_args.create_embeddings and not cli_args.train and not cli_args.visualize:
        cli_args.train = True

    if cli_args.create_embeddings:
        create_embeddings(cli_args)

    if cli_args.train:
        train_diffusion_unet(cli_args)

    if cli_args.visualize:
        visualize_from_checkpoint(cli_args)


if __name__ == "__main__":
    main()
