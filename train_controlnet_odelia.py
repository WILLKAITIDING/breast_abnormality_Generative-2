#!/usr/bin/env python
"""Train ControlNet with pixel-level image conditioning for ODELIA breast MRI.

The preprocessed MRI volume (intensity-normed, resized to [256,256,128]) is used
as single-channel conditioning input. The ControlNet learns to guide generation
given the real anatomy, enabling abnormality detection via reconstruction comparison.

Usage (multi-GPU):
    torchrun --nproc_per_node=2 train_controlnet_odelia.py \
        -e configs/environment_controlnet_odelia.json \
        -c configs/config_controlnet_odelia.json \
        -t configs/config_network_rflow_odelia_controlnet.json \
        -g 2 --run_name v1_image_cond
"""

import argparse
import glob
import json
import logging
import multiprocessing
import os
import random as _random
import time
from datetime import timedelta
from pathlib import Path

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer
from monai.networks.utils import copy_model_state
from monai.utils import RankFilter
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from scripts.utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp, dynamic_infer
from scripts.sample import ReconModel
from scripts.diff_model_setting import load_config, setup_logging


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_slices_comparison(vol_a, vol_b, save_path, title_a="A", title_b="B", suptitle=""):
    h, w, d = vol_a.shape
    slices = [
        (vol_a[h // 2, :, :], vol_b[h // 2, :, :], "Sagittal"),
        (vol_a[:, w // 2, :], vol_b[:, w // 2, :], "Coronal"),
        (vol_a[:, :, d // 2], vol_b[:, :, d // 2], "Axial"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, (s_a, s_b, view) in enumerate(slices):
        axes[0, col].imshow(s_a.T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"{title_a} - {view}")
        axes[0, col].axis("off")
        axes[1, col].imshow(s_b.T, cmap="gray", origin="lower")
        axes[1, col].set_title(f"{title_b} - {view}")
        axes[1, col].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_video_comparison(vol_a, vol_b, save_path, fps=10, cond_vol=None, mask_vol=None, mse_vol=None, title=None):
    from matplotlib.cm import hot as hot_cmap

    def to_uint8(arr):
        vmin, vmax = np.percentile(arr, [1, 99])
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return (arr * 255).astype(np.uint8)

    def to_uint8_rgb_yellow_mask(gray_slice, mask_slice):
        g = np.clip(gray_slice, 0, 1)
        rgb = np.stack([g, g, g], axis=-1)
        masked = mask_slice < 0.5
        rgb[masked] = [1.0, 1.0, 0.0]
        return (rgb * 255).astype(np.uint8)

    mse_99 = np.percentile(mse_vol, 99.5) if mse_vol is not None else 1.0

    frames = []
    for s in range(vol_a.shape[2]):
        sa = to_uint8(vol_a[:, :, s].T)
        sb = to_uint8(vol_b[:, :, s].T)
        divider = np.full((sa.shape[0], 4, 3), 128, dtype=np.uint8)
        sa_rgb = np.stack([sa, sa, sa], axis=-1)
        sb_rgb = np.stack([sb, sb, sb], axis=-1)

        panels = []
        if cond_vol is not None and mask_vol is not None:
            cs = cond_vol.shape[2]
            si = int(s * cs / vol_a.shape[2])
            si = min(si, cs - 1)
            sc = to_uint8_rgb_yellow_mask(cond_vol[:, :, si].T, mask_vol[:, :, si].T)
            if sc.shape[:2] != sa.shape[:2]:
                from PIL import Image
                sc = np.array(Image.fromarray(sc).resize((sa.shape[1], sa.shape[0]), Image.BILINEAR))
            panels.extend([sc, divider])

        panels.extend([sa_rgb, divider, sb_rgb])

        if mse_vol is not None:
            mse_slice = mse_vol[:, :, s].T
            mse_normed = np.clip(mse_slice / (mse_99 + 1e-8), 0, 1)
            overlay_color = (hot_cmap(mse_normed)[:, :, :3] * 255).astype(np.uint8)
            alpha = np.clip(mse_normed * 2, 0, 0.7)[:, :, np.newaxis]
            blended = (sa_rgb.astype(np.float32) * (1 - alpha) + overlay_color.astype(np.float32) * alpha)
            panels.extend([divider, np.clip(blended, 0, 255).astype(np.uint8)])

        frame = np.concatenate(panels, axis=1)
        if title:
            title_bar = np.zeros((28, frame.shape[1], 3), dtype=np.uint8)
            frame = np.concatenate([title_bar, frame], axis=0)
        frames.append(frame)

    if title:
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            for i, f in enumerate(frames):
                pil_img = PILImage.fromarray(f)
                draw = ImageDraw.Draw(pil_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
                except OSError:
                    font = ImageFont.load_default()
                draw.text((8, 6), f"{title}  |  slice {i+1}/{len(frames)}", fill=(255, 255, 255), font=font)
                frames[i] = np.array(pil_img)
        except ImportError:
            pass

    imageio.mimwrite(save_path, frames, fps=fps)


def random_mask_3d(x, max_masks=3, min_frac=0.1, max_frac=0.4, p_nomask=0.1, fill_value=0.0, return_info=False):
    """Zero out 1-N random cubic patches in a 5D tensor [B,C,H,W,D].

    With probability p_nomask the input is returned unmodified so the model
    also sees fully intact conditioning during training.

    If return_info=True, returns (masked, mask, info_list) where info_list
    contains dicts with origin/size of each cube for visualization.
    """
    if torch.rand(1).item() < p_nomask:
        ret = (x, torch.ones_like(x[:, :1]))
        return (*ret, []) if return_info else ret
    masked = x.clone()
    B, C, H, W, D = x.shape
    mask = torch.ones(B, 1, H, W, D, device=x.device)
    n_masks = torch.randint(1, max_masks + 1, (1,)).item()
    info_list = []
    for _ in range(n_masks):
        fh = torch.empty(1).uniform_(min_frac, max_frac).item()
        fw = torch.empty(1).uniform_(min_frac, max_frac).item()
        fd = torch.empty(1).uniform_(min_frac, max_frac).item()
        sh, sw, sd = int(H * fh), int(W * fw), int(D * fd)
        h0 = torch.randint(0, max(H - sh, 1), (1,)).item()
        w0 = torch.randint(0, max(W - sw, 1), (1,)).item()
        d0 = torch.randint(0, max(D - sd, 1), (1,)).item()
        masked[:, :, h0:h0 + sh, w0:w0 + sw, d0:d0 + sd] = fill_value
        mask[:, :, h0:h0 + sh, w0:w0 + sw, d0:d0 + sd] = 0.0
        info_list.append({"origin": (h0, w0, d0), "size": (sh, sw, sd),
                          "frac": (fh, fw, fd)})
    ret = (masked, mask)
    return (*ret, info_list) if return_info else ret


@torch.inference_mode()
def generate_image_conditioned_sample(
    controlnet, unet, noise_scheduler_cfg, scale_factor, device,
    image_cond, spacing_tensor, modality_tensor, latent_shape,
    num_inference_steps=30,
):
    """Full denoising from noise, conditioned on pixel-level image."""
    import argparse as _argparse
    scheduler = define_instance(_argparse.Namespace(**noise_scheduler_cfg), "noise_scheduler")

    image = torch.randn((1, *latent_shape), device=device, dtype=torch.float32)

    if isinstance(scheduler, RFlowScheduler):
        scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            input_img_size_numel=torch.prod(torch.tensor(image.shape[2:])),
        )
    else:
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    all_timesteps = scheduler.timesteps
    all_next = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))

    with torch.amp.autocast("cuda"):
        for t, next_t in zip(all_timesteps, all_next):
            t_tensor = torch.tensor((t,), device=device)
            cn_in = {"x": image, "timesteps": t_tensor, "controlnet_cond": image_cond}
            if modality_tensor is not None:
                cn_in["class_labels"] = modality_tensor
            down, mid = controlnet(**cn_in)

            unet_in = {
                "x": image, "timesteps": t_tensor, "spacing_tensor": spacing_tensor,
                "down_block_additional_residuals": down, "mid_block_additional_residual": mid,
            }
            if modality_tensor is not None:
                unet_in["class_labels"] = modality_tensor
            out = unet(**unet_in)

            if not isinstance(scheduler, RFlowScheduler):
                image, _ = scheduler.step(out, t, image)
            else:
                image, _ = scheduler.step(out, t, image, next_t)

    return image


@torch.inference_mode()
def decode_latent(latent, autoencoder, scale_factor, device):
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    latent = latent.to(device)
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    inferer = SlidingWindowInferer(roi_size=[80, 80, 80], sw_batch_size=1, mode="gaussian",
                                   overlap=0.4, sw_device=device, device=device)
    with torch.amp.autocast("cuda"):
        recon = dynamic_infer(inferer, recon_model, latent)
    decoded = recon[0, 0].cpu().float().numpy()
    del recon, recon_model
    return decoded


@torch.inference_mode()
def sliding_window_infer_lesion(
    controlnet, unet, autoencoder, noise_scheduler_cfg, scale_factor, device,
    cond_volume, spacing_tensor, modality_tensor, latent_shape,
    window_frac=0.25, num_inference_steps=30,
):
    """Tile the conditioning volume with non-overlapping windows, mask each one,
    generate via ControlNet, decode, and assemble a full reconstruction + MSE map.

    Args:
        cond_volume: [1, 1, H, W, D] conditioning image at 4x-latent resolution.
        latent_shape: [C, lH, lW, lD] shape of the latent.
        window_frac: fraction of each spatial dim for the window size.

    Returns:
        (original_np, reconstructed_np, mse_map) all as numpy [H, W, D].
    """
    _, _, H, W, D = cond_volume.shape
    wh, ww, wd = int(H * window_frac), int(W * window_frac), int(D * window_frac)
    wh, ww, wd = max(wh, 1), max(ww, 1), max(wd, 1)

    original_np = cond_volume[0, 0].cpu().float().numpy()
    reconstructed = np.zeros_like(original_np)
    counts = np.zeros_like(original_np)

    positions = []
    for h0 in range(0, H, wh):
        for w0 in range(0, W, ww):
            for d0 in range(0, D, wd):
                positions.append((h0, w0, d0))

    logger = logging.getLogger("maisi.controlnet.odelia")
    logger.info(f"    Sliding window: {len(positions)} positions, "
                f"window=[{wh},{ww},{wd}], volume=[{H},{W},{D}]")

    for pi, (h0, w0, d0) in enumerate(positions):
        h1, w1, d1 = min(h0 + wh, H), min(w0 + ww, W), min(d0 + wd, D)
        masked_cond = cond_volume.clone()
        masked_cond[:, :, h0:h1, w0:w1, d0:d1] = 0.0

        gen_latent = generate_image_conditioned_sample(
            controlnet, unet, noise_scheduler_cfg, scale_factor, device,
            masked_cond, spacing_tensor, modality_tensor, latent_shape,
            num_inference_steps=num_inference_steps,
        )
        gen_decoded = decode_latent(gen_latent, autoencoder, scale_factor, device)

        gh, gw, gd = gen_decoded.shape
        rh0 = int(h0 * gh / H); rh1 = int(h1 * gh / H)
        rw0 = int(w0 * gw / W); rw1 = int(w1 * gw / W)
        rd0 = int(d0 * gd / D); rd1 = int(d1 * gd / D)

        reconstructed[rh0:rh1, rw0:rw1, rd0:rd1] += gen_decoded[rh0:rh1, rw0:rw1, rd0:rd1]
        counts[rh0:rh1, rw0:rw1, rd0:rd1] += 1.0

        del gen_latent, gen_decoded, masked_cond
        torch.cuda.empty_cache()

        if (pi + 1) % 10 == 0 or (pi + 1) == len(positions):
            logger.info(f"    Window {pi+1}/{len(positions)} done")

    counts = np.maximum(counts, 1.0)
    reconstructed /= counts
    mse_map = (original_np - reconstructed) ** 2

    return original_np, reconstructed, mse_map


@torch.inference_mode()
def single_pass_infer_lesion(
    controlnet, unet, autoencoder, noise_scheduler_cfg, scale_factor, device,
    cond_volume, spacing_tensor, modality_tensor, latent_shape,
    num_inference_steps=30,
):
    """Single-pass inference: feed the full unmasked image as conditioning,
    generate one reconstruction, decode, and compute MSE.

    Much faster than sliding_window_infer_lesion (1 pass vs 64+).
    The assumption is that a model trained only on healthy data with random
    masking will still reconstruct "healthy-looking" anatomy even when given
    an abnormal image as conditioning, producing high MSE at lesion sites.

    Returns:
        (original_np, reconstructed_np, mse_map) all as numpy [H, W, D].
    """
    logger = logging.getLogger("maisi.controlnet.odelia")
    logger.info("    Single-pass inference (full image conditioning)")

    original_np = cond_volume[0, 0].cpu().float().numpy()

    gen_latent = generate_image_conditioned_sample(
        controlnet, unet, noise_scheduler_cfg, scale_factor, device,
        cond_volume, spacing_tensor, modality_tensor, latent_shape,
        num_inference_steps=num_inference_steps,
    )
    reconstructed = decode_latent(gen_latent, autoencoder, scale_factor, device)

    del gen_latent
    torch.cuda.empty_cache()

    mse_map = (original_np - reconstructed) ** 2
    return original_np, reconstructed, mse_map


def save_lesion_inference_figure(original, reconstructed, mse_map, save_path,
                                 name="", epoch=0):
    """Save a 4-row comparison: Original | Reconstructed | MSE heatmap | Overlay."""
    from matplotlib.cm import hot as hot_cmap
    h, w, d = original.shape
    mean_mse = float(np.mean(mse_map))
    max_mse = float(np.max(mse_map))
    mse_99 = float(np.percentile(mse_map, 99.5))

    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    views = [
        ("Sagittal", original[h//2,:,:], reconstructed[h//2,:,:], mse_map[h//2,:,:]),
        ("Coronal",  original[:,w//2,:], reconstructed[:,w//2,:], mse_map[:,w//2,:]),
        ("Axial",    original[:,:,d//2], reconstructed[:,:,d//2], mse_map[:,:,d//2]),
    ]
    for col, (view, sl_o, sl_r, sl_m) in enumerate(views):
        axes[0, col].imshow(sl_o.T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Original - {view}"); axes[0, col].axis("off")

        axes[1, col].imshow(sl_r.T, cmap="gray", origin="lower")
        axes[1, col].set_title(f"Reconstructed - {view}"); axes[1, col].axis("off")

        im = axes[2, col].imshow(sl_m.T, cmap="hot", origin="lower", vmin=0, vmax=mse_99)
        axes[2, col].set_title(f"MSE - {view}"); axes[2, col].axis("off")
        plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)

        def _to_gray_rgb(s):
            vmin, vmax = np.percentile(s, [1, 99])
            normed = np.clip((s - vmin) / (vmax - vmin + 1e-8), 0, 1)
            return np.stack([normed] * 3, axis=-1)

        rgb = _to_gray_rgb(sl_o.T)
        mse_normed = np.clip(sl_m.T / (mse_99 + 1e-8), 0, 1)
        overlay_color = hot_cmap(mse_normed)[:, :, :3]
        alpha = np.clip(mse_normed * 2, 0, 0.7)[:, :, np.newaxis]
        blended = rgb * (1 - alpha) + overlay_color * alpha
        axes[3, col].imshow(np.clip(blended, 0, 1), origin="lower")
        axes[3, col].set_title(f"Overlay - {view}"); axes[3, col].axis("off")

    fig.suptitle(f"{name}  |  Epoch {epoch+1}\n"
                 f"Mean MSE: {mean_mse:.6f}  |  Max MSE: {max_mse:.6f}  |  99.5th: {mse_99:.6f}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_controlnet_odelia(
    env_config_path: str, model_config_path: str, model_def_path: str,
    num_gpus: int, run_name: str, checkpoint_root: str, vae_checkpoint: str = None,
    diffusion_ckpt: str = None,
    n_epochs_override: int = None, batch_size_override: int = None,
    lr_override: float = None, num_workers_override: int = None,
    cache_rate_override: float = None,
    mask_max_masks: int = 3, mask_min_frac: float = 0.1,
    mask_max_frac: float = 0.4, mask_p_nomask: float = 0.1,
    lesion_prep_dir: str = None, infer_interval: int = 50,
    infer_window_frac: float = 0.25, infer_n_samples: int = 3,
) -> None:
    logger = logging.getLogger("maisi.controlnet.odelia")
    use_ddp = num_gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    is_main = rank == 0
    logger.info(f"World_size: {world_size}")

    args = load_config(env_config_path, model_config_path, model_def_path)

    # --- Auto-versioning ---
    base_run_dir = os.path.join(checkpoint_root, run_name)
    resume_controlnet_path = None
    run_dir = None

    if is_main:
        if os.path.exists(base_run_dir):
            best_version = -1
            for folder in sorted(os.listdir(base_run_dir)):
                if not folder.startswith("version_"):
                    continue
                try:
                    vid = int(folder.split("_")[1])
                except (IndexError, ValueError):
                    continue
                ckpt_file = os.path.join(base_run_dir, folder, "checkpoints", "controlnet_current.pt")
                if os.path.isfile(ckpt_file) and vid > best_version:
                    best_version = vid
                    resume_controlnet_path = ckpt_file

        next_version = 0
        if os.path.exists(base_run_dir):
            existing = [d for d in os.listdir(base_run_dir) if d.startswith("version_")]
            if existing:
                vids = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
                if vids:
                    next_version = max(vids) + 1

        run_dir = os.path.join(base_run_dir, f"version_{next_version}")
        logger.info(f"Run directory: {run_dir}")
        if resume_controlnet_path:
            logger.info(f"Resuming from: {resume_controlnet_path}")

    if use_ddp:
        info = {"run_dir": run_dir, "resume_controlnet_path": resume_controlnet_path} if is_main else None
        info_list = [info]
        dist.broadcast_object_list(info_list, src=0)
        info = info_list[0]
        run_dir = info["run_dir"]
        resume_controlnet_path = info["resume_controlnet_path"]

    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        tb_dir = Path(run_dir) / "tfevent"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(tb_dir))

    # --- Models ---
    unet = define_instance(args, "diffusion_unet_def").to(device)
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    diff_path = diffusion_ckpt if diffusion_ckpt else args.trained_diffusion_path
    diff_ckpt_data = torch.load(diff_path, map_location=device, weights_only=False)
    unet.load_state_dict(diff_ckpt_data["unet_state_dict"], strict=False)
    scale_factor = diff_ckpt_data["scale_factor"]
    if isinstance(scale_factor, (int, float)):
        scale_factor = torch.tensor(scale_factor, device=device)
    logger.info(f"Loaded UNet from {diff_path}, scale_factor={scale_factor.item():.4f}")

    controlnet = define_instance(args, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())

    # Resume from previous version or pretrained
    start_epoch = 0
    if resume_controlnet_path:
        ckpt = torch.load(resume_controlnet_path, map_location=device, weights_only=False)
        controlnet.load_state_dict(ckpt["controlnet_state_dict"], strict=False)
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("val_loss", ckpt.get("loss", 1e4))
        if isinstance(best_loss, torch.Tensor):
            best_loss = best_loss.item()
        logger.info(f"Resumed controlnet from epoch {start_epoch}")
        del ckpt
    elif args.existing_ckpt_filepath and os.path.exists(args.existing_ckpt_filepath):
        controlnet.load_state_dict(
            torch.load(args.existing_ckpt_filepath, map_location=device, weights_only=False)["controlnet_state_dict"],
            strict=False,
        )
        logger.info(f"Loaded pretrained controlnet from {args.existing_ckpt_filepath}")
        best_loss = 1e4
    else:
        logger.info("Training controlnet from scratch (image conditioning).")
        best_loss = 1e4

    for p in unet.parameters():
        p.requires_grad = False
    unet.eval()

    noise_scheduler = define_instance(args, "noise_scheduler")

    # VAE for validation decoding
    autoencoder = None
    if vae_checkpoint:
        autoencoder = define_instance(args, "autoencoder_def").to(device)
        vae_ckpt = torch.load(vae_checkpoint, map_location=device, weights_only=False)
        if "unet_state_dict" in vae_ckpt:
            vae_ckpt = vae_ckpt["unet_state_dict"]
        autoencoder.load_state_dict(vae_ckpt)
        autoencoder.eval()
        for p in autoencoder.parameters():
            p.requires_grad = False
        logger.info(f"Loaded frozen VAE from {vae_checkpoint}")
        del vae_ckpt

    if use_ddp:
        controlnet = DDP(controlnet, device_ids=[device], output_device=rank, find_unused_parameters=True)
    raw_controlnet = controlnet.module if use_ddp else controlnet

    # --- Data ---
    if include_modality:
        with open(args.modality_mapping_path, "r") as f:
            args.modality_mapping = json.load(f)
    else:
        args.modality_mapping = None

    batch_size = batch_size_override if batch_size_override is not None else args.controlnet_train["batch_size"]
    cache_rate = cache_rate_override if cache_rate_override is not None else args.controlnet_train["cache_rate"]
    num_workers = num_workers_override if num_workers_override is not None else 8

    train_loader, val_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank, world_size=world_size,
        batch_size=batch_size,
        cache_rate=cache_rate,
        fold=args.controlnet_train["fold"],
        modality_mapping=args.modality_mapping,
        num_workers=num_workers,
    )

    # --- Optimizer ---
    lr = lr_override if lr_override is not None else args.controlnet_train["lr"]
    n_epochs = n_epochs_override if n_epochs_override is not None else args.controlnet_train["n_epochs"]
    optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=lr)
    total_steps = (n_epochs * len(train_loader.dataset)) / batch_size
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)
    if is_main:
        logger.info(f"Training: lr={lr}, n_epochs={n_epochs}, total_steps={total_steps:.0f}")
    scaler = GradScaler("cuda")
    total_step = start_epoch * len(train_loader)

    # Read model def for inference scheduler
    with open(model_def_path) as f:
        model_def_dict = json.load(f)

    # --- Training loop ---
    controlnet.train()
    prev_time = time.time()

    for epoch in range(start_epoch, n_epochs):
        epoch_loss_ = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=True, disable=not is_main)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(device) * scale_factor
            labels = batch["label"].to(device)
            spacing_tensor = batch["spacing"].to(device)
            modality_tensor = batch["modality"].to(device) if include_modality else None

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=True):
                noise = torch.randn_like(images).to(device)
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(images)
                else:
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

                target_cond_size = [s * 4 for s in images.shape[2:]]
                controlnet_cond = F.interpolate(labels.float(), size=target_cond_size, mode="trilinear", align_corners=False)
                controlnet_cond, _ = random_mask_3d(
                    controlnet_cond,
                    max_masks=mask_max_masks, min_frac=mask_min_frac,
                    max_frac=mask_max_frac, p_nomask=mask_p_nomask,
                )
                noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

                cn_in = {"x": noisy_latent, "timesteps": timesteps, "controlnet_cond": controlnet_cond}
                if include_modality:
                    cn_in["class_labels"] = modality_tensor
                down, mid = controlnet(**cn_in)

                unet_in = {
                    "x": noisy_latent, "timesteps": timesteps, "spacing_tensor": spacing_tensor,
                    "down_block_additional_residuals": down, "mid_block_additional_residual": mid,
                }
                if include_modality:
                    unet_in["class_labels"] = modality_tensor
                model_output = unet(**unet_in)

                if noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                    model_gt = images - noise
                elif noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                    model_gt = noise
                else:
                    model_gt = images
                loss = F.l1_loss(model_output.float(), model_gt.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_step += 1

            if is_main:
                writer.add_scalar("train/loss_iter", loss.detach().cpu().item(), total_step)
                prev_time = time.time()
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
            epoch_loss_ += loss.detach()

        pbar.close()
        epoch_loss = epoch_loss_ / (step + 1)
        if use_ddp:
            dist.barrier()
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.AVG)

        # --- Validation loss ---
        val_loss_ = torch.tensor(0.0, device=device)
        val_steps = 0
        if val_loader is not None and len(val_loader) > 0:
            controlnet.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_images = val_batch["image"].to(device) * scale_factor
                    val_labels = val_batch["label"].to(device)
                    val_spacing = val_batch["spacing"].to(device)
                    val_mod = val_batch["modality"].to(device) if include_modality else None

                    with autocast("cuda", enabled=True):
                        val_noise = torch.randn_like(val_images).to(device)
                        if isinstance(noise_scheduler, RFlowScheduler):
                            val_ts = noise_scheduler.sample_timesteps(val_images)
                        else:
                            val_ts = torch.randint(0, noise_scheduler.num_train_timesteps, (val_images.shape[0],), device=device).long()

                        val_cond_size = [s * 4 for s in val_images.shape[2:]]
                        val_cond = F.interpolate(val_labels.float(), size=val_cond_size, mode="trilinear", align_corners=False)
                        val_cond, _ = random_mask_3d(
                            val_cond, max_masks=mask_max_masks, min_frac=mask_min_frac,
                            max_frac=mask_max_frac, p_nomask=mask_p_nomask,
                        )
                        val_noisy = noise_scheduler.add_noise(original_samples=val_images, noise=val_noise, timesteps=val_ts)

                        cn_in = {"x": val_noisy, "timesteps": val_ts, "controlnet_cond": val_cond}
                        if include_modality:
                            cn_in["class_labels"] = val_mod
                        down, mid = controlnet(**cn_in)

                        unet_in = {
                            "x": val_noisy, "timesteps": val_ts, "spacing_tensor": val_spacing,
                            "down_block_additional_residuals": down, "mid_block_additional_residual": mid,
                        }
                        if include_modality:
                            unet_in["class_labels"] = val_mod
                        val_output = unet(**unet_in)

                        if noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                            val_gt = val_images - val_noise
                        elif noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                            val_gt = val_noise
                        else:
                            val_gt = val_images
                        val_loss_ += F.l1_loss(val_output.float(), val_gt.float()).detach()
                        val_steps += 1

            controlnet.train()

        if val_steps > 0:
            val_loss = val_loss_ / val_steps
            if use_ddp:
                dist.barrier()
                dist.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG)
        else:
            val_loss = torch.tensor(float("inf"), device=device)

        if is_main:
            writer.add_scalar("train/loss_epoch", epoch_loss.cpu().item(), epoch + 1)
            if val_steps > 0:
                writer.add_scalar("val/loss_epoch", val_loss.cpu().item(), epoch + 1)
            logger.info(f"Epoch {epoch+1}/{n_epochs} | train_loss={epoch_loss.item():.4f} | val_loss={val_loss.item():.4f}")

            cn_sd = raw_controlnet.state_dict()
            torch.save({"epoch": epoch + 1, "train_loss": epoch_loss, "val_loss": val_loss,
                         "controlnet_state_dict": cn_sd, "scale_factor": scale_factor},
                        str(ckpt_dir / "controlnet_current.pt"))

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"epoch": epoch + 1, "train_loss": epoch_loss, "val_loss": best_loss,
                             "controlnet_state_dict": cn_sd, "scale_factor": scale_factor},
                            str(ckpt_dir / f"controlnet_best_epoch{epoch+1}.pt"))
                logger.info(f"  ** New best val_loss! Saved to controlnet_best_epoch{epoch+1}.pt")

            # --- Validation: generate decoded MRI from image conditioning ---
            if autoencoder is not None and val_loader is not None and len(val_loader) > 0:
                try:
                    controlnet.eval()
                    val_idx = _random.randint(0, len(val_loader.dataset) - 1)
                    val_sample = val_loader.dataset[val_idx]
                    val_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in val_sample.items()}
                    val_images = val_batch["image"].to(device) * scale_factor
                    val_cond_raw = val_batch["label"].to(device).float()
                    target_cond_size = [s * 4 for s in val_images.shape[2:]]
                    val_cond = F.interpolate(val_cond_raw, size=target_cond_size, mode="trilinear", align_corners=False)
                    val_cond_masked, val_mask, val_mask_info = random_mask_3d(
                        val_cond, max_masks=mask_max_masks, min_frac=mask_min_frac,
                        max_frac=mask_max_frac, p_nomask=0.0, return_info=True,
                    )
                    val_spacing = val_batch["spacing"].to(device)
                    val_modality = val_batch["modality"].to(device) if include_modality else None
                    latent_shape = val_images.shape[1:]

                    logger.info("  Generating validation MRI from masked image conditioning...")
                    gen_latent = generate_image_conditioned_sample(
                        raw_controlnet, unet,
                        {"noise_scheduler": model_def_dict["noise_scheduler"]},
                        scale_factor, device,
                        val_cond_masked, val_spacing, val_modality, latent_shape,
                    )

                    gen_decoded = decode_latent(gen_latent, autoencoder, scale_factor, device)
                    real_decoded = decode_latent(val_images, autoencoder, scale_factor, device)
                    cond_masked_np = val_cond_masked[0, 0].cpu().float().numpy()
                    mask_np = val_mask[0, 0].cpu().float().numpy()

                    def gray_with_yellow_mask(gray_slice, mask_slice):
                        """Convert grayscale slice to RGB with yellow overlay on masked (0) regions."""
                        g = np.clip(gray_slice, 0, 1)
                        rgb = np.stack([g, g, g], axis=-1)
                        masked_region = mask_slice < 0.5
                        rgb[masked_region] = [1.0, 1.0, 0.0]
                        return rgb

                    vis_dir = Path(run_dir) / "val_images"
                    vid_dir = Path(run_dir) / "val_videos"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vid_dir.mkdir(parents=True, exist_ok=True)
                    base = f"val_epoch{epoch:04d}"

                    h, w, d = real_decoded.shape
                    ch, cw, cd = cond_masked_np.shape
                    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
                    for col, (view, sl_c, sl_m, sl_r, sl_g) in enumerate([
                        ("Sagittal", cond_masked_np[ch//2,:,:], mask_np[ch//2,:,:], real_decoded[h//2,:,:], gen_decoded[h//2,:,:]),
                        ("Coronal",  cond_masked_np[:,cw//2,:], mask_np[:,cw//2,:], real_decoded[:,w//2,:], gen_decoded[:,w//2,:]),
                        ("Axial",    cond_masked_np[:,:,cd//2], mask_np[:,:,cd//2], real_decoded[:,:,d//2], gen_decoded[:,:,d//2]),
                    ]):
                        axes[0, col].imshow(gray_with_yellow_mask(sl_c.T, sl_m.T), origin="lower")
                        axes[0, col].set_title(f"Masked Conditioning - {view}")
                        axes[0, col].axis("off")
                        axes[1, col].imshow(sl_r.T, cmap="gray", origin="lower")
                        axes[1, col].set_title(f"Real (decoded) - {view}")
                        axes[1, col].axis("off")
                        axes[2, col].imshow(sl_g.T, cmap="gray", origin="lower")
                        axes[2, col].set_title(f"Generated - {view}")
                        axes[2, col].axis("off")
                    mask_desc_parts = []
                    for ci, c in enumerate(val_mask_info):
                        s = c['size']
                        f = c['frac']
                        mask_desc_parts.append(f"Cube {ci+1}: size=({s[0]},{s[1]},{s[2]}), frac=({f[0]:.2f},{f[1]:.2f},{f[2]:.2f})")
                    mask_title = f"n_masks={len(val_mask_info)}, cond=[{ch},{cw},{cd}]"
                    if mask_desc_parts:
                        mask_title += "\n" + "  |  ".join(mask_desc_parts)
                    fig.suptitle(f"Epoch {epoch+1} | Image-conditioned ControlNet\n{mask_title}",
                                 fontsize=12, family="monospace")
                    fig.tight_layout(rect=[0, 0, 1, 0.94])
                    fig.savefig(str(vis_dir / f"{base}_comparison.png"), dpi=100, bbox_inches="tight")
                    plt.close(fig)

                    save_video_comparison(
                        real_decoded, gen_decoded, str(vid_dir / f"{base}_decoded.mp4"),
                        cond_vol=cond_masked_np, mask_vol=mask_np,
                        title=f"Epoch {epoch+1} | {mask_title.split(chr(10))[0]}",
                    )

                    logger.info(f"  Saved validation to {vis_dir} and {vid_dir}")
                    del gen_latent, gen_decoded, real_decoded, cond_masked_np, mask_np
                    torch.cuda.empty_cache()
                    controlnet.train()
                except Exception as e:
                    logger.warning(f"  Validation generation failed: {e}")
                    controlnet.train()

            # --- Lesion inference: sliding-window MSE heatmap ---
            if (lesion_prep_dir and autoencoder is not None
                    and os.path.isdir(lesion_prep_dir)
                    and (epoch + 1) % infer_interval == 0):
                try:
                    controlnet.eval()
                    lesion_files = sorted(glob.glob(os.path.join(lesion_prep_dir, "*.nii.gz")))
                    if lesion_files:
                        _random.seed(epoch)
                        chosen = _random.sample(lesion_files, min(infer_n_samples, len(lesion_files)))

                        infer_img_dir = Path(run_dir) / "lesion_infer"
                        infer_vid_dir = Path(run_dir) / "lesion_infer_videos"
                        infer_img_dir.mkdir(parents=True, exist_ok=True)
                        infer_vid_dir.mkdir(parents=True, exist_ok=True)

                        with open(args.modality_mapping_path, "r") as f:
                            mod_map_infer = json.load(f)

                        def _modality_from_filename(fname):
                            lower = fname.lower()
                            if "sub" in lower: return "mri_breast_sub"
                            elif "post" in lower: return "mri_breast_post"
                            elif "pre" in lower: return "mri_breast_pre"
                            elif "t2" in lower: return "mri_breast_t2"
                            return "mri_breast_post"

                        for li, lf in enumerate(chosen):
                            lname = os.path.basename(lf).replace(".nii.gz", "")
                            sample_mod = _modality_from_filename(lname)
                            infer_modality = torch.tensor(mod_map_infer[sample_mod],
                                                          dtype=torch.long, device=device).unsqueeze(0) if include_modality else None
                            infer_spacing = torch.tensor([0.7, 0.7, 0.75], device=device).unsqueeze(0) * 1e2
                            logger.info(f"  Lesion inference [{li+1}/{len(chosen)}]: {lname}  (modality={sample_mod})")

                            vol = nib.load(lf).get_fdata().astype(np.float32)
                            cond_t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(device)
                            target_cond_size = [64 * 4, 64 * 4, 32 * 4]
                            cond_t = F.interpolate(cond_t, size=target_cond_size,
                                                   mode="trilinear", align_corners=False)
                            latent_shape = (4, 64, 64, 32)

                            orig_np, recon_np, mse_np = sliding_window_infer_lesion(
                                raw_controlnet, unet, autoencoder,
                                {"noise_scheduler": model_def_dict["noise_scheduler"]},
                                scale_factor, device,
                                cond_t, infer_spacing, infer_modality, latent_shape,
                                window_frac=infer_window_frac,
                            )

                            base = f"lesion_epoch{epoch:04d}_{li:02d}_{lname}"
                            save_lesion_inference_figure(
                                orig_np, recon_np, mse_np,
                                str(infer_img_dir / f"{base}.png"),
                                name=lname, epoch=epoch,
                            )
                            save_video_comparison(orig_np, recon_np,
                                                  str(infer_vid_dir / f"{base}.mp4"),
                                                  mse_vol=mse_np,
                                                  title=f"{lname} | MSE={float(np.mean(mse_np)):.6f}")

                            mean_mse = float(np.mean(mse_np))
                            writer.add_scalar(f"lesion_infer/mse_{lname}", mean_mse, epoch + 1)
                            logger.info(f"    Mean MSE: {mean_mse:.6f}")

                            del cond_t, orig_np, recon_np, mse_np
                            torch.cuda.empty_cache()

                        logger.info(f"  Lesion inference saved to {infer_img_dir}")
                    controlnet.train()
                except Exception as e:
                    logger.warning(f"  Lesion inference failed: {e}")
                    import traceback; traceback.print_exc()
                    controlnet.train()

        torch.cuda.empty_cache()

    if use_ddp:
        dist.destroy_process_group()
    if is_main:
        writer.close()
        logger.info("Training complete.")


def _load_models(env_config_path, model_config_path, model_def_path, controlnet_ckpt,
                  vae_checkpoint, device, diffusion_ckpt_path=None):
    """Load ControlNet, UNet, VAE, and return (controlnet, unet, autoencoder, scale_factor, args, model_def_dict)."""
    logger = logging.getLogger("maisi.controlnet.odelia")
    args = load_config(env_config_path, model_config_path, model_def_path)

    unet = define_instance(args, "diffusion_unet_def").to(device)
    diff_path = diffusion_ckpt_path if diffusion_ckpt_path else args.trained_diffusion_path
    diff_ckpt_data = torch.load(diff_path, map_location=device, weights_only=False)
    unet.load_state_dict(diff_ckpt_data["unet_state_dict"], strict=False)
    scale_factor = diff_ckpt_data["scale_factor"]
    if isinstance(scale_factor, (int, float)):
        scale_factor = torch.tensor(scale_factor, device=device)
    for p in unet.parameters():
        p.requires_grad = False
    unet.eval()
    logger.info(f"Loaded UNet from {diff_path}, scale_factor={scale_factor.item():.4f}")

    controlnet = define_instance(args, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())
    ckpt = torch.load(controlnet_ckpt, map_location=device, weights_only=False)
    controlnet.load_state_dict(ckpt["controlnet_state_dict"], strict=False)
    ckpt_epoch = ckpt.get("epoch", "?")
    logger.info(f"Loaded ControlNet from {controlnet_ckpt} (epoch {ckpt_epoch})")
    del ckpt
    controlnet.eval()
    for p in controlnet.parameters():
        p.requires_grad = False

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    vae_ckpt = torch.load(vae_checkpoint, map_location=device, weights_only=False)
    if "unet_state_dict" in vae_ckpt:
        vae_ckpt = vae_ckpt["unet_state_dict"]
    autoencoder.load_state_dict(vae_ckpt)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False
    logger.info(f"Loaded VAE from {vae_checkpoint}")
    del vae_ckpt

    with open(model_def_path) as f:
        model_def_dict = json.load(f)

    include_modality = unet.num_class_embeds is not None
    mod_map = None
    if include_modality:
        with open(args.modality_mapping_path, "r") as f:
            mod_map = json.load(f)

    return controlnet, unet, autoencoder, scale_factor, args, model_def_dict, mod_map, include_modality


def visualize_standalone(env_config_path, model_config_path, model_def_path,
                         controlnet_ckpt, vae_checkpoint, output_dir,
                         n_samples=5, prep_dir=None, datalist_file=None,
                         mask_max_masks=3, mask_min_frac=0.1, mask_max_frac=0.4,
                         diffusion_ckpt=None):
    """Generate masked-conditioning -> reconstruction visualizations from a trained checkpoint.

    Only samples from the validation fold (fold=0) to avoid data leakage.
    """
    logger = logging.getLogger("maisi.controlnet.odelia")
    setup_logging()
    device = torch.device("cuda:0")

    controlnet, unet, autoencoder, scale_factor, args, model_def_dict, \
        mod_map, include_modality = _load_models(
            env_config_path, model_config_path, model_def_path,
            controlnet_ckpt, vae_checkpoint, device, diffusion_ckpt_path=diffusion_ckpt)

    if prep_dir is None:
        prep_dir = args.data_base_dir[0] if isinstance(args.data_base_dir, list) else args.data_base_dir
        prep_dir = os.path.dirname(prep_dir)

    entry_lookup = {}
    if datalist_file and os.path.isfile(datalist_file):
        with open(datalist_file) as f:
            datalist = json.load(f)["training"]
        val_entries = [e for e in datalist if e.get("fold", 1) == 0]
        base_dir = os.path.dirname(prep_dir)
        files = []
        for e in val_entries:
            fpath = os.path.join(base_dir, e["label"])
            if os.path.isfile(fpath):
                files.append(fpath)
                entry_lookup[fpath] = e
        logger.info(f"Using {len(files)} validation samples from datalist (fold=0)")
    else:
        files = sorted(glob.glob(os.path.join(prep_dir, "*.nii.gz")))
        logger.warning(f"No datalist provided — sampling from ALL {len(files)} files in {prep_dir}")

    if not files:
        logger.error(f"No .nii.gz files found")
        return
    _random.seed(42)
    chosen = _random.sample(files, min(n_samples, len(files)))

    vis_dir = Path(output_dir) / "images"
    vid_dir = Path(output_dir) / "videos"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    latent_shape = (4, 64, 64, 32)
    target_cond_size = [64 * 4, 64 * 4, 32 * 4]

    for idx, fpath in enumerate(chosen):
        name = os.path.basename(fpath).replace(".nii.gz", "")
        entry = entry_lookup.get(fpath, {})
        sample_modality = entry.get("modality", "mri_breast_post")
        sample_spacing = entry.get("spacing", [0.7, 0.7, 0.75])

        modality_tensor = None
        if include_modality and mod_map:
            modality_tensor = torch.tensor(mod_map[sample_modality],
                                            dtype=torch.long, device=device).unsqueeze(0)
        spacing_tensor = torch.tensor(sample_spacing, device=device).unsqueeze(0) * 1e2

        logger.info(f"[{idx+1}/{len(chosen)}] Visualizing: {name}  (modality={sample_modality}, spacing={sample_spacing})")

        vol = nib.load(fpath).get_fdata().astype(np.float32)
        cond_t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(device)
        cond_t = F.interpolate(cond_t, size=target_cond_size, mode="trilinear", align_corners=False)

        cond_masked, mask, mask_info = random_mask_3d(
            cond_t, max_masks=mask_max_masks, min_frac=mask_min_frac,
            max_frac=mask_max_frac, p_nomask=0.0, return_info=True,
        )

        with torch.no_grad():
            gen_latent = generate_image_conditioned_sample(
                controlnet, unet,
                {"noise_scheduler": model_def_dict["noise_scheduler"]},
                scale_factor, device,
                cond_masked, spacing_tensor, modality_tensor, latent_shape,
            )
            gen_decoded = decode_latent(gen_latent, autoencoder, scale_factor, device)
            real_decoded = decode_latent(
                torch.randn(1, *latent_shape, device=device) * 0,
                autoencoder, scale_factor, device,
            )

        cond_np = cond_t[0, 0].cpu().numpy()
        cond_masked_np = cond_masked[0, 0].cpu().numpy()
        mask_np = mask[0, 0].cpu().numpy()
        gen_np = gen_decoded

        def _gray_yellow(gray_sl, mask_sl):
            g = np.clip(gray_sl, 0, 1)
            rgb = np.stack([g, g, g], axis=-1)
            rgb[mask_sl < 0.5] = [1.0, 1.0, 0.0]
            return rgb

        ch, cw, cd = cond_masked_np.shape
        gh, gw, gd = gen_np.shape
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        for col, (view, sl_o, sl_c, sl_m, sl_g) in enumerate([
            ("Sagittal", cond_np[ch//2,:,:], cond_masked_np[ch//2,:,:], mask_np[ch//2,:,:], gen_np[gh//2,:,:]),
            ("Coronal",  cond_np[:,cw//2,:], cond_masked_np[:,cw//2,:], mask_np[:,cw//2,:], gen_np[:,gw//2,:]),
            ("Axial",    cond_np[:,:,cd//2], cond_masked_np[:,:,cd//2], mask_np[:,:,cd//2], gen_np[:,:,gd//2]),
        ]):
            axes[0, col].imshow(sl_o.T, cmap="gray", origin="lower")
            axes[0, col].set_title(f"Original Conditioning - {view}"); axes[0, col].axis("off")
            axes[1, col].imshow(_gray_yellow(sl_c.T, sl_m.T), origin="lower")
            axes[1, col].set_title(f"Masked Conditioning - {view}"); axes[1, col].axis("off")
            axes[2, col].imshow(sl_g.T, cmap="gray", origin="lower")
            axes[2, col].set_title(f"Generated - {view}"); axes[2, col].axis("off")

        mask_desc = []
        for ci, c in enumerate(mask_info):
            s, f = c['size'], c['frac']
            mask_desc.append(f"Cube {ci+1}: size=({s[0]},{s[1]},{s[2]}), frac=({f[0]:.2f},{f[1]:.2f},{f[2]:.2f})")
        mask_title = f"n_masks={len(mask_info)}, cond=[{ch},{cw},{cd}]"
        if mask_desc:
            mask_title += "\n" + "  |  ".join(mask_desc)
        fig.suptitle(f"{name}\n{mask_title}", fontsize=12, family="monospace")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(str(vis_dir / f"vis_{idx:02d}_{name}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

        save_video_comparison(
            cond_np, gen_np, str(vid_dir / f"vis_{idx:02d}_{name}.mp4"),
            cond_vol=cond_masked_np, mask_vol=mask_np,
            title=f"{name} | {mask_title.split(chr(10))[0]}",
        )
        logger.info(f"  Saved to {vis_dir} and {vid_dir}")
        del cond_t, cond_masked, gen_latent, gen_decoded
        torch.cuda.empty_cache()

    logger.info(f"Done. {len(chosen)} visualizations saved to {output_dir}")


def inference_standalone(env_config_path, model_config_path, model_def_path,
                         controlnet_ckpt, vae_checkpoint, output_dir,
                         lesion_prep_dir, n_samples=10, window_frac=0.25,
                         strategy="sliding_window", diffusion_ckpt=None):
    """Run sliding-window lesion inference from a trained checkpoint."""
    logger = logging.getLogger("maisi.controlnet.odelia")
    setup_logging()
    device = torch.device("cuda:0")

    controlnet, unet, autoencoder, scale_factor, args, model_def_dict, \
        mod_map, include_modality = _load_models(
            env_config_path, model_config_path, model_def_path,
            controlnet_ckpt, vae_checkpoint, device, diffusion_ckpt_path=diffusion_ckpt)

    lesion_files = sorted(glob.glob(os.path.join(lesion_prep_dir, "*.nii.gz")))
    if not lesion_files:
        logger.error(f"No .nii.gz files in {lesion_prep_dir}")
        return
    _random.seed(42)
    chosen = _random.sample(lesion_files, min(n_samples, len(lesion_files)))

    img_dir = Path(output_dir) / "images"
    vid_dir = Path(output_dir) / "videos"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    latent_shape = (4, 64, 64, 32)
    target_cond_size = [64 * 4, 64 * 4, 32 * 4]

    def _modality_from_filename(fname):
        """Infer modality key from filename convention: ..._Pre.nii.gz, ..._Post_1.nii.gz, etc."""
        lower = fname.lower()
        if "sub" in lower:
            return "mri_breast_sub"
        elif "post" in lower:
            return "mri_breast_post"
        elif "pre" in lower:
            return "mri_breast_pre"
        elif "t2" in lower:
            return "mri_breast_t2"
        return "mri_breast_post"

    for idx, fpath in enumerate(chosen):
        name = os.path.basename(fpath).replace(".nii.gz", "")
        sample_modality = _modality_from_filename(name)

        modality_tensor = None
        if include_modality and mod_map:
            modality_tensor = torch.tensor(mod_map[sample_modality],
                                            dtype=torch.long, device=device).unsqueeze(0)
        spacing_tensor = torch.tensor([0.7, 0.7, 0.75], device=device).unsqueeze(0) * 1e2

        logger.info(f"[{idx+1}/{len(chosen)}] Lesion inference: {name}  (modality={sample_modality})")

        vol = nib.load(fpath).get_fdata().astype(np.float32)
        cond_t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(device)
        cond_t = F.interpolate(cond_t, size=target_cond_size, mode="trilinear", align_corners=False)

        with torch.no_grad():
            infer_fn_args = dict(
                controlnet=controlnet, unet=unet, autoencoder=autoencoder,
                noise_scheduler_cfg={"noise_scheduler": model_def_dict["noise_scheduler"]},
                scale_factor=scale_factor, device=device,
                cond_volume=cond_t, spacing_tensor=spacing_tensor,
                modality_tensor=modality_tensor, latent_shape=latent_shape,
            )
            if strategy == "single_pass":
                orig_np, recon_np, mse_np = single_pass_infer_lesion(**infer_fn_args)
            else:
                orig_np, recon_np, mse_np = sliding_window_infer_lesion(
                    **infer_fn_args, window_frac=window_frac)

        base = f"infer_{idx:02d}_{name}"
        save_lesion_inference_figure(orig_np, recon_np, mse_np,
                                     str(img_dir / f"{base}.png"), name=name, epoch=0)
        mean_mse_val = float(np.mean(mse_np))
        save_video_comparison(orig_np, recon_np, str(vid_dir / f"{base}.mp4"),
                              mse_vol=mse_np,
                              title=f"{name} | {strategy} | MSE={mean_mse_val:.6f}")

        mean_mse = float(np.mean(mse_np))
        logger.info(f"  Mean MSE: {mean_mse:.6f}")

        del cond_t, orig_np, recon_np, mse_np
        torch.cuda.empty_cache()

    logger.info(f"Done. {len(chosen)} inference results saved to {output_dir}")


def save_mama_mia_figure(original, reconstructed, mse_map, seg, save_path, name=""):
    """5-row figure: Original | Reconstructed | MSE heatmap | Overlay | Seg contour overlay."""
    from matplotlib.cm import hot as hot_cmap
    h, w, d = original.shape
    mean_mse = float(np.mean(mse_map))
    max_mse = float(np.max(mse_map))
    mse_99 = float(np.percentile(mse_map, 99.5))

    def _to_gray_rgb(s):
        vmin, vmax = np.percentile(s, [1, 99])
        normed = np.clip((s - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return np.stack([normed] * 3, axis=-1)

    fig, axes = plt.subplots(5, 3, figsize=(18, 30))
    views = [
        ("Sagittal", h//2, lambda v: v[h//2,:,:]),
        ("Coronal",  w//2, lambda v: v[:,w//2,:]),
        ("Axial",    d//2, lambda v: v[:,:,d//2]),
    ]
    for col, (view, _, slicer) in enumerate(views):
        sl_o, sl_r, sl_m, sl_s = slicer(original), slicer(reconstructed), slicer(mse_map), slicer(seg)

        axes[0, col].imshow(sl_o.T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Original - {view}"); axes[0, col].axis("off")

        axes[1, col].imshow(sl_r.T, cmap="gray", origin="lower")
        axes[1, col].set_title(f"Reconstructed - {view}"); axes[1, col].axis("off")

        im = axes[2, col].imshow(sl_m.T, cmap="hot", origin="lower", vmin=0, vmax=mse_99)
        axes[2, col].set_title(f"MSE - {view}"); axes[2, col].axis("off")
        plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)

        rgb = _to_gray_rgb(sl_o.T)
        mse_normed = np.clip(sl_m.T / (mse_99 + 1e-8), 0, 1)
        overlay_color = hot_cmap(mse_normed)[:, :, :3]
        alpha = np.clip(mse_normed * 2, 0, 0.7)[:, :, np.newaxis]
        blended = rgb * (1 - alpha) + overlay_color * alpha
        axes[3, col].imshow(np.clip(blended, 0, 1), origin="lower")
        axes[3, col].set_title(f"MSE Overlay - {view}"); axes[3, col].axis("off")

        rgb_seg = _to_gray_rgb(sl_o.T).copy()
        seg_mask = sl_s.T > 0.5
        rgb_seg[seg_mask] = rgb_seg[seg_mask] * 0.4 + np.array([0.0, 1.0, 0.0]) * 0.6
        axes[4, col].imshow(np.clip(rgb_seg, 0, 1), origin="lower")
        axes[4, col].set_title(f"GT Seg (green) - {view}"); axes[4, col].axis("off")

    fig.suptitle(f"{name}\nMean MSE: {mean_mse:.6f}  |  Max MSE: {max_mse:.6f}  |  99.5th: {mse_99:.6f}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def inference_mama_mia(env_config_path, model_config_path, model_def_path,
                       controlnet_ckpt, vae_checkpoint, output_dir,
                       mama_mia_dir, n_samples=30, strategy="single_pass",
                       window_frac=0.25, diffusion_ckpt=None,
                       modality_key=None):
    """Run inference on MAMA-MIA dataset with ground-truth segmentation overlay."""
    logger = logging.getLogger("maisi.controlnet.odelia")
    setup_logging()
    device = torch.device("cuda:0")

    controlnet, unet, autoencoder, scale_factor, args, model_def_dict, \
        mod_map, include_modality = _load_models(
            env_config_path, model_config_path, model_def_path,
            controlnet_ckpt, vae_checkpoint, device, diffusion_ckpt_path=diffusion_ckpt)

    all_files = sorted(os.listdir(mama_mia_dir))
    img_files = [f for f in all_files if "label" not in f and f.endswith(".nii.gz")]
    seg_files = [f for f in all_files if "label" in f and f.endswith(".nii.gz")]
    logger.info(f"MAMA-MIA: {len(img_files)} images, {len(seg_files)} segmentations")

    pairs = []
    for img_f in img_files:
        base = img_f.split("_0001")[0]
        matching = [s for s in seg_files if s.startswith(base)]
        if matching:
            pairs.append((img_f, matching[0]))
    logger.info(f"Paired (img+seg): {len(pairs)}")

    _random.seed(42)
    chosen = _random.sample(pairs, min(n_samples, len(pairs)))

    img_dir = Path(output_dir) / "images"
    vid_dir = Path(output_dir) / "videos"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    latent_shape = (4, 64, 64, 32)
    target_size = [256, 256, 128]
    target_cond_size = [64 * 4, 64 * 4, 32 * 4]

    _modality_key = modality_key or "mri_breast_post"
    modality_tensor = None
    if include_modality and mod_map:
        modality_tensor = torch.tensor(mod_map.get(_modality_key, 21),
                                        dtype=torch.long, device=device).unsqueeze(0)
    logger.info(f"Modality conditioning: {_modality_key} (id={mod_map.get(_modality_key, '?') if mod_map else 'N/A'})")
    spacing_tensor = torch.tensor([0.7, 0.7, 0.75], device=device).unsqueeze(0) * 1e2

    for idx, (img_f, seg_f) in enumerate(chosen):
        name = img_f.replace(".nii.gz", "").replace(" ", "_")
        logger.info(f"[{idx+1}/{len(chosen)}] {name}")

        img = nib.load(os.path.join(mama_mia_dir, img_f))
        vol = img.get_fdata().astype(np.float32)
        seg_vol = nib.load(os.path.join(mama_mia_dir, seg_f)).get_fdata().astype(np.float32)

        p995 = np.percentile(vol, 99.5)
        vol = np.clip(vol / (p995 + 1e-8), 0, 1)

        vol_t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
        vol_t = F.interpolate(vol_t, size=target_size, mode="trilinear", align_corners=False)
        vol_resized = vol_t[0, 0].numpy()

        seg_t = torch.from_numpy(seg_vol).float().unsqueeze(0).unsqueeze(0)
        seg_t = F.interpolate(seg_t, size=target_size, mode="nearest")
        seg_resized = seg_t[0, 0].numpy()

        cond_t = F.interpolate(vol_t, size=target_cond_size, mode="trilinear", align_corners=False).to(device)

        with torch.no_grad():
            infer_args = dict(
                controlnet=controlnet, unet=unet, autoencoder=autoencoder,
                noise_scheduler_cfg={"noise_scheduler": model_def_dict["noise_scheduler"]},
                scale_factor=scale_factor, device=device,
                cond_volume=cond_t, spacing_tensor=spacing_tensor,
                modality_tensor=modality_tensor, latent_shape=latent_shape,
            )
            if strategy == "single_pass":
                orig_np, recon_np, mse_np = single_pass_infer_lesion(**infer_args)
            else:
                orig_np, recon_np, mse_np = sliding_window_infer_lesion(
                    **infer_args, window_frac=window_frac)

        base = f"mama_{idx:02d}_{name}"
        save_mama_mia_figure(orig_np, recon_np, mse_np, seg_resized,
                             str(img_dir / f"{base}.png"), name=name)
        save_video_comparison(orig_np, recon_np, str(vid_dir / f"{base}.mp4"),
                              mse_vol=mse_np,
                              title=f"{name} | {strategy} | MSE={float(np.mean(mse_np)):.6f}")

        logger.info(f"  MSE={float(np.mean(mse_np)):.6f}, saved to {img_dir}")
        del cond_t, orig_np, recon_np, mse_np
        torch.cuda.empty_cache()

    logger.info(f"Done. {len(chosen)} MAMA-MIA results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ControlNet ODELIA Image Conditioning")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "visualize", "infer", "infer_mama_mia"],
                        help="Mode: train, visualize, infer (lesion), or infer_mama_mia")
    parser.add_argument("-e", "--env_config_path", type=str, required=True)
    parser.add_argument("-c", "--model_config_path", type=str, required=True)
    parser.add_argument("-t", "--model_def_path", type=str, required=True)
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="v1_image_cond")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints/controlnet_odelia")
    parser.add_argument("--vae_checkpoint", type=str, default=None)
    parser.add_argument("--diffusion_ckpt", type=str, default=None,
                        help="Override diffusion UNet checkpoint path (overrides env config)")
    parser.add_argument("--controlnet_ckpt", type=str, default=None,
                        help="ControlNet checkpoint for visualize/infer modes")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for visualize/infer modes")
    parser.add_argument("--n_epochs", type=int, default=None, help="Override n_epochs from config")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size from config")
    parser.add_argument("--lr", type=float, default=None, help="Override lr from config")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers for dataloaders")
    parser.add_argument("--cache_rate", type=float, default=None, help="Override cache_rate from config")
    parser.add_argument("--mask_max_masks", type=int, default=3, help="Max random cubic masks per sample")
    parser.add_argument("--mask_min_frac", type=float, default=0.1, help="Min mask size as fraction of each dim")
    parser.add_argument("--mask_max_frac", type=float, default=0.4, help="Max mask size as fraction of each dim")
    parser.add_argument("--mask_p_nomask", type=float, default=0.1, help="Probability of skipping masking")
    parser.add_argument("--prep_dir", type=str, default=None, help="Preprocessed image dir for visualize mode")
    parser.add_argument("--datalist_file", type=str, default=None, help="Datalist JSON to filter validation fold")
    parser.add_argument("--lesion_prep_dir", type=str, default=None, help="Dir with preprocessed lesion NIfTIs")
    parser.add_argument("--mama_mia_dir", type=str, default=None, help="MAMA-MIA dir with img+seg .nii.gz pairs")
    parser.add_argument("--infer_interval", type=int, default=50, help="Run lesion inference every N epochs")
    parser.add_argument("--infer_window_frac", type=float, default=0.25, help="Sliding window size as fraction of each dim")
    parser.add_argument("--infer_strategy", type=str, default="sliding_window",
                        choices=["sliding_window", "single_pass"],
                        help="Inference strategy: sliding_window (64 passes) or single_pass (1 pass)")
    parser.add_argument("--infer_n_samples", type=int, default=3, help="Number of samples for inference/visualize")
    parser.add_argument("--infer_modality", type=str, default=None,
                        help="Override modality conditioning key (e.g. mri_breast_pre, mri_breast_post)")
    args = parser.parse_args()

    if args.mode == "train":
        train_controlnet_odelia(
            args.env_config_path, args.model_config_path, args.model_def_path,
            args.num_gpus, args.run_name, args.checkpoint_root, args.vae_checkpoint,
            diffusion_ckpt=args.diffusion_ckpt,
            n_epochs_override=args.n_epochs, batch_size_override=args.batch_size,
            lr_override=args.lr, num_workers_override=args.num_workers,
            cache_rate_override=args.cache_rate,
            mask_max_masks=args.mask_max_masks, mask_min_frac=args.mask_min_frac,
            mask_max_frac=args.mask_max_frac, mask_p_nomask=args.mask_p_nomask,
            lesion_prep_dir=args.lesion_prep_dir, infer_interval=args.infer_interval,
            infer_window_frac=args.infer_window_frac, infer_n_samples=args.infer_n_samples,
        )
    elif args.mode == "visualize":
        assert args.controlnet_ckpt, "--controlnet_ckpt required for visualize mode"
        assert args.vae_checkpoint, "--vae_checkpoint required for visualize mode"
        out = args.output_dir or "./outputs/controlnet_odelia/visualize_standalone"
        visualize_standalone(
            args.env_config_path, args.model_config_path, args.model_def_path,
            args.controlnet_ckpt, args.vae_checkpoint, out,
            n_samples=args.infer_n_samples, prep_dir=args.prep_dir,
            datalist_file=args.datalist_file,
            mask_max_masks=args.mask_max_masks, mask_min_frac=args.mask_min_frac,
            mask_max_frac=args.mask_max_frac, diffusion_ckpt=args.diffusion_ckpt,
        )
    elif args.mode == "infer":
        assert args.controlnet_ckpt, "--controlnet_ckpt required for infer mode"
        assert args.vae_checkpoint, "--vae_checkpoint required for infer mode"
        assert args.lesion_prep_dir, "--lesion_prep_dir required for infer mode"
        out = args.output_dir or "./outputs/controlnet_odelia/inference_standalone"
        inference_standalone(
            args.env_config_path, args.model_config_path, args.model_def_path,
            args.controlnet_ckpt, args.vae_checkpoint, out,
            args.lesion_prep_dir, n_samples=args.infer_n_samples,
            window_frac=args.infer_window_frac,
            strategy=args.infer_strategy, diffusion_ckpt=args.diffusion_ckpt,
        )
    elif args.mode == "infer_mama_mia":
        assert args.controlnet_ckpt, "--controlnet_ckpt required for infer_mama_mia mode"
        assert args.vae_checkpoint, "--vae_checkpoint required for infer_mama_mia mode"
        assert args.mama_mia_dir, "--mama_mia_dir required for infer_mama_mia mode"
        out = args.output_dir or "./outputs/controlnet_odelia/inference_mama_mia"
        inference_mama_mia(
            args.env_config_path, args.model_config_path, args.model_def_path,
            args.controlnet_ckpt, args.vae_checkpoint, out,
            args.mama_mia_dir, n_samples=args.infer_n_samples,
            strategy=args.infer_strategy, window_frac=args.infer_window_frac,
            diffusion_ckpt=args.diffusion_ckpt,
            modality_key=args.infer_modality,
        )
