# fine_tuning/finetune_decoder.py
"""
Fine-tune the SAM2 video predictor on paired ultrasound images and binary masks.

This script is intended for ultrasound-domain adaptation of SAM2 in the
ablation-monitoring pipeline. The SAM2 image encoder/backbone is frozen, while
the prompt- and mask-related components are trained using simulated point
prompts generated from the ground-truth masks.

The script assumes that the image and mask folders have matching relative
subfolder structures and matching file names.

Important:
    The SAM2 configuration and initialization checkpoint used here should match
    the SAM2 backbone used in the monitoring application. The default settings
    below use the SAM2.1 large model, consistent with the default app.py setup.
"""

import os
import glob
import json
import random
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


class USMaskDataset(Dataset):
    """Ultrasound image-mask dataset with mirrored image and annotation folders."""

    def __init__(self, img_root, mask_root, size=1024):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)

        exts = ("*.jpg", "*.jpeg", "*.png")
        img_paths = []
        for e in exts:
            img_paths += glob.glob(str(self.img_root / "**" / e), recursive=True)
        self.img_paths = sorted(img_paths)

        self.tr_img = T.Compose([T.Resize(size, antialias=True), T.ToTensor()])
        self.tr_mask = T.Compose([T.Resize(size, antialias=True), T.ToTensor()])
        self.aug_color = T.ColorJitter(0.1, 0.1, 0.1, 0.05)
        self.size = size

    def __len__(self):
        return len(self.img_paths)

    def _find_mask_for(self, img_path: str) -> Path:
        """Find the corresponding annotation mask using the same relative path."""
        ip = Path(img_path)
        rel = ip.relative_to(self.img_root)
        stem = rel.stem
        subdir = rel.parent

        for ext in (".png", ".jpg", ".jpeg"):
            cand = self.mask_root / subdir / f"{stem}{ext}"
            if cand.exists():
                return cand

        fallback = self.mask_root / subdir / rel.name
        if fallback.exists():
            return fallback

        raise FileNotFoundError(f"Mask not found for image: {img_path}")

    def __getitem__(self, i):
        ip = self.img_paths[i]
        mp = self._find_mask_for(ip)

        img = Image.open(ip).convert("RGB")
        m = Image.open(mp).convert("L")

        img = self.tr_img(img)
        m = self.tr_mask(m)

        # Light intensity augmentation improves robustness to ultrasound contrast variation.
        if random.random() < 0.8:
            img = self.aug_color(img)

        # Low-amplitude noise approximates frame-to-frame ultrasound speckle variation.
        if random.random() < 0.5:
            noise = torch.randn_like(img[:1]) * 0.04
            img = torch.clamp(img + noise, 0, 1)

        m = (m > 0.5).float()

        # Simulate one positive prompt from the annotated foreground region.
        ys, xs = torch.where(m[0] > 0.5)
        if xs.numel() > 0:
            idx = int(torch.randint(0, xs.numel(), (1,)).item())
            pos = torch.tensor([xs[idx].item(), ys[idx].item()], dtype=torch.float32)
        else:
            H, W = m.shape[-2], m.shape[-1]
            pos = torch.tensor([W // 2, H // 2], dtype=torch.float32)

        # Simulate one nearby negative prompt and keep it inside the image boundary.
        H, W = m.shape[-2], m.shape[-1]
        neg = pos + torch.tensor(
            [random.randint(-40, 40), random.randint(-40, 40)],
            dtype=torch.float32,
        )
        neg = torch.stack([
            torch.clamp(neg[0], 0, W - 1),
            torch.clamp(neg[1], 0, H - 1),
        ])

        return {
            "image": img,
            "mask": m,
            "points": torch.stack([pos, neg], dim=0),
            "labels": torch.tensor([1, 0], dtype=torch.int32),
        }


class DiceBCELoss(nn.Module):
    """Combined BCE-with-logits and soft Dice loss for binary mask prediction."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, target):
        bce = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * target).sum(dim=(1, 2, 3)) + self.smooth
        den = (probs + target).sum(dim=(1, 2, 3)) + self.smooth
        dice = 1.0 - (num / den).mean()

        return bce + dice


def _freeze_image_encoder_like_things(module):
    """Freeze modules corresponding to the SAM2 image encoder or visual backbone."""
    frozen = 0

    for name, sub in module.named_modules():
        if any(k in name.lower() for k in ["image_encoder", "backbone", "vision_encoder"]):
            for p in sub.parameters(recurse=False):
                p.requires_grad = False
                frozen += 1

    for attr in ["image_encoder", "backbone", "vision_encoder"]:
        if hasattr(module, attr):
            for p in getattr(module, attr).parameters():
                p.requires_grad = False
                frozen += 1

    return frozen


def _collect_decoder_prompt_params(module):
    """Collect trainable parameters from prompt- and mask-related SAM2 modules."""
    cand_modules = []

    for attr in ["mask_decoder", "prompt_encoder", "iou_pred", "decoder", "mask_head"]:
        if hasattr(module, attr):
            cand_modules.append(getattr(module, attr))

    if hasattr(module, "model"):
        m = module.model
        for attr in ["mask_decoder", "prompt_encoder", "iou_pred", "decoder", "mask_head"]:
            if hasattr(m, attr):
                cand_modules.append(getattr(m, attr))

        if hasattr(m, "sam"):
            s = m.sam
            for attr in ["mask_decoder", "prompt_encoder", "iou_pred", "decoder", "mask_head"]:
                if hasattr(s, attr):
                    cand_modules.append(getattr(s, attr))

    params, names = [], []
    for mod in cand_modules:
        for n, p in mod.named_parameters(recurse=True):
            params.append(p)
            names.append(f"{type(mod).__name__}.{n}")

    params = list({id(p): p for p in params}.values())
    return params, names


def _fallback_non_image_params(module):
    """Fallback parameter selection when module names differ across SAM2 versions."""
    params, names = [], []

    for n, p in module.named_parameters():
        if any(k in n.lower() for k in ["image_encoder", "backbone", "vision_encoder"]):
            continue
        params.append(p)
        names.append(n)

    return params, names


def _patch_predictor_objscore(predictor, device):
    """
    Patch SAM2 predictor output for versions that do not always return object scores.

    Some SAM2 builds expect `object_score_logits` in downstream outputs. When the
    key is absent, a zero-valued tensor is inserted so that training can continue
    without modifying the SAM2 source code.
    """
    if not hasattr(predictor, "_run_single_frame_inference"):
        return

    orig = predictor._run_single_frame_inference

    def wrapped(*args, **kwargs):
        out, aux = orig(*args, **kwargs)

        if isinstance(out, dict) and "object_score_logits" not in out:
            dev = device
            for v in out.values():
                if torch.is_tensor(v):
                    dev = v.device
                    break
                if isinstance(v, (list, tuple)) and len(v) and torch.is_tensor(v[0]):
                    dev = v[0].device
                    break
            out["object_score_logits"] = torch.zeros(1, 1, device=dev)

        return out, aux

    predictor._run_single_frame_inference = wrapped


def forward_one_sample_via_tempdir(predictor, img_tensor, points, labels):
    """
    Run a gradient-enabled one-frame SAM2 forward pass.

    The SAM2 video predictor expects a video directory. For each training sample,
    this function temporarily writes the image as a one-frame sequence, initializes
    the predictor state, converts point prompts to SAM2 coordinates, and directly
    calls the internal tracking step to obtain trainable mask logits.
    """
    tmpdir = tempfile.mkdtemp(prefix="sam2_oneframe_")

    try:
        img_np = (
            img_tensor.clamp(0, 1)
            .mul(255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        Image.fromarray(img_np).save(os.path.join(tmpdir, "000000.jpg"))

        state = predictor.init_state(video_path=tmpdir)
        device = state["device"]

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)

        if points.dim() == 2:
            points = points.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        video_H, video_W = state["video_height"], state["video_width"]

        points = points.to(device, dtype=torch.float32)
        labels = labels.to(device)

        scale = torch.tensor([video_W, video_H], dtype=torch.float32, device=device)
        pts = (points / scale) * float(predictor.image_size)

        point_inputs = {
            "point_coords": pts,
            "point_labels": labels,
        }

        (
            _img_b,
            _img_embed,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = predictor._get_image_feature(state, frame_idx=0, batch_size=1)

        current_out = predictor.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=None,
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )

        logits = current_out["pred_masks"]

        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        elif logits.size(1) > 1:
            logits = logits[:, :1, ...]

        target_H, target_W = img_tensor.shape[-2], img_tensor.shape[-1]
        if logits.shape[-2:] != (target_H, target_W):
            logits = F.interpolate(
                logits,
                size=(target_H, target_W),
                mode="bilinear",
                align_corners=False,
            )

        return logits

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _write_training_metadata(
    out_ckpt,
    model_cfg,
    init_checkpoint,
    size,
    epochs,
    lr,
    best_iou,
    best_threshold,
):
    """
    Save the metadata file used by app.py to verify checkpoint compatibility.

    The monitoring app reads this file and checks that `model_cfg` matches the
    selected SAM2 configuration before loading the fine-tuned heads.
    """
    meta_path = out_ckpt.replace(".pt", "_meta.json")

    metadata = {
        "model_cfg": Path(model_cfg).name,
        "init_checkpoint": Path(init_checkpoint).name,
        "backbone": "sam2.1_hiera_large",
        "image_size": int(size),
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "best_val_iou": float(best_iou),
        "best_threshold": float(best_threshold),
        "training_strategy": "image encoder frozen; prompt and mask components fine-tuned",
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ↳ saved metadata {meta_path}")


def finetune(
    model_cfg: str,
    init_checkpoint: str,
    img_root: str,
    mask_root: str,
    out_ckpt: str,
    size=1024,
    epochs=10,
    lr=1e-4,
    device="cuda",
    best_threshold=0.5,
):
    """Fine-tune SAM2 prompt/mask components on ultrasound image-mask pairs."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(model_cfg, init_checkpoint, device=device)

    frozen = _freeze_image_encoder_like_things(predictor)
    if hasattr(predictor, "model"):
        frozen += _freeze_image_encoder_like_things(predictor.model)
        if hasattr(predictor.model, "sam"):
            frozen += _freeze_image_encoder_like_things(predictor.model.sam)

    print(f"[info] froze ~{frozen} parameter tensors from image/backbone modules")

    train_params, train_names = _collect_decoder_prompt_params(predictor)
    if len(train_params) == 0:
        print(
            "[warn] Could not find decoder/prompt modules by name; "
            "falling back to all non-image-encoder parameters."
        )
        train_params, train_names = _fallback_non_image_params(predictor)

    for p in train_params:
        p.requires_grad = True

    tot_train = sum(p.numel() for p in train_params)
    print(f"[train] trainable tensors: {len(train_params)}  (~{tot_train / 1e6:.2f}M params)")
    print("        examples:", train_names[:8])

    _patch_predictor_objscore(predictor, device)

    ds = USMaskDataset(img_root, mask_root, size=size)
    if len(ds) == 0:
        raise RuntimeError(f"No images found under {img_root}")

    # Batch size is one because the SAM2 video predictor state is initialized per sample.
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    # This quick validation pass uses the same dataset. For final reporting,
    # use a separate validation/test split and report the split explicitly.
    vl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.01)
    loss_fn = DiceBCELoss()

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    best_iou = -1.0
    predictor.train()

    for ep in range(1, epochs + 1):
        predictor.train()
        tot = 0.0

        for batch in dl:
            img = batch["image"].to(device)[0]
            mgt = batch["mask"].to(device)[0:1]
            pts = batch["points"].to(device)[0]
            lbs = batch["labels"].to(device)[0]

            opt.zero_grad()
            logits = forward_one_sample_via_tempdir(predictor, img, pts, lbs)
            loss = loss_fn(logits, mgt)
            loss.backward()
            opt.step()

            tot += loss.item()

        predictor.eval()
        with torch.no_grad():
            inter = 0.0
            union = 0.0

            for batch in vl:
                img = batch["image"].to(device)[0]
                mgt = batch["mask"].to(device)[0:1]
                pts = batch["points"].to(device)[0]
                lbs = batch["labels"].to(device)[0]

                logits = forward_one_sample_via_tempdir(predictor, img, pts, lbs)
                if logits.dim() == 3:
                    logits = logits.unsqueeze(0)

                pr = (torch.sigmoid(logits) > best_threshold).float()
                inter += (pr * mgt).sum().item()
                union += ((pr + mgt) > 0).float().sum().item()

            iou = inter / max(1e-6, union)

        print(f"[ep {ep:02d}] train_loss={tot / len(dl):.4f}  valIoU={iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            torch.save(predictor.state_dict(), out_ckpt)
            print(f"  ↳ saved {out_ckpt} (best IoU {best_iou:.4f})")

            _write_training_metadata(
                out_ckpt=out_ckpt,
                model_cfg=model_cfg,
                init_checkpoint=init_checkpoint,
                size=size,
                epochs=epochs,
                lr=lr,
                best_iou=best_iou,
                best_threshold=best_threshold,
            )


if __name__ == "__main__":
    # These defaults use SAM2.1 large, matching the default app.py configuration.
    # Update the local paths as needed before running the script on a new machine.
    model_cfg = "C:/Users/a3taghip/samproject/sam2-main/configs/sam2.1/sam2.1_hiera_l.yaml"
    init_checkpoint = "C:/Users/a3taghip/samproject/sam2-main/checkpoints/sam2.1_hiera_large.pt"

    img_root = "C:/Users/a3taghip/sam2/sam2/data/Annotations/Images"
    mask_root = "C:/Users/a3taghip/sam2/sam2/data/Annotations/Annotations"

    out_ckpt = "C:/Users/a3taghip/sam2/runs/finetuned/decoder_ultrasound.pt"

    finetune(
        model_cfg,
        init_checkpoint,
        img_root,
        mask_root,
        out_ckpt,
        size=1024,
        epochs=10,
        lr=1e-4,
        device="cuda",
        best_threshold=0.5,
    )
