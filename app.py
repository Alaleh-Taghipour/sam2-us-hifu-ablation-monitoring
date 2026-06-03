
# app.py
# Interactive SAM2-based ultrasound ablation monitoring pipeline.
# The script launches a Gradio interface for prompt-based SAM2 video segmentation,
# applies ultrasound-specific mask refinement, exports contour/focal-point CSV files,
# evaluates segmentation metrics, and supports motion-aware treatment-plan updates.

# Core numerical, visualization, image-processing, and UI dependencies.
import csv
import glob
import importlib.util
import json
import os
import queue
import re
import shutil
import sys
import time
import traceback
from threading import Thread

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Response, jsonify, request
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor



# Directory used for all runtime outputs: contours, focal points, overlays, and metrics.
PREDICTIONS_DIR = r"C:\Users\a3taghip\sam2\sam2_pipeline\out_csv"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
print(f"[✔] Predictions will be saved in: {PREDICTIONS_DIR}")

# Optional heads-only SAM2 checkpoint trained on the ultrasound ablation dataset.
# The corresponding metadata file stores the compatible SAM2 config and best mask threshold.
FINETUNED_CKPT = r"C:\Users\a3taghip\sam2\runs\finetuned\decoder_ultrasound.pt"
FINETUNED_META = FINETUNED_CKPT.replace(".pt", "_meta.json")
# Runtime weight selection. The value is assigned from the Gradio radio button.
USE_FINETUNED = None
# Default mask threshold. It is overwritten when a compatible fine-tuning metadata file is loaded.
BEST_THR = 0.5

# Motion-aware update thresholds used during live monitoring.
# Tier 1 applies a geometric shift; Tiers 2 and 3 re-run segmentation on the reference frame.
MOTION_POLL_SECS = 20.0
TIER1_MAX = 2.0
TIER2_MAX = 6.0

# Stable live-plan filename read by the downstream robotic targeting workflow.
PLAN_LIVE_NAME = "plan_points_live.csv"
PLAN_LIVE = os.path.join(PREDICTIONS_DIR, PLAN_LIVE_NAME)


# Contour coordinates are scaled before CSV export to match the physical coordinate convention.
SCALE_CONTOUR = 1e-3
CSV_FLOAT_FMT = "{:.6f}"
Z_CONST = 0.380000

# Shared motion state updated from the Gradio interface and consumed by the motion gate.
latest_motion = {"d": None, "dx": None, "dy": None, "ts": 0.0}

# File-management utilities used for CSV copying, discovery, and focal-point updates.


# -----------------------------------------------------------------------------
# CSV and geometry utilities
# -----------------------------------------------------------------------------
# Return the most recently modified CSV in a directory.
def _find_latest_csv(dirpath):
    csvs = [os.path.join(dirpath, f) for f in os.listdir(dirpath)
            if f.lower().endswith(".csv")]
    if not csvs: return None
    csvs.sort(key=lambda p: os.path.getmtime(p))
    return csvs[-1]

# Copy a CSV only when the source exists; this keeps update steps fail-safe.
def _copy_csv(src, dst):
    if src and os.path.isfile(src):
        shutil.copyfile(src, dst)

# Lightweight numeric check used when detecting x/y columns in flexible CSV files.
def _is_float(v):
    try:
        float(v)
        return True
    except Exception:
        return False


# Shift boundary or focal-point coordinates by the measured tissue/target motion.
def _shift_csv(in_csv, out_csv, dx_mm, dy_mm):
    """
    Generic shifter:
      - works for boundary CSV ("x","y") AND focal-points CSV
      - tries standard names first, then falls back to first two numeric columns
    """
    # Convert GUI motion input from millimetres to the same scaled units used in CSV output.
    dx_add = dx_mm * SCALE_CONTOUR
    dy_add = dy_mm * SCALE_CONTOUR

    with open(in_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    # Empty files are preserved so that downstream scripts do not fail unexpectedly.
    if not rows:
        print(f"[motion] {in_csv}: empty CSV, nothing to shift")
        shutil.copyfile(in_csv, out_csv)
        return

    # Prefer common x/y column names, but allow a numeric-column fallback for compatibility.
    cand_x = ["x_mm", "X_mm", "x", "X"]
    cand_y = ["y_mm", "Y_mm", "y", "Y"]

    xk = next((k for k in cand_x if k in fieldnames), None)
    yk = next((k for k in cand_y if k in fieldnames), None)

    # If named columns are not available, use the first two numeric columns as x and y.
    if xk is None or yk is None:
        for r in rows:
            numeric_keys = [k for k in fieldnames if _is_float(r.get(k, ""))]
            if len(numeric_keys) >= 2:
                xk, yk = numeric_keys[0], numeric_keys[1]
                break

    # If named columns are not available, use the first two numeric columns as x and y.
    if xk is None or yk is None:
        print(f"[motion] {in_csv}: cannot find numeric x/y columns, leave unchanged")
        shutil.copyfile(in_csv, out_csv)
        return

    # Apply the same in-plane shift to every valid coordinate row.
    for r in rows:
        if _is_float(r.get(xk, "")):
            r[xk] = f"{float(r[xk]) + dx_add:.6f}"
        if _is_float(r.get(yk, "")):
            r[yk] = f"{float(r[yk]) + dy_add:.6f}"

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[motion] shifted {os.path.basename(in_csv)} using columns ({xk}, {yk})")



# Apply the same motion correction to all focal-point CSV files in the output directory.
def _shift_all_focal_points(dx_mm, dy_mm):
    """
    Shift all focal-point CSVs (mask_contours_frame*_obj*_focal_points_1mm.csv)
    by the same motion (dx_mm, dy_mm) in mm.
    """
    pattern = os.path.join(PREDICTIONS_DIR, "*_focal_points_1mm.csv")
    focal_files = glob.glob(pattern)
    if not focal_files:
        print("[motion] no *_focal_points_1mm.csv files to shift")
        return

    for fp in focal_files:
        tmp = fp + ".tmp"
        _shift_csv(fp, tmp, dx_mm, dy_mm)
        os.replace(tmp, fp)
        print(f"[motion] shifted focal points file: {os.path.basename(fp)}")


# Determine the next geometry index so each monitoring update is stored as a new snapshot.
def _next_geom_index():
    """
    Find the next available geometry index for:
      mask_contours_frame{N}_obj1.csv
    so we don't overwrite existing ones.
    """
    pattern = os.path.join(PREDICTIONS_DIR, "mask_contours_frame*_obj1.csv")
    files = glob.glob(pattern)
    max_idx = 0
    for fp in files:
        base = os.path.basename(fp)
        # Ignore files that do not follow the expected frame-index naming pattern.
        try:
            mid = base.split("frame")[1].split("_obj")[0]
            idx = int(mid)
            if idx > max_idx:
                max_idx = idx
        except Exception:
            continue
    return max_idx + 1


# Save the current live boundary and focal points as a versioned geometry snapshot.
def _snapshot_current_geometry(dx_mm=None, dy_mm=None):
    """
    Take the CURRENT live plan + latest focal_points_1mm and
    save them as a new version:

      mask_contours_frame{N}_obj1.csv
      mask_contours_frame{N}_obj1_focal_points_1mm.csv

    If dx_mm, dy_mm are provided (in mm), the focal-points snapshot
    is shifted by that amount before saving.
    """
    idx = _next_geom_index()

    # Snapshot the current boundary plan first.
    if os.path.isfile(PLAN_LIVE):
        dst_boundary = os.path.join(
            PREDICTIONS_DIR, f"mask_contours_frame{idx}_obj1.csv"
        )
        _copy_csv(PLAN_LIVE, dst_boundary)
        print(f"[geom] snapshot boundary → {dst_boundary}")
    else:
        print("[geom] PLAN_LIVE not found; no boundary snapshot")

    # Then snapshot the latest focal-point plan, optionally shifted by the same motion update.
    focal_files = glob.glob(os.path.join(PREDICTIONS_DIR, "*focal_points_1mm.csv"))
    if focal_files:
        focal_files.sort(key=os.path.getmtime)
        src_focal = focal_files[-1]
        dst_focal = os.path.join(
            PREDICTIONS_DIR, f"mask_contours_frame{idx}_obj1_focal_points_1mm.csv"
        )

        if dx_mm is not None and dy_mm is not None:
            _shift_csv(src_focal, dst_focal, dx_mm, dy_mm)
            print(
                f"[geom] snapshot focal (shifted by dx={dx_mm:.3f} mm, "
                f"dy={dy_mm:.3f} mm) → {dst_focal}"
            )
        else:
            _copy_csv(src_focal, dst_focal)
            print(f"[geom] snapshot focal → {dst_focal}")
    else:
        print("[geom] no focal_points_1mm.csv found; no focal snapshot")

    # Keep nearest-neighbour ordered focal-point files synchronized with the snapshots.
    _build_all_nn_focals()

    return idx


# Add a constant z coordinate to focal-point CSVs when the helper script outputs only x and y.
def _ensure_z_column(csv_path, z_val=Z_CONST):
    """Add a z column with constant value if it doesn't exist yet."""
    import csv, tempfile

    if not os.path.isfile(csv_path):
        return

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return

    header = rows[0]
    # Existing z columns are preserved to avoid overwriting manually prepared files.
    if any(h.lower() == "z" for h in header):
        return

    z_str = CSV_FLOAT_FMT.format(z_val)
    header.append("z")
    for i in range(1, len(rows)):
        while len(rows[i]) < len(header) - 1:
            rows[i].append("")
        rows[i].append(z_str)

    tmp_path = csv_path + ".tmpz"
    with open(tmp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    os.replace(tmp_path, csv_path)
    print(f"[z-column] added z={z_str} to {os.path.basename(csv_path)}")


# Reorder focal points with a greedy nearest-neighbour path for smoother robotic traversal.
def _nn_reorder_single_focal_csv(in_csv, out_csv):
    """
    Read a focal-points CSV (x,y,...) and write an NN-reordered version to out_csv.
    If anything looks weird, just copy the file unchanged.
    """
    try:
        with open(in_csv, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        # Preserve empty files rather than failing the full batch update.
        if not rows:
            shutil.copyfile(in_csv, out_csv)
            print(f"[NN] {os.path.basename(in_csv)}: empty, copied as-is.")
            return

        # The focal-point files are expected to contain x/y columns in scaled physical units.
        cand_x = ["x_mm", "X_mm", "x", "X"]
        cand_y = ["y_mm", "Y_mm", "y", "Y"]

        xk = next((k for k in cand_x if k in fieldnames), None)
        yk = next((k for k in cand_y if k in fieldnames), None)

        if xk is None or yk is None:
            print(f"[NN] {os.path.basename(in_csv)}: x/y columns not found, copied as-is.")
            shutil.copyfile(in_csv, out_csv)
            return

        # Collect only rows with valid numeric coordinates before ordering.
        idxs = []
        xs = []
        ys = []
        for i, r in enumerate(rows):
            vx = r.get(xk, "")
            vy = r.get(yk, "")
            if not _is_float(vx) or not _is_float(vy):
                continue
            idxs.append(i)
            xs.append(float(vx))
            ys.append(float(vy))

        n = len(idxs)
        if n <= 1:
            shutil.copyfile(in_csv, out_csv)
            print(f"[NN] {os.path.basename(in_csv)}: <=1 numeric point, copied as-is.")
            return

        # Greedy ordering starts from the first point and repeatedly selects the closest unused point.
        used = [False] * n
        order_local = [0]
        used[0] = True

        for _ in range(n - 1):
            last = order_local[-1]
            best_j = None
            best_d2 = None
            for j in range(n):
                if used[j]:
                    continue
                dx = xs[j] - xs[last]
                dy = ys[j] - ys[last]
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_j = j
            if best_j is None:
                break
            used[best_j] = True
            order_local.append(best_j)

        # Convert local nearest-neighbour indices back to the original CSV row indices.
        ordered_row_indices = [idxs[j] for j in order_local]
        ordered_rows = [rows[i] for i in ordered_row_indices]

        # Write the ordered focal-point sequence using the original header.
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered_rows)

        print(f"[NN] wrote NN-reordered focal points → {os.path.basename(out_csv)}")

    except Exception as e:
        print(f"[NN] ERROR on {os.path.basename(in_csv)}:", e)
        traceback.print_exc()
        # If ordering fails, fall back to copying the original focal-point file.
        try:
            shutil.copyfile(in_csv, out_csv)
        except Exception:
            pass


# Refresh nearest-neighbour ordered copies for all focal-point CSV files.
def _build_all_nn_focals():
    """
    For every mask_contours_frame*_obj*_focal_points_1mm.csv in PREDICTIONS_DIR,
    create/refresh an NN-ordered version:

      mask_contours_frameN_objM_focal_points_1mm.csv
        → NN_frameN_objM_focal_points_1mm.csv  (preferred naming)

    If the NN file already exists and is newer than the source, skip it.
    """
    pattern = os.path.join(PREDICTIONS_DIR, "*_focal_points_1mm.csv")
    all_focals = glob.glob(pattern)
    if not all_focals:
        print("[NN] no *_focal_points_1mm.csv files found.")
        return

    for fp in all_focals:
        base = os.path.basename(fp)

        # Avoid recursively reordering files that were already generated by this function.
        if base.startswith("NN_"):
            continue

        # Use a clean output name when the source follows the expected mask-contour naming pattern.
        m = re.match(r"mask_contours_frame(\d+)_obj(\d+)_focal_points_1mm\.csv", base)
        if m:
            nn_name = f"NN_frame{m.group(1)}_obj{m.group(2)}_focal_points_1mm.csv"
        else:
            nn_name = "NN_" + base

        nn_path = os.path.join(PREDICTIONS_DIR, nn_name)

        # Skip files whose NN copy is already current.
        try:
            if os.path.exists(nn_path) and os.path.getmtime(nn_path) >= os.path.getmtime(fp):
                print(f"[NN] up-to-date: {nn_name}")
                continue
        except Exception:
            pass

        _nn_reorder_single_focal_csv(fp, nn_path)

# Export mask contours as scaled x/y coordinates with a constant z plane for robot-side use.
def _write_contours_csv(contours_list, out_path):
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Each contour vertex is stored in the same CSV format expected by the targeting pipeline.
        writer.writerow(["x", "y", "z"])
        z_str = CSV_FLOAT_FMT.format(Z_CONST)
        for contour in contours_list:
            for x_, y_ in contour:
                xs = CSV_FLOAT_FMT.format(float(x_) * SCALE_CONTOUR)
                ys = CSV_FLOAT_FMT.format(float(y_) * SCALE_CONTOUR)
                writer.writerow([xs, ys, z_str])





# -----------------------------------------------------------------------------
# Prediction-vs-ground-truth overlays and segmentation metrics
# -----------------------------------------------------------------------------
def _save_overlay_and_metrics_for_frame(frame_idx: int, background_png_path: str):
    """
    Build predicted mask from saved contours JSON, compare with GT in out_csv/masks,
    draw FP/FN/TP overlay on the given background image, and save metrics JSON.
    Writes:
      - overlay_cmp_frame{idx}.png
      - metrics_frame{idx}.json
    """
    try:
        # Load the original B-mode frame as the visualization background.
        bg = cv2.imread(background_png_path, cv2.IMREAD_COLOR)
        if bg is None:
            return "overlay: no background"
        H, W = bg.shape[:2]

        # Reconstruct the predicted binary mask from the saved contour JSON file.
        json_path_exact = os.path.join(PREDICTIONS_DIR, f"mask_contours_frame{frame_idx}_obj1.json")
        if not os.path.isfile(json_path_exact):
            # If the exact frame file is unavailable, use the newest contour file as a fallback.
            cands = glob.glob(os.path.join(PREDICTIONS_DIR, "mask_contours_frame*_obj*.json"))
            if not cands:
                return "overlay: no pred json"
            cands.sort(key=os.path.getmtime)
            json_path_exact = cands[-1]

        with open(json_path_exact, "r") as f:
            contours_list = json.load(f)
        pred = np.zeros((H, W), dtype=np.uint8)
        pts = [np.asarray(c, dtype=np.int32).reshape(-1,2) for c in contours_list if len(c)]
        if pts:
            cv2.fillPoly(pred, pts, 255)
        pred = (pred > 0).astype(np.uint8)

        # Ground-truth masks are expected in PREDICTIONS_DIR/masks for metric calculation.
        gt_dir = os.path.join(PREDICTIONS_DIR, "masks")
        gt_cands = glob.glob(os.path.join(gt_dir, "*.png")) + glob.glob(os.path.join(gt_dir, "*.jpg"))
        if not gt_cands:
            return "overlay: no GT masks"
        # Prefer the ground-truth mask whose filename contains the current frame index.
        gt_path = next((p for p in gt_cands if str(frame_idx) in os.path.basename(p)), None)
        if gt_path is None:
            gt_cands.sort(key=os.path.getmtime)
            gt_path = gt_cands[-1]
        # Read alpha masks without discarding transparency; this avoids edge artifacts in PNG masks.
        gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt_raw is None:
            return "overlay: failed to read GT"

        # Use alpha when available, otherwise convert color masks to grayscale.
        if gt_raw.ndim == 3 and gt_raw.shape[2] == 4:
            gt_gray = gt_raw[:, :, 3]
        elif gt_raw.ndim == 3:
            gt_gray = cv2.cvtColor(gt_raw, cv2.COLOR_BGR2GRAY)
        else:
            gt_gray = gt_raw

        # A hard threshold avoids JPEG halos; Otsu is used only if the mask is not recovered.
        _, gt_bin = cv2.threshold(gt_gray, 127, 255, cv2.THRESH_BINARY)
        if gt_bin.sum() == 0:
            _, gt_bin = cv2.threshold(gt_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Nearest-neighbour resizing preserves binary mask boundaries.
        gt_bin = cv2.resize(gt_bin, (W, H), interpolation=cv2.INTER_NEAREST)

        # Small morphology suppresses isolated speckles and fills tiny pinholes in GT masks.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gt_bin = cv2.morphologyEx(gt_bin, cv2.MORPH_OPEN,  k, iterations=1)
        gt_bin = cv2.morphologyEx(gt_bin, cv2.MORPH_CLOSE, k, iterations=1)

        gt = (gt_bin > 0).astype(np.uint8)


        tp = ((pred==1) & (gt==1)).astype(np.uint8)
        fp = ((pred==1) & (gt==0)).astype(np.uint8)
        fn = ((pred==0) & (gt==1)).astype(np.uint8)

        # Overlay helper used to mark false positives, false negatives, and true positives.
        def blend(img, mask, color_bgr, a):
            m3 = np.repeat(mask[:, :, None], 3, axis=2)
            color = np.zeros_like(img, dtype=np.uint8); color[:] = color_bgr
            return np.where(m3==1, cv2.addWeighted(img, 1.0-a, color, a, 0), img)

        vis = bg.copy()
        vis = blend(vis, fp, (0,0,255), 0.45)
        vis = blend(vis, fn, (255,0,0), 0.45)
        vis = blend(vis, tp, (0,255,0), 0.25)

        out_png = os.path.join(PREDICTIONS_DIR, f"overlay_cmp_frame{frame_idx}.png")
        cv2.imwrite(out_png, vis)

        # Pixel counts used for all reported segmentation metrics.
        tpN = int(tp.sum())
        fpN = int(fp.sum())
        fnN = int(fn.sum())
        tnN = int(((pred==0) & (gt==0)).sum())

        # Denominators are clamped to avoid division-by-zero on empty masks.
        den_iou  = max(1, (tpN + fpN + fnN))
        den_dice = max(1, (2*tpN + fpN + fnN))
        den_all  = max(1, (tpN + fpN + fnN + tnN))

        iou   = tpN / den_iou
        dice  = (2*tpN) / den_dice
        acc   = (tpN + tnN) / den_all
        prec  = tpN / max(1, (tpN + fpN))
        rec   = tpN / max(1, (tpN + fnN))
        loss_iou  = 1.0 - iou
        loss_dice = 1.0 - dice

        # Report percentages in the console for quick inspection during experiments.
        print(
            f"[metrics][frame {frame_idx}] "
            f"IoU={iou*100:.2f}%  Dice={dice*100:.2f}%  "
            f"Acc={acc*100:.2f}%  Prec={prec*100:.2f}%  Rec={rec*100:.2f}%  "
            f"Loss(IoU)={loss_iou*100:.2f}%  Loss(Dice)={loss_dice*100:.2f}%"
        )

        # Save raw metric values and source filenames for later analysis.
        with open(os.path.join(PREDICTIONS_DIR, f"metrics_frame{frame_idx}.json"), "w") as f:
            json.dump({
                "IoU": float(iou), "Dice": float(dice), "Accuracy": float(acc),
                "Precision": float(prec), "Recall": float(rec),
                "Loss_IoU": float(loss_iou), "Loss_Dice": float(loss_dice),
                "tp": tpN, "fp": fpN, "fn": fnN, "tn": tnN,
                "pred_json": os.path.basename(json_path_exact),
                "gt": os.path.basename(gt_path)
            }, f, indent=2)

        return "overlay: ok"

    except Exception as e:
        return f"overlay err: {e}"


# -----------------------------------------------------------------------------
# Fine-tuned SAM2 head loading
# -----------------------------------------------------------------------------
# Resolve the underlying SAM2 module regardless of predictor wrapper naming.
def _get_inner_model(obj):
    for name in ["model", "sam", "net", "_model", "sam_model", "module"]:
        if hasattr(obj, name):
            return getattr(obj, name)
    return obj

# Confirm that the fine-tuned heads were trained with the selected SAM2 configuration.
def _is_meta_compatible(current_cfg_path: str):
    """Return (bool_ok, why_str). Requires meta json to contain 'model_cfg'."""
    try:
        if not os.path.isfile(FINETUNED_META):
            return False, "no meta"
        with open(FINETUNED_META, "r") as f:
            meta = json.load(f)
        finetune_cfg = str(meta.get("model_cfg", "")).replace("\\", "/").split("/")[-1]
        current_cfg  = str(current_cfg_path).replace("\\", "/").split("/")[-1]
        return (finetune_cfg == current_cfg), f"finetune_cfg={finetune_cfg}, current_cfg={current_cfg}"
    except Exception as e:
        return False, f"meta error: {e}"

# Load only the prompt encoder and mask decoder from the ultrasound fine-tuned checkpoint.
def _load_finetuned_heads_into_predictor(predictor, device, verbose=True, current_cfg_path=None):
    """Load prompt/mask heads and best threshold only if compatible with current backbone."""
    global BEST_THR
    if not os.path.isfile(FINETUNED_CKPT):
        msg = f"[fine-tune] ckpt not found, using base weights: {FINETUNED_CKPT}"
        if verbose: print(msg)
        return msg

    ok, why = _is_meta_compatible(current_cfg_path or "")
    if not ok:
        msg = f"[fine-tune] SKIP: finetuned heads are incompatible with this backbone ({why})."
        if verbose: print(msg)
        return msg

    try:
        sd = torch.load(FINETUNED_CKPT, map_location=device)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]

        # Keep the heads-only adaptation and leave the frozen SAM2 image backbone unchanged.
        filtered = {k: v for k, v in sd.items()
                    if ("prompt_encoder" in k) or ("mask_decoder" in k)}

        target = _get_inner_model(predictor)
        missing, unexpected = target.load_state_dict(filtered, strict=False)
        if verbose:
            print("[fine-tune] Loaded heads from:", FINETUNED_CKPT)
            print(f"[fine-tune] Loaded {len(filtered)} tensors (prompt_encoder + mask_decoder).")
            if missing:
                print("[fine-tune] Missing (ignored):",
                      [m for m in missing if ("prompt_encoder" in m or "mask_decoder" in m)])
            if unexpected:
                print("[fine-tune] Unexpected (ignored):",
                      [u for u in unexpected if ("prompt_encoder" in u or "mask_decoder" in u)])
        target.eval()

        # Load the validation-selected threshold used for converting logits into binary masks.
        if os.path.isfile(FINETUNED_META):
            with open(FINETUNED_META, "r") as f:
                info = json.load(f)
                BEST_THR = float(info.get("best_threshold", 0.5))
                if verbose:
                    print(f"[fine-tune] Using best threshold from meta: {BEST_THR:.3f}")
        else:
            if verbose:
                print("[fine-tune] Meta JSON not found; using default threshold 0.5")
        return "Fine-tuned heads loaded"
    except Exception as e:
        print("[fine-tune] Failed to load heads:", e)
        return f"Failed to load heads: {e}"


# -----------------------------------------------------------------------------
# Focal-point generation and ordering
# -----------------------------------------------------------------------------
# External helper script that converts contours/masks into 1 mm focal-point grids.
FOCAL_SCRIPT_PATH = os.path.join(PREDICTIONS_DIR, "make_focal_points_from_masks.py")

# Run the focal-point generator after the first contour and overlay files are available.
def _run_make_focal_points_once():
    """
    Import make_focal_points_from_masks.py as a module and call main().
    Safe to call multiple times: the script itself skips already-processed masks.
    """
    if not os.path.isfile(FOCAL_SCRIPT_PATH):
        print(f"[FOCAL] script not found at: {FOCAL_SCRIPT_PATH}")
        return

    try:
        print(f"[FOCAL] loading script: {FOCAL_SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("focal_module", FOCAL_SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Call main() directly so focal-point generation runs in the same Python process.
        if hasattr(module, "main"):
            print("[FOCAL] calling main() in make_focal_points_from_masks.py ...")
            module.main()
        else:
            print("[FOCAL] module has no main(), nothing to do.")

        # Standardize focal-point files to include z and nearest-neighbour ordered copies.
        pattern = os.path.join(PREDICTIONS_DIR, "*_focal_points_1mm.csv")
        for fp in glob.glob(pattern):
            _ensure_z_column(fp, z_val=Z_CONST)

        _build_all_nn_focals()

        print("[FOCAL] finished successfully.")


    except Exception as e:
        print("[FOCAL] ERROR while running focal-points script:", e)
        traceback.print_exc()




# -----------------------------------------------------------------------------
# Runtime state shared by Gradio callbacks
# -----------------------------------------------------------------------------
predictor = None
inference_state = None
frame_names = []
points = []
labels = []
video_dir = None
stop_prediction = False
box = None


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
# Display a semi-transparent segmentation mask on a matplotlib axis.
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Display SAM2 positive and negative point prompts.
def show_points(coords, labels_arr, ax, marker_size=200):
    pos_points = coords[labels_arr == 1]
    neg_points = coords[labels_arr == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], pos_points[:, 1]*0 + neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# Display the optional bounding-box prompt.
def show_box(b, ax):
    x0, y0, x1, y1 = b
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               edgecolor='green', facecolor='none',
                               linewidth=2))


# -----------------------------------------------------------------------------
# Mask cleanup and contour extraction
# -----------------------------------------------------------------------------
# Convert a binary mask into external OpenCV contours.
def get_contours_from_mask(mask):
    m = np.asarray(mask)
    if m.ndim == 3:
        m = np.squeeze(m)
    if m.ndim != 2:
        m = m[..., 0]
    m = (m > 0).astype(np.uint8) * 255
    m = np.ascontiguousarray(m)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Smooth the predicted ablation mask and retain the dominant connected component.
def _postprocess_mask(binary_mask, min_area_ratio=0.002, close_ks=5):
    """binary_mask: (H,W) {0,1} -> cleaned (H,W) {0,1}"""
    m = (binary_mask.astype(np.uint8) * 255)
    # Closing fills small holes and connects small gaps along the ablation boundary.
    if close_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (m > 127).astype(np.uint8)
    H, W = m.shape[:2]
    min_area = max(1, int(min_area_ratio * H * W))
    # Remove small disconnected regions before selecting the final mask component.
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return (m > 127).astype(np.uint8)
    largest = max(cnts, key=cv2.contourArea)
    m2 = np.zeros_like(m)
    cv2.drawContours(m2, [largest], -1, 255, thickness=-1)
    return (m2 > 127).astype(np.uint8)

# Create a borderless canvas so saved overlays match the original frame dimensions.
def _full_bleed_fig(W, H):
    dpi = 100.0
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    return fig, ax

# Load frames in numeric order when filenames are indexed, otherwise fall back to lexicographic order.
def _list_frames_sorted(folder):
    exts = (".jpg", ".jpeg", ".png")
    names = [p for p in os.listdir(folder) if os.path.splitext(p)[-1].lower() in exts]
    try:
        names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except Exception:
        names.sort()
    return names


# -----------------------------------------------------------------------------
# Motion-aware re-masking and plan update logic
# -----------------------------------------------------------------------------
# Re-segment the reference frame with the current prompt and export an updated contour CSV.
def _remask_frame0_and_write_csv(threshold=0.5):
    """Re-run mask on frame 0 with current prompt, write a new CSV, return its path (or None)."""
    global predictor, inference_state, video_dir, frame_names, points, labels, box
    if predictor is None or not frame_names:
        print("[remask] predictor or frames not ready"); return None

    # Rebuild SAM2 state if needed, then reset before applying the prompt again.
    if inference_state is None:
        inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=True)
    predictor.reset_state(inference_state)

    if (points is None and box is None) or \
       ((points is not None) and (len(points)==0)):
        print("[remask] no prompt available (point/box)"); return None

    # Reuse the same point or box prompt on frame 0.
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
        box=box
    )
    raw = torch.sigmoid(out_mask_logits[0]).cpu().numpy()
    mask = (raw > threshold).astype(np.uint8).squeeze()

    # Apply the same smoothing and largest-component cleanup used in the main propagation path.
    m8 = cv2.medianBlur((mask * 255).astype(np.uint8), 5)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        eps = 0.005 * cv2.arcLength(c, True)
        c2 = cv2.approxPolyDP(c, eps, True)
        sm = np.zeros_like(m8)
        cv2.drawContours(sm, [c2], -1, 255, thickness=-1)
        mask = (sm > 127).astype(np.uint8)
    else:
        mask = (m8 > 127).astype(np.uint8)

    mask = _postprocess_mask(mask)
    contours = get_contours_from_mask(mask)
    contours_list = [c.reshape(-1, 2).tolist() for c in contours]

    # Store the re-masked contour as a separate file before copying it to the live plan.
    out_csv = os.path.join(PREDICTIONS_DIR, f"mask_contours_frame0_obj1_remask.csv")
    _write_contours_csv(contours_list, out_csv)
    print(f"[remask] wrote {out_csv}")
    return out_csv
# Apply a motion-aware update immediately from the UI instead of waiting for the timed gate.
def _apply_motion_now(d, dx, dy):
    """
    Apply tiered action immediately:
      Tier-1 (d <= TIER1_MAX): shift plan by (dx,dy) if provided
      Tier-2 (d <= TIER2_MAX): re-mask frame 0 and refresh plan
      Tier-3 (d >  TIER2_MAX): same as Tier-2 (fallback full replan)
    Returns a human-readable status string.
    """
    try:
        if d is None or (isinstance(d, float) and (np.isnan(d) or d < 0)):
            return "No motion value provided."

        if d <= TIER1_MAX:
            if dx is not None and dy is not None:
                base_csv = PLAN_LIVE if os.path.isfile(PLAN_LIVE) else _find_latest_csv(PREDICTIONS_DIR)
                if not base_csv:
                    return "Tier-1: no CSV to shift yet."

                # Tier 1 keeps the existing segmentation and shifts the live geometry.
                shifted_tmp = os.path.join(PREDICTIONS_DIR, "plan_points_shifted.csv")
                _shift_csv(base_csv, shifted_tmp, dx, dy)
                _copy_csv(shifted_tmp, PLAN_LIVE)

                # Store the shifted boundary and focal points as a new geometry version.
                _snapshot_current_geometry(dx_mm=dx, dy_mm=dy)

                return f"Tier-1: shifted boundary + focal points by ({dx:.2f}, {dy:.2f}) mm"

            else:
                return "Tier-1: dx/dy not provided → plan unchanged."



        # Tiers 2 and 3 refresh the segmentation from the current prompt.
        new_csv = _remask_frame0_and_write_csv(threshold=BEST_THR)
        if new_csv:
            _copy_csv(new_csv, PLAN_LIVE)

            _snapshot_current_geometry()

            return f"Tier-{'2' if d <= TIER2_MAX else '3'}: re-masked frame 0 → {PLAN_LIVE}"
        return "Re-mask failed (no prompt or no frames)."

    except Exception as e:
        return f"Motion apply error: {e}"


# -----------------------------------------------------------------------------
# Gradio callbacks: model initialization and prompt handling
# -----------------------------------------------------------------------------
# Initialize the SAM2 video predictor and optionally load ultrasound fine-tuned heads.
def initialize_model(sam2_checkpoint, model_cfg, video_directory, which_weights):
    global predictor, video_dir, frame_names, USE_FINETUNED
    USE_FINETUNED = which_weights
    try:
        # Use GPU acceleration when available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

        # Fine-tuned loading is optional; errors fall back to the base SAM2 model.
        try:
            # Fine-tuned mode loads only compatible prompt/mask heads.
            if USE_FINETUNED and USE_FINETUNED.lower().startswith("fine"):
                status_ft = _load_finetuned_heads_into_predictor(
                    predictor, device, current_cfg_path=model_cfg
                )
            else:
                status_ft = "Base weights (no finetune)"
            print("[init]", status_ft)
        except Exception as e:
            print("[init] Fine-tune load error:", e)
            status_ft = f"Fine-tune load error: {e}"

        video_dir = video_directory
        # The first frame is shown in the UI so the user can initialize the SAM2 prompt.
        frame_names = _list_frames_sorted(video_dir)
        if not frame_names:
            return None, "No .jpg/.jpeg/.png frames found in the directory."
        first_frame_path = os.path.join(video_dir, frame_names[0])
        first_frame = Image.open(first_frame_path).convert("RGB")
        return first_frame, f"Model initialized. {status_ft}"
    except Exception as e:
        return None, f"Error initializing model: {str(e)}"

# Handle point or box prompt selection on the first ultrasound frame.
def handle_click(image, evt: gr.SelectData, drawing_mode="point"):
    global labels, points, inference_state, box
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)

        img_height, img_width = image.shape[:2]
        x, y = evt.index[0], evt.index[1]
        print(f"Click at ({x}, {y}) - Image size: {img_width}x{img_height}")

        # A point prompt marks the initial ablation region; box mode uses a fixed local window.
        if drawing_mode == "point":
            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            box = None
        else:
            box = np.array([x - 80, y - 80, x + 80, y + 80], dtype=np.float32)
            points = None
            labels = None

        inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=True)
        predictor.reset_state(inference_state)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
            box=box
        )

        # Convert SAM2 logits to a binary mask using the selected/best threshold.
        raw = torch.sigmoid(out_mask_logits[0]).cpu().numpy()
        mask = (raw > BEST_THR).astype(np.uint8).squeeze()

        # Refine the initial mask before displaying it back in the Gradio image panel.
        m8 = cv2.medianBlur((mask * 255).astype(np.uint8), 5)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            eps = 0.005 * cv2.arcLength(c, True)
            c2 = cv2.approxPolyDP(c, eps, True)
            sm = np.zeros_like(m8)
            cv2.drawContours(sm, [c2], -1, 255, thickness=-1)
            mask = (sm > 127).astype(np.uint8)
        else:
            mask = (m8 > 127).astype(np.uint8)

        mask = _postprocess_mask(mask)

        # Render the prompt and refined mask without changing the frame size.
        fig, ax = _full_bleed_fig(img_width, img_height)
        ax.imshow(image)
        if points is not None:
            show_points(points, labels, ax)
        if box is not None:
            show_box(box, ax)
        show_mask(mask, ax, obj_id=out_obj_ids[0])

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf.shape = (h, w, 4)
        img_array = buf[:, :, :3]
        plt.close(fig)
        return img_array

    except Exception as e:
        print(f"Error in handle_click: {str(e)}")
        traceback.print_exc()
        return image

# Kept for compatibility with earlier versions of the app.
save_path = "./"


# -----------------------------------------------------------------------------
# Prediction paths
# -----------------------------------------------------------------------------
# Streaming prediction path retained for Flask-style responses.
def start_prediction():
    """Streaming generator version. Now also saves overlay PNG and triggers focal points."""
    global predictor, inference_state, video_dir, frame_names, points, labels
    print("Starting prediction process...")
    try:
        if inference_state is None:
            print("Inference state not initialized. Initializing now...")
            inference_state = predictor.init_state(video_path=video_dir)
            if not inference_state.get("obj_idx_to_id"):
                print("No objects defined. Please add points first.")
                return jsonify({
                    'status': 'error',
                    'message': 'No objects defined. Please add points first.'
                }), 400

        save_predictions = True
        save_path = os.path.join(os.getcwd(), 'predictions')
        if save_predictions:
            print(f"Creating save directory at: {save_path}")
            os.makedirs(save_path, exist_ok=True)

        def generate():
            try:
                print(f"Starting predictions for {len(frame_names)} frames...")
                # Track the timed motion gate during streaming propagation.
                last_gate = time.time()
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    try:
                        frame_path = os.path.join(video_dir, frame_names[out_frame_idx])
                        image = Image.open(frame_path).convert("RGB")
                        image_array = np.array(image)
                        H, W = image_array.shape[:2]

                        fig, ax = _full_bleed_fig(W, H)
                        ax.imshow(image_array)

                        any_mask_drawn = False
                        for i, out_obj_id in enumerate(out_obj_ids):
                            # Convert frame-level SAM2 logits into a binary mask.
                            raw = torch.sigmoid(out_mask_logits[i]).cpu().numpy()
                            mask = (raw > BEST_THR).astype(np.uint8).squeeze()

                            # Apply edge-aware smoothing before contour extraction.
                            m8 = cv2.medianBlur((mask * 255).astype(np.uint8), 5)
                            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            if cnts:
                                c = max(cnts, key=cv2.contourArea)
                                eps = 0.005 * cv2.arcLength(c, True)
                                c2 = cv2.approxPolyDP(c, eps, True)
                                sm = np.zeros_like(m8)
                                cv2.drawContours(sm, [c2], -1, 255, thickness=-1)
                                mask = (sm > 127).astype(np.uint8)
                            else:
                                mask = (m8 > 127).astype(np.uint8)

                            mask = _postprocess_mask(mask)
                            show_mask(mask, ax, obj_id=out_obj_id)
                            any_mask_drawn = True

                            # Save both contour JSON and scaled CSV for each propagated frame.
                            contours = get_contours_from_mask(mask)
                            contours_list = [c.reshape(-1, 2).tolist() for c in contours]

                            json_path = os.path.join(PREDICTIONS_DIR, f"mask_contours_frame{out_frame_idx}_obj{out_obj_id}.json")
                            with open(json_path, "w") as f:
                                json.dump(contours_list, f)

                            
                            import csv
                            csv_path = os.path.join(PREDICTIONS_DIR, f"mask_contours_frame{out_frame_idx}_obj{out_obj_id}.csv")
                            with open(csv_path, "w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(["x", "y", "z"])
                                z_str = CSV_FLOAT_FMT.format(Z_CONST)
                                for contour in contours_list:
                                    for x_, y_ in contour:
                                        xs = CSV_FLOAT_FMT.format(float(x_) * SCALE_CONTOUR)
                                        ys = CSV_FLOAT_FMT.format(float(y_) * SCALE_CONTOUR)
                                        writer.writerow([xs, ys, z_str])



                            # Update the stable live plan for downstream motion/robot modules.
                            _copy_csv(csv_path, PLAN_LIVE)

                        if out_frame_idx == 0 and points is not None and len(points):
                            show_points(np.array(points), np.array(labels), ax)

                        fig.canvas.draw()
                        w, h = fig.canvas.get_width_height()
                        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                        buf.shape = (h, w, 4)
                        img_array = buf[:, :, :3]

                        if any_mask_drawn:
                            # Store the display overlay and the unmodified frame for later review/metrics.
                            disp_png = os.path.join(PREDICTIONS_DIR, f"output_image_{out_frame_idx}.png")
                            Image.fromarray(img_array).save(disp_png)

                            bg_png = os.path.join(PREDICTIONS_DIR, f"bg_image_{out_frame_idx}.png")
                            Image.fromarray(image_array).save(bg_png)

                            # Generate focal points only after the first valid mask has been saved.
                            if out_frame_idx == 0:
                                _run_make_focal_points_once()

                            # Build qualitative GT overlays and quantitative IoU/Dice metrics.
                            msg = _save_overlay_and_metrics_for_frame(out_frame_idx, bg_png)

                            if msg != "overlay: ok":
                                print("[overlay]", msg)

                                

                        plt.close(fig)

                        # Timed motion gate allows the live plan to respond to measured target motion.
                        now = time.time()
                        if (now - last_gate) >= MOTION_POLL_SECS:
                            d = latest_motion["d"]
                            dx = latest_motion["dx"]; dy = latest_motion["dy"]
                            print(f"[motion] gate tick. d={d}, dx={dx}, dy={dy}")
                            try:
                                if d is None or (isinstance(d, float) and (np.isnan(d) or d < 0)):
                                    print("[motion] no motion provided → continue")
                                else:
                                    if d <= TIER1_MAX:
                                        if dx is not None and dy is not None:
                                            base_csv = PLAN_LIVE if os.path.isfile(PLAN_LIVE) else _find_latest_csv(PREDICTIONS_DIR)
                                            if base_csv:
                                                shifted_tmp = os.path.join(PREDICTIONS_DIR, "plan_points_shifted.csv")
                                                _shift_csv(base_csv, shifted_tmp, dx, dy)
                                                _copy_csv(shifted_tmp, PLAN_LIVE)
                                                print(f"[motion] Tier-1: shifted plan by ({dx:.2f},{dy:.2f}) mm")
                                            else:
                                                print("[motion] Tier-1: no CSV to shift → continue")
                                        else:
                                            print("[motion] Tier-1: no dx/dy → continue unchanged")
                                    elif d <= TIER2_MAX:
                                        new_csv = _remask_frame0_and_write_csv(threshold=BEST_THR)
                                        if new_csv: _copy_csv(new_csv, PLAN_LIVE)
                                        print("[motion] Tier-2: re-masked frame 0 and refreshed CSV")
                                    else:
                                        new_csv = _remask_frame0_and_write_csv(threshold=BEST_THR)
                                        if new_csv: _copy_csv(new_csv, PLAN_LIVE)
                                        print("[motion] Tier-3: full replan fallback = re-mask frame 0 and refresh CSV")
                                latest_motion.update({"d": None, "dx": None, "dy": None})
                            except Exception as e:
                                print("[motion] gate error:", e)
                            last_gate = now

                        yield img_array

                    except Exception as e:
                        print(f"Error in prediction: {str(e)}")
                        yield f"Error during prediction: {str(e)}"

                yield None
            except Exception as e:
                print(f"Error in prediction: {str(e)}")
                yield f"Error during prediction: {str(e)}"

        return Response(generate(), content_type='image/jpeg')

    except Exception as e:
        print(f"Error in start_prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error starting prediction: {str(e)}"
        }), 500

# Threaded Gradio prediction path used by the Start Prediction button.
def process_video(speed_slider):
    """Threaded version used by the Start Prediction button."""
    global predictor, inference_state, video_dir, frame_names, points, labels, stop_prediction, box
    stop_prediction = False
    try:
        if inference_state is None:
            inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=True)
            predictor.reset_state(inference_state)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
                box=box,
            )
        frame_queue = queue.Queue(maxsize=100)
        frame_delay = speed_slider
        print(f"Frame delay: {frame_delay} seconds")

        # Motion updates are checked periodically during threaded propagation.
        last_gate = time.time()

        # Background worker performs SAM2 propagation and file export while the UI remains responsive.
        def process_frames():
            nonlocal last_gate
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if stop_prediction:
                    print("Prediction stopped by user")
                    break
                try:
                    frame_path = os.path.join(video_dir, frame_names[out_frame_idx])
                    original_img = Image.open(frame_path).convert("RGB")
                    original_array = np.array(original_img)
                    H, W = original_array.shape[:2]

                    fig, ax = _full_bleed_fig(W, H)
                    ax.imshow(original_array)

                    # A frame is exported only after at least one object mask is available.
                    any_mask_drawn = False
                    for i, out_obj_id in enumerate(out_obj_ids):
                        # Convert logits to a thresholded ablation mask for the current frame/object.
                        raw = torch.sigmoid(out_mask_logits[i]).cpu().numpy()
                        mask = (raw > BEST_THR).astype(np.uint8).squeeze()

                        # Apply the same smoothing and component filtering as used for the initial prompt.
                        m8 = cv2.medianBlur((mask * 255).astype(np.uint8), 5)
                        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        if cnts:
                            c = max(cnts, key=cv2.contourArea)
                            eps = 0.005 * cv2.arcLength(c, True)
                            c2 = cv2.approxPolyDP(c, eps, True)
                            sm = np.zeros_like(m8)
                            cv2.drawContours(sm, [c2], -1, 255, thickness=-1)
                            mask = (sm > 127).astype(np.uint8)
                        else:
                            mask = (m8 > 127).astype(np.uint8)

                        mask = _postprocess_mask(mask)
                        show_mask(mask, ax, obj_id=out_obj_id)
                        any_mask_drawn = True

                        # Extract and save the mask boundary for planning/monitoring outputs.
                        contours = get_contours_from_mask(mask)
                        contours_list = [c.reshape(-1, 2).tolist() for c in contours]

                        json_path = os.path.join(PREDICTIONS_DIR, f"mask_contours_frame{out_frame_idx}_obj{out_obj_id}.json")
                        with open(json_path, "w") as f:
                            json.dump(contours_list, f)

                        import csv
                        csv_path = os.path.join(PREDICTIONS_DIR, f"mask_contours_frame{out_frame_idx}_obj{out_obj_id}.csv")
                        with open(csv_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["x", "y", "z"])
                            z_str = CSV_FLOAT_FMT.format(Z_CONST)
                            for contour in contours_list:
                                for x_, y_ in contour:
                                    xs = CSV_FLOAT_FMT.format(float(x_) * SCALE_CONTOUR)
                                    ys = CSV_FLOAT_FMT.format(float(y_) * SCALE_CONTOUR)
                                    writer.writerow([xs, ys, z_str])



                        # Refresh the stable live CSV after each frame-level contour export.
                        _copy_csv(csv_path, PLAN_LIVE)

                    fig.canvas.draw()
                    w, h = fig.canvas.get_width_height()
                    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    masked_array = buf.reshape(h, w, 4)[:, :, :3]

                    if any_mask_drawn:
                        # Save both the rendered overlay and original background frame.
                        disp_png = os.path.join(PREDICTIONS_DIR, f"output_image_{out_frame_idx}.png")
                        Image.fromarray(masked_array).save(disp_png)

                        bg_png = os.path.join(PREDICTIONS_DIR, f"bg_image_{out_frame_idx}.png")
                        Image.fromarray(original_array).save(bg_png)

                        # Snapshot the starting geometry so each prediction run has a saved baseline.
                        if out_frame_idx == 0:
                            _run_make_focal_points_once()

                        # Compare prediction against available manual masks and save metric JSON files.
                        msg = _save_overlay_and_metrics_for_frame(out_frame_idx, bg_png)

                        if msg != "overlay: ok":
                            print("[overlay]", msg)

                        # Snapshot the starting geometry so each prediction run has a saved baseline.
                        if out_frame_idx == 0:
                            _snapshot_current_geometry()

                    plt.close(fig)

                    # Periodically apply submitted motion updates during video propagation.
                    now = time.time()
                    if (now - last_gate) >= MOTION_POLL_SECS:
                        d = latest_motion["d"]
                        dx = latest_motion["dx"]; dy = latest_motion["dy"]
                        print(f"[motion] gate tick. d={d}, dx={dx}, dy={dy}")

                        try:
                            if d is None or (isinstance(d, float) and (np.isnan(d) or d < 0)):
                                print("[motion] no motion provided → continue")
                            else:
                                if d <= TIER1_MAX:
                                    if dx is not None and dy is not None:
                                        base_csv = PLAN_LIVE if os.path.isfile(PLAN_LIVE) else _find_latest_csv(PREDICTIONS_DIR)
                                        if base_csv:
                                            shifted_tmp = os.path.join(PREDICTIONS_DIR, "plan_points_shifted.csv")
                                            _shift_csv(base_csv, shifted_tmp, dx, dy)
                                            _copy_csv(shifted_tmp, PLAN_LIVE)
                                            print(f"[motion] Tier-1: shifted plan by ({dx:.2f},{dy:.2f}) mm")
                                        else:
                                            print("[motion] Tier-1: no CSV to shift → continue")
                                    else:
                                        print("[motion] Tier-1: no dx/dy → continue unchanged")

                                elif d <= TIER2_MAX:
                                    new_csv = _remask_frame0_and_write_csv(threshold=BEST_THR)
                                    if new_csv: _copy_csv(new_csv, PLAN_LIVE)
                                    print("[motion] Tier-2: re-masked frame 0 and refreshed CSV")

                                else:
                                    new_csv = _remask_frame0_and_write_csv(threshold=BEST_THR)
                                    if new_csv: _copy_csv(new_csv, PLAN_LIVE)
                                    print("[motion] Tier-3: full replan fallback = re-mask frame 0 and refresh CSV")

                            latest_motion.update({"d": None, "dx": None, "dy": None})
                        except Exception as e:
                            print("[motion] gate error:", e)

                        last_gate = now

                    # Send the original and segmented frame back to the Gradio display loop.
                    frame_queue.put((original_array, masked_array, out_frame_idx))
                except Exception as e:
                    print(f"Error processing frame {out_frame_idx}: {str(e)}")
                    traceback.print_exc()
            frame_queue.put(None)

        # Start propagation in a worker thread and stream frames to the UI at the selected speed.
        process_thread = Thread(target=process_frames)
        process_thread.start()

        last_frame_time = time.time()
        while True:
            frames = frame_queue.get()
            if frames is None or stop_prediction:
                break
            original, masked, frame_idx = frames
            current_time = time.time()
            time_since_last = current_time - last_frame_time
            if time_since_last < frame_delay:
                time.sleep(frame_delay - time_since_last)
            yield [original, masked]
            last_frame_time = time.time()

    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        traceback.print_exc()

# Stop flag used by the Gradio Stop Prediction button.
def stop_prediction_fn():
    global stop_prediction
    stop_prediction = True
    return "Stopping prediction..."

# Store a manual motion measurement and immediately update the live geometry.
def _push_motion(d, dx, dy):
    latest_motion["d"]  = None if d  is None else float(d)
    latest_motion["dx"] = None if dx is None else float(dx)
    latest_motion["dy"] = None if dy is None else float(dy)
    latest_motion["ts"] = time.time()
    # Apply the update immediately so the user does not have to wait for the polling interval.
    msg = _apply_motion_now(latest_motion["d"], latest_motion["dx"], latest_motion["dy"])
    print("[motion]", msg)
    return msg


# -----------------------------------------------------------------------------
# Gradio user interface
# -----------------------------------------------------------------------------
with gr.Blocks() as demo:
    demo.load()
    with gr.Row():
        with gr.Column():
            # SAM2 base checkpoint and config paths can be edited from the UI before initialization.
            sam2_checkpoint = gr.Textbox(
                label="SAM2 Checkpoint Path",
                value="C:/Users/a3taghip/samproject/sam2-main/checkpoints/sam2.1_hiera_large.pt"
            )
            model_cfg = gr.Textbox(
                label="Model Config Path",
                value="C:/Users/a3taghip/samproject/sam2-main/configs/sam2.1/sam2.1_hiera_l.yaml"
            )
            video_dir = gr.Textbox(
                label="Video Directory",
                value="C:/Users/a3taghip/samproject/sam2-main/data/Annotations/Images/image_seq_0"
            )
            # The app can run either the original SAM2 checkpoint or the ultrasound fine-tuned heads.
            use_finetuned_radio = gr.Radio(
                choices=["Base (no finetune)", "Fine-tuned heads"],
                value="Fine-tuned heads",
                label="Weights to use"
            )
            init_button = gr.Button("Initialize Model")

        with gr.Column():
            status = gr.Textbox(label="Status")

    with gr.Row():
        speed_slider = gr.Slider(
            minimum=0.1,
            maximum=5.0,
            value=2.5,
            step=0.1,
            label="Frame Interval (seconds)"
        )

    with gr.Row():
        image_input = gr.Image(
            label="Click to add points",
            interactive=True,
            type="numpy",
            height=480,
            show_download_button=False
        )

    with gr.Row():
        # Motion inputs allow manual testing of the tiered motion-aware update logic.
        motion_d   = gr.Number(label="Motion d (mm)", value=None, precision=2)
        motion_dx  = gr.Number(label="dx (mm, optional)", value=None, precision=2)
        motion_dy  = gr.Number(label="dy (mm, optional)", value=None, precision=2)
        push_motion_btn = gr.Button("Submit Motion")
        motion_status = gr.Textbox(label="Motion status", interactive=False)

    with gr.Row():
        start_button = gr.Button("Start Prediction")
        stop_button = gr.Button("Stop Prediction")

    with gr.Row():
        original_frame = gr.Image(label="Original Video", height=720)
        masked_frame = gr.Image(label="Segmented Video", height=720)

    with gr.Row():
        drawing_mode = gr.Radio(
            choices=["point", "box"],
            value="point",
            label="Drawing Mode",
            interactive=True
        )

    # Initialize the model and load the first frame for prompt selection.
    init_button.click(
        fn=initialize_model,
        inputs=[sam2_checkpoint, model_cfg, video_dir, use_finetuned_radio],
        outputs=[image_input, status]
    )

    # Clicking the image creates the initial SAM2 prompt and preview mask.
    image_input.select(
        fn=handle_click,
        inputs=[image_input, drawing_mode],
        outputs=[image_input]
    )

    # Submit motion values and update the current live plan.
    push_motion_btn.click(
        fn=_push_motion,
        inputs=[motion_d, motion_dx, motion_dy],
        outputs=[motion_status]
    )

    # Start frame propagation and display original/segmented frames side by side.
    start_button.click(
        fn=process_video,
        inputs=[speed_slider],
        outputs=[original_frame, masked_frame]
    )

    # Set the stop flag for the active prediction thread.
    stop_button.click(
        fn=stop_prediction_fn,
        inputs=[],
        outputs=[status]
    )

# Launch locally by default; share=False avoids exposing the interface publicly.
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
