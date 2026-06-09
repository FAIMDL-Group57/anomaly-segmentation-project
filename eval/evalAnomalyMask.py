"""Mask-based anomaly segmentation baselines for EoMT (project step 8).

The pixel baseline (step 7) lives in ``eval/evalAnomaly.py`` and scores an ERFNet
per-pixel classifier. This is its mask-architecture counterpart: it evaluates an
EoMT checkpoint on the same anomaly validation datasets and reports AuPRC /
FPR@95TPR for the post-hoc methods MSP, Max Logit, Max Entropy and RbA, plus
temperature-scaled MSP.

Key difference from the pixel baseline: EoMT is a mask architecture, so the model
emits per-query mask logits and per-query class logits. ``to_per_pixel_logits_semantic``
collapses those into per-pixel, per-class *presence* scores
``f_c(x) = sum_q sigmoid(mask_q(x)) * softmax(class_q)[c]`` (the trailing
"no-object" class is dropped). All four anomaly scores are derived from that same
single forward pass:

  * MSP         : 1 - max_c softmax(f/T)_c
  * Max Logit   : -max_c f_c
  * Max Entropy : Shannon entropy of softmax(f)
  * RbA         : -sum_c f_c   (low total known-class evidence => anomalous)

This lives in eval/ next to the pixel baseline (evalAnomaly.py). The EoMT model
code lives under eomt/, so we add that to sys.path and resolve checkpoint paths
against it -- the script therefore runs from any working directory, e.g.:

  cd eval
  python evalAnomalyMask.py \
      --models finetuned cityscapes coco \
      --methods msp maxlogit maxentropy rba \
      --temperatures 0.5 0.75 1.1
"""
import os
import sys
import glob
import random
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr

# This file is in <repo>/eval/, but the EoMT package (models.*, training.*) and
# its checkpoints live under <repo>/eomt/. Resolve everything against absolute
# paths so the script works regardless of the current working directory.
HERE = os.path.dirname(os.path.abspath(__file__))            # <repo>/eval
REPO_ROOT = os.path.dirname(HERE)                            # <repo>
EOMT_ROOT = os.path.join(REPO_ROOT, "eomt")                  # <repo>/eomt
sys.path.insert(0, EOMT_ROOT)

from models.vit import ViT
from models.eomt import EoMT
from training.mask_classification_semantic import MaskClassificationSemantic


def _under_eomt(path):
    """Resolve a checkpoint path stored relative to eomt/ (as in eval_finetune.py)."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(EOMT_ROOT, path))

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Images are fed at their native resolution and evaluated against the native GT.
# EoMT's documented inference (window_imgs_semantic / revert_window_logits_semantic)
# resizes the short side to the crop size, runs OVERLAPPING crops, stitches them and
# rescales back to the input size. Pre-squashing to a fixed 2:1 shape would collapse
# that overlap to zero (the seam artefact), so we deliberately keep native sizes.

# The three EoMT checkpoints differ in architecture, so each needs its own preset.
# Checkpoint paths are relative to eomt/ (same convention as eval_finetune.py) and
# resolved to absolute paths via _under_eomt() at load time.
MODEL_PRESETS = {
    "finetuned": dict(ckpt="checkpoints/ft_coco_full_long", num_q=200,
                      img_size=(640, 640), num_classes=19),
    "cityscapes": dict(ckpt="../../eomt_cityscapes.bin", num_q=100,
                       img_size=(1024, 1024), num_classes=19),
    "coco": dict(ckpt="../../eomt_coco.bin", num_q=200,
                 img_size=(640, 640), num_classes=133),
}

# Folder names of the five anomaly validation sets inside --datadir.
DATASET_NAMES = ["RoadAnomaly21", "RoadObsticle21", "FS_LostFound_full",
                 "fs_static", "RoadAnomaly"]

ALL_METHODS = ["msp", "maxlogit", "maxentropy", "rba"]


# ---------------------------------------------------------------------------
# Model construction / loading (same recipe as eval_finetune.py)
# ---------------------------------------------------------------------------
def build_model(num_q, img_size, num_classes, num_blocks=3):
    enc = ViT(img_size=img_size, backbone_name="vit_base_patch14_reg4_dinov2")
    net = EoMT(encoder=enc, num_classes=num_classes, num_q=num_q,
               num_blocks=num_blocks, masked_attn_enabled=False)
    m = MaskClassificationSemantic(network=net, img_size=img_size,
                                   num_classes=num_classes,
                                   attn_mask_annealing_enabled=False)
    return m.eval().to(device)


def find_ckpt(path):
    """Resolve a checkpoint path; a directory means 'best .ckpt inside it'.

    Tolerates both checkpoint layouts seen in this repo: <eomt>/<name> and
    <eomt>/checkpoints/<name>. A bare ``last.ckpt`` is used only as a fallback.
    """
    base = _under_eomt(path)
    # name without any leading "checkpoints/" component, e.g. "ft_coco_full_long"
    stem = path[len("checkpoints/"):] if path.startswith("checkpoints/") else path
    candidates = [base, _under_eomt(os.path.join("checkpoints", stem)),
                  _under_eomt(stem)]
    resolved = next((c for c in candidates if os.path.exists(c)), None)
    if resolved is None:
        raise FileNotFoundError(
            f"Checkpoint not found. Tried: {[c for c in candidates]}")
    if os.path.isdir(resolved):
        ckpts = sorted(glob.glob(os.path.join(resolved, "*.ckpt")))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt in {resolved}")
        nonlast = [c for c in ckpts if "last" not in os.path.basename(c)]
        return (nonlast or ckpts)[0]
    return resolved


def load_ckpt_into(model, ckpt_path):
    ckpt = torch.load(find_ckpt(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k.replace("._orig_mod", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


# ---------------------------------------------------------------------------
# Anomaly scoring -- all methods share one forward pass
# ---------------------------------------------------------------------------
def compute_anomaly_scores(logits, methods, temperatures):
    """logits: per-pixel class-presence scores (C, H, W) from
    to_per_pixel_logits_semantic. Returns {score_name: (H, W) numpy array}
    where higher = more anomalous.
    """
    logits = logits.float()
    out = {}
    if "maxlogit" in methods:
        out["maxlogit"] = (-logits.max(dim=0).values).cpu().numpy()
    if "rba" in methods:
        # Rejected by All: little total known-class evidence => anomalous.
        out["rba"] = (-logits.sum(dim=0)).cpu().numpy()
    if "maxentropy" in methods:
        probs = F.softmax(logits, dim=0)
        out["maxentropy"] = (
            -(probs * torch.log(probs + 1e-12)).sum(dim=0)).cpu().numpy()
    if "msp" in methods:
        out["msp"] = _msp(logits, 1.0)
        for t in temperatures:
            out[f"msp_t{t:g}"] = _msp(logits, t)
    return out


def _msp(logits, t):
    # Temperature is applied to the logits before softmax (confidence calibration).
    probs = F.softmax(logits / t, dim=0)
    return (1.0 - probs.max(dim=0).values).cpu().numpy()


# ---------------------------------------------------------------------------
# Windowed inference (identical pipeline to eval_finetune.py)
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer_logits(model, img_uint8_chw, crop_size):
    """Return per-pixel class-presence scores (C, H, W) at the input's native
    resolution, via EoMT's documented overlapping sliding-window inference."""
    imgs = [img_uint8_chw.to(device)]
    img_sizes = [im.shape[-2:] for im in imgs]
    dev_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.amp.autocast(dtype=torch.float16, device_type=dev_type,
                            enabled=(dev_type == "cuda")):
        crops, origins = model.window_imgs_semantic(imgs)
        mask_logits, class_logits = model(crops)
        mask_logits_up = F.interpolate(mask_logits[-1], crop_size, mode="bilinear")
        crop_logits = model.to_per_pixel_logits_semantic(mask_logits_up, class_logits[-1])
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)[0]
    return logits


# ---------------------------------------------------------------------------
# Ground-truth label mapping (consistent with eval/evalAnomaly.py)
# ---------------------------------------------------------------------------
def map_ood_labels(raw_gts, dataset_name):
    """Standardise a raw GT mask to {0: inlier, 1: anomaly, 255: ignore}.

    The provided validation bundle already ships standardised {0,1,255} masks for
    most sets (RoadAnomaly encodes the anomaly as 2); the raw-Fishyscapes branch
    is kept so the script also works on un-processed LostAndFound / fs_static.
    """
    if raw_gts.ndim == 3:
        raw_gts = raw_gts[:, :, 0]
    name = dataset_name.lower()
    ood = np.full_like(raw_gts, 255, dtype=np.uint8)

    already_standard = set(np.unique(raw_gts)).issubset({0, 1, 255})
    if already_standard or "roadanomaly" in name or "roadobsticle" in name:
        ood[raw_gts == 0] = 0
        ood[raw_gts == 1] = 1
        ood[raw_gts == 2] = 1            # RoadAnomaly encodes anomaly as label 2
    elif "lostfound" in name or "static" in name:
        ood[raw_gts == 1] = 0            # raw Fishyscapes: 1 = road/inlier
        ood[(raw_gts > 1) & (raw_gts < 205)] = 1
        ood[raw_gts == 0] = 255
    return ood


def store_heatmap(anomaly_map, img_path, model_name, dataset, score_name):
    # One subtree per model so the three model runs don't overwrite each other,
    # then per dataset (filenames can collide across datasets) and per score.
    out_dir = os.path.join("anomaly_vis_mask", model_name, dataset, score_name)
    os.makedirs(out_dir, exist_ok=True)
    norm = (anomaly_map - anomaly_map.min()) / (np.ptp(anomaly_map) + 1e-8)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, Image.open(img_path).size)  # PIL .size = (w, h)
    fname = os.path.basename(img_path).rsplit(".", 1)[0] + ".png"
    cv2.imwrite(os.path.join(out_dir, fname), heat)


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------
def gt_path_for(img_path):
    gt = img_path.replace(os.sep + "images" + os.sep,
                          os.sep + "labels_masks" + os.sep)
    return gt.rsplit(".", 1)[0] + ".png"


def evaluate_dataset(model, model_name, crop_size, dataset_dir, dataset_name,
                     methods, temperatures, store=False):
    """Return {score_name: (auprc, fpr95)} for one dataset."""
    img_paths = sorted(
        p for p in glob.glob(os.path.join(dataset_dir, "images", "*"))
        if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if not img_paths:
        print(f"  [skip] no images in {dataset_dir}/images")
        return {}

    score_lists, gt_list = {}, []
    for path in img_paths:
        # Native resolution: GT mask is matched to the image size (no downscaling),
        # so EoMT's sliding window keeps its overlap and we score at full res.
        img_pil = Image.open(path).convert("RGB")
        try:
            mask = Image.open(gt_path_for(path)).resize(img_pil.size, Image.NEAREST)
        except FileNotFoundError:
            continue
        ood_gts = map_ood_labels(np.array(mask), dataset_name)
        valid = (ood_gts == 0) | (ood_gts == 1)
        if not valid.any() or 1 not in np.unique(ood_gts):
            continue  # need anomaly pixels (same rule as the pixel baseline)

        img_chw = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).byte()
        logits = infer_logits(model, img_chw, crop_size)
        scores = compute_anomaly_scores(logits, methods, temperatures)

        gt_list.append(ood_gts[valid])
        for name, amap in scores.items():
            score_lists.setdefault(name, []).append(amap[valid])
            if store:
                store_heatmap(amap, path, model_name, dataset_name, name)

        del logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not gt_list:
        print(f"  [skip] {dataset_name}: no valid images with anomaly pixels")
        return {}

    val_label = np.concatenate(gt_list)
    results = {}
    for name, chunks in score_lists.items():
        val_out = np.concatenate(chunks)
        auprc = average_precision_score(val_label, val_out) * 100.0
        fpr = fpr_at_95_tpr(val_out, val_label) * 100.0
        results[name] = (auprc, fpr)
    return results


# ---------------------------------------------------------------------------
def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--datadir",
                    default=os.path.join(HERE, "Anomaly_Validation_Datasets",
                                         "Validation_Dataset"),
                    help="folder containing the five dataset subfolders")
    ap.add_argument("--models", nargs="+", default=list(MODEL_PRESETS),
                    choices=list(MODEL_PRESETS))
    ap.add_argument("--datasets", nargs="+", default=DATASET_NAMES,
                    choices=DATASET_NAMES)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS,
                    choices=ALL_METHODS)
    ap.add_argument("--temperatures", nargs="*", type=float,
                    default=[0.5, 0.75, 1.1],
                    help="extra temperatures for MSP (t=1.0 is always included)")
    ap.add_argument("--store", action="store_true",
                    help="dump per-image anomaly heatmaps to anomaly_vis_mask/")
    ap.add_argument("--results-file", default="anomaly_mask_results.txt")
    args = ap.parse_args()

    results_fh = open(args.results_file, "a")

    for model_name in args.models:
        preset = MODEL_PRESETS[model_name]
        print(f"\n{'='*78}\n Model: {model_name}  ({preset['ckpt']})\n{'='*78}",
              flush=True)
        model = build_model(preset["num_q"], preset["img_size"], preset["num_classes"])
        load_ckpt_into(model, preset["ckpt"])
        crop_size = preset["img_size"]

        for dataset_name in args.datasets:
            dataset_dir = os.path.join(args.datadir, dataset_name)
            if not os.path.isdir(dataset_dir):
                print(f"  [skip] missing {dataset_dir}")
                continue
            print(f"\n-- {dataset_name} --", flush=True)
            res = evaluate_dataset(model, model_name, crop_size, dataset_dir,
                                   dataset_name, args.methods, args.temperatures,
                                   args.store)
            for score_name, (auprc, fpr) in res.items():
                line = (f"{model_name:<11} {dataset_name:<18} {score_name:<10} "
                        f"AuPRC {auprc:6.2f}  FPR95 {fpr:6.2f}")
                print("  " + line, flush=True)
                results_fh.write(line + "\n")
            results_fh.flush()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_fh.close()
    print(f"\nDone. Appended results to {args.results_file}")


if __name__ == "__main__":
    main()
