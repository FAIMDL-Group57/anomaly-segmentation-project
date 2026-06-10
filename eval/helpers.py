import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr


def resolve_gt_path(img_path):
    """Image path -> ground-truth mask path (``.../labels_masks/<name>.png``).

    Masks in every validation set are stored as PNGs under ``labels_masks/`` even
    when the images are .jpg/.webp, so we swap the folder and force the .png
    extension. Used by both the pixel (evalAnomaly.py) and mask
    (evalAnomalyMask.py) baselines.
    """
    gt = img_path.replace(os.sep + "images" + os.sep,
                          os.sep + "labels_masks" + os.sep)
    return gt.rsplit(".", 1)[0] + ".png"


def standardize_ood_gts(raw_gts, name):
    """Standardise a raw GT mask to {0: inlier, 1: anomaly, 255: ignore}.

    ``name`` may be a dataset folder name or a full mask path -- only lowercased
    substring membership is used, so both ``"FS_LostFound_full"`` and a path that
    contains it match the same branch.

    The provided validation bundle already ships standardised {0,1,255} masks for
    most sets (RoadAnomaly encodes the anomaly as label 2), which the
    ``already_standard`` fast-path covers. The raw-Fishyscapes and StreetHazard
    branches are kept so the same code also works on un-processed masks.
    """
    raw = np.asarray(raw_gts)
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    n = name.lower()
    ood = np.full_like(raw, 255, dtype=np.uint8)

    already_standard = set(np.unique(raw)).issubset({0, 1, 255})
    if already_standard or "roadanomaly" in n or "roadobsticle" in n:
        ood[raw == 0] = 0
        ood[raw == 1] = 1
        ood[raw == 2] = 1                     # RoadAnomaly encodes anomaly as label 2
    elif "lostfound" in n or "lostandfound" in n or "static" in n:
        ood[raw == 1] = 0                     # raw Fishyscapes: 1 = road/inlier
        ood[(raw > 1) & (raw < 205)] = 1
        ood[raw == 0] = 255
    elif "streethazard" in n:
        s = raw.copy()
        s = np.where(s == 14, 255, s)
        s = np.where(s < 20, 0, s)
        s = np.where(s == 255, 1, s)
        ood = s.astype(np.uint8)
    return ood


def summarize_scores(gt_list, score_lists):
    """Per-image GT/score chunks -> {score_name: (auprc, fpr95)} (both in %).

    ``gt_list`` is a list of 1-D arrays of standardised labels; ``score_lists``
    maps each score name to a list of matching 1-D anomaly-score arrays.
    """
    val_label = np.concatenate(gt_list)
    results = {}
    for name, chunks in score_lists.items():
        val_out = np.concatenate(chunks)
        auprc = average_precision_score(val_label, val_out) * 100.0
        fpr = fpr_at_95_tpr(val_out, val_label) * 100.0
        results[name] = (auprc, fpr)
    return results


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
