"""Class-level Panoptic Quality (PQ) for the 7-macro cross-dataset evaluation.

Standalone companion to ``inference.ipynb``. It scores the SAME 7-macro
restricted-vocabulary setup as the notebook's ``evaluate_semantic_mIoU_simplified``
(identical windowed semantic inference, identical COCO/Cityscapes -> macro mapping,
identical ground truth) but with Panoptic Quality instead of mIoU, so the PQ numbers
are directly comparable to the mIoU table.

Because ``CityscapesSemantic`` yields only semantic maps (no instance IDs), each of the
7 macro classes is treated as a single "stuff" segment per image: the predicted region
for class ``c`` is matched to the GT region for ``c`` and counts as a TP iff their
IoU > 0.5 (otherwise the GT region is an FN and the predicted region an FP -- a double
penalty). ``mPQ`` is the mean over present macro classes.

    PQ = sum_{(p,g) in TP} IoU(p,g) / (|TP| + 0.5|FP| + 0.5|FN|) = SQ * RQ

Usage from the notebook (after the 7-class mIoU cell, which defines
``config_cs/config_coco``, ``state_dict_path_*``, ``data_loader_instance``,
``build_model_instance`` and the loaded ``ft_model``)::

    from eval_pq import evaluate_panoptic_quality_classlevel, evaluate_checkpoint_pq, summarize_pq

    mPQ_cs   = evaluate_checkpoint_pq(config_cs,   state_dict_path_cityscapes,
                                      data_loader_instance, device, build_model_instance, "cityscapes")
    mPQ_coco = evaluate_checkpoint_pq(config_coco, state_dict_path_coco,
                                      data_loader_instance, device, build_model_instance, "coco")
    mPQ_ft, *_ = evaluate_panoptic_quality_classlevel(
        ft_model, data_loader_instance.val_dataloader(), (640, 640),
        is_coco_model=False, device=device)

    summarize_pq({"COCO zero-shot": mPQ_coco,
                  "COCO->CS fine-tuned (head)": mPQ_ft,
                  "Cityscapes model": mPQ_cs})
"""

import numpy as np
import torch
from torch.nn import functional as F

# --------------------------------------------------------------------------- #
# Constants + 7-macro mapping (kept identical to the notebook's cell so PQ and
# mIoU score the exact same label space). If you change the mapping in the
# notebook, pass your own groups/lookup via the function kwargs to stay in sync.
# --------------------------------------------------------------------------- #
EVAL_NUM_CLASSES = 7
IGNORE_INDEX = 255
PQ_IOU_THRESH = 0.5
MACRO_NAMES = ["Flat", "Construction", "Object", "Nature", "Sky", "Human", "Vehicle"]

# Cityscapes TrainID -> 7 macro class
CS_TRAINID_TO_7CLASS = {
    0: 0, 1: 0,                              # Flat: road, sidewalk
    2: 1, 3: 1, 4: 1,                        # Construction: building, wall, fence
    5: 2, 6: 2, 7: 2,                        # Object: pole, traffic light, traffic sign
    8: 3, 9: 3,                              # Nature: vegetation, terrain
    10: 4,                                   # Sky
    11: 5, 12: 5,                            # Human: person, rider
    13: 6, 14: 6, 15: 6, 16: 6, 17: 6, 18: 6,  # Vehicle
}

# COCO panoptic ID -> 7 macro class (everything else -> IGNORE)
COCO_TO_7CLASS = {i: IGNORE_INDEX for i in range(133)}
COCO_TO_7CLASS.update({
    100: 0, 123: 0,                                                            # Flat
    82: 1, 91: 1, 101: 1, 106: 1, 109: 1, 110: 1, 111: 1, 112: 1, 117: 1, 129: 1, 131: 1,  # Construction
    9: 2, 10: 2, 11: 2, 12: 2,                                                 # Object
    88: 3, 90: 3, 99: 3, 102: 3, 103: 3, 105: 3, 113: 3, 116: 3, 124: 3, 125: 3, 126: 3, 130: 3,  # Nature
    119: 4,                                                                    # Sky
    0: 5,                                                                      # Human
    1: 6, 2: 6, 3: 6, 5: 6, 6: 6, 7: 6,                                        # Vehicle
})


def _build_groups(id_to_macro):
    groups = {m: [] for m in range(EVAL_NUM_CLASSES)}
    for cid, macro in id_to_macro.items():
        if macro != IGNORE_INDEX:
            groups[macro].append(cid)
    return groups


COCO_MACRO_GROUPS = _build_groups(COCO_TO_7CLASS)
CS_MACRO_GROUPS = _build_groups(CS_TRAINID_TO_7CLASS)

TARGET_LOOKUP_TABLE = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _cs_id, _macro in CS_TRAINID_TO_7CLASS.items():
    TARGET_LOOKUP_TABLE[_cs_id] = _macro


# --------------------------------------------------------------------------- #
# Core evaluator
# --------------------------------------------------------------------------- #
def evaluate_panoptic_quality_classlevel(
    model,
    dataloader,
    img_size,
    is_coco_model=False,
    device="cuda",
    coco_groups=COCO_MACRO_GROUPS,
    cs_groups=CS_MACRO_GROUPS,
    target_lookup=TARGET_LOOKUP_TABLE,
    verbose=True,
):
    """Class-level ("stuff") PQ over the 7 macro classes.

    Each macro class is one segment per image; a TP requires IoU > ``PQ_IOU_THRESH``.
    Returns ``(mPQ, pq, sq, rq)`` as floats / per-class numpy arrays.
    """
    model.eval()
    tp = np.zeros(EVAL_NUM_CLASSES, dtype=np.int64)
    fp = np.zeros(EVAL_NUM_CLASSES, dtype=np.int64)
    fn = np.zeros(EVAL_NUM_CLASSES, dtype=np.int64)
    iou_sum = np.zeros(EVAL_NUM_CLASSES, dtype=np.float64)

    raw_dev_str = device.type if isinstance(device, torch.device) else str(device)
    dev_type = "cuda" if "cuda" in raw_dev_str else "cpu"
    groups = coco_groups if is_coco_model else cs_groups

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if verbose and i % 100 == 0:
                print(f"Processing: Image [{i}/{len(dataloader)}] for PQ...")

            # --- batch unpacking identical to the mIoU evaluator ---
            if isinstance(batch, (list, tuple)):
                if len(batch) == 1 and isinstance(batch[0], dict):
                    batch = batch[0]
                elif len(batch) >= 2:
                    img, target = batch[0][0], batch[1][0]
                    batch = None
            if batch is not None and isinstance(batch, dict):
                img = batch.get("image", batch.get("img"))[0]
                target = batch.get("mask", batch.get("target"))[0]

            imgs = [img.to(device)]
            img_sizes = [im.shape[-2:] for im in imgs]

            with torch.amp.autocast(dtype=torch.float16, device_type=dev_type):
                crops, origins = model.window_imgs_semantic(imgs)
                mask_logits, class_logits = model(crops)
                mask_logits_up = F.interpolate(mask_logits[-1], img_size, mode="bilinear")
                crop_logits = model.to_per_pixel_logits_semantic(mask_logits_up, class_logits[-1])
                logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)[0]

            # --- identical 7-macro restricted-vocabulary mapping (argmax) ---
            mapped_logits = torch.full(
                (EVAL_NUM_CLASSES, logits.shape[1], logits.shape[2]),
                -float("inf"), device=logits.device, dtype=logits.dtype)
            for macro_id, ids in groups.items():
                if ids:
                    mapped_logits[macro_id] = logits[ids].max(dim=0)[0]
            pred_array = mapped_logits.argmax(0).cpu().numpy()

            target_array = model.to_per_pixel_targets_semantic(
                [target], IGNORE_INDEX)[0].cpu().numpy().copy()
            target_array[(target_array < 0) | (target_array > 255)] = IGNORE_INDEX
            target_array = target_lookup[target_array.astype(np.uint8)]

            # --- class-level PQ scoring (one segment per class per image) ---
            valid = target_array != IGNORE_INDEX
            for c in range(EVAL_NUM_CLASSES):
                gt_c = target_array == c            # GT already excludes ignore
                pred_c = (pred_array == c) & valid  # preds over ignore don't count
                gt_area = int(gt_c.sum())
                pred_area = int(pred_c.sum())
                if gt_area == 0 and pred_area == 0:
                    continue                        # class absent in both (TN)
                inter = int(np.logical_and(pred_c, gt_c).sum())
                union = pred_area + gt_area - inter
                iou = inter / union if union > 0 else 0.0
                if gt_area > 0 and pred_area > 0 and iou > PQ_IOU_THRESH:
                    tp[c] += 1
                    iou_sum[c] += iou
                else:
                    if gt_area > 0:
                        fn[c] += 1
                    if pred_area > 0:
                        fp[c] += 1

            del crops, mask_logits, class_logits, mask_logits_up, crop_logits, logits, mapped_logits

    denom = tp + 0.5 * fp + 0.5 * fn
    sq = np.divide(iou_sum, tp, out=np.zeros_like(iou_sum), where=tp > 0)      # segmentation quality
    rq = np.divide(tp, denom, out=np.zeros_like(denom), where=denom > 0)       # recognition quality
    pq = np.divide(iou_sum, denom, out=np.zeros_like(denom), where=denom > 0)  # PQ = SQ * RQ
    present = denom > 0

    if verbose:
        print("\n  Per-class class-level PQ:")
        print("  %-13s %7s %7s %7s %6s %6s %6s"
              % ("class", "PQ", "SQ", "RQ", "TP", "FP", "FN"))
        for c in range(EVAL_NUM_CLASSES):
            if present[c]:
                print("  %-13s %6.2f%% %6.2f%% %6.2f%% %5d %5d %5d"
                      % (MACRO_NAMES[c], pq[c] * 100, sq[c] * 100, rq[c] * 100,
                         tp[c], fp[c], fn[c]))
        mpq = float(pq[present].mean()) if present.any() else 0.0
        print("  %-13s %6.2f%%" % ("mPQ", mpq * 100))

    mPQ = float(pq[present].mean()) if present.any() else 0.0
    return mPQ, pq, sq, rq


def evaluate_checkpoint_pq(target_config, target_state_dict_path, data_loader_instance,
                           device, build_model_instance, model_type="cityscapes"):
    """Build + load a checkpoint (via the notebook's ``build_model_instance``) and score PQ.

    ``build_model_instance`` is passed in so this module stays standalone; use the one
    already defined in the notebook's 7-class eval cell.
    """
    is_coco = "coco" in target_state_dict_path.lower() or model_type in ("coco", "finetuned")
    print("\n" + "=" * 60)
    print(f"Panoptic Quality (class-level) - {model_type.upper()} model")
    print("=" * 60)

    model, img_size = build_model_instance(
        target_config, target_state_dict_path, data_loader_instance, device=device)
    checkpoint = torch.load(target_state_dict_path, map_location=device, weights_only=True)
    sd = (checkpoint["state_dict"]
          if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint)
    model.load_state_dict(sd, strict=False)

    mPQ, *_ = evaluate_panoptic_quality_classlevel(
        model=model, dataloader=data_loader_instance.val_dataloader(),
        img_size=img_size, is_coco_model=is_coco, device=device)
    return mPQ


def summarize_pq(results, title="Class-level Panoptic Quality (7-macro, IoU>0.5)"):
    """Pretty-print a {label: mPQ_fraction} dict."""
    print("\n=== " + title + " ===")
    width = max(len(k) for k in results)
    for label, mpq in results.items():
        print(f"{label:<{width}} : {mpq * 100:6.2f}% mPQ")


# --------------------------------------------------------------------------- #
# Shared-taxonomy evaluation (the "common classes" alternative to the 7-macro
# mapping). -inf masking restricts predictions to the genuinely-shared classes;
# all non-shared GT pixels are ignored. No hand-built many->1 grouping except
# the single defensible CS-side merge rider->person.
#
# Set: Things + sky, rider->person (9 classes).
# --------------------------------------------------------------------------- #
SHARED_NAMES = ["person", "car", "truck", "bus", "train",
                "motorcycle", "bicycle", "traffic light", "sky"]
SHARED_NUM_CLASSES = len(SHARED_NAMES)

# shared idx -> Cityscapes train-ids (max over these; person merges rider)
SHARED_CS_GROUPS = {
    0: [11, 12],  # person (incl. rider)
    1: [13],      # car
    2: [14],      # truck
    3: [15],      # bus
    4: [16],      # train
    5: [17],      # motorcycle
    6: [18],      # bicycle
    7: [6],       # traffic light
    8: [10],      # sky
}
# shared idx -> COCO contiguous panoptic ids (1:1)
SHARED_COCO_GROUPS = {
    0: [0],       # person
    1: [2],       # car
    2: [7],       # truck
    3: [5],       # bus
    4: [6],       # train
    5: [3],       # motorcycle
    6: [1],       # bicycle
    7: [9],       # traffic light
    8: [119],     # sky-other-merged
}
SHARED_TARGET_LOOKUP = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _idx, _ids in SHARED_CS_GROUPS.items():
    for _cid in _ids:
        SHARED_TARGET_LOOKUP[_cid] = _idx


def evaluate_shared_taxonomy(model, dataloader, img_size, is_coco_model=False,
                             device="cuda", verbose=True):
    """Restricted-vocabulary eval on the 9 shared classes (Things + sky, rider->person).

    -inf masking restricts predictions to the shared classes; non-shared GT pixels are
    ignored (so this measures only the taxonomy both datasets agree on). Computes BOTH
    shared-class mIoU and class-level mPQ in a single inference pass.

    Returns a dict with: ``miou``, ``iou`` (per-class), ``mpq``, ``pq``/``sq``/``rq``
    (per-class), ``names``, and the ``confusion`` matrix.
    """
    model.eval()
    N = SHARED_NUM_CLASSES
    groups = SHARED_COCO_GROUPS if is_coco_model else SHARED_CS_GROUPS

    hist = np.zeros((N, N), dtype=np.float64)            # for mIoU
    tp = np.zeros(N, dtype=np.int64)                     # for class-level PQ
    fp = np.zeros(N, dtype=np.int64)
    fn = np.zeros(N, dtype=np.int64)
    iou_sum = np.zeros(N, dtype=np.float64)

    raw_dev_str = device.type if isinstance(device, torch.device) else str(device)
    dev_type = "cuda" if "cuda" in raw_dev_str else "cpu"

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if verbose and i % 100 == 0:
                print(f"Processing: Image [{i}/{len(dataloader)}] for shared-taxonomy eval...")

            if isinstance(batch, (list, tuple)):
                if len(batch) == 1 and isinstance(batch[0], dict):
                    batch = batch[0]
                elif len(batch) >= 2:
                    img, target = batch[0][0], batch[1][0]
                    batch = None
            if batch is not None and isinstance(batch, dict):
                img = batch.get("image", batch.get("img"))[0]
                target = batch.get("mask", batch.get("target"))[0]

            imgs = [img.to(device)]
            img_sizes = [im.shape[-2:] for im in imgs]

            with torch.amp.autocast(dtype=torch.float16, device_type=dev_type):
                crops, origins = model.window_imgs_semantic(imgs)
                mask_logits, class_logits = model(crops)
                mask_logits_up = F.interpolate(mask_logits[-1], img_size, mode="bilinear")
                crop_logits = model.to_per_pixel_logits_semantic(mask_logits_up, class_logits[-1])
                logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)[0]

            # -inf masking: only the shared classes can be predicted (argmax over N).
            mapped_logits = torch.full(
                (N, logits.shape[1], logits.shape[2]),
                -float("inf"), device=logits.device, dtype=logits.dtype)
            for k, ids in groups.items():
                if ids:
                    mapped_logits[k] = logits[ids].max(dim=0)[0]
            pred_array = mapped_logits.argmax(0).cpu().numpy()

            target_array = model.to_per_pixel_targets_semantic(
                [target], IGNORE_INDEX)[0].cpu().numpy().copy()
            target_array[(target_array < 0) | (target_array > 255)] = IGNORE_INDEX
            target_array = SHARED_TARGET_LOOKUP[target_array.astype(np.uint8)]

            valid = target_array != IGNORE_INDEX  # ignore all non-shared GT pixels

            # mIoU confusion over shared-GT pixels only.
            t, p = target_array[valid].astype(np.int64), pred_array[valid].astype(np.int64)
            hist += np.bincount(N * t + p, minlength=N * N).reshape(N, N)

            # class-level PQ (one segment per class per image; preds masked to valid).
            for c in range(N):
                gt_c = target_array == c
                pred_c = (pred_array == c) & valid
                gt_area = int(gt_c.sum())
                pred_area = int(pred_c.sum())
                if gt_area == 0 and pred_area == 0:
                    continue
                inter = int(np.logical_and(pred_c, gt_c).sum())
                union = pred_area + gt_area - inter
                iou = inter / union if union > 0 else 0.0
                if gt_area > 0 and pred_area > 0 and iou > PQ_IOU_THRESH:
                    tp[c] += 1
                    iou_sum[c] += iou
                else:
                    if gt_area > 0:
                        fn[c] += 1
                    if pred_area > 0:
                        fp[c] += 1

            del crops, mask_logits, class_logits, mask_logits_up, crop_logits, logits, mapped_logits

    # mIoU aggregate
    inter = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    iou_present = union > 0
    miou = float(iou[iou_present].mean()) if iou_present.any() else 0.0

    # PQ aggregate
    denom = tp + 0.5 * fp + 0.5 * fn
    sq = np.divide(iou_sum, tp, out=np.zeros_like(iou_sum), where=tp > 0)
    rq = np.divide(tp, denom, out=np.zeros_like(denom), where=denom > 0)
    pq = np.divide(iou_sum, denom, out=np.zeros_like(denom), where=denom > 0)
    pq_present = denom > 0
    mpq = float(pq[pq_present].mean()) if pq_present.any() else 0.0

    if verbose:
        print("\n  Shared-taxonomy per-class (restricted vocab, non-shared GT ignored):")
        print("  %-13s %7s %7s %7s %7s %6s %6s %6s"
              % ("class", "IoU", "PQ", "SQ", "RQ", "TP", "FP", "FN"))
        for c in range(N):
            if iou_present[c] or pq_present[c]:
                print("  %-13s %6.2f%% %6.2f%% %6.2f%% %6.2f%% %5d %5d %5d"
                      % (SHARED_NAMES[c], iou[c] * 100, pq[c] * 100, sq[c] * 100,
                         rq[c] * 100, tp[c], fp[c], fn[c]))
        print("  %-13s %6.2f%% %6.2f%%" % ("MEAN", miou * 100, mpq * 100))

    return {"miou": miou, "iou": iou, "mpq": mpq, "pq": pq, "sq": sq, "rq": rq,
            "names": SHARED_NAMES, "confusion": hist}


def evaluate_checkpoint_shared(target_config, target_state_dict_path, data_loader_instance,
                               device, build_model_instance, model_type="cityscapes"):
    """Build + load a checkpoint and run the shared-taxonomy eval (mIoU + mPQ)."""
    is_coco = "coco" in target_state_dict_path.lower() or model_type in ("coco", "finetuned")
    print("\n" + "=" * 60)
    print(f"Shared-taxonomy eval - {model_type.upper()} model")
    print("=" * 60)

    model, img_size = build_model_instance(
        target_config, target_state_dict_path, data_loader_instance, device=device)
    checkpoint = torch.load(target_state_dict_path, map_location=device, weights_only=True)
    sd = (checkpoint["state_dict"]
          if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint)
    model.load_state_dict(sd, strict=False)

    return evaluate_shared_taxonomy(
        model=model, dataloader=data_loader_instance.val_dataloader(),
        img_size=img_size, is_coco_model=is_coco, device=device)
