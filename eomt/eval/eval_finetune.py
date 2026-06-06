"""Standalone evaluation mirroring the inference.ipynb eval cells.

Computes, for the fine-tuned (COCO->Cityscapes head-only) model and the original
Cityscapes model, on the full Cityscapes val set:
  (a) 7-macro restricted-vocabulary mIoU  (consistent with the previous step)
  (b) full 19-class Cityscapes mIoU (per class + mean)
"""
import glob
import os
import sys
import importlib

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# This script lives in eval/ but its imports/paths assume the eomt/ root.
# Add the repo root (parent of this file's dir) so models.*/training.* resolve,
# then run it from eomt/ so the relative config/checkpoint paths below work:
#   cd eomt && python eval/eval_finetune.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit import ViT
from models.eomt import EoMT
from training.mask_classification_semantic import MaskClassificationSemantic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = 255
EVAL_NUM_CLASSES = 7

DATA_PATH = "../../cityscapes"
CS_BIN = "../../eomt_cityscapes.bin"
FT_DIR = "checkpoints/ft_coco_headonly"
UNFREEZE_DIR = "checkpoints/ft_coco_unfreeze"
FULL_DIR = "checkpoints/ft_coco_full_long"

# ---- 7-macro mappings (from inference.ipynb) ----
CS_TRAINID_TO_7CLASS = {
    0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3,
    10: 4, 11: 5, 12: 5, 13: 6, 14: 6, 15: 6, 16: 6, 17: 6, 18: 6,
}
CS_MACRO_GROUPS = {m: [] for m in range(EVAL_NUM_CLASSES)}
for cs_id, m in CS_TRAINID_TO_7CLASS.items():
    CS_MACRO_GROUPS[m].append(cs_id)
TARGET_LOOKUP_TABLE = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for cs_id, m in CS_TRAINID_TO_7CLASS.items():
    TARGET_LOOKUP_TABLE[cs_id] = m

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def compute_confusion_matrix(pred, target, num_classes, ignore_index):
    mask = (target >= 0) & (target < num_classes) & (target != ignore_index)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def build_data():
    with open("configs/dinov2/cityscapes/semantic/eomt_base_640.yaml") as f:
        cfg = yaml.safe_load(f)
    mod_name, cls_name = cfg["data"]["class_path"].rsplit(".", 1)
    data_cls = getattr(importlib.import_module(mod_name), cls_name)
    data = data_cls(path=DATA_PATH, batch_size=1, num_workers=0,
                    check_empty_targets=False, **cfg["data"].get("init_args", {}))
    return data.setup()


def build_model(num_q, img_size, num_classes, num_blocks=3):
    enc = ViT(img_size=img_size, backbone_name="vit_base_patch14_reg4_dinov2")
    net = EoMT(encoder=enc, num_classes=num_classes, num_q=num_q,
               num_blocks=num_blocks, masked_attn_enabled=False)
    m = MaskClassificationSemantic(network=net, img_size=img_size,
                                   num_classes=num_classes,
                                   attn_mask_annealing_enabled=False)
    return m.eval().to(device)


def load_ckpt_into(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k.replace("._orig_mod", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


@torch.no_grad()
def evaluate(model, dataloader, img_size, tag):
    """Single pass: accumulate both 7-macro and 19-class confusion matrices."""
    hist7 = np.zeros((EVAL_NUM_CLASSES, EVAL_NUM_CLASSES))
    hist19 = np.zeros((19, 19))
    dev_type = "cuda" if torch.cuda.is_available() else "cpu"

    for i, batch in enumerate(dataloader):
        if i % 100 == 0:
            print(f"  [{tag}] image {i}/{len(dataloader)}", flush=True)
        img, target = batch[0][0], batch[1][0]
        imgs = [img.to(device)]
        img_sizes = [im.shape[-2:] for im in imgs]

        with torch.amp.autocast(dtype=torch.float16, device_type=dev_type):
            crops, origins = model.window_imgs_semantic(imgs)
            mask_logits, class_logits = model(crops)
            mask_logits_up = F.interpolate(mask_logits[-1], img_size, mode="bilinear")
            crop_logits = model.to_per_pixel_logits_semantic(mask_logits_up, class_logits[-1])
            logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)[0]

        # full 19-class prediction
        pred19 = logits.argmax(0).cpu().numpy()

        # 7-macro prediction (CS path: max over the train-ids in each macro group)
        mapped = torch.full((EVAL_NUM_CLASSES, logits.shape[1], logits.shape[2]),
                            -float("inf"), device=logits.device, dtype=logits.dtype)
        for macro_id, cs_ids in CS_MACRO_GROUPS.items():
            if cs_ids:
                mapped[macro_id] = logits[cs_ids].max(dim=0)[0]
        pred7 = mapped.argmax(0).cpu().numpy()

        tgt19 = model.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].cpu().numpy().copy()
        tgt19[(tgt19 < 0) | (tgt19 > 255)] = IGNORE_INDEX
        tgt7 = TARGET_LOOKUP_TABLE[tgt19.astype(np.uint8)]

        hist7 += compute_confusion_matrix(pred7, tgt7, EVAL_NUM_CLASSES, IGNORE_INDEX)
        hist19 += compute_confusion_matrix(pred19, tgt19, 19, IGNORE_INDEX)

    def miou(hist):
        inter = np.diag(hist)
        union = hist.sum(1) + hist.sum(0) - inter
        return inter / (union + 1e-10)

    return miou(hist7), miou(hist19)


def best_ckpt(d):
    """Best (monitored) checkpoint = the non-'last' .ckpt saved by save_top_k=1."""
    nonlast = [c for c in sorted(glob.glob(os.path.join(d, "*.ckpt")))
               if "last" not in os.path.basename(c)]
    return nonlast[0] if nonlast else sorted(glob.glob(os.path.join(d, "*.ckpt")))[0]


def main():
    data = build_data()
    val = data.val_dataloader()

    head_ckpt = best_ckpt(FT_DIR)
    unf_ckpt = best_ckpt(UNFREEZE_DIR)
    full_ckpt = best_ckpt(FULL_DIR)
    print("head-only checkpoint:", head_ckpt, flush=True)
    print("unfreeze  checkpoint:", unf_ckpt, flush=True)
    print("full      checkpoint:", full_ckpt, flush=True)

    # The three fine-tuned models share architecture (num_q=200, 640px, 19-class);
    # the original Cityscapes model is num_q=100, 1024px.
    print("\n=== FT head-only (COCO->Cityscapes) | num_q=200, 640px ===", flush=True)
    m = build_model(num_q=200, img_size=(640, 640), num_classes=19)
    load_ckpt_into(m, head_ckpt)
    h7, h19 = evaluate(m, val, (640, 640), "head")
    del m
    torch.cuda.empty_cache()

    print("\n=== FT unfreeze (COCO->Cityscapes, blocks 8-11) | num_q=200, 640px ===", flush=True)
    m = build_model(num_q=200, img_size=(640, 640), num_classes=19)
    load_ckpt_into(m, unf_ckpt)
    u7, u19 = evaluate(m, val, (640, 640), "unfreeze")
    del m
    torch.cuda.empty_cache()

    print("\n=== FT full backbone (COCO->Cityscapes, 60ep) | num_q=200, 640px ===", flush=True)
    m = build_model(num_q=200, img_size=(640, 640), num_classes=19)
    load_ckpt_into(m, full_ckpt)
    f7, f19 = evaluate(m, val, (640, 640), "full")
    del m
    torch.cuda.empty_cache()

    print("\n=== ORIGINAL Cityscapes | num_q=100, 1024px ===", flush=True)
    cs = build_model(num_q=100, img_size=(1024, 1024), num_classes=19)
    load_ckpt_into(cs, CS_BIN)
    cs7, cs19 = evaluate(cs, val, (1024, 1024), "cs")

    print("\n\n############## RESULTS ##############")
    print("\n(a) 7-macro shared-vocabulary mIoU (consistent with previous step)")
    print(f"  COCO zero-shot (from previous step):      77.37%")
    print(f"  COCO->Cityscapes FT (head-only):          {h7.mean()*100:.2f}%")
    print(f"  COCO->Cityscapes FT (unfreeze 8-11):      {u7.mean()*100:.2f}%")
    print(f"  COCO->Cityscapes FT (full backbone, 60ep):{f7.mean()*100:.2f}%")
    print(f"  Original Cityscapes model:                {cs7.mean()*100:.2f}%")

    print("\n(b) Full 19-class Cityscapes mIoU")
    print("  %-16s %9s %9s %9s %9s" % ("class", "original", "ft(head)", "ft(unfrz)", "ft(full)"))
    print("  " + "-" * 58)
    for i, name in enumerate(CITYSCAPES_CLASS_NAMES):
        print("  %-16s %8.2f%% %8.2f%% %8.2f%% %8.2f%%" % (
            name, cs19[i]*100, h19[i]*100, u19[i]*100, f19[i]*100))
    print("  " + "-" * 58)
    print("  %-16s %8.2f%% %8.2f%% %8.2f%% %8.2f%%" % (
        "mIoU (19)", cs19.mean()*100, h19.mean()*100, u19.mean()*100, f19.mean()*100))


if __name__ == "__main__":
    main()
