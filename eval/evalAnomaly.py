# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Folder names of the anomaly validation sets used by --full-report. Same set as
# the EoMT mask baseline (eval/evalAnomalyMask.py) so the two reports line up.
DATASET_NAMES = ["RoadAnomaly21", "RoadObsticle21", "FS_LostFound_full",
                 "fs_static", "RoadAnomaly"]

# Post-hoc methods the pixel baseline supports (no RbA -- that is mask-specific).
ALL_METHODS = ["msp", "maxlogit", "maxentropy"]

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)

def compute_anomaly_score(result, method):
    # result: raw ERFNet logits, shape [1, C, H, W]. Returns a [H, W] anomaly
    # map where higher = more anomalous (out-of-distribution).
    logits = result.squeeze(0)  # [C, H, W]
    if method == "maxlogit":
        # negative max logit: confident (high logit) -> low anomaly score
        anomaly = -torch.max(logits, dim=0).values
    elif method == "msp":
        # 1 - maximum softmax probability
        probs = torch.softmax(logits, dim=0)
        anomaly = 1.0 - torch.max(probs, dim=0).values
    elif method == "maxentropy":
        # Shannon entropy of the softmax distribution
        probs = torch.softmax(logits, dim=0)
        anomaly = -torch.sum(probs * torch.log(probs + 1e-12), dim=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    return anomaly.data.cpu().numpy()

def _store_anomaly_result(anomaly_result, path, method):
    # --- save anomaly heatmap for visual inspection ---
    # one sub-folder per validation dataset, then per method
    # (msp / maxlogit / maxentropy), so heatmaps are grouped by the set they
    # came from and the scoring method used.
    dataset = osp.basename(osp.dirname(osp.dirname(path)))  # .../<dataset>/images/<file>
    out_dir = osp.join("anomaly_vis", dataset, method)
    os.makedirs(out_dir, exist_ok=True)
    norm = (anomaly_result - anomaly_result.min()) / (np.ptp(anomaly_result) + 1e-8)
    heatmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # resize heatmap back to the original image resolution so they match
    orig = Image.open(path)
    heatmap = cv2.resize(heatmap, orig.size)  # PIL .size is (width, height)
    fname = osp.basename(path).rsplit(".", 1)[0] + ".png"
    cv2.imwrite(osp.join(out_dir, fname), heatmap)


# ---------------------------------------------------------------------------
# Full-report helpers: evaluate every dataset with every method in one pass.
# ---------------------------------------------------------------------------
def _msp(logits, t):
    # 1 - max softmax probability, with temperature scaling applied to the logits.
    probs = torch.softmax(logits / t, dim=0)
    return (1.0 - torch.max(probs, dim=0).values).data.cpu().numpy()

def evaluate_anomaly_scores_gpu(logits_tensor, t):
    probs = torch.softmax(logits_tensor / t, dim=0)
    scores = (1.0 - torch.max(probs, dim=0)[0])
    return scores.cpu().numpy()

def compute_all_anomaly_scores(result, methods, temperatures):
    # One forward pass -> every requested anomaly map. Mirrors compute_anomaly_score
    # but returns {score_name: [H, W] array} (higher = more anomalous) and adds the
    # temperature-scaled MSP variants used in the report.
    logits = result.squeeze(0)  # [C, H, W]
    out = {}
    if "maxlogit" in methods:
        out["maxlogit"] = (-torch.max(logits, dim=0).values).data.cpu().numpy()
    if "maxentropy" in methods:
        probs = torch.softmax(logits, dim=0)
        out["maxentropy"] = (
            -torch.sum(probs * torch.log(probs + 1e-12), dim=0)).data.cpu().numpy()
    if "msp" in methods:
        out["msp"] = _msp(logits, 1.0)
        for t in temperatures:
            out[f"msp_t{t:g}"] = _msp(logits, t)
    return out

def load_ood_gts(path):
    # Resolve the ground-truth mask for an image and standardise it to
    # {0: inlier, 1: anomaly, 255: ignore}, matching the per-dataset remapping the
    # single-method path uses below.
    pathGT = path.replace("images", "labels_masks")
    if "RoadObsticle21" in pathGT:
        pathGT = pathGT.replace("webp", "png")
    if "fs_static" in pathGT:
        pathGT = pathGT.replace("jpg", "png")
    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace("jpg", "png")

    mask = target_transform(Image.open(pathGT))
    ood_gts = np.array(mask)

    if "RoadAnomaly" in pathGT:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)
    if "LostFound" in pathGT:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    if "Streethazard" in pathGT:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)
    return ood_gts

def evaluate_dataset_full(model, dataset_dir, args, methods, temperatures):
    """Return {score_name: (auprc, fpr95)} for one anomaly dataset."""
    img_paths = sorted(
        p for p in glob.glob(osp.join(dataset_dir, "images", "*"))
        if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    score_lists, gt_list = {}, []
    fixed_t_scores_pool = []
    all_labels_list = []
  
    for path in img_paths:
        images = input_transform(Image.open(path).convert("RGB")).unsqueeze(0).float()
        if not args.cpu:
            images = images.cuda()
        with torch.no_grad():
            result = model(images)
        scores = compute_all_anomaly_scores(result, methods, temperatures)
        try:
            ood_gts = load_ood_gts(path)
        except FileNotFoundError:
            continue
        if 1 not in np.unique(ood_gts):
            continue  # need anomaly pixels (same rule as the single-method path)
        valid = (ood_gts == 0) | (ood_gts == 1)
        gt_list.append(ood_gts[valid])
        for name, amap in scores.items():
            score_lists.setdefault(name, []).append(amap[valid])
        if "msp" in methods:
            logits_raw = result.squeeze(0).float()[:19, :] 
            valid_mask_torch = torch.from_numpy(valid).to(logits_raw.device)
            fixed_t_scores_pool.append(logits_raw[:, valid_mask_torch].cpu().half())
            all_labels_list.append(ood_gts[valid])
        del result
        if not args.cpu:
            torch.cuda.empty_cache()

    if not gt_list:
        return {}

    val_label = np.concatenate(gt_list)
    results = {}
    for name, chunks in score_lists.items():
        val_out = np.concatenate(chunks)
        auprc = average_precision_score(val_label, val_out) * 100.0
        fpr = fpr_at_95_tpr(val_out, val_label) * 100.0
        results[name] = (auprc, fpr)

    if len(all_labels_list) > 0 and "msp" in methods:
        global_labels_np = np.concatenate(all_labels_list, axis=0)
        device_type = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        global_logits_torch = torch.cat(fixed_t_scores_pool, dim=1).to(device_type, dtype=torch.float32)

        best_t, best_auprc, best_fpr95 = 1.0, 0.0, 100.0
        search_space = np.arange(0.1, 2.55, 0.05)
        for t in search_space:
            t = round(t, 2)
            global_scores_np = evaluate_anomaly_scores_gpu(global_logits_torch, t)
            auprc = average_precision_score(global_labels_np, global_scores_np) * 100.0
            if auprc > best_auprc:
                best_auprc = auprc
                best_t = t
                best_fpr95 = fpr_at_95_tpr(global_scores_np, global_labels_np) * 100.0
        
        results[f"msp_best(t={best_t:.2f})"] = (best_auprc, best_fpr95)
        print(f"  [FOUND] Best Temperature: t={best_t:.2f} | Best AuPRC: {best_auprc:.2f}%")

    return results

def run_full_report(model, args):
    report_fh = open(args.report_file, "a") if args.report_file else None
    for dataset_name in DATASET_NAMES:
        dataset_dir = osp.join(args.anomaly_datadir, dataset_name)
        if not osp.isdir(dataset_dir):
            print(f"  [skip] missing {dataset_dir}")
            continue
        header = f"-- {dataset_name} --"
        print(f"\n{header}", flush=True)
        if report_fh:
            report_fh.write(f"\n{header}\n")
        res = evaluate_dataset_full(model, dataset_dir, args,
                                    ALL_METHODS, args.temperatures)
        for score_name, (auprc, fpr) in res.items():
            line = (f"  {dataset_name:<18} {score_name:<10} "
                    f"AuPRC {auprc:6.2f}  FPR95 {fpr:6.2f}")
            print(line, flush=True)
            if report_fh:
                report_fh.write(line + "\n")
        if report_fh:
            report_fh.flush()
    if report_fh:
        report_fh.close()
        print(f"\nDone. Appended full report to {args.report_file}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default='msp',
                        choices=['msp', 'maxlogit', 'maxentropy'],
                        help='post-hoc anomaly score: MSP, Max Logit or Max Entropy')
    parser.add_argument('--full-report', action='store_true',
                        help='evaluate ALL datasets with ALL methods (incl. '
                             'temperature-scaled MSP) and print a grouped report')
    parser.add_argument('--anomaly-datadir',
                        default=osp.join(osp.dirname(osp.abspath(__file__)),
                                         'Anomaly_Validation_Datasets',
                                         'Validation_Dataset'),
                        help='folder containing the anomaly dataset subfolders '
                             '(used by --full-report)')
    parser.add_argument('--temperatures', nargs='*', type=float,
                        default=[0.5, 0.75, 1.1],
                        help='extra MSP temperatures for --full-report '
                             '(t=1.0 is always included)')
    parser.add_argument('--report-file', default='all_results_table.txt',
                        help='append the --full-report output to this file')
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()

    if args.full_report:
        run_full_report(model, args)
        return

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float()
        if not args.cpu:
            images = images.cuda()
        with torch.no_grad():
            result = model(images)
        anomaly_result = compute_anomaly_score(result, args.method)
        _store_anomaly_result(anomaly_result, path, args.method)
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'Method: {args.method}')
    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('method:' + args.method + '   input:' + str(args.input[0]) +
                '   AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()
