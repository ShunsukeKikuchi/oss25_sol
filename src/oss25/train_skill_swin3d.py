import os
import sys
import argparse
import json
import copy
from typing import Dict, List, Tuple
from timm.utils.model_ema import ModelEmaV3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.swin3d_aux import Swin3DWithSegAux
from .datasets.skill_video_dataset import SkillVideoDataset


def load_splits(split_root: str, fold: int) -> Tuple[List[str], List[str]]:
    tv = os.path.join(split_root, f'fold_{fold}_train_videos.txt')
    vv = os.path.join(split_root, f'fold_{fold}_val_videos.txt')
    with open(tv) as f:
        train_videos = [x.strip() for x in f if x.strip()]
    with open(vv) as f:
        val_videos = [x.strip() for x in f if x.strip()]
    return train_videos, val_videos


def aggregate_labels(xlsx_path: str, osats_full_xlsx: str | None = None) -> Dict[str, Dict]:
    df = pd.read_excel(xlsx_path)
    # Normalize column names of OSATS and GRS
    osats_cols = [
        'OSATS_RESPECT', 'OSATS_MOTION', 'OSATS_INSTRUMENT', 'OSATS_SUTURE',
        'OSATS_FLOW', 'OSATS_KNOWLEDGE', 'OSATS_PERFORMANCE', 'OSATS_FINAL_QUALITY'
    ]
    grs_col = 'GLOBA_RATING_SCORE'
    assert grs_col in df.columns
    gb = df.groupby('VIDEO')
    labels: Dict[str, Dict] = {}
    labels: Dict[str, Dict] = {}
    for vid, g in gb:
        osats = g[osats_cols].mean().values.astype(np.float32)
        grs = float(g[grs_col].mean())
        # Map GRS 8..40 -> 4 classes by bins [8-15]=0, [16-23]=1, [24-31]=2, [32-40]=3
        if grs <= 15:
            grs_cls = 0
        elif grs <= 23:
            grs_cls = 1
        elif grs <= 31:
            grs_cls = 2
        else:
            grs_cls = 3
        labels[vid] = {
            'osats': osats,  # (8,)
            'grs_cls': grs_cls,
            'grs_num': grs,
        }
    # Optional: enrich with TIME/GROUP/SUTURES from OSATS.xlsx
    try:
        if osats_full_xlsx is None:
            osats_full_xlsx = os.path.join(os.path.dirname(xlsx_path), 'OSATS.xlsx')
        df2 = pd.read_excel(osats_full_xlsx)
        # Normalize columns
        # TIME: PRE/POST -> 0/1
        # GROUP: map to fixed order
        group_map_order = ['E-LEARNING', 'HMD-BASED', 'TUTOR-LED']
        gmap = {name: i for i, name in enumerate(group_map_order)}
        agg = df2.groupby('VIDEO').agg({
            'TIME': 'first',
            'GROUP': 'first',
            'SUTURES': 'mean',
        }).reset_index()
        for _, row in agg.iterrows():
            vid = row['VIDEO']
            if vid not in labels:
                continue
            time_bin = 0 if str(row['TIME']).strip().upper().startswith('PRE') else 1
            group_idx = gmap.get(str(row['GROUP']).strip().upper().replace('’', "'"), None)
            if group_idx is None:
                # try exact match first
                group_idx = gmap.get(str(row['GROUP']).strip(), 0)
            sut = float(row['SUTURES']) if pd.notna(row['SUTURES']) else 0.0
            labels[vid]['time_bin'] = int(time_bin)
            labels[vid]['group_idx'] = int(group_idx)
            labels[vid]['sutures'] = float(sut)
    except Exception as e:
        print(f"[warn] enrich labels failed: {e}")
    return labels


def build_dataloaders(args, labels):
    split_root = args.split_root
    train_videos, val_videos = load_splits(split_root, args.fold)

    tenfps_root = args.tenfps_root
    pseudo6_root = args.pseudo6_root

    ds_train = SkillVideoDataset(
        train_videos, labels, tenfps_root, pseudo6_root,
        task=args.task, model_mode=args.model, n_frames=args.n_frames,
        last5s_sec=args.last5s_sec, image_size=args.image_size,
        deterministic_val=False, frames1fps_root=getattr(args, 'frames1fps_root', None),
    )
    ds_val = SkillVideoDataset(
        val_videos, labels, tenfps_root, pseudo6_root,
        task=args.task, model_mode=args.model, n_frames=args.n_frames,
        last5s_sec=args.last5s_sec, image_size=args.image_size,
        deterministic_val=True, frames1fps_root=getattr(args, 'frames1fps_root', None),
    )
    # Build optional balanced sampler based on class counts (TaskA only)
    sampler = None
    if args.task == 'A' and args.balanced_sampler:
        counts = [0, 0, 0, 0]
        y_list = []
        for vid in ds_train.video_ids:
            y = int(labels[vid]['grs_cls'])
            y_list.append(y)
            counts[y] += 1
        weights = []
        for y in y_list:
            if args.ce_class_weighting == 'inv':
                w = 1.0 / max(1, counts[y])
            elif args.ce_class_weighting == 'inv_sqrt':
                w = 1.0 / max(1.0, float(counts[y]) ** 0.5)
            else:
                # uniform sampling but keep option to enable sampler API
                w = 1.0
            weights.append(w)
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return loader_train, loader_val


def compute_metrics_taskA(records: List[Tuple[str, int, int]]) -> Dict[str, float]:
    # records: list of (video_id, gt_cls, pred_cls)
    import importlib
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'snippet', 'src')))
    dice = importlib.import_module('metrics4MICCAI24.dice_score')
    ec = importlib.import_module('metrics4MICCAI24.expected_cost')
    df = pd.DataFrame(records, columns=['item_id', 'ground_truth', 'prediction'])
    # keep snippet prints (per-class F1) visible for bias inspection
    f1 = dice.get_f1(df, num_classes=4)
    cost = ec.get_expected_cost(df, num_classes=4)
    # Print confusion matrix for visibility
    cm = importlib.import_module('metrics4MICCAI24.utils').get_multi_class_confusion_matrix(df, num_classes=4)
    print('Confusion matrix (rows=GT, cols=Pred):')
    for row in cm:
        print(row)
    acc = (df['ground_truth'] == df['prediction']).mean()
    return {'f1': float(f1), 'expected_cost': float(cost), 'acc': float(acc)}


def compute_metrics_taskB(records: List[Tuple[str, int, int]], preds_raw: List[np.ndarray], gts_raw: List[np.ndarray]) -> Dict[str, float]:
    # records: flattened (item_id, gt_int, pred_int) where classes 0..4
    import importlib
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'snippet', 'src')))
    dice = importlib.import_module('metrics4MICCAI24.dice_score')
    ec = importlib.import_module('metrics4MICCAI24.expected_cost')
    utils = importlib.import_module('metrics4MICCAI24.utils')
    df = pd.DataFrame(records, columns=['item_id', 'ground_truth', 'prediction'])
    # keep snippet prints for class-wise F1 visibility
    f1 = dice.get_f1(df, num_classes=5)
    cost = ec.get_expected_cost(df, num_classes=5)
    # Also print per-target (category) class-wise F1
    cat_names = ['RESPECT', 'MOTION', 'INSTRUMENT', 'SUTURE', 'FLOW', 'KNOWLEDGE', 'PERFORMANCE', 'FINAL_QUALITY']
    print('Per-target class-wise F1 (classes 0..4):')
    for cn in cat_names:
        mask = df['item_id'].str.endswith(':' + cn)
        if not mask.any():
            continue
        # metrics snippet expects a 0..N-1 contiguous index; reset_index avoids KeyError
        df_cat = df[mask].reset_index(drop=True)
        tp, tn, fp, fn, _ = utils.get_multiclass_tp_tn_fp_fn(df_cat, num_classes=5)
        f1_per = []
        for i in range(5):
            if tp[i] > 0 and (tp[i] + fp[i]) > 0 and (tp[i] + fn[i]) > 0:
                precision = tp[i] / (tp[i] + fp[i])
                recall = tp[i] / (tp[i] + fn[i])
                f1_pc = 2 * precision * recall / (precision + recall)
            else:
                f1_pc = 0.0
            f1_per.append(f1_pc)
        print(f'  {cn}: ' + ', '.join(f'{v:.3f}' for v in f1_per))
        # Per-category confusion matrix
        cm_cat = utils.get_multi_class_confusion_matrix(df_cat, num_classes=5)
        print(f'  {cn} Confusion (rows=GT, cols=Pred):')
        for row in cm_cat:
            print('   ', row)
    # MAE over raw predictions (clamped) across 8 dims
    mae_list = []
    for pr, gt in zip(preds_raw, gts_raw):
        mae_list.append(np.abs(pr - gt).mean())
    mae = float(np.mean(mae_list))
    return {'f1': float(f1), 'expected_cost': float(cost), 'mae': mae}


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        # compute CE with optional label smoothing
        if self.label_smoothing > 0.0:
            num_classes = logits.shape[-1]
            with torch.no_grad():
                smooth = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth.scatter_(-1, target.unsqueeze(-1), 1.0 - self.label_smoothing)
            log_prob = torch.log_softmax(logits, dim=-1)
            ce = -(smooth * log_prob)
            if self.weight is not None:
                ce = ce * self.weight.view(1, -1)
            ce = ce.sum(dim=-1)
        else:
            ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction='none')
        pt = torch.softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1.0)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def seg_dice_loss_3d(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor | None = None,
                     ignore_index: int = -100, eps: float = 1e-6) -> torch.Tensor:
    """Multi-class Dice over 3D logits.
    logits: (B,C,T,H,W), target: (B,T,H,W) int with ignore_index.
    Returns mean 1-Dice weighted by class_weights (default equal weights).
    """
    B, C, T, H, W = logits.shape
    # mask valid voxels
    valid = (target != ignore_index)  # (B,T,H,W)
    if valid.sum() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=1)  # (B,C,T,H,W)
    # one-hot target
    tgt = target.clamp_min(0)
    onehot = torch.zeros((B, C, T, H, W), device=logits.device, dtype=probs.dtype)
    onehot.scatter_(1, tgt.unsqueeze(1), 1.0)
    v = valid.unsqueeze(1).to(probs.dtype)
    probs = probs * v
    onehot = onehot * v
    # per-class dice across (B,T,H,W)
    dims = (0, 2, 3, 4)
    inter = (probs * onehot).sum(dim=dims)
    denom = probs.sum(dim=dims) + onehot.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)  # (C,)
    if class_weights is None:
        loss = 1.0 - dice.mean()
    else:
        w = class_weights.to(dice.dtype).to(dice.device)
        w = w / (w.sum() + eps)
        loss = 1.0 - (dice * w).sum()
        return loss


def grs_to_cls_scalar(v: float) -> int:
    if v <= 15:
        return 0
    if v <= 23:
        return 1
    if v <= 31:
        return 2
    return 3


def soft_binning_probs(y_pred: torch.Tensor, centers: torch.Tensor, tau: float) -> torch.Tensor:
    """
    y_pred: (B,) predicted scalar (TaskA)
    centers: (K,) bin centers in same scale as y_pred
    Returns p: (B,K) softmax over negative distances/temperature
    """
    # compute distances
    # ensure shapes: (B,1) vs (1,K)
    y = y_pred.view(-1, 1)
    c = centers.view(1, -1)
    d = torch.abs(y - c)
    logits = -d / max(tau, 1e-6)
    p = torch.softmax(logits, dim=-1)
    return p


"""
EMA is provided by timm.utils.model_ema.ModelEmaV3. We enable warmup.
"""


def train_one_epoch(model, loader, optimizer, scaler, device, task: str, aux_weight: float, clamp_minmax=(0.0, 4.0), seg_class_weights: List[float] | None = None,
                    ce_weight_vec: List[float] | None = None, focal: bool = False, focal_gamma: float = 2.0, label_smoothing: float = 0.0,
                    grad_clip: float | None = None, loss_ec_weight: float = 0.1,
                    taskA_ec_weight: float = 0.0, taskA_ec_tau: float = 2.0,
                    w_time: float = 0.1, w_group: float = 0.1, w_sutures: float = 0.1,
                    ema: object | None = None):
    model.train()
    weight_tensor = None
    if ce_weight_vec is not None:
        weight_tensor = torch.tensor(ce_weight_vec, dtype=torch.float32, device=device)
    if focal:
        ce_fn = FocalLoss(weight=weight_tensor, gamma=focal_gamma, label_smoothing=label_smoothing)
    else:
        ce_fn = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    l1 = nn.SmoothL1Loss()
    if seg_class_weights is not None:
        w = torch.tensor(seg_class_weights, dtype=torch.float32, device=device)
    else:
        w = None
    # seg loss will use Dice; keep weights via w

    total = 0.0
    for batch in tqdm(loader, desc='train'):
        clip = batch['clip'].to(device)  # BxCxTxHxW (via DataLoader collation -> torch stacks)
        seg_labels = batch['seg_labels'].to(device)  # BxT'x7x7 after collation

        optimizer.zero_grad(set_to_none=True)
        # New AMP API
        with torch.amp.autocast('cuda', enabled=(scaler is not None and device.type == 'cuda')):
            out = model(clip)
            seg_logits = out['seg_logits']  # Bx7xT'x7x7
            # Dice loss over (B,T,H,W); guard when no valid labels
            if (seg_labels != -100).any():
                seg_loss = seg_dice_loss_3d(seg_logits, seg_labels, class_weights=w, ignore_index=-100)
            else:
                seg_loss = torch.zeros((), device=device)

            if task == 'A':
                y = batch['yA'].to(device)  # scalar GRS
                preds = out['main'].squeeze(-1)
                # Important: do NOT clamp before loss, to keep gradients
                l1_main = nn.functional.l1_loss(preds, y)
                # Soft-binning EC loss (no extra head):
                if taskA_ec_weight > 0.0:
                    # bin centers in GRS space (8..15,16..23,24..31,32..40)
                    centers = torch.tensor([11.5, 19.5, 27.5, 36.0], device=preds.device, dtype=preds.dtype)
                    psoft = soft_binning_probs(preds, centers, taskA_ec_tau)  # (B,4)
                    # ground-truth class index
                    y_cls = torch.tensor([grs_to_cls_scalar(float(v)) for v in y.detach().cpu().tolist()], device=preds.device, dtype=torch.long)
                    idx = torch.arange(4, device=preds.device, dtype=preds.dtype).view(1, 4)
                    cost = torch.abs(idx - y_cls.view(-1, 1).to(preds.dtype)) / 3.0
                    ec_loss_a = (psoft * cost).sum(dim=-1).mean()
                else:
                    ec_loss_a = torch.zeros((), device=preds.device, dtype=preds.dtype)
                loss_main = l1_main + taskA_ec_weight * ec_loss_a
            else:
                # TaskB: regression on 8 numeric OSATS scores in [1,5]
                y = batch['yB'].to(device)  # (B,8)
                preds = out['main']         # (B,8)
                preds_clamped = torch.clamp(preds, min=1.0, max=5.0)
                loss_main = nn.functional.smooth_l1_loss(preds_clamped, y)

            # Aux multitask losses (if labels provided)
            aux_losses = 0.0
            if 'y_time' in batch:
                y_time = batch['y_time'].to(device)
                aux_losses = aux_losses + w_time * nn.functional.cross_entropy(out['time_logits'], y_time)
            if 'y_group' in batch:
                y_group = batch['y_group'].to(device)
                aux_losses = aux_losses + w_group * nn.functional.cross_entropy(out['group_logits'], y_group)
            if 'y_sutures' in batch:
                y_sut = batch['y_sutures'].to(device)
                aux_losses = aux_losses + w_sutures * l1(out['sutures_pred'], y_sut)

            loss = loss_main + aux_weight * seg_loss + aux_losses

        if scaler is None:
            loss.backward()
            # optional grad clipping
            if grad_clip and grad_clip > 0:
                try:
                    import torch.nn.utils as nn_utils
                    nn_utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                except Exception:
                    pass
            optimizer.step()
            if ema is not None:
                ema.update(model)
        else:
            scaler.scale(loss).backward()
            # optional grad clipping (unscale first)
            if grad_clip and grad_clip > 0:
                try:
                    import torch.nn.utils as nn_utils
                    scaler.unscale_(optimizer)
                    nn_utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                except Exception:
                    pass
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device, task: str, clamp_minmax=(0.0, 4.0)):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    l1 = nn.L1Loss(reduction='sum')

    if task == 'A':
        # Regression on GRS (8..40). Report L1 and bin into 4 classes for F1/Expected Cost
        recs = []
        loss_sum = 0.0
        n = 0
        for batch in tqdm(loader, desc='val'):
            clip = batch['clip'].to(device)
            y = batch['yA'].to(device)  # scalar GRS
            out = model(clip)
            preds_raw = out['main'].squeeze(1)
            # Compute loss on raw preds for consistency with training
            loss_sum += nn.L1Loss(reduction='sum')(preds_raw, y).item()
            # to 4-class for metrics
            def grs_to_cls(v: float) -> int:
                if v <= 15: return 0
                if v <= 23: return 1
                if v <= 31: return 2
                return 3
            p = float(torch.clamp(preds_raw, min=8.0, max=40.0).item())
            g = float(y.item())
            recs.append((batch['video_id'][0], grs_to_cls(g), grs_to_cls(p)))
            n += 1
        metrics = compute_metrics_taskA(recs)
        metrics['l1'] = float(loss_sum / max(1, n))
        return metrics
    else:
        recs = []
        preds_raw = []  # list of numpy arrays (pred numeric in [1,5]), for MAE reporting
        gts_raw = []
        loss_sum = 0.0
        n = 0
        for batch in tqdm(loader, desc='val'):
            clip = batch['clip'].to(device)
            y = batch['yB'].to(device)
            out = model(clip)
            logits = out['main']
            if logits.dim() == 2 and logits.shape[1] == 40:
                logits = logits.view(logits.shape[0], 8, 5)
                p = torch.softmax(logits, dim=-1)
                bins = torch.arange(5, device=logits.device, dtype=logits.dtype).view(1, 1, 5)
                expv0 = (p * bins).sum(dim=-1)  # (B,8) zero-based expectation
                # Convert to numeric scale [1,5] for reporting/metrics
                pred_num = expv0 + 1.0
                # loss over numeric prediction vs GT
                loss_sum += l1(pred_num, y).item()
                pr = torch.clamp(pred_num, min=1.0, max=5.0).squeeze(0).cpu().numpy()
            else:
                preds_t = logits
                loss_sum += l1(preds_t, y).item()
                pr = torch.clamp(preds_t, min=1.0, max=5.0).squeeze(0).cpu().numpy()
            gt = y.squeeze(0).cpu().numpy()
            preds_raw.append(pr)
            gts_raw.append(gt)
            # build integer predictions for official metrics
            pr_int = np.clip(np.rint(pr) - 1, 0, 4).astype(int)
            gt_int = np.clip(np.rint(gt) - 1, 0, 4).astype(int)
            cat_names = ['RESPECT', 'MOTION', 'INSTRUMENT', 'SUTURE', 'FLOW', 'KNOWLEDGE', 'PERFORMANCE', 'FINAL_QUALITY']
            vid = batch['video_id'][0]
            for k in range(8):
                recs.append((f'{vid}:{cat_names[k]}', int(gt_int[k]), int(pr_int[k])))
                n += 1
        metrics = compute_metrics_taskB(recs, preds_raw, gts_raw)
        metrics['loss'] = float(loss_sum / max(1, n))
        return metrics


def main():
    ap = argparse.ArgumentParser(description='Train Swin3D-B for TaskA/B with SAM2 pseudo-mask aux loss')
    ap.add_argument('--task', choices=['A', 'B'], required=True)
    ap.add_argument('--model', choices=['model1', 'model2'], required=True, help='model1: full-clip bins; model2: last-5s')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--n-frames', type=int, default=96)
    ap.add_argument('--last5s-sec', type=int, default=5)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--opt', type=str, default='sf-radam', choices=['adamw', 'sf-radam', 'sf-adamw'],
                    help='optimizer: adamw or schedule-free variants (sf-radam/sf-adamw)')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--aux-weight', type=float, default=1.0)
    ap.add_argument('--balanced-sampler', action='store_true')
    ap.add_argument('--ce-class-weighting', choices=['none', 'inv', 'inv_sqrt'], default='inv_sqrt')
    ap.add_argument('--label-smoothing', type=float, default=0.05)
    ap.add_argument('--focal-loss', action='store_true')
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--freeze-backbone-epochs', type=int, default=1)
    ap.add_argument('--loss-ec-weight', type=float, default=0.1, help='weight for differentiable expected-cost loss (TaskB)')
    ap.add_argument('--taskA-ec-weight', type=float, default=0.1, help='weight for TaskA soft-binning expected-cost loss')
    ap.add_argument('--taskA-ec-tau', type=float, default=2.0, help='temperature for TaskA soft-binning (higher=softer)')
    ap.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay for model parameters (0=off)')
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--out-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'skill_swin3d')))
    ap.add_argument('--split-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'splits', 'oss25_5fold_v1')))
    ap.add_argument('--xlsx', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'OSATS_MICCAI_trainset.xlsx')))
    ap.add_argument('--osats-xlsx', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'OSATS.xlsx')))
    ap.add_argument('--tenfps-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'sam2_frames_10fps')))
    ap.add_argument('--pseudo6-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'seg_pseudo6')))
    ap.add_argument('--frames1fps-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'frames_1fps')))
    ap.add_argument('--seg-bg-weight', type=float, default=1.0, help='class weight for background (others=1.0)')
    # Aux multitask weights
    ap.add_argument('--w-time', type=float, default=0.1, help='weight for TIME (binary) aux loss')
    ap.add_argument('--w-group', type=float, default=0.1, help='weight for GROUP (3-class) aux loss')
    ap.add_argument('--w-sutures', type=float, default=0.1, help='weight for SUTURES regression aux loss')
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    labels = aggregate_labels(args.xlsx, args.osats_xlsx)
    loader_train, loader_val = build_dataloaders(args, labels)

    # Define model
    if args.task == 'A':
        # Regression head: scalar GRS
        num_outputs = 1
    else:
        # TaskB: regression head (8 numeric scores in [1,5])
        num_outputs = 8
    model = Swin3DWithSegAux(task=args.task, num_outputs=num_outputs, weights='KINETICS400_IMAGENET22K_V1')
    model = model.to(device)
    ema = ModelEmaV3(model, decay=args.ema_decay, device=device, use_warmup=True) if args.ema_decay and args.ema_decay > 0 else None

    # Optimizer: prefer schedule-free if requested and available
    optimizer = None
    if args.opt.startswith('sf'):
        try:
            from schedulefree import RAdamScheduleFree, AdamWScheduleFree
            if args.opt == 'sf-radam':
                optimizer = RAdamScheduleFree(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            print(f"[opt] Using schedule-free optimizer: {args.opt}")
        except Exception as e:
            print(f"[warn] schedulefree not available ({e}); falling back to AdamW")
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Output dir
    out_dir = os.path.join(args.out_root, f'task{args.task}', args.model, f'fold_{args.fold}')
    os.makedirs(out_dir, exist_ok=True)

    best_metric = None
    best_tie = None
    best_path = os.path.join(out_dir, 'best.pt')
    last_path = os.path.join(out_dir, 'last.pt')

    hist = []
    for epoch in range(1, args.epochs + 1):
        # put schedule-free optimizers in train mode if supported
        if hasattr(optimizer, 'train') and callable(getattr(optimizer, 'train')):
            try:
                optimizer.train()
            except Exception:
                pass
        # Optional warmup: freeze backbone for first few epochs to avoid collapse
        if epoch <= max(0, args.freeze_backbone_epochs):
            for n, p in model.backbone.named_parameters():
                p.requires_grad = False
            for p in model.main_head.parameters():
                p.requires_grad = True
            for p in model.seg_head.parameters():
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True
        seg_w = [args.seg_bg_weight] + [1.0] * 6
        # Build CE class weights for TaskA if requested
        ce_w = None
        if args.task == 'A' and args.ce_class_weighting != 'none':
            # compute counts from train split
            train_videos, _ = load_splits(args.split_root, args.fold)
            counts = [0, 0, 0, 0]
            for vid in train_videos:
                if vid in labels:
                    counts[int(labels[vid]['grs_cls'])] += 1
            ce_w = []
            for c in range(4):
                if args.ce_class_weighting == 'inv':
                    ce_w.append(1.0 / max(1, counts[c]))
                else:
                    ce_w.append(1.0 / max(1.0, float(counts[c]) ** 0.5))
        tr_loss = train_one_epoch(model, loader_train, optimizer, scaler, device, task=args.task, aux_weight=args.aux_weight,
                                   seg_class_weights=seg_w, ce_weight_vec=ce_w, focal=args.focal_loss, focal_gamma=args.focal_gamma,
                                   label_smoothing=args.label_smoothing, grad_clip=getattr(args, 'grad_clip', None),
                                   loss_ec_weight=getattr(args, 'loss_ec_weight', 0.1),
                                   taskA_ec_weight=getattr(args, 'taskA_ec_weight', 0.0), taskA_ec_tau=getattr(args, 'taskA_ec_tau', 2.0),
                                   w_time=getattr(args, 'w_time', 0.1), w_group=getattr(args, 'w_group', 0.1), w_sutures=getattr(args, 'w_sutures', 0.1),
                                   ema=ema)
        # switch to eval mode for schedule-free optimizers during validation
        if hasattr(optimizer, 'eval') and callable(getattr(optimizer, 'eval')):
            try:
                optimizer.eval()
            except Exception:
                pass
        val_metrics = validate(ema.module if ema is not None else model, loader_val, device, task=args.task)
        row = {'epoch': epoch, 'train_loss': tr_loss, **val_metrics}
        hist.append(row)
        with open(os.path.join(out_dir, 'log.jsonl'), 'a') as f:
            f.write(json.dumps(row) + '\n')

        # Choose best by F1 (higher is better). If tie on F1:
        #  - TaskA: prefer lower 'l1'
        #  - TaskB: prefer lower 'mae'
        score = val_metrics.get('f1', 0.0)
        tie_val = val_metrics.get('l1', float('inf')) if args.task == 'A' else val_metrics.get('mae', float('inf'))
        is_best = False
        if best_metric is None or score > best_metric:
            is_best = True
        elif best_metric is not None and score == best_metric:
            if best_tie is None or tie_val < best_tie:
                is_best = True
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'model_ema': (ema.module.state_dict() if ema is not None else None), 'args': vars(args), 'metrics': val_metrics}, last_path)
        if is_best:
            best_metric = score
            best_tie = tie_val
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'model_ema': (ema.module.state_dict() if ema is not None else None), 'args': vars(args), 'metrics': val_metrics}, best_path)

        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val={val_metrics}")


if __name__ == '__main__':
    main()
