#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from oss25.task3_kp_dataset import Task3KeypointDataset
from oss25.models.kpnet import KPNet
import timm


def gaussian_heatmap_targets(batch_targets, sigma=2.0):
    # Already generated in dataset; just placeholder if future changes
    return batch_targets

def softargmax_coords(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    # logits: B,C,H,W -> coords: B,C,2 (x,y)
    B, C, H, W = logits.shape
    flat = (logits / max(temperature, 1e-6)).view(B, C, -1)
    prob = torch.softmax(flat, dim=-1)
    xs = torch.arange(W, device=logits.device, dtype=logits.dtype)
    ys = torch.arange(H, device=logits.device, dtype=logits.dtype)
    grid_x = xs.repeat(H)
    grid_y = ys.repeat_interleave(W)
    ex = (prob * grid_x.view(1, 1, H * W)).sum(dim=-1)
    ey = (prob * grid_y.view(1, 1, H * W)).sum(dim=-1)
    return torch.stack([ex, ey], dim=-1)


def multiclass_dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, eps: float = 1e-6) -> torch.Tensor:
    # logits: (B,C,H,W), target: (B,H,W) in {0..C-1} or -100 to ignore
    B, C, H, W = logits.shape
    valid = (target != ignore_index)
    if valid.sum() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=1)
    # one-hot target
    t = torch.zeros((B, C, H, W), device=logits.device, dtype=probs.dtype)
    t.scatter_(1, target.clamp_min(0).unsqueeze(1), 1.0)
    # mask invalid pixels
    v = valid.unsqueeze(1).to(probs.dtype)
    probs = probs * v
    t = t * v
    # per-class dice then average over classes present in valid region
    dims = (0, 2, 3)
    intersection = (probs * t).sum(dim=dims)
    denom = probs.sum(dim=dims) + t.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    # average only over classes that appear in target (to reduce bg dominance)
    present = (t.sum(dim=dims) > 0).to(dice.dtype)
    if present.sum() > 0:
        dice_mean = (dice * present).sum() / present.sum()
    else:
        dice_mean = dice.mean()
    return 1.0 - dice_mean


def focal_ce_from_logits(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-8) -> torch.Tensor:
    # Multiclass focal based on CE. logits: (B,C,H,W), target: (B,H,W)
    B, C, H, W = logits.shape
    valid = (target != ignore_index)
    if valid.sum() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    logp = torch.log_softmax(logits, dim=1)  # (B,C,H,W)
    # gather log-prob for target classes
    tgt = target.clamp_min(0)
    logpt = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)  # (B,H,W)
    pt = logpt.exp()
    # alpha weighting: treat background(0) vs foreground(>0) differently
    alpha_map = torch.where(tgt == 0, 1.0 - alpha, alpha).to(logp.dtype)
    loss = -alpha_map * (1 - pt) ** gamma * logpt
    loss = loss[valid]
    return loss.mean()


def _build_model_input(batch, mean, std, in_chans: int, flow_scale: float = 20.0, device=None):
    img = batch['image']  # B,3,H,W (already on device in norm_loader)
    x_img = img  # in norm_loader we will pre-normalize image
    if in_chans <= 3 or ('flow_uv' not in batch and 'fg_mask' not in batch):
        return x_img
    flow = batch.get('flow_uv', None)
    fg = batch.get('fg_mask', None)
    parts = [x_img]
    if flow is not None:
        flow_n = torch.clamp(flow / flow_scale, -1.0, 1.0)
        parts.append(flow_n)
    if fg is not None:
        fg_n = torch.clamp(fg * 2.0 - 1.0, -1.0, 1.0)
        parts.append(fg_n)
    return torch.cat(parts, dim=1)


def _warp_with_flow(tensor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    # tensor: B,C,H,W; flow: B,2,H,W (dx,dy) in pixels
    B, C, H, W = tensor.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=tensor.device), torch.arange(W, device=tensor.device), indexing='ij')
    grid_x = (xx.float() + flow[:, 0])
    grid_y = (yy.float() + flow[:, 1])
    gx = (grid_x / (W - 1)).clamp(0, 1) * 2 - 1
    gy = (grid_y / (H - 1)).clamp(0, 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)
    return torch.nn.functional.grid_sample(tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)


def _bilinear_sample(flow: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    # flow: B,2,H,W; coords: B,C,2 at image-scale pixel coords
    B, _, H, W = flow.shape
    C = coords.shape[1]
    x = coords[..., 0].clamp(0, W - 1)
    y = coords[..., 1].clamp(0, H - 1)
    gx = (x / (W - 1)) * 2 - 1  # B,C
    gy = (y / (H - 1)) * 2 - 1  # B,C
    grid = torch.stack([gx, gy], dim=-1).view(B, C, 1, 2)  # B,H_out(=C),W_out(=1),2
    u = torch.nn.functional.grid_sample(flow[:, 0:1], grid, mode='bilinear', align_corners=True).squeeze(1).squeeze(-1)  # B,C
    v = torch.nn.functional.grid_sample(flow[:, 1:2], grid, mode='bilinear', align_corners=True).squeeze(1).squeeze(-1)  # B,C
    return torch.stack([u, v], dim=-1)  # B,C,2


def _meta_get(meta: dict, key: str, index: int):
    v = meta[key]
    # tuple-of-sequences (e.g., (H_list, W_list)) after default_collate
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and all(hasattr(x, '__getitem__') for x in v):
            a, b = v[0][index], v[1][index]
            if torch.is_tensor(a):
                a = a.item()
            if torch.is_tensor(b):
                b = b.item()
            return a, b
        elem = v[index]
        if torch.is_tensor(elem) and elem.numel() == 1:
            return elem.item()
        return elem
    if torch.is_tensor(v) and v.numel() == 1:
        return v.item()
    return v


def _resize_to_original_map(t: torch.Tensor, meta: dict, index: int, *, is_label: bool = False) -> torch.Tensor:
    """Resize/crop a spatial map (C,H,W) or (H,W) back to original (H0,W0).
    Handles letterbox by cropping valid region before resizing.
    """
    H0, W0 = _meta_get(meta, 'orig_size', index)
    # letterbox cropping is intentionally skipped in this pipeline to evaluate on full original canvas
    if t.dim() == 3:
        # (C,H,W)
        if is_label:
            orig_dtype = t.dtype
            t = t.to(torch.float32)
            t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(H0, W0), mode='nearest').squeeze(0)
            t = t.to(orig_dtype)
        else:
            t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(H0, W0), mode='bilinear').squeeze(0)
    elif t.dim() == 2:
        # (H,W)
        if is_label:
            orig_dtype = t.dtype
            t = t.to(torch.float32)
            t = torch.nn.functional.interpolate(t.unsqueeze(0).unsqueeze(0), size=(H0, W0), mode='nearest').squeeze(0).squeeze(0)
            t = t.to(orig_dtype)
        else:
            t = torch.nn.functional.interpolate(t.unsqueeze(0).unsqueeze(0), size=(H0, W0), mode='bilinear').squeeze(0).squeeze(0)
    else:
        raise ValueError('Unsupported tensor shape for resize_to_original_map')
    return t


def _coords_to_original(coords_hw: torch.Tensor, meta: dict, index: int) -> torch.Tensor:
    """Convert coordinates from heatmap/image canvas to original pixel coords.
    coords_hw: (C,2) in current heatmap/image canvas.
    """
    sc = _meta_get(meta, 'scale', index)
    if isinstance(sc, (tuple, list)):
        sx, sy = float(sc[0]), float(sc[1])
    else:
        sx = sy = float(sc)
    x = coords_hw[:, 0] / max(sx, 1e-6)
    y = coords_hw[:, 1] / max(sy, 1e-6)
    return torch.stack([x, y], dim=-1)


def train_one_epoch(model, loader, optimizer, device, scaler=None, *, pos_weight: float = 50.0,
                    coord_w: float = 0.2, coord_temp: float = 1.0, seg_w: float = 0.0,
                    hm_consist_w: float = 0.0, coord_consist_w: float = 0.0) -> Dict:
    model.train()
    ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
    total_loss = 0.0
    total_hm = 0.0
    total_vis = 0.0
    total_coord = 0.0
    total_seg = 0.0
    count = 0
    for batch in loader:
        img = batch.get('input', batch['image'])
        hm_t = batch['heatmaps']
        vis_t = batch['vis_labels']  # (B,C)
        coord_t = batch['coord_targets']  # (B,C,2) in heatmap coords
        coord_wt = batch['coord_weights']  # (B,C)
        prev_img = batch.get('prev_input', None)
        flow_uv = batch.get('flow_uv', None)
        meta = batch.get('meta', None)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            if seg_w > 0:
                hm_p, vis_p, seg_p = model(img, return_seg=True)  # hm: B,C,H,W; vis: B,C,3; seg: B,7,H,W
            else:
                hm_p, vis_p = model(img, return_seg=False)
                seg_p = None
            # BCE with logits for heatmap with positive class upweighting at original resolution
            B, C, Hh, Wh = hm_p.shape
            pw = torch.full((C,), float(pos_weight), device=hm_p.device, dtype=hm_p.dtype)
            # per-channel visibility weights: 0 -> 0, 1/2 -> 1.0
            w_ch = torch.where((vis_t >= 1), torch.ones_like(vis_t, dtype=hm_p.dtype), torch.zeros_like(vis_t, dtype=hm_p.dtype))  # (B,C)
            hm_bce_sum = torch.zeros((), device=hm_p.device, dtype=hm_p.dtype)
            hm_denom_sum = torch.zeros((), device=hm_p.device, dtype=hm_p.dtype)
            for i in range(B):
                # resize preds/targets to original canvas (H0,W0)
                pred_i = _resize_to_original_map(hm_p[i], meta, i, is_label=False)  # (C,H0,W0)
                tgt_i = _resize_to_original_map(hm_t[i], meta, i, is_label=False)   # (C,H0,W0)
                H0, W0 = _meta_get(meta, 'orig_size', i)
                # weight map per pixel per channel
                w_b_i = w_ch[i].view(1, 1, C).expand(H0, W0, C)
                # compute BCE in HWxC layout for pos_weight broadcasting
                inp_i = pred_i.permute(1, 2, 0)
                tgt_i2 = tgt_i.permute(1, 2, 0)
                bce_i = torch.nn.functional.binary_cross_entropy_with_logits(inp_i, tgt_i2, pos_weight=pw, weight=w_b_i, reduction='sum')
                hm_bce_sum = hm_bce_sum + bce_i
                hm_denom_sum = hm_denom_sum + w_b_i.sum()
            hm_loss = hm_bce_sum / hm_denom_sum.clamp(min=1.0)
            # Visibility CE (size-agnostic)
            Bv, Cv, _H, _W = hm_p.shape
            vis_loss = ce(vis_p.view(Bv * Cv, 3), vis_t.view(Bv * Cv))
            # Coordinate L1 on soft-argmax locations, evaluated in original coords
            coord_pred = softargmax_coords(hm_p, temperature=coord_temp)  # (B,C,2)
            coord_num = torch.zeros((), device=hm_p.device, dtype=hm_p.dtype)
            coord_den = torch.zeros((), device=hm_p.device, dtype=hm_p.dtype)
            for i in range(B):
                pred_xy_hw = coord_pred[i]  # (C,2)
                tgt_xy_hw = coord_t[i]
                pred_xy_ori = _coords_to_original(pred_xy_hw, meta, i)
                tgt_xy_ori = _coords_to_original(tgt_xy_hw, meta, i)
                l1_i = torch.abs(pred_xy_ori - tgt_xy_ori).sum(dim=-1)  # (C,)
                w_i = coord_wt[i]
                coord_num = coord_num + (l1_i * w_i).sum()
                coord_den = coord_den + w_i.sum()
            coord_loss = coord_num / coord_den.clamp(min=1.0)
            # Segmentation: Dice + Focal at original resolution (if labels exist)
            seg_loss = torch.tensor(0.0, device=device)
            if seg_w > 0 and seg_p is not None and 'seg_labels6' in batch:
                seg_t = batch['seg_labels6']  # (B,Hh,Wh)
                seg_sum = torch.zeros((), device=hm_p.device, dtype=hm_p.dtype)
                seg_cnt = 0
                for i in range(B):
                    seg_t_i = seg_t[i]
                    # consider valid only if any label != -100
                    if isinstance(seg_t_i, torch.Tensor) and (seg_t_i != -100).any():
                        seg_logits_i = _resize_to_original_map(seg_p[i], meta, i, is_label=False).unsqueeze(0)  # (1,7,H0,W0)
                        seg_lbl_i = _resize_to_original_map(seg_t_i, meta, i, is_label=True).unsqueeze(0)       # (1,H0,W0)
                        dice = multiclass_dice_loss_from_logits(seg_logits_i, seg_lbl_i.long(), ignore_index=-100)
                        focal = focal_ce_from_logits(seg_logits_i, seg_lbl_i.long(), ignore_index=-100, gamma=2.0, alpha=0.25)
                        seg_sum = seg_sum + (0.5 * dice + 0.5 * focal)
                        seg_cnt += 1
                if seg_cnt > 0:
                    seg_loss = seg_sum / float(seg_cnt)
            # temporal consistency (unchanged; defined on current canvas)
            hm_tc = torch.tensor(0.0, device=device)
            coord_tc = torch.tensor(0.0, device=device)
            if (hm_consist_w > 0 or coord_consist_w > 0) and (prev_img is not None) and (flow_uv is not None):
                with torch.no_grad():
                    hm_prev, _ = model(prev_img, return_seg=False)
                if hm_consist_w > 0:
                    # resize flow to heatmap and scale magnitude per axis
                    H_img, W_img = flow_uv.shape[-2], flow_uv.shape[-1]
                    flow_small = torch.nn.functional.interpolate(flow_uv, size=(Hh, Wh), mode='bilinear', align_corners=False)
                    scale_u = float(Wh) / float(W_img)
                    scale_v = float(Hh) / float(H_img)
                    scale_map = torch.tensor([scale_u, scale_v], device=flow_uv.device, dtype=flow_uv.dtype).view(1, 2, 1, 1)
                    flow_small = flow_small * scale_map
                    hm_prev_prob = torch.sigmoid(hm_prev)
                    hm_prev_warp = _warp_with_flow(hm_prev_prob, flow_small)
                    hm_curr_prob = torch.sigmoid(hm_p)
                    hm_tc = torch.mean((hm_prev_warp - hm_curr_prob) ** 2)
                if coord_consist_w > 0:
                    coords_prev = softargmax_coords(hm_prev, temperature=coord_temp)  # heatmap coords (xh, yh)
                    H_img, W_img = flow_uv.shape[-2], flow_uv.shape[-1]
                    scale_x = float(W_img) / float(Wh)
                    scale_y = float(H_img) / float(Hh)
                    coords_prev_img = torch.stack([coords_prev[..., 0] * scale_x, coords_prev[..., 1] * scale_y], dim=-1)
                    flow_at_prev = _bilinear_sample(flow_uv, coords_prev_img)
                    coords_warp_img = coords_prev_img + flow_at_prev
                    coords_curr = torch.stack([coord_pred[..., 0] * scale_x, coord_pred[..., 1] * scale_y], dim=-1)
                    l1c = torch.abs(coords_warp_img - coords_curr).sum(dim=-1)
                    coord_tc = (l1c * coord_wt).sum() / coord_wt.sum().clamp(min=1.0)
            loss = hm_loss + 0.1 * vis_loss + coord_w * coord_loss + seg_w * seg_loss + hm_consist_w * hm_tc + coord_consist_w * coord_tc
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_hm += float(hm_loss.detach().cpu())
        total_vis += float(vis_loss.detach().cpu())
        total_coord += float(coord_loss.detach().cpu())
        if seg_w > 0:
            total_seg += float(seg_loss.detach().cpu())
        count += 1
    return {
        'loss': total_loss / max(count, 1),
        'hm_loss': total_hm / max(count, 1),
        'vis_loss': total_vis / max(count, 1),
        'coord_loss': total_coord / max(count, 1),
        'seg_loss': (total_seg / max(count, 1) if seg_w > 0 else 0.0),
        'hm_tc': 0.0, 'coord_tc': 0.0
    }


@torch.no_grad()
def evaluate(model, loader, device, *, pos_weight: float = 50.0, coord_w: float = 0.2, coord_temp: float = 1.0, seg_w: float = 0.0,
             hm_consist_w: float = 0.0, coord_consist_w: float = 0.0) -> Dict:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
    total_loss = 0.0
    total_hm = 0.0
    total_vis = 0.0
    total_coord = 0.0
    total_seg = 0.0
    count = 0
    # per-class IoU accumulators for segmentation (0..6 where 0=bg)
    i_inter = torch.zeros(7, dtype=torch.float64)
    i_union = torch.zeros(7, dtype=torch.float64)
    for batch in loader:
        img = batch.get('input', batch['image'])
        hm_t = batch['heatmaps']
        vis_t = batch['vis_labels']
        coord_t = batch['coord_targets']
        coord_wt = batch['coord_weights']
        meta = batch.get('meta', None)
        if seg_w > 0:
            hm_p, vis_p, seg_p = model(img, return_seg=True)
        else:
            hm_p, vis_p = model(img, return_seg=False)
            seg_p = None
        B, C, Hh, Wh = hm_p.shape
        pw = torch.full((C,), float(pos_weight), device=hm_p.device, dtype=hm_p.dtype)
        # visibility-based per-channel weights
        w_ch = torch.where((vis_t >= 1), torch.ones_like(vis_t, dtype=hm_p.dtype), torch.zeros_like(vis_t, dtype=hm_p.dtype))
        hm_bce_sum = torch.zeros((), device=hm_p.device, dtype=hm_p.dtype)
        hm_denom_sum = torch.zeros((), device=hm_p.device, dtype=hm_p.dtype)
        for i in range(B):
            pred_i = _resize_to_original_map(hm_p[i], meta, i, is_label=False)
            tgt_i = _resize_to_original_map(hm_t[i], meta, i, is_label=False)
            H0, W0 = _meta_get(meta, 'orig_size', i)
            w_b_i = w_ch[i].view(1, 1, C).expand(H0, W0, C)
            inp_i = pred_i.permute(1, 2, 0)
            tgt_i2 = tgt_i.permute(1, 2, 0)
            bce_i = torch.nn.functional.binary_cross_entropy_with_logits(inp_i, tgt_i2, pos_weight=pw, weight=w_b_i, reduction='sum')
            hm_bce_sum = hm_bce_sum + bce_i
            hm_denom_sum = hm_denom_sum + w_b_i.sum()
        hm_loss = hm_bce_sum / hm_denom_sum.clamp(min=1.0)
        Bv, Cv, _H, _W = hm_p.shape
        vis_loss = ce(vis_p.view(Bv * Cv, 3), vis_t.view(Bv * Cv))
        coord_pred = softargmax_coords(hm_p, temperature=coord_temp)
        coord_num = torch.zeros((), device=hm_p.device, dtype=hm_p.dtype)
        coord_den = torch.zeros((), device=hm_p.device, dtype=hm_p.dtype)
        for i in range(B):
            pred_xy_ori = _coords_to_original(coord_pred[i], meta, i)
            tgt_xy_ori = _coords_to_original(coord_t[i], meta, i)
            l1_i = torch.abs(pred_xy_ori - tgt_xy_ori).sum(dim=-1)
            w_i = coord_wt[i]
            coord_num = coord_num + (l1_i * w_i).sum()
            coord_den = coord_den + w_i.sum()
        coord_loss = coord_num / coord_den.clamp(min=1.0)
        seg_loss = torch.tensor(0.0, device=device)
        if seg_w > 0 and seg_p is not None and 'seg_labels6' in batch:
            seg_t = batch['seg_labels6']
            seg_sum = torch.zeros((), device=hm_p.device, dtype=hm_p.dtype)
            seg_cnt = 0
            for i in range(B):
                seg_t_i = seg_t[i]
                if isinstance(seg_t_i, torch.Tensor) and (seg_t_i != -100).any():
                    seg_logits_i = _resize_to_original_map(seg_p[i], meta, i, is_label=False).unsqueeze(0)
                    seg_lbl_i = _resize_to_original_map(seg_t_i, meta, i, is_label=True).unsqueeze(0)
                    dice = multiclass_dice_loss_from_logits(seg_logits_i, seg_lbl_i.long(), ignore_index=-100)
                    focal = focal_ce_from_logits(seg_logits_i, seg_lbl_i.long(), ignore_index=-100, gamma=2.0, alpha=0.25)
                    seg_sum = seg_sum + (0.5 * dice + 0.5 * focal)
                    # IoU accumulation on original canvas
                    pred = seg_logits_i.argmax(dim=1)[0].detach().cpu().to(torch.int64)  # H0,W0
                    tgt = seg_lbl_i[0].detach().cpu().to(torch.int64)
                    mask = (tgt != -100)
                    if mask.any():
                        p = pred[mask]
                        t = tgt[mask]
                        for c in range(0, 7):
                            pc = (p == c)
                            tc = (t == c)
                            inter = (pc & tc).sum().item()
                            union = (pc | tc).sum().item()
                            i_inter[c] += inter
                            i_union[c] += union
                    seg_cnt += 1
            if seg_cnt > 0:
                seg_loss = seg_sum / float(seg_cnt)
        # combine
        loss = hm_loss + 0.1 * vis_loss + coord_w * coord_loss + seg_w * seg_loss
        total_loss += float(loss.detach().cpu())
        total_hm += float(hm_loss.detach().cpu())
        total_vis += float(vis_loss.detach().cpu())
        total_coord += float(coord_loss.detach().cpu())
        if seg_w > 0:
            total_seg += float(seg_loss.detach().cpu())
        count += 1
    res = {'val_loss': total_loss / max(count, 1), 'val_hm_loss': total_hm / max(count, 1), 'val_vis_loss': total_vis / max(count, 1), 'val_coord_loss': total_coord / max(count, 1), 'val_seg_loss': (total_seg / max(count, 1) if seg_w > 0 else 0.0)}
    if seg_w > 0 and i_union.sum() > 0:
        iou = torch.where(i_union > 0, i_inter / i_union.clamp(min=1.0), torch.zeros_like(i_union))
        # report per-class IoU for 1..6 and mIoU (1..6前景平均)
        for c in range(1, 7):
            res[f'val_seg_iou_{c}'] = float(iou[c].item())
        fg = iou[1:7]
        res['val_seg_mIoU'] = float(fg[fg==fg].mean().item())  # ignore NaN
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')))
    # sizes can be int (N) or 'H×W' (e.g., '540x960')
    ap.add_argument('--image-size', type=str, default='384')
    ap.add_argument('--heatmap-size', type=str, default='128')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--encoder', default='convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384')
    ap.add_argument('--out-dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'kpnet')))
    # augmentation and loss tuning
    ap.add_argument('--aug-rotate-deg', type=float, default=45.0, help='max abs degrees for random rotation (train only)')
    ap.add_argument('--hm-pos-weight', type=float, default=50.0, help='BCE pos_weight for heatmaps')
    ap.add_argument('--coord-loss-weight', type=float, default=0.2, help='loss weight for coordinate L1')
    ap.add_argument('--coord-temp', type=float, default=1.0, help='temperature for soft-argmax')
    ap.add_argument('--pseudo6-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'seg_pseudo6')),
                    help='root dir for SAM2 6-class indexed masks')
    ap.add_argument('--seg-loss-weight', type=float, default=0.15, help='aux segmentation loss weight (Dice+Focal)')
    # flow/temporal consistency
    ap.add_argument('--in-chans', type=int, default=6, help='input channels (3=RGB, 6=RGB+flow_uv+fg)')
    ap.add_argument('--flow-norm', type=float, default=20.0, help='flow normalization scale for extra channels')
    ap.add_argument('--hm-consist-weight', type=float, default=0.05, help='temporal consistency weight for heatmap')
    ap.add_argument('--coord-consist-weight', type=float, default=0.05, help='temporal consistency weight for coords')
    ap.add_argument('--resize-only', action='store_true', help='resize to square without letterbox (aspect ratio not preserved)')
    # HOTA eval & visualization (optional)
    ap.add_argument('--val-hota', action='store_true', help='run HOTA eval via TrackEval during training')
    ap.add_argument('--val-hota-interval', type=int, default=0, help='epochs interval for HOTA (0=only final epoch)')
    ap.add_argument('--hota-gt-folder', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'kp_eval', 'data', 'gt')),
                    help='aggregated GT folder for TrackEval')
    ap.add_argument('--hota-mode', choices=['det', 'cot'], default='det', help='HOTA eval inference mode')
    ap.add_argument('--hota-max-videos', type=int, default=0, help='limit number of val videos (0=all)')
    ap.add_argument('--val-vis-video', default='', help='video id to visualize after HOTA (empty=skip)')
    # dump predicted segmentation on val during training (Stage1向け)
    ap.add_argument('--dump-seg-val', action='store_true', help='dump predicted seg masks on val after epochs (seg_loss_weight>0 only)')
    ap.add_argument('--dump-seg-interval', type=int, default=0, help='epochs interval to dump seg (0=only final epoch)')
    ap.add_argument('--dump-seg-limit', type=int, default=8, help='max val samples to dump per epoch (0=all; be careful)')
    ap.add_argument('--dump-seg-video', default='', help='dump only this video id (empty=all)')
    ap.add_argument('--dump-seg-alpha', type=float, default=0.6, help='global alpha scale for seg overlay during dump (0..1)')
    ap.add_argument('--init', type=str, default='', help='init model weights from checkpoint (model only)')
    ap.add_argument('--resume', type=str, default='', help='resume full training from checkpoint (model+optimizer+epoch)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def _parse_hw(v: str) -> tuple[int,int]:
        try:
            if isinstance(v, (int, float)):
                n = int(v)
                return n, n
            s = str(v).lower().replace('×', 'x').replace(',', 'x')
            if 'x' in s:
                a, b = s.split('x')
                return int(a), int(b)
            n = int(s)
            return n, n
        except Exception:
            n = 384
            return n, n

    # parse sizes and force heatmap == image as依頼
    img_h, img_w = _parse_hw(args.image_size)
    hm_h, hm_w = _parse_hw(args.heatmap_size)
    # force identical
    hm_h, hm_w = img_h, img_w
    args.image_h, args.image_w = img_h, img_w
    args.heatmap_h, args.heatmap_w = hm_h, hm_w
    # keep string option consistent for downstream tools
    args.heatmap_size = args.image_size

    allow_seg_only = args.seg_loss_weight is not None and args.seg_loss_weight > 0
    train_ds = Task3KeypointDataset(args.data_root, split='train', image_size=(args.image_h, args.image_w), heatmap_size=(args.heatmap_h, args.heatmap_w),
                                    aug_rotate_deg=args.aug_rotate_deg, pseudo6_root=args.pseudo6_root, use_flow=(args.in_chans>3),
                                    allow_seg_only=allow_seg_only, resize_only=args.resize_only)
    val_ds = Task3KeypointDataset(args.data_root, split='val', image_size=(args.image_h, args.image_w), heatmap_size=(args.heatmap_h, args.heatmap_w),
                                  aug_rotate_deg=0.0, pseudo6_root=args.pseudo6_root, use_flow=(args.in_chans>3),
                                  allow_seg_only=False, resize_only=args.resize_only)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = KPNet(encoder_name=args.encoder, image_size=(args.image_h, args.image_w), heatmap_size=(args.heatmap_h, args.heatmap_w), num_keypoints=27, in_chans=args.in_chans)
    model = model.to(device)

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # checkpoint load
    start_epoch = 1
    if args.resume:
        sd = torch.load(args.resume, map_location='cpu')
        m = sd['model'] if 'model' in sd else sd
        model.load_state_dict(m, strict=False)
        if isinstance(sd, dict) and 'optimizer' in sd:
            try:
                optimizer.load_state_dict(sd['optimizer'])
            except Exception:
                pass
        if isinstance(sd, dict) and 'epoch' in sd:
            start_epoch = int(sd['epoch']) + 1
        print(f"[resume] loaded from {args.resume} -> start_epoch={start_epoch}")
    elif args.init:
        sd = torch.load(args.init, map_location='cpu')
        m = sd['model'] if 'model' in sd else sd
        model.load_state_dict(m, strict=False)
        print(f"[init] loaded model weights from {args.init}")

    best_val = float('inf')
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Data normalization per timm config
        data_cfg = timm.data.resolve_model_data_config(model.encoder)
        mean = torch.tensor(data_cfg.get('mean', (0.5, 0.5, 0.5)), dtype=torch.float32, device=device).view(1,3,1,1)
        std = torch.tensor(data_cfg.get('std', (0.5, 0.5, 0.5)), dtype=torch.float32, device=device).view(1,3,1,1)

        def norm_loader(loader):
            for batch in loader:
                # move tensors
                for k in ['image','prev_image','flow_uv','fg_mask','heatmaps','vis_labels','coord_targets','coord_weights','seg_labels6']:
                    if k in batch:
                        batch[k] = batch[k].to(device)
                # normalize image only, extras handled separately
                batch['image'] = (batch['image'] - mean) / std
                if 'prev_image' in batch:
                    batch['prev_image'] = (batch['prev_image'] - mean) / std
                # build model inputs with extras
                batch['input'] = _build_model_input(batch, mean, std, args.in_chans, flow_scale=args.flow_norm, device=device)
                if 'prev_image' in batch:
                    prev = batch['prev_image']
                    if args.in_chans > 3:
                        B, _, H, W = prev.shape
                        zeros = torch.zeros(B, args.in_chans - 3, H, W, device=device, dtype=prev.dtype)
                        prev = torch.cat([prev, zeros], dim=1)
                    batch['prev_input'] = prev
                batch['heatmaps'] = batch['heatmaps'].to(device)
                batch['vis_labels'] = batch['vis_labels'].to(device)
                batch['coord_targets'] = batch['coord_targets'].to(device)
                batch['coord_weights'] = batch['coord_weights'].to(device)
                if 'seg_labels6' in batch:
                    batch['seg_labels6'] = batch['seg_labels6'].to(device)
                yield batch

        tr = train_one_epoch(model, norm_loader(train_loader), optimizer, device, scaler,
                             pos_weight=args.hm_pos_weight, coord_w=args.coord_loss_weight, coord_temp=args.coord_temp,
                             seg_w=args.seg_loss_weight, hm_consist_w=args.hm_consist_weight, coord_consist_w=args.coord_consist_weight)
        ev = evaluate(model, norm_loader(val_loader), device,
                      pos_weight=args.hm_pos_weight, coord_w=args.coord_loss_weight, coord_temp=args.coord_temp,
                      seg_w=args.seg_loss_weight, hm_consist_w=args.hm_consist_weight, coord_consist_w=args.coord_consist_weight)
        # optional seg IoU in log
        seg_iou_str = ''
        if 'val_seg_mIoU' in ev:
            seg_iou_str = f" val_mIoU={ev['val_seg_mIoU']:.3f}"
        print(f"epoch {epoch}: loss={tr['loss']:.4f} hm={tr['hm_loss']:.4f} coord={tr['coord_loss']:.4f} seg={tr['seg_loss']:.4f} vis={tr['vis_loss']:.4f} | "
              f"val_loss={ev['val_loss']:.4f} val_hm={ev['val_hm_loss']:.4f} val_coord={ev['val_coord_loss']:.4f} val_seg={ev['val_seg_loss']:.4f} val_vis={ev['val_vis_loss']:.4f}{seg_iou_str}")
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(args.out_dir, 'last.pt'))
        if ev['val_loss'] < best_val:
            best_val = ev['val_loss']
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(args.out_dir, 'best.pt'))

        # Optional HOTA eval/visualization
        do_hota = False
        if args.val_hota:
            if args.val_hota_interval and args.val_hota_interval > 0:
                do_hota = ((epoch - start_epoch + 1) % args.val_hota_interval == 0)
            else:
                do_hota = (epoch == (start_epoch + args.epochs - 1))
        if do_hota:
            try:
                _run_hota_eval(args, epoch)
            except Exception as e:
                print(f"[warn] HOTA eval failed at epoch {epoch}: {e}")

        # Optional: dump predicted seg maps on val (only when seg loss enabled)
        do_dump_seg = False
        if args.dump_seg_val:
            if args.dump_seg_interval and args.dump_seg_interval > 0:
                do_dump_seg = ((epoch - start_epoch + 1) % args.dump_seg_interval == 0)
            else:
                do_dump_seg = (epoch == (start_epoch + args.epochs - 1))
        if do_dump_seg:
            try:
                _dump_val_seg_predictions(args, model, device, epoch)
            except Exception as e:
                print(f"[warn] dump seg failed at epoch {epoch}: {e}")


def _run_hota_eval(args, epoch: int):
    ckpt = os.path.join(args.out_dir, 'last.pt')
    trackers_dir = os.path.join(args.out_dir, 'hota', f'epoch_{epoch}', 'trackers')
    out_eval_dir = os.path.join(args.out_dir, 'hota', f'epoch_{epoch}', 'eval')
    os.makedirs(trackers_dir, exist_ok=True)
    os.makedirs(out_eval_dir, exist_ok=True)

    # subset videos if requested
    videos_opt = []
    val_root = os.path.join(args.data_root, 'val', 'frames')
    if args.hota_max_videos and args.hota_max_videos > 0 and os.path.isdir(val_root):
        vids = sorted([d for d in os.listdir(val_root) if os.path.isdir(os.path.join(val_root, d))])
        videos_opt = vids[: args.hota_max_videos]

    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    py = sys.executable
    infer_mod = 'oss25.infer_kpnet_with_cotracker' if args.hota_mode == 'cot' else 'oss25.infer_kpnet_cotracker'
    infer_cmd = [
        py, '-m', infer_mod,
        '--data-root', args.data_root,
        '--ckpt', ckpt,
        '--out-dir', trackers_dir,
        '--device', args.device,
        '--image-size', str(args.image_size),
        '--heatmap-size', str(args.heatmap_size),
    ]
    if getattr(args, 'resize_only', False):
        infer_cmd += ['--resize-only']
    if videos_opt:
        infer_cmd += ['--videos', *videos_opt]
    print('[hota] running inference:', ' '.join(infer_cmd))
    subprocess.run(infer_cmd, check=True, env=env)

    # TrackEval
    trackeval_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'TrackEval', 'scripts', 'run_mot_challenge_kp.py'))
    te_cmd = [
        py, trackeval_script,
        '--GT_FOLDER', args.hota_gt_folder,
        '--TRACKERS_FOLDER', trackers_dir,
        '--OUTPUT_FOLDER', out_eval_dir,
        '--TRACKERS_TO_EVAL', 'oss25_det',
    ]
    print('[hota] running TrackEval:', ' '.join(te_cmd))
    subprocess.run(te_cmd, check=True, env=env)

    # Visualization (optional)
    if args.val_vis_video:
        vis_out = os.path.join(args.out_dir, 'hota', f'epoch_{epoch}', 'vis', args.val_vis_video)
        os.makedirs(vis_out, exist_ok=True)
        pred_path = os.path.join(trackers_dir, f'{args.val_vis_video}_pred.txt')
        if os.path.exists(pred_path):
            vis_cmd = [
                py, '-m', 'oss25.visualize_val_kp',
                '--data-root', args.data_root,
                '--gt-folder', args.hota_gt_folder,
                '--pred', pred_path,
                '--video', args.val_vis_video,
                '--out', vis_out,
                '--limit', '12',
            ]
            print('[hota] running visualize:', ' '.join(vis_cmd))
            subprocess.run(vis_cmd, check=True, env=env)
        else:
            print(f"[hota] skip visualize: pred not found for video {args.val_vis_video}")


@torch.no_grad()
def _dump_val_seg_predictions(args, model, device, epoch: int):
    model.eval()
    from PIL import Image, ImageDraw
    import numpy as np
    # small loader (batch=1, no workers) to avoid interfering main loaders
    from torch.utils.data import DataLoader
    val_ds = Task3KeypointDataset(args.data_root, split='val', image_size=(args.image_h, args.image_w), heatmap_size=(args.heatmap_h, args.heatmap_w),
                                  aug_rotate_deg=0.0, pseudo6_root=args.pseudo6_root, use_flow=(args.in_chans>3), allow_seg_only=False, resize_only=args.resize_only)
    small_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    out_root = os.path.join(args.out_dir, 'seg_snap', f'epoch_{epoch}')
    os.makedirs(out_root, exist_ok=True)
    # normalization as training
    data_cfg = timm.data.resolve_model_data_config(model.encoder)
    mean = torch.tensor(data_cfg.get('mean', (0.5,0.5,0.5)), dtype=torch.float32, device=device).view(1,3,1,1)
    std = torch.tensor(data_cfg.get('std', (0.5,0.5,0.5)), dtype=torch.float32, device=device).view(1,3,1,1)

    dumped = 0
    for batch in small_loader:
        # filter by video id if requested
        name = batch['meta']['frame_name'][0]
        video_id = name.split('_frame_')[0]
        if args.dump_seg_video and video_id != args.dump_seg_video:
            continue
        # build input
        img = (batch['image'].to(device) - mean) / std
        flow = batch.get('flow_uv')
        fg = batch.get('fg_mask')
        if flow is not None:
            flow = flow.to(device)
        if fg is not None:
            fg = fg.to(device)
        x = img
        if args.in_chans > 3:
            flow_n = torch.clamp(flow / args.flow_norm, -1.0, 1.0) if flow is not None else torch.zeros(img.shape[0], 2, img.shape[-2], img.shape[-1], device=device)
            fg_n = torch.clamp(fg * 2.0 - 1.0, -1.0, 1.0) if fg is not None else torch.zeros(img.shape[0], 1, img.shape[-2], img.shape[-1], device=device)
            x = torch.cat([img, flow_n, fg_n], dim=1)
        # forward
        hm_p, vis_p, seg_p = model(x, return_seg=True)
        seg_l = seg_p.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)  # Hh x Wh
        # do not save raw label map; overlay only
        # save overlay
        # rebuild base RGB for overlay (ensure all operands on CPU)
        mean_c = mean[0].detach().cpu()
        std_c = std[0].detach().cpu()
        rgb = img[0, :3].detach().cpu()
        base = (rgb * std_c + mean_c).clamp(0, 1).numpy()
        base = (base.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        base_im = Image.fromarray(base).convert('RGBA')
        # color palette for predicted seg
        palette = [
            (0, 0, 0, 0),
            (0, 128, 255, 80),
            (0, 200, 200, 80),
            (255, 0, 255, 80),
            (255, 0, 128, 80),
            (128, 0, 255, 80),
            (255, 0, 64, 80),
        ]
        h, w = seg_l.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        alpha_scale = max(0.0, min(1.0, float(getattr(args, 'dump_seg_alpha', 1.0))))
        for lb in range(1, 7):
            mask = (seg_l == lb)
            if not mask.any():
                continue
            r, g, b, a = palette[lb]
            a = int(max(0, min(255, round(a * alpha_scale))))
            rgba[mask] = (r, g, b, a)
        # clip overlay to the valid (non-pad) region using meta (scale & pad) only when letterbox is used
        if bool(_meta_get(batch['meta'], 'letterbox', 0)):
            pl, pt = _meta_get(batch['meta'], 'pad', 0)
            s = float(_meta_get(batch['meta'], 'scale', 0))
            H0, W0 = _meta_get(batch['meta'], 'orig_size', 0)
            nw, nh = int(round(W0 * s)), int(round(H0 * s))
            # zero alpha outside [pt:pt+nh, pl:pl+nw]
            alpha = rgba[..., 3]
            if pt > 0:
                alpha[:pt, :] = 0
            if pl > 0:
                alpha[:, :pl] = 0
            if pt + nh < alpha.shape[0]:
                alpha[pt + nh :, :] = 0
            if pl + nw < alpha.shape[1]:
                alpha[:, pl + nw :] = 0
            rgba[..., 3] = alpha
        # resize seg overlay to match base image canvas size
        overlay = Image.fromarray(rgba).resize(base_im.size, Image.NEAREST)
        base_im.alpha_composite(overlay)
        # draw keypoints (GT: red, Pred: cyan) and correspondence lines
        draw = ImageDraw.Draw(base_im, 'RGBA')
        # predicted keypoints from heatmap (soft-argmax)
        coords_hm = softargmax_coords(hm_p, temperature=1.0)  # (B,C,2)
        W_img, H_img = base_im.size
        sx = float(W_img) / float(args.heatmap_w)
        sy = float(H_img) / float(args.heatmap_h)
        coords_np = coords_hm[0].detach().cpu().numpy()
        coords_img = np.stack([coords_np[..., 0] * sx, coords_np[..., 1] * sy], axis=-1)
        vis_pred = vis_p.softmax(dim=-1).argmax(dim=-1)[0].detach().cpu().numpy()
        for (x, y), v in zip(coords_img, vis_pred):
            r = 3
            col = (0, 255, 255, 220) if int(v) > 0 else (0, 255, 255, 120)
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=col)
        # GT keypoints from coord_targets (heatmap座標)
        if 'coord_targets' in batch:
            ct = batch['coord_targets'][0].detach().cpu().numpy()
            cw = batch['coord_weights'][0].detach().cpu().numpy() if 'coord_weights' in batch else None
            gt_xy = []
            for i, (xh, yh) in enumerate(ct):
                if cw is not None and cw[i] <= 0:
                    continue
                xg = xh * sx
                yg = yh * sy
                r = 3
                draw.ellipse([(xg - r, yg - r), (xg + r, yg + r)], fill=(255, 0, 0, 220))
                gt_xy.append((i, xg, yg))
            # correspondence lines (pred↔GT) for supervised points
            for i, xg, yg in gt_xy:
                xp, yp = coords_img[i]
                draw.line([(xg, yg), (xp, yp)], fill=(255, 255, 0, 160), width=2)
        base_rgb = base_im.convert('RGB')
        base_rgb.save(os.path.join(out_root, f'{name}_overlay.jpg'), quality=90)
        dumped += 1
        if args.dump_seg_limit and args.dump_seg_limit > 0 and dumped >= args.dump_seg_limit:
            break


if __name__ == '__main__':
    main()
