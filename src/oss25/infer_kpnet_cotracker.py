#!/usr/bin/env python3
import os
import argparse
import glob
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from oss25.models.kpnet import KPNet
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
import timm


KP_LAYOUT = {
    0: (0, 6),   # left hand (6)
    1: (6, 6),   # right hand (6)
    2: (12, 3),  # scissors (3)
    3: (15, 3),  # tweezers (3)
    4: (18, 3),  # needle holder (3)
    5: (21, 6),  # needle (two instances x3 -> 6)
}


def refine_coords_sigmoid(heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Peak coordinates via argmax + local 3x3 weighted refinement (sigmoid probs).
    Returns coords (B,C,2) in heatmap space and conf (B,C) in [0,1].
    """
    B, C, H, W = heatmaps.shape
    prob = torch.sigmoid(heatmaps)
    flat = prob.view(B, C, -1)
    idx = flat.argmax(dim=-1)
    xs0 = (idx % W).float()
    ys0 = (idx // W).float()
    conf = flat.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    # local refinement
    xs = xs0.clone()
    ys = ys0.clone()
    for b in range(B):
        for c in range(C):
            x0 = int(xs0[b, c].item())
            y0 = int(ys0[b, c].item())
            x1 = max(0, x0 - 1); x2 = min(W - 1, x0 + 1)
            y1 = max(0, y0 - 1); y2 = min(H - 1, y0 + 1)
            patch = prob[b, c, y1:y2+1, x1:x2+1]
            if patch.numel() == 0:
                continue
            w = patch / (patch.sum() + 1e-6)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(y1, y2+1, device=prob.device, dtype=torch.float32),
                torch.arange(x1, x2+1, device=prob.device, dtype=torch.float32),
                indexing='ij'
            )
            xr = (w * grid_x).sum()
            yr = (w * grid_y).sum()
            xs[b, c] = xr
            ys[b, c] = yr
    coords = torch.stack([xs, ys], dim=-1)
    return coords, conf

def reorder_tool_kps(cls_id: int, kp: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    """Reorder 3-point tool KP to match spec ordering using simple heuristics.
    - scissors/tweezers/needle holder: [left, right, joint|nub]
    - needle block handled upstream as two instances of 3 points each
    """
    if len(kp) != 3:
        return kp
    # left/right by x coordinate
    sorted_by_x = sorted(enumerate(kp), key=lambda t: t[1][0])
    left_idx, left = sorted_by_x[0]
    right_idx, right = sorted_by_x[-1]
    mid_idx = [i for i in range(3) if i not in (left_idx, right_idx)][0]
    mid = kp[mid_idx]  # joint/nub
    return [left, right, mid]


def best_perm_to_prev(prev: List[Tuple[float, float, int]], cur: List[Tuple[float, float, int]]):
    # brute force 3! permutations to minimize L2 sum to prev
    import itertools
    if len(prev) != 3 or len(cur) != 3:
        return cur
    best = None
    best_d = 1e18
    for perm in itertools.permutations(range(3)):
        d = 0.0
        ok = True
        for i, j in enumerate(perm):
            px, py, _ = prev[i]
            cx, cy, c = cur[j]
            if cx < 0 or cy < 0:  # skip invalid
                ok = False
                break
            d += (px - cx) ** 2 + (py - cy) ** 2
        if ok and d < best_d:
            best_d = d
            best = [cur[j] for j in perm]
    return best if best is not None else cur


def load_val_video_frames(data_root: str, video_id: str) -> List[str]:
    frame_dir = os.path.join(data_root, 'val', 'frames', video_id)
    files = sorted(glob.glob(os.path.join(frame_dir, f'{video_id}_frame_*.png')),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_frame_')[1]))
    return files


def run_detector_on_frame(model: KPNet, img: Image.Image, image_size=384, heatmap_size=96, device='cuda', prev_img: Image.Image = None, resize_only: bool = False):
    # prepare input (letterbox or stretch) for arbitrary target size (H,W) or int
    w0, h0 = img.size
    if isinstance(image_size, (tuple, list)):
        ih, iw = int(image_size[0]), int(image_size[1])
    else:
        ih = iw = int(image_size)
    if isinstance(heatmap_size, (tuple, list)):
        hh, hw = int(heatmap_size[0]), int(heatmap_size[1])
    else:
        hh = hw = int(heatmap_size)
    if resize_only:
        sx = iw / float(w0)
        sy = ih / float(h0)
        canvas = img.resize((iw, ih), Image.BILINEAR)
        pl = pt = 0
        scale_meta = (sx, sy)
    else:
        s = min(iw / float(w0), ih / float(h0))
        nw, nh = int(round(w0 * s)), int(round(h0 * s))
        img_resized = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (iw, ih), color=(0, 0, 0))
        pl = (iw - nw) // 2
        pt = (ih - nh) // 2
        canvas.paste(img_resized, (pl, pt))
        scale_meta = s
    curr_np = np.array(canvas).astype(np.float32) / 255.0
    x_img = torch.from_numpy(curr_np).permute(2, 0, 1).unsqueeze(0).to(device)
    # build extras if model expects >3 channels
    extras = []
    if getattr(model, 'in_chans', 3) > 3 and _HAS_CV2:
        if prev_img is None:
            prev_img_l = canvas
        else:
            if resize_only:
                prev_img_l = prev_img.resize((iw, ih), Image.BILINEAR)
            else:
                pw, ph = prev_img.size
                s2 = min(iw / float(pw), ih / float(ph))
                nw2, nh2 = int(round(pw * s2)), int(round(ph * s2))
                prev_resized = prev_img.resize((nw2, nh2), Image.BILINEAR)
                prev_canvas = Image.new('RGB', (iw, ih), color=(0, 0, 0))
                pl2 = (iw - nw2) // 2
                pt2 = (ih - nh2) // 2
                prev_canvas.paste(prev_resized, (pl2, pt2))
                prev_img_l = prev_canvas
        prev_np = np.array(prev_img_l).astype(np.uint8)
        g0 = cv2.cvtColor(prev_np, cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor((curr_np*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_uv = torch.from_numpy(flow.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
        flow_uv = torch.clamp(flow_uv / 20.0, -1.0, 1.0)
        diff = cv2.absdiff(g1, g0).astype(np.float32) / 255.0
        fg = torch.from_numpy(diff).unsqueeze(0).unsqueeze(0).float().to(device)
        fg = torch.clamp(fg * 2.0 - 1.0, -1.0, 1.0)
        extras = [flow_uv, fg]
    # normalize only RGB channels, then concat extras
    data_cfg = timm.data.resolve_model_data_config(model.encoder)
    mean = torch.tensor(data_cfg.get('mean', (0.5,0.5,0.5)), dtype=torch.float32, device=device).view(1,3,1,1)
    std = torch.tensor(data_cfg.get('std', (0.5,0.5,0.5)), dtype=torch.float32, device=device).view(1,3,1,1)
    x_rgb = (x_img - mean) / std
    x = x_rgb
    if extras:
        x = torch.cat([x_rgb, *extras], dim=1)
    # pad zeros if model expects more channels (e.g., in_chans=6 but CV2 not available)
    need = getattr(model, 'in_chans', x.shape[1]) - x.shape[1]
    if need > 0:
        B, _, H, W = x.shape
        zeros = torch.zeros(B, need, H, W, device=x.device, dtype=x.dtype)
        x = torch.cat([x, zeros], dim=1)
    with torch.no_grad():
        hm, vis_logits = model(x)
    coords_hm, conf = refine_coords_sigmoid(hm)
    # scale coords from heatmap to image canvas
    sx_hm = iw / float(hw)
    sy_hm = ih / float(hh)
    coords_img = coords_hm.clone()
    coords_img[..., 0] = coords_img[..., 0] * sx_hm
    coords_img[..., 1] = coords_img[..., 1] * sy_hm
    # map to original image space
    if resize_only:
        sx, sy = scale_meta
        coords_img[..., 0] = coords_img[..., 0] / sx
        coords_img[..., 1] = coords_img[..., 1] / sy
    else:
        s = scale_meta
        coords_img[..., 0] = (coords_img[..., 0] - pl) / s
        coords_img[..., 1] = (coords_img[..., 1] - pt) / s
    vis = vis_logits.softmax(dim=-1).argmax(dim=-1)  # (B,C)
    return coords_img[0].cpu().numpy(), conf[0].cpu().numpy(), vis[0].cpu().numpy()


def write_tracker_file(out_path: str, all_rows: List[str]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for r in all_rows:
            f.write(r + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')))
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'kp_eval', 'data', 'trackers')))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--image-size', type=str, default='384')
    ap.add_argument('--heatmap-size', type=str, default='128')
    ap.add_argument('--videos', nargs='*', help='subset of video ids (val)')
    ap.add_argument('--resize-only', action='store_true', help='resize to square without letterbox (no padding)')
    ap.add_argument('--in-chans', type=int, default=3, help='model input channels (3 or 6)')
    ap.add_argument('--swap-hand-mid-idx', action='store_true', help='Swap hand KP order: [thumb, index, middle, ring, pinky, back] from current [thumb, middle, index, ring, pinky, back]')
    ap.add_argument('--reorder-tools', action='store_true', help='Reorder 3-pt tools by x to [left,right,mid] (default: off)')
    ap.add_argument('--stabilize-tool-order', action='store_true', help='Stabilize 3-pt tools order over time by nearest-neighbor to previous frame (default: off)')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    sd = torch.load(args.ckpt, map_location='cpu')
    weights = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
    # auto-detect input channels from checkpoint if possible
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

    # parse sizes and force heatmap == image
    img_h, img_w = _parse_hw(args.image_size)
    hm_h, hm_w = _parse_hw(args.heatmap_size)
    hm_h, hm_w = img_h, img_w
    args.image_h, args.image_w = img_h, img_w
    args.heatmap_h, args.heatmap_w = hm_h, hm_w
    args.heatmap_size = args.image_size
    def _guess_in_chans(w):
        for k in ['encoder.stem_0.weight', 'encoder.stem.conv.weight']:
            if k in w and w[k].dim() == 4:
                return int(w[k].shape[1])
        for k, v in w.items():
            if k.endswith('stem_0.weight') or k.endswith('stem.conv.weight'):
                if v.dim() == 4:
                    return int(v.shape[1])
        return None
    ckpt_in = _guess_in_chans(weights) or args.in_chans
    if ckpt_in != args.in_chans:
        print(f"[info] overriding in_chans: ckpt={ckpt_in} arg={args.in_chans}")
    model = KPNet(image_size=(args.image_h, args.image_w), heatmap_size=(args.heatmap_h, args.heatmap_w), in_chans=ckpt_in)
    res = model.load_state_dict(weights, strict=False)
    missing = getattr(res, 'missing_keys', [])
    unexpected = getattr(res, 'unexpected_keys', [])
    if missing or unexpected:
        print(f"[warn] state_dict mismatch: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print('  missing (first 10):', missing[:10])
        if unexpected:
            print('  unexpected (first 10):', unexpected[:10])
    model = model.to(device)
    model.eval()

    # Optional: try to import CoTracker3 for later extension (not used in minimal baseline)
    try:
        _ = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        has_cotracker = True
    except Exception:
        has_cotracker = False
        print('[warn] CoTracker not available; running detector-only baseline')

    # videos from val frames
    frame_root = os.path.join(args.data_root, 'val', 'frames')
    video_ids = args.videos if args.videos else sorted(os.listdir(frame_root))
    for vid in video_ids:
        frame_files = load_val_video_frames(args.data_root, vid)
        if not frame_files:
            print(f'[skip] no frames for {vid}')
            continue
        rows: List[str] = []
        prev_tool: Dict[Tuple[int,int], List[Tuple[float,float,int]]] = {}
        prev_img = None
        for fpath in frame_files:
            frame_idx = int(os.path.splitext(os.path.basename(fpath))[0].split('_frame_')[1])
            img = Image.open(fpath).convert('RGB')
            coords, conf, vis = run_detector_on_frame(model, img, image_size=(args.image_h, args.image_w), heatmap_size=(args.heatmap_h, args.heatmap_w), device=device, prev_img=prev_img, resize_only=args.resize_only)
            # Compose rows per object
            # Track IDs: 0=left hand,1=right hand,2=scissors,3=tweezers,4=needle holder,5=needle0,6=needle1
            def row(frame, tid, cls, kp_coords):
                # bbox xywh unused
                bbox = '-1,-1,-1,-1'
                # kp: x,y,c per point
                kp_str = ','.join([f'{x:.1f},{y:.1f},{int(c)}' for (x,y,c) in kp_coords])
                return f'{frame},{tid},{cls},{bbox},{kp_str}'
            # hands
            for cls, tid in [(0,0),(1,1)]:
                base, n = KP_LAYOUT[cls]
                kp = []
                for i in range(n):
                    x, y = coords[base+i]
                    c = int(vis[base+i])
                    kp.append((x, y, c))
                if args.swap_hand_mid_idx and len(kp) >= 3:
                    # swap index(2) and middle(1): from [thumb, middle, index, ring, pinky, back] -> [thumb, index, middle, ring, pinky, back]
                    kp[1], kp[2] = kp[2], kp[1]
                rows.append(row(frame_idx, tid, cls, kp))
            # single-instance tools
            for cls, tid in [(2,2),(3,3),(4,4)]:
                base, n = KP_LAYOUT[cls]
                kp = []
                for i in range(n):
                    x, y = coords[base+i]
                    c = int(vis[base+i])
                    kp.append((x, y, c))
                if args.stabilize_tool_order and (cls,0) in prev_tool:
                    kp = best_perm_to_prev(prev_tool.get((cls,0), []), kp)
                elif args.reorder_tools:
                    kp = reorder_tool_kps(cls, kp)
                rows.append(row(frame_idx, tid, cls, kp))
                prev_tool[(cls,0)] = kp
            # needles (two instances)
            base, n_total = KP_LAYOUT[5]
            n = 3
            for inst, tid in enumerate([5,6]):
                kp = []
                for i in range(n):
                    x, y = coords[base + inst*n + i]
                    c = int(vis[base + inst*n + i])
                    kp.append((x, y, c))
                key = (5, inst)
                if args.stabilize_tool_order and key in prev_tool:
                    kp = best_perm_to_prev(prev_tool.get(key, []), kp)
                elif args.reorder_tools:
                    kp = reorder_tool_kps(5, kp)
                rows.append(row(frame_idx, tid, 5, kp))
                prev_tool[key] = kp
            prev_img = img
        out_path = os.path.join(args.out_dir, f'{vid}_pred.txt')
        write_tracker_file(out_path, rows)
        print(f'[pred] wrote {out_path}')


if __name__ == '__main__':
    main()
