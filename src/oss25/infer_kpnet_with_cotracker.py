#!/usr/bin/env python3
import os
import glob
import argparse
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from oss25.models.kpnet import KPNet
import timm


# Channel layout: class -> (base_index, count)
KP_LAYOUT = {
    0: (0, 6),   # left hand
    1: (6, 6),   # right hand
    2: (12, 3),  # scissors
    3: (15, 3),  # tweezers
    4: (18, 3),  # needle holder
    5: (21, 6),  # needle (two instances x3)
}


def argmax_2d_sigmoid(heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = heatmaps.shape
    prob = torch.sigmoid(heatmaps)
    flat = prob.view(B, C, -1)
    idx = flat.argmax(dim=-1)
    xs0 = (idx % W).float()
    ys0 = (idx // W).float()
    conf = flat.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    xs = xs0.clone(); ys = ys0.clone()
    for b in range(B):
        for c in range(C):
            x0 = int(xs0[b, c].item()); y0 = int(ys0[b, c].item())
            x1 = max(0, x0-1); x2 = min(W-1, x0+1)
            y1 = max(0, y0-1); y2 = min(H-1, y0+1)
            patch = prob[b, c, y1:y2+1, x1:x2+1]
            if patch.numel() == 0:
                continue
            w = patch / (patch.sum() + 1e-6)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(y1, y2+1, device=prob.device, dtype=torch.float32),
                torch.arange(x1, x2+1, device=prob.device, dtype=torch.float32),
                indexing='ij')
            xr = (w * grid_x).sum(); yr = (w * grid_y).sum()
            xs[b, c] = xr; ys[b, c] = yr
    coords = torch.stack([xs, ys], dim=-1)
    return coords, conf

def reorder_tool_kps(cls_id: int, kp: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    if len(kp) != 3:
        return kp
    sorted_by_x = sorted(enumerate(kp), key=lambda t: t[1][0])
    left_idx, left = sorted_by_x[0]
    right_idx, right = sorted_by_x[-1]
    mid_idx = [i for i in range(3) if i not in (left_idx, right_idx)][0]
    mid = kp[mid_idx]
    return [left, right, mid]


def load_val_frames_list(data_root: str, vid: str) -> List[str]:
    frame_dir = os.path.join(data_root, 'val', 'frames', vid)
    files = sorted(
        glob.glob(os.path.join(frame_dir, f'{vid}_frame_*.png')),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_frame_')[1])
    )
    return files


def letterbox(img: Image.Image, image_size: int) -> Tuple[torch.Tensor, float, int, int]:
    w0, h0 = img.size
    if isinstance(image_size, (tuple, list)):
        ih, iw = int(image_size[0]), int(image_size[1])
    else:
        ih = iw = int(image_size)
    s = min(iw / float(w0), ih / float(h0))
    nw, nh = int(round(w0 * s)), int(round(h0 * s))
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (iw, ih), color=(0, 0, 0))
    pl = (iw - nw) // 2
    pt = (ih - nh) // 2
    canvas.paste(img_resized, (pl, pt))
    x = torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0).permute(2, 0, 1)
    return x, s, pl, pt


def detector_infer_frame(model: KPNet, img: Image.Image, image_size, heatmap_size, device: torch.device, prev_img: Image.Image = None, resize_only: bool = False):
    x, s, pl, pt = letterbox(img, image_size)
    x = x.unsqueeze(0).to(device)
    extras = []
    if getattr(model, 'in_chans', 3) > 3:
        try:
            import cv2
            x_prev, _, _, _ = letterbox(prev_img if prev_img is not None else img, image_size)
            prev_np = (x_prev.numpy().transpose(1,2,0) * 255.0).astype('uint8')
            curr_np = (x[0].cpu().numpy().transpose(1,2,0) * 255.0).astype('uint8')
            g0 = cv2.cvtColor(prev_np, cv2.COLOR_RGB2GRAY)
            g1 = cv2.cvtColor(curr_np, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_uv = torch.from_numpy(flow.astype('float32')).permute(2,0,1).unsqueeze(0).to(device)
            flow_uv = torch.clamp(flow_uv/20.0, -1.0, 1.0)
            diff = cv2.absdiff(g1, g0).astype('float32')/255.0
            fg = torch.from_numpy(diff).unsqueeze(0).unsqueeze(0).float().to(device)
            fg = torch.clamp(fg*2.0-1.0, -1.0, 1.0)
            extras = [flow_uv, fg]
        except Exception:
            pass
    # normalize as in training
    data_cfg = timm.data.resolve_model_data_config(model.encoder)
    mean = torch.tensor(data_cfg.get('mean', (0.5,0.5,0.5)), dtype=torch.float32, device=device).view(1,3,1,1)
    std = torch.tensor(data_cfg.get('std', (0.5,0.5,0.5)), dtype=torch.float32, device=device).view(1,3,1,1)
    x_rgb = (x - mean) / std
    x = x_rgb
    if extras:
        x = torch.cat([x_rgb, *extras], dim=1)
    need = getattr(model, 'in_chans', x.shape[1]) - x.shape[1]
    if need > 0:
        B, _, H, W = x.shape
        zeros = torch.zeros(B, need, H, W, device=x.device, dtype=x.dtype)
        x = torch.cat([x, zeros], dim=1)
    with torch.no_grad():
        hm, vis_logits = model(x)
    coords_hm, conf = argmax_2d_sigmoid(hm)
    # scale heatmap coords to image canvas size (per-axis)
    if isinstance(image_size, (tuple, list)):
        ih, iw = int(image_size[0]), int(image_size[1])
    else:
        ih = iw = int(image_size)
    if isinstance(heatmap_size, (tuple, list)):
        hh, hw = int(heatmap_size[0]), int(heatmap_size[1])
    else:
        hh = hw = int(heatmap_size)
    sx_hm = iw / float(hw)
    sy_hm = ih / float(hh)
    coords_img = coords_hm.clone()
    coords_img[..., 0] = coords_img[..., 0] * sx_hm
    coords_img[..., 1] = coords_img[..., 1] * sy_hm
    # map back to original image coordinates
    coords_img[..., 0] = (coords_img[..., 0] - pl) / s
    coords_img[..., 1] = (coords_img[..., 1] - pt) / s
    vis = vis_logits.softmax(dim=-1).argmax(dim=-1)  # (B,C)
    return coords_img[0].cpu().numpy(), conf[0].cpu().numpy(), vis[0].cpu().numpy()


def build_video_tensor(frames: List[str], image_size, device: torch.device, resize_only: bool = False) -> Tuple[torch.Tensor, List[Tuple[float,int,int]]]:
    vs = []
    meta = []  # (scale, pl, pt)
    for fp in frames:
        img = Image.open(fp).convert('RGB')
        x, s, pl, pt = letterbox(img, image_size)
        vs.append(x)
        meta.append((s, pl, pt))
    vid = torch.stack(vs, dim=0)  # T,3,H,W
    vid = vid.unsqueeze(0).to(device)  # 1,T,3,H,W
    return vid, meta


def assign_tracks_to_kps(grid_tracks: np.ndarray, kp_coords: np.ndarray, used: set) -> List[int]:
    # grid_tracks: (N,2) points at current frame; kp_coords: (C,2)
    N = grid_tracks.shape[0]
    C = kp_coords.shape[0]
    assigned = [-1] * C
    # greedy nearest with uniqueness
    dists = np.linalg.norm(grid_tracks[None, :, :] - kp_coords[:, None, :], axis=-1)  # CxN
    pairs = [(i, j, dists[i, j]) for i in range(C) for j in range(N) if j not in used]
    pairs.sort(key=lambda x: x[2])
    taken_k = set()
    for i, j, d in pairs:
        if i in taken_k or j in used:
            continue
        assigned[i] = j
        used.add(j)
        taken_k.add(i)
    return assigned


def rows_from_tracks(frame_idx: int, kp_coords: np.ndarray, kp_vis: np.ndarray) -> List[str]:
    # compose MOT rows for this frame
    rows = []
    def row(frame, tid, cls, kp):
        bbox = '-1,-1,-1,-1'
        kp_str = ','.join([f'{x:.1f},{y:.1f},{int(c)}' for (x,y,c) in kp])
        return f'{frame},{tid},{cls},{bbox},{kp_str}'
    # hands
    for cls, tid in [(0,0),(1,1)]:
        base, n = KP_LAYOUT[cls]
        kp = [(kp_coords[base+i,0], kp_coords[base+i,1], kp_vis[base+i]) for i in range(n)]
        rows.append(row(frame_idx, tid, cls, kp))
    # tools single inst
    for cls, tid in [(2,2),(3,3),(4,4)]:
        base, n = KP_LAYOUT[cls]
        kp = [(kp_coords[base+i,0], kp_coords[base+i,1], kp_vis[base+i]) for i in range(n)]
        rows.append(row(frame_idx, tid, cls, kp))
    # needles two inst
    base, n_total = KP_LAYOUT[5]
    n = 3
    for inst, tid in enumerate([5,6]):
        kp = [(kp_coords[base+inst*n+i,0], kp_coords[base+inst*n+i,1], kp_vis[base+inst*n+i]) for i in range(n)]
        rows.append(row(frame_idx, tid, 5, kp))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')))
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'kp_eval', 'data', 'trackers')))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--image-size', type=str, default='384')
    ap.add_argument('--heatmap-size', type=str, default='128')
    ap.add_argument('--videos', nargs='*', help='subset of val video ids')
    ap.add_argument('--resize-only', action='store_true', help='resize to square without letterbox (no padding)')
    ap.add_argument('--in-chans', type=int, default=3, help='model input channels (3 or 6)')
    ap.add_argument('--k', type=int, default=20, help='re-detection interval (frames)')
    ap.add_argument('--grid-size', type=int, default=20)
    ap.add_argument('--vis-th', type=float, default=0.5)
    ap.add_argument('--swap-hand-mid-idx', action='store_true',
                    help='Swap hand KP order: [thumb, index, middle, ring, pinky, back] from current [thumb, middle, index, ring, pinky, back]')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    sd = torch.load(args.ckpt, map_location='cpu')
    weights = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
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

    # try loading CoTracker3
    cot = None
    try:
        cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
        print('[info] CoTracker3 loaded')
    except Exception as e:
        print(f'[warn] CoTracker3 not available ({e}); fallback to detector-only')

    val_root = os.path.join(args.data_root, 'val', 'frames')
    video_ids = args.videos if args.videos else sorted(os.listdir(val_root))
    os.makedirs(args.out_dir, exist_ok=True)
    for vid in video_ids:
        frame_files = load_val_frames_list(args.data_root, vid)
        if not frame_files:
            print(f'[skip] no frames for {vid}')
            continue
        T = len(frame_files)

        if cot is None:
            # Detector-only baseline
            rows_all = []
            prev_img = None
            for f in frame_files:
                frame_idx = int(os.path.splitext(os.path.basename(f))[0].split('_frame_')[1])
                img = Image.open(f).convert('RGB')
                coords, conf, vis = detector_infer_frame(model, img, (args.image_h, args.image_w), (args.heatmap_h, args.heatmap_w), device, prev_img=prev_img, resize_only=args.resize_only)
                rows_all.extend(rows_from_tracks(frame_idx, coords, vis))
                prev_img = img
            out_path = os.path.join(args.out_dir, f'{vid}_pred.txt')
            with open(out_path, 'w') as fw:
                for r in rows_all:
                    fw.write(r + '\n')
            print(f'[pred] {vid}: detector-only -> {out_path}')
            continue

        # Build video tensor for CoTracker3
        video, meta = build_video_tensor(frame_files, (args.image_h, args.image_w), device, resize_only=args.resize_only)
        # Initialize CoTracker with grid
        cot(video_chunk=video, is_first_step=True, grid_size=args.grid_size)
        step = getattr(cot, 'step', 8)
        # collect tracks over time using overlapping windows
        # We'll accumulate predictions for the first `step` frames of each 2*step window
        tracks_T = None
        vis_T = None
        for ind in range(0, video.shape[1] - step, step):
            chunk = video[:, ind : min(ind + step * 2, video.shape[1])]
            pred_tracks, pred_visibility = cot(video_chunk=chunk)
            # pred_tracks: B, Tc, N, 2
            # pred_visibility: B, Tc, N (or B, Tc, N, 1) depending on version
            Tc = pred_tracks.shape[1]
            take = min(step, Tc)
            pts = pred_tracks[:, :take]  # 1,take,N,2
            if pred_visibility.dim() == 4:
                visb = pred_visibility[:, :take, :, 0]
            else:
                visb = pred_visibility[:, :take, :]
            if tracks_T is None:
                N = pts.shape[2]
                tracks_T = torch.zeros(video.shape[1], N, 2, device=pts.device)
                vis_T = torch.zeros(video.shape[1], N, device=pts.device)
            tracks_T[ind:ind+take] = pts[0]
            vis_T[ind:ind+take] = visb[0]
        # convert to numpy
        tracks_np = tracks_T.detach().cpu().numpy()  # T,N,2 at image_size coords
        vis_np = vis_T.detach().cpu().numpy()  # T,N in [0,1]

        # Assign grid tracks to KPs at re-detection frames
        rows_all = []
        kp2grid: Dict[int, int] = {}
        prev_img = None
        for t_idx, f in enumerate(frame_files):
            frame_no = int(os.path.splitext(os.path.basename(f))[0].split('_frame_')[1])
            # re-detect every K frames or first frame
            if (t_idx % args.k) == 0:
                img = Image.open(f).convert('RGB')
                coords, conf, vis = detector_infer_frame(model, img, (args.image_h, args.image_w), (args.heatmap_h, args.heatmap_w), device, prev_img=prev_img)
                grid_pts = tracks_np[t_idx]  # N,2 (in image_size canvas coords)
                # map grid coords back to original image space using meta
                s, pl, pt = meta[t_idx]
                grid_pts_img = np.zeros_like(grid_pts)
                if isinstance(s, tuple):
                    sx, sy = s
                    grid_pts_img[:,0] = grid_pts[:,0] / sx
                    grid_pts_img[:,1] = grid_pts[:,1] / sy
                else:
                    grid_pts_img[:,0] = (grid_pts[:,0] - pl) / s
                    grid_pts_img[:,1] = (grid_pts[:,1] - pt) / s
                used = set()
                kp2grid = {}
                # assign per-block to encourage diversity
                order_blocks = [0,1,2,3,4,5]
                for cls in order_blocks:
                    base, cnt = KP_LAYOUT[cls]
                    sel = assign_tracks_to_kps(grid_pts_img, coords[base:base+cnt], used)
                    for i, j in enumerate(sel):
                        if j >= 0:
                            kp2grid[base+i] = j
            # compose output rows for current frame using current assignment
            kp_coords_frame = np.zeros((27,2), dtype=np.float32)
            kp_vis_frame = np.zeros((27,), dtype=np.int64)
            # default vis from CoTracker
            visb = vis_np[t_idx]
            for c in range(27):
                j = kp2grid.get(c, None)
                if j is not None:
                    # map co-tracker coords back to original
                    s, pl, pt = meta[t_idx]
                    if isinstance(s, tuple):
                        sx, sy = s
                        x = tracks_np[t_idx, j, 0] / sx
                        y = tracks_np[t_idx, j, 1] / sy
                    else:
                        x = (tracks_np[t_idx, j, 0] - pl) / s
                        y = (tracks_np[t_idx, j, 1] - pt) / s
                    kp_coords_frame[c] = (x, y)
                    kp_vis_frame[c] = 2 if visb[j] >= args.vis_th else 1
                else:
                    kp_coords_frame[c] = (0.0, 0.0)
                    kp_vis_frame[c] = 0
            # optional swap of hand mid/index
            if hasattr(args, 'swap_hand_mid_idx') and args.swap_hand_mid_idx:
                # left hand block
                lb, ln = KP_LAYOUT[0]
                if ln >= 3:
                    kp_coords_frame[lb+1], kp_coords_frame[lb+2] = kp_coords_frame[lb+2].copy(), kp_coords_frame[lb+1].copy()
                    kp_vis_frame[lb+1], kp_vis_frame[lb+2] = kp_vis_frame[lb+2], kp_vis_frame[lb+1]
                # right hand block
                rb, rn = KP_LAYOUT[1]
                if rn >= 3:
                    kp_coords_frame[rb+1], kp_coords_frame[rb+2] = kp_coords_frame[rb+2].copy(), kp_coords_frame[rb+1].copy()
                    kp_vis_frame[rb+1], kp_vis_frame[rb+2] = kp_vis_frame[rb+2], kp_vis_frame[rb+1]

            # optional swap of hand mid/index
            if hasattr(args, 'swap_hand_mid_idx') and args.swap_hand_mid_idx:
                lb, ln = KP_LAYOUT[0]
                if ln >= 3:
                    kp_coords_frame[lb+1], kp_coords_frame[lb+2] = kp_coords_frame[lb+2].copy(), kp_coords_frame[lb+1].copy()
                    kp_vis_frame[lb+1], kp_vis_frame[lb+2] = kp_vis_frame[lb+2], kp_vis_frame[lb+1]
                rb, rn = KP_LAYOUT[1]
                if rn >= 3:
                    kp_coords_frame[rb+1], kp_coords_frame[rb+2] = kp_coords_frame[rb+2].copy(), kp_coords_frame[rb+1].copy()
                    kp_vis_frame[rb+1], kp_vis_frame[rb+2] = kp_vis_frame[rb+2], kp_vis_frame[rb+1]

            # reorder tools blocks
            for cls in [2,3,4]:
                base, n = KP_LAYOUT[cls]
                if n == 3:
                    a = kp_coords_frame[base:base+3].tolist()
                    v = kp_vis_frame[base:base+3].tolist()
                    kp3 = [(a[i][0], a[i][1], int(v[i])) for i in range(3)]
                    kp3r = reorder_tool_kps(cls, kp3)
                    for i in range(3):
                        kp_coords_frame[base+i,0] = kp3r[i][0]
                        kp_coords_frame[base+i,1] = kp3r[i][1]
                        kp_vis_frame[base+i] = kp3r[i][2]
            # reorder needle instances
            base, n_total = KP_LAYOUT[5]
            for inst in range(2):
                a = kp_coords_frame[base+inst*3:base+inst*3+3].tolist()
                v = kp_vis_frame[base+inst*3:base+inst*3+3].tolist()
                kp3 = [(a[i][0], a[i][1], int(v[i])) for i in range(3)]
                kp3r = reorder_tool_kps(5, kp3)
                for i in range(3):
                    kp_coords_frame[base+inst*3+i,0] = kp3r[i][0]
                    kp_coords_frame[base+inst*3+i,1] = kp3r[i][1]
                    kp_vis_frame[base+inst*3+i] = kp3r[i][2]

            rows_all.extend(rows_from_tracks(frame_no, kp_coords_frame, kp_vis_frame))
            prev_img = Image.open(f).convert('RGB')
        out_path = os.path.join(args.out_dir, f'{vid}_pred.txt')
        with open(out_path, 'w') as fw:
            for r in rows_all:
                fw.write(r + '\n')
        print(f'[pred] {vid}: detector+CoTracker -> {out_path}')


if __name__ == '__main__':
    main()
