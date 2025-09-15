#!/usr/bin/env python3
import os
import argparse
import glob
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import timm

try:
    import cv2
except Exception as e:
    cv2 = None

from oss25.models.kpnet import KPNet


KP_LAYOUT = {
    0: (0, 6),   # left hand (6)
    1: (6, 6),   # right hand (6)
    2: (12, 3),  # scissors (3)
    3: (15, 3),  # tweezers (3)
    4: (18, 3),  # needle holder (3)
    5: (21, 6),  # needle (two instances x3 -> 6)
}


def letterbox(img: Image.Image, image_size: int):
    w, h = img.size
    if isinstance(image_size, (tuple, list)):
        ih, iw = int(image_size[0]), int(image_size[1])
    else:
        ih = iw = int(image_size)
    s = min(iw / w, ih / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (iw, ih), color=(0, 0, 0))
    pl = (iw - nw) // 2
    pt = (ih - nh) // 2
    canvas.paste(img_resized, (pl, pt))
    x = torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0).permute(2, 0, 1)
    return x, s, pl, pt, canvas


@torch.no_grad()
def detect_frame(model: KPNet, img: Image.Image, image_size, heatmap_size, device: torch.device):
    x, s, pl, pt, canvas = letterbox(img, image_size)
    x = x.unsqueeze(0).to(device)
    data_cfg = timm.data.resolve_model_data_config(model.encoder)
    mean = torch.tensor(data_cfg.get('mean', (0.5,0.5,0.5)), dtype=torch.float32, device=device).view(1,3,1,1)
    std = torch.tensor(data_cfg.get('std', (0.5,0.5,0.5)), dtype=torch.float32, device=device).view(1,3,1,1)
    x_rgb = (x - mean) / std
    # pad zeros for extra channels if model expects >3
    need = getattr(model, 'in_chans', 3) - x_rgb.shape[1]
    if need > 0:
        B, _, H, W = x_rgb.shape
        zeros = torch.zeros(B, need, H, W, device=device, dtype=x_rgb.dtype)
        x_in = torch.cat([x_rgb, zeros], dim=1)
    else:
        x_in = x_rgb
    hm, vis_logits = model(x_in)
    prob = torch.sigmoid(hm)
    B, C, H, W = prob.shape
    flat = prob.view(B, C, -1)
    idx = flat.argmax(dim=-1)
    xs0 = (idx % W).float()
    ys0 = (idx // W).float()
    coords_hm = torch.stack([xs0, ys0], dim=-1)
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
    coords_img[..., 0] = (coords_img[..., 0] - pl) / s
    coords_img[..., 1] = (coords_img[..., 1] - pt) / s
    vis = vis_logits.softmax(dim=-1).argmax(dim=-1)
    return coords_img[0].cpu().numpy(), vis[0].cpu().numpy(), canvas


def rows_from_coords(frame_idx: int, coords: np.ndarray, vis: np.ndarray) -> List[str]:
    rows = []
    def row(frame, tid, cls, kp):
        bbox = '-1,-1,-1,-1'
        kp_str = ','.join([f'{x:.1f},{y:.1f},{int(c)}' for (x,y,c) in kp])
        return f'{frame},{tid},{cls},{bbox},{kp_str}'
    for cls, tid in [(0,0),(1,1)]:
        base, n = KP_LAYOUT[cls]
        kp = [(coords[base+i,0], coords[base+i,1], int(vis[base+i])) for i in range(n)]
        rows.append(row(frame_idx, tid, cls, kp))
    for cls, tid in [(2,2),(3,3),(4,4)]:
        base, n = KP_LAYOUT[cls]
        kp = [(coords[base+i,0], coords[base+i,1], int(vis[base+i])) for i in range(n)]
        rows.append(row(frame_idx, tid, cls, kp))
    base, n_total = KP_LAYOUT[5]
    n = 3
    for inst, tid in enumerate([5,6]):
        kp = [(coords[base+inst*n+i,0], coords[base+inst*n+i,1], int(vis[base+inst*n+i])) for i in range(n)]
        rows.append(row(frame_idx, tid, 5, kp))
    return rows


def load_val_frames(data_root: str, video_id: str) -> List[str]:
    frame_dir = os.path.join(data_root, 'val', 'frames', video_id)
    return sorted(glob.glob(os.path.join(frame_dir, f'{video_id}_frame_*.png')),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_frame_')[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')))
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'kp_eval', 'data', 'trackers_klt')))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--image-size', type=str, default='384')
    ap.add_argument('--heatmap-size', type=str, default='128')
    ap.add_argument('--videos', nargs='*')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--in-chans', '--in_chans', dest='in_chans', type=int, default=3,
                    help='model input channels (3 or 6). If omitted, guessed from ckpt')
    args = ap.parse_args()

    assert cv2 is not None, 'OpenCV (cv2) is required for KLT mode'

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    sd = torch.load(args.ckpt, map_location='cpu')
    weights = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
    # guess input channels from checkpoint
    def _guess_in_chans(w):
        for k in ['encoder.stem_0.weight', 'encoder.stem.conv.weight']:
            if k in w and w[k].dim() == 4:
                return int(w[k].shape[1])
        for k, v in w.items():
            if (k.endswith('stem_0.weight') or k.endswith('stem.conv.weight')) and v.dim() == 4:
                return int(v.shape[1])
        return None
    ckpt_in = _guess_in_chans(weights) or args.in_chans
    if ckpt_in != args.in_chans:
        print(f"[info] overriding in_chans: ckpt={ckpt_in} arg={args.in_chans}")
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
    img_h, img_w = _parse_hw(args.image_size)
    hm_h, hm_w = _parse_hw(args.heatmap_size)
    hm_h, hm_w = img_h, img_w
    args.image_h, args.image_w = img_h, img_w
    args.heatmap_h, args.heatmap_w = hm_h, hm_w
    args.heatmap_size = args.image_size
    model = KPNet(image_size=(args.image_h, args.image_w), heatmap_size=(args.heatmap_h, args.heatmap_w), in_chans=ckpt_in)
    model.load_state_dict(weights, strict=False)
    model = model.to(device).eval()

    frame_root = os.path.join(args.data_root, 'val', 'frames')
    videos = args.videos if args.videos else sorted(os.listdir(frame_root))
    os.makedirs(args.out_dir, exist_ok=True)
    for vid in videos:
        files = load_val_frames(args.data_root, vid)
        if not files:
            print(f'[skip] no frames for {vid}')
            continue
        rows = []
        prev_canvas = None
        prev_pts = None  # (27,2) at image coords
        for t, f in enumerate(files):
            idx = int(os.path.splitext(os.path.basename(f))[0].split('_frame_')[1])
            img = Image.open(f).convert('RGB')
            if t == 0 or (t % args.k) == 0 or prev_pts is None:
                coords, vis, canvas = detect_frame(model, img, (args.image_h, args.image_w), (args.heatmap_h, args.heatmap_w), device)
                prev_pts = coords.copy()
                prev_canvas = canvas
            else:
                # KLT track from prev_canvas to current canvas
                pc = np.array(prev_canvas)
                cc = np.array(letterbox(img, (args.image_h, args.image_w))[4])
                g0 = cv2.cvtColor(pc, cv2.COLOR_RGB2GRAY)
                g1 = cv2.cvtColor(cc, cv2.COLOR_RGB2GRAY)
                pts0 = prev_pts.reshape(-1,1,2).astype(np.float32)
                pts1, st, err = cv2.calcOpticalFlowPyrLK(g0, g1, pts0, None, winSize=(21,21), maxLevel=3,
                                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
                pts1 = pts1.reshape(-1,2)
                # map back to original coords
                coords = pts1.copy()
                # convert from image_size canvas to original using current frame params
                _, s, pl, pt, _ = letterbox(img, (args.image_h, args.image_w))
                coords[:,0] = (coords[:,0] - pl) / s
                coords[:,1] = (coords[:,1] - pt) / s
                vis = np.full((coords.shape[0],), 2, dtype=np.int64)
                prev_canvas = Image.fromarray(cc)
                prev_pts = pts1.copy()
            rows.extend(rows_from_coords(idx, coords, vis))
        out_path = os.path.join(args.out_dir, f'{vid}_pred.txt')
        with open(out_path, 'w') as fw:
            for r in rows:
                fw.write(r + '\n')
        print(f'[pred-klt] {vid}: {out_path}')


if __name__ == '__main__':
    main()
