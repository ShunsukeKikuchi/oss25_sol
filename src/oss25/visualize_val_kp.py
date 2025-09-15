#!/usr/bin/env python3
import os
import argparse
from typing import List, Tuple, Dict
import numpy as np

from PIL import Image, ImageDraw


CLASS_NAMES = {
    0: 'left hand',
    1: 'right hand',
    2: 'scissors',
    3: 'tweezers',
    4: 'needle holder',
    5: 'needle',
}

CLASS_COLORS_GT = {
    0: (255, 0, 0, 220),      # red
    1: (255, 128, 0, 220),    # orange
    2: (255, 0, 255, 220),    # magenta
    3: (255, 0, 128, 220),    # pink
    4: (128, 0, 255, 220),    # purple
    5: (255, 0, 64, 220),     # red-ish
}

CLASS_COLORS_PRED = {
    0: (0, 128, 255, 220),    # blue
    1: (0, 200, 200, 220),    # cyan
    2: (0, 200, 0, 220),      # green
    3: (0, 150, 0, 220),
    4: (0, 100, 0, 220),
    5: (0, 255, 128, 220),
}

# Segmentation color palettes (RGBA)
SEG_COLORS_GT = [
    (0, 0, 0, 0),          # 0 background -> transparent
    (255, 0, 0, 120),      # 1 left hand
    (255, 128, 0, 120),    # 2 right hand
    (0, 200, 0, 120),      # 3 scissors
    (0, 150, 0, 120),      # 4 tweezers
    (0, 100, 0, 120),      # 5 needle holder
    (0, 255, 128, 120),    # 6 needle
]

SEG_COLORS_PRED = [
    (0, 0, 0, 0),          # 0 background -> transparent
    (0, 128, 255, 80),     # alt palette for predicted seg
    (0, 200, 200, 80),
    (255, 0, 255, 80),
    (255, 0, 128, 80),
    (128, 0, 255, 80),
    (255, 0, 64, 80),
]


def parse_line(line: str) -> Tuple[int, int, int, List[Tuple[float, float, int]]]:
    parts = [p.strip() for p in line.strip().split(',') if p.strip()]
    frame = int(float(parts[0]))
    track_id = int(float(parts[1]))
    class_id = int(float(parts[2]))
    rest = parts[7:]
    triples: List[Tuple[float, float, int]] = []
    for i in range(0, len(rest), 3):
        try:
            x = float(rest[i]); y = float(rest[i+1]); c = int(float(rest[i+2]))
            triples.append((x, y, c))
        except Exception:
            break
    return frame, track_id, class_id, triples


def load_kp_file(path: str) -> Dict[int, Dict[int, List[Tuple[float, float, int]]]]:
    # returns: frame -> class_id -> list of keypoints (x,y,c)
    data: Dict[int, Dict[int, List[Tuple[float, float, int]]]] = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            frame, tid, cls, triples = parse_line(line)
            data.setdefault(frame, {})[cls] = triples
    return data


def draw_kps(im: Image.Image, kps_by_class: Dict[int, List[Tuple[float, float, int]]], is_gt: bool):
    draw = ImageDraw.Draw(im, 'RGBA')
    colors = CLASS_COLORS_GT if is_gt else CLASS_COLORS_PRED
    for cls, triples in kps_by_class.items():
        col = colors.get(cls, (255, 255, 255, 200))
        for (x, y, c) in triples:
            r = 5
            alpha = 220 if c == 2 else (140 if c == 1 else 80)
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(col[0], col[1], col[2], alpha))


def draw_correspondences(im: Image.Image,
                         gt_by_class: Dict[int, List[Tuple[float, float, int]]],
                         pred_by_class: Dict[int, List[Tuple[float, float, int]]]):
    """Draw straight lines connecting corresponding GT and Pred keypoints per class, index-wise.
    Assumes ordering一致（手6点、ツール3点、針3点x2）.
    """
    draw = ImageDraw.Draw(im, 'RGBA')
    # line color: yellow-ish with alpha, thicker when vis=2
    for cls, gt_kps in gt_by_class.items():
        pred_kps = pred_by_class.get(cls, [])
        n = min(len(gt_kps), len(pred_kps))
        for i in range(n):
            gx, gy, gc = gt_kps[i]
            px, py, pc = pred_kps[i]
            # transparency reflects visibility agreement; thicker when both visible
            both_vis = (gc == 2 and pc == 2)
            width = 3 if both_vis else 2
            alpha = 220 if both_vis else 140
            draw.line([(gx, gy), (px, py)], fill=(255, 255, 0, alpha), width=width)


def overlay_seg_mask(base: Image.Image, mask_path: str, palette: List[Tuple[int, int, int, int]], alpha_scale: float = 1.0):
    if not os.path.exists(mask_path):
        return
    try:
        m = Image.open(mask_path).convert('L')
        arr = np.array(m).astype(np.int64)
        h, w = arr.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        # map labels to colors
        for lb in range(min(len(palette), 7)):
            mask = (arr == lb)
            if not mask.any():
                continue
            r, g, b, a = palette[lb]
            if alpha_scale != 1.0:
                a = int(max(0, min(255, round(a * alpha_scale))))
            rgba[mask] = (r, g, b, a)
        overlay = Image.fromarray(rgba, mode='RGBA')
        base.alpha_composite(overlay)
    except Exception:
        return


def pick_frames(frames: List[int], limit: int, step: int) -> List[int]:
    frames = sorted(set(frames))
    if step > 1:
        frames = frames[::step]
    return frames[:limit] if limit > 0 else frames


def main():
    ap = argparse.ArgumentParser(description='Visualize GT vs Pred keypoints for val video')
    ap.add_argument('--data-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')))
    ap.add_argument('--gt-folder', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'kp_eval', 'data', 'gt')))
    ap.add_argument('--pred', required=True, help='Path to predicted tracker file (<video>_pred.txt)')
    ap.add_argument('--video', required=True, help='Video ID (e.g., E66F)')
    ap.add_argument('--out', required=True, help='Output folder to save overlays')
    ap.add_argument('--limit', type=int, default=12, help='Max frames to visualize (0 for all)')
    ap.add_argument('--step', type=int, default=1, help='Stride when picking frames (by sorted unique frame numbers)')
    # segmentation overlay options
    ap.add_argument('--seg-root', default='', help='Root dir for GT/pseudo seg masks (e.g., artifacts/seg_pseudo6)')
    ap.add_argument('--seg-pred-root', default='', help='Root dir for predicted seg masks (optional)')
    ap.add_argument('--seg-alpha', type=float, default=1.0, help='Global alpha scale for seg overlays (0..1)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gt_path = os.path.join(args.gt_folder, f'{args.video}.txt')
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f'GT file not found: {gt_path}')
    pred_path = args.pred
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f'Pred file not found: {pred_path}')

    gt = load_kp_file(gt_path)
    pred = load_kp_file(pred_path)
    frames = sorted(set(list(gt.keys()) + list(pred.keys())))
    frames_show = pick_frames(frames, args.limit, args.step)

    # frame image path pattern
    frame_dir = os.path.join(args.data_root, 'val', 'frames', args.video)
    for fr in frames_show:
        img_path = os.path.join(frame_dir, f'{args.video}_frame_{fr}.png')
        if not os.path.exists(img_path):
            # skip if frame image not found
            continue
        im = Image.open(img_path).convert('RGBA')
        # compose layers (RGBA)
        layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
        # optional segmentation overlays
        if args.seg_root:
            seg_path = os.path.join(args.seg_root, args.video, f'{args.video}_frame_{fr}.png')
            overlay_seg_mask(layer, seg_path, SEG_COLORS_GT, alpha_scale=args.seg_alpha)
        if args.seg_pred_root:
            segp_path = os.path.join(args.seg_pred_root, args.video, f'{args.video}_frame_{fr}.png')
            overlay_seg_mask(layer, segp_path, SEG_COLORS_PRED, alpha_scale=args.seg_alpha)
        # GT then Pred + correspondence lines
        gt_k = gt.get(fr, {})
        pred_k = pred.get(fr, {})
        draw_correspondences(layer, gt_k, pred_k)
        draw_kps(layer, gt_k, is_gt=True)
        draw_kps(layer, pred_k, is_gt=False)
        out_im = Image.alpha_composite(im, layer).convert('RGB')
        out_path = os.path.join(args.out, f'{args.video}_frame_{fr}_overlay.jpg')
        out_im.save(out_path, quality=92)
        print(f'[vis] saved {out_path}')


if __name__ == '__main__':
    main()
