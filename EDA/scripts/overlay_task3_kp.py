#!/usr/bin/env python3
import os
import random
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

import argparse
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.abspath(os.path.join(ROOT, '../data'))
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
FRAMES_DIR_TR = os.path.join(TRAIN_DIR, 'frames')
MOT_DIR_TR = os.path.join(TRAIN_DIR, 'mot')
FRAMES_DIR_VAL = os.path.join(VAL_DIR, 'frames')
MOT_DIR_VAL = os.path.join(VAL_DIR, 'mot')
OUT_DIR = os.path.join(ROOT, 'outputs', 'task3_overlay')

CLASS_COLORS = {
    0: (66, 135, 245, 200),   # Left Hand - blue
    1: (245, 130, 48, 200),   # Right Hand - orange
    2: (60, 180, 75, 200),    # Scissors - green
    3: (145, 30, 180, 200),   # Tweezers - purple
    4: (230, 25, 75, 200),    # Needle Holder - red
    5: (128, 64, 0, 200),     # Needle - brown
}

def parse_line(line: str) -> Tuple[int, int, int, List[Tuple[float, float, int]]]:
    parts = [p.strip() for p in line.strip().split(',') if len(p.strip())]
    if len(parts) < 7:
        return None
    try:
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
        class_id = int(float(parts[2]))
    except Exception:
        return None
    rest = parts[7:] if len(parts) >= 11 else []
    triples = []
    for i in range(0, len(rest), 3):
        try:
            x = float(rest[i]); y = float(rest[i+1]); v = int(float(rest[i+2]))
            triples.append((x, y, v))
        except Exception:
            break
    return frame, track_id, class_id, triples

def overlay_one(frame_png: str, mot_txt: str, out_png: str):
    im = Image.open(frame_png).convert('RGBA')
    draw = ImageDraw.Draw(im, 'RGBA')
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    with open(mot_txt, 'r') as f:
        for line in f:
            parsed = parse_line(line)
            if not parsed:
                continue
            frame_idx, tid, cid, kps = parsed
            color = CLASS_COLORS.get(cid, (255, 255, 255, 200))
            for j, (x, y, v) in enumerate(kps):
                r = 5
                # adjust alpha by visibility (2 visible, 1 hidden, 0 out)
                a = 200 if v == 2 else (120 if v == 1 else 60)
                col = (color[0], color[1], color[2], a)
                bbox = [(x - r, y - r), (x + r, y + r)]
                draw.ellipse(bbox, fill=col, outline=(0, 0, 0, 180))
                if j == 0 and font:
                    draw.text((x + 6, y - 6), f"{tid}", fill=(255, 255, 255, 220), font=font)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    im.save(out_png)

def sample_and_overlay(dataset: str, max_images: int, seed: int) -> int:
    random.seed(seed)
    if dataset == 'train':
        mot_dir = MOT_DIR_TR
        frames_dir = FRAMES_DIR_TR
        out_sub = os.path.join(OUT_DIR, 'train')
        mot_files = sorted(os.listdir(mot_dir))
        sample = mot_files if max_images >= len(mot_files) else random.sample(mot_files, max_images)
        count = 0
        for fname in sample:
            stem = os.path.splitext(fname)[0]
            frame_png = os.path.join(frames_dir, stem + '.png')
            mot_txt = os.path.join(mot_dir, fname)
            if not os.path.exists(frame_png):
                continue
            out_png = os.path.join(out_sub, stem + '_overlay.png')
            overlay_one(frame_png, mot_txt, out_png)
            count += 1
        return count
    elif dataset == 'val':
        mot_dir = MOT_DIR_VAL
        out_sub = os.path.join(OUT_DIR, 'val')
        mot_files = sorted(os.listdir(mot_dir))
        sample = mot_files if max_images >= len(mot_files) else random.sample(mot_files, max_images)
        count = 0
        for fname in sample:
            stem = os.path.splitext(fname)[0]
            video_id = stem.split('_frame_')[0]
            frame_png = os.path.join(FRAMES_DIR_VAL, video_id, stem + '.png')
            mot_txt = os.path.join(mot_dir, fname)
            if not os.path.exists(frame_png):
                continue
            out_png = os.path.join(out_sub, stem + '_overlay.png')
            overlay_one(frame_png, mot_txt, out_png)
            count += 1
        return count
    else:
        raise ValueError('dataset must be train or val')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both', choices=['train','val','both'])
    parser.add_argument('--max', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    if args.dataset in ('train','both'):
        total += sample_and_overlay('train', args.max, args.seed)
    if args.dataset in ('val','both'):
        total += sample_and_overlay('val', args.max, args.seed)
    print('Saved', total, 'overlay images to', OUT_DIR)

if __name__ == '__main__':
    main()
