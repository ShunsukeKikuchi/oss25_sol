#!/usr/bin/env python3
import os
import argparse
import json
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .models.swin3d_aux import Swin3DWithSegAux
from .train_skill_swin3d import aggregate_labels, load_splits
from .datasets.skill_video_dataset import SkillVideoDataset


def grs_to_cls(v: float) -> int:
    if v <= 15:
        return 0
    if v <= 23:
        return 1
    if v <= 31:
        return 2
    return 3


@torch.no_grad()
def infer_video(model, ds: SkillVideoDataset, vid: str, tta: int, device: torch.device, n_frames: int,
                seed_base: int = 12345) -> Dict:
    # Build TTA by sampling multiple clips via different RNG seeds
    preds = []
    for rep in range(tta):
        # control RNG to make sampling reproducible per video+rep
        seed = seed_base + rep * 997 + (hash(vid) & 0x7fffffff)
        random.seed(seed)
        # fetch item
        idx = ds.video_ids.index(vid)
        sample = ds[idx]
        clip = sample['clip'].unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(clip)
            main = out['main']  # (1,1) for TaskA, (1,8) for TaskB
        preds.append(main.squeeze(0).cpu().numpy())
    preds = np.stack(preds, axis=0)  # (TTA, D)
    pred_mean = preds.mean(axis=0)
    return {
        'pred_mean': pred_mean,
        'pred_all': preds,
    }


def main():
    ap = argparse.ArgumentParser(description='Inference with TTA (multi sampling per video) for TaskA/B Swin3D')
    ap.add_argument('--task', choices=['A', 'B'], required=True)
    ap.add_argument('--model', choices=['model1', 'model2'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--videos', nargs='*', help='subset of videos; default=val split of fold')
    ap.add_argument('--n-frames', type=int, default=96)
    ap.add_argument('--last5s-sec', type=int, default=5)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--tta', type=int, default=5)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out-csv', default='')
    ap.add_argument('--out-json', default='')
    ap.add_argument('--split-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'splits', 'oss25_5fold_v1')))
    ap.add_argument('--xlsx', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'OSATS_MICCAI_trainset.xlsx')))
    ap.add_argument('--tenfps-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'sam2_frames_10fps')))
    ap.add_argument('--frames1fps-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'frames_1fps')))
    ap.add_argument('--pseudo6-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'seg_pseudo6')))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # videos list
    if args.videos:
        video_ids = args.videos
    else:
        _, val_videos = load_splits(args.split_root, args.fold)
        video_ids = val_videos

    # dummy labels for dataset construction (only structure needed)
    labels = aggregate_labels(args.xlsx)
    # For unknown videos (e.g., test), synthesize placeholder labels
    for v in video_ids:
        if v not in labels:
            labels[v] = {'osats': np.zeros(8, dtype=np.float32), 'grs_cls': 0, 'grs_num': 24.0}

    ds = SkillVideoDataset(
        video_ids, labels, args.tenfps_root, args.pseudo6_root,
        task=args.task, model_mode=args.model, n_frames=args.n_frames,
        last5s_sec=args.last5s_sec, image_size=args.image_size,
        deterministic_val=False, frames1fps_root=args.frames1fps_root,
    )

    # build model
    out_dim = 1 if args.task == 'A' else 8
    model = Swin3DWithSegAux(task=args.task, num_outputs=out_dim, weights=None)
    sd = torch.load(args.ckpt, map_location='cpu')
    if isinstance(sd, dict) and sd.get('model_ema') is not None:
        weights = sd['model_ema']
    else:
        weights = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()

    # run
    results = {}
    for vid in tqdm(video_ids, desc='infer'):
        out = infer_video(model, ds, vid, args.tta, device, args.n_frames)
        p = out['pred_mean']
        if args.task == 'A':
            grs = float(np.clip(p[0], 8.0, 40.0))
            results[vid] = {'GRS': grs}
        else:
            vec = np.clip(p, 0.0, 4.0)
            results[vid] = {
                'OSATS_RESPECT': float(vec[0]),
                'OSATS_MOTION': float(vec[1]),
                'OSATS_INSTRUMENT': float(vec[2]),
                'OSATS_SUTURE': float(vec[3]),
                'OSATS_FLOW': float(vec[4]),
                'OSATS_KNOWLEDGE': float(vec[5]),
                'OSATS_PERFORMANCE': float(vec[6]),
                'OSATS_FINAL_QUALITY': float(vec[7]),
            }

    # write outputs
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        import csv
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            if args.task == 'A':
                w.writerow(['VIDEO', 'GRS'])
                for vid in video_ids:
                    w.writerow([vid, results[vid]['GRS']])
            else:
                w.writerow(['VIDEO', 'OSATS_RESPECT', 'OSATS_MOTION', 'OSATS_INSTRUMENT', 'OSATS_SUTURE', 'OSATS_FLOW', 'OSATS_KNOWLEDGE', 'OSATS_PERFORMANCE', 'OSATS_FINAL_QUALITY'])
                for vid in video_ids:
                    r = results[vid]
                    w.writerow([vid, r['OSATS_RESPECT'], r['OSATS_MOTION'], r['OSATS_INSTRUMENT'], r['OSATS_SUTURE'], r['OSATS_FLOW'], r['OSATS_KNOWLEDGE'], r['OSATS_PERFORMANCE'], r['OSATS_FINAL_QUALITY']])

    # also print quick summary
    print(f'[done] videos={len(video_ids)}, wrote: csv={bool(args.out_csv)}, json={bool(args.out_json)}')


if __name__ == '__main__':
    main()
