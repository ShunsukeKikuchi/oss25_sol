#!/usr/bin/env python3
import os
import argparse
import shutil
from typing import Dict, List

def collect_val_mot(val_mot_dir: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    txts = [n for n in sorted(os.listdir(val_mot_dir)) if n.endswith('.txt')]
    if not txts:
        return groups
    # Case A: per-frame files exist
    per_frame = [n for n in txts if '_frame_' in os.path.splitext(n)[0]]
    if per_frame:
        for name in per_frame:
            stem = os.path.splitext(name)[0]
            video = stem.split('_frame_')[0]
            groups.setdefault(video, []).append(os.path.join(val_mot_dir, name))
        # sort by frame index
        for k in groups:
            groups[k].sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_frame_')[1]))
        return groups
    # Case B: already aggregated per video: just pass through
    for name in txts:
        stem = os.path.splitext(name)[0]
        groups[stem] = [os.path.join(val_mot_dir, name)]
    return groups


def aggregate_to_single_files(groups: Dict[str, List[str]], out_gt_dir: str):
    os.makedirs(out_gt_dir, exist_ok=True)
    for video, files in groups.items():
        out_path = os.path.join(out_gt_dir, f'{video}.txt')
        # If only one aggregated file, copy it
        if len(files) == 1 and os.path.basename(files[0]).startswith(video):
            shutil.copy2(files[0], out_path)
            print(f'[gt] copied {files[0]} -> {out_path}')
            continue
        # Otherwise concatenate per-frame files
        with open(out_path, 'w') as out:
            for fp in files:
                with open(fp, 'r') as f:
                    for line in f:
                        out.write(line.rstrip('\n') + '\n')
        print(f'[gt] wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')))
    ap.add_argument('--out-root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'kp_eval', 'data')))
    args = ap.parse_args()

    val_mot_dir = os.path.join(args.data_root, 'val', 'mot')
    out_gt_dir = os.path.join(args.out_root, 'gt')
    groups = collect_val_mot(val_mot_dir)
    aggregate_to_single_files(groups, out_gt_dir)
    print('[done] GT aggregation complete.')


if __name__ == '__main__':
    main()
