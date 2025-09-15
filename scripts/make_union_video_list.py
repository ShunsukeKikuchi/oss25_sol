#!/usr/bin/env python3
import os
import argparse

SPLITS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'splits', 'oss25_5fold_v2'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits-dir', default=SPLITS_DIR)
    ap.add_argument('--out', default=os.path.join(SPLITS_DIR, 'all_videos.txt'))
    args = ap.parse_args()

    vids = set()
    for name in os.listdir(args.splits_dir):
        if name.endswith('_train_videos.txt') or name.endswith('_val_videos.txt'):
            with open(os.path.join(args.splits_dir, name)) as f:
                for l in f:
                    l = l.strip()
                    if l:
                        vids.add(l)
    vids = sorted(vids)
    with open(args.out, 'w') as f:
        f.write('\n'.join(vids))
    print(f'wrote {len(vids)} videos to', args.out)


if __name__ == '__main__':
    main()

