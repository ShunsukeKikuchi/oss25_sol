#!/usr/bin/env python3
import os
import argparse
import subprocess
import json
from typing import List

DATA_MOVIE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'movie'))
OUT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'frames_1fps'))


def ffprobe_fps_duration(path: str):
    cmd = [
        'ffprobe','-v','error',
        '-select_streams','v:0',
        '-show_entries','stream=avg_frame_rate',
        '-show_entries','format=duration',
        '-of','json', path
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    data = json.loads(out.decode('utf-8', 'ignore'))
    rate = data['streams'][0]['avg_frame_rate']
    if '/' in rate:
        a, b = rate.split('/')
        fps = float(a) / float(b) if float(b) != 0 else 0.0
    else:
        fps = float(rate)
    duration = float(data['format'].get('duration', 0.0))
    return fps, duration


def extract_1fps(video: str, dry_run=False) -> List[str]:
    vid_id = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.join(OUT_BASE, vid_id)
    os.makedirs(out_dir, exist_ok=True)
    fps, duration = ffprobe_fps_duration(video)
    # 1 fps sampling using ffmpeg, starting at 0
    tmp_pattern = os.path.join(out_dir, f"{vid_id}_frame_%06d_tmp.jpg")
    if not dry_run:
        cmd = [
            'ffmpeg','-y','-hide_banner','-loglevel','error',
            '-i', video,
            '-vf','fps=1',
            '-start_number','0',
            '-q:v','2',
            tmp_pattern
        ]
        subprocess.check_call(cmd)
    # rename to original frame indices: idx = round(second * fps)
    generated = []
    k = 0
    while True:
        src = os.path.join(out_dir, f"{vid_id}_frame_{k:06d}_tmp.jpg")
        if not os.path.exists(src):
            break
        frame_idx = int(round(k * fps))
        dst = os.path.join(out_dir, f"{vid_id}_frame_{frame_idx}.jpg")
        os.replace(src, dst)
        generated.append(dst)
        k += 1
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', nargs='*', help='Specific VIDEO ids (without .mp4). If omitted, process all.')
    parser.add_argument('--list', type=str, help='Path to a txt file with VIDEO ids to process')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.list:
        with open(args.list, 'r') as f:
            ids = [l.strip() for l in f if l.strip()]
    elif args.videos:
        ids = args.videos
    else:
        ids = [os.path.splitext(f)[0] for f in os.listdir(DATA_MOVIE) if f.lower().endswith('.mp4')]

    os.makedirs(OUT_BASE, exist_ok=True)
    for vid in ids:
        video_path = os.path.join(DATA_MOVIE, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"[skip] missing: {video_path}")
            continue
        print(f"[extract] {vid} -> {OUT_BASE}/{vid}/")
        outs = extract_1fps(video_path, dry_run=args.dry_run)
        print(f"  saved {len(outs)} frames")


if __name__ == '__main__':
    main()

