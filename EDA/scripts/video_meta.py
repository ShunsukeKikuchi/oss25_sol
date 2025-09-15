#!/usr/bin/env python3
import os
import json
import random
import subprocess
import shutil
from typing import Optional, Dict, Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_MOVIE = os.path.abspath(os.path.join(ROOT, '../data/movie'))
OUT_DIR = os.path.join(ROOT, 'outputs')

def list_videos():
    return sorted([f for f in os.listdir(DATA_MOVIE) if f.lower().endswith('.mp4')])

def parse_rate(s: str) -> Optional[float]:
    if not s:
        return None
    if '/' in s:
        a, b = s.split('/')
        try:
            return float(a) / float(b) if float(b) != 0 else None
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None

def meta_ffprobe(path: str) -> Optional[Dict[str, Any]]:
    if not shutil.which('ffprobe'):
        return None
    try:
        cmd = [
            'ffprobe','-v','error',
            '-select_streams','v:0',
            '-show_entries','stream=width,height,avg_frame_rate',
            '-show_entries','format=duration',
            '-of','json', path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode('utf-8', 'ignore'))
        width = data['streams'][0].get('width')
        height = data['streams'][0].get('height')
        fps = parse_rate(data['streams'][0].get('avg_frame_rate'))
        duration = float(data['format'].get('duration', 0.0))
        return {'width': width, 'height': height, 'fps': fps, 'duration': duration, 'source': 'ffprobe'}
    except Exception:
        return None

def meta_cv2(path: str) -> Optional[Dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration = float(frame_count / fps) if fps and fps > 0 else None
        return {'width': width, 'height': height, 'fps': float(fps) if fps else None, 'duration': duration, 'source': 'opencv'}
    except Exception:
        return None

def get_meta(path: str) -> Dict[str, Any]:
    meta = meta_ffprobe(path)
    if meta is None:
        meta = meta_cv2(path)
    return meta or {'width': None, 'height': None, 'fps': None, 'duration': None, 'source': None}

def main(sample_size: int = 20, seed: int = 0):
    os.makedirs(OUT_DIR, exist_ok=True)
    vids = list_videos()
    random.seed(seed)
    sample = vids if sample_size >= len(vids) else sorted(random.sample(vids, sample_size))
    results = {}
    for v in sample:
        p = os.path.join(DATA_MOVIE, v)
        results[v] = get_meta(p)
    # aggregate summary
    widths = [m['width'] for m in results.values() if m['width']]
    heights = [m['height'] for m in results.values() if m['height']]
    fpss = [m['fps'] for m in results.values() if m['fps']]
    durations = [m['duration'] for m in results.values() if m['duration']]
    summary = {
        'count': len(vids),
        'sampled': len(sample),
        'sources': sorted(list(set([m['source'] for m in results.values() if m['source']]))),
        'width_mode': max(set(widths), key=widths.count) if widths else None,
        'height_mode': max(set(heights), key=heights.count) if heights else None,
        'fps_mode': max(set(fpss), key=fpss.count) if fpss else None,
        'duration_mean_sec': (sum(durations)/len(durations)) if durations else None,
    }
    with open(os.path.join(OUT_DIR, 'video_meta_samples.json'), 'w') as f:
        json.dump({'summary': summary, 'samples': results}, f, indent=2)
    print('Video meta summary:', summary)

if __name__ == '__main__':
    main()

