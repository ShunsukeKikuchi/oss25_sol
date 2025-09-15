#!/usr/bin/env python3
import os
import json
import subprocess
import shutil
from statistics import mean

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_MOVIE = os.path.abspath(os.path.join(ROOT, '../data/movie'))
OUT_DIR = os.path.join(ROOT, 'outputs')

def list_videos():
    return sorted([f for f in os.listdir(DATA_MOVIE) if f.lower().endswith('.mp4')])

def parse_rate(s: str):
    if not s:
        return None
    if '/' in s:
        num, den = s.split('/')
        try:
            num = float(num); den = float(den)
            return (num / den) if den else None
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None

def ffprobe_meta(path):
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
        stream = data['streams'][0]
        width = int(stream.get('width'))
        height = int(stream.get('height'))
        fps = parse_rate(stream.get('avg_frame_rate'))
        duration = float(data['format'].get('duration', 0.0))
        return {'width': width, 'height': height, 'fps': fps, 'duration': duration}
    except Exception:
        return None

def summarize(metas):
    res_counts = {}
    fps_counts = {}
    durations = []
    for m in metas:
        if not m: continue
        res = f"{m['width']}x{m['height']}" if m.get('width') and m.get('height') else 'None'
        res_counts[res] = res_counts.get(res, 0) + 1
        if m.get('fps'):
            fr = round(m['fps'], 3)
            fps_counts[fr] = fps_counts.get(fr, 0) + 1
        if m.get('duration'):
            durations.append(m['duration'])
    return {
        'resolution_counts': dict(sorted(res_counts.items(), key=lambda x: (-x[1], x[0]))),
        'fps_counts': dict(sorted(fps_counts.items(), key=lambda x: (-x[1], x[0]))),
        'duration_min_sec': min(durations) if durations else None,
        'duration_max_sec': max(durations) if durations else None,
        'duration_mean_sec': mean(durations) if durations else None,
        'count': len(metas),
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vids = list_videos()
    results = {}
    metas = []
    for v in vids:
        p = os.path.join(DATA_MOVIE, v)
        m = ffprobe_meta(p)
        results[v] = m
        metas.append(m)
    summ = summarize(metas)
    out_json = {
        'summary': summ,
        'files': results
    }
    with open(os.path.join(OUT_DIR, 'video_meta_full.json'), 'w') as f:
        json.dump(out_json, f, indent=2)
    # write a quick markdown
    md_lines = []
    md_lines.append('# Video Meta Summary (Full)')
    md_lines.append('')
    md_lines.append('## Resolutions')
    for k, v in summ['resolution_counts'].items():
        md_lines.append(f'- {k}: {v}')
    md_lines.append('')
    md_lines.append('## FPS (rounded 3 decimals)')
    for k, v in summ['fps_counts'].items():
        md_lines.append(f'- {k}: {v}')
    md_lines.append('')
    md_lines.append('## Durations (sec)')
    md_lines.append(f"- min: {summ['duration_min_sec']}")
    md_lines.append(f"- mean: {summ['duration_mean_sec']}")
    md_lines.append(f"- max: {summ['duration_max_sec']}")
    with open(os.path.join(OUT_DIR, 'video_meta_summary.md'), 'w') as f:
        f.write('\n'.join(md_lines))
    print('Processed', len(vids), 'videos')

if __name__ == '__main__':
    main()

