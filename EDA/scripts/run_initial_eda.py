#!/usr/bin/env python3
import os
import sys
import json
import glob
import re
from collections import Counter, defaultdict

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs'))
os.makedirs(OUT_DIR, exist_ok=True)

summary = {
    'paths': {},
    'movie': {},
    'labels': {},
    'task3': {},
    'notes': []
}

def safe_listdir(path):
    try:
        return sorted(os.listdir(path))
    except Exception as e:
        summary['notes'].append(f'listdir failed for {path}: {e}')
        return []

def gather_movies():
    movie_dir = os.path.join(DATA_ROOT, 'movie')
    files = [f for f in safe_listdir(movie_dir) if f.lower().endswith('.mp4')]
    ids = [os.path.splitext(f)[0] for f in files]
    summary['movie'] = {
        'dir': movie_dir,
        'count': len(files),
        'sample': files[:10],
    }
    return set(ids)

def _norm_key(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())

def _find_col(actual_cols, candidates_norm):
    by_norm = {_norm_key(c): c for c in actual_cols}
    for target in candidates_norm:
        if target in by_norm:
            return by_norm[target]
    return None

def try_read_labels():
    label_path = os.path.join(DATA_ROOT, 'OSATS_MICCAI_trainset.xlsx')
    df = None
    reason = None
    try:
        import pandas as pd  # type: ignore
        df = pd.read_excel(label_path)
    except Exception as e:
        reason = f'pandas read_excel failed: {e!r}'
        # try openpyxl directly
        if 'openpyxl' in (str(e) if e else ''):
            try:
                import openpyxl  # noqa: F401
                import pandas as pd
                df = pd.read_excel(label_path)
                reason = None
            except Exception as e2:
                reason = f'openpyxl path failed: {e2!r}'
    if df is None:
        summary['labels'] = {
            'path': label_path,
            'loaded': False,
            'reason': reason,
        }
        return None
    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    actual_cols = list(df.columns)
    # resolve canonical columns with aliasing by normalized name
    video_col = _find_col(actual_cols, [_norm_key('VIDEO')]) or 'VIDEO'
    grs_col = _find_col(actual_cols, [
        _norm_key('GLOBARATINGSCORE'),
        _norm_key('GLOBALRATINGSCORE'),
        _norm_key('GLOBA_RATING_SCORE'),
        _norm_key('GLOBAL_RATING_SCORE'),
    ])
    osats_canonical = [
        'OSATS_RESPECT','OSATS_MOTION','OSATS_INSTRUMENT','OSATS_SUTURE',
        'OSATS_FLOW','OSATS_KNOWLEDGE','OSATS_PERFORMANCE','OSATSFINALQUALITY'
    ]
    osats_aliases = {
        'OSATSFINALQUALITY': ['OSATS_FINAL_QUALITY','OSATS FINAL QUALITY','OSATS-FINAL-QUALITY']
    }
    osats_cols_resolved = {}
    for name in osats_canonical:
        candidates = [name] + osats_aliases.get(name, [])
        found = _find_col(actual_cols, [_norm_key(c) for c in candidates])
        if found:
            osats_cols_resolved[name] = found
    summary['labels'] = {
        'path': label_path,
        'loaded': True,
        'rows': int(df.shape[0]),
        'cols': actual_cols,
        'has_required': {
            'VIDEO': (video_col in df.columns),
            'GRS(aliased)': bool(grs_col),
        },
        'has_osats': {c: (c in osats_cols_resolved) for c in osats_canonical},
        'resolved': {
            'video_col': video_col,
            'grs_col': grs_col,
            'osats_cols': osats_cols_resolved,
        }
    }
    # extract basics
    videos = set(df[video_col].astype(str).str.strip()) if video_col in df.columns else set()
    # GRS to 4-class
    grs_counts = {}
    if grs_col:
        grs = df[grs_col]
        def to_cls(v):
            try:
                v = float(v)
            except Exception:
                return None
            if 8 <= v <= 15: return 0
            if 16 <= v <= 23: return 1
            if 24 <= v <= 31: return 2
            if 32 <= v <= 40: return 3
            return None
        classes = [to_cls(v) for v in grs]
        c = Counter([x for x in classes if x is not None])
        grs_counts = {str(k): int(v) for k, v in sorted(c.items())}
    # OSATS range check
    osats_stats = {}
    for c_canon, c_real in osats_cols_resolved.items():
        if c_real in df.columns:
            vals = df[c_real].dropna().tolist()
            try:
                vals = [int(round(float(v))) for v in vals]
            except Exception:
                pass
            if vals:
                osats_stats[c_canon] = {
                    'min': int(min(vals)),
                    'max': int(max(vals)),
                    'counts': {str(k): int(v) for k, v in sorted(Counter(vals).items())}
                }
    summary['labels'].update({
        'video_ids_in_xlsx': len(videos),
        'grs_class_counts': grs_counts,
        'osats_stats': osats_stats,
    })
    return videos

def compare_video_lists(movie_ids, label_ids):
    missing_videos = sorted(list((label_ids or set()) - (movie_ids or set()))) if label_ids is not None else []
    orphan_movies = sorted(list((movie_ids or set()) - (label_ids or set()))) if movie_ids is not None and label_ids is not None else []
    summary['movie'].update({
        'unique_ids': len(movie_ids or []),
        'orphan_movies_count': len(orphan_movies),
        'orphan_movies_sample': orphan_movies[:20],
    })
    summary['labels'].update({
        'missing_video_files_count': len(missing_videos),
        'missing_video_files_sample': missing_videos[:20],
    })
    # write detailed lists
    with open(os.path.join(OUT_DIR, 'missing_video_files.txt'), 'w') as f:
        f.write('\n'.join(missing_videos))
    with open(os.path.join(OUT_DIR, 'orphan_movies.txt'), 'w') as f:
        f.write('\n'.join(orphan_movies))

def parse_task3():
    tdir = os.path.join(DATA_ROOT, 'train')
    vdir = os.path.join(DATA_ROOT, 'val')
    mot_train = sorted(safe_listdir(os.path.join(tdir, 'mot')))
    mot_val = sorted(safe_listdir(os.path.join(vdir, 'mot')))
    frames_train = sorted(safe_listdir(os.path.join(tdir, 'frames')))
    frames_val_dirs = sorted(safe_listdir(os.path.join(vdir, 'frames')))

    # Parse a subset of MOT files for stats
    class_counts = Counter()
    kp_stats = defaultdict(lambda: Counter())  # class_id -> Counter(kp_count)
    lines_sampled = 0
    for fname in mot_train[:50]:  # limit for speed
        path = os.path.join(tdir, 'mot', fname)
        try:
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 200:  # cap per file
                        break
                    parts = [p.strip() for p in line.strip().split(',') if len(p.strip())]
                    if len(parts) < 7:
                        continue
                    frame = parts[0]
                    track_id = parts[1]
                    class_id = parts[2]
                    # skip bbox (next 4 fields)
                    rest = parts[7:] if len(parts) >= 11 else []
                    # count triples
                    triples = len(rest) // 3
                    try:
                        cid = int(float(class_id))
                    except Exception:
                        continue
                    class_counts[cid] += 1
                    kp_stats[cid][triples] += 1
                    lines_sampled += 1
        except Exception as e:
            summary['notes'].append(f'Failed to parse {path}: {e!r}')

    summary['task3'] = {
        'train': {
            'mot_files': len(mot_train),
            'frames_files': len(frames_train),
        },
        'val': {
            'mot_files': len(mot_val),
            'frame_dirs': len(frames_val_dirs),
        },
        'mot_sampled_lines': lines_sampled,
        'class_row_counts': {str(k): int(v) for k, v in sorted(class_counts.items())},
        'kp_triple_counts_by_class': {str(k): {str(kk): int(vv) for kk, vv in sorted(c.items())}
                                      for k, c in kp_stats.items()},
    }

def write_outputs():
    # JSON summary
    with open(os.path.join(OUT_DIR, 'initial_eda_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # Text summary
    lines = []
    lines.append('# Initial EDA Summary')
    lines.append('')
    lines.append('## Movies')
    lines.append(json.dumps(summary.get('movie', {}), ensure_ascii=False, indent=2))
    lines.append('')
    lines.append('## Labels')
    lines.append(json.dumps(summary.get('labels', {}), ensure_ascii=False, indent=2))
    lines.append('')
    lines.append('## Task3')
    lines.append(json.dumps(summary.get('task3', {}), ensure_ascii=False, indent=2))
    lines.append('')
    lines.append('## Notes')
    lines.append('\n'.join(summary.get('notes', [])))
    with open(os.path.join(OUT_DIR, 'initial_eda_summary.md'), 'w') as f:
        f.write('\n'.join(lines))

def main():
    summary['paths'] = {
        'DATA_ROOT': DATA_ROOT,
        'OUT_DIR': OUT_DIR,
    }
    movie_ids = gather_movies()
    label_ids = try_read_labels()
    compare_video_lists(movie_ids, label_ids)
    parse_task3()
    write_outputs()
    # print concise stdout
    print('Movies:', summary['movie']['count'], 'files')
    if summary['labels'].get('loaded'):
        print('Labels: rows =', summary['labels'].get('rows'))
        print('GRS classes:', summary['labels'].get('grs_class_counts'))
    else:
        print('Labels: not loaded ->', summary['labels'].get('reason'))
    print('Task3 train mot files:', summary['task3']['train']['mot_files'])
    print('Task3 val mot files:', summary['task3']['val']['mot_files'])
    print('KP triple counts by class (sample):', summary['task3']['kp_triple_counts_by_class'])

if __name__ == '__main__':
    main()
