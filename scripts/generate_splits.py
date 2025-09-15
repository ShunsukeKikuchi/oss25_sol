#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(REPO_ROOT, '..', 'data')
MOVIE_DIR = os.path.join(DATA_ROOT, 'movie')
OUT_ROOT = os.path.join(REPO_ROOT, '..', 'splits', 'oss25_5fold_v1')


def norm_key(s: str) -> str:
    import re
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def find_col(actual_cols: List[str], candidates: List[str]) -> str:
    by_norm = {norm_key(c): c for c in actual_cols}
    for cand in candidates:
        nc = norm_key(cand)
        if nc in by_norm:
            return by_norm[nc]
    return ''


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns to canonical names
    col_map = {}
    for c in df.columns:
        nc = norm_key(c)
        if nc == 'VIDEO':
            col_map[c] = 'VIDEO'
        elif nc == 'STUDENT':
            col_map[c] = 'STUDENT'
        elif nc == 'GROUP':
            col_map[c] = 'GROUP'
        elif nc == 'TIME':
            col_map[c] = 'TIME'
        elif nc in ('GLOBARATINGSCORE','GLOBALRATINGSCORE','GLOBA_RATING_SCORE','GLOBAL_RATING_SCORE'):
            col_map[c] = 'GRS'
        elif nc == 'OSATSRESPECT':
            col_map[c] = 'OSATS_RESPECT'
        elif nc == 'OSATSMOTION':
            col_map[c] = 'OSATS_MOTION'
        elif nc == 'OSATSINSTRUMENT':
            col_map[c] = 'OSATS_INSTRUMENT'
        elif nc == 'OSATSSUTURE':
            col_map[c] = 'OSATS_SUTURE'
        elif nc == 'OSATSFLOW':
            col_map[c] = 'OSATS_FLOW'
        elif nc == 'OSATSKNOWLEDGE':
            col_map[c] = 'OSATS_KNOWLEDGE'
        elif nc == 'OSATSPERFORMANCE':
            col_map[c] = 'OSATS_PERFORMANCE'
        elif nc in ('OSATSFINALQUALITY','OSATS_FINAL_QUALITY','OSATSFINALQUALITY'):
            col_map[c] = 'OSATSFINALQUALITY'
        else:
            # keep original
            pass
    out = df.rename(columns=col_map).copy()
    # ensure required columns exist
    for req in ['VIDEO','STUDENT','GRS']:
        if req not in out.columns:
            out[req] = np.nan
    return out


def read_labels() -> pd.DataFrame:
    # Merge OSATS.xlsx (has GROUP/TIME) and OSATS_MICCAI_trainset.xlsx
    frames = []
    xlsx1 = os.path.join(DATA_ROOT, 'OSATS.xlsx')
    xlsx2 = os.path.join(DATA_ROOT, 'OSATS_MICCAI_trainset.xlsx')
    if os.path.exists(xlsx1):
        df1 = pd.read_excel(xlsx1)
        df1.columns = [str(c).strip() for c in df1.columns]
        frames.append(_normalize_cols(df1))
    if os.path.exists(xlsx2):
        df2 = pd.read_excel(xlsx2)
        df2.columns = [str(c).strip() for c in df2.columns]
        frames.append(_normalize_cols(df2))
    if not frames:
        raise FileNotFoundError('No label Excel files found in data/')
    # outer concat, then keep canonical columns
    df = pd.concat(frames, ignore_index=True, sort=False)
    # keep only canonical columns we'll use
    keep = ['VIDEO','STUDENT','GROUP','TIME','GRS','OSATS_RESPECT','OSATS_MOTION','OSATS_INSTRUMENT','OSATS_SUTURE','OSATS_FLOW','OSATS_KNOWLEDGE','OSATS_PERFORMANCE','OSATSFINALQUALITY']
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    df = df[keep]
    return df


def map_video_exists(df: pd.DataFrame, video_col: str) -> Dict[str, bool]:
    vids = set([str(v).strip() for v in df[video_col].dropna().unique().tolist()])
    existing = {v: os.path.exists(os.path.join(MOVIE_DIR, f"{v}.mp4")) for v in vids}
    return existing


def grs_to_class(v: float) -> int:
    try:
        v = float(v)
    except Exception:
        return -1
    if 8 <= v <= 15: return 0
    if 16 <= v <= 23: return 1
    if 24 <= v <= 31: return 2
    if 32 <= v <= 40: return 3
    return -1


def build_student_vectors(df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    cols = list(df.columns)
    c_VIDEO = find_col(cols, ['VIDEO'])
    c_STUDENT = find_col(cols, ['STUDENT'])
    c_GROUP = find_col(cols, ['GROUP'])
    c_TIME = find_col(cols, ['TIME'])
    c_GRS = find_col(cols, ['GRS','GLOBARATINGSCORE','GLOBALRATINGSCORE','GLOBA_RATING_SCORE','GLOBAL_RATING_SCORE'])
    osats_canon = [
        'OSATS_RESPECT','OSATS_MOTION','OSATS_INSTRUMENT','OSATS_SUTURE',
        'OSATS_FLOW','OSATS_KNOWLEDGE','OSATS_PERFORMANCE','OSATSFINALQUALITY'
    ]
    osats_map = {k: k for k in osats_canon if k in cols}

    # Filter only rows with existing movie files
    exist_map = map_video_exists(df, c_VIDEO)
    df = df[df[c_VIDEO].astype(str).str.strip().map(exist_map.get).fillna(False)].copy()

    # Precompute unique sets
    groups = sorted([g for g in df[c_GROUP].astype(str).str.strip().unique().tolist() if g and g.lower() != 'nan']) if c_GROUP else []
    group_index = {g: i for i, g in enumerate(groups)}

    # Build dimension names
    dim_names: List[str] = []
    # 0) video count
    dim_names.append('N_VIDEOS')
    # 1) GRS classes
    dim_names += [f'GRS_C{i}' for i in range(4)]
    # 2) TIME
    dim_names += ['TIME_PRE', 'TIME_POST']
    # 3) GROUP one-hot
    dim_names += [f'GROUP_{g}' for g in groups]
    # 4) OSATS distributions (per category 1..5)
    for cat in osats_canon:
        for v in range(1, 6):
            dim_names.append(f'{cat}_V{v}')

    D = len(dim_names)
    # Helper to index dims
    idx = {n: i for i, n in enumerate(dim_names)}

    # Video count per student (unique videos)
    vids_per_student: Dict[str, set] = defaultdict(set)

    # Initialize student vectors
    vecs: Dict[str, np.ndarray] = {}
    meta: Dict[str, Dict] = defaultdict(lambda: {'videos': set(), 'groups': Counter(), 'times': Counter()})

    for _, row in df.iterrows():
        stu = str(row[c_STUDENT]).strip()
        vid = str(row[c_VIDEO]).strip()
        grp = str(row[c_GROUP]).strip() if c_GROUP else ''
        tim = str(row[c_TIME]).strip().lower() if c_TIME else ''
        if stu not in vecs:
            vecs[stu] = np.zeros(D, dtype=float)
        v = vecs[stu]

        # count unique videos per student later
        vids_per_student[stu].add(vid)
        meta[stu]['videos'].add(vid)
        if grp:
            meta[stu]['groups'][grp] += 1
        if tim:
            meta[stu]['times'][tim] += 1

        # GRS
        if c_GRS:
            cls = grs_to_class(row[c_GRS])
            if 0 <= cls <= 3:
                v[idx[f'GRS_C{cls}']] += 1.0

        # TIME
        if tim.startswith('pre'):
            v[idx['TIME_PRE']] += 1.0
        elif tim.startswith('post'):
            v[idx['TIME_POST']] += 1.0

        # GROUP
        if grp in group_index:
            v[idx[f'GROUP_{grp}']] += 1.0

        # OSATS
        for cat in osats_canon:
            ccol = osats_map[cat]
            if not ccol:
                continue
            try:
                val = int(round(float(row[ccol])))
            except Exception:
                continue
            if 1 <= val <= 5:
                v[idx[f'{cat}_V{val}']] += 1.0

    # finalize video counts
    for stu, vids in vids_per_student.items():
        if stu not in vecs:
            vecs[stu] = np.zeros(D, dtype=float)
        vecs[stu][idx['N_VIDEOS']] = float(len(vids))

    # convert meta videos set to list
    for stu in meta:
        meta[stu]['videos'] = sorted(list(meta[stu]['videos']))

    # capture dim info
    meta_info = {
        'dim_names': dim_names,
        'groups': groups,
        'osats_canon': osats_canon,
    }

    return (vecs, meta, meta_info)


def build_weights(dim_names: List[str]) -> np.ndarray:
    w = np.ones(len(dim_names), dtype=float)
    for i, n in enumerate(dim_names):
        if n == 'N_VIDEOS':
            w[i] = 1.0
        elif n.startswith('GRS_C'):
            w[i] = 4.0
        elif n.startswith('TIME_'):
            w[i] = 2.0
        elif n.startswith('GROUP_'):
            w[i] = 1.5
        elif '_V' in n:  # OSATS bins
            w[i] = 1.0
        else:
            w[i] = 1.0
    return w


def objective_cost(fold_sums: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    # relative squared error per dim
    denom = np.maximum(target, 1.0)
    diff = (fold_sums - target) / denom
    return float(np.sum(weights * np.sum(diff * diff, axis=0)))


def assign_greedy(vecs: Dict[str, np.ndarray], dim_names: List[str], kfold: int = 5, seed: int = 0, moves: int = 3, restarts: int = 10) -> Dict[str, int]:
    rng = np.random.default_rng(seed)
    students = list(vecs.keys())
    V = np.stack([vecs[s] for s in students], axis=0)
    total = V.sum(axis=0)
    target = total / float(kfold)
    weights = build_weights(dim_names)

    # order by weighted norm desc
    norms = (np.abs(V) * weights).sum(axis=1)
    order = list(np.argsort(-norms))

    best_assign = None
    best_cost = float('inf')

    for r in range(restarts):
        # randomize slight perturbation of order
        rng.shuffle(order)
        assign = -np.ones(len(students), dtype=int)
        fold_sums = np.zeros((kfold, V.shape[1]), dtype=float)
        # initial greedy placement
        for idx_i in order:
            v = V[idx_i]
            best_f = None
            best_delta = None
            for f in range(kfold):
                fold_sums[f] += v
                cost = objective_cost(fold_sums, target, weights)
                fold_sums[f] -= v
                if best_delta is None or cost < best_delta:
                    best_delta = cost
                    best_f = f
            assign[idx_i] = best_f
            fold_sums[best_f] += v

        # local improvement by moves
        improved = True
        pass_no = 0
        while improved and pass_no < moves:
            improved = False
            pass_no += 1
            for idx_i in rng.permutation(len(students)):
                cur_f = assign[idx_i]
                v = V[idx_i]
                current_cost = objective_cost(fold_sums, target, weights)
                best_f = cur_f
                best_cost_local = current_cost
                for f in range(kfold):
                    if f == cur_f: continue
                    fold_sums[cur_f] -= v
                    fold_sums[f] += v
                    new_cost = objective_cost(fold_sums, target, weights)
                    fold_sums[f] -= v
                    fold_sums[cur_f] += v
                    if new_cost + 1e-9 < best_cost_local:
                        best_cost_local = new_cost
                        best_f = f
                if best_f != cur_f:
                    fold_sums[cur_f] -= v
                    fold_sums[best_f] += v
                    assign[idx_i] = best_f
                    improved = True

        final_cost = objective_cost(fold_sums, target, weights)
        if final_cost < best_cost:
            best_cost = final_cost
            best_assign = assign.copy()

    # build mapping
    mapping = {students[i]: int(best_assign[i]) for i in range(len(students))}
    return mapping


def save_outputs(mapping: Dict[str, int], df_main: pd.DataFrame, dim_info: Dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Canonical columns standardized by read_labels()
    c_VIDEO = 'VIDEO'
    c_STUDENT = 'STUDENT'
    c_GROUP = 'GROUP'
    c_TIME = 'TIME'
    c_GRS = 'GRS'
    # aggregate to unique video rows with student's fold
    base_cols = [c_VIDEO, c_STUDENT, c_TIME, c_GROUP, c_GRS]
    sel_cols = [c for c in base_cols if c]
    df = df_main[sel_cols].drop_duplicates(subset=[c_VIDEO, c_STUDENT]).copy()
    # ensure required columns exist
    for req in [c_TIME, c_GROUP, c_GRS]:
        if req and req not in df.columns:
            df[req] = np.nan
    df['FOLD'] = df[c_STUDENT].map(mapping)
    df['GRS_CLASS'] = df[c_GRS].map(grs_to_class)
    df['TIME_N'] = df[c_TIME].astype(str).str.strip().str.lower()
    df['TIME_N'] = df['TIME_N'].replace({'pre ': 'pre', 'post ': 'post'})

    # Merge in additional videos from OSATS_MICCAI_trainset.xlsx (if present)
    extra_path = os.path.join(DATA_ROOT, 'OSATS_MICCAI_trainset.xlsx')
    if os.path.exists(extra_path):
        df2 = pd.read_excel(extra_path)
        df2.columns = [str(c).strip() for c in df2.columns]
        cols2 = list(df2.columns)
        v2 = find_col(cols2, ['VIDEO'])
        s2 = find_col(cols2, ['STUDENT'])
        t2 = find_col(cols2, ['TIME'])  # may not exist
        g2 = find_col(cols2, ['GROUP']) # may not exist
        r2 = find_col(cols2, ['GLOBARATINGSCORE','GLOBALRATINGSCORE','GLOBA_RATING_SCORE','GLOBAL_RATING_SCORE'])
        df2 = df2[[c for c in [v2, s2, t2, g2, r2] if c]].drop_duplicates(subset=[v2, s2])
        # restrict to students in mapping
        df2 = df2[df2[s2].astype(str).str.strip().isin(mapping.keys())].copy()
        # align columns to main names
        df2 = df2.rename(columns={v2: c_VIDEO, s2: c_STUDENT})
        if t2: df2 = df2.rename(columns={t2: c_TIME})
        if g2: df2 = df2.rename(columns={g2: c_GROUP})
        if r2: df2 = df2.rename(columns={r2: c_GRS})
        df2['FOLD'] = df2[c_STUDENT].map(mapping)
        df2['GRS_CLASS'] = df2[c_GRS].map(grs_to_class) if c_GRS in df2.columns else -1
        if c_TIME in df2.columns:
            df2['TIME_N'] = df2[c_TIME].astype(str).str.strip().str.lower().replace({'pre ': 'pre', 'post ': 'post'})
        else:
            df2['TIME_N'] = 'unknown'
        # union (prefer main for overlapping videos)
        df = pd.concat([df, df2[~df2[c_VIDEO].isin(df[c_VIDEO])]], ignore_index=True)

    # drop videos not present in movie dir
    exist_map = map_video_exists(df, c_VIDEO)
    videos = df[df[c_VIDEO].astype(str).str.strip().map(exist_map.get).fillna(False)].copy()

    # per fold lists
    for f in sorted(videos['FOLD'].unique().tolist()):
        fold_df = videos[videos['FOLD'] == f]
        fold_df[[c_VIDEO, c_STUDENT, 'FOLD', 'TIME_N', 'GRS_CLASS']].to_csv(
            os.path.join(out_dir, f'fold_{f}.csv'), index=False
        )
        # train/val split for this fold
        val_videos = sorted(fold_df[c_VIDEO].astype(str).str.strip().tolist())
        train_videos = sorted(videos[videos['FOLD'] != f][c_VIDEO].astype(str).str.strip().tolist())
        with open(os.path.join(out_dir, f'fold_{f}_val_videos.txt'), 'w') as ftxt:
            ftxt.write('\n'.join(val_videos))
        with open(os.path.join(out_dir, f'fold_{f}_train_videos.txt'), 'w') as ftxt:
            ftxt.write('\n'.join(train_videos))

    # Global assignment per student
    df_students = pd.DataFrame({
        'STUDENT': list(mapping.keys()),
        'FOLD': [mapping[s] for s in mapping]
    })
    df_students.to_csv(os.path.join(out_dir, 'students_folds.csv'), index=False)

    # Summary markdown
    lines: List[str] = []
    lines.append('# Split Summary (oss25_5fold_v1)')
    lines.append('')
    lines.append(f'- Students: {df_students.shape[0]}')
    lines.append(f'- Videos: {videos.shape[0]}')
    # folds counts
    lines.append('')
    lines.append('## Per-fold counts')
    for f in sorted(videos['FOLD'].unique().tolist()):
        fold_df = videos[videos['FOLD'] == f]
        lines.append(f'- Fold {f}: videos={fold_df.shape[0]}, students={fold_df[c_STUDENT].nunique()}')
    # distributions
    lines.append('')
    lines.append('## GRS class distribution per fold (videos)')
    for f in sorted(videos['FOLD'].unique().tolist()):
        fold_df = videos[videos['FOLD'] == f]
        cnt = fold_df['GRS_CLASS'].value_counts().sort_index().to_dict()
        lines.append(f'- Fold {f}: {cnt}')
    lines.append('')
    lines.append('## TIME distribution per fold (videos)')
    for f in sorted(videos['FOLD'].unique().tolist()):
        fold_df = videos[videos['FOLD'] == f]
        cnt = fold_df['TIME_N'].value_counts().to_dict()
        lines.append(f'- Fold {f}: {cnt}')

    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines))

    # Save config
    config = {
        'kfold': 5,
        'dim_names': dim_info['dim_names'],
        'weights_rule': {'GRS_C*': 4.0, 'TIME_*': 2.0, 'GROUP_*': 1.5, 'OSATS_*_V*': 1.0, 'N_VIDEOS': 1.0},
        'notes': 'Student-level group split optimized for label balance across 5 folds.'
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--restarts', type=int, default=20)
    parser.add_argument('--moves', type=int, default=4)
    parser.add_argument('--out', default=OUT_ROOT)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = read_labels()
    vecs, meta, dim_info = build_student_vectors(df)
    mapping = assign_greedy(vecs, dim_info['dim_names'], kfold=args.kfold, seed=args.seed, moves=args.moves, restarts=args.restarts)
    save_outputs(mapping, df, dim_info, args.out)
    print('Saved splits to', args.out)


if __name__ == '__main__':
    main()
