#!/usr/bin/env python3
import os
import json
import math
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.abspath(os.path.join(ROOT, '../data'))
OUT_DIR = os.path.join(ROOT, 'outputs')


def _norm_key(s: str) -> str:
    import re
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _find_col(actual_cols: List[str], candidates: List[str]) -> str:
    by_norm = {_norm_key(c): c for c in actual_cols}
    for cand in candidates:
        nc = _norm_key(cand)
        if nc in by_norm:
            return by_norm[nc]
    return ''


def load_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


OSATS_CANON = [
    'OSATS_RESPECT','OSATS_MOTION','OSATS_INSTRUMENT','OSATS_SUTURE',
    'OSATS_FLOW','OSATS_KNOWLEDGE','OSATS_PERFORMANCE','OSATSFINALQUALITY'
]

OSATS_ALIASES: Dict[str, List[str]] = {
    'OSATSFINALQUALITY': ['OSATS_FINAL_QUALITY','OSATS FINAL QUALITY','OSATS-FINAL-QUALITY']
}


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)
    res = {}
    res['VIDEO'] = _find_col(cols, ['VIDEO'])
    res['STUDENT'] = _find_col(cols, ['STUDENT'])
    res['TIME'] = _find_col(cols, ['TIME', 'SESSION'])
    res['GRS'] = _find_col(cols, ['GLOBARATINGSCORE','GLOBALRATINGSCORE','GLOBA_RATING_SCORE','GLOBAL_RATING_SCORE'])
    for k in OSATS_CANON:
        cand = [k] + OSATS_ALIASES.get(k, [])
        res[k] = _find_col(cols, cand)
    return res


def to_numeric(s):
    try:
        return pd.to_numeric(s, errors='coerce')
    except Exception:
        return pd.Series([np.nan] * len(s))


def corr_analysis_video(df: pd.DataFrame) -> Dict:
    cols = resolve_columns(df)
    vid = cols['VIDEO']
    grs = cols['GRS']
    keep_cols = [vid, grs] + [cols[k] for k in OSATS_CANON if cols[k]]
    df2 = df[keep_cols].copy()
    # numeric
    df2[grs] = to_numeric(df2[grs])
    for k in OSATS_CANON:
        c = cols[k]
        if c:
            df2[c] = to_numeric(df2[c])
    # aggregate per video (mean of raters)
    agg = df2.groupby(vid, dropna=True).mean(numeric_only=True)
    # Pearson and Spearman with GRS
    pearson = {}
    spearman = {}
    for k in OSATS_CANON:
        c = cols[k]
        if not c or c not in agg.columns:
            continue
        try:
            pearson[k] = float(agg[c].corr(agg[grs], method='pearson'))
            spearman[k] = float(agg[c].corr(agg[grs], method='spearman'))
        except Exception:
            pass
    # Inter-OSATS correlation (Pearson)
    osats_real = [cols[k] for k in OSATS_CANON if cols[k] in agg.columns]
    inter = agg[osats_real].corr(method='pearson') if osats_real else pd.DataFrame()
    # Heatmap for OSATS vs GRS (Pearson)
    try:
        vals = [pearson.get(k, np.nan) for k in OSATS_CANON]
        fig, ax = plt.subplots(figsize=(6, 1.8))
        im = ax.imshow([vals], cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_yticks([0]); ax.set_yticklabels(['Pearson'])
        ax.set_xticks(list(range(len(OSATS_CANON))))
        ax.set_xticklabels(OSATS_CANON, rotation=45, ha='right', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        os.makedirs(OUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUT_DIR, 'corr_osats_vs_grs_pearson.png'), dpi=150)
        plt.close(fig)
    except Exception:
        pass
    # Inter-OSATS heatmap
    try:
        if not inter.empty:
            fig, ax = plt.subplots(figsize=(4.5, 4))
            im = ax.imshow(inter.values, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(list(range(len(inter.columns))))
            ax.set_yticks(list(range(len(inter.index))))
            ax.set_xticklabels([k.replace('OSATS_', '') for k in inter.columns], rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels([k.replace('OSATS_', '') for k in inter.index], fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, 'corr_inter_osats_pearson.png'), dpi=150)
            plt.close(fig)
    except Exception:
        pass
    return {
        'n_videos': int(agg.shape[0]),
        'pearson_osats_vs_grs': pearson,
        'spearman_osats_vs_grs': spearman,
        'inter_osats_pearson_matrix': inter.to_dict() if not inter.empty else {},
    }


def pre_post_analysis(df: pd.DataFrame) -> Dict:
    cols = resolve_columns(df)
    student = cols['STUDENT']
    time = cols['TIME']
    grs = cols['GRS']
    if not (student and time and grs and student in df.columns and time in df.columns and grs in df.columns):
        return {
            'available': False,
            'reason': 'Required columns missing (need STUDENT, TIME, GRS)'
        }
    # normalize time values
    d = df[[student, time, grs] + [cols[k] for k in OSATS_CANON if cols[k]]].copy()
    d[grs] = to_numeric(d[grs])
    for k in OSATS_CANON:
        c = cols[k]
        if c:
            d[c] = to_numeric(d[c])
    tnorm = d[time].astype(str).str.strip().str.lower().replace({'pre ': 'pre', 'post ': 'post'})
    d[time] = tnorm
    # aggregate by student, time
    agg = d.groupby([student, time]).mean(numeric_only=True).reset_index()
    # pivot to get pre/post columns for GRS and OSATS
    def pivot_metric(col):
        piv = agg.pivot(index=student, columns=time, values=col)
        return piv
    res = {}
    # GRS delta
    piv_grs = pivot_metric(grs)
    if 'pre' in piv_grs.columns and 'post' in piv_grs.columns:
        delta_grs = (piv_grs['post'] - piv_grs['pre']).dropna()
        res['n_students_with_pre_post'] = int(delta_grs.shape[0])
        res['grs_delta_stats'] = {
            'mean': float(delta_grs.mean()),
            'median': float(delta_grs.median()),
            'min': float(delta_grs.min()),
            'max': float(delta_grs.max()),
        }
        # boxplot
        try:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.boxplot([piv_grs['pre'].dropna().values, piv_grs['post'].dropna().values], labels=['pre', 'post'])
            ax.set_title('GRS pre vs post')
            plt.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, 'grs_pre_post_box.png'), dpi=150)
            plt.close(fig)
        except Exception:
            pass
        # correlate OSATS deltas with GRS delta
        osats_delta_corr = {}
        for k in OSATS_CANON:
            c = cols[k]
            if not c or c not in agg.columns:
                continue
            piv = pivot_metric(c)
            if 'pre' in piv.columns and 'post' in piv.columns:
                delta = (piv['post'] - piv['pre']).reindex(delta_grs.index)
                try:
                    osats_delta_corr[k] = float(delta.corr(delta_grs, method='pearson'))
                except Exception:
                    pass
        res['osats_delta_vs_grs_delta_pearson'] = osats_delta_corr
    else:
        res['n_students_with_pre_post'] = 0
        res['grs_delta_stats'] = {}
        res['osats_delta_vs_grs_delta_pearson'] = {}
    res['available'] = True
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # 1) Correlations using MICCAI_trainset.xlsx (per video mean)
    df1 = load_xlsx(os.path.join(DATA_ROOT, 'OSATS_MICCAI_trainset.xlsx'))
    corr1 = corr_analysis_video(df1)
    # 2) Pre/Post analysis using OSATS.xlsx (has TIME column)
    df2 = load_xlsx(os.path.join(DATA_ROOT, 'OSATS.xlsx'))
    prepost = pre_post_analysis(df2)
    # save JSON
    out = {'corr_video': corr1, 'pre_post': prepost}
    with open(os.path.join(OUT_DIR, 'relations_analysis.json'), 'w') as f:
        json.dump(out, f, indent=2)
    # save Markdown summary
    lines = []
    lines.append('# Relations Analysis Summary')
    lines.append('')
    lines.append('## Correlations (per-video means)')
    lines.append(f"Videos analyzed: {corr1['n_videos']}")
    lines.append('### Pearson OSATS vs GRS')
    for k in OSATS_CANON:
        v = corr1['pearson_osats_vs_grs'].get(k, None)
        if v is not None:
            lines.append(f'- {k}: {v:.3f}')
    lines.append('')
    lines.append('### Spearman OSATS vs GRS')
    for k in OSATS_CANON:
        v = corr1['spearman_osats_vs_grs'].get(k, None)
        if v is not None:
            lines.append(f'- {k}: {v:.3f}')
    lines.append('')
    lines.append('Artifacts: corr_osats_vs_grs_pearson.png, corr_inter_osats_pearson.png')
    lines.append('')
    lines.append('## Pre/Post Analysis (by student)')
    if prepost.get('available'):
        lines.append(f"Students with pre+post: {prepost.get('n_students_with_pre_post', 0)}")
        stats = prepost.get('grs_delta_stats', {})
        if stats:
            lines.append(f"GRS delta (post-pre): mean={stats.get('mean'):.3f}, median={stats.get('median'):.3f}, min={stats.get('min'):.3f}, max={stats.get('max'):.3f}")
        else:
            lines.append('GRS delta stats: N/A')
        lines.append('### Correlation of OSATS delta with GRS delta (Pearson)')
        for k in OSATS_CANON:
            v = prepost.get('osats_delta_vs_grs_delta_pearson', {}).get(k, None)
            if v is not None:
                lines.append(f'- {k}: {v:.3f}')
        lines.append('Artifacts: grs_pre_post_box.png')
    else:
        lines.append(f"Pre/Post analysis unavailable: {prepost.get('reason')}")
    with open(os.path.join(OUT_DIR, 'relations_analysis.md'), 'w') as f:
        f.write('\n'.join(lines))
    print('Saved relations analysis artifacts to', OUT_DIR)


if __name__ == '__main__':
    main()

