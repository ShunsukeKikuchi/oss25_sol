#!/usr/bin/env python3
import json
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(ROOT, 'outputs')
IN_JSON = os.path.join(OUT_DIR, 'initial_eda_summary.json')

def load_summary(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_grs(grs_counts, out_path):
    keys = ['0','1','2','3']
    vals = [grs_counts.get(k, 0) for k in keys]
    plt.figure(figsize=(4,3))
    bars = plt.bar(keys, vals, color=['#4C78A8','#F58518','#54A24B','#E45756'])
    for b, v in zip(bars, vals):
        plt.text(b.get_x()+b.get_width()/2, b.get_height(), str(v), ha='center', va='bottom', fontsize=9)
    plt.title('GRS class distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_osats(osats_stats, out_path):
    cats = [
        'OSATS_RESPECT','OSATS_MOTION','OSATS_INSTRUMENT','OSATS_SUTURE',
        'OSATS_FLOW','OSATS_KNOWLEDGE','OSATS_PERFORMANCE','OSATSFINALQUALITY'
    ]
    n = len(cats)
    cols = 4
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*2.6))
    axes = axes.flatten()
    for i, cat in enumerate(cats):
        ax = axes[i]
        stats = osats_stats.get(cat, {})
        counts = stats.get('counts', {})
        keys = ['1','2','3','4','5']
        vals = [counts.get(k, 0) for k in keys]
        ax.bar(keys, vals, color='#4C78A8')
        ax.set_title(cat, fontsize=9)
        ax.set_ylim(0, max(vals)+10 if vals else 10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    # hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    fig.suptitle('OSATS distributions', y=0.98)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_summary(IN_JSON)
    labels = data.get('labels', {})
    # GRS
    grs_counts = labels.get('grs_class_counts', {})
    if grs_counts:
        plot_grs(grs_counts, os.path.join(OUT_DIR, 'grs_class_distribution.png'))
    # OSATS
    osats_stats = labels.get('osats_stats', {})
    if osats_stats:
        plot_osats(osats_stats, os.path.join(OUT_DIR, 'osats_distributions.png'))
    print('Saved plots to', OUT_DIR)

if __name__ == '__main__':
    main()

