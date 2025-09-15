#!/usr/bin/env python3
import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

import numpy as np
from PIL import Image

# Ensure local sam2 package is importable
import torch
try:
    from torch._inductor import config as inductor_config
    inductor_config.max_autotune = False
    if hasattr(inductor_config, 'triton'):
        inductor_config.triton.cudagraphs = False
    if hasattr(inductor_config, 'cudagraphs'):
        inductor_config.cudagraphs = False
except Exception:
    pass
# Add inner package path to avoid build guard complaint when not pip-installed
PKG_INNER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sam2', 'sam2'))
if os.path.isdir(PKG_INNER):
    sys.path.insert(0, PKG_INNER)
from sam2.build_sam import build_sam2_video_predictor

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
MOVIE_DIR = os.path.join(DATA_ROOT, 'movie')
MASKS_DIR = os.path.join(DATA_ROOT, 'train', 'masks')
ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
DONE_MARK = '_DONE'
LOCK_MARK = '.lock'

# Class mapping per challenge spec
# 0: Left Hand, 1: Right Hand, 2: Scissors, 3: Tweezers, 4: Needle Holder, 5: Needle
TOOL_NAME_TO_CLASS = {
    'left_hand': 0,
    'right_hand': 1,
    'scissors': 2,
    'tweezers': 3,
    'needle_holder': 4,
    'needleholder': 4,
    'needle-holder': 4,
    'needle': 5,
}


def parse_mask_filename(fn: str) -> Tuple[str, int, str]:
    # e.g., A36O_frame_0_tweezers_mask.png
    #        <video>_frame_<frame>_<tool><suffix>_mask.png
    m = re.match(r"^(?P<vid>[^_]+)_frame_(?P<frame>\d+)_(?P<tool>.+)_mask\.png$", fn)
    if not m:
        return '', -1, ''
    vid = m.group('vid')
    frame = int(m.group('frame'))
    tool = m.group('tool')
    return vid, frame, tool


def norm_tool(tool: str) -> Tuple[str, str]:
    t = tool.lower()
    inst = ''
    if t.startswith('needle1'):
        t = 'needle'
        inst = '1'
    # normalize separators
    t = t.replace(' ', '_').replace('-', '_')
    return t, inst


def tool_to_class(tool: str) -> int:
    t, _ = norm_tool(tool)
    # try direct mapping, then contains logic
    if t in TOOL_NAME_TO_CLASS:
        return TOOL_NAME_TO_CLASS[t]
    for k, v in TOOL_NAME_TO_CLASS.items():
        if k in t:
            return v
    return -1


def collect_annotations(video_id: str) -> Dict[Tuple[int, str], List[Tuple[int, str]]]:
    # key: (class_id, instance_suffix) -> list of (frame_idx, mask_path)
    by_obj: Dict[Tuple[int, str], List[Tuple[int, str]]] = defaultdict(list)
    for fn in os.listdir(MASKS_DIR):
        if not fn.startswith(f"{video_id}_frame_") or not fn.endswith('_mask.png'):
            continue
        vid, frame_idx, tool = parse_mask_filename(fn)
        if vid != video_id:
            continue
        cls = tool_to_class(tool)
        if cls < 0:
            continue
        norm, inst = norm_tool(tool)
        by_obj[(cls, inst)].append((frame_idx, os.path.join(MASKS_DIR, fn)))
    # sort lists by frame index
    for k in by_obj:
        by_obj[k].sort(key=lambda x: x[0])
    return by_obj


def read_mask_bool(path: str) -> np.ndarray:
    im = Image.open(path).convert('L')
    arr = np.array(im)
    return arr >= 128


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_indexed_png(path: str, arr: np.ndarray):
    # uint8 labels
    im = Image.fromarray(arr.astype(np.uint8), mode='L')
    im.save(path)


def build_predictor(config: str, ckpt: str, device: str = 'cuda', vos_optimized: bool = True):
    predictor = build_sam2_video_predictor(
        config_file=config,
        ckpt_path=ckpt,
        device=device,
        mode='eval',
        vos_optimized=vos_optimized,
        # add all frames receiving masks as conditioning frames
        hydra_overrides_extra=["++model.add_all_frames_to_correct_as_cond=true"],
    )
    return predictor


def _acquire_lock(dir_path: str) -> bool:
    os.makedirs(dir_path, exist_ok=True)
    lock_path = os.path.join(dir_path, LOCK_MARK)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False


def _release_lock(dir_path: str):
    lock_path = os.path.join(dir_path, LOCK_MARK)
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
        except Exception:
            pass


def ensure_fullfps_jpg_dir(video_path: str, video_id: str) -> str:
    out_dir = os.path.join(ARTIFACTS_DIR, 'sam2_fullfps_frames', video_id)
    # check if frames exist
    if not (os.path.isdir(out_dir) and any(f.lower().endswith(('.jpg', '.jpeg')) for f in os.listdir(out_dir))):
        ensure_dir(out_dir)
        pattern = os.path.join(out_dir, '%05d.jpg')
        import subprocess
        cmd = [
            'ffmpeg','-y','-hide_banner','-loglevel','error',
            '-i', video_path,
            '-q:v','2',
            '-start_number','0',
            pattern
        ]
        subprocess.check_call(cmd)
    return out_dir


def _ffprobe_fps(mp4_path: str) -> float:
    try:
        import subprocess, json as _json
        cmd = [
            'ffprobe','-v','error',
            '-select_streams','v:0',
            '-show_entries','stream=avg_frame_rate',
            '-of','json', mp4_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = _json.loads(out.decode('utf-8','ignore'))
        rate = data['streams'][0]['avg_frame_rate']
        if '/' in rate:
            a,b = rate.split('/')
            return float(a)/float(b) if float(b)!=0 else 0.0
        return float(rate)
    except Exception:
        return 0.0


def propagate_multi_cond(video_path: str, video_id: str, annotations: Dict[Tuple[int, str], List[Tuple[int, str]]],
                         predictor, out6_dir: str, out3_dir: str, fps_down: int = 0, fps_orig: float = 0.0):
    H = W = None
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        # Try direct video init; fallback to pre-extracted JPG frames if decord fails
        try:
            state = predictor.init_state(video_path, offload_video_to_cpu=True, offload_state_to_cpu=False)
        except Exception as e:
            print(f"[warn] decord failed on {video_id} ({e}); falling back to JPG frames")
            jpg_dir = ensure_fullfps_jpg_dir(video_path, video_id)
            state = predictor.init_state(jpg_dir, offload_video_to_cpu=True, offload_state_to_cpu=False)
        # frame index mapper when using fps-downsampled JPEG input
        def map_idx(i: int) -> int:
            if fps_down and fps_orig and state.get('num_frames', 0) > 0:
                j = int(round(i * (fps_down / fps_orig)))
                j = max(0, min(j, state['num_frames'] - 1))
                return j
            return i
        # add per-object masks at their annotated frames
        for (cls, inst), items in annotations.items():
            obj_id = f"cls{cls}_inst{inst or '0'}"
            for frame_idx, mask_path in items:
                mbool = read_mask_bool(mask_path)
                if H is None:
                    H, W = mbool.shape
                predictor.add_new_mask(state, frame_idx=map_idx(frame_idx), obj_id=obj_id, mask=mbool)
        # precompute nearest-annotation distance per class for temporal weighting
        class_ann_frames: Dict[int, List[int]] = defaultdict(list)
        for (cls, _), items in annotations.items():
            for fidx, _ in items:
                class_ann_frames[cls].append(fidx)
        for c in class_ann_frames:
            class_ann_frames[c] = sorted(set(class_ann_frames[c]))

        # propagate (forward and reverse) with simple per-object overwrite like minimal example
        frame_count = 0
        # forward pass
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            h = out_mask_logits.shape[-2]
            w = out_mask_logits.shape[-1]
            six = np.zeros((h, w), dtype=np.uint8)
            three = np.zeros((h, w), dtype=np.uint8)
            for i, oid in enumerate(out_obj_ids):
                # binarize per-object mask at 0-logit threshold and overwrite labels
                mask_bin = (out_mask_logits[i, 0] > 0).detach().cpu().numpy()
                m = re.match(r"^cls(\d+)_inst(\d+)$", oid)
                if not m:
                    continue
                c = int(m.group(1))
                six[mask_bin] = c + 1  # 1..6, 0 is background
                if c in (0, 1):
                    three[mask_bin] = 1
                elif c in (2, 3, 4, 5):
                    three[mask_bin] = 2
            ensure_dir(out6_dir)
            ensure_dir(out3_dir)
            six_path = os.path.join(out6_dir, f"{video_id}_frame_{out_frame_idx}.png")
            three_path = os.path.join(out3_dir, f"{video_id}_frame_{out_frame_idx}.png")
            # resume-friendly: skip if already exists
            if not os.path.exists(six_path):
                save_indexed_png(six_path, six)
            if not os.path.exists(three_path):
                save_indexed_png(three_path, three)
            frame_count += 1
        # reverse pass (optional, improves coverage)
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state, reverse=True):
            h = out_mask_logits.shape[-2]
            w = out_mask_logits.shape[-1]
            six = np.zeros((h, w), dtype=np.uint8)
            three = np.zeros((h, w), dtype=np.uint8)
            for i, oid in enumerate(out_obj_ids):
                mask_bin = (out_mask_logits[i, 0] > 0).detach().cpu().numpy()
                m = re.match(r"^cls(\d+)_inst(\d+)$", oid)
                if not m:
                    continue
                c = int(m.group(1))
                six[mask_bin] = c + 1
                if c in (0, 1):
                    three[mask_bin] = 1
                elif c in (2, 3, 4, 5):
                    three[mask_bin] = 2
            six_path = os.path.join(out6_dir, f"{video_id}_frame_{out_frame_idx}.png")
            three_path = os.path.join(out3_dir, f"{video_id}_frame_{out_frame_idx}.png")
            if not os.path.exists(six_path):
                save_indexed_png(six_path, six)
            if not os.path.exists(three_path):
                save_indexed_png(three_path, three)
            frame_count += 1
        return frame_count


def main():
    ap = argparse.ArgumentParser(description='Run SAM2 propagation with multi-annotation fusion')
    ap.add_argument('--video', help='VIDEO id (without .mp4)')
    ap.add_argument('--list', help='Text file with VIDEO ids (one per line)')
    ap.add_argument('--config', required=True, help='Path to SAM2 config yaml (e.g., sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml)')
    ap.add_argument('--ckpt', required=True, help='Path to SAM2 checkpoint .pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out-base', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts')))
    ap.add_argument('--fps', type=int, default=0, help='If >0, downsample video to this fps via JPEG frames before propagation')
    args = ap.parse_args()

    if not args.video and not args.list:
        ap.error('Specify --video or --list')

    if args.list:
        with open(args.list) as f:
            videos = [l.strip() for l in f if l.strip()]
    else:
        videos = [args.video]

    out6_root = os.path.join(args.out_base, 'seg_pseudo6')
    out3_root = os.path.join(args.out_base, 'seg_pseudo3')
    os.makedirs(out6_root, exist_ok=True)
    os.makedirs(out3_root, exist_ok=True)

    predictor = build_predictor(args.config, args.ckpt, device=args.device, vos_optimized=False)

    for vid in videos:
        video_path = os.path.join(MOVIE_DIR, f'{vid}.mp4')
        if not os.path.exists(video_path):
            print(f"[skip] missing video: {video_path}")
            continue
        ann = collect_annotations(vid)
        if not ann:
            print(f"[skip] no TaskC masks found for {vid} in {MASKS_DIR} — skipping propagation")
            continue
        out6_dir = os.path.join(out6_root, vid)
        out3_dir = os.path.join(out3_root, vid)
        # if fps downsampling requested, pre-extract  to JPEGs and use the folder as input
        input_path = video_path
        if args.fps and args.fps > 0:
            # extract only once per video
            fps_dir = os.path.join(ARTIFACTS_DIR, f'sam2_frames_{args.fps}fps', vid)
            if not (os.path.isdir(fps_dir) and any(fn.lower().endswith(('.jpg','.jpeg')) for fn in os.listdir(fps_dir))):
                os.makedirs(fps_dir, exist_ok=True)
                pattern = os.path.join(fps_dir, '%05d.jpg')
                import subprocess
                cmd = [
                    'ffmpeg','-y','-hide_banner','-loglevel','error',
                    '-i', video_path,
                    '-vf', f'fps={args.fps}',
                    '-q:v','2',
                    '-start_number','0',
                    pattern
                ]
                subprocess.check_call(cmd)
            input_path = fps_dir
        # skip if already completed
        if os.path.exists(os.path.join(out6_dir, DONE_MARK)):
            print(f"[skip] {vid}: DONE mark exists -> {out6_dir}")
            continue
        # acquire lock to be multi-process safe
        if not _acquire_lock(out6_dir):
            print(f"[lock] {vid}: another process is using {out6_dir}; skipping")
            continue
        try:
            src_desc = input_path if os.path.isdir(input_path) else video_path
            print(f"[run] {vid}: {len(ann)} objects; input={src_desc} -> out={out6_dir}")
            # determine original fps if downsampled
            fps_orig = 0.0
            if args.fps and args.fps > 0:
                mp4_src = os.path.join(MOVIE_DIR, f'{vid}.mp4')
                fps_orig = _ffprobe_fps(mp4_src)
            n = propagate_multi_cond(input_path, vid, ann, predictor, out6_dir, out3_dir, fps_down=args.fps, fps_orig=fps_orig)
            # mark done
            with open(os.path.join(out6_dir, DONE_MARK), 'w') as f:
                f.write('ok')
            print(f"  frames saved: {n}")
        finally:
            _release_lock(out6_dir)


if __name__ == '__main__':
    main()
