#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import json
import math
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

try:
    import decord
    from decord import VideoReader
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

# Fallback: OpenCV-based reader
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# Simple wrappers to unify frame access
class _DecordVRWrapper:
    def __init__(self, path: str):
        self.vr = VideoReader(path)
        self.length = len(self.vr)
        try:
            fps = float(self.vr.get_avg_fps())
            self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        except Exception:
            self.fps = 30.0
    def __len__(self):
        return self.length
    def get(self, i: int):
        return self.vr[int(i)].asnumpy()

class _CV2VRWrapper:
    def __init__(self, path: str):
        if not _HAS_CV2:
            raise RuntimeError('opencv not available')
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError('cv2 failed to open video')
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n is None or n <= 0:
            # Fallback probing (rare)
            frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                frames += 1
            n = frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap = cap
        self.length = max(0, int(n))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
    def __len__(self):
        return self.length
    def get(self, i: int):
        i = max(0, min(self.length - 1, int(i)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Return a tiny black frame to avoid hard crash; caller will letterbox anyway
            return np.zeros((8, 8, 3), dtype=np.uint8)
        # BGR->RGB
        return frame[:, :, ::-1]


def open_video_reader(video_path: str):
    # Prefer decord if available and works
    if _HAS_DECORD:
        try:
            vr = _DecordVRWrapper(video_path)
            return vr, len(vr), vr.fps
        except Exception:
            pass
    # Fallback to OpenCV
    if _HAS_CV2:
        vr = _CV2VRWrapper(video_path)
        return vr, len(vr), vr.fps
    raise RuntimeError('No suitable video backend available (decord/cv2)')

from .models.swin3d_aux import Swin3DWithSegAux


class EnsembleSwin3D(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.return_key = 'main'
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*R,C,T,H,W) -> returns (B*R,D) averaged over models
        outs = []
        for m in self.models:
            y = m(x)
            y = y[self.return_key] if isinstance(y, dict) else y
            outs.append(y)
        y_stack = torch.stack(outs, dim=0)  # (M,B*R,D)
        return y_stack.mean(dim=0)


def resize_square_pil(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BILINEAR)


def normalize_clip(clip_np: np.ndarray) -> torch.Tensor:
    # clip_np: (T,H,W,3) in [0,255]
    clip = torch.from_numpy(clip_np.astype(np.float32) / 255.0).permute(3, 0, 1, 2)  # C,T,H,W
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)
    clip = (clip - mean) / std
    return clip


def even_sample_indices(length: int, n: int, deterministic: bool) -> List[int]:
    if length <= 0:
        return []
    if n <= 0:
        return []
    if deterministic or length >= n:
        xs = np.linspace(0, max(0, length - 1), num=n)
        idx = [int(round(x)) for x in xs]
    else:
        # length < n: upsample with repeats, jitter lightly
        base = list(range(length))
        idx = []
        while len(idx) < n:
            idx.extend(base)
        idx = idx[:n]
        # small jitter within valid range
        jitter = np.random.randint(low=-1, high=2, size=n)
        idx = [min(length - 1, max(0, i + int(j))) for i, j in zip(idx, jitter)]
    return idx


def decode_video_frames_decord(video_path: str):
    # Deprecated in favor of open_video_reader; keep for backward compatibility
    return open_video_reader(video_path)


def build_clip_from_indices(vr, indices: List[int], image_size: int) -> torch.Tensor:
    frames_np = []
    L = len(vr)
    for i in indices:
        i0 = max(0, min(L - 1, int(i)))
        frame = vr.get(i0)  # H,W,3 uint8 RGB (wrapper handles backend)
        img = Image.fromarray(frame)
        img_sq = resize_square_pil(img, image_size)
        frames_np.append(np.array(img_sq, dtype=np.uint8))
    clip_np = np.stack(frames_np, axis=0)  # T,H,W,3
    clip = normalize_clip(clip_np)
    return clip


@torch.inference_mode()
def infer_one_video(model: torch.nn.Module,
                    video_path: str,
                    task: str,
                    tta: int,
                    device: torch.device,
                    n_frames: int,
                    image_size: int,
                    model_mode: str = 'model2',
                    last5s_sec: int = 5,
                    seed_base: int = 12345) -> np.ndarray:
    vr, length, fps = decode_video_frames_decord(video_path)
    preds = []
    vid_key = os.path.splitext(os.path.basename(video_path))[0]
    for rep in range(tta):
        seed = seed_base + rep * 997 + (hash(vid_key) & 0x7fffffff)
        random.seed(seed)
        np.random.seed(seed & 0x7fffffff)
        # choose temporal range
        if model_mode == 'model2' and fps > 0:
            last_frames = int(round(float(last5s_sec) * fps))
            start = max(0, length - last_frames)
            end = length
            seg_len = max(1, end - start)
            base_indices = [start + i for i in range(seg_len)]
        else:
            base_indices = list(range(length))
        if len(base_indices) == 0:
            # fallback to single black frame
            clip = torch.zeros(3, n_frames, image_size, image_size, dtype=torch.float32)
        else:
            # sample T indices
            det = False  # allow stochastic for TTA
            rel_idx = even_sample_indices(len(base_indices), n_frames, deterministic=det)
            abs_idx = [base_indices[i] for i in rel_idx]
            clip = build_clip_from_indices(vr, abs_idx, image_size)
        clip = clip.unsqueeze(0).to(device)  # 1,C,T,H,W
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(clip)
            main = out['main']  # (1,1) or (1,8)
        preds.append(main.squeeze(0).detach().cpu().numpy())
    preds = np.stack(preds, axis=0)
    pred_mean = preds.mean(axis=0)
    return pred_mean


@torch.inference_mode()
def infer_video_with_vr(model: torch.nn.Module,
                        vr,
                        length: int,
                        fps: float,
                        vid_key: str,
                        tta: int,
                        device: torch.device,
                        n_frames: int,
                        image_size: int,
                        model_mode: str = 'model2',
                        last5s_sec: int = 5,
                        seed_base: int = 12345) -> np.ndarray:
    preds = []
    for rep in range(tta):
        seed = seed_base + rep * 997 + (hash(vid_key) & 0x7fffffff)
        random.seed(seed)
        np.random.seed(seed & 0x7fffffff)
        if model_mode == 'model2' and fps > 0:
            last_frames = int(round(float(last5s_sec) * fps))
            start = max(0, length - last_frames)
            end = length
            seg_len = max(1, end - start)
            base_indices = [start + i for i in range(seg_len)]
        else:
            base_indices = list(range(length))
        if len(base_indices) == 0:
            clip = torch.zeros(3, n_frames, image_size, image_size, dtype=torch.float32)
        else:
            det = False
            rel_idx = even_sample_indices(len(base_indices), n_frames, deterministic=det)
            abs_idx = [base_indices[i] for i in rel_idx]
            clip = build_clip_from_indices(vr, abs_idx, image_size)
        clip = clip.unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(clip)
            main = out['main']
        preds.append(main.squeeze(0).detach().cpu().numpy())
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0)


def write_taskA_csv(out_csv: str, video_ids: List[str], results: Dict[str, int]):
    import csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['VIDEO', 'GRS'])
        for vid in video_ids:
            w.writerow([vid, int(results[vid])])


def write_taskB_csv(out_csv: str, video_ids: List[str], results: Dict[str, List[int]]):
    import csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    # NOTE: per spec, the last header is OSATSFINALQUALITY (no underscore)
    header = ['VIDEO', 'OSATS_RESPECT', 'OSATS_MOTION', 'OSATS_INSTRUMENT', 'OSATS_SUTURE',
              'OSATS_FLOW', 'OSATS_KNOWLEDGE', 'OSATS_PERFORMANCE', 'OSATSFINALQUALITY']
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for vid in video_ids:
            row = [vid] + list(results[vid])
            w.writerow(row)


def find_mp4s(input_dir: str) -> List[str]:
    files = []
    for ext in ('*.mp4', '*.MP4', '*.mkv', '*.avi'):
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    # also search one level deeper
    for root, dirs, _ in os.walk(input_dir):
        for ext in ('*.mp4', '*.MP4'):
            files.extend(glob.glob(os.path.join(root, ext)))
    # deduplicate
    files = sorted(list({os.path.abspath(p) for p in files}))
    return files


def list_fold_weights(weights_dir: str) -> List[str]:
    if not weights_dir:
        return []
    if not os.path.isdir(weights_dir):
        return []
    cand = []
    # common patterns
    cand.extend(glob.glob(os.path.join(weights_dir, 'fold*.pt')))
    cand.extend(glob.glob(os.path.join(weights_dir, 'fold*.pth')))
    # fallback: any .pt under dir
    if not cand:
        cand.extend(glob.glob(os.path.join(weights_dir, '*.pt')))
        cand.extend(glob.glob(os.path.join(weights_dir, '*.pth')))
    # sort by name
    cand = sorted(list({os.path.abspath(p) for p in cand}))
    # keep at most 5
    if len(cand) > 5:
        cand = cand[:5]
    return cand


def load_state_dict_any(path: str):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and sd.get('model_ema') is not None:
        return sd['model_ema']
    if isinstance(sd, dict) and 'model' in sd:
        return sd['model']
    return sd


def build_model(task: str, out_dim: int, device: torch.device) -> torch.nn.Module:
    model = Swin3DWithSegAux(task=task, num_outputs=out_dim, weights=None)
    model = model.to(device).eval()
    return model


def _build_base_indices(length: int, fps: float, model_mode: str, last5s_sec: int) -> List[int]:
    if model_mode == 'model2' and fps > 0:
        last_frames = int(round(float(last5s_sec) * fps))
        start = max(0, length - last_frames)
        end = length
        return [start + i for i in range(max(1, end - start))]
    else:
        return list(range(length))


def _generate_tta_indices(base_indices: List[int], n_frames: int, tta: int, seed_base: int, vid_key: str) -> List[List[int]]:
    rep_indices: List[List[int]] = []
    for rep in range(tta):
        seed = seed_base + rep * 997 + (hash(vid_key) & 0x7fffffff)
        random.seed(seed)
        np.random.seed(seed & 0x7fffffff)
        det = False
        rel_idx = even_sample_indices(len(base_indices), n_frames, deterministic=det)
        abs_idx = [base_indices[i] for i in rel_idx]
        rep_indices.append(abs_idx)
    return rep_indices


def _prefetch_letterboxed_frames(vr, indices: List[int], image_size: int) -> Dict[int, np.ndarray]:
    uniq = sorted({int(i) for i in indices})
    cache: Dict[int, np.ndarray] = {}
    L = len(vr)
    for i in uniq:
        i0 = max(0, min(L - 1, int(i)))
        frame = vr.get(i0)
        img = Image.fromarray(frame)
        img_sq = resize_square_pil(img, image_size)
        cache[i0] = np.array(img_sq, dtype=np.uint8)
    return cache


def _clip_from_cached(cache: Dict[int, np.ndarray], indices: List[int]) -> torch.Tensor:
    frames_np = [cache[int(i)] for i in indices]
    clip_np = np.stack(frames_np, axis=0)  # T,H,W,3
    return normalize_clip(clip_np)


@torch.inference_mode()
def ensemble_infer_video_reuse(models: List[torch.nn.Module],
                               video_path: str,
                               task: str,
                               tta: int,
                               device: torch.device,
                               n_frames: int,
                               image_size: int,
                               model_mode: str = 'model2',
                               last5s_sec: int = 5,
                               seed_base: int = 12345) -> np.ndarray:
    vr, length, fps = decode_video_frames_decord(video_path)
    vid_key = os.path.splitext(os.path.basename(video_path))[0]
    base_indices = _build_base_indices(length, fps, model_mode, last5s_sec)
    # Generate TTA indices once (shared by all folds)
    rep_indices = _generate_tta_indices(base_indices, n_frames, tta, seed_base, vid_key)
    # Prefetch frames once for all TTA reps
    all_needed = [j for arr in rep_indices for j in arr]
    cache = _prefetch_letterboxed_frames(vr, all_needed, image_size)

    # Accumulate predictions over models and reps
    acc = None
    for rep_idx in rep_indices:
        clip = _clip_from_cached(cache, rep_idx).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            for m in models:
                out = m(clip)
                main = out['main']  # (1,D)
                p = main.squeeze(0).detach().cpu().numpy().astype(np.float32)
                if acc is None:
                    acc = p
                else:
                    acc += p
    total = float(len(models) * tta)
    pred_mean = acc / max(1.0, total)
    return pred_mean


class VideoTTADataset(Dataset):
    def __init__(self,
                 video_paths: List[str],
                 video_ids: List[str],
                 n_frames: int,
                 tta: int,
                 image_size: int,
                 model_mode: str = 'model1',
                 last5s_sec: int = 5,
                 seed_base: int = 12345):
        self.video_paths = video_paths
        self.video_ids = video_ids
        self.n_frames = int(n_frames)
        self.tta = int(tta)
        self.image_size = int(image_size)
        self.model_mode = str(model_mode)
        self.last5s_sec = int(last5s_sec)
        self.seed_base = int(seed_base)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        path = self.video_paths[idx]
        vid = self.video_ids[idx]
        vr, length, fps = decode_video_frames_decord(path)
        base_indices = _build_base_indices(length, fps, self.model_mode, self.last5s_sec)
        rep_indices = _generate_tta_indices(base_indices, self.n_frames, self.tta, self.seed_base, vid)
        all_needed = [j for arr in rep_indices for j in arr]
        cache = _prefetch_letterboxed_frames(vr, all_needed, self.image_size)
        clips = []
        for arr in rep_indices:
            x = _clip_from_cached(cache, arr)
            clips.append(x)
        clips_t = torch.stack(clips, dim=0)  # (TTA,C,T,H,W)
        return {
            'video_id': vid,
            'clips': clips_t,
        }


# ===== Fast GPU decode path (Decord NVDEC + batched + Torch preprocessing) =====
if _HAS_DECORD:
    try:
        decord.bridge.set_bridge('torch')
    except Exception:
        pass
import torch.nn.functional as F
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1,1)
_WORKER_VR_CACHE: Dict[Tuple[str,str], object] = {}

def _get_vr_gpu(path: str):
    key = ('gpu', path)
    vr = _WORKER_VR_CACHE.get(key)
    if vr is None:
        if not _HAS_DECORD:
            raise RuntimeError('decord not available')
        try:
            vr = decord.VideoReader(path, ctx=decord.gpu(0))
        except Exception:
            vr = decord.VideoReader(path, ctx=decord.cpu(0))
        _WORKER_VR_CACHE[key] = vr
    return vr

def _torch_letterbox_batch(frames_nchw: torch.Tensor, size: int | Tuple[int,int]) -> torch.Tensor:
    N, C, H, W = frames_nchw.shape
    if isinstance(size, (tuple, list)):
        out_h, out_w = int(size[0]), int(size[1])
    else:
        out_h = out_w = int(size)
    scale = min(out_w / float(W), out_h / float(H))
    new_w = max(1, int(math.floor(W * scale)))
    new_h = max(1, int(math.floor(H * scale)))
    resized = F.interpolate(frames_nchw, size=(new_h, new_w), mode='bilinear', align_corners=False)
    pad_l = (out_w - new_w) // 2
    pad_r = out_w - new_w - pad_l
    pad_t = (out_h - new_h) // 2
    pad_b = out_h - new_h - pad_t
    out = F.pad(resized, (pad_l, pad_r, pad_t, pad_b))
    return out

def _normalize_inplace(clip: torch.Tensor) -> torch.Tensor:
    mean = _MEAN.to(clip.device, clip.dtype)
    std  = _STD.to(clip.device, clip.dtype)
    return (clip - mean) / std

def build_clip_from_indices_decord_gpu(vr, indices: List[int], image_size: int, device: torch.device, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    L = len(vr)
    idx = [max(0, min(L - 1, int(i))) for i in indices]
    uniq = sorted(set(idx))
    batch = vr.get_batch(uniq)  # (N,H,W,3) uint8 on CPU or GPU (Torch tensor when bridge=Torch)
    inv = {k: j for j, k in enumerate(uniq)}
    order = torch.tensor([inv[i] for i in idx], device=batch.device, dtype=torch.long)
    batch = batch.index_select(0, order)             # (T,H,W,3)
    frames = batch.permute(0, 3, 1, 2).to(dtype)     # (T,C,H,W) in chosen dtype
    frames = frames.div_(255)                        # [0,1]
    frames = _torch_letterbox_batch(frames, image_size)               # (T,C,S,S)
    clip = frames.permute(1, 0, 2, 3).contiguous()                    # (C,T,S,S)
    mean = _MEAN.to(clip.device, dtype)
    std  = _STD.to(clip.device, dtype)
    clip = (clip - mean) / std
    if clip.device != device:
        clip = clip.to(device, non_blocking=True)
    return clip


class VideoTTADatasetFastGPU(Dataset):
    def __init__(self,
                 video_paths: List[str],
                 video_ids: List[str],
                 n_frames: int,
                 tta: int,
                 image_size: int,
                 model_mode: str = 'model1',
                 last5s_sec: int = 5,
                 seed_base: int = 12345,
                 device: str = 'cuda'):
        self.video_paths = video_paths
        self.video_ids = video_ids
        self.n_frames = int(n_frames)
        self.tta = int(tta)
        self.image_size = int(image_size)
        self.model_mode = str(model_mode)
        self.last5s_sec = int(last5s_sec)
        self.seed_base = int(seed_base)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        path = self.video_paths[idx]
        vid = self.video_ids[idx]
        if not _HAS_DECORD:
            raise RuntimeError('decord is required for GPU decode path')
        vr = _get_vr_gpu(path)
        length = len(vr)
        try:
            fps = float(vr.get_avg_fps())
            fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        except Exception:
            fps = 30.0
        base_indices = _build_base_indices(length, fps, self.model_mode, self.last5s_sec)
        rep_indices = _generate_tta_indices(base_indices, self.n_frames, self.tta, self.seed_base, vid)
        clips = []
        for arr in rep_indices:
            clip = build_clip_from_indices_decord_gpu(vr, arr, self.image_size, self.device)
            clips.append(clip.to('cpu', non_blocking=False))  # move each rep back to CPU to reduce GPU peak
            del clip
        clips_t = torch.stack(clips, dim=0)  # (TTA,C,T,H,W) on CPU
        return {'video_id': vid, 'clips': clips_t}

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description='Task A/B inference from mp4 inputs (Swin3D with TTA + 5-fold ensemble)')
    ap.add_argument('--task', choices=['A', 'B'], required=True)
    ap.add_argument('--ckpt', default='', help='Fallback: single checkpoint (.pt or .pth)')
    ap.add_argument('--weights-dir', default='', help='Directory containing fold checkpoints (e.g., /app/weight/TaskA)')
    ap.add_argument('--input-dir', default='/input', help='Directory containing input videos (mp4)')
    ap.add_argument('--out-csv', default='', help='Output CSV path; if empty, defaults to /output/taskA_GRS.csv or /output/taskB_OSATS.csv')
    ap.add_argument('--model-mode', choices=['model1', 'model2'], default='model2')
    ap.add_argument('--n-frames', type=int, default=96)
    ap.add_argument('--last5s-sec', type=int, default=5)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--tta', type=int, default=5)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    if not _HAS_DECORD:
        print('[error] decord is not available. Please ensure decord is installed.', file=sys.stderr)
        sys.exit(2)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # CUDA backend knobs
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True

    # list videos
    mp4s = find_mp4s(args.input_dir)
    if not mp4s:
        print(f'[error] no input videos found in {args.input_dir}', file=sys.stderr)
        sys.exit(1)
    video_ids = [os.path.splitext(os.path.basename(p))[0] for p in mp4s]

    # discover weights directory (default by task) if not provided
    weights_dir = args.weights_dir
    if not weights_dir:
        default_dir = '/app/weight/TaskA' if args.task == 'A' else '/app/weight/TaskB'
        if os.path.isdir(default_dir):
            weights_dir = default_dir
    fold_paths = list_fold_weights(weights_dir)

    out_dim = 1 if args.task == 'A' else 8

    models: List[torch.nn.Module] = []
    # Load ensemble or single model fallback
    if fold_paths:
        for pi, p in enumerate(fold_paths):
            m = build_model(args.task, out_dim, device)
            w = load_state_dict_any(p)
            m.load_state_dict(w, strict=True)
            models.append(m)
        print(f"[info] loaded ensemble folds: {len(models)} from {weights_dir}")
    else:
        if not args.ckpt:
            print('[error] no weights found: provide --weights-dir or --ckpt', file=sys.stderr)
            sys.exit(2)
        m = build_model(args.task, out_dim, device)
        w = load_state_dict_any(args.ckpt)
        m.load_state_dict(w, strict=True)
        models.append(m)
        print(f"[info] loaded single checkpoint: {args.ckpt}")

    # Wrap models into an ensemble module
    ensemble = EnsembleSwin3D(models).to(device)
    # Optional: torch.compile for speed (PyTorch 2.0+)
    ensemble = torch.compile(ensemble)
    ensemble.eval()

    # run
    resultsA: Dict[str, int] = {}
    resultsB: Dict[str, List[int]] = {}

    # DataLoader fast path (model1 fixed)
    is_gpu_decode = (_HAS_DECORD and device.type == 'cuda')
    if is_gpu_decode:
        ds = VideoTTADatasetFastGPU(mp4s, video_ids, n_frames=args.n_frames, tta=args.tta, image_size=args.image_size,
                                    model_mode='model1', last5s_sec=args.last5s_sec, seed_base=12345, device=args.device)
    else:
        ds = VideoTTADataset(mp4s, video_ids, n_frames=args.n_frames, tta=args.tta, image_size=args.image_size,
                             model_mode='model1', last5s_sec=args.last5s_sec, seed_base=12345)
    # Defaults tuned for overlap of I/O and compute
    if is_gpu_decode:
        num_workers = 0
        pin_memory = False
        prefetch_factor = 2
    else:
        num_workers = 16
        pin_memory = True
        prefetch_factor = 2
    loader_kwargs = dict(batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=prefetch_factor))
    loader = DataLoader(ds, **loader_kwargs)

    for batch in tqdm(loader, desc='infer', total=len(ds)):
        vids = batch['video_id']  # List[str] or list-like
        clips = batch['clips']    # (B,TTA,C,T,H,W)
        if isinstance(vids, str):
            vids = [vids]
        B, R = clips.shape[0], clips.shape[1]
        y_acc = []
        for r in range(R):
            x_r = clips[:, r].to(device, non_blocking=True)  # (B,C,T,H,W)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                y_r = ensemble(x_r)  # (B,D), already fold-averaged
            y_acc.append(y_r)
            del x_r
            torch.cuda.synchronize() if device.type == 'cuda' else None
        y = torch.stack(y_acc, dim=1).mean(dim=1)  # (B,D) mean over TTA reps
        y_np = y.detach().cpu().numpy().astype(np.float32)
        # write per video in batch order
        for i, vid in enumerate(vids):
            p = y_np[i]
            if args.task == 'A':
                grs_num = float(np.clip(float(p[0]), 8.0, 40.0))
                if grs_num <= 15:
                    grs_cls = 0
                elif grs_num <= 23:
                    grs_cls = 1
                elif grs_num <= 31:
                    grs_cls = 2
                else:
                    grs_cls = 3
                resultsA[vid] = int(grs_cls)
            else:
                vec = np.clip(p, 1.0, 5.0)
                vec_i = np.clip(np.rint(vec - 1.0).astype(int), 0, 4)
                resultsB[vid] = [int(x) for x in vec_i.tolist()]

    # write outputs
    if args.task == 'A':
        out_csv = args.out_csv or '/output/taskA_GRS.csv'
        write_taskA_csv(out_csv, video_ids, resultsA)
    else:
        out_csv = args.out_csv or '/output/taskB_OSATS.csv'
        write_taskB_csv(out_csv, video_ids, resultsB)

    print(f"[done] videos={len(video_ids)} wrote: {out_csv}")


if __name__ == '__main__':
    main() 