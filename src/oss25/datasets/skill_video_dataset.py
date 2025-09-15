import os
import glob
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def letterbox(im: Image.Image, size: int = 224, color=(0, 0, 0)) -> Tuple[Image.Image, Dict]:
    """Resize to a square (size x size) without padding (aspect ratio is not preserved)."""
    w, h = im.size
    im_resized = im.resize((size, size), Image.BILINEAR)
    return im_resized, {"scale": None, "pad": (0, 0), "orig_size": (h, w)}


def letterbox_mask(im: Image.Image, size: int = 224, fill=0) -> Tuple[Image.Image, Dict]:
    """Resize mask to a square (size x size) without padding (nearest)."""
    w, h = im.size
    im_resized = im.resize((size, size), Image.NEAREST)
    return im_resized, {"scale": None, "pad": (0, 0), "orig_size": (h, w)}


def _pair_10fps_and_masks(video_root_10fps: str, video_root_masks: str) -> Tuple[List[str], List[str]]:
    jpgs = sorted(glob.glob(os.path.join(video_root_10fps, '*.jpg')))
    pngs = sorted(glob.glob(os.path.join(video_root_masks, '*.png')))
    # Filter out non-frame files like '_DONE'
    pngs = [p for p in pngs if p.endswith('.png')]
    if not jpgs or not pngs or len(jpgs) != len(pngs):
        return [], []
    return jpgs, pngs


def _even_bins_indices(length: int, n: int, deterministic: bool) -> List[int]:
    if length <= 0:
        return []
    if n <= 1:
        return [length // 2]
    # Split [0, length) into n bins; pick one index per bin
    step = length / n
    indices = []
    for i in range(n):
        start = int(round(i * step))
        end = int(round((i + 1) * step))
        if end <= start:
            end = min(start + 1, length)
        if deterministic:
            idx = (start + end - 1) // 2
        else:
            idx = random.randrange(start, end)
        idx = max(0, min(length - 1, idx))
        indices.append(idx)
    return indices


class SkillVideoDataset(Dataset):
    """
    Video-level dataset using 10fps frames and corresponding pseudo masks so that
    the auxiliary loss can be applied to all frames.

    Modes:
      - model1: sample 96 frames across entire 10fps sequence (bin-wise random)
      - model2: sample 96 frames from the last `last5s_sec` seconds; if available
                frames < 96, repeat/upsample indices deterministically.
    """

    def __init__(
        self,
        video_ids: List[str],
        labels: Dict[str, Dict],
        tenfps_root: str,
        pseudo6_root: str,
        task: str,
        model_mode: str,
        n_frames: int = 96,
        last5s_sec: int = 5,
        image_size: int = 224,
        deterministic_val: bool = False,
        frames1fps_root: Optional[str] = None,
    ):
        self.video_ids = video_ids
        self.labels = labels
        self.tenfps_root = tenfps_root
        self.pseudo6_root = pseudo6_root
        self.task = task
        self.model_mode = model_mode
        self.n_frames = n_frames
        self.last5s_sec = last5s_sec
        self.image_size = image_size
        self.deterministic_val = deterministic_val
        self.frames1fps_root = frames1fps_root

        # Keep all provided videos; availability of pseudo will be handled per-item

    def __len__(self):
        return len(self.video_ids)

    def _sample_indices(self, L: int) -> List[int]:
        det = self.deterministic_val
        if self.model_mode == 'model1':
            idx = _even_bins_indices(L, self.n_frames, deterministic=det)
        else:
            # last K seconds at 10fps
            # If we are using 10fps frames, K=sec*10; if fallback to 1fps, K=sec*1.
            # We don't know here whether 10fps is used; approximate using n_frames density:
            # If L is large enough (>= self.last5s_sec * 10), assume 10fps; else assume 1fps.
            K10 = self.last5s_sec * 10
            K1 = self.last5s_sec * 1
            K = K10 if L >= K10 else min(L, K1)
            start = max(0, L - K)
            window = L - start
            if window <= 0:
                # fallback to whole clip
                idx = _even_bins_indices(L, self.n_frames, deterministic=det)
            else:
                # select n_frames over [start, L)
                base = _even_bins_indices(window, self.n_frames, deterministic=det)
                idx = [start + b for b in base]
        # ensure length == n_frames by padding/repeating if needed
        if len(idx) < self.n_frames:
            # repeat last indices deterministically
            pad = [idx[-1]] * (self.n_frames - len(idx)) if idx else [0] * self.n_frames
            idx = idx + pad
        elif len(idx) > self.n_frames:
            idx = idx[: self.n_frames]
        return idx

    def __getitem__(self, i: int):
        vid = self.video_ids[i]
        # Prefer 10fps frames
        v10 = os.path.join(self.tenfps_root, vid)
        vmsk = os.path.join(self.pseudo6_root, vid)
        jpgs_10, pngs_10 = _pair_10fps_and_masks(v10, vmsk)
        use_10fps = len(jpgs_10) > 0 and len(pngs_10) > 0

        # Fallback to 1fps frames if available
        frames = []
        if use_10fps:
            frames = jpgs_10
        elif self.frames1fps_root:
            v1 = os.path.join(self.frames1fps_root, vid)
            if os.path.isdir(v1):
                frames = sorted(glob.glob(os.path.join(v1, '*.jpg')) + glob.glob(os.path.join(v1, '*.png')))

        # If still empty, raise
        if not frames:
            raise RuntimeError(f'No frames found for video {vid}')

        L = len(frames)
        sel = self._sample_indices(L)

        # Build clip tensor
        imgs = []
        for idx in sel:
            im = Image.open(frames[idx]).convert('RGB')
            im_sq, _ = letterbox(im, self.image_size)
            imgs.append(torch.from_numpy(np.array(im_sq)).permute(2, 0, 1))
        clip = torch.stack(imgs, dim=1).float()  # CxTxHxW

        # Normalize per Swin3D_B weights
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        clip = (clip / 255.0 - mean) / std

        # Build seg labels if 10fps+pseudo6 available; else fill ignore_index (-100)
        T = clip.shape[1]
        T_out = (T - 2) // 2 + 1
        if use_10fps:
            # Robust mapping: 10fps frame index k -> pseudo mask frame id
            # Observed mapping: k=0 -> frame_0.png, k>=1 -> frame_{1000 + (k-1)}.png
            video_id = vid
            map_idx = [min(len(sel) - 1, 2 * j + 1) for j in range(T_out)]
            seg_labels = []
            for j in map_idx:
                k = sel[j]
                mask_id = 0 if k == 0 else (1000 + (k - 1))
                mpath = os.path.join(self.pseudo6_root, video_id, f"{video_id}_frame_{mask_id}.png")
                if not os.path.exists(mpath):
                    # fallback: skip
                    seg_labels.append(torch.full((7, 7), fill_value=-100, dtype=torch.int64))
                    continue
                m = Image.open(mpath).convert('L')
                m_sq, _ = letterbox_mask(m, self.image_size)
                m_small = m_sq.resize((7, 7), Image.NEAREST)
                seg_labels.append(torch.from_numpy(np.array(m_small).astype(np.int64)))
            seg_labels = torch.stack(seg_labels, dim=0)  # T' x 7 x 7
        else:
            seg_labels = torch.full((T_out, 7, 7), fill_value=-100, dtype=torch.int64)

        lab = self.labels[vid]
        sample = {
            'video_id': vid,
            'clip': clip,             # CxTxHxW
            'seg_labels': seg_labels, # T' x 7 x 7 (or -100 to ignore)
        }
        if self.task == 'A':
            # Use numeric GRS for regression (8..40)
            # classification bins will be computed only for metrics at validation time
            grs = float(lab.get('grs_num', 0.0))
            sample['yA'] = torch.tensor(grs, dtype=torch.float32)
        else:
            sample['yB'] = torch.tensor(lab['osats'], dtype=torch.float32)  # (8,)
        # optional aux labels if available
        if 'time_bin' in lab:
            sample['y_time'] = torch.tensor(lab['time_bin'], dtype=torch.long)
        if 'group_idx' in lab:
            sample['y_group'] = torch.tensor(lab['group_idx'], dtype=torch.long)
        if 'sutures' in lab:
            sample['y_sutures'] = torch.tensor(lab['sutures'], dtype=torch.float32)

        return sample
