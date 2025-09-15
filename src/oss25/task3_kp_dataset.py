import os
import glob
from typing import Dict, List, Tuple, Optional

import math
import numpy as np
from PIL import Image
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

import torch
from torch.utils.data import Dataset


CLASS_NAMES = {
    0: 'left_hand',
    1: 'right_hand',
    2: 'scissors',
    3: 'tweezers',
    4: 'needle_holder',
    5: 'needle',
}

# Per-class keypoint counts
KP_COUNTS = {0: 6, 1: 6, 2: 3, 3: 3, 4: 3, 5: 3}

# Output channel layout (27 channels total)
# 0..5: left hand (6), 6..11: right hand (6), 12..14: scissors (3), 15..17: tweezers (3),
# 18..20: needle holder (3), 21..26: needle (two instances x3 -> 6)
CHANNEL_LAYOUT: Dict[int, Tuple[int, int]] = {
    0: (0, 6),
    1: (6, 6),
    2: (12, 3),
    3: (15, 3),
    4: (18, 3),
    5: (21, 6),
}


def parse_mot_line(line: str) -> Optional[Tuple[int, int, int, List[Tuple[float, float, int]]]]:
    parts = [p.strip() for p in line.strip().split(',') if len(p.strip())]
    if len(parts) < 7:
        return None
    try:
        frame = int(float(parts[0]))
        tid = int(float(parts[1]))
        cls = int(float(parts[2]))
    except Exception:
        return None
    rest = parts[7:] if len(parts) >= 11 else []
    triples = []
    for i in range(0, len(rest), 3):
        try:
            x = float(rest[i]); y = float(rest[i+1]); v = int(float(rest[i+2]))
            triples.append((x, y, v))
        except Exception:
            break
    return frame, tid, cls, triples


def resize_and_pad_to_square(img: Image.Image, size: int, fill: int = 0) -> Tuple[Image.Image, float, float, int, int]:
    """Resize image to fit in size x size, keeping aspect ratio, pad with fill.
    Returns: image, scale, inv_scale, pad_left, pad_top.
    """
    w, h = img.size
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (size, size), color=(fill, fill, fill))
    pl = (size - nw) // 2
    pt = (size - nh) // 2
    canvas.paste(img_resized, (pl, pt))
    return canvas, s, (1.0 / s if s != 0 else 0.0), pl, pt


def draw_gaussian(heatmap: np.ndarray, x: float, y: float, sigma: float = 2.0):
    h, w = heatmap.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return
    radius = max(1, int(3 * sigma))
    xs = np.arange(int(x) - radius, int(x) + radius + 1)
    ys = np.arange(int(y) - radius, int(y) + radius + 1)
    xs = xs[(xs >= 0) & (xs < w)]
    ys = ys[(ys >= 0) & (ys < h)]
    for yy in ys:
        for xx in xs:
            val = math.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma * sigma))
            heatmap[yy, xx] = max(heatmap[yy, xx], val)


class Task3KeypointDataset(Dataset):
    def __init__(self, data_root: str, split: str = 'train', image_size=384, heatmap_size=96,
                 aug_rotate_deg: float = 0.0, pseudo6_root: Optional[str] = None,
                 use_flow: bool = True, allow_seg_only: bool = False, resize_only: bool = False):
        assert split in ['train', 'val']
        self.data_root = data_root
        self.split = split
        self.frames_dir = os.path.join(data_root, split, 'frames')
        self.mot_dir = os.path.join(data_root, split, 'mot')
        if isinstance(image_size, (tuple, list)):
            self.image_h, self.image_w = int(image_size[0]), int(image_size[1])
        else:
            self.image_h = self.image_w = int(image_size)
        if isinstance(heatmap_size, (tuple, list)):
            self.heatmap_h, self.heatmap_w = int(heatmap_size[0]), int(heatmap_size[1])
        else:
            self.heatmap_h = self.heatmap_w = int(heatmap_size)
        self.image_size = max(self.image_h, self.image_w)
        self.heatmap_size = max(self.heatmap_h, self.heatmap_w)
        # augmentation config
        self.aug_rotate_deg = float(aug_rotate_deg) if split == 'train' else 0.0
        # optional SAM2 pseudo-label root (6-class indexed mask)
        if pseudo6_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            pseudo6_root = os.path.join(repo_root, 'artifacts', 'seg_pseudo6')
        self.pseudo6_root = pseudo6_root
        self.use_flow = use_flow
        self.resize_only = bool(resize_only)
        # build item index supporting per-frame (train) and aggregated-per-video (val)
        self.items = []  # list of dicts: {mot_path, frame_name, img_path}
        if split == 'train':
            for mp in sorted(glob.glob(os.path.join(self.mot_dir, '*.txt'))):
                base = os.path.basename(mp)
                if base.startswith('SYNAPSE_'):  # skip manifest
                    continue
                name = os.path.splitext(base)[0]  # <video>_frame_<id>
                img_path = os.path.join(self.frames_dir, f'{name}.png')
                self.items.append({'mot_path': mp, 'frame_name': name, 'img_path': img_path})
            # Optionally add segmentation-only frames (no MOT) if masks exist
            if allow_seg_only and os.path.isdir(self.pseudo6_root):
                existing = set([it['frame_name'] for it in self.items])
                for vid in sorted(os.listdir(self.pseudo6_root)):
                    vdir = os.path.join(self.pseudo6_root, vid)
                    if not os.path.isdir(vdir):
                        continue
                    for mpath in sorted(glob.glob(os.path.join(vdir, f'{vid}_frame_*.png'))):
                        fname = os.path.splitext(os.path.basename(mpath))[0]  # <video>_frame_<id>
                        if fname in existing:
                            continue
                        img_path = os.path.join(self.frames_dir, f'{fname}.png')
                        if os.path.exists(img_path):
                            self.items.append({'mot_path': None, 'frame_name': fname, 'img_path': img_path})
        else:
            # val MOT are aggregated per video: <video>.txt
            for mp in sorted(glob.glob(os.path.join(self.mot_dir, '*.txt'))):
                base = os.path.basename(mp)
                if base.startswith('SYNAPSE_'):
                    continue
                video = os.path.splitext(base)[0]
                vdir = os.path.join(self.frames_dir, video)
                if not os.path.isdir(vdir):
                    # fallback: maybe flat structure (unlikely)
                    pngs = []
                else:
                    pngs = sorted(glob.glob(os.path.join(vdir, f'{video}_frame_*.png')),
                                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_frame_')[1]))
                for img_path in pngs:
                    name = os.path.splitext(os.path.basename(img_path))[0]
                    self.items.append({'mot_path': mp, 'frame_name': name, 'img_path': img_path})

    def __len__(self) -> int:
        return len(self.items)

    def _channels_for_class(self, cls: int) -> Tuple[int, int]:
        base, count = CHANNEL_LAYOUT[cls]
        return base, count

    def _assign_needle_slots(self, entries: List[Tuple[int, int, int, List[Tuple[float, float, int]]]]):
        # entries of class 5, sort by track id, keep up to 2
        needle_entries = [(tid, triples) for (_, tid, cls, triples) in entries if cls == 5]
        needle_entries.sort(key=lambda x: x[0])
        return needle_entries[:2]

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        mot_path = rec['mot_path']
        name = rec['frame_name']  # <video>_frame_<id>
        img_path = rec['img_path']

        img = Image.open(img_path).convert('RGB')
        W0, H0 = img.size
        if self.resize_only:
            img_sq = img.resize((self.image_w, self.image_h), Image.BILINEAR)
            scale_x = self.image_w / float(W0)
            scale_y = self.image_h / float(H0)
            pl = 0; pt = 0
            letterbox = False
        else:
            img_sq, scale, inv_scale, pl, pt = resize_and_pad_to_square(img, self.image_size, fill=0)
            scale_x = scale_y = scale
            letterbox = True
        # optional rotation augmentation (no flip) around image center
        rot_deg = 0.0
        if self.aug_rotate_deg > 0.0:
            rot_deg = np.random.uniform(-self.aug_rotate_deg, self.aug_rotate_deg)
            img_sq = img_sq.rotate(rot_deg, resample=Image.BILINEAR)
        img_np = np.array(img_sq).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)

        # Prepare previous frame (for optical flow / foreground)
        prev_img_t = None
        flow_uv = None
        fg_mask = None
        if self.use_flow:
            # find previous frame path
            prev_name = None
            try:
                video_id, frame_str = name.split('_frame_')
                frame_idx = int(frame_str)
                if frame_idx > 0:
                    prev_name = f"{video_id}_frame_{frame_idx-1}"
            except Exception:
                prev_name = None
            if prev_name is not None:
                prev_img_path = os.path.join(self.frames_dir, f'{prev_name}.png')
                if not os.path.exists(prev_img_path):
                    prev_name = None
            if prev_name is None:
                prev_img_sq = img_sq.copy()
            else:
                prev_img = Image.open(os.path.join(self.frames_dir, f'{prev_name}.png')).convert('RGB')
                if self.resize_only:
                    prev_img_sq = prev_img.resize((self.image_w, self.image_h), Image.BILINEAR)
                else:
                    prev_img_sq, _, _, _, _ = resize_and_pad_to_square(prev_img, self.image_size, fill=0)
                if self.aug_rotate_deg > 0.0 and rot_deg != 0.0:
                    prev_img_sq = prev_img_sq.rotate(rot_deg, resample=Image.BILINEAR)
            prev_np = np.array(prev_img_sq).astype(np.float32) / 255.0
            prev_img_t = torch.from_numpy(prev_np).permute(2, 0, 1)
            # optical flow + foreground mask (computed on letterboxed & rotated frames)
            if _HAS_CV2:
                g0 = cv2.cvtColor((prev_np*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                g1 = cv2.cvtColor((img_np*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                try:
                    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    # H,W,2 -> 2,H,W float32
                    flow_uv = torch.from_numpy(flow.astype(np.float32)).permute(2, 0, 1)
                except Exception:
                    flow_uv = torch.zeros(2, img_sq.size[1], img_sq.size[0], dtype=torch.float32)
                # foreground via abs diff
                diff = cv2.absdiff(g1, g0).astype(np.float32) / 255.0
                # simple normalization
                fg_mask = torch.from_numpy(diff).unsqueeze(0).float()
            else:
                flow_uv = torch.zeros(2, img_sq.size[1], img_sq.size[0], dtype=torch.float32)
                fg_mask = torch.zeros(1, img_sq.size[1], img_sq.size[0], dtype=torch.float32)

        # Targets
        C = 27
        Hh = self.heatmap_h
        Wh = self.heatmap_w
        heatmaps = np.zeros((C, Hh, Wh), dtype=np.float32)
        vis_labels = np.full((C,), -100, dtype=np.int64)  # -100 ignored
        coord_targets = np.zeros((C, 2), dtype=np.float32)  # (x_h, y_h)
        coord_weights = np.zeros((C,), dtype=np.float32)    # 0 (ignore), 0.5 (hidden), 1.0 (visible)
        seg6 = np.full((Hh, Wh), -100, dtype=np.int64)      # 0..6 labels, -100 ignored by CE

        # Parse MOT entries: handle per-frame files (train) and aggregated per-video (val)
        entries = []
        has_mot = isinstance(mot_path, str) and os.path.exists(mot_path)
        if has_mot:
            with open(mot_path, 'r') as f:
                if '_frame_' in os.path.splitext(os.path.basename(mot_path))[0]:
                    for line in f:
                        parsed = parse_mot_line(line)
                        if parsed is None:
                            continue
                        entries.append(parsed)
                else:
                    # aggregated file: filter by current frame index
                    try:
                        frame_idx = int(name.split('_frame_')[1])
                    except Exception:
                        frame_idx = None
                    for line in f:
                        parsed = parse_mot_line(line)
                        if parsed is None:
                            continue
                        if frame_idx is None or parsed[0] == frame_idx:
                            entries.append(parsed)

        # For each class, process entries
        used = set()
        # Hands and tools (single instance per class expected)
        for cls in [0, 1, 2, 3, 4]:
            base, count = self._channels_for_class(cls)
            # pick first occurrence for this cls
            found = None
            for (frame, tid, c, triples) in entries:
                if c == cls:
                    found = (tid, triples)
                    break
            if found is None:
                continue
            tid, triples = found
            used.add((cls, tid))
            kp_required = KP_COUNTS[cls]
            if len(triples) < kp_required:
                # pad missing
                triples = triples + [(0.0, 0.0, 0)] * (kp_required - len(triples))
            triples = triples[:kp_required]
            # place heatmaps and vis labels
            for i, (x, y, v) in enumerate(triples):
                # transform coords
                x_n = x * scale_x + pl
                y_n = y * scale_y + pt
                # apply rotation if any
                if self.aug_rotate_deg > 0.0 and rot_deg != 0.0:
                    cx = self.image_size * 0.5
                    cy = self.image_size * 0.5
                    rad = np.deg2rad(rot_deg)
                    cos_a, sin_a = np.cos(rad), np.sin(rad)
                    xr = cos_a * (x_n - cx) - sin_a * (y_n - cy) + cx
                    yr = sin_a * (x_n - cx) + cos_a * (y_n - cy) + cy
                    x_n, y_n = xr, yr
                x_h = x_n * (Wh / (self.image_w if self.resize_only else self.image_size))
                y_h = y_n * (Hh / (self.image_h if self.resize_only else self.image_size))
                ch = base + i
                in_bounds = (x_h >= 0 and y_h >= 0 and x_h < Wh and y_h < Hh)
                if v > 0 and in_bounds:  # supervise only if inside after aug
                    draw_gaussian(heatmaps[ch], x_h, y_h, sigma=2.0)
                # visibility label: if moved out by aug, mark as 0 (outside)
                vis_labels[ch] = int(v if in_bounds else 0)
                # coord targets and weights
                if v > 0 and in_bounds:
                    coord_targets[ch, 0] = float(x_h)
                    coord_targets[ch, 1] = float(y_h)
                    # occluded(1) も visible(2) と同等に重み1.0
                    coord_weights[ch] = 1.0
                else:
                    coord_targets[ch, 0] = 0.0
                    coord_targets[ch, 1] = 0.0
                    coord_weights[ch] = 0.0

        # Needles (up to 2 instances)
        needle_pairs = self._assign_needle_slots(entries)
        for slot_idx, (tid, triples) in enumerate(needle_pairs):
            base, count = self._channels_for_class(5)
            offset = slot_idx * KP_COUNTS[5]
            triples = triples[:KP_COUNTS[5]]
            if len(triples) < KP_COUNTS[5]:
                triples = triples + [(0.0, 0.0, 0)] * (KP_COUNTS[5] - len(triples))
            for i, (x, y, v) in enumerate(triples):
                x_n = x * scale_x + pl
                y_n = y * scale_y + pt
                if self.aug_rotate_deg > 0.0 and rot_deg != 0.0:
                    cx = self.image_size * 0.5
                    cy = self.image_size * 0.5
                    rad = np.deg2rad(rot_deg)
                    cos_a, sin_a = np.cos(rad), np.sin(rad)
                    xr = cos_a * (x_n - cx) - sin_a * (y_n - cy) + cx
                    yr = sin_a * (x_n - cx) + cos_a * (y_n - cy) + cy
                    x_n, y_n = xr, yr
                x_h = x_n * (Wh / (self.image_w if self.resize_only else self.image_size))
                y_h = y_n * (Hh / (self.image_h if self.resize_only else self.image_size))
                ch = base + offset + i
                in_bounds = (x_h >= 0 and y_h >= 0 and x_h < Wh and y_h < Hh)
                if v > 0 and in_bounds:
                    draw_gaussian(heatmaps[ch], x_h, y_h, sigma=2.0)
                vis_labels[ch] = int(v if in_bounds else 0)
                if v > 0 and in_bounds:
                    coord_targets[ch, 0] = float(x_h)
                    coord_targets[ch, 1] = float(y_h)
                    coord_weights[ch] = 1.0
                else:
                    coord_targets[ch, 0] = 0.0
                    coord_targets[ch, 1] = 0.0
                    coord_weights[ch] = 0.0

        # Optional SAM2 pseudo segmentation: artifacts/seg_pseudo6/<video>/<name>.png (indexed 0..6)
        try:
            video_id = name.split('_frame_')[0]
            seg_path = os.path.join(self.pseudo6_root, video_id, f'{name}.png')
            if os.path.exists(seg_path):
                seg_im = Image.open(seg_path).convert('L')
                if self.resize_only:
                    canvas = seg_im.resize((self.image_w, self.image_h), resample=Image.NEAREST)
                    if self.aug_rotate_deg > 0.0 and rot_deg != 0.0:
                        canvas = canvas.rotate(rot_deg, resample=Image.NEAREST)
                    seg_small = canvas.resize((Wh, Hh), resample=Image.NEAREST)
                else:
                    # letterbox path
                    s = scale_x
                    nw, nh = int(round(W0 * s)), int(round(H0 * s))
                    seg_resized = seg_im.resize((nw, nh), resample=Image.NEAREST)
                    canvas = Image.new('L', (self.image_size, self.image_size), color=0)
                    canvas.paste(seg_resized, (pl, pt))
                    if self.aug_rotate_deg > 0.0 and rot_deg != 0.0:
                        canvas = canvas.rotate(rot_deg, resample=Image.NEAREST)
                    seg_small = canvas.resize((Wh, Hh), resample=Image.NEAREST)
                seg_arr = np.array(seg_small).astype(np.int64)
                seg6 = seg_arr
        except Exception:
            pass

        # MOTが存在するフレームのみ、未設定(-100)の可視性を枠外(0)として分類監督に含める
        if has_mot:
            vis_labels[vis_labels == -100] = 0

        sample = {
            'image': img_t,
            'prev_image': (prev_img_t if prev_img_t is not None else img_t.clone()),
            'flow_uv': (flow_uv if flow_uv is not None else torch.zeros(2, img_t.shape[-2], img_t.shape[-1], dtype=torch.float32)),
            'fg_mask': (fg_mask if fg_mask is not None else torch.zeros(1, img_t.shape[-2], img_t.shape[-1], dtype=torch.float32)),
            'heatmaps': torch.from_numpy(heatmaps),
            'vis_labels': torch.from_numpy(vis_labels),
            'coord_targets': torch.from_numpy(coord_targets),  # (C,2) in heatmap coords (x,y)
            'coord_weights': torch.from_numpy(coord_weights),  # (C,) weight per kp
            'seg_labels6': torch.from_numpy(seg6),             # (Hh,Wh) int labels 0..6 or -100
            'meta': {
                'mot_path': mot_path,
                'frame_name': name,
                'orig_size': (H0, W0),
                'scale': (scale_x, scale_y) if not letterbox else scale,
                'pad': (pl, pt),
                'letterbox': letterbox,
            }
        }
        return sample
