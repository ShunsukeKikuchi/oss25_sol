#!/usr/bin/env python3
import os, sys, argparse, json, subprocess, time
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm.auto import tqdm
import timm

from oss25.models.kpnet import KPNet

# ---- Layout: matches KPNet (27 channels)
KP_LAYOUT: Dict[int, Tuple[int, int]] = {
    0: (0, 6),   # left hand (6)
    1: (6, 6),   # right hand (6)
    2: (12, 3),  # scissors (3)
    3: (15, 3),  # tweezers (3)
    4: (18, 3),  # needle holder (3)
    5: (21, 6),  # needle (two instances x3 -> 6)
}

def parse_hw(v: str) -> Tuple[int, int]:
    try:
        if isinstance(v, (tuple, list)): return int(v[0]), int(v[1])
        s = str(v).lower().replace('×','x').replace(',','x')
        if 'x' in s:
            a,b = s.split('x'); return int(a), int(b)
        n = int(s); return n, n
    except Exception:
        return 384, 384

# ---- Vectorized coordinate refinement (reshape-safe)
def refine_coords_sigmoid_fast(heatmaps: torch.Tensor):
    B, C, H, W = heatmaps.shape
    prob = heatmaps.sigmoid()
    flat = prob.reshape(B, C, -1)
    idx = flat.argmax(dim=-1)                          # (B, C)
    conf = flat.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    xs0 = (idx % W).float()
    ys0 = (idx // W).float()

    pc = prob.reshape(B * C, 1, H, W)
    patches = F.unfold(pc, kernel_size=3, padding=1)   # (B*C, 9, H*W)
    sel = idx.reshape(-1)
    patch = patches[torch.arange(B * C, device=pc.device), :, sel]  # (B*C,9)
    w = patch / (patch.sum(dim=1, keepdim=True) + 1e-6)

    dx = torch.tensor([-1,0,1,-1,0,1,-1,0,1], device=pc.device, dtype=prob.dtype)
    dy = torch.tensor([-1,-1,-1,0,0,0,1,1,1], device=pc.device, dtype=prob.dtype)
    offx = (w * dx).sum(dim=1)
    offy = (w * dy).sum(dim=1)

    xs = xs0.reshape(-1) + offx
    ys = ys0.reshape(-1) + offy
    coords = torch.stack([xs, ys], dim=-1).reshape(B, C, 2)
    return coords, conf

def row_for(cls: int, tid: int, frame_idx: int, coords: np.ndarray, vis: np.ndarray) -> str:
    base, n_total = KP_LAYOUT[cls]
    if cls == 5 and tid in (5, 6):
        offset = base + (tid - 5) * 3
        n = 3
    else:
        offset = base
        n = n_total if cls != 5 else n_total
    kp: List[str] = []
    for i in range(n):
        x, y = coords[offset + i]
        c = int(vis[offset + i])
        kp.extend([f'{x:.1f}', f'{y:.1f}', f'{c}'])
    bbox = '-1,-1,-1,-1'
    return f'{frame_idx},{tid},{cls},{bbox},' + ','.join(kp)

def write_tracker_file(out_dir: str, video_id: str, rows: List[str]):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{video_id}.csv')
    with open(out_path, 'w') as f:
        for r in rows: f.write(r + '\n')
    return out_path

def ffprobe_meta(path: str) -> Tuple[int,int,Optional[int],float]:
    """Return (w,h,nb_frames_or_None,fps_float)."""
    cmd = [
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,nb_frames,avg_frame_rate",
        "-of","json", path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
    js = json.loads(p.stdout)
    st = js["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    nbf = None
    if "nb_frames" in st and str(st["nb_frames"]).isdigit():
        nbf = int(st["nb_frames"])
    fps = 0.0
    if "avg_frame_rate" in st and st["avg_frame_rate"] not in ("0/0","N/A"):
        num, den = st["avg_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    return w, h, nbf, fps

def spawn_ffmpeg_reader_resize(path: str, out_w: int, out_h: int, hwaccel: bool=False):
    """
    resize_only 前提: アスペクト無視で (out_w,out_h) にスケール。
    """
    vf = f"scale={out_w}:{out_h}:flags=bicubic"
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin"]
    if hwaccel:
        cmd += ["-hwaccel","cuda"]
    cmd += ["-i", path, "-vf", vf,
            "-pix_fmt","rgb24","-f","rawvideo","-vsync","0","pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    frame_bytes = out_w * out_h * 3
    return proc, frame_bytes

def read_exact(stream, nbytes: int) -> bytes:
    """Read exactly nbytes from stream or return b'' on EOF."""
    chunks = []; got = 0
    while got < nbytes:
        b = stream.read(nbytes - got)
        if not b: break
        chunks.append(b); got += len(b)
    return b''.join(chunks)

def compute_flow_pair(prev_rgb: np.ndarray, cur_rgb: np.ndarray, downscale: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    prev_rgb, cur_rgb: (H,W,3) RGB uint8
    returns (flow_uv [2,H,W] float32 ~[-1,1], fg [1,H,W] float32 [-1,1])
    """
    H, W = cur_rgb.shape[:2]
    if downscale > 1:
        cur_s = cv2.resize(cur_rgb, (W//downscale, H//downscale), interpolation=cv2.INTER_LINEAR)
        prv_s = cv2.resize(prev_rgb, (W//downscale, H//downscale), interpolation=cv2.INTER_LINEAR)
    else:
        cur_s, prv_s = cur_rgb, prev_rgb
    g1 = cv2.cvtColor(cur_s, cv2.COLOR_RGB2GRAY)
    g0 = cv2.cvtColor(prv_s, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5,3,15,3,5,1.2,0)  # (h,w,2)
    if downscale > 1:
        flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
        g0 = cv2.resize(g0, (W, H), interpolation=cv2.INTER_LINEAR)
        g1 = cv2.resize(g1, (W, H), interpolation=cv2.INTER_LINEAR)
    diff = cv2.absdiff(g1, g0).astype(np.float32) / 255.0
    flow_uv = (flow.transpose(2,0,1).astype(np.float32) / 20.0).clip(-1.0, 1.0)  # (2,H,W)
    fg = (diff[None, ...] * 2.0 - 1.0).astype(np.float32)                         # (1,H,W)
    return flow_uv, fg

def infer_one_video_ffmpeg_stride29(model: KPNet, video_path: str,
                                    src_w: int, src_h: int,
                                    img_h: int, img_w: int, hm_h: int, hm_w: int,
                                    device: torch.device, batch_size: int,
                                    mean: torch.Tensor, std: torch.Tensor,
                                    use_hwaccel: bool, flow_downscale: int) -> List[str]:
    """
    ffmpegで (img_w,img_h) にリサイズしたRGB24 rawを読み、フレーム idx%29==0 のみ推論。
    フローは「直前フレーム vs 選択フレーム」で計算。
    """
    proc, frame_bytes = spawn_ffmpeg_reader_resize(video_path, img_w, img_h, hwaccel=use_hwaccel)
    assert proc.stdout is not None

    rows: List[str] = []
    last_frame_uint8: Optional[np.ndarray] = None

    # スケール係数（heatmap->resized->original）
    sx_hm = img_w / float(hm_w)
    sy_hm = img_h / float(hm_h)
    sx = img_w / float(src_w)
    sy = img_h / float(src_h)

    # 進捗：選択フレーム数ベース
    _, _, nb_frames, _ = ffprobe_meta(video_path)
    total_sel = (nb_frames + 29 - 1) // 29 if nb_frames and nb_frames > 0 else None
    pbar = tqdm(total=total_sel, desc=os.path.basename(video_path), unit='sel-f')

    t0 = time.time(); done_sel = 0
    frame_idx = 0

    # バッチ用バッファ（選択フレームのみを貯める）
    batch_frames: List[np.ndarray] = []
    batch_flows:  List[np.ndarray] = []
    batch_fgs:    List[np.ndarray] = []
    batch_fidx:   List[int] = []

    mean = mean.to(device); std = std.to(device)

    try:
        while True:
            raw = read_exact(proc.stdout, frame_bytes)
            if not raw:
                break
            cur_uint8 = np.frombuffer(raw, dtype=np.uint8).reshape(img_h, img_w, 3)

            # フローの連続性用に直前フレームは常に保持
            if last_frame_uint8 is None:
                last_frame_uint8 = cur_uint8.copy()

            # 選択フレームか？
            if frame_idx % 29 == 0:
                flow_uv, fg = compute_flow_pair(last_frame_uint8, cur_uint8, flow_downscale)
                batch_frames.append(cur_uint8)
                batch_flows.append(flow_uv)
                batch_fgs.append(fg)
                batch_fidx.append(frame_idx)

                # バッチが満たされたら推論
                if len(batch_frames) >= batch_size:
                    _run_infer_batch(model, batch_frames, batch_flows, batch_fgs, batch_fidx,
                                     device, mean, std, img_h, img_w,
                                     sx_hm, sy_hm, sx, sy, rows)
                    done_sel += len(batch_frames)
                    elapsed = max(time.time() - t0, 1e-6)
                    pbar.update(len(batch_frames))
                    pbar.set_postfix(fps=f"{done_sel/elapsed:.1f}")
                    batch_frames.clear(); batch_flows.clear(); batch_fgs.clear(); batch_fidx.clear()

            # 次のフレーム用に更新
            last_frame_uint8 = cur_uint8
            frame_idx += 1

        # 端数バッチを処理
        if batch_frames:
            _run_infer_batch(model, batch_frames, batch_flows, batch_fgs, batch_fidx,
                             device, mean, std, img_h, img_w,
                             sx_hm, sy_hm, sx, sy, rows)
            done_sel += len(batch_frames)
            elapsed = max(time.time() - t0, 1e-6)
            pbar.update(len(batch_frames))
            pbar.set_postfix(fps=f"{done_sel/elapsed:.1f}")

        proc.stdout.close()
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
        pbar.close()

    return rows

def _run_infer_batch(model: KPNet,
                     frames_uint8: List[np.ndarray],
                     flows_list: List[np.ndarray],
                     fgs_list: List[np.ndarray],
                     fidx_list: List[int],
                     device: torch.device,
                     mean: torch.Tensor, std: torch.Tensor,
                     img_h: int, img_w: int,
                     sx_hm: float, sy_hm: float, sx: float, sy: float,
                     rows_out: List[str]) -> None:
    """
    選択フレームのみでバッチ推論を行い、rows_out に追記する。
    """
    B = len(frames_uint8)
    frames = np.stack(frames_uint8, axis=0)                    # (B,H,W,3) uint8
    flows  = np.stack(flows_list,  axis=0)                     # (B,2,H,W) float32
    fgs    = np.stack(fgs_list,    axis=0)                     # (B,1,H,W) float32

    # to device & normalize
    x_rgb = torch.from_numpy(frames).to(device, non_blocking=True).permute(0,3,1,2).float().div_(255.0)
    x_rgb = (x_rgb - mean) / std
    flow_uv = torch.from_numpy(flows).to(device, non_blocking=True).float().clamp_(-1.0, 1.0)
    fg = torch.from_numpy(fgs).to(device, non_blocking=True).float()

    x = torch.cat([x_rgb, flow_uv, fg], dim=1)                 # (B,6,H,W)

    need = getattr(model, 'in_chans', x.shape[1]) - x.shape[1]
    if need > 0:
        pad = torch.zeros(B, need, img_h, img_w, dtype=x.dtype, device=device)
        x = torch.cat([x, pad], dim=1)
    elif need < 0:
        raise RuntimeError(f"in_chans mismatch: model expects {getattr(model,'in_chans','?')} but provided {x.shape[1]}")

    with torch.inference_mode():
        hm, vis_logits = model(x)

    coords_hm, _ = refine_coords_sigmoid_fast(hm.float())      # (B,27,2)

    # heatmap -> resized image -> original
    coords_img = coords_hm.clone()
    coords_img[..., 0] = coords_img[..., 0] * sx_hm
    coords_img[..., 1] = coords_img[..., 1] * sy_hm
    coords_img[..., 0] = coords_img[..., 0] / (sx + 1e-12)
    coords_img[..., 1] = coords_img[..., 1] / (sy + 1e-12)

    vis = vis_logits.softmax(dim=-1).argmax(dim=-1)            # (B,27)
    c_np = coords_img.cpu().numpy()
    v_np = vis.cpu().numpy()

    for bi in range(B):
        fi = fidx_list[bi]
        rows_out.append(row_for(0, 0, fi, c_np[bi], v_np[bi]))
        rows_out.append(row_for(1, 1, fi, c_np[bi], v_np[bi]))
        rows_out.append(row_for(2, 2, fi, c_np[bi], v_np[bi]))
        rows_out.append(row_for(3, 3, fi, c_np[bi], v_np[bi]))
        rows_out.append(row_for(4, 4, fi, c_np[bi], v_np[bi]))
        rows_out.append(row_for(5, 5, fi, c_np[bi], v_np[bi]))
        rows_out.append(row_for(5, 6, fi, c_np[bi], v_np[bi]))

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description='ffmpeg pipe inference (resize_only, flow enforced, stride=29)')
    ap.add_argument('--input-dir', default='/input')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', default='/output')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--image-size', default='384')
    ap.add_argument('--heatmap-size', default='128')
    ap.add_argument('--batch-size', type=int, default=8, help='batch of SELECTED frames')
    ap.add_argument('--flow-downscale', type=int, default=1, help='1=full-res flow; >1 downsamples flow compute')
    ap.add_argument('--ffmpeg-hwaccel', action='store_true', help='Use ffmpeg -hwaccel cuda if available')
    ap.add_argument('--compile', action='store_true')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    img_h, img_w = parse_hw(args.image_size)
    hm_h, hm_w = parse_hw(args.heatmap_size)

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

    # load checkpoint
    sd = torch.load(args.ckpt, map_location='cpu')
    weights = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd

    def _guess_in_chans(w):
        for k in ['encoder.stem_0.weight','encoder.stem.conv.weight']:
            if k in w and hasattr(w[k],'dim') and w[k].dim()==4: return int(w[k].shape[1])
        for k,v in w.items():
            if (k.endswith('stem_0.weight') or k.endswith('stem.conv.weight')) and hasattr(v,'dim') and v.dim()==4:
                return int(v.shape[1])
        return None

    ckpt_in = _guess_in_chans(weights) or 6
    if ckpt_in < 6:
        raise RuntimeError(f"flow is enforced but checkpoint in_chans={ckpt_in}. Use a checkpoint trained with in_chans>=6 (RGB+flow+fg).")

    model = KPNet(image_size=(img_h,img_w), heatmap_size=(hm_h,hm_w), in_chans=ckpt_in)
    _ = model.load_state_dict(weights, strict=False)
    model = model.to(device).eval()
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # mean/std once
    try:
        data_cfg = timm.data.resolve_model_data_config(model.encoder)
        mean = torch.tensor(data_cfg.get('mean',(0.5,0.5,0.5)), dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor(data_cfg.get('std', (0.5,0.5,0.5)), dtype=torch.float32).view(1,3,1,1)
    except Exception:
        mean = torch.tensor((0.5,0.5,0.5), dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor((0.5,0.5,0.5), dtype=torch.float32).view(1,3,1,1)

    # list videos
    mp4s = []
    for root, _, files in os.walk(args.input_dir):
        for name in files:
            if name.lower().endswith('.mp4'):
                mp4s.append(os.path.join(root, name))
    mp4s.sort()
    if not mp4s:
        print(f'[error] no input videos found in {args.input_dir}', file=sys.stderr); sys.exit(1)

    for mp4 in mp4s:
        vid = os.path.splitext(os.path.basename(mp4))[0]
        src_w, src_h, _, _ = ffprobe_meta(mp4)
        rows = infer_one_video_ffmpeg_stride29(model, mp4, src_w, src_h,
                                               img_h, img_w, hm_h, hm_w,
                                               device, args.batch_size,
                                               mean, std,
                                               use_hwaccel=args.ffmpeg_hwaccel,
                                               flow_downscale=args.flow_downscale)
        out_path = write_tracker_file(args.out_dir, vid, rows)
        print(f'[pred] wrote {out_path}')

if __name__ == '__main__':
    main()
