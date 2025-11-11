#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full evaluator for video editing with per-instance metrics — now supports
both video files AND image-sequence folders, plus flexible mask filenames.

Metrics
-------
- CIA / LTF / IA / LTC  (instance-aware; uses per-frame masks if provided)
- PSNR / SSIM           (original vs edited, full-frame)
- LPIPS                 (perceptual distance; original vs edited, full-frame; lower is better)
- CLIP-F / CLIP-T       (text–video alignment using the global edited description)
                        NOTE: In this version CLIP-F / CLIP-T are SCALED BY 100 to match common paper reporting.
- Warp-Err              (temporal warp error on edited video)

Inputs
------
Use a JSON spec describing the edited instances. Example:
{
  "description": "A man in a red shirt walking a black dog on the beach.",
  "instances": [
    {"instance_id": "1", "source_caption": "a man wearing a white shirt", "target_caption": "a man wearing a red shirt"},
    {"instance_id": "2", "source_caption": "a brown dog", "target_caption": "a black dog"}
  ],
  "mask_root": "Video2/data/clips/clip_001/masks"  // optional
}

CLI (video files)
-----------------
python measure.py \
  --orig /path/original.mp4 \
  --edit /path/edited.mp4 \
  --spec /path/spec.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 240 \
  --save_json ./metrics.json

CLI (image sequences)
--------------------
python measure.py \
  --orig_frames_root data/clips/clip_001/frames_orig \
  --edit_frames_root data/clips/clip_001/frames \
  --spec data/clips/clip_001/spec.json \
  --device cuda \
  --save_json data/clips/clip_001/metrics.json

Dependencies
------------
pip install numpy pillow opencv-python torch torchvision open_clip_torch transformers scikit-image tqdm lpips

Notes
-----
• If masks are not provided, the script falls back to frame-level (mask-free) approximations
  for CIA/LTF/IA/LTC by treating the whole frame as the region; CIA then behaves like 2-class IA.
• With masks, place them under mask_root/<instance_id> and name them to match frame filenames
  (e.g., 000.png, 001.png). The loader also accepts zero-padded fallbacks {t:05d}.png / {t:03d}.png.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# ---------------------- CLIP backend ----------------------
class ClipEncoder:
    def __init__(self, device: str = "cpu", model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = torch.device(device)
        self.backend = None
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        try:
            import open_clip
            self.backend = "open_clip"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.model.eval().to(self.device)
            self.tokenizer = open_clip.get_tokenizer(model_name)
        except Exception:
            from transformers import CLIPProcessor, CLIPModel
            self.backend = "hf"
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.model.eval()
            self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        if len(pil_images) == 0:
            return torch.empty(0, 512, device=self.device)
        if self.backend == "open_clip":
            imgs = torch.stack([self.preprocess(im).to(self.device) for im in pil_images])
            feats = self.model.encode_image(imgs)
        else:
            feats_list = []
            for im in pil_images:
                x = self.preprocess(im).unsqueeze(0).to(self.device)  # batch of size 1
                with torch.no_grad():
                    f = self.model.encode_image(x)
                feats_list.append(f.cpu())
            feats = torch.cat(feats_list, dim=0)
        feats = F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        if self.backend == "open_clip":
            toks = self.tokenizer(texts).to(self.device)
            feats = self.model.encode_text(toks)
        else:
            inputs = self.preprocess(text=texts, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_text_features(**inputs)
        feats = F.normalize(feats, dim=-1)
        return feats

# ---------------------- Video / Image-sequence I/O ----------------------

def read_video_frames(path: str, stride: int = 1, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], List[str]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames = []
    names = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame[:, :, ::-1].copy())  # BGR->RGB
            names.append(f"{len(names):03d}.png")  # synthetic names
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames, names


def read_image_sequence(root: str, stride: int = 1, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], List[str]]:
    p = Path(root)
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    frames: List[np.ndarray] = []
    names: List[str] = []
    for i, f in enumerate(files):
        if i % stride != 0:
            continue
        img = Image.open(f).convert("RGB")
        frames.append(np.array(img))
        names.append(f.name)
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames, names

# ---------------------- Helpers ----------------------

def list_mask_paths(mask_root: Optional[Path], instance_id: str, T: int,
                    frame_names: Optional[List[str]] = None, debug: bool = True) -> List[Optional[Path]]:
    if mask_root is None:
        return [None] * T
    out: List[Optional[Path]] = []
    mask_root = Path(str(mask_root).strip())   # ← add this
    instance_id = str(instance_id).strip()     # ← and this
    miss = 0
    for t in range(T):
        cand: Optional[Path] = None
        tried = []
        if frame_names and t < len(frame_names):
            fn = frame_names[t]
            p1 = mask_root / str(instance_id) / fn
            tried.append(p1)
            if p1.exists():
                cand = p1
        if cand is None:
            p2 = mask_root / str(instance_id) / f"{t:05d}.png"
            p3 = mask_root / str(instance_id) / f"{t:03d}.png"
            tried += [p2, p3]
            cand = p2 if p2.exists() else (p3 if p3.exists() else None)
        if debug:
            if cand is None:
                miss += 1
                print(f"[MASK][MISS] inst={instance_id} t={t} name={frame_names[t] if frame_names else None}")
                print("             tried:", " | ".join(str(x) for x in tried))
            else:
                print(f"[MASK][HIT ] inst={instance_id} t={t} -> {cand.name}")
        out.append(cand)
    if debug:
        hit = T - miss
        print(f"[MASK][SUMMARY] inst={instance_id}: hits={hit} / {T}, misses={miss}")
    return out



def tight_bbox(mask_arr: np.ndarray, pad: int = 4) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = mask_arr.shape
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(h - 1, y1 + pad); x1 = min(w - 1, x1 + pad)
    return int(y0), int(x0), int(y1 + 1), int(x1 + 1)


def crop_by_mask(image: Image.Image, mask_img: Optional[Image.Image], fallback_full_frame: bool = True) -> Optional[Image.Image]:
    if mask_img is None:
        return image.copy() if fallback_full_frame else None
    m = np.array(mask_img.convert("L"))
    bb = tight_bbox(m, pad=4)
    if bb is None:
        return image.copy() if fallback_full_frame else None
    y0, x0, y1, x1 = bb
    return image.crop((x0, y0, x1, y1))

# ---------------------- Metrics ----------------------

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20.0 * np.log10(PIXEL_MAX / np.sqrt(mse))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        s, _ = ssim_fn(img1, img2, channel_axis=2, full=True)
        return float(s)
    except Exception:
        def _to_gray(x):
            return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        y1 = _to_gray(img1); y2 = _to_gray(img2)
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        mu1 = cv2.GaussianBlur(y1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(y2, (11, 11), 1.5)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(y1 * y1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(y2 * y2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(y1 * y2, (11, 11), 1.5) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean())

# ---------------------- LPIPS ----------------------
class LPIPSComputer:
    """
    Thin wrapper around the 'lpips' package.
    Produces a lower-is-better perceptual distance between two images.
    Returns NaN if lpips isn't available.
    """
    def __init__(self, device: str = "cpu", net: str = "alex"):
        self.device = torch.device(device)
        self.ok = False
        self.loss_fn = None
        try:
            import lpips  # pip install lpips
            self.loss_fn = lpips.LPIPS(net=net).to(self.device).eval()
            self.ok = True
        except Exception:
            # Gracefully degrade if lpips isn't installed or fails to init
            self.loss_fn = None
            self.ok = False

    @torch.no_grad()
    def distance(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        img1/img2: HxWx3 uint8 RGB
        Returns a scalar float LPIPS distance.
        """
        if not self.ok:
            return float("nan")

        # Convert [0,255] RGB -> [-1,1] CHW float32
        def to_tensor(x: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
            t = t * 2.0 - 1.0
            return t.unsqueeze(0).to(self.device)

        t1 = to_tensor(img1)
        t2 = to_tensor(img2)
        d = self.loss_fn(t1, t2)
        return float(d.item())

# ---------------------- Temporal ----------------------

def warp_error_mean_abs(frames: List[np.ndarray]) -> float:
    if len(frames) < 2:
        return float("nan")
    errs = []
    for t in range(len(frames) - 1):
        f0 = frames[t]
        f1 = frames[t + 1]
        g0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, pyr_scale=0.5, levels=3,
                                            winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        h, w = g0.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(f0, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        err = np.mean(np.abs(warped.astype(np.float32) - f1.astype(np.float32))) / 255.0
        errs.append(err)
    return float(np.mean(errs)) if errs else float("nan")

# ---------------------- Core computation ----------------------

def compute_instance_embeddings(edit_frames: List[np.ndarray], mask_paths: List[Optional[Path]], enc: ClipEncoder) -> List[torch.Tensor]:
    crops: List[Image.Image] = []
    for t, f in enumerate(edit_frames):
        img = Image.fromarray(f)
        mpath = mask_paths[t] if t < len(mask_paths) else None
        mask_img = Image.open(mpath).convert("L") if (mpath and mpath.exists()) else None
        crop = crop_by_mask(img, mask_img, fallback_full_frame=True)
        crops.append(crop)
    if len(crops) == 0:
        return []
    emb = enc.encode_images(crops)  # [T,D]
    return [e for e in emb]


def instance_texts(inst: Dict) -> Tuple[str, str]:
    src = (inst.get("source_caption") or "object").strip()
    tgt = (inst.get("target_caption") or "object").strip()
    return src, tgt


def compute_metrics(orig_frames: List[np.ndarray], edit_frames: List[np.ndarray], spec: Dict, device: str = "cpu",
                     frame_names: Optional[List[str]] = None) -> Dict[str, float]:
    T = min(len(orig_frames), len(edit_frames))
    orig_frames = orig_frames[:T]
    edit_frames = edit_frames[:T]

    # 1) PSNR/SSIM (full-frame)
    psnrs = [psnr(o, e) for o, e in zip(orig_frames, edit_frames)]
    ssims = [ssim(o, e) for o, e in zip(orig_frames, edit_frames)]
    PSNR = float(np.mean(psnrs))
    SSIM = float(np.mean(ssims))

    # 1b) LPIPS (full-frame, lower is better)
    lpips_comp = LPIPSComputer(device=device)
    lpips_vals = [lpips_comp.distance(o, e) for o, e in zip(orig_frames, edit_frames)]
    LPIPS = float(np.mean(lpips_vals)) if len(lpips_vals) else float("nan")

    # 2) Warp-Err on edited video
    WarpErr = warp_error_mean_abs(edit_frames)

    # 3) CLIP global metrics (use edited description)
    enc = ClipEncoder(device=device)
    pil_frames = [Image.fromarray(f) for f in edit_frames]
    img_emb = enc.encode_images(pil_frames)                          # [T, D], L2-normalized
    target_desc = (spec.get("description") or "").strip() or "the scene"
    txt_tgt_desc = enc.encode_texts([target_desc])[0]                # [D], L2-normalized

    # CLIP-T: average image–text cosine over frames (×100)
    per_frame_it = (img_emb @ txt_tgt_desc)                          # [T]
    CLIP_T = float(per_frame_it.mean().item() * 100.0)

    # CLIP-F: average cosine between consecutive frame embeddings (×100)
    if img_emb.shape[0] >= 2:
        # cosine between img_emb[t] and img_emb[t+1] since embeddings are normalized
        per_pair_ff = (img_emb[:-1] * img_emb[1:]).sum(dim=1)        # [T-1]
        CLIP_F = float(per_pair_ff.mean().item() * 100.0)
    else:
        CLIP_F = float("nan")

    # 4) Instance-aware metrics (CIA/LTF/IA/LTC)
    insts = spec.get("instances", [])
    mask_root = Path(spec["mask_root"]) if spec.get("mask_root") else None

    # Encode per-instance crops
    inst_ids = [str(inst.get("instance_id", i+1)) for i, inst in enumerate(insts)]
    inst_emb_seq: Dict[str, List[torch.Tensor]] = {}
    for inst_id in inst_ids:
        masks = list_mask_paths(mask_root, inst_id, T, frame_names=frame_names)
        inst_emb_seq[inst_id] = compute_instance_embeddings(edit_frames, masks, enc)

    # Aggregate instance embeddings (mean over time)
    inst_img_emb: Dict[str, torch.Tensor] = {}
    for inst_id, seq in inst_emb_seq.items():
        if len(seq) == 0:
            continue
        mat = torch.stack(seq, dim=0)
        inst_img_emb[inst_id] = F.normalize(mat.mean(dim=0, keepdim=False), dim=-1)

    # Text embeddings per instance
    id2src: Dict[str, torch.Tensor] = {}
    id2tgt: Dict[str, torch.Tensor] = {}
    for inst_id, inst in zip(inst_ids, insts):
        src, tgt = instance_texts(inst)
        id2src[inst_id] = enc.encode_texts([src])[0]
        id2tgt[inst_id] = enc.encode_texts([tgt])[0]

    # Build arrays of valid instances
    valid_ids = [iid for iid in inst_ids if iid in inst_img_emb]
    if len(valid_ids) == 0:
        return {
            "PSNR": PSNR, "SSIM": SSIM, "LPIPS": LPIPS,
            "Warp-Err": WarpErr,
            "CLIP-F": CLIP_F, "CLIP-T": CLIP_T,
            "CIA": float("nan"), "LTF": float("nan"), "IA": float("nan"), "LTC": float("nan")
        }

    # --- 新版：CIA 仍用時間平均；LTF/IA 改為逐幀再平均 ---
    
    # 先保留你的 CIA（用時間平均的影像嵌入）
    img_mat = torch.stack([inst_img_emb[iid] for iid in valid_ids], dim=0)  # [N,D]
    tgt_mat = torch.stack([id2tgt[iid] for iid in valid_ids], dim=0)        # [N,D]
    S = img_mat @ tgt_mat.T                                                 # [N,N]
    preds = torch.argmax(S, dim=1)
    cia = (preds == torch.arange(S.shape[0], device=S.device)).float().mean().item()
    
    # LTF：對每個 instance，逐幀和該 instance 的 target 文本做 cos，相似度再對幀取平均，最後跨 instance 取平均
    ltf_vals = []
    for iid in valid_ids:
        seq = inst_emb_seq.get(iid, [])
        if len(seq) == 0:
            continue
        txt_tgt = id2tgt[iid]  # 已 L2 normalize
        # 逐幀相似度
        sims = torch.stack([ (e * txt_tgt).sum() for e in seq ], dim=0)  # [T_i]
        ltf_vals.append(float(sims.mean().item()))
    ltf = float(np.mean(ltf_vals)) if ltf_vals else float("nan")
    
    # IA：對每個 instance，分別計算 逐幀與 target 的 cos 平均、逐幀與 source 的 cos 平均，再比較
    ia_list = []
    for iid in valid_ids:
        seq = inst_emb_seq.get(iid, [])
        if len(seq) == 0:
            continue
        txt_tgt = id2tgt[iid]
        txt_src = id2src[iid]
        # sim_t = torch.stack([ (e * txt_tgt).sum() for e in seq ], dim=0).mean().item()
        # sim_s = torch.stack([ (e * txt_src).sum() for e in seq ], dim=0).mean().item()
        # ia_list.append(1.0 if sim_t >= sim_s else 0.0)

        # 建逐幀相似度 Tensor（與 target、與 source）
        sims_t = torch.stack([(e * txt_tgt).sum() for e in seq], dim=0)  # [T]
        sims_s = torch.stack([(e * txt_src).sum() for e in seq], dim=0)  # [T]
    
        # 逐幀投票：target 勝過 source 記 1，否則 0
        votes = (sims_t > sims_s).float().mean().item()  # 比較得到 bool，再轉 float 求平均
        ia_list.append(votes)

    ia = float(np.mean(ia_list)) if ia_list else float("nan")

    # LTC: temporal consistency of per-frame instance embeddings
    ltc_vals = []
    for iid in valid_ids:
        seq = inst_emb_seq.get(iid, [])
        if len(seq) < 2:
            continue
        mat = torch.stack(seq, dim=0)  # [T,D]
        sims = F.cosine_similarity(mat[:-1], mat[1:], dim=1)
        ltc_vals.append(float(sims.mean().item()))
    ltc = float(np.mean(ltc_vals)) if ltc_vals else float("nan")

    return {
        "PSNR": float(PSNR), "SSIM": float(SSIM), "LPIPS": float(LPIPS),
        "Warp-Err": float(WarpErr),
        "CLIP-F": float(CLIP_F), "CLIP-T": float(CLIP_T),  # scaled by 100
        "CIA": float(cia), "LTF": float(ltf), "IA": float(ia), "LTC": float(ltc)
    }

# ---------------------- Orchestration ----------------------

def main():
    ap = argparse.ArgumentParser()
    # Either videos OR image sequences
    ap.add_argument("--orig", type=str, default=None, help="Path to original video (e.g., .mp4)")
    ap.add_argument("--edit", type=str, default=None, help="Path to edited video (e.g., .mp4)")
    ap.add_argument("--orig_frames_root", type=str, default=None, help="Folder of original frames (PNG/JPG)")
    ap.add_argument("--edit_frames_root", type=str, default=None, help="Folder of edited frames (PNG/JPG)")

    ap.add_argument("--spec", type=str, required=True, help="JSON with description, instances and optional mask_root")
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--sample_stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--save_json", type=str, default=None)
    args = ap.parse_args()

    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    # Read frames (video OR image sequence)
    frame_names: List[str]
    if args.orig_frames_root and args.edit_frames_root:
        orig_frames, _orig_names = read_image_sequence(args.orig_frames_root, args.sample_stride, args.max_frames)
        edit_frames, frame_names = read_image_sequence(args.edit_frames_root, args.sample_stride, args.max_frames)
    elif args.orig and args.edit:
        orig_frames, _orig_names = read_video_frames(args.orig, stride=args.sample_stride, max_frames=args.max_frames)
        edit_frames, frame_names = read_video_frames(args.edit, stride=args.sample_stride, max_frames=args.max_frames)
    else:
        raise ValueError("Provide either --orig/--edit (videos) OR --orig_frames_root/--edit_frames_root (image folders).")

    if len(orig_frames) == 0 or len(edit_frames) == 0:
        raise RuntimeError("No frames read from one or both inputs.")

    metrics = compute_metrics(orig_frames, edit_frames, spec, device=args.device, frame_names=frame_names)

    print("\n==== Metrics ====")
    for k in ["CIA","LTF","IA","LTC","CLIP-F","CLIP-T","PSNR","SSIM","LPIPS","Warp-Err"]:
        v = metrics[k]
        if isinstance(v, float) and np.isfinite(v):
            if k in ("CLIP-F","CLIP-T"):
                print(f"{k:8s}: {v:.2f}")  # show two decimals since it's 0–100 scale now
            else:
                print(f"{k:8s}: {v:.4f}")
        else:
            print(f"{k:8s}: {v}")

    if args.save_json:
        p = Path(args.save_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {p}")

if __name__ == "__main__":
    main()
