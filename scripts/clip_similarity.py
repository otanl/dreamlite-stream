"""Compute CLIP text-image similarity for output mp4s.

For each output video, decodes frames at a sample interval, encodes them
with CLIP image encoder, encodes the prompt with CLIP text encoder, and
averages cosine similarity across frames.

Reports per-(prompt, sequence) similarity and aggregates per-prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_ROOT = Path(__file__).resolve().parent.parent


PROMPTS = {
    "oilpaint":  "transfer this to oil painting style, vibrant colors",
    "comic":     "transfer this to comic book style, halftone shading, bold outlines",
    "ukiyoe":    "transfer this to ukiyo-e woodblock print style, flat colors, line art",
    "vangogh":   "transfer this to van gogh impressionist style, swirling brushstrokes",
}


def read_video_frames(path: str, every: int = 4, max_frames: int = 32):
    cap = cv2.VideoCapture(path)
    out = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % every == 0:
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if len(out) >= max_frames:
                break
        idx += 1
    cap.release()
    return out


@torch.no_grad()
def compute_clip_sim(model, processor, frames_np, prompt: str, device):
    pils = [Image.fromarray(f) for f in frames_np]
    inputs = processor(
        text=[prompt], images=pils, return_tensors="pt", padding=True,
    ).to(device)
    out = model(**inputs)
    # logits_per_image is (B, 1); compute cosine via image_embeds / text_embeds
    img_emb = out.image_embeds            # (B, D)
    txt_emb = out.text_embeds             # (1, D)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    cos = (img_emb @ txt_emb.T).squeeze(-1)  # (B,)
    return cos.cpu().tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--frame_stride", type=int, default=4,
                   help="sample every Nth frame")
    p.add_argument("--max_frames_per_video", type=int, default=32)
    p.add_argument("--out_jsonl", default=str(_ROOT / "out" / "clip_similarity.jsonl"))
    p.add_argument("--heldout_dir", default=str(_ROOT / "out" / "heldout_prompts_eval"))
    args = p.parse_args()

    print(f"[load] {args.clip_model}")
    model = CLIPModel.from_pretrained(args.clip_model).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    # Define which videos to score
    # In-domain (oil painting champion) + 3 held-out prompts
    jobs = []
    champion_dir = _ROOT / "out" / "champion" / "champion"
    for mp4 in sorted(champion_dir.glob("*.mp4")):
        jobs.append(("oilpaint", mp4.stem, str(mp4)))

    heldout_dir = Path(args.heldout_dir)
    for mp4 in sorted(heldout_dir.glob("*_*.mp4")):
        stem = mp4.stem  # e.g. "comic_blackswan"
        tag, seq = stem.split("_", 1)
        if tag in PROMPTS and tag != "oilpaint":
            jobs.append((tag, seq, str(mp4)))

    print(f"[jobs] {len(jobs)} videos to score")
    results = []
    for tag, seq, path in jobs:
        frames = read_video_frames(path, args.frame_stride, args.max_frames_per_video)
        if not frames:
            print(f"  skip {tag}/{seq}: no frames")
            continue
        sims = compute_clip_sim(model, processor, frames, PROMPTS[tag], args.device)
        row = {
            "prompt_tag": tag, "sequence": seq, "n_frames": len(frames),
            "clip_sim_mean": float(np.mean(sims)),
            "clip_sim_std": float(np.std(sims)),
        }
        results.append(row)
        print(f"  {tag:9s} {seq:18s} n={len(frames):3d} clip_sim={row['clip_sim_mean']:.4f} ± {row['clip_sim_std']:.4f}")

    # Aggregates per prompt_tag
    print()
    print("=" * 70)
    print("Per-prompt CLIP similarity aggregates:")
    print(f"  {'prompt_tag':<12s} {'N':>3s}  {'mean':>7s}  {'std':>7s}")
    by_tag = {}
    for r in results:
        by_tag.setdefault(r["prompt_tag"], []).append(r["clip_sim_mean"])
    for tag in ["oilpaint", "comic", "ukiyoe", "vangogh"]:
        if tag in by_tag:
            xs = by_tag[tag]
            print(f"  {tag:<12s} {len(xs):>3d}  {mean(xs):>7.4f}  "
                  f"{stdev(xs) if len(xs)>1 else 0:>7.4f}")

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[saved] {args.out_jsonl}")


if __name__ == "__main__":
    main()
