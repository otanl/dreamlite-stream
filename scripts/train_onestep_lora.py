"""Train a one-step LoRA: make ONE denoising step reproduce the K-step output.

Setting (the lean path, see dreamlite_stream/lean.py): every frame uses the
same init-noise tensor and the prompt embedding is computed once per clip, so
the K-step output is a deterministic function of the input frame. Each
training frame therefore has exactly one target, and distilling K steps into
one is paired regression:

    student:  x0 = eps - unet_lora([eps | ref], t=1)        (one step)
    target:   the K-step Euler output from the same eps, ref and prompt
    loss:     Huber(latent) + w * LPIPS(decoded crop)

The pixel-space perceptual term is what keeps the result sharp; latent L2
alone is the documented cause of blur in one-step students. Nothing is
trained except the LoRA, and the base UNet stays in bf16.

The LoRA is tied to the resolution (--size) and the init-noise tensor it was
trained with, and to the prompts it saw. Train one per resolution. The noise
tensor is drawn from --seed and stored in the file, so the LoRA does not
depend on a later run drawing the same tensor again.

Data: frames from any folders of clips (--clips_dirs). Each clip is paired
with a few of the prompts; the prompt embedding is taken from the clip's
first frame, exactly as the lean path does at run time. Teacher targets are
cached under --cache_dir so reruns skip that stage.

Usage:
    python scripts/train_onestep_lora.py --size 768
    python scripts/train_onestep_lora.py --size 640 --steps 3000 --rank 32 --targets attn,ff

Then run with the result merged into the UNet:
    python scripts/champion_eval.py --lean --size 768 \
        --onestep_lora runs/onestep_lora_768/onestep_lora.safetensors

Re-score an existing LoRA without training:
    python scripts/train_onestep_lora.py --size 768 --eval_only

Outputs (under --out_dir, default runs/onestep_lora_<size>/):
    onestep_lora.safetensors   EMA weights + training metadata
    args.json, log.jsonl       recipe and the periodic held-out metrics
    final_eval.json            held-out comparison with the base UNet at 1 and 2 steps
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from dreamlite import DreamLiteMobilePipeline  # noqa: E402

from dreamlite_stream import onestep_lora  # noqa: E402
from dreamlite_stream import pipeline_ops as ops  # noqa: E402
from dreamlite_stream.metrics import compute_temporal, hf_density  # noqa: E402
from dreamlite_stream.runtime import iter_video_frames  # noqa: E402

DEV, BF16, FP16 = "cuda", torch.bfloat16, torch.float16

DEFAULT_PROMPTS = [
    "transfer this to oil painting style, vibrant colors",
    "transfer this to watercolor painting style, soft edges",
    "transfer this to pencil sketch style, fine line work",
    "transfer this to anime art style, clean cel shading",
    "transfer this to 3D render style, ray-traced lighting",
    "transfer this to comic book style, halftone shading, bold outlines",
    "transfer this to ukiyo-e woodblock print style, flat colors, line art",
    "transfer this to van gogh impressionist style, swirling brushstrokes",
]
# Never trained on; reported separately to show how far the LoRA carries.
UNSEEN_PROMPTS = [
    "transfer this to pixel art style, 16-bit game graphics",
    "transfer this to ink wash painting style, sumi-e brushwork",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--seed", type=int, default=42,
                   help="Noise seed of the shared init-noise tensor; must match the seed used at run time.")
    p.add_argument("--teacher_steps", type=int, default=4)
    p.add_argument("--clips_dirs", nargs="+", default=[
        str(_ROOT / "assets" / "davis_heldout20_mp4"), str(_ROOT / "assets" / "crossds_512"),
    ])
    p.add_argument("--frames_per_clip", type=int, default=16)
    p.add_argument("--frame_stride", type=int, default=3)
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--prompts_per_clip", type=int, default=3)
    p.add_argument("--eval_dir", default=str(_ROOT / "assets" / "davis_mp4"))
    p.add_argument("--eval_clips", nargs="+", default=["blackswan", "parkour", "bmx-trees", "libby"])
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=float, default=None, help="Defaults to --rank.")
    p.add_argument("--targets", default="attn,ff",
                   help="Comma-separated layer groups: attn, ff, pw (see dreamlite_stream/onestep_lora.py).")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lpips_weight", type=float, default=5.0)
    p.add_argument("--crop", type=int, default=384, help="Side of the decoded crop the LPIPS term sees.")
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--max_memory_fraction", type=float, default=0.90,
                   help="Fail with out-of-memory above this share of VRAM instead of spilling to shared memory.")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--skip_final_eval", action="store_true")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; run the final evaluation on the LoRA already in --out_dir.")
    args = p.parse_args()
    if args.size % 32:
        p.error("--size must be a multiple of 32")
    args.alpha = float(args.rank) if args.alpha is None else args.alpha
    args.targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    args.out_dir = args.out_dir or str(_ROOT / "runs" / f"onestep_lora_{args.size}")
    args.cache_dir = args.cache_dir or str(_ROOT / "data" / f"onestep_targets_{args.size}")
    return args


def edge_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of blurred gradient-magnitude maps: does the output keep the input's structure."""
    def e(x):
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
        m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        return cv2.GaussianBlur(m, (0, 0), 2.0).ravel()
    return float(np.corrcoef(e(a), e(b))[0, 1])


class Trainer:
    def __init__(self, args):
        self.args = args
        torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction)
        self.pipe = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=BF16).to(DEV)
        self.unet = self.pipe.unet
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self.vae16 = copy.deepcopy(self.pipe.vae).to(DEV, FP16).to(memory_format=torch.channels_last).eval()
        self.vae32 = copy.deepcopy(self.pipe.vae).to(DEV, torch.float32).eval()
        import lpips
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
        for m in (self.vae32, self.lpips):
            for p in m.parameters():
                p.requires_grad_(False)
        s = args.size
        self.lat = s // self.pipe.vae_scale_factor
        self.tid = ops.make_time_ids(s, s, DEV, BF16)
        # The tensor BatchedEditWorker(shared_noise=True, seed=...) draws. It is
        # saved with the LoRA, and --eval_only replaces it by the saved one.
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        self.eps = ops.make_init_noise(self.pipe, height=s, width=s, device=DEV, dtype=BF16,
                                       generator=gen, batch_size=1, draw_dtype=torch.float32)
        self.ts1 = self.sched(1)[0]
        self.wrappers = {}

    # ------------------------------------------------------------------ sampling
    def sched(self, k):
        ts, _ = ops.set_timesteps(self.pipe, self.args.size, self.args.size, k, DEV)
        return ts.clone(), self.pipe.scheduler.sigmas.clone().to(DEV)

    def call(self, x, ref, t, emb, mask):
        n = x.shape[0]
        v = self.unet(torch.cat([x, ref], dim=3), timestep=t.expand(n).to(BF16),
                      encoder_hidden_states=emb, encoder_attention_mask=mask,
                      added_cond_kwargs={"time_ids": self.tid.expand(n, -1)}, return_dict=False)[0]
        return v[..., : self.lat]

    def one_step(self, ref, emb, mask):
        e = self.eps.expand(ref.shape[0], -1, -1, -1)
        return e - self.call(e, ref, self.ts1[0], emb, mask)

    @torch.no_grad()
    def multi_step(self, ref, emb, mask, k):
        ts, sg = self.sched(k)
        x = self.eps.expand(ref.shape[0], -1, -1, -1).contiguous()
        for i in range(k):
            x = x + (sg[i + 1] - sg[i]) * self.call(x, ref, ts[i], emb, mask)
        return x

    @torch.no_grad()
    def encode(self, frames):
        return ops.encode_uint8_batch_to_latent(
            self.vae16.encoder, torch.from_numpy(np.stack(frames)), DEV, FP16, BF16)

    @torch.no_grad()
    def decode(self, lat):
        return list(ops.decode_latent_to_uint8(self.pipe, self.vae16.decoder, lat, FP16))

    @torch.no_grad()
    def embed(self, prompt, frame):
        return ops.encode_prompt_edit_batch(self.pipe, [prompt], [frame], DEV, BF16)

    def place_text_encoder(self, device: str) -> None:
        """The 4 GB text encoder is only needed to build the data and eval sets; park it on the
        CPU while training so the UNet's activations have the VRAM."""
        self.pipe.text_encoder.to(device)
        if device == "cpu":
            torch.cuda.empty_cache()

    def set_lora_scale(self, scale: float) -> None:
        for w in self.wrappers.values():
            w.scale = scale

    # ------------------------------------------------------------------ data
    def load_frames(self, path, n, stride=1):
        fr = [np.asarray(f) for _, f, _ in iter_video_frames(str(path), self.args.size)]
        fr = fr[::stride][:n]
        return fr[: (len(fr) // 8) * 8]

    def build_dataset(self):
        a = self.args
        clips = sorted(p for d in a.clips_dirs for p in Path(d).glob("*.mp4"))
        if not clips:
            raise FileNotFoundError(f"no .mp4 clips under {a.clips_dirs}")
        key = hashlib.sha1(json.dumps({
            "clips": [f"{p.name}:{p.stat().st_size}" for p in clips], "size": a.size, "seed": a.seed,
            "k": a.teacher_steps, "prompts": a.prompts, "ppc": a.prompts_per_clip,
            "fpc": a.frames_per_clip, "stride": a.frame_stride, "data_seed": a.data_seed,
            # targets depend on which hidden state is the prompt embedding (pipeline_ops.prompt_hidden_state)
            "prompt_hidden": "pre-norm",
            "noise_draw": "fp32",
        }, sort_keys=True).encode()).hexdigest()[:12]
        cache = Path(a.cache_dir) / f"dataset_{key}.pt"
        if cache.exists():
            d = torch.load(cache, map_location=DEV)
            print(f"[data] loaded {d['refs'].shape[0]} cached pairs from {cache}", flush=True)
            return d
        rng = np.random.default_rng(a.data_seed)
        refs, tgts, embs, masks, idx = [], [], [], [], []
        t0 = time.perf_counter()
        for ci, path in enumerate(clips):
            frames = self.load_frames(path, a.frames_per_clip, a.frame_stride)
            if not frames:
                continue
            ref = torch.cat([self.encode(frames[b:b + 8]) for b in range(0, len(frames), 8)])
            for pi in rng.choice(len(a.prompts), size=min(a.prompts_per_clip, len(a.prompts)), replace=False):
                emb, mask = self.embed(a.prompts[int(pi)], frames[0])
                embs.append(emb)
                masks.append(mask)
                for b in range(0, len(frames), 8):
                    r = ref[b:b + 8]
                    tgts.append(self.multi_step(r, emb.expand(8, -1, -1), mask.expand(8, -1), a.teacher_steps))
                    refs.append(r)
                    idx += [len(embs) - 1] * 8
            if ci % 5 == 0:
                print(f"[data] {ci + 1}/{len(clips)} clips, {len(idx)} pairs ({time.perf_counter() - t0:.0f}s)",
                      flush=True)
        d = dict(refs=torch.cat(refs), tgts=torch.cat(tgts), embs=torch.cat(embs), masks=torch.cat(masks),
                 idx=torch.tensor(idx, device=DEV))
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(d, cache)
        print(f"[data] {d['refs'].shape[0]} pairs from {len(clips)} clips x {a.prompts_per_clip} prompts "
              f"in {time.perf_counter() - t0:.0f}s -> {cache}", flush=True)
        return d

    @torch.no_grad()
    def build_eval(self, combos, n_frames):
        """Per (clip, prompt): frames, embedding, reference latents and the teacher's K-step output."""
        items = []
        rs = lambda fr: [cv2.resize(f, (512, 512), interpolation=cv2.INTER_AREA) for f in fr]
        for clip, prompt, seen in combos:
            frames = self.load_frames(Path(self.args.eval_dir) / f"{clip}.mp4", n_frames)
            emb, mask = self.embed(prompt, frames[0])
            emb, mask = emb.expand(8, -1, -1), mask.expand(8, -1)
            ref = torch.cat([self.encode(frames[b:b + 8]) for b in range(0, len(frames), 8)])
            k_out = [f for b in range(0, len(frames), 8)
                     for f in self.decode(self.multi_step(ref[b:b + 8], emb, mask, self.args.teacher_steps))]
            k5 = rs(k_out)
            items.append(dict(clip=clip, prompt=prompt, seen=seen, emb=emb, mask=mask, ref=ref,
                              src5=rs(frames), k5=k5, k_t=self.to_t(k5), k_sobel=hf_density(k5)[0]))
        return items

    @staticmethod
    def to_t(frames5):
        return torch.from_numpy(np.stack(frames5)).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(DEV)

    @torch.no_grad()
    def score(self, items, k=1, full=False):
        """Mean metrics of the current UNet (with whatever LoRA state is active) against the teacher."""
        rows = []
        rs = lambda fr: [cv2.resize(f, (512, 512), interpolation=cv2.INTER_AREA) for f in fr]
        for it in items:
            outs = []
            for b in range(0, it["ref"].shape[0], 8):
                ref = it["ref"][b:b + 8]
                lat = self.one_step(ref, it["emb"], it["mask"]) if k == 1 else \
                    self.multi_step(ref, it["emb"], it["mask"], k)
                outs += self.decode(lat)
            o5 = rs(outs)
            to = self.to_t(o5)
            lp = float(torch.cat([self.lpips(to[i:i + 16], it["k_t"][i:i + 16])
                                  for i in range(0, len(o5), 16)]).mean())
            row = dict(seen=it["seen"], lpips=lp, sobel_ratio=hf_density(o5)[0] / it["k_sobel"])
            if full:
                row["edge"] = mean(edge_corr(a, b) for a, b in zip(it["src5"][2::5], o5[2::5]))
                row["warp"] = compute_temporal(it["src5"], o5).warping_error
            rows.append(row)
        out = {}
        for tag, sel in (("seen", True), ("unseen", False)):
            r = [x for x in rows if x["seen"] is sel]
            if r:
                out[tag] = {k2: mean(x[k2] for x in r) for k2 in r[0] if k2 != "seen"}
        return out

    # ------------------------------------------------------------------ training
    def lat_to_img(self, lat32):
        shift = getattr(self.pipe.vae.config, "shift_factor", 0.0) or 0.0
        return self.vae32.decoder(lat32 / self.pipe.vae.config.scaling_factor + shift)

    def loss(self, x0, tgt):
        x0, tgt = x0.float(), tgt.float()
        l_lat = F.smooth_l1_loss(x0, tgt, beta=0.1)
        c = min(self.args.crop // self.pipe.vae_scale_factor, self.lat)
        cy, cx = np.random.randint(0, self.lat - c + 1, size=2)
        img_s = self.lat_to_img(x0[:, :, cy:cy + c, cx:cx + c]).clamp(-1, 1)
        with torch.no_grad():
            img_t = self.lat_to_img(tgt[:, :, cy:cy + c, cx:cx + c]).clamp(-1, 1)
        l_lp = self.lpips(img_s, img_t).mean()
        return l_lat + self.args.lpips_weight * l_lp, l_lat.item(), l_lp.item()

    def train(self, data, quick_items, log_path):
        a = self.args
        self.wrappers = onestep_lora.attach(self.unet, rank=a.rank, alpha=a.alpha, targets=a.targets)
        params = [p for w in self.wrappers.values() for p in (w.A, w.B)]
        n_train = sum(p.numel() for p in params)
        print(f"[lora] targets={a.targets} rank={a.rank}: {len(self.wrappers)} layers, "
              f"{n_train / 1e6:.2f}M trainable", flush=True)
        ema = [p.detach().clone() for p in params]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        n = data["refs"].shape[0]
        bs, step, t0 = a.batch_size, 0, time.perf_counter()
        log = open(log_path, "w", encoding="utf-8")

        def with_ema(fn):
            live = [p.detach().clone() for p in params]
            with torch.no_grad():
                for p, e in zip(params, ema):
                    p.copy_(e)
            try:
                return fn()
            finally:
                with torch.no_grad():
                    for p, l in zip(params, live):
                        p.copy_(l)

        self.set_lora_scale(0.0)
        base = self.score(quick_items)
        self.set_lora_scale(1.0)
        print(f"[eval] step    0 (base 1-step)  " + self.fmt(base), flush=True)
        log.write(json.dumps({"step": 0, **base}) + "\n")

        while step < a.steps:
            for g in opt.param_groups:
                g["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * step / a.steps))
            i = torch.randint(0, n, (bs,), device=DEV)
            x0 = loss = None
            oom = False
            try:
                with torch.autocast("cuda", dtype=BF16):
                    j = data["idx"][i]
                    x0 = self.one_step(data["refs"][i], data["embs"][j], data["masks"][j])
                loss, l_lat, l_lp = self.loss(x0, data["tgts"][i])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            except torch.OutOfMemoryError:
                oom = True
            if oom:
                # Release the failed step here, outside the except block: inside it the
                # traceback still holds the graph, and empty_cache() frees nothing.
                del x0, loss
                opt.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                if bs == 1:
                    raise RuntimeError(
                        "out of memory at batch size 1; lower --rank, drop 'ff' from --targets, "
                        "or raise --max_memory_fraction")
                bs -= 1
                print(f"[train] out of memory, batch size -> {bs}", flush=True)
                continue
            with torch.no_grad():
                for p, e in zip(params, ema):
                    e.mul_(a.ema).add_(p.detach(), alpha=1 - a.ema)
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"[train] step {step:5d}/{a.steps}  huber {l_lat:.4f}  lpips {l_lp:.4f}  bs {bs}  "
                      f"{(time.perf_counter() - t0) / step:.2f}s/step", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                r = with_ema(lambda: self.score(quick_items))
                print(f"[eval] step {step:4d} (EMA)           " + self.fmt(r)
                      + f"   peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                log.write(json.dumps({"step": step, "huber": l_lat, "lpips_train": l_lp, **r}) + "\n")
                log.flush()
        log.close()
        with torch.no_grad():
            for p, e in zip(params, ema):
                p.copy_(e)
        return time.perf_counter() - t0

    @staticmethod
    def fmt(r):
        return "   ".join(
            f"{tag}: LPIPS->teacher {v['lpips']:.3f} sobel x{v['sobel_ratio']:.2f}"
            + (f" edge {v['edge']:.3f} warp {v['warp']:.2f}" if "edge" in v else "")
            for tag, v in r.items())


def final_eval(tr, args, state, meta, out_dir, check=None):
    """Held-out comparison against the base UNet's K-step output.

    `tr.unet` must be the bare base UNet here: the reference and the base
    rows are computed first, then the LoRA is merged, so its row comes from
    exactly the weights the runtime executes. `check` is an optional
    (ref, emb, mask, output) tuple from the unmerged LoRA branch.
    """
    combos = [(c, p, True) for p in args.prompts for c in args.eval_clips[:2]] \
        + [(c, p, False) for p in UNSEEN_PROMPTS for c in args.eval_clips[:2]]
    items = tr.build_eval(combos, n_frames=32)
    res = {}
    res["1 step, base"] = tr.score(items, k=1, full=True)
    res["2 steps, base"] = tr.score(items, k=2, full=True)
    handle = onestep_lora.merge(tr.unet, state, meta)
    if check is not None:
        ref, emb, mask, wrapped = check
        with torch.no_grad():
            merged = tr.one_step(ref, emb, mask).float()
        print(f"[merge] merged-weights output vs LoRA-branch output: relative difference "
              f"{((merged - wrapped).abs().mean() / wrapped.abs().mean()).item():.4f}  ({handle.n_layers} layers)",
              flush=True)
    res["1 step, LoRA merged"] = tr.score(items, k=1, full=True)
    handle.restore()
    (out_dir / "final_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nFINAL held-out evaluation at {args.size}, vs the base {args.teacher_steps}-step output "
          f"({len(args.eval_clips[:2])} clips x 32 frames per prompt)")
    for tag in ("seen", "unseen"):
        n_p = len(args.prompts) if tag == "seen" else len(UNSEEN_PROMPTS)
        print(f"FINAL  {tag} prompts ({n_p}):")
        for mode, r in res.items():
            if tag in r:
                v = r[tag]
                print(f"FINAL    {mode:<22} LPIPS {v['lpips']:.3f}  sobel x{v['sobel_ratio']:.2f}  "
                      f"edge {v['edge']:.3f}  warp_err {v['warp']:.2f}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "onestep_lora.safetensors"
    if args.eval_only:
        state, meta = onestep_lora.load(path)
        for k in ("size", "seed", "teacher_steps"):
            if meta.get(k) != getattr(args, k):
                raise SystemExit(f"{path} was trained with {k}={meta.get(k)}, not {getattr(args, k)}")
        args.prompts = meta["prompts"]
        print(f"[config] eval only: {path}", flush=True)
        tr = Trainer(args)
        noise = onestep_lora.init_noise(state)
        if noise is not None:
            tr.eps = noise.to(DEV, BF16)    # the tensor it was fitted to, whatever this build would draw
        final_eval(tr, args, state, meta, out_dir)
        return

    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"[config] size={args.size} seed={args.seed} teacher_steps={args.teacher_steps} "
          f"targets={args.targets} rank={args.rank} steps={args.steps} bs={args.batch_size} lr={args.lr}", flush=True)
    tr = Trainer(args)
    data = tr.build_dataset()

    quick = tr.build_eval(
        [(c, args.prompts[0], True) for c in args.eval_clips[:2]]
        + [(c, args.prompts[min(3, len(args.prompts) - 1)], True) for c in args.eval_clips[:2]]
        + [(c, UNSEEN_PROMPTS[0], False) for c in args.eval_clips[:2]], n_frames=8)
    tr.place_text_encoder("cpu")
    seconds = tr.train(data, quick, out_dir / "log.jsonl")

    state = onestep_lora.state_from_wrappers(tr.wrappers)
    state[onestep_lora.NOISE_KEY] = tr.eps.detach().float().cpu().contiguous()
    meta = dict(format="dreamlite-stream/onestep-lora/1", base="DreamLite-mobile", size=args.size, seed=args.seed,
                teacher_steps=args.teacher_steps, rank=args.rank, alpha=args.alpha, targets=args.targets,
                prompts=args.prompts, train_steps=args.steps, train_pairs=int(data["refs"].shape[0]),
                lpips_weight=args.lpips_weight, lr=args.lr, train_seconds=round(seconds))
    onestep_lora.save(path, state, meta)
    print(f"[saved] {path}  ({sum(v.numel() for v in state.values()) / 1e6:.2f}M values, "
          f"trained in {seconds / 60:.1f} min)", flush=True)

    if args.skip_final_eval:
        return

    # One batch through the LoRA branch as trained, then put the base layers
    # back: everything final_eval computes before its merge must come from
    # the base UNet, the reference above all.
    it = quick[0]
    with torch.no_grad():
        wrapped = tr.one_step(it["ref"][:8], it["emb"], it["mask"]).float()
    onestep_lora.detach(tr.unet, tr.wrappers)
    tr.wrappers = {}
    tr.place_text_encoder(DEV)
    final_eval(tr, args, state, meta, out_dir, check=(it["ref"][:8], it["emb"], it["mask"], wrapped))


if __name__ == "__main__":
    main()
