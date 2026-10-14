"""Build a TensorRT engine of the UNet for one resolution and batch size.

Two stages, each skipped when its result is already on disk:

1. export   PyTorch -> ONNX (fp16, static shapes). Needs the model weights.
            A one-step LoRA (--onestep_lora) is merged into the weights first,
            the Temporal LLLite adapter is traced into the graph with its
            per-hook cond embeddings as inputs, and the prompt projection
            (encoder_hid_proj) is left OUT of the graph: raw Qwen3-VL
            embeddings reach ~1000 in magnitude and its RMSNorm squares them,
            which overflows fp16. The runtime runs that projection in fp32.
            Also writes <name>.check.safetensors: real inputs and the fp32
            PyTorch output for them.
2. build    ONNX -> engine, with TensorRT. Then runs the engine on the check
            inputs and reports the error against the fp32 PyTorch output and
            the time per call. Needs only tensorrt and the stage-1 files, so
            it can run under a different interpreter from stage 1.

An engine only loads in the TensorRT version that built it, on the platform
that built it. Run stage 2 with the interpreter that will run the engine;
when Windows and WSL share one checkout, give each its own --out folder
(the stage-1 files can be copied across, they are platform-independent).

Usage:
    python scripts/build_trt_unet.py --size 768 --batch_size 8 \
        --onestep_lora runs/onestep_lora_768/onestep_lora.safetensors
    python scripts/build_trt_unet.py --size 640 --batch_size 8 --lllite_multiplier 0

Then:
    python scripts/champion_eval.py --lean --size 768 \
        --trt_engine out/trt/unet_768_b8_lora_lllite.engine

Outputs (default out/trt/): <name>.onnx, <name>.check.safetensors,
<name>.export.json, <name>.noise.safetensors (the init noise of the LoRA,
which the runtime uses as the shared noise), <name>.engine and <name>.json
(the engine's spec).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from types import MethodType

import torch

_ROOT = Path(__file__).resolve().parent.parent
_DREAMLITE = _ROOT.parent / "dreamlite"
_LLLITE = _ROOT.parent / "dreamlite-lllite"
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DREAMLITE))
sys.path.insert(0, str(_LLLITE / "src"))

warnings.filterwarnings("ignore")

from dreamlite_stream.lean import noise_path_for  # noqa: E402
from dreamlite_stream.trt_engine import (  # noqa: E402
    BASE_INPUTS, EngineSpec, cond_input_name, file_sha256,
)

DEV = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=str(_DREAMLITE / "models" / "DreamLite-mobile"))
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--onestep_lora", default=None, metavar="FILE",
                   help="One-step LoRA to merge into the weights before export.")
    p.add_argument("--lllite_weights",
                   default=str(_ROOT / "runs" / "temporal_lllite_v3" / "temporal_lllite_step001440.safetensors"))
    p.add_argument("--lllite_blocks", default="down_blocks")
    p.add_argument("--lllite_multiplier", type=float, default=1.0,
                   help="Adapter scale built into the engine. 0 = build without the adapter.")
    p.add_argument("--prompt", default="transfer this to oil painting style, vibrant colors",
                   help="Prompt of the check inputs (the engine itself is prompt-independent).")
    p.add_argument("--clip", default=str(_ROOT / "assets" / "davis_mp4" / "parkour.mp4"),
                   help="Clip the check frames come from.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="Engine path. Default: out/trt/unet_<size>_b<batch>[_lora][_lllite].engine")
    p.add_argument("--precision", choices=["auto", "fp16", "typed"], default="auto",
                   help="fp16: TensorRT picks fp16 or fp32 per layer (TensorRT 10 only; the faster of the two "
                        "there). typed: strongly typed, every layer in the dtype of the fp16 ONNX graph (all that "
                        "TensorRT 11+ offers). auto: fp16 on TensorRT 10, typed on 11+.")
    p.add_argument("--opt_level", type=int, default=3, help="TensorRT builder optimization level (0-5).")
    p.add_argument("--workspace_gb", type=int, default=10)
    p.add_argument("--force", action="store_true", help="Redo both stages even if their outputs exist.")
    args = p.parse_args()
    if args.size % 32:
        p.error("--size must be a multiple of 32")
    if args.out is None:
        name = f"unet_{args.size}_b{args.batch_size}" + ("_lora" if args.onestep_lora else "") \
            + ("_lllite" if args.lllite_multiplier != 0 else "")
        args.out = str(_ROOT / "out" / "trt" / f"{name}.engine")
    return args


def _export_forward(self, x):
    """LLLite hook forward for tracing: the cond embedding is a graph input."""
    merged = self.mid(torch.cat([self._export_cond_emb, self.down(x)], dim=2))
    return self._org_forward(x + self.up(merged) * self._multiplier_t)


class _Exportable(torch.nn.Module):
    """UNet with positional inputs: the five base tensors, then one cond embedding per LLLite hook."""

    def __init__(self, unet, hooks):
        super().__init__()
        self.unet = unet
        self._hooks = list(hooks)

    def forward(self, model_input, timestep, encoder_hidden_states, encoder_attention_mask, time_ids, *cond_embs):
        for m, ce in zip(self._hooks, cond_embs):
            m._export_cond_emb = ce
        return self.unet(model_input, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
                         encoder_attention_mask=encoder_attention_mask,
                         added_cond_kwargs={"time_ids": time_ids}, return_dict=False)[0]


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    a = a.float()
    if not torch.isfinite(a).all():
        return float("nan")
    return ((a - ref).abs().mean() / ref.abs().mean()).item()


def requested_export(args) -> dict:
    """What stage 1 depends on; compared with <name>.export.json to decide whether to reuse it."""
    with_lllite = args.lllite_multiplier != 0
    return {
        "size": args.size, "batch_size": args.batch_size, "seed": args.seed,
        "onestep_lora": Path(args.onestep_lora).name if args.onestep_lora else None,
        "onestep_lora_sha256": file_sha256(args.onestep_lora) if args.onestep_lora else None,
        "lllite_weights": Path(args.lllite_weights).name if with_lllite else None,
        "lllite_weights_sha256": file_sha256(args.lllite_weights) if with_lllite else None,
        "lllite_blocks": args.lllite_blocks if with_lllite else None,
        "lllite_multiplier": args.lllite_multiplier if with_lllite else 0.0,
    }


@torch.no_grad()
def export(args, onnx_path: Path, check_path: Path, meta_path: Path) -> None:
    import numpy as np
    from dreamlite import DreamLiteMobilePipeline
    from safetensors.torch import load_file, save_file

    import dreamlite_stream.pipeline_ops as ops
    from dreamlite_stream import onestep_lora
    from dreamlite_stream.runtime import iter_video_frames

    B, size = args.batch_size, args.size
    pipe = DreamLiteMobilePipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(DEV)
    frames = [np.asarray(f) for _, f, _ in iter_video_frames(args.clip, size)][:B]
    if len(frames) < B:
        raise SystemExit(f"{args.clip} has fewer than {B} frames")
    # The lean path's inputs: one prompt embedding broadcast to the batch, one shared noise draw.
    emb, mask = ops.encode_prompt_edit_batch(pipe, [args.prompt], frames[:1], DEV, torch.bfloat16)
    emb, mask = emb.expand(B, -1, -1).float().contiguous(), mask.expand(B, -1).contiguous()
    ref_lat = ops.encode_image_to_latent_batch(pipe, frames, size, size, DEV, torch.bfloat16).float()
    # The check runs on the noise the engine will see: the LoRA's own tensor when it has one.
    state = meta = lora_noise = None
    if args.onestep_lora:
        state, meta = onestep_lora.load(args.onestep_lora)
        lora_noise = onestep_lora.init_noise(state)
    noise_path = noise_path_for(onnx_path)
    if lora_noise is not None:
        save_file({onestep_lora.NOISE_KEY: lora_noise.contiguous()}, str(noise_path))
        eps = lora_noise.to(DEV, torch.float32)
    else:
        noise_path.unlink(missing_ok=True)
        eps = ops.make_init_noise(pipe, height=size, width=size, device=DEV, dtype=torch.float32,
                                  generator=torch.Generator(device="cpu").manual_seed(args.seed), batch_size=1)
    x = torch.cat([eps.expand(B, -1, -1, -1), ref_lat], dim=3).contiguous()
    ts, _ = ops.set_timesteps(pipe, size, size, 1, DEV)
    t = ts[0].expand(B).float().contiguous()
    tid = ops.make_time_ids(size, size, DEV, torch.float32).expand(B, -1).contiguous()
    latent_hw = size // pipe.vae_scale_factor
    del pipe.text_encoder
    torch.cuda.empty_cache()

    unet = pipe.unet.float().eval()
    req = requested_export(args)
    lora_meta = None
    if args.onestep_lora:
        handle = onestep_lora.merge(unet, state, meta)
        lora_meta = {k: meta.get(k) for k in ("size", "seed", "teacher_steps", "prompts")}
        print(f"[export] merged one-step LoRA: {handle.n_layers} layers from {args.onestep_lora}", flush=True)
        for note in onestep_lora.compatibility_notes(meta, size=size, seed=args.seed, steps=1, shared_noise=True,
                                                     has_noise=lora_noise is not None):
            print(f"[export] note: {note}", flush=True)

    hooks, hook_names = [], []
    if req["lllite_weights"]:
        from dreamlite_lllite import apply_lllite
        blocks = [s.strip() for s in args.lllite_blocks.split(",")]
        controller = apply_lllite(unet, cond_emb_dim=32, mlp_dim=64, cond_image_size=size, sample_size=latent_hw,
                                  inference_mode=False, max_batch_size=B, block_filter=blocks)
        controller.load_state_dict(load_file(args.lllite_weights), strict=False)
        controller.to(device=DEV, dtype=torch.float32).eval()
        controller.set_multiplier(args.lllite_multiplier)
        # The cond the lean path holds for a clip: its first batch of frames.
        cond = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(127.5).sub(1.0).to(DEV)
        controller.set_cond_image(cond)
        hook_names = list(controller.modules_dict.keys())
        hooks = [controller.modules_dict[n] for n in hook_names]
        print(f"[export] LLLite adapter: {len(hooks)} hooks ({args.lllite_blocks}), "
              f"multiplier {args.lllite_multiplier:g}", flush=True)

    plain = _Exportable(unet, [])
    ref32 = plain(x, t, emb, mask, tid)                  # fp32 reference, projection inside, hooks live
    proj = unet.encoder_hid_proj(emb)
    print(f"[export] prompt embeddings: max |x| {emb.abs().max().item():.0f} raw, "
          f"{proj.abs().max().item():.1f} after the projection (kept outside the engine)", flush=True)

    unet.encoder_hid_proj = torch.nn.Identity()
    unet.half()
    cond_embs = []
    for m in hooks:
        m.half()
        cond_embs.append(m.cond_emb.half().contiguous())
        m._host_ref[0].forward = MethodType(_export_forward, m)
    model = _Exportable(unet, hooks).eval()
    inputs = (x.half(), t.half(), proj.half().contiguous(), mask.long(), tid.half(), *cond_embs)
    err16 = rel_err(model(*inputs), ref32)
    print(f"[export] PyTorch fp16 with the projection outside vs fp32: relative error {err16:.4f}", flush=True)

    names = list(BASE_INPUTS) + [cond_input_name(i) for i in range(len(hooks))]
    t0 = time.perf_counter()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, inputs, str(onnx_path), input_names=names, output_names=["noise_pred"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"[export] {onnx_path.name} ({onnx_path.stat().st_size / 2**20:.0f} MB) in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    check = {n: v.cpu().contiguous() for n, v in zip(names, inputs)}
    check["ref32"] = ref32.cpu().contiguous()
    save_file(check, str(check_path))
    meta_path.write_text(json.dumps({**req, "onestep_lora_meta": lora_meta, "has_init_noise": lora_noise is not None,
                                     "seq_len": int(proj.shape[1]), "hook_names": hook_names,
                                     "pytorch_fp16_rel_err": err16}, indent=2), encoding="utf-8")
    del pipe, unet, model, plain
    torch.cuda.empty_cache()


def build(args, onnx_path: Path, check_path: Path, meta: dict, engine_path: Path) -> None:
    import tensorrt as trt
    from safetensors.torch import load_file

    major = int(trt.__version__.split(".")[0])
    typed = args.precision == "typed" or (args.precision == "auto" and major >= 11)
    if not typed and major >= 11:
        raise SystemExit(f"TensorRT {trt.__version__} builds strongly typed networks only; use --precision typed")
    precision = "fp16-typed" if typed else "fp16"
    logger = trt.Logger(trt.Logger.WARNING)

    spec_file = EngineSpec.path_for(engine_path)
    stale = False
    if engine_path.exists() and spec_file.exists():
        old = EngineSpec.load(engine_path)
        if old.platform and old.platform != sys.platform and not args.force:
            raise SystemExit(f"{engine_path} was built on {old.platform} and this is {sys.platform}. Pass --out "
                             "to keep a separate engine for this platform, or --force to replace that one.")
        stale = (old.precision, old.tensorrt) != (precision, trt.__version__)
    if args.force or stale or not engine_path.exists():
        builder = trt.Builder(logger)
        flags = (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if (typed and major < 11) else 0
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            for i in range(parser.num_errors):
                print("[build] parse error:", parser.get_error(i))
            raise SystemExit("ONNX parse failed")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)
        config.builder_optimization_level = args.opt_level
        if not typed:
            config.set_flag(trt.BuilderFlag.FP16)
        print(f"[build] TensorRT {trt.__version__}, {precision}, optimization level {args.opt_level} ...", flush=True)
        t0 = time.perf_counter()
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise SystemExit("TensorRT build failed")
        engine_path.write_bytes(bytes(blob))
        print(f"[build] {engine_path.name} ({engine_path.stat().st_size / 2**20:.0f} MB) in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        del builder, network, parser, config, blob

    # Run the engine on the check inputs.
    check = load_file(str(check_path))
    ref32 = check.pop("ref32").to(DEV)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    ctx = engine.create_execution_context()
    dtypes = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32,
              trt.DataType.INT64: torch.int64, trt.DataType.INT32: torch.int32}
    keep, out = {}, None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dt, shape = dtypes[engine.get_tensor_dtype(name)], tuple(engine.get_tensor_shape(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            keep[name] = check[name].to(DEV, dt).contiguous()
            if tuple(keep[name].shape) != shape:
                raise SystemExit(f"engine input {name} is {shape}, check input is {tuple(keep[name].shape)}")
        else:
            keep[name] = out = torch.empty(shape, dtype=dt, device=DEV)
        ctx.set_tensor_address(name, keep[name].data_ptr())
    stream = torch.cuda.Stream()

    def run():
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    run()
    err = rel_err(out, ref32)
    for _ in range(5):
        run()
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        run()
        times.append((time.perf_counter() - t0) * 1000)
    ms = sorted(times)[len(times) // 2]
    B = meta["batch_size"]
    print(f"[check] engine vs fp32 PyTorch on real inputs: relative error {err:.4f}  "
          f"(PyTorch fp16: {meta['pytorch_fp16_rel_err']:.4f})", flush=True)
    print(f"[check] {ms:.1f} ms per call at batch {B} = {ms / B:.2f} ms per frame", flush=True)
    if not err < 0.05:
        print("[check] WARNING: the error is far above the ~0.01 expected of fp16; do not use this engine", flush=True)

    spec = EngineSpec(
        size=meta["size"], batch_size=B, seq_len=meta["seq_len"], tensorrt=trt.__version__,
        platform=sys.platform, precision=precision,
        onestep_lora=meta["onestep_lora"], onestep_lora_sha256=meta["onestep_lora_sha256"],
        onestep_lora_meta=meta.get("onestep_lora_meta"),
        lllite_weights=meta["lllite_weights"], lllite_blocks=meta["lllite_blocks"],
        lllite_multiplier=meta["lllite_multiplier"], hook_names=meta["hook_names"],
        validation={"rel_err_vs_fp32": err, "ms_per_call": ms},
    )
    print(f"[done] {engine_path}\n       spec: {spec.save(engine_path)}", flush=True)


def main():
    args = parse_args()
    engine_path = Path(args.out)
    onnx_path = engine_path.with_suffix(".onnx")
    check_path = engine_path.with_suffix(".check.safetensors")
    meta_path = engine_path.with_suffix(".export.json")
    print(f"[config] size={args.size} batch={args.batch_size} lora={'yes' if args.onestep_lora else 'no'} "
          f"lllite_multiplier={args.lllite_multiplier:g}  torch {torch.__version__}", flush=True)

    req = requested_export(args)
    have = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    reuse = (not args.force and have is not None and onnx_path.exists() and check_path.exists()
             and all(have.get(k) == v for k, v in req.items())
             and (noise_path_for(engine_path).exists() or not have.get("has_init_noise")))
    if reuse:
        print(f"[export] reusing {onnx_path.name}", flush=True)
    else:
        export(args, onnx_path, check_path, meta_path)
        have = json.loads(meta_path.read_text(encoding="utf-8"))
        if engine_path.exists():
            engine_path.unlink()  # built from the previous export
    build(args, onnx_path, check_path, have, engine_path)


if __name__ == "__main__":
    main()
