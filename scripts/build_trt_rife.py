"""Build a TensorRT engine of the RIFE interpolation network for one frame size and batch.

The engine takes `frames` (batch + 1 consecutive frames) and a time position
`t`, and returns the `batch` in-between frames; see dreamlite_stream/rife.py.
It is exported in fp16 with the warp coordinates in fp32, and built strongly
typed so TensorRT keeps that split.

The script then runs the engine on real frames and reports its agreement
with the fp32 PyTorch network and the time per call.

Weights: `flownet.pkl` of RIFE v4.25 (or v4.25 lite / v4.26) from the links
in https://github.com/hzwer/Practical-RIFE.

Usage:
    python scripts/build_trt_rife.py --weights data/rife/v4.25/flownet.pkl --size 768 --batch_size 8

Then:
    python scripts/demo_camera.py --lean --size 768 --interp_factor 2 --interp_method rife \
        --rife_engine out/trt/rife_768_b8.engine

An engine only loads in the TensorRT version that built it, on the platform
that built it; with Windows and WSL sharing a checkout, give each its own
--out folder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dreamlite_stream import rife  # noqa: E402
from dreamlite_stream.runtime import iter_video_frames  # noqa: E402
from dreamlite_stream.trt_engine import file_sha256  # noqa: E402

DEV = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--weights", default=str(_ROOT / "data" / "rife" / "v4.25" / "flownet.pkl"))
    p.add_argument("--size", type=int, default=768, help="Frame side; padded up to a multiple of 64.")
    p.add_argument("--batch_size", type=int, default=8, help="In-between frames per call (the pipeline's batch size).")
    p.add_argument("--clip", default=str(_ROOT / "assets" / "davis_mp4" / "parkour.mp4"),
                   help="Clip the check frames come from.")
    p.add_argument("--out", default=None, help="Engine path. Default: out/trt/rife_<size>_b<batch>.engine")
    p.add_argument("--opt_level", type=int, default=3)
    p.add_argument("--workspace_gb", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.out is None:
        args.out = str(_ROOT / "out" / "trt" / f"rife_{args.size}_b{args.batch_size}.engine")
    return args


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else -10.0 * np.log10(mse)


def median_ms(fn, n=30, warm=5) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    v = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        v.append((time.perf_counter() - t0) * 1000)
    return sorted(v)[len(v) // 2]


@torch.no_grad()
def main():
    import tensorrt as trt

    args = parse_args()
    engine_path = Path(args.out)
    onnx_path = engine_path.with_suffix(".onnx")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    side = args.size + (-args.size) % rife.MULTIPLE
    n = args.batch_size
    print(f"[config] {side}x{side}, {n} in-between frames per call, torch {torch.__version__}, "
          f"TensorRT {trt.__version__}", flush=True)

    net32 = rife.load_ifnet(args.weights).to(DEV)
    frames = [np.asarray(f) for _, f, _ in iter_video_frames(args.clip, side)][: n + 1]
    if len(frames) < n + 1:
        raise SystemExit(f"{args.clip} has fewer than {n + 1} frames")
    x32 = torch.from_numpy(np.stack(frames)).to(DEV).permute(0, 3, 1, 2).float().div(255.0).contiguous()
    t32 = torch.tensor([0.5], device=DEV)
    ref = net32(x32, t32).clamp(0, 1)
    blend = (0.5 * (x32[:-1] + x32[1:]))
    print(f"[check] the frames move: plain cross-fade vs RIFE fp32 = {psnr(blend, ref):.1f} dB", flush=True)

    net16 = rife.load_ifnet(args.weights).to(DEV).half()
    x16, t16 = x32.half(), t32.half()
    pt16 = net16(x16, t16)
    print(f"[check] PyTorch fp16 vs fp32: {psnr(pt16.clamp(0, 1), ref):.1f} dB", flush=True)
    net_cl = rife.load_ifnet(args.weights).to(DEV).half().to(memory_format=torch.channels_last)
    x_cl = x16.contiguous(memory_format=torch.channels_last)
    pt_ms = median_ms(lambda: net_cl(x_cl, t16))
    print(f"[check] PyTorch fp16 channels_last: {pt_ms:.1f} ms per call", flush=True)

    if args.force or not onnx_path.exists():
        t0 = time.perf_counter()
        torch.onnx.export(net16, (x16, t16), str(onnx_path), input_names=["frames", "t"],
                          output_names=["mids"], opset_version=17, do_constant_folding=True, dynamo=False)
        print(f"[export] {onnx_path.name} ({onnx_path.stat().st_size / 2**20:.0f} MB) in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        if engine_path.exists():
            engine_path.unlink()
    del net32, net16, net_cl
    torch.cuda.empty_cache()

    logger = trt.Logger(trt.Logger.WARNING)
    spec_file = rife.engine_spec_path(engine_path)
    if engine_path.exists() and spec_file.exists() and not args.force:
        built_on = json.loads(spec_file.read_text(encoding="utf-8")).get("platform")
        if built_on not in (None, "", sys.platform):
            raise SystemExit(f"{engine_path} was built on {built_on} and this is {sys.platform}. Pass --out to "
                             "keep a separate engine for this platform, or --force to replace that one.")
    if args.force or not engine_path.exists():
        builder = trt.Builder(logger)
        major = int(trt.__version__.split(".")[0])
        flags = (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if major < 11 else 0
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            for i in range(parser.num_errors):
                print("[build] parse error:", parser.get_error(i))
            raise SystemExit("ONNX parse failed")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)
        config.builder_optimization_level = args.opt_level
        t0 = time.perf_counter()
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise SystemExit("TensorRT build failed")
        engine_path.write_bytes(bytes(blob))
        print(f"[build] {engine_path.name} ({engine_path.stat().st_size / 2**20:.0f} MB) in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        del builder, network, parser, config, blob

    spec = {"format": rife.ENGINE_FORMAT, "height": side, "width": side, "batch_size": n,
            "weights": Path(args.weights).parent.name + "/" + Path(args.weights).name,
            "weights_sha256": file_sha256(args.weights), "tensorrt": trt.__version__,
            "platform": sys.platform}
    rife.engine_spec_path(engine_path).write_text(json.dumps(spec, indent=2), encoding="utf-8")

    runner = rife._TRTRunner(engine_path, torch.device(DEV))
    out = runner(x16, t16)
    db = psnr(out.clamp(0, 1), ref)
    ms = median_ms(lambda: runner(x16, t16))
    print(f"[check] engine vs fp32 PyTorch: {db:.1f} dB", flush=True)
    print(f"[check] engine: {ms:.1f} ms per call = {ms / n:.2f} ms per in-between frame "
          f"(PyTorch fp16: {pt_ms:.1f} ms)", flush=True)
    if db < 35:
        print("[check] WARNING: below 35 dB against fp32; do not use this engine", flush=True)
    spec["validation"] = {"psnr_vs_fp32_db": db, "ms_per_call": ms, "pytorch_fp16_ms_per_call": pt_ms}
    rife.engine_spec_path(engine_path).write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"[done] {engine_path}", flush=True)


if __name__ == "__main__":
    main()
