"""Build a TensorRT engine from the exported UNet ONNX.

Static shapes (B=8 + 512²); fp16 mode. The resulting `.engine` is the
compiled-and-optimized executable kernel set for this shape.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import tensorrt as trt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default=str(Path(__file__).resolve().parent.parent / "out" / "trt" / "unet_b8_512.onnx"))
    p.add_argument("--engine", default=str(Path(__file__).resolve().parent.parent / "out" / "trt" / "unet_b8_512.engine"))
    p.add_argument("--workspace_gb", type=int, default=8,
                   help="GPU workspace memory budget for TRT search")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                   help="enable FP16 (default). Use --no-fp16 for pure FP32 engine.")
    return p.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    print(f"[parse] {onnx_path}  ({onnx_path.stat().st_size / 1024 / 1024:.0f} MB)")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  parse error {i}: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30),
    )
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[config] FP16 enabled")

    print("[build] compiling engine - this can take 5-15 minutes...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("TRT build failed (returned None)")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = engine_path.stat().st_size / 1024 / 1024
    print(f"[done] saved {engine_path}  ({size_mb:.0f} MB)  in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
