"""Build TRT engine with FP16 default but selected layers forced to FP32.

The pure-FP16 engine produces visibly wrong outputs because Qwen3-VL TE has
wide-range outputs (std=37, max=1160) that lose precision in the FP16
attention softmax / LayerNorm. This script lets us keep the speed of FP16
matmuls while forcing the precision-sensitive layers to FP32.

Usage:
    python scripts/build_trt_engine_mixed.py \
        --onnx out/trt/unet_b8_512.onnx \
        --engine out/trt/unet_b8_512_mixed.engine \
        --fp32_layers normalization,softmax
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Set

import tensorrt as trt


LAYER_TYPES = {
    "normalization": trt.LayerType.NORMALIZATION,
    "softmax": trt.LayerType.SOFTMAX,
    "matrix_multiply": trt.LayerType.MATRIX_MULTIPLY,
    "elementwise": trt.LayerType.ELEMENTWISE,
    "reduce": trt.LayerType.REDUCE,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--workspace_gb", type=int, default=12)
    p.add_argument("--fp32_layers", default="normalization,softmax",
                   help="comma list from {normalization,softmax,matrix_multiply,elementwise,reduce}")
    p.add_argument("--fp32_layer_names", default=None,
                   help="optional comma list of substrings; layers whose name contains any "
                        "are forced FP32 in addition to --fp32_layers")
    p.add_argument("--obey", action="store_true",
                   help="set OBEY_PRECISION_CONSTRAINTS (TRT will error rather than ignore)")
    return p.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    fp32_kinds: Set[trt.LayerType] = set()
    for name in args.fp32_layers.split(","):
        name = name.strip().lower()
        if name in LAYER_TYPES:
            fp32_kinds.add(LAYER_TYPES[name])
    name_substr = (
        [s.strip() for s in args.fp32_layer_names.split(",")]
        if args.fp32_layer_names else []
    )

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    print(f"[parse] {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.0f} MB)")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  parse error {i}: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30))
    config.set_flag(trt.BuilderFlag.FP16)
    if args.obey:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    # Per-layer FP32 enforcement
    n_total = network.num_layers
    fp32_count = 0
    type_count: dict = {}
    for i in range(n_total):
        layer = network.get_layer(i)
        type_count[layer.type] = type_count.get(layer.type, 0) + 1
        force_fp32 = False
        if layer.type in fp32_kinds:
            force_fp32 = True
        elif name_substr and any(s in layer.name for s in name_substr):
            force_fp32 = True
        if force_fp32:
            layer.precision = trt.DataType.FLOAT
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.DataType.FLOAT)
            fp32_count += 1

    print(f"[layers] total={n_total}  forced FP32={fp32_count}")
    print(f"[layer types] (top 8 by count):")
    for k, v in sorted(type_count.items(), key=lambda x: -x[1])[:8]:
        print(f"  {str(k):40s} {v}")

    print("[build] compiling engine...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("TRT build failed (returned None)")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = engine_path.stat().st_size / 1024 / 1024
    print(f"[done] saved {engine_path} ({size_mb:.0f} MB) in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
