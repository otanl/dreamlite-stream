"""TensorRT wrapper that mimics DreamLite UNet's call signature.

Drop-in replacement for `pipeline.unet(model_input, timestep=..., ...)` that
runs the compiled TRT engine instead of the PyTorch UNet. Pre-allocated
device buffers are reused across calls; only input data pointers change.

Limitations vs PyTorch UNet:
  - Fixed shapes (must match what the engine was built for)
  - No autograd / gradient flow
  - No LLLite hooks (the LLLite is monkey-patched into PyTorch's host.forward
    and isn't visible to the exported graph). For LLLite + TRT we'd need to
    bake hooks into the export, or use a custom plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import tensorrt as trt
import torch


class TRTUNetWrapper:
    """Loads a TRT engine and exposes a `__call__` mirroring DreamLite UNet.

    Usage:
        wrapper = TRTUNetWrapper("unet_b8_512.engine", device="cuda")
        # Replace pipeline.unet
        pipeline.unet = wrapper
        # ... pipeline_ops.denoise() will now run via TRT
    """

    def __init__(self, engine_path: str, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.dtype = torch.float16  # engine was built fp16
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Discover input / output names (TRT 10 API).
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # Engine was built with FIXED prompt sequence length. Read it back
        # so we can pad/truncate runtime inputs to match — pipeline's
        # encode_prompt produces variable-length embeddings (~100 tokens
        # depending on prompt) that won't match the engine's binding.
        self.engine_seq_len: Optional[int] = None
        for name in self.input_names:
            if name == "encoder_hidden_states":
                shape = self.engine.get_tensor_shape(name)
                self.engine_seq_len = int(shape[1])
                break

        # Pre-allocate output buffer (single output for our UNet).
        # Engine has fixed shapes, so we can query the static shape.
        self._output_buffer: Optional[torch.Tensor] = None

        self._stream = torch.cuda.Stream(device=device)

    def _ensure_output_buffer(self, shape, dtype) -> torch.Tensor:
        if (
            self._output_buffer is None
            or tuple(self._output_buffer.shape) != tuple(shape)
            or self._output_buffer.dtype != dtype
        ):
            self._output_buffer = torch.empty(
                shape, dtype=dtype, device=self.device,
            )
        return self._output_buffer

    @torch.no_grad()
    def __call__(
        self,
        model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        added_cond_kwargs: dict,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor]:
        """Run TRT engine. Inputs must be on cuda and match engine shapes.

        encoder_attention_mask is expected to be int64 (long) — the engine
        was compiled with that binding. timestep / encoder_hidden_states /
        time_ids are fp16.
        """
        time_ids = added_cond_kwargs["time_ids"]

        # Cast as needed to match engine bindings (engine was built fp16)
        if model_input.dtype != self.dtype:
            model_input = model_input.to(self.dtype)
        if timestep.dtype != self.dtype:
            timestep = timestep.to(self.dtype)
        if encoder_hidden_states.dtype != self.dtype:
            encoder_hidden_states = encoder_hidden_states.to(self.dtype)
        if encoder_attention_mask.dtype != torch.long:
            encoder_attention_mask = encoder_attention_mask.long()
        if time_ids.dtype != self.dtype:
            time_ids = time_ids.to(self.dtype)

        # Pad / truncate prompt embeddings to engine's compiled seq_len. The
        # attention mask zeros out padded positions so they're ignored.
        if self.engine_seq_len is not None:
            B, L_real = encoder_hidden_states.shape[:2]
            L_eng = self.engine_seq_len
            if L_real < L_eng:
                pad_emb = torch.zeros(
                    B, L_eng - L_real, encoder_hidden_states.shape[2],
                    dtype=encoder_hidden_states.dtype,
                    device=encoder_hidden_states.device,
                )
                encoder_hidden_states = torch.cat([encoder_hidden_states, pad_emb], dim=1)
                pad_mask = torch.zeros(
                    B, L_eng - L_real,
                    dtype=encoder_attention_mask.dtype,
                    device=encoder_attention_mask.device,
                )
                encoder_attention_mask = torch.cat([encoder_attention_mask, pad_mask], dim=1)
            elif L_real > L_eng:
                encoder_hidden_states = encoder_hidden_states[:, :L_eng]
                encoder_attention_mask = encoder_attention_mask[:, :L_eng]

        # Make every input contiguous & on cuda
        inputs = {
            "model_input": model_input.contiguous(),
            "timestep": timestep.contiguous(),
            "encoder_hidden_states": encoder_hidden_states.contiguous(),
            "encoder_attention_mask": encoder_attention_mask.contiguous(),
            "time_ids": time_ids.contiguous(),
        }

        # Output shape: same as model_input (UNet preserves layout).
        out = self._ensure_output_buffer(model_input.shape, self.dtype)

        # Bind addresses
        for name, t in inputs.items():
            self.context.set_tensor_address(name, t.data_ptr())
        # Sole output
        out_name = self.output_names[0]
        self.context.set_tensor_address(out_name, out.data_ptr())

        # Execute on the CURRENT stream so subsequent PyTorch ops on default
        # stream see the output without cross-stream sync issues.
        current_stream = torch.cuda.current_stream(self.device).cuda_stream
        self.context.execute_async_v3(current_stream)

        return (out,)
