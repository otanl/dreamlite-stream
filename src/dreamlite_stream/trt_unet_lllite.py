"""TensorRT wrapper for LLLite-baked DreamLite UNet.

The TRT engine was exported with LLLite hooks fused in. Each hook's `cond_emb`
is a separate engine input (38 inputs for the down_blocks subset).

The wrapper keeps the LLLite controller's `conditioning1` CNNs in PyTorch
(they're tiny — 32 channels in/out — so TRT'ing them would save microseconds).
On `set_cond_image()` we run the CNNs to produce 38 cond_embs and stash them;
on each `__call__` we bind them as engine inputs alongside the 5 base inputs.

Layout assumption (must match export-time):
  cond_emb_i corresponds to hook_names[i] from the .hooks.json sidecar file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import tensorrt as trt
import torch


class TRTUNetLLLiteWrapper:
    """TRT engine wrapper that produces noise_pred WITH LLLite contribution.

    Usage:
        wrapper = TRTUNetLLLiteWrapper(
            engine_path="unet_lllite_b8_512.engine",
            hooks_json="unet_lllite_b8_512.hooks.json",
            controller=lllite_controller,   # for conditioning1 CNNs
        )
        wrapper.set_cond_image(cond_img_tensor)   # populates internal cond_embs
        out = wrapper(model_input, timestep=..., encoder_hidden_states=..., ...)[0]
    """

    def __init__(
        self,
        engine_path: str,
        hooks_json: str,
        controller,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.controller = controller

        with open(hooks_json) as f:
            meta = json.load(f)
        self.hook_names: List[str] = meta["hook_names"]
        self.n_hooks: int = meta["n_hooks"]
        self.cond_emb_shape = tuple(meta["cond_emb_shape"])
        assert len(self.hook_names) == self.n_hooks

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Discover inputs / outputs and check binding completeness
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # Expected base inputs
        expected_base = {
            "model_input", "timestep", "encoder_hidden_states",
            "encoder_attention_mask", "time_ids",
        }
        for name in expected_base:
            if name not in self.input_names:
                raise RuntimeError(f"engine missing base input: {name}")
        # cond_emb_0 ... cond_emb_{N-1}
        for i in range(self.n_hooks):
            if f"cond_emb_{i}" not in self.input_names:
                raise RuntimeError(f"engine missing cond_emb_{i}")

        # Engine's encoder_hidden_states seq_len (used for pad/truncate)
        eh_shape = self.engine.get_tensor_shape("encoder_hidden_states")
        self.engine_seq_len = int(eh_shape[1])

        # Pre-allocate cond_emb cache slots (filled on set_cond_image)
        self._cond_embs: List[Optional[torch.Tensor]] = [None] * self.n_hooks

        # Output buffer (allocated on first call)
        self._output_buffer: Optional[torch.Tensor] = None

    # ----------------------------------------------------------
    @torch.no_grad()
    def set_cond_image(self, cond_image: Optional[torch.Tensor]) -> None:
        """Run all conditioning1 CNNs and stash results as engine inputs.

        cond_image: (B, 3, H, W) in [-1, 1]. Pass None to zero out the cond_embs.
        """
        if cond_image is None:
            for i in range(self.n_hooks):
                if self._cond_embs[i] is not None:
                    self._cond_embs[i].zero_()
            return

        cond_image = cond_image.to(device=self.device, dtype=self.dtype)
        for i, name in enumerate(self.hook_names):
            m = self.controller.modules_dict[name]
            # Mirror the path inside LLLiteModule.set_cond_image
            m.conditioning1.to(device=self.device, dtype=self.dtype)
            cx = m.conditioning1(cond_image)  # (B, C, h, w)
            if m.width_concat_factor > 1:
                n, c, h, w = cx.shape
                pad = torch.zeros(
                    n, c, h, w * (m.width_concat_factor - 1),
                    device=cx.device, dtype=cx.dtype,
                )
                cx = torch.cat([cx, pad], dim=3)
            n, c, h, w = cx.shape
            cx = cx.view(n, c, h * w).permute(0, 2, 1).contiguous()  # (B, h*w, c)
            self._cond_embs[i] = cx

    # ----------------------------------------------------------
    def _ensure_output_buffer(self, shape, dtype) -> torch.Tensor:
        if (
            self._output_buffer is None
            or tuple(self._output_buffer.shape) != tuple(shape)
            or self._output_buffer.dtype != dtype
        ):
            self._output_buffer = torch.empty(shape, dtype=dtype, device=self.device)
        return self._output_buffer

    # ----------------------------------------------------------
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
        time_ids = added_cond_kwargs["time_ids"]

        # dtype casts
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

        # Pad/truncate prompt embeddings to engine seq_len
        B, L_real = encoder_hidden_states.shape[:2]
        L_eng = self.engine_seq_len
        if L_real < L_eng:
            pad_emb = torch.zeros(
                B, L_eng - L_real, encoder_hidden_states.shape[2],
                dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device,
            )
            encoder_hidden_states = torch.cat([encoder_hidden_states, pad_emb], dim=1)
            pad_mask = torch.zeros(
                B, L_eng - L_real,
                dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device,
            )
            encoder_attention_mask = torch.cat([encoder_attention_mask, pad_mask], dim=1)
        elif L_real > L_eng:
            encoder_hidden_states = encoder_hidden_states[:, :L_eng]
            encoder_attention_mask = encoder_attention_mask[:, :L_eng]

        # Check cond_embs are populated
        for i, ce in enumerate(self._cond_embs):
            if ce is None:
                raise RuntimeError(
                    f"cond_emb_{i} not set; call set_cond_image first")

        inputs = {
            "model_input": model_input.contiguous(),
            "timestep": timestep.contiguous(),
            "encoder_hidden_states": encoder_hidden_states.contiguous(),
            "encoder_attention_mask": encoder_attention_mask.contiguous(),
            "time_ids": time_ids.contiguous(),
        }
        for i in range(self.n_hooks):
            inputs[f"cond_emb_{i}"] = self._cond_embs[i].contiguous()

        out = self._ensure_output_buffer(model_input.shape, self.dtype)

        # Bind addresses
        for name, t in inputs.items():
            self.context.set_tensor_address(name, t.data_ptr())
        out_name = self.output_names[0]
        self.context.set_tensor_address(out_name, out.data_ptr())

        current_stream = torch.cuda.current_stream(self.device).cuda_stream
        self.context.execute_async_v3(current_stream)

        return (out,)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self
