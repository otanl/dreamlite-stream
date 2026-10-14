"""TensorRT backend for the UNet, built by `scripts/build_trt_unet.py`.

`TRTEngineUNet` stands in for `pipeline.unet`: same call signature, the
denoise loop does not change. What differs from the older `TRTUNetWrapper`:

* The prompt projection (`encoder_hid_proj`, a Linear + RMSNorm) runs in
  PyTorch fp32 and the engine takes its output. Raw Qwen3-VL embeddings
  reach ~1000 in magnitude and the RMSNorm squares them, which overflows
  fp16 inside an engine; after the projection the values are small.
* A one-step LoRA is merged into the weights before export, so the engine
  already contains it.
* The Temporal LLLite adapter is part of the exported graph when the engine
  was built with it. Its per-hook cond embeddings are engine inputs, read
  straight from the controller's buffers, so `controller.set_cond_image`
  keeps working as it does for the PyTorch UNet. The adapter's multiplier is
  a constant of the engine.

An engine is tied to one resolution, one batch size, one TensorRT version
and the platform it was built on (a Windows engine does not load under
Linux / WSL, and the reverse). The sidecar `<engine>.json` records what it
was built from; `EngineSpec` reads it and `mismatches` compares it with how
a script is about to run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

SPEC_FORMAT = "dreamlite-stream/trt-unet/1"
BASE_INPUTS = ("model_input", "timestep", "encoder_hidden_states", "encoder_attention_mask", "time_ids")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cond_input_name(i: int) -> str:
    return f"cond_emb_{i}"


@dataclass
class EngineSpec:
    """What an engine was built from. Stored as `<engine>.json`."""

    size: int
    batch_size: int
    seq_len: int
    tensorrt: str = ""
    platform: str = ""                          # sys.platform of the build; "" = not recorded
    precision: str = "fp16"
    onestep_lora: Optional[str] = None          # file name, for the log
    onestep_lora_sha256: Optional[str] = None
    onestep_lora_meta: Optional[dict] = None    # size / seed / prompts it was trained for
    lllite_weights: Optional[str] = None
    lllite_blocks: Optional[str] = None
    lllite_multiplier: float = 0.0
    hook_names: List[str] = field(default_factory=list)
    validation: Dict[str, float] = field(default_factory=dict)
    format: str = SPEC_FORMAT

    @staticmethod
    def path_for(engine_path) -> Path:
        return Path(engine_path).with_suffix(".json")

    def save(self, engine_path) -> Path:
        p = self.path_for(engine_path)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, engine_path) -> "EngineSpec":
        p = cls.path_for(engine_path)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; build the engine with scripts/build_trt_unet.py")
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("format") != SPEC_FORMAT:
            raise ValueError(f"{p}: unknown engine spec format {d.get('format')!r}")
        return cls(**d)

    @property
    def has_lllite(self) -> bool:
        return bool(self.hook_names)

    def mismatches(self, *, size: int, batch_size: int, onestep_lora: Optional[str] = None,
                   lllite_multiplier: float = 1.0) -> List[str]:
        """Reasons this engine cannot stand in for the run described by the arguments."""
        out: List[str] = []
        if self.platform and self.platform != sys.platform:
            out.append(f"engine was built on {self.platform}, this is {sys.platform}; build one here "
                       "(scripts/build_trt_unet.py with its own --out)")
        if size != self.size:
            out.append(f"engine is {self.size}x{self.size}, run is {size}x{size}")
        if batch_size != self.batch_size:
            out.append(f"engine batch size is {self.batch_size}, run uses {batch_size}")
        if onestep_lora:
            sha = file_sha256(onestep_lora)
            if sha != self.onestep_lora_sha256:
                out.append("--onestep_lora names a different file from the one built into the engine "
                           f"({self.onestep_lora or 'none'})")
        engine_mult = self.lllite_multiplier if self.has_lllite else 0.0
        if abs(engine_mult - lllite_multiplier) > 1e-6:
            out.append(f"engine was built with the adapter at multiplier {engine_mult:g}, "
                       f"run asks for {lllite_multiplier:g}")
        return out


class TRTEngineUNet:
    """A TensorRT engine with the call signature of the DreamLite UNet.

    hid_proj:    the PyTorch UNet's `encoder_hid_proj`; run here in fp32.
    controller:  the LLLite controller, required when the engine contains
                 the adapter. Its `_cond_emb_buf` buffers feed the engine.
    """

    is_trt_engine = True

    def __init__(self, engine_path, hid_proj: torch.nn.Module, controller=None,
                 device: str = "cuda") -> None:
        import tensorrt as trt

        self.spec = EngineSpec.load(engine_path)
        self.device = torch.device(device)
        self.dtype = torch.float16
        # A private fp32 copy: the PyTorch UNet it came from keeps its own dtype.
        self.hid_proj = copy.deepcopy(hid_proj).to(device=self.device, dtype=torch.float32).eval()
        built, running = self.spec.tensorrt, trt.__version__
        if built and built != running:
            raise RuntimeError(f"{engine_path} was built with TensorRT {built}; this is {running}. "
                               "Engines do not carry over between versions; rebuild it here.")
        if self.spec.platform and self.spec.platform != sys.platform:
            raise RuntimeError(f"{engine_path} was built on {self.spec.platform}; this is {sys.platform}. "
                               "Engines do not carry over between platforms; build one here.")

        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        self.engine = self._runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not load {engine_path}")
        self.context = self.engine.create_execution_context()

        dtypes = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32,
                  trt.DataType.INT64: torch.int64, trt.DataType.INT32: torch.int32,
                  trt.DataType.BOOL: torch.bool}
        self._in_dtype: Dict[str, torch.dtype] = {}
        self._in_shape: Dict[str, Tuple[int, ...]] = {}
        self._out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(int(d) for d in self.engine.get_tensor_shape(name))
            dt = dtypes[self.engine.get_tensor_dtype(name)]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._in_dtype[name], self._in_shape[name] = dt, shape
            else:
                self._out_name = name
                self._out = torch.empty(shape, dtype=dt, device=self.device)
        missing = [n for n in BASE_INPUTS if n not in self._in_shape]
        if missing or self._out_name is None:
            raise RuntimeError(f"{engine_path}: not a build_trt_unet.py engine (missing {missing or 'output'})")
        self.batch_size = self._in_shape["model_input"][0]
        self.seq_len = self._in_shape["encoder_hidden_states"][1]

        self._hooks = []
        if self.spec.has_lllite:
            if controller is None:
                raise ValueError("this engine contains the LLLite adapter; pass its controller")
            for i, name in enumerate(self.spec.hook_names):
                if name not in controller.modules_dict:
                    raise ValueError(f"engine expects LLLite hook {name!r}, which the controller lacks "
                                     f"(built with --lllite_blocks {self.spec.lllite_blocks})")
                if cond_input_name(i) not in self._in_shape:
                    raise RuntimeError(f"{engine_path}: engine input {cond_input_name(i)} is missing")
                self._hooks.append(controller.modules_dict[name])
        self._cond: List[Optional[torch.Tensor]] = [None] * len(self._hooks)
        self._cond_key: List[Optional[tuple]] = [None] * len(self._hooks)
        self._keep: Dict[str, torch.Tensor] = {}
        # The engine gets a stream of its own: on the default stream TensorRT
        # inserts host-side synchronisations.
        self._stream = torch.cuda.Stream(device=self.device)

    # ------------------------------------------------------------------
    def _pad_rows(self, t: torch.Tensor) -> torch.Tensor:
        n = t.shape[0]
        if n == self.batch_size:
            return t
        return torch.cat([t, t[-1:].expand(self.batch_size - n, *t.shape[1:])], dim=0)

    def _cond_inputs(self) -> Dict[str, torch.Tensor]:
        """The controller's cond buffers as engine inputs, converted only when they changed."""
        feeds = {}
        for i, m in enumerate(self._hooks):
            buf = m._cond_emb_buf
            key = (buf.data_ptr(), buf._version, buf.dtype)
            if key != self._cond_key[i]:
                name = cond_input_name(i)
                if tuple(buf.shape) != self._in_shape[name]:
                    raise RuntimeError(f"LLLite cond buffer {tuple(buf.shape)} does not match the engine's "
                                       f"{self._in_shape[name]}; batch size or resolution differs from the build")
                self._cond[i] = buf.to(self._in_dtype[name]).contiguous().clone()
                self._cond_key[i] = key
            feeds[cond_input_name(i)] = self._cond[i]
        return feeds

    @torch.no_grad()
    def __call__(self, model_input: torch.Tensor, timestep: torch.Tensor,
                 encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor,
                 added_cond_kwargs: dict, return_dict: bool = False) -> Tuple[torch.Tensor]:
        n = model_input.shape[0]
        if n > self.batch_size:
            raise ValueError(f"batch of {n} frames exceeds the engine's batch size {self.batch_size}")
        if tuple(model_input.shape[1:]) != self._in_shape["model_input"][1:]:
            raise ValueError(f"input {tuple(model_input.shape)} does not match the engine's "
                             f"{self._in_shape['model_input']}; the engine is for {self.spec.size}px")
        out_dtype = model_input.dtype

        # Prompt projection in fp32, outside the engine.
        ehs = self.hid_proj(encoder_hidden_states.to(torch.float32))
        mask = encoder_attention_mask
        seq = ehs.shape[1]
        if seq < self.seq_len:  # masked padding up to the engine's fixed length
            ehs = torch.cat([ehs, ehs.new_zeros(n, self.seq_len - seq, ehs.shape[2])], dim=1)
            mask = torch.cat([mask, mask.new_zeros(n, self.seq_len - seq)], dim=1)
        elif seq > self.seq_len:
            ehs, mask = ehs[:, : self.seq_len], mask[:, : self.seq_len]

        feeds = {
            "model_input": model_input,
            "timestep": timestep.reshape(-1).expand(n) if timestep.numel() == 1 else timestep,
            "encoder_hidden_states": ehs,
            "encoder_attention_mask": mask,
            "time_ids": added_cond_kwargs["time_ids"],
        }
        feeds = {k: self._pad_rows(v).to(self._in_dtype[k]).contiguous() for k, v in feeds.items()}
        feeds.update(self._cond_inputs())

        for name, t in feeds.items():
            self.context.set_tensor_address(name, t.data_ptr())
        self.context.set_tensor_address(self._out_name, self._out.data_ptr())
        # The engine runs asynchronously; its inputs must outlive it, so hold
        # them until the next call replaces them. Ordering across the two
        # streams: the engine waits for the caller's pending work (the
        # inputs), and the caller's later work waits for the engine.
        self._keep = feeds
        caller = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(caller)
        if not self.context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        caller.wait_stream(self._stream)
        return (self._out[:n].to(out_dtype),)

    # nn.Module look-alikes used by callers that treat this as the UNet
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self
