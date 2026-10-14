"""The "lean" operating point of BatchedEditWorker, and its CLI plumbing.

The default path recomputes everything per batch: the Qwen3-VL prompt
embedding for every frame, the LLLite cond on a schedule, fresh init noise,
and it moves frames through the PIL-based image_processor. The lean path
keeps the per-batch work down to the UNet and the VAE:

  te_refresh_every=0 + te_batch_one   prompt embedding computed once per clip
                                      (from one frame) and reused
  cond_refresh_every=0                LLLite cond set once per clip and held
  shared_noise                        one init-noise draw for every frame
  fast_vae + vae_compile              fp16 / channels_last compiled VAE with
                                      GPU-side (de)normalisation, uint8 arrays
                                      in and out

Because the lean path makes the output a deterministic function of the input
frame, one denoising step can also be taught to reproduce the multi-step
output: ``--onestep_lora FILE`` merges such a LoRA (see ``onestep_lora.py``,
trained by ``scripts/train_onestep_lora.py``) into the UNet weights.
``--trt_engine FILE`` runs the UNet as a TensorRT engine built by
``scripts/build_trt_unet.py`` instead of the torch.compile'd module.

Everything here is opt-in. Scripts that reproduce the paper tables keep the
default path unless one of these flags is passed.

Usage in a script::

    add_lean_args(parser)                       # after the script's own args
    args = resolve_lean_args(parser.parse_args())
    pipeline = ...                              # load
    apply_onestep_lora(pipeline, args)          # before the worker compiles the UNet
    controller = None                           # --lllite_multiplier 0: attach nothing
    if args.lllite_multiplier != 0:
        controller = apply_lllite(pipeline.unet, ...)
        controller.set_multiplier(args.lllite_multiplier)
    apply_trt_engine(pipeline, args, controller)  # after the adapter is attached
    worker = BatchedEditWorker(..., lllite_controller=controller, **lean_worker_kwargs(args))

The script's own ``--cond_refresh_every`` (and ``--te_refresh_every``, if it
defines one) should default to ``None`` so that ``--lean`` can tell an
explicit value from an unset one.
"""

from __future__ import annotations

import argparse


def add_lean_args(p: argparse.ArgumentParser, *, te_args: bool = True) -> None:
    """Add the lean-path flags to `p`.

    Pass ``te_args=False`` when the script already defines
    ``--te_refresh_every`` / ``--te_batch_one`` itself.
    """
    g = p.add_argument_group(
        "lean path",
        "Opt-in options that cut per-batch work. The defaults reproduce the "
        "paper's configuration.",
    )
    g.add_argument("--lean", action="store_true",
                   help="Preset: --te_refresh_every 0 --te_batch_one "
                        "--cond_refresh_every 0 --shared_noise --fast_vae "
                        "--vae_compile. Explicit values for the two refresh "
                        "options still win.")
    g.add_argument("--shared_noise", action="store_true",
                   help="Use one init-noise draw for every frame of every "
                        "batch. Removes the frame-to-frame noise difference "
                        "that --fixed_noise leaves in place.")
    g.add_argument("--fast_vae", action="store_true",
                   help="Run the VAE in fp16 / channels_last with "
                        "normalisation on the GPU instead of the PIL "
                        "image_processor; outputs become uint8 arrays.")
    g.add_argument("--vae_compile", action="store_true",
                   help="torch.compile that VAE (implies --fast_vae). Adds a "
                        "compile at start-up.")
    g.add_argument("--onestep_lora", default=None, metavar="FILE",
                   help="Merge a one-step LoRA (scripts/train_onestep_lora.py) "
                        "into the UNet so one step reproduces the multi-step "
                        "output. Trained for one --size and --seed and for the "
                        "shared-noise setting; use with --lean. No run-time "
                        "cost: the weights are merged.")
    g.add_argument("--lllite_multiplier", type=float, default=None,
                   help="Scale of the Temporal LLLite adapter's residuals. "
                        "1.0 = as trained (default). 0 = no adapter: its "
                        "hooks are not attached at all. With --trt_engine "
                        "the default is what the engine was built with.")
    g.add_argument("--trt_engine", default=None, metavar="FILE",
                   help="Run the UNet as a TensorRT engine "
                        "(scripts/build_trt_unet.py) instead of torch.compile. "
                        "An engine is built for one --size and --batch_size "
                        "and already contains its one-step LoRA and adapter "
                        "multiplier.")
    if te_args:
        g.add_argument("--te_refresh_every", type=int, default=None,
                       help="Recompute the Qwen3-VL prompt embedding every N "
                            "batches. 1 = every batch (default). 0 = once per "
                            "clip (default with --lean).")
        g.add_argument("--te_batch_one", action="store_true",
                       help="On a refresh, encode one frame and broadcast the "
                            "embedding to the batch.")


def resolve_lean_args(
    args: argparse.Namespace,
    *,
    cond_refresh_default: int = 8,
    te_refresh_default: int = 1,
) -> argparse.Namespace:
    """Fill in what ``--lean`` implies. Explicit refresh values win."""
    lean = bool(getattr(args, "lean", False))
    if getattr(args, "cond_refresh_every", None) is None:
        args.cond_refresh_every = 0 if lean else cond_refresh_default
    if getattr(args, "te_refresh_every", None) is None:
        args.te_refresh_every = 0 if lean else te_refresh_default
    if lean:
        args.te_batch_one = True
        args.shared_noise = True
        args.fast_vae = True
        args.vae_compile = True
    if getattr(args, "vae_compile", False):
        args.fast_vae = True
    if getattr(args, "lllite_multiplier", None) is None:
        args.lllite_multiplier = 1.0
        if getattr(args, "trt_engine", None):
            from .trt_engine import EngineSpec
            spec = EngineSpec.load(args.trt_engine)
            args.lllite_multiplier = spec.lllite_multiplier if spec.has_lllite else 0.0
    return args


def lean_worker_kwargs(args: argparse.Namespace) -> dict:
    """BatchedEditWorker keyword arguments for the resolved flags."""
    return {
        "cond_refresh_every": args.cond_refresh_every,
        "te_refresh_every": args.te_refresh_every,
        "te_batch_one": bool(getattr(args, "te_batch_one", False)),
        "shared_noise": bool(getattr(args, "shared_noise", False)),
        "fast_vae": bool(getattr(args, "fast_vae", False)),
        "vae_compile": bool(getattr(args, "vae_compile", False)),
        "output_type": "np" if getattr(args, "fast_vae", False) else "pil",
    }


def apply_onestep_lora(pipeline, args: argparse.Namespace):
    """Merge the LoRA named by ``--onestep_lora`` into ``pipeline.unet``.

    Call after loading the pipeline and before the worker is constructed
    (torch.compile must see the merged weights). The UNet is shared by every
    worker built on this pipeline; the returned handle's ``restore()`` undoes
    the merge. Returns None when no LoRA was requested.
    """
    path = getattr(args, "onestep_lora", None)
    if not path:
        return None
    if getattr(args, "trt_engine", None):
        # The engine was built from merged weights; apply_trt_engine checks
        # that they came from this file.
        return None
    from . import onestep_lora

    state, meta = onestep_lora.load(path)
    handle = onestep_lora.merge(pipeline.unet, state, meta)
    print(
        f"[onestep_lora] merged {handle.n_layers} layers from {path} "
        f"(trained at {meta.get('size')}px, seed {meta.get('seed')}, "
        f"{meta.get('teacher_steps')}-step teacher, {len(meta.get('prompts') or [])} prompts)"
    )
    for note in onestep_lora.compatibility_notes(
        meta, size=args.size, seed=getattr(args, "seed", None),
        steps=getattr(args, "steps", 1), shared_noise=bool(getattr(args, "shared_noise", False)),
        prompt=getattr(args, "prompt", None),
    ):
        print(f"[onestep_lora] note: {note}")
    return handle


def apply_trt_engine(pipeline, args: argparse.Namespace, controller=None):
    """Replace ``pipeline.unet`` with the engine named by ``--trt_engine``.

    Call after the LLLite controller is attached to the PyTorch UNet and
    before the worker is constructed. The engine takes over the UNet call
    only; the prompt projection keeps running in PyTorch and the controller
    keeps feeding the adapter's cond. Returns the engine wrapper, or None
    when no engine was requested.
    """
    path = getattr(args, "trt_engine", None)
    if not path:
        return None
    from . import onestep_lora
    from .trt_engine import EngineSpec, TRTEngineUNet

    spec = EngineSpec.load(path)
    problems = spec.mismatches(
        size=args.size, batch_size=args.batch_size,
        onestep_lora=getattr(args, "onestep_lora", None),
        lllite_multiplier=args.lllite_multiplier,
    )
    if problems:
        raise SystemExit(f"--trt_engine {path} does not fit this run:\n  " + "\n  ".join(problems))
    unet = getattr(pipeline.unet, "_orig_mod", pipeline.unet)
    engine = TRTEngineUNet(path, unet.encoder_hid_proj,
                           controller if spec.has_lllite else None, device=args.device)
    pipeline.unet = engine
    print(
        f"[trt] UNet runs as {path} (TensorRT {spec.tensorrt}, {spec.precision}; "
        f"one-step LoRA: {spec.onestep_lora or 'none'}; "
        f"adapter: {f'x{spec.lllite_multiplier:g}' if spec.has_lllite else 'not in the engine'})"
    )
    if spec.onestep_lora_meta:
        for note in onestep_lora.compatibility_notes(
            spec.onestep_lora_meta, size=args.size, seed=getattr(args, "seed", None),
            steps=getattr(args, "steps", 1), shared_noise=bool(getattr(args, "shared_noise", False)),
            prompt=getattr(args, "prompt", None),
        ):
            print(f"[trt] note: the engine's one-step LoRA was {note}")
    return engine


def holds_clip_state(args: argparse.Namespace) -> bool:
    """True if the resolved options keep state that must be reset per clip.

    That is the case once the prompt embedding is reused across batches or
    the LLLite cond is held; such runs should call
    ``worker.reset_sequence()`` between unrelated clips.
    """
    return args.te_refresh_every != 1 or args.cond_refresh_every == 0
