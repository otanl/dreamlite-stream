# Attribution

`dreamlite-stream` is a derivative work that builds on the following
projects. **Code** in this repository is released under Apache-2.0
(see `LICENSE`); **trained adapter weights** in this repository
(`runs/temporal_lllite_v3/temporal_lllite_step001440.safetensors`)
are derivatives of DreamLite-mobile and inherit DreamLite's
non-commercial weight license — see below.

## DreamLite

This repository's runtime, training scripts, and trained adapter
weights are **Adapted Material** (CC BY-NC 4.0 §1(a)) of
[DreamLite](https://github.com/ByteVisionLab/DreamLite) by Kailai
Feng et al. (ByteDance Ltd.).

- Paper: *DreamLite: A Lightweight On-Device Unified Model for
  Image Generation and Editing.* arXiv:2603.28713 (2026).
- License (weights): CC BY-NC 4.0 — see the upstream
  [`WEIGHTS_LICENSE`](https://github.com/ByteVisionLab/DreamLite/blob/main/WEIGHTS_LICENSE).
- Modifications in this repo: We add an asymmetric side-stream /
  main-stream inference pipeline, a periodic cond-refresh schedule,
  a `down_blocks` hook-subset selector for the
  [`dreamlite-lllite`](../dreamlite-lllite) adapter, a 1-step
  inference path that does not retrain DreamLite-mobile, and the
  Temporal LLLite v3 trained adapter. **No DreamLite UNet or
  Qwen3-VL TE weights are redistributed by this repository.**

## ControlNet-LLLite (kohya-ss)

The adapter architecture used here is the diffusers-side port in
the sibling [`dreamlite-lllite`](../dreamlite-lllite) repo, itself
derived from [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
(Apache-2.0) and
[kohya-ss/ControlNet-LLLite-ComfyUI](https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI).
See [`dreamlite-lllite/ATTRIBUTION.md`](../dreamlite-lllite/ATTRIBUTION.md)
for the adapter-side modification log.

## Qwen3-VL (text encoder)

DreamLite-mobile uses [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
as its text encoder. Our pipeline routes Qwen3-VL features through
the LLLite cond encoder; users of the trained adapter weights in
this repository are bound by the relevant Qwen3-VL license terms in
addition to the DreamLite license.

## DAVIS-2017 (evaluation and demo video)

The headline evaluation in the paper uses
[DAVIS-2017](https://davischallenge.org/davis2017/code.html). Its
organizers license the annotations they created under CC BY 4.0
(challenge rules page), and the site's dataset metadata lists
CC BY-NC 4.0; no separate licence is stated for the video frames.
We use the frames only, for non-commercial research: the 480p source
frames are re-encoded as mp4 fixtures under `assets/davis_mp4/`, and
the demo video in the README (made by `scripts/make_demo.py`) shows
five of these clips next to their stylized outputs.

## Cross-dataset evaluation sources

The cross-dataset evaluation (Table X (c)) uses public
clips from:
- Big Buck Bunny — Blender Foundation, CC-BY
- Sintel — Blender Foundation, CC-BY
- Jellyfish — test-videos.co.uk
- MDN samples — Mozilla, CC0
- learning-container CC samples
- Intel IoT public sample videos

These clips are not redistributed by this repository; see
`assets/README.md` for source URLs and trim windows used.

## Diffusers / Transformers / TensorRT

Pipeline integration uses
[`diffusers`](https://github.com/huggingface/diffusers) and
[`transformers`](https://github.com/huggingface/transformers),
both Apache-2.0. The TRT export path uses NVIDIA TensorRT (proprietary
SDK; users must accept the NVIDIA TensorRT license).

## RAFT (§IV-C)

The optical-flow sanity check in §IV-C uses RAFT-Large
weights distributed via `torchvision.models.optical_flow.raft_large`
(`C_T_SKHT_V2`); see torchvision's license. RAFT itself is by
[Teed and Deng, ECCV 2020](https://github.com/princeton-vl/RAFT).

## RIFE (display-side frame interpolation)

`src/dreamlite_stream/rife.py` contains the flow network of RIFE v4.25
(`Head`, `ResConv`, `IFBlock`, `IFNet` and the warp), re-written for
batched inference from `IFNet_HDv3.py` and `warplayer.py` of
[Practical-RIFE](https://github.com/hzwer/Practical-RIFE) by Zhewei Huang
et al., with parameter names kept so that the upstream checkpoint loads.
RIFE is described in *Real-Time Intermediate Flow Estimation for Video
Frame Interpolation*, ECCV 2022. **No RIFE weights are redistributed by
this repository**; users download `flownet.pkl` from the upstream links.

Practical-RIFE is released under the MIT License:

> MIT License
>
> Copyright (c) 2021 hzwer
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## LPIPS (scene-cut evaluation + spatial fidelity probes)

The LPIPS spatial-fidelity probe uses
[`lpips`](https://github.com/richzhang/PerceptualSimilarity)
with the AlexNet backbone (BSD-2-Clause).

---

If you redistribute the trained Temporal LLLite weights produced
by this code, you must preserve all attributions in this file and
clearly state that the weights inherit DreamLite's CC BY-NC 4.0
license (non-commercial use only).
