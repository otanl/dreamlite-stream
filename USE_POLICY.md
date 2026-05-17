# Use Policy

This repository's **trained Temporal LLLite weights**
(`runs/temporal_lllite_v3/temporal_lllite_step001440.safetensors`)
are Adapted Material of DreamLite-mobile and inherit DreamLite's
weight licence (CC BY-NC 4.0). Anyone using these adapter weights,
or training new adapter weights with the code in this repository
on top of DreamLite-mobile, is bound by **both** the DreamLite
weight licence and the upstream content/use policy reproduced
below verbatim.

## Upstream DreamLite use notice

> ⚠️ **Important Usage and Compliance Notice:**
> By accessing and using these models, you agree to abide by our
> ethical guidelines. These models **MUST NOT** be used to
> generate, edit, or distribute any content that is sexually
> explicit, pornographic, violent, discriminatory, or otherwise
> illegal. These models are released for **research and
> non-commercial use only**, and **MUST NOT** be used for any
> commercial purposes. We strictly prohibit the use of DreamLite
> for malicious purposes.

Source: notice attached to the DreamLite-mobile early-access
release. The above text is the binding policy; this repository
restates it for visibility but does not modify or relax it.

## What this means in practice

- **Non-commercial only.** The trained adapter weights here may
  not be used in any commercial product, paid service, or
  revenue-generating application. This restriction is inherited
  from DreamLite and does not depend on whether the code is
  Apache-2.0 (it is).
- **No NSFW / violent / discriminatory / illegal content.** This
  applies to inputs *and* outputs — both as evaluation material
  and as deployed applications.
- **No malicious use.** Identity manipulation, deepfake / consent-
  violating synthesis, surveillance targeting individuals, and
  similar misuses are prohibited.
- **Token security.** If you obtained an early-access token to
  download DreamLite-mobile, do not commit it to any repository
  or share it publicly. We do not redistribute the token or the
  underlying base-model weights.

## What this repository contributes on top

- **Code** is released under Apache-2.0 (see `LICENSE`). The code
  permits commercial use *in principle*, but in practice it does
  nothing useful without DreamLite-mobile weights, which are
  themselves non-commercial. So a commercial application of this
  pipeline would require either (a) a commercial licence from the
  DreamLite authors, or (b) substituting a different base model
  with a commercial-permissive licence.
- **Trained adapter weights** are dual-bound: distributed here at
  no cost for non-commercial research, under CC BY-NC 4.0 with
  attribution to DreamLite (see `ATTRIBUTION.md`).

If you are unsure whether your intended use is compliant, please
contact the DreamLite authors directly; their decision on the
weight licence is authoritative.
