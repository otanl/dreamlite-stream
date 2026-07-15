"""Score OpenVE-Bench edited videos against (source, instruction) via VLM-judge.

Implements the protocol described in arXiv:2512.07826 §4.2: for each
(src_mp4, edit_mp4, instruction) triple, prompt a VLM judge to emit three
1-5 scores along Instruction Compliance / Consistency & Detail Fidelity /
Visual Quality & Stability axes, with the monotonicity constraint that the
2nd and 3rd score must not exceed the 1st.

The exact judge prompts live in OpenVE-3M Appendix F (paper PDF). They must
be lifted verbatim before the actual scored run; the placeholders in
``_JUDGE_PROMPT_TEMPLATE`` below are pre-flight scaffolding.

Supported judges (see notes/openve_bench_plan.md §2 §4 §7):
- seed-1.6vl  (closed, ByteDance Volcengine API — headline in paper Table 2)
- gemini-2.5-pro (closed, Google AI Studio API — headline in paper Table 3)
- internvl-3.5-38b (open weights, runs locally — tie-break / reproducibility)

Caches one response per (pair_id, judge) to ``out/openve/judge_<judge>.jsonl``
so a partial run can resume without re-paying the API cost.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# TODO(openve): replace placeholder with verbatim Appendix F prompt.
# Three components: system + user_template + parser. The user_template must
# embed (instruction, src_mp4, edit_mp4) per the official protocol.
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """\
You are an evaluator for instruction-guided video editing. Given a source
video, an edited video, and an editing instruction, rate the edit on three
axes from 1 (worst) to 5 (best):

1. Instruction Compliance: how well does the edited video follow the
   instruction?
2. Consistency & Detail Fidelity: are non-edited regions preserved and is
   per-frame detail coherent? This score must be no higher than (1).
3. Visual Quality & Stability: is the edited video temporally stable and
   visually high quality? This score must be no higher than (1).

Instruction: {instruction}

Source video: {src_mp4}
Edited video: {edit_mp4}

Respond with strict JSON only, e.g.:
{{"instruction_compliance": 4, "consistency_detail": 3, "visual_quality_stability": 3, "rationale": "..."}}
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results",
                   default=str(_ROOT / "out" / "openve" / "results.jsonl"))
    p.add_argument("--judge", choices=["seed-1.6vl", "gemini-2.5-pro", "internvl-3.5-38b"],
                   required=True)
    p.add_argument("--out_dir", default=str(_ROOT / "out" / "openve"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sleep_s", type=float, default=0.0,
                   help="Sleep between API calls (rate-limit pacing).")
    return p.parse_args()


def load_results(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    cache: dict[str, dict] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cache[rec["pair_id"]] = rec
    return cache


def judge_seed_1_6vl(prompt: str, src_mp4: str, edit_mp4: str) -> dict:
    """Call Seed-1.6VL via ByteDance Volcengine.

    TODO(openve): wire up. Requires ``VOLCENGINE_API_KEY`` env var and the
    multimodal endpoint that accepts mp4 inputs. Return shape:
        {"instruction_compliance": int, "consistency_detail": int,
         "visual_quality_stability": int, "rationale": str, "raw": <response>}.
    """
    raise NotImplementedError("Seed-1.6VL judge: wire to Volcengine API.")


def judge_gemini(prompt: str, src_mp4: str, edit_mp4: str) -> dict:
    """Call Gemini 2.5 Pro via Google AI Studio.

    TODO(openve): wire up. Requires ``GOOGLE_API_KEY`` env var and the
    multimodal endpoint that accepts mp4 inputs (uploaded via File API).
    """
    raise NotImplementedError("Gemini 2.5 Pro judge: wire to GAI Studio API.")


def judge_internvl(prompt: str, src_mp4: str, edit_mp4: str) -> dict:
    """Call InternVL3.5-38B locally.

    TODO(openve): wire up. The model is gated on HF; once downloaded, frame-
    sample both videos (e.g. 8 evenly-spaced frames each), tile into a single
    prompt context, and run the standard chat template. Open-weights judge
    is the reproducibility anchor; expect ~10s per pair on a 3090 Ti.
    """
    raise NotImplementedError("InternVL3.5-38B judge: wire to local HF model.")


_JUDGES = {
    "seed-1.6vl": judge_seed_1_6vl,
    "gemini-2.5-pro": judge_gemini,
    "internvl-3.5-38b": judge_internvl,
}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"judge_{args.judge}.jsonl"

    rows = load_results(Path(args.results))
    if args.limit > 0:
        rows = rows[: args.limit]
    cache = load_cache(cache_path)
    print(f"[judge={args.judge}] {len(rows)} pairs, {len(cache)} cached")

    judge_fn = _JUDGES[args.judge]
    with cache_path.open("a", encoding="utf-8") as fout:
        for i, rec in enumerate(rows):
            if rec["pair_id"] in cache:
                continue
            prompt = _JUDGE_PROMPT_TEMPLATE.format(
                instruction=rec.get("prompt", ""),
                src_mp4=rec["src_mp4"],
                edit_mp4=rec["edit_mp4"],
            )
            try:
                scores = judge_fn(prompt, rec["src_mp4"], rec["edit_mp4"])
            except NotImplementedError as e:
                sys.exit(f"[abort] {e}")
            except Exception as e:
                print(f"[error] {rec['pair_id']}: {e}")
                continue
            record = {
                "pair_id": rec["pair_id"],
                "category": rec["category"],
                **scores,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i+1}/{len(rows)}] {rec['category']}/{rec['pair_id']} "
                  f"IC={scores.get('instruction_compliance')} "
                  f"CD={scores.get('consistency_detail')} "
                  f"VQ={scores.get('visual_quality_stability')}")
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
    print(f"[saved] {cache_path}")


if __name__ == "__main__":
    main()
