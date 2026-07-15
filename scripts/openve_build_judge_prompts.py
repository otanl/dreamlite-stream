"""Parse the verbatim OpenVE-Bench Appendix-F judge prompts into JSON.

Input : notes/openve_appendixF_raw.txt (extracted from arXiv:2512.07826v2
        HTML; LaTeX-to-text artifacts possible -- review the JSON once).
Output: data/openve_bench/judge_prompts.json
        {category: full_prompt_text}

Category mapping (index.jsonl category -> Appendix prompt title):
    global_style      -> Prompt 1: Global Style
    background_change -> Prompt 2: Background Change
    local_change      -> Prompt 3: Local Change
    local_remove      -> Prompt 4: Local Remove
    local_add         -> Prompt 5: Local Add
    subtitle_edit     -> Prompt 6: Subtitles Edit
    camera_edit       -> Prompt 7: Camera Multi-shot Edit
    creative_edit     -> Prompt 8 if present in the raw dump, else
                         fallback to Prompt 1's structure with a generic
                         header (flagged in the JSON so the paper can
                         disclose the substitution).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
RAW = _ROOT / "notes" / "openve_appendixF_raw.txt"
OUT = _ROOT / "data" / "openve_bench" / "judge_prompts.json"

CATEGORY_TO_TITLE = {
    "global_style": "Global Style",
    "background_change": "Background Change",
    "local_change": "Local Change",
    "local_remove": "Local Remove",
    "local_add": "Local Add",
    "subtitle_edit": "Subtitles Edit",
    "camera_edit": "Camera Multi-shot Edit",
    "creative_edit": "Creative Edit",
}


def main() -> None:
    raw = RAW.read_text(encoding="utf-8")
    # Split on "Prompt. N: Title" headers.
    parts = re.split(r"Prompt\.\s*(\d+):\s*([^\n]+)\n", raw)
    # parts = [pre, num, title, body, num, title, body, ...]
    prompts: dict[str, str] = {}
    for i in range(1, len(parts) - 2, 3):
        title = parts[i + 1].strip()
        body = parts[i + 2]
        # Body ends where the next non-prompt content begins; cut at the
        # "Below are the videos" sentinel (inclusive) when present.
        m = re.search(r"Below are the videos[^\n]*", body)
        if m:
            body = body[: m.end()]
        prompts[title] = body.strip()

    out: dict[str, dict] = {}
    for cat, title in CATEGORY_TO_TITLE.items():
        if title in prompts:
            out[cat] = {"verbatim": True, "title": title,
                        "prompt": prompts[title]}
        else:
            # Fallback: generic header + Global Style's rubric skeleton.
            base = prompts.get("Global Style", "")
            out[cat] = {
                "verbatim": False,
                "title": title,
                "prompt": ("You are a data rater specializing in grading "
                           "creative video edits. You will be given two "
                           "videos (before and after editing) and the "
                           "editing instruction. Evaluate on a 5-point "
                           "scale from three perspectives as below.\n"
                           + base[base.find("Instruction Compliance"):]),
            }
            print(f"WARNING: no verbatim prompt for '{title}' -- "
                  f"generic fallback recorded (disclose in paper).")

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    for cat, rec in out.items():
        print(f"{cat}: verbatim={rec['verbatim']} len={len(rec['prompt'])}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
