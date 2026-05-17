"""Generate multi-prompt distillation pairs for v4 LLLite training.

Drives generate_temporal_pairs.py over a fixed prompt list, accumulating
into a single output dir. Each pair manifest entry carries its own prompt
so the trainer can shuffle across prompts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PROMPTS = [
    "transfer this to oil painting style, vibrant colors",
    "transfer this to watercolor painting style, soft edges",
    "transfer this to pencil sketch style, fine line work",
    "transfer this to anime art style, clean cel shading",
    "transfer this to 3D render style, ray-traced lighting",
]

DEFAULT_SEQUENCES = [
    "blackswan", "libby", "swing", "camel", "dance-twirl",
    "goat", "scooter-black", "bmx-trees", "parkour", "kite-surf",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    p.add_argument("--mp4_dir", default=str(_ROOT / "assets" / "davis_mp4"))
    p.add_argument("--max_frames_per_seq", type=int, default=40)
    p.add_argument("--out_dir", default=str(_ROOT / "data" / "temporal_pairs_v4_multiprompt"))
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--blend_alpha", type=float, default=0.85)
    p.add_argument("--no_compile", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe previous manifest so accumulate works correctly
    manifest = out_dir / "manifest.jsonl"
    if manifest.exists():
        manifest.unlink()

    inputs = [str(Path(args.mp4_dir) / f"{s}.mp4") for s in args.sequences]
    inputs = [p for p in inputs if Path(p).exists()]

    for i, prompt in enumerate(args.prompts):
        print(f"\n========== prompt {i+1}/{len(args.prompts)}: {prompt!r} ==========")
        cmd = [
            sys.executable,
            str(_ROOT / "scripts" / "generate_temporal_pairs.py"),
            "--inputs", *inputs,
            "--prompt", prompt,
            "--size", "512",
            "--steps", str(args.steps),
            "--max_frames_per_seq", str(args.max_frames_per_seq),
            "--blend_alpha", str(args.blend_alpha),
            "--out_dir", str(out_dir),
        ]
        if args.no_compile:
            cmd.append("--no_compile")
        # generate_temporal_pairs.py wipes manifest at startup; we'd lose
        # earlier prompts. Workaround: read its manifest after each call and
        # accumulate to a master manifest file.
        master = out_dir.parent / "_master_manifest.jsonl"
        if i == 0 and master.exists():
            master.unlink()
        # Run subprocess so each prompt run gets fresh state; capture its output.
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  generate_temporal_pairs.py failed for prompt {i}")
            continue
        # Append per-prompt manifest to master
        if manifest.exists():
            with master.open("a", encoding="utf-8") as fout, manifest.open(encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
            # Also rename per-prompt npz files to avoid collisions
            pairs_dir = out_dir / "pairs"
            for npz in list(pairs_dir.glob("*.npz")):
                if not npz.stem.startswith(f"p{i}_"):
                    new_name = pairs_dir / f"p{i}_{npz.stem}.npz"
                    npz.rename(new_name)

    # Restore master as the canonical manifest
    if (out_dir.parent / "_master_manifest.jsonl").exists():
        (out_dir.parent / "_master_manifest.jsonl").rename(manifest)
        # Also rewrite manifest entries to point at renamed npz files
        new_lines = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            import json
            row = json.loads(line)
            stem = row["stem"]
            # Find prompt index by which prompt occurs in row
            for i, prompt in enumerate(args.prompts):
                if row.get("prompt") == prompt:
                    row["stem"] = f"p{i}_{stem}"
                    break
            new_lines.append(json.dumps(row))
        manifest.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print(f"\n[done] master manifest -> {manifest}")
        print(f"  total pairs: {sum(1 for _ in manifest.open())}")


if __name__ == "__main__":
    main()
