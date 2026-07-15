"""Download and verify OpenVE-Bench (Lewandofski/OpenVE-Bench, CC-BY-NC-4.0).

431 video-edit pairs across 8 categories (Global Style, Background Change,
Local Change, Local Add, Local Remove, Subtitles Edit, Camera Edit, Creative
Edit). See notes/openve_bench_plan.md for the integration plan.

After download, this script writes ``data/openve_bench/index.jsonl`` with one
flat record per pair: ``{pair_id, category, src_mp4, prompt}``. Downstream
``openve_eval.py`` consumes that index.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(_ROOT / "data" / "openve_bench"))
    p.add_argument("--repo_id", default="Lewandofski/OpenVE-Bench")
    p.add_argument("--skip_download", action="store_true",
                   help="Dataset already on disk; just rebuild index.jsonl.")
    return p.parse_args()


def download(out_dir: Path, repo_id: str) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(out_dir))


def build_index(out_dir: Path) -> int:
    csv_path = out_dir / "benchmark_videos.csv"
    if not csv_path.exists():
        sys.exit(f"missing {csv_path}")

    index_path = out_dir / "index.jsonl"
    n = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f_csv, \
         index_path.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_csv)
        for row in reader:
            category = row["edited_type"].strip()
            prompt = row["prompt"].strip()
            src = row["original_video"].strip()
            src_name = Path(src).name
            pair_id = src_name.split("_", 1)[0]
            record = {
                "pair_id": pair_id,
                "category": category,
                "src_mp4": str(out_dir / "videos" / src_name),
                "prompt": prompt,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(f"[index] {index_path}  ({n} pairs)")
    return n


def verify(out_dir: Path) -> None:
    videos_dir = out_dir / "videos"
    if not videos_dir.exists():
        sys.exit(f"missing {videos_dir}")
    n_mp4 = sum(1 for _ in videos_dir.glob("*.mp4"))
    print(f"[verify] {n_mp4} mp4 files in {videos_dir}")
    if n_mp4 != 431:
        print(f"[warn] expected 431 mp4 files, found {n_mp4}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print(f"[download] {args.repo_id} -> {out_dir}")
        download(out_dir, args.repo_id)

    verify(out_dir)
    build_index(out_dir)


if __name__ == "__main__":
    main()
