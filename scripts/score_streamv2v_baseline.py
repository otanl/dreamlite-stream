"""Score the StreamV2V baseline outputs with OUR temporal metric, identically
to champion_eval.py, and merge with the recorded timing to produce the
head-to-head comparison table (StreamV2V vs DreamLite-stream champion).

Input:
  out/streamv2v_baseline/streamv2v_timing.jsonl   (from bench_streamv2v.py)
  out/streamv2v_baseline/<clip>.mp4               (StreamV2V outputs)
  assets/davis_mp4/<clip>.mp4                      (identical flow-source input)
  data/champion_main.jsonl                         (our per-clip champion numbers)

Run (main dreamlite env with cv2):
  python scripts/score_streamv2v_baseline.py
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean, stdev
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from dreamlite_stream.metrics import compute_temporal, read_video_frames  # noqa

BASE = _ROOT / "out" / "streamv2v_baseline"
IN_DIR = _ROOT / "assets" / "davis_mp4"
CHAMP = _ROOT / "data" / "champion_main.jsonl"
SIZE = 512


def main() -> None:
    champ = {json.loads(l)["sequence"]: json.loads(l) for l in open(CHAMP)}
    timing = {json.loads(l)["clip"]: json.loads(l)
              for l in open(BASE / "streamv2v_timing.jsonl")}

    rows = []
    for clip, t in timing.items():
        in_frames = read_video_frames(str(IN_DIR / f"{clip}.mp4"), size=SIZE)
        out_frames = read_video_frames(str(BASE / f"{clip}.mp4"))
        n = min(len(in_frames), len(out_frames))
        m = compute_temporal(in_frames[:n], out_frames[:n])
        c = champ.get(clip, {})
        rows.append({
            "clip": clip, "n": n,
            "sv2v_fps": t["fps_steady"],
            "sv2v_warp": m.warping_error,
            "ours_fps": c.get("fps"),
            "ours_warp": c.get("warp_err"),
        })
        print(f"{clip:14s} n={n:3d}  SV2V fps={t['fps_steady']:5.2f} "
              f"warp={m.warping_error:6.2f}   OURS fps={c.get('fps'):5.2f} "
              f"warp={c.get('warp_err'):6.2f}")

    def agg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)

    print("\n=== aggregate (10 DAVIS clips, 512x512, RTX 3090 Ti, no TensorRT) ===")
    for label, k in [("StreamV2V fps", "sv2v_fps"),
                     ("StreamV2V warp", "sv2v_warp"),
                     ("Ours fps", "ours_fps"),
                     ("Ours warp", "ours_warp")]:
        mu, sd = agg(k)
        print(f"  {label:16s} {mu:7.2f} ± {sd:5.2f}")

    mu_s, _ = agg("sv2v_fps")
    mu_o, _ = agg("ours_fps")
    print(f"\n  throughput ratio (ours / StreamV2V): {mu_o / mu_s:.2f}x")

    out = BASE / "comparison_table.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
