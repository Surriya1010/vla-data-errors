#!/usr/bin/env python
"""Run the label-free detectors on a dataset copy and score them against the
injected ground truth in its manifest.json.

    python scripts/05_run_detectors.py --condition action_lag_2 [--suite-only]

Writes results/detectors/<condition>.csv (one row per episode) and prints a
summary. Lag is judged relative to the median lag of the clean copy, because
the simulator's own response delay is not zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vla_errors import estimate_lag, jitter_score, stall_runs  # noqa: E402


def episodes(root: Path, only: set[int] | None):
    df = pd.concat(pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet")))
    df = df.sort_values(["episode_index", "frame_index"])
    for ep, g in df.groupby("episode_index"):
        if only is None or int(ep) in only:
            yield int(ep), np.stack(g["action"].to_numpy()), np.stack(g["observation.state"].to_numpy())


def score(root: Path, only: set[int] | None) -> pd.DataFrame:
    rows = [
        dict(episode=ep, lag=estimate_lag(a, s), jitter=jitter_score(a), stall=stall_runs(a))
        for ep, a, s in episodes(root, only)
    ]
    return pd.DataFrame(rows)


def prf(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--suite-only", action="store_true", help="restrict to the training suite's episodes")
    args = ap.parse_args()

    data = REPO / "data"
    only = None
    if args.suite_only:
        env = dict(l.split("#")[0].strip().split("=", 1) for l in (REPO / "configs/train.env").read_text().splitlines()
                   if "=" in l.split("#")[0] and not l.startswith("export"))
        only = set(json.loads((data / f"suite_episodes_{env['SUITE']}.txt").read_text()))

    clean = score(data / "conditions" / "clean", only)
    cond = score(data / "conditions" / args.condition, only)
    truth = json.loads((data / "conditions" / args.condition / "manifest.json").read_text())

    base_lag = int(np.median(clean["lag"]))
    jitter_thr = float(np.quantile(clean["jitter"], 0.95))  # flag above clean 95th percentile
    cond["lag_vs_clean"] = cond["lag"] - base_lag
    cond["flag_lag"] = cond["lag_vs_clean"] != 0
    cond["flag_jitter"] = cond["jitter"] > jitter_thr

    affected = set(truth.get("affected_episodes", [])) | set(map(int, truth.get("swapped", {})))
    cond["injected"] = cond["episode"].isin(affected)

    out = REPO / "results" / "detectors"; out.mkdir(parents=True, exist_ok=True)
    cond.to_csv(out / f"{args.condition}.csv", index=False)

    print(f"condition={args.condition} episodes={len(cond)} injected={int(cond['injected'].sum())}")
    print(f"clean median lag={base_lag}  clean jitter 95th pct={jitter_thr:.4f}")
    if truth["type"] == "action_lag":
        k = truth["params"]["k"]
        err = (cond["lag_vs_clean"] - k).abs()
        print(f"lag estimate: exact={np.mean(err == 0):.1%}  within 1 frame={np.mean(err <= 1):.1%}")
    for name in ("flag_lag", "flag_jitter"):
        print(name, prf(cond[name].to_numpy(), cond["injected"].to_numpy()))
    if truth["type"] == "instruction_swap":
        print("instruction_swap: no motion-based detector (known gap); flags above are expected to be chance")


if __name__ == "__main__":
    main()
