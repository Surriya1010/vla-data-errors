#!/usr/bin/env python
"""Write a corrupted copy of the LeRobot LIBERO dataset for one condition.

    python scripts/02_make_corrupted.py --list-suite-episodes
    python scripts/02_make_corrupted.py --condition action_lag_2
    python scripts/02_make_corrupted.py --condition clean

Output: data/conditions/<condition>/
    meta/      copied (tasks/episodes metadata updated for instruction swaps)
    data/      parquet files rewritten with the corrupted columns
    videos/    symlinked to the source (camera frames are never modified)
    manifest.json   ground truth: what was injected where, with seeds

Only the action, state and task-index columns are changed. Episode lengths
and frame order are preserved, so camera frames stay aligned.

STATUS: written against LeRobot dataset format v3.0 as documented on
2026-11-23, but NOT yet run on the real dataset. It validates the format first
and stops with a clear message if anything differs from these assumptions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vla_errors import action_jitter, action_lag, instruction_swap, state_bias  # noqa: E402

ENV = dict(
    line.split("#")[0].strip().split("=", 1)
    for line in (REPO / "configs/train.env").read_text().splitlines()
    if "=" in line.split("#")[0] and not line.startswith("export")
)
REQUIRED_COLS = ["action", "observation.state", "episode_index", "frame_index", "task_index"]


def source_root(revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(ENV["DATASET_REPO"], repo_type="dataset", revision=revision))


def load_tasks(root: Path) -> pd.DataFrame:
    """meta/tasks.parquet -> DataFrame with columns task_index, task."""
    t = pd.read_parquet(root / "meta" / "tasks.parquet")
    if "task" not in t.columns:  # v3 stores the task string as the index
        t = t.reset_index().rename(columns={t.index.name or "index": "task"})
    if not {"task", "task_index"} <= set(t.columns):
        raise SystemExit(f"Unexpected meta/tasks.parquet columns: {list(t.columns)}")
    return t[["task_index", "task"]]


def suite_task_strings(suite: str) -> set[str]:
    from libero.libero import benchmark

    s = benchmark.get_benchmark_dict()[suite]()
    return {s.get_task(i).language for i in range(s.n_tasks)}


def read_data(root: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files under {root/'data'}")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df["_file"] = str(f.relative_to(root))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset format differs from assumptions. Missing columns {missing}; "
                         f"found {list(df.columns)}")
    return df, files


def stack(col: pd.Series) -> np.ndarray:
    return np.stack(col.to_numpy()).astype(np.float32)


def list_suite_episodes(root: Path, suite: str) -> list[int]:
    tasks = load_tasks(root)
    wanted = suite_task_strings(suite)
    idx = set(tasks.loc[tasks["task"].isin(wanted), "task_index"])
    if len(idx) != len(wanted):
        raise SystemExit(f"Matched {len(idx)} of {len(wanted)} {suite} task strings in the dataset")
    df, _ = read_data(root)
    eps = df.loc[df["task_index"].isin(idx), "episode_index"].unique()
    return sorted(int(e) for e in eps)


def apply_condition(df: pd.DataFrame, cond: dict, suite_tasks: np.ndarray, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    kind = cond["type"]
    truth: dict = {"type": kind, "params": {k: v for k, v in cond.items() if k != "type"}, "seed": seed}
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    groups = df.groupby("episode_index").indices

    if kind == "none":
        return df, truth

    if kind in ("action_lag", "action_jitter"):
        actions = stack(df["action"])
        ref_std = actions.std(axis=0)
        new = actions.copy()
        for ep, rows in groups.items():
            if kind == "action_lag":
                new[rows] = action_lag(actions[rows], int(cond["k"]))
            else:
                new[rows] = action_jitter(actions[rows], float(cond["strength"]), ref_std, rng)
        df["action"] = list(new)
        truth["affected_episodes"] = sorted(int(e) for e in groups)
        return df, truth

    if kind == "state_bias":
        states = stack(df["observation.state"])
        df["observation.state"] = list(state_bias(states, float(cond["offset_m"])))
        truth["affected_episodes"] = sorted(int(e) for e in groups)
        return df, truth

    if kind == "instruction_swap":
        ep_ids = np.array(sorted(groups))
        ep_task = np.array([df.loc[groups[e][0], "task_index"] for e in ep_ids])
        in_suite = np.isin(ep_task, suite_tasks)
        new_task, mask_suite = instruction_swap(ep_task[in_suite], suite_tasks, float(cond["fraction"]), rng)
        relabel = dict(zip(ep_ids[in_suite][mask_suite], new_task[mask_suite]))
        for ep, t in relabel.items():
            df.loc[groups[ep], "task_index"] = t
        truth["swapped"] = {int(e): {"true": int(ep_task[ep_ids == e][0]), "new": int(t)} for e, t in relabel.items()}
        return df, truth

    raise SystemExit(f"Unknown condition type {kind!r}")


def write_copy(src: Path, out: Path, df: pd.DataFrame, truth: dict, tasks: pd.DataFrame) -> None:
    if out.exists():
        raise SystemExit(f"{out} already exists; delete it first to regenerate")
    out.mkdir(parents=True)
    shutil.copytree(src / "meta", out / "meta")
    if (src / "videos").exists():
        (out / "videos").symlink_to((src / "videos").resolve(), target_is_directory=True)
    for rel, part in df.groupby("_file"):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        part.drop(columns="_file").to_parquet(dst, index=False)
    if truth.get("swapped"):  # keep per-episode task metadata consistent, if present
        name = dict(zip(tasks["task_index"], tasks["task"]))
        for f in sorted((out / "meta" / "episodes").rglob("*.parquet")):
            ep = pd.read_parquet(f)
            if "tasks" in ep.columns and "episode_index" in ep.columns:
                swapped = truth["swapped"]
                ep["tasks"] = [
                    [name[swapped[int(e)]["new"]]] if int(e) in swapped else t
                    for e, t in zip(ep["episode_index"], ep["tasks"])
                ]
                ep.to_parquet(f, index=False)
    (out / "manifest.json").write_text(json.dumps(truth, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", help="name from configs/conditions.yaml")
    ap.add_argument("--list-suite-episodes", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="corruption seed")
    args = ap.parse_args()

    suite, revision = ENV["SUITE"], ENV["DATASET_REVISION"]
    src = source_root(revision)
    print(f"source dataset: {src}")

    if args.list_suite_episodes:
        eps = list_suite_episodes(src, suite)
        path = REPO / "data" / f"suite_episodes_{suite}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text("[" + ",".join(map(str, eps)) + "]")
        print(f"{len(eps)} episodes in {suite} -> {path}")
        return

    if not args.condition:
        ap.error("--condition or --list-suite-episodes is required")
    conds = yaml.safe_load((REPO / "configs/conditions.yaml").read_text())
    if args.condition not in conds:
        raise SystemExit(f"Unknown condition. Choose from: {', '.join(conds)}")

    tasks = load_tasks(src)
    suite_tasks = tasks.loc[tasks["task"].isin(suite_task_strings(suite)), "task_index"].to_numpy()
    df, _ = read_data(src)
    df, truth = apply_condition(df, conds[args.condition], suite_tasks, args.seed)
    truth.update(condition=args.condition, source=str(src), dataset_revision=revision, suite=suite)
    out = REPO / "data" / "conditions" / args.condition
    write_copy(src, out, df, truth, tasks)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
