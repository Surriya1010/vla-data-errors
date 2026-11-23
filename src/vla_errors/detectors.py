"""Label-free detectors for training-data errors, one episode at a time.

They use only the recorded action and state streams, never the injection
labels, so they can be run on real datasets too. Their accuracy is measured
against the known injected errors (scripts/05_run_detectors.py).
"""

from __future__ import annotations

import numpy as np

from .corruptions import ARM_DIMS, POS_DIMS


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return np.divide(x, sd, out=np.zeros_like(x), where=sd > 1e-12)


def lag_profile(
    actions: np.ndarray,
    states: np.ndarray,
    max_lag: int = 8,
    action_dims: slice = POS_DIMS,
    state_dims: slice = POS_DIMS,
) -> dict[int, float]:
    """Correlation between commanded translation and end-effector motion, per delay.

    For a candidate delay d, pairs action[t + d] with the motion
    state[t + 1] - state[t]. If the action stream was delayed by d frames,
    that pairing restores the original alignment and the correlation peaks.
    Returns {d: mean correlation over the translation dims}.
    """
    a = np.asarray(actions, dtype=np.float64)[:, action_dims]
    s = np.asarray(states, dtype=np.float64)[:, state_dims]
    motion = np.diff(s, axis=0)  # motion[t] = s[t+1] - s[t], length T-1
    T = motion.shape[0]
    out: dict[int, float] = {}
    for d in range(-max_lag, max_lag + 1):
        t0, t1 = max(0, -d), min(T, a.shape[0] - d)
        if t1 - t0 < 3:
            continue
        x = _zscore(a[t0 + d : t1 + d])
        y = _zscore(motion[t0:t1])
        out[d] = float((x * y).mean())
    return out


def estimate_lag(actions: np.ndarray, states: np.ndarray, max_lag: int = 8) -> int:
    """Estimated delay (frames) of the action stream relative to the observations.

    0 means aligned (relative to how the simulator responds). In real data the
    controller itself adds some response delay, so compare episodes against the
    clean dataset's typical value rather than against 0.
    """
    prof = lag_profile(actions, states, max_lag=max_lag)
    if not prof:
        raise ValueError("episode too short to estimate lag")
    return max(prof, key=prof.get)


def jitter_score(actions: np.ndarray, cutoff: float = 0.5, dims: slice = ARM_DIMS) -> float:
    """Share of arm-action energy above `cutoff` x Nyquist frequency.

    Smooth teleoperation puts most energy at low frequency; added noise spreads
    it across all frequencies, so the score rises with jitter. Returns a value
    in [0, 1]; 0 for a constant signal.
    """
    a = np.asarray(actions, dtype=np.float64)[:, dims]
    a = a - a.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(a, axis=0)) ** 2
    total = power.sum()
    if total <= 1e-12:
        return 0.0
    freqs = np.fft.rfftfreq(a.shape[0])  # 0 .. 0.5 cycles/frame
    high = freqs >= cutoff * 0.5
    return float(power[high].sum() / total)


def stall_runs(actions: np.ndarray, eps: float = 1e-3, dims: slice = ARM_DIMS) -> int:
    """Length (frames) of the longest run of near-zero arm action."""
    a = np.asarray(actions, dtype=np.float64)[:, dims]
    still = np.linalg.norm(a, axis=1) < eps
    best = run = 0
    for s in still:
        run = run + 1 if s else 0
        best = max(best, run)
    return best
