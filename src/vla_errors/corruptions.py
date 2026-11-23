"""Error injection for one episode (or one dataset) of robot demonstrations.

Every function is pure: it returns a new array and never modifies its input.
Arrays are time-major, shape (T, D) for one episode.

LIBERO conventions (LeRobot `lerobot/libero`):
    action[:, 0:6]  end-effector delta (translation 0:3, rotation 3:6), in [-1, 1]
    action[:, 6]    gripper command
    state[:, 0:3]   end-effector position (m)
"""

from __future__ import annotations

import numpy as np

ARM_DIMS = slice(0, 6)
POS_DIMS = slice(0, 3)


def action_lag(actions: np.ndarray, k: int) -> np.ndarray:
    """Delay the action stream by k frames relative to the observations.

    Row t of the result holds the action originally at row t - k. The first k
    rows repeat the first action. k < 0 makes actions lead instead (row t holds
    the action from t + |k|, the last rows repeat the last action).
    The episode length is unchanged, so camera frames stay aligned with state.
    """
    a = np.asarray(actions)
    if a.ndim != 2:
        raise ValueError(f"expected (T, D) array, got shape {a.shape}")
    T = a.shape[0]
    if k == 0 or T == 0:
        return a.copy()
    idx = np.clip(np.arange(T) - k, 0, T - 1)
    return a[idx].copy()


def action_jitter(
    actions: np.ndarray,
    strength: float,
    ref_std: np.ndarray,
    rng: np.random.Generator,
    dims: slice = ARM_DIMS,
    low: float = -1.0,
    high: float = 1.0,
) -> np.ndarray:
    """Add zero-mean Gaussian noise to the arm action dims.

    Noise std per dim = strength * ref_std[dim], where ref_std is the per-dim
    standard deviation of the clean actions over the whole dataset (pass it in
    so every episode uses the same scale). Result is clipped to [low, high].
    The gripper dim is untouched by default.
    """
    a = np.asarray(actions, dtype=np.float64).copy()
    std = np.asarray(ref_std, dtype=np.float64)[dims]
    noise = rng.normal(0.0, 1.0, size=a[:, dims].shape) * (strength * std)
    a[:, dims] = np.clip(a[:, dims] + noise, low, high)
    return a.astype(np.asarray(actions).dtype, copy=False)


def state_bias(states: np.ndarray, offset_m: np.ndarray | float, dims: slice = POS_DIMS) -> np.ndarray:
    """Add a constant calibration offset (metres) to the state position dims."""
    s = np.asarray(states, dtype=np.float64).copy()
    s[:, dims] = s[:, dims] + np.asarray(offset_m, dtype=np.float64)
    return s.astype(np.asarray(states).dtype, copy=False)


def instruction_swap(
    episode_task: np.ndarray,
    allowed_tasks: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel a fraction of episodes with a different task from the same pool.

    episode_task: (N,) task index per episode.
    allowed_tasks: task indices to draw wrong labels from (e.g. the suite's).
    Returns (new_episode_task, swapped_mask). Each swapped episode gets a task
    that differs from its true one. Exactly round(fraction * N) episodes swap.
    """
    true = np.asarray(episode_task)
    pool = np.unique(np.asarray(allowed_tasks))
    if pool.size < 2:
        raise ValueError("need at least two tasks to swap labels")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    n = true.shape[0]
    n_swap = int(round(fraction * n))
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=n_swap, replace=False)] = True
    new = true.copy()
    for i in np.flatnonzero(mask):
        choices = pool[pool != true[i]]
        new[i] = rng.choice(choices)
    return new, mask
