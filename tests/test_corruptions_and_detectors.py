"""Unit tests on synthetic episodes (no LeRobot, no GPU)."""

import numpy as np
import pytest

from vla_errors import (
    action_jitter,
    action_lag,
    estimate_lag,
    instruction_swap,
    jitter_score,
    stall_runs,
    state_bias,
)


def synthetic_episode(T=200, seed=0):
    """Smooth actions; state integrates the action one step later (aligned)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    phases = rng.uniform(0, 2 * np.pi, size=(1, 7))
    freqs = rng.uniform(0.01, 0.04, size=(1, 7))
    actions = 0.5 * np.sin(2 * np.pi * freqs * t + phases)
    states = np.zeros((T, 8))
    for i in range(T - 1):
        states[i + 1, :3] = states[i, :3] + 0.01 * actions[i, :3]
    return actions.astype(np.float32), states.astype(np.float32)


# ---- corruptions -------------------------------------------------------------

def test_action_lag_shifts_and_pads():
    a = np.arange(10, dtype=float)[:, None].repeat(2, axis=1)
    out = action_lag(a, 3)
    assert out.shape == a.shape
    assert np.all(out[:3, 0] == 0)             # padded with first action
    assert np.all(out[3:, 0] == a[:-3, 0])      # row t holds action t-3
    lead = action_lag(a, -2)
    assert np.all(lead[:-2, 0] == a[2:, 0])
    assert np.all(lead[-2:, 0] == a[-1, 0])


def test_action_lag_zero_is_copy_and_input_untouched():
    a, _ = synthetic_episode()
    before = a.copy()
    out = action_lag(a, 0)
    assert np.array_equal(out, a) and out is not a
    action_lag(a, 4)
    assert np.array_equal(a, before)


def test_action_jitter_scale_clip_and_gripper():
    a, _ = synthetic_episode()
    ref = a.std(axis=0)
    rng = np.random.default_rng(1)
    out = action_jitter(a, 0.5, ref, rng)
    assert np.all(out <= 1.0) and np.all(out >= -1.0)
    assert np.array_equal(out[:, 6], a[:, 6])   # gripper untouched
    resid = (out - a)[:, :6].std(axis=0)
    assert np.allclose(resid, 0.5 * ref[:6], rtol=0.25)
    assert np.array_equal(action_jitter(a, 0.0, ref, rng), a)


def test_state_bias_only_position():
    _, s = synthetic_episode()
    out = state_bias(s, 0.02)
    assert np.allclose(out[:, :3] - s[:, :3], 0.02, atol=1e-6)
    assert np.array_equal(out[:, 3:], s[:, 3:])


def test_instruction_swap_count_and_always_wrong():
    rng = np.random.default_rng(2)
    tasks = np.repeat(np.arange(10), 50)
    new, mask = instruction_swap(tasks, np.arange(10), 0.25, rng)
    assert mask.sum() == round(0.25 * tasks.size)
    assert np.all(new[mask] != tasks[mask])
    assert np.array_equal(new[~mask], tasks[~mask])
    assert set(np.unique(new)) <= set(range(10))


def test_instruction_swap_rejects_bad_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        instruction_swap(np.zeros(5, int), np.array([0]), 0.5, rng)
    with pytest.raises(ValueError):
        instruction_swap(np.zeros(5, int), np.arange(3), 1.5, rng)


# ---- detectors ---------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 1, 2, 4, -2])
def test_estimate_lag_recovers_injected_delay(k):
    a, s = synthetic_episode(seed=3)
    assert estimate_lag(action_lag(a, k), s) == k


def test_jitter_score_rises_with_strength():
    a, _ = synthetic_episode(seed=4)
    ref = a.std(axis=0)
    scores = [
        jitter_score(action_jitter(a, st, ref, np.random.default_rng(5)))
        for st in (0.0, 0.1, 0.25, 0.5)
    ]
    assert scores == sorted(scores)
    assert scores[0] < 0.05 < scores[-1]


def test_stall_runs_finds_inserted_pause():
    a, _ = synthetic_episode(seed=6)
    assert stall_runs(a) < 5
    a2 = a.copy()
    a2[50:80, :6] = 0.0
    assert stall_runs(a2) >= 30
