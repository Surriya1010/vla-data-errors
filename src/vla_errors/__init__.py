"""Controlled training-data errors for robot policies, and detectors for them."""

from .corruptions import action_jitter, action_lag, instruction_swap, state_bias
from .detectors import estimate_lag, jitter_score, stall_runs

__all__ = [
    "action_jitter",
    "action_lag",
    "instruction_swap",
    "state_bias",
    "estimate_lag",
    "jitter_score",
    "stall_runs",
]
