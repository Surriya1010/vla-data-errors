# Experiment design

## Research questions

- **RQ1.** For each realistic error type in the *training* data, how does task
  success of a fine-tuned SmolVLA change as the error strength increases?
- **RQ2.** Under the same training budget, which error types cost the most?
- **RQ3.** Can simple, label-free checks detect each error type, and how well,
  measured against the known injected errors?

## Data

- Dataset: [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero),
  LeRobot format v3.0. 1,693 episodes, 273,465 frames, 40 tasks, two 256x256
  cameras, 8-D state, 7-D action.
- **Pin the dataset revision** (`--dataset.revision=<commit-sha>`) and record it
  in `results/`. Hub datasets can be re-uploaded.
- State: end-effector position (3), axis-angle orientation (3), gripper (2).
- Action: 6-D end-effector delta + 1-D gripper, in [-1, 1].

**Scope reduction for an 8 GB laptop:** train and evaluate on **one suite**
(`libero_object` first: 10 tasks, the suite where published models score
highest, so a clean policy sits well above the floor). The episode indices
belonging to that suite are selected with `--dataset.episodes`; the helper in
`scripts/02_make_corrupted.py --list-suite-episodes` prints them.

## Error types (minimum version: the first three)

| ID | Error | What it simulates | How it is injected | Strengths |
|---|---|---|---|---|
| `action_lag_k` | **Action/observation time offset** | Camera and joint/command streams captured with a timing offset; logging latency | Within each episode, row *t* gets the action from row *t-k* (start padded with the first action). Videos and state are untouched | k = 1, 2, 4 frames |
| `instruction_swap_p` | **Wrong task label** | Episodes saved under the wrong task name | A fraction *p* of episodes get the task index of a different task from the same suite | p = 0.1, 0.25, 0.5 |
| `action_jitter_s` | **Operator noise** | Shaky teleoperation | Zero-mean Gaussian noise on the 6 arm action dims, std = s x per-dim std of the clean actions; clipped to [-1, 1]; gripper untouched | s = 0.1, 0.25, 0.5 |
| `state_bias_b` *(optional)* | Calibration offset | Miscalibrated end-effector position | Constant offset of b metres on the state position dims, same for the whole dataset | b = 0.005, 0.02 |

Why these: timing offset is named as a main real-data problem but rarely
measured on training data (see related work). Wrong labels are the language
counterpart. Jitter is the reference point, because prior work (Belkhale et al.
2023) already showed action noise hurts; reproducing that direction is a check
that the pipeline is sensitive at all. Calibration is optional because a 2026
paper already measures calibration sensitivity on LIBERO.

All injection code is in `src/vla_errors/corruptions.py` and is deterministic
given a seed.

## Training recipe (identical for every condition)

- Start from `lerobot/smolvla_base` (not from the LIBERO checkpoint, which has
  already seen clean data).
- Same suite, same episodes, same number of steps, same batch size, same seed.
- Budget is set by step 3 of the protocol below; record it in
  `configs/conditions.yaml` before running any corrupted condition and do not
  change it afterwards.
- Three training seeds for the clean control and for the strongest strength of
  each error type; one seed for intermediate strengths if time is short (state
  this in the write-up).

## Evaluation

- `lerobot-eval`, LIBERO, the training suite, `--env.init_states=true`, hard
  resets, fixed `--seed`, the same `--policy.n_action_steps` for every
  condition (record it: the execution horizon changes SmolVLA's LIBERO score a
  lot, see related work).
- At least 50 episodes per task where time allows (500 per suite). Report the
  95% Wilson interval `lerobot-eval` prints.

## Protocol

1. **Pipeline check.** Evaluate `lerobot/smolvla_libero` on the suite. Compare
   with its published level. A large shortfall means the evaluation setup is
   wrong; fix it first.
2. **Memory check.** Find the largest batch size that fine-tunes SmolVLA within
   8 GB (gradient accumulation if needed). Record it.
3. **Clean control.** Fine-tune on clean data with a budget large enough that
   success is clearly above the floor. If it stays near 0%, the study cannot
   detect damage (this is exactly how a similar study ended up
   inconclusive); increase the budget or switch to an easier suite before
   continuing.
4. **Corrupted conditions**, strongest strength first. If the strongest
   strength shows no effect, the weaker ones are skipped for that error type.
5. **Detectors** on every dataset copy (`scripts/05_run_detectors.py`).

## Detectors (RQ3)

| Detector | Targets | Signal |
|---|---|---|
| `estimate_lag` | action_lag | Lag that maximises the correlation between commanded arm translation and the next-step change in end-effector position |
| `jitter_score` | action_jitter | Share of action energy at high frequency |
| `stall_runs` | operator hesitation (reported, not injected) | Longest run of near-zero arm action |
| *(none)* | instruction_swap | Not detectable from motion alone; listed as a known gap |

Metrics: per-episode precision, recall, F1 against the known injected episodes,
and for `estimate_lag`, the absolute error of the estimated lag.

## What counts as a result

- Each error type gets a dose-response plot: success (with interval) vs strength.
- "Error X matters more than error Y" is claimed only when their intervals do
  not overlap at the strengths compared.
- The write-up states the smallest difference the experiment could detect.
