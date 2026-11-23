# Which data errors break a robot policy?

*A controlled study of realistic training-data errors on a vision-language-action
(VLA) policy for a Franka Panda arm, in simulation.*

> **Status: setup.** No experiment has been run yet. Every number in this repo
> that is not under `results/` is a plan or a value reported by someone else,
> with its source.
>
> | Part | State |
> |---|---|
> | Course core (Isaac Lab Franka controllers) | Runs on the laptop (checked 2026-11-23) |
> | `src/vla_errors` corruptions and detectors | 13 unit tests pass on synthetic episodes |
> | `scripts/02` (dataset corruption) | Passes on a small fake dataset in LeRobot v3 layout (all four condition types); **not yet run on the real LIBERO data**. It checks the format and stops if it differs |
> | `scripts/05` (detectors on real data) | Written, not yet run |
> | LeRobot environment, training, evaluation | Not installed yet (`setup/install.sh`) |

## The question

Real robot data is always slightly wrong: camera frames and joint readings are
captured at slightly different moments, episodes get the wrong task label,
operators move hesitantly. In real data the true values are unknown, so nobody
can say how wrong a dataset is or what that costs.

In simulation the truth is known. This project takes clean Franka
demonstrations, adds **one realistic error type at a time** at controlled
strengths, fine-tunes the same VLA on each version, and measures:

1. **How much each error type damages task success** (a dose-response chart).
2. **Which errors matter and which don't**, under the same training budget.
3. **Whether cheap checks can detect each error** before training, tested
   against errors whose presence and size are known exactly.

See [docs/experiment-design.md](docs/experiment-design.md) for the full design
and [docs/related-work.md](docs/related-work.md) for what is and is not new.

## Setup at a glance

| | Choice |
|---|---|
| Robot | Franka Panda (simulated) |
| Benchmark | [LIBERO](https://libero-project.github.io), via [LeRobot](https://huggingface.co/docs/lerobot/en/libero) |
| Training data | [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) (1,693 episodes, 40 tasks, 1.9 GB) |
| Policy | [SmolVLA](https://huggingface.co/lerobot/smolvla_base) (450M parameters) |
| Course part | Franka kinematics and control in NVIDIA Isaac Lab ([course/](course/)) |
| Hardware | One laptop GPU, RTX 5060 8 GB |

SmolVLA was chosen because it is the only VLA with LIBERO checkpoints that can
also be *fine-tuned* within 8 GB. Larger models are listed in
[docs/decisions.md](docs/decisions.md).

## Repository layout

```
README.md                 this file
docs/
  experiment-design.md    conditions, protocol, statistics, success criteria
  related-work.md         prior work and the narrowed claim
  decisions.md            dated log of choices and why
  proposal-email.md       course proposal draft (not sent)
course/                   Manipulator-course core: Isaac Lab Franka control
setup/install.sh          creates .venv with LeRobot + LIBERO + SmolVLA
configs/conditions.yaml   every experimental condition, in one place
scripts/
  00_check_env.sh         GPU, CUDA, LeRobot, LIBERO import checks
  01_eval_official.sh     evaluate the official SmolVLA LIBERO checkpoint
  02_make_corrupted.py    write a corrupted copy of the LIBERO dataset
  03_finetune.sh          fine-tune SmolVLA on one dataset condition
  04_eval_checkpoint.sh   evaluate a fine-tuned checkpoint
  05_run_detectors.py     run the error detectors on a dataset copy
src/vla_errors/
  corruptions.py          error injection (pure numpy, unit-tested)
  detectors.py            error detection (pure numpy, unit-tested)
tests/                    unit tests for the two modules above
results/                  outputs; see results/README.md
```

## Quick start

All commands run from the repository root.

```bash
# 0. Unit tests for the corruption and detector code (no GPU needed)
python3 -m pytest tests -q

# 1. Create the environment (downloads PyTorch, LeRobot, LIBERO: several GB)
bash setup/install.sh
source .venv/bin/activate

# 2. Check GPU, CUDA and imports
bash scripts/00_check_env.sh

# 3. Sanity check: does the official checkpoint reproduce on this laptop?
bash scripts/01_eval_official.sh libero_object 10

# 4. Make a corrupted dataset copy, e.g. actions delayed by 2 frames
python scripts/02_make_corrupted.py --condition action_lag_2

# 5. Fine-tune, then evaluate (see docs/experiment-design.md for the protocol)
bash scripts/03_finetune.sh action_lag_2
bash scripts/04_eval_checkpoint.sh action_lag_2
```

## Order of work

1. **Course core** in Isaac Lab ([course/README.md](course/README.md)). Runs today.
2. **Pipeline check:** step 3 above. If the official checkpoint scores far below
   its published level on this laptop, stop and fix the evaluation setup before
   anything else.
3. **Clean control:** fine-tune SmolVLA on clean data with the exact recipe used
   for every corrupted condition. This, not the official checkpoint, is the
   baseline every condition is compared against.
4. **Corrupted conditions**, one error type at a time.
5. **Detectors** on every dataset copy.
6. Write-up: one chart per question, with confidence intervals.

## Honesty rules for this project

- No result is reported without the command, seed, dataset revision and number
  of episodes that produced it.
- Success rates come with the 95% interval `lerobot-eval` reports. Overlapping
  intervals mean "no difference shown", not "no difference".
- If an error type shows no effect, that is reported as a result, with the
  smallest effect the experiment could have detected.
