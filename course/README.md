# Course core: Franka Panda modelling and control (Isaac Lab)

This part stands on its own for the Manipulator course and does not depend on
the VLA experiment.

## Environment

- Isaac Sim 6.0.1 (pre-release) at `~/isaacsim`, Isaac Lab 3.0 at `~/IsaacLab`.
- **Asset fix (done 2026-11-23):** the 6.0 asset folder does not exist on
  NVIDIA's server yet, so robots failed to load. All `~/IsaacLab/apps/*.kit`
  files now point to the 5.1 assets; the originals are kept as
  `*.bak-6.0-assets`. Revert once NVIDIA publishes 6.0 assets:
  ```bash
  cd ~/IsaacLab/apps && for f in *.bak-6.0-assets; do mv "$f" "${f%.bak-6.0-assets}"; done
  ```
- Verified 2026-11-23: `run_diff_ik.py --robot franka_panda --num_envs 4
  --headless` reaches "Setup complete" with ~1.8 GB GPU memory.

## Run the controllers

```bash
cd ~/IsaacLab
# Differential inverse kinematics (with a window: drop --headless)
./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py --robot franka_panda --num_envs 4
# Operational-space control
./isaaclab.sh -p scripts/tutorials/05_controllers/run_osc.py --num_envs 4
```

Both loop until the window is closed (or Ctrl+C when headless).

## Work plan

| # | Task | Output |
|---|---|---|
| 1 | Franka DH (or URDF-derived) parameters; forward kinematics | Table + code; FK error vs simulator end-effector pose over random joint samples |
| 2 | Geometric Jacobian | Code; check against finite differences of FK |
| 3 | Singularities and manipulability | Manipulability index along test trajectories; behaviour near singular poses |
| 4 | Differential IK (damped least squares) | Tracking error vs damping factor; behaviour near joint limits |
| 5 | Operational-space control | Tracking error and settling time vs differential IK on the same targets |
| 6 | Trajectory generation | Joint-space vs Cartesian, quintic vs trapezoidal; velocity/acceleration profiles |

Write the analysis scripts in `course/` (for example `course/fk_check.py`),
run them with `~/IsaacLab/isaaclab.sh -p <script>`, and save plots to
`results/course/`.
