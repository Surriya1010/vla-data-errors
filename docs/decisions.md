# Decision log

## 2026-11-23

- **Simulation only.** No physical robot access during term. Simulation also
  gives the ground truth the study needs.
- **Franka Panda.** Standard research arm; used by both LIBERO and Isaac Lab.
- **SmolVLA as the trained model.** Hugging Face Hub check (2026-11-23):

  | Model | Parameters | Weights | LIBERO | Verdict |
  |---|---|---|---|---|
  | `lerobot/smolvla_libero` | 450M | 0.9 GB | ~87% avg (published) | **Main model**: fits fine-tuning in 8 GB |
  | `zuoxingdong/evo1_libero` | 776M | 3.1 GB | 96.65% overall (model card) | Stronger comparison if it fits; Qwen licence |
  | `lerobot/xvla-libero` | 880M | 3.5 GB | see paper | Possible comparison |
  | `lerobot/smolvla_libero_plus` | 450M | 0.9 GB | trained on perturbed LIBERO | Free evaluation-only comparison |
  | pi0.5 / GR00T N1.6 / MolmoAct2 / OpenVLA-OFT | 3.3B-7B | 7.5-15 GB | high | Too large for 8 GB |

- **Fine-tune from `smolvla_base`**, not from the LIBERO checkpoint, so every
  condition starts from the same model that has never seen clean LIBERO data.
- **Separate Python environment** (`.venv`, Python 3.12) so LeRobot never
  touches the Isaac Sim / Isaac Lab installation.
- **Course core in Isaac Lab**, research part in LIBERO (MuJoCo).
- **Parked:** Rhoban BAM actuator models (fits Microduck or SO-100, not Franka);
  real-robot sim-to-real; LLM "chat" planner.
