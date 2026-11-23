# Related work and the narrowed claim

Checked 2026-11-23 by web search. Not a systematic review; extend before
writing a report. Several sources were seen only as search summaries and must
be read in full before being cited.

| Work | What it did | Consequence for this project |
|---|---|---|
| [Belkhale et al., *Data Quality in Imitation Learning*, NeurIPS 2023](https://arxiv.org/abs/2306.02437) | Controlled noise study: action (policy) noise hurts far more than system/state noise, worst with small datasets | Cite. This project does **not** claim to be the first controlled study of demonstration noise. It uses different error types (timing, labels) and a VLA |
| [Liu et al., *Robust Imitation Learning from Corrupted Demonstrations*](https://arxiv.org/abs/2201.12594) | A fraction of samples replaced by outliers; plain behaviour cloning fails | Background |
| [DMDR, *Restoring Noisy Demonstration with Diffusion Models*](https://arxiv.org/abs/2510.14467) | Gaussian/Laplacian action noise; diffusion policies more sensitive than BC | Background; synthetic noise only |
| [*Hidden Variables Behind Real-Robot Data Quality*, arXiv 2611.22388](https://arxiv.org/abs/2611.22388) | Names timestamp misalignment and calibration as main sources of inconsistent real data; experiments mostly on joint zero-offset miscalibration | Closest motivation: this project measures the timing case directly |
| [*Towards Pose-Robust Few-Shot VLA Models*, arXiv 2611.24360](https://arxiv.org/abs/2611.24360) | Calibration-error sensitivity on LIBERO-Spatial | Calibration is therefore optional here, not the headline |
| [RAVEL, arXiv 2609.34170](https://arxiv.org/abs/2609.34170); [RESTA, arXiv 2610.18163](https://arxiv.org/abs/2610.18163) | Observation delay at *inference* time on LIBERO; offset augmentation | Different problem (deployment latency, not misaligned training data). Cite to make the distinction explicit |
| [DemoDoctor (blog)](https://dev.to/raihan-js/demodoctor-bad-robot-demonstrations-are-detectable-do-they-matter-548p) | Injected lag/stall/jitter/truncation/duplicates into PushT; stall detector F1 0.87, jitter 0.69, lag weak; 12 ACT policies showed no measurable effect, but clean success was ~1%, so the test was underpowered | Nearest prior attempt. Design lesson: clean baseline must be well above floor; report the detectable effect size |
| [PSD metric, arXiv 2605.01544](https://arxiv.org/abs/2605.01544); [Demo-SCORE, arXiv 2503.03707](https://arxiv.org/abs/2503.03707); [QoQ, arXiv 2603.09056](https://arxiv.org/abs/2603.09056); [CUPID, arXiv 2506.19121](https://arxiv.org/abs/2506.19121) | Score or filter demonstrations (spectral power, rollout classifier, influence functions) | Baselines and context for the detectors |
| [Scale Labs, robot data QC](https://labs.scale.com/blog/from-video-review-to-measurement) | Small "linter" checks; keep a human in the loop | Industry framing for the detectors |
| [arXiv 2610.31859](https://arxiv.org/abs/2610.31859) | Official SmolVLA LIBERO checkpoint: 67.6% with 50-action chunks, 93.4% replanning every 10; small differences between GPUs | Fix and report the execution horizon |

## Narrowed claim

A controlled study, on LIBERO with SmolVLA, of how **training-data** timing
offsets, wrong task labels and operator-style noise affect a fine-tuned VLA at
several strengths, with a clean baseline well above floor, plus detectors
validated against known injected errors.

Not claimed: a general law about robot data; the first noise study; real-robot
validity.
