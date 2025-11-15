# TC-DDPG Framework (Minimal Reproducibility Archive)

This archive contains a **minimal, ready-to-run experiment** that reproduces the
main quantitative results reported in **Table 3** of the paper:

> "Physics-Informed Reinforcement Learning for Thermodynamically-Constrained
> HVAC Control" (submitted to Energies).

The provided code is **research-grade** and intended to document the framework
and experimental pipeline, not as a production-ready BMS implementation.

---

## 1. Software Environment

The minimal experiment has been tested with the following stack:

- Python 3.10
- PyTorch 2.1.0
- NumPy 1.26.4
- SciPy 1.11.x
- Matplotlib 3.8.x
- Gymnasium (or OpenAI Gym) 0.28.x (for environment API)
- PyYAML 6.0.x

You can recreate the environment using either `conda` or `pip`.

### Conda (recommended)

```bash
conda create -n tcddpg python=3.10
conda activate tcddpg
pip install -r requirements.txt
