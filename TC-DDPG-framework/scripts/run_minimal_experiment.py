# scripts/run_minimal_experiment.py

import argparse
import yaml
from pathlib import Path

from tcddpg.utils_seeds import set_global_seed
from tcddpg.env_rc_model import make_rc_env
from tcddpg.agent_tcddpg import train_ddpg_baseline, train_tcddpg

def main(config_path: str):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Set global seed (NumPy, Python, PyTorch, Gym)
    seed = cfg.get("seed", 42)
    set_global_seed(seed)

    # Build environment
    env_cfg = cfg["env"]
    env = make_rc_env(env_cfg)

    # Train baseline DDPG
    print("\n=== Training Baseline DDPG ===")
    res_base = train_ddpg_baseline(env, cfg["training"])

    # Reset environment for TC-DDPG
    set_global_seed(seed)
    env = make_rc_env(env_cfg)

    # Train TC-DDPG (ours)
    print("\n=== Training TC-DDPG (with thermodynamic & physics layers) ===")
    res_tc = train_tcddpg(env, cfg["training"], cfg["physics"], cfg["tcddpg"])

    # Output directory
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save simple CSV comparing both controllers
    out_csv = out_dir / "minimal_table3.csv"
    with out_csv.open("w") as f:
        f.write("method,energy_kwh,comfort_drift,violations_per_day\n")
        f.write(f"DDPG,{res_base['energy']:.3f},{res_base['comfort']:.3f},{res_base['violations']:.3f}\n")
        f.write(f"TC-DDPG,{res_tc['energy']:.3f},{res_tc['comfort']:.3f},{res_tc['violations']:.3f}\n")

    print(f"\nSaved summary table → {out_csv}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to YAML configuration file.")
    args = parser.parse_args()
    main(args.config)
