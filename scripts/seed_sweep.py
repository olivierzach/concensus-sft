import argparse
import copy
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from consensus_sft.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--seeds", default="13,21,1337")
    parser.add_argument("--output_dir", default="outputs/consensus_seed_sweep")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["project"]["seed"] = seed
        config["project"]["output_dir"] = args.output_dir
        run_name = f"{config['project']['run_name']}_seed{seed}"
        config["project"]["run_name"] = run_name

        # Write a temporary config file per run.
        temp_config = f"/tmp/seed_sweep_{seed}.yaml"
        with open(temp_config, "w", encoding="utf-8") as f:
            import yaml

            yaml.safe_dump(config, f, sort_keys=False)

        cmd = [
            sys.executable,
            "scripts/consensus/train_clean.py",
            "--config",
            temp_config,
        ]
        print("Running:", " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
