"""
Run bench_model.py across a grid of configs via subprocess.

Usage:
    python bench_sweep.py              # single GPU
    python bench_sweep.py --gpus 2     # distributed
"""

import subprocess
import sys
import argparse

CONFIGS = [
    # (hidden, heads, batch, seq)
    (512, 8, 8, 512),
    (512, 8, 32, 512),
    (512, 8, 8, 2048),
    (512, 8, 32, 2048),
    (1024, 8, 8, 4096),
    (1024, 8, 32, 4096),
]


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}")
        if result.stderr:
            # Print last 5 lines of stderr for debugging.
            for line in result.stderr.strip().split("\n")[-5:]:
                print(f"  {line}")
    else:
        print(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="GPUs for distributed (0 = single-GPU only)")
    args = parser.parse_args()

    distributed = args.gpus > 1

    mode = f"Distributed ({args.gpus} GPUs)" if distributed else "Single GPU"
    print(f"{'=' * 80}\n  {mode}\n{'=' * 80}")

    for h, nh, b, s in CONFIGS:
        for model_type in ["baseline", "efficient"]:
            if distributed:
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={args.gpus}",
                    "bench_model.py",
                ]
            else:
                cmd = [sys.executable, "bench_model.py"]

            cmd += [
                "--hidden",
                str(h),
                "--heads",
                str(nh),
                "--batch",
                str(b),
                "--seq",
                str(s),
                "--type",
                model_type,
            ]
            run(cmd)
        print()


if __name__ == "__main__":
    main()
