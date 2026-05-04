from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all advanced retrieval experiment scripts.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", default="strict", choices=["strict", "balanced", "creative"])
    parser.add_argument("--print-top", type=int, default=10)
    parser.add_argument("--multiquery-count", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    scripts = [
        ("baseline", base_dir / "01_baseline.py", []),
        ("multiquery", base_dir / "02_multiquery.py", ["--multiquery-count", str(args.multiquery_count)]),
        ("hyde", base_dir / "03_hyde.py", []),
        ("selfcheck", base_dir / "04_selfcheck.py", []),
    ]
    for name, script, extra_args in scripts:
        print(f"\n===== {name} =====")
        command = [
            sys.executable,
            str(script),
            "--query",
            args.query,
            "--mode",
            args.mode,
            "--print-top",
            str(args.print_top),
            *extra_args,
        ]
        subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
