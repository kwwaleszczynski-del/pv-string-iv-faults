"""Command-line entry point for pv-string-iv-faults."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_full_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pv-iv-faults",
        description="PV string fault diagnosis from 15-minute inverter MPPT voltage and current alone.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "verify", "plots", "dashboard"],
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional root directory for results/ and docs/ outputs.",
    )
    args = parser.parse_args(argv)
    run_full_pipeline(output_root=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
