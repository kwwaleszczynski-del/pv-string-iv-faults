"""Command-line entry point for pv-string-iv-faults.

Usage:
    python -m pv_iv_faults                 # full run: tuning, held-out evaluation, showcase, figures, docs/data.js
    python -m pv_iv_faults --workers 1     # same, without a process pool
    python -m pv_iv_faults --out /tmp/run  # write results/ and docs/ under another root
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import run_full_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pv-iv-faults",
        description="PV string fault diagnosis from 15-minute inverter MPPT voltage and current, evaluated on held-out synthetic inverters.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Root directory for results/ and docs/ (default: repository root).")
    parser.add_argument("--workers", type=int, default=None, help="Parallel processes for the synthetic inverters (default: CPU count).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = run_full_pipeline(output_root=args.out, workers=args.workers)
    ver = out["verification"]
    logging.info("held-out macro-F1 %.3f (tuning, in-sample %.3f)", ver["heldout"]["macro_f1"], ver["tuning"]["macro_f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
