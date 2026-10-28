"""Allow `python -m pv_iv_faults` execution."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
