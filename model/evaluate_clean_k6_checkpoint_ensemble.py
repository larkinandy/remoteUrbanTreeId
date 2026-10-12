#!/usr/bin/env python3
"""Compatibility entry point for the renamed checkpoint evaluator.

New workflows should call ``model/stage2/evaluate_checkpoint_ensemble.py``.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "stage2" / "evaluate_checkpoint_ensemble.py"),
        run_name="__main__",
    )
