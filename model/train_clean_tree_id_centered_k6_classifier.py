#!/usr/bin/env python3
"""Compatibility entry point for the renamed fixed-partition trainer.

New workflows should call ``model/stage2/train_fixed_partition_classifier.py``.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "stage2" / "train_fixed_partition_classifier.py"),
        run_name="__main__",
    )
