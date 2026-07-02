#!/usr/bin/env python3
"""Compatibility wrapper for the multimodal taxon training script.

The implementation currently lives under ``classifyModel.deprecated.scripts``
after the repository reorganization. Keeping this wrapper lets existing
commands such as

    python -m classifyModel.scripts.train_multimodal_taxa

continue to work, including Windows multiprocessing DataLoader workers that
need to re-import the launched module.
"""

from __future__ import annotations

from multiprocessing import freeze_support

from classifyModel.deprecated.scripts.train_multimodal_taxa import main


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
