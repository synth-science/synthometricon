#!/usr/bin/env python3
"""Backward-compatible shim — delegates to ``python -m extraction``."""
from extraction.__main__ import main

if __name__ == "__main__":
    main()
