#!/usr/bin/env python3
"""Deprecated compatibility wrapper for convert_kugelaudio_to_gguf.py."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).with_name("convert_kugelaudio_to_gguf.py")
    print(
        "warning: scripts/convert_vibevoice_to_gguf.py is deprecated; "
        "use scripts/convert_kugelaudio_to_gguf.py instead",
        file=sys.stderr,
    )
    runpy.run_path(str(target), run_name="__main__")
