"""Top-level MOVE package conveniences.

This package wraps the canonical implementation located in ``MOVE`` so users can
simply run ``python -m move`` from the repository root or import
``move.main``/``move.MOVE`` in their own scripts.
"""

from MOVE.move import MOVE, main

__all__ = ["MOVE", "main"]
