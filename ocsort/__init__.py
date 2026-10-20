"""Vendored OC-SORT core (MIT); see ocsort/LICENSE.

Only the tracker (Kalman + observation-centric association) is included; the
upstream detector/training stack is intentionally not vendored.
"""
from .ocsort import OCSort

__all__ = ["OCSort"]
