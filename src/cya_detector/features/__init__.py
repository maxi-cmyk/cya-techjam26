"""Deterministic auxiliary feature extractors (Tasks 7-9).

Every extractor in this package is a fixed algorithm, not a learned model:
given the same image and parameters it must return the same output. None of
them may independently return an authenticity verdict; each returns a
feature vector plus a validity/confidence flag for downstream fusion.
"""

from __future__ import annotations
