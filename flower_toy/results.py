"""In-memory experiment results shared by analysis, reporting, and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import ModelHistory


@dataclass
class Baseline:
    green: np.ndarray
    ancestors: dict[str, np.ndarray]
    histories: tuple[ModelHistory, ...]


@dataclass
class ExperimentResult:
    summary: dict = field(default_factory=dict)
    tables: dict[str, list[dict]] = field(default_factory=dict)  # Keys become CSV basenames.


@dataclass
class SuiteResult:
    baseline: Baseline
    summary: dict = field(default_factory=dict)
    tables: dict[str, list[dict]] = field(default_factory=dict)  # Keys become CSV basenames.
