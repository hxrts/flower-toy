"""Shared scientific defaults and independent figure measurement settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSettings:
    green_steps: int = 10
    green_lr: float = .15
    specialization_steps: int = 190
    specialization_lr: float = .18
    reader_steps: int = 800
    reader_lr: float = .15
    competition_steps: int = 160
    competition_lr: float = .15
    acquisition_lr: float = .07
    acquisition_budget: int = 300
    matched_budget: int = 600
    # Planned sweep cells are reported even when acquisition fails.
    sensitivity_green_steps: tuple[int, ...] = (0, 5, 10, 20)
    sensitivity_lrs: tuple[float, ...] = (.04, .07, .10)
    sensitivity_targets: tuple[float, ...] = (.90, .95, .975)
    # The article selects two targets from the wider sensitivity sweep.
    comparison_targets: tuple[float, ...] = (.95, .975)
    target_score: float = .95
    strict_target_score: float = .975
    capacity: float = 1.30
    decision_threshold: float = .5
    max_reader_error: float = .05
    handoff_phase_steps: int = 40
    handoff_lr: float = .2
    handoff_threshold: float = .45
    ancestry_temperature: float = .25
    acquisition_temperature: float = 7.


@dataclass(frozen=True)
class BandSettings:
    # Intervals are half-open update ranges; scales are source TikZ points.
    competition_intervals: tuple[tuple[int, int], ...] = ((0, 10), (10, 30))
    contribution_checkpoint: int = 0
    reader_checkpoint: int = 10
    competition_scale: float = 80.
    feature_scale: float = 240.
    connection_scale: float = 48.
    specialization_scale: float = 200.
    reader_scale: float = 200.


TRAINING = TrainingSettings()
BANDS = BandSettings()
