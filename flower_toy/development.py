"""Controlled comparisons of reader-conditioned inheritance and continuity.

Role targets, ancestry, update schedules and reader routing are declared.
They are interventions, not inferred outcomes of unstructured training.
"""
from __future__ import annotations

import numpy as np
from .model import (FLOWER, TEXTURE, ALL_ROLES, GOOD_READERS, train_green,
                    create_history, history_arrays, response_assay,
                    attributed_price, reader_outputs)
from .results import ExperimentResult
from .settings import TRAINING


def matched_inheritance(ancestors: dict[str, np.ndarray] | None = None) -> tuple[list[dict], float]:
    if ancestors is None:
        ancestors = dict(A=train_green(FLOWER), B=TEXTURE.copy())
    # Change ancestry while holding learned role targets and final reader wiring fixed.
    histories = [create_history(name, ancestors, parents) for name, parents in (
        ('Retention', dict(F1='A', F2='A', G='A', T='B')),
        ('Mixed inheritance', dict(F1='A', F2='B', G='A', T='A')))]
    rows, outputs = [], []
    for history in histories:
        weights, credit = history_arrays(history)
        outputs.append(reader_outputs(weights, GOOD_READERS))
        for group, roles in (('all', ALL_ROLES), ('flower', GOOD_READERS.flower),
                             ('broccoli', GOOD_READERS.broccoli)):
            indices = [ALL_ROLES.index(role) for role in roles]
            rows.append(dict(history=history.name, population=group,
                             **attributed_price(np.array([1., 0.]),
                                 response_assay(weights, True)[indices], credit[indices])))
    return rows, float(np.max(np.abs(outputs[0] - outputs[1])))


def handoff(schedule, phase_steps=TRAINING.handoff_phase_steps, lr=TRAINING.handoff_lr):
    """Two flower channels, fixed mean reader, threshold .45, equal update budgets.

    Each update descends half-squared error to a prescribed local target.
    Acquire B before retiring A, or reverse that order. All checkpoint
    inputs are tested, including the initial state and the phase boundary.
    """
    if schedule not in ('continuous', 'interrupted'):
        raise ValueError('Unknown handoff schedule')
    weights = np.array([1., 0.])
    rows = []
    for step in range(2 * phase_steps + 1):
        score = float(weights.mean())
        predictions = np.array([0., score]) >= TRAINING.handoff_threshold
        rows.append(dict(schedule=schedule, step=step, a_flower=float(weights[0]),
                         b_flower=float(weights[1]), flower_score=score,
                         negative_score=0.,
                         positive_margin=score-TRAINING.handoff_threshold,
                         negative_margin=TRAINING.handoff_threshold,
                         flower_accuracy=float(np.mean(predictions == [False, True]))))
        if step < 2 * phase_steps:
            # Reverse the phase order while preserving each channel's update budget.
            acquiring = (step < phase_steps) == (schedule == 'continuous')
            coordinate, target = (1, 1.) if acquiring else (0, 0.)
            weights[coordinate] -= lr * (weights[coordinate] - target)
    # One-to-one ancestry removes selection, isolating the history of availability.
    endpoint = attributed_price(np.array([1., 0.]), (weights >= TRAINING.decision_threshold).astype(float), np.eye(2))
    return rows, weights, endpoint


def handoff_thresholds(traces):
    """Evaluate fixed thresholds across every recorded update, without retraining."""
    rows = []
    for schedule in ('continuous', 'interrupted'):
        trace = [r for r in traces if r['schedule'] == schedule]
        for threshold in (.001, .01, .1, .25, .45, .5):
            unavailable = [r['step'] for r in trace
                           if not r['negative_score'] < threshold <= r['flower_score']]
            rows.append(dict(schedule=schedule, threshold=threshold,
                             unavailable_updates=len(unavailable),
                             first_unavailable=unavailable[0] if unavailable else '',
                             last_unavailable=unavailable[-1] if unavailable else '',
                             minimum_positive_margin=min(r['flower_score'] for r in trace)-threshold,
                             minimum_negative_margin=threshold-max(r['negative_score'] for r in trace)))
    return rows


def development_experiment(ancestors: dict[str, np.ndarray]) -> ExperimentResult:
    matched, difference = matched_inheritance(ancestors)
    traces, endpoints, accounting = [], [], []
    for schedule in ('continuous', 'interrupted'):
        rows, weights, price = handoff(schedule)
        traces.extend(rows)
        endpoints.append(weights)
        accounting.append(dict(schedule=schedule, **price))
    result = dict(matched_reader_max_difference=difference,
                  handoff_endpoint_max_difference=float(np.max(np.abs(endpoints[0]-endpoints[1]))),
                  minimum_accuracy={s:min(r['flower_accuracy'] for r in traces if r['schedule']==s)
                                    for s in ('continuous','interrupted')},
                  scope='Declared local targets and schedules; fixed mean flower reader. '
                        'Identical full endpoints imply identical future deterministic updates under matched conditions.')
    return ExperimentResult(summary={"development_comparisons": result}, tables={
        "matched_reader_inheritance": matched, "handoff_trajectories": traces,
        "handoff_price": accounting, "handoff_thresholds": handoff_thresholds(traces)})
