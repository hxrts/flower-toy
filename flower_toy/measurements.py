"""Pure trajectory, ancestry, and gradient-band measurements."""

from __future__ import annotations

import numpy as np

from .model import (
    ALL_ROLES,
    READER_ASSIGNMENTS,
    FEATURES,
    GOOD_READERS,
    ModelHistory,
    ROLE_TARGETS,
    price,
)

from .settings import BANDS, TRAINING


def history_tables(histories: tuple[ModelHistory, ...]) -> dict[str, list[dict]]:
    trajectories, accounting = [], []
    for history in histories:
        for role in ALL_ROLES:
            use = history.uses[role]
            consumer = "flower" if role in GOOD_READERS.flower else "broccoli"
            for step, weights in enumerate(use.trajectory):
                trajectories.append(dict(zip(
                    ("history", "gradient_step", "use", "parent", "consumer_role",
                     "flower_response", "green_response", "texture_response"),
                    (history.name, step, role, use.parent, consumer, *map(float, weights)))))
        # These are fixed endpoint assignments, before any reader-driven learning.
        for assignment, mapping in READER_ASSIGNMENTS:
            for population, roles in (("all", ALL_ROLES), ("flower_reader", mapping.flower),
                                      ("broccoli_reader", mapping.broccoli)):
                measured = price(history, roles)
                accounting.append(dict(zip(
                    ("history", "reader_assignment", "population", "baseline", "selection",
                     "transmission", "final", "delta", "A_uses", "B_uses"),
                    (history.name, assignment, population, measured.baseline, measured.selection,
                     measured.transmission, measured.final, measured.total, *measured.parent_counts))))
    return {"learning_trajectories": trajectories, "price_decompositions": accounting}


def gradient_measurements(competition: list[dict],
                          histories: tuple[ModelHistory, ...]) -> list[dict]:
    """Measure signed loss derivatives and actual GD increments in fixed coordinates.

    Pressure is -dL/dparameter, before multiplying by the learning rate.
    Reader contributions include the 1/n task averaging in the actual loss.
    Terminal checkpoints have measured gradients but no applied update.
    """
    rows = []
    def record(experiment, condition, step, source, parameter, gradient, lr, applied):
        rows.append(dict(experiment=experiment, condition=condition, step=step,
                         source=source, parameter=parameter, loss_gradient=float(gradient),
                         descent_pressure=float(-gradient), learning_rate=lr,
                         update_applied=applied,
                         gradient_update=float(-lr * gradient) if applied else 0.))
    for row in competition:
        condition = ("warm" if row["head_start"] else "cold") + (
            "_conflicting" if row["conflicting"] else "_compatible")
        for i in range(2):
            record("competition", condition, row["step"], "all_readers", f"feature_{i}",
                   row[f"feature_gradient_{i}"], TRAINING.competition_lr, row["update_applied"])
            for r in range(2):
                record("competition", condition, row["step"], f"reader_{r}", f"feature_{i}",
                       row[f"reader_{r}_feature_gradient_{i}"], TRAINING.competition_lr, row["update_applied"])
                record("competition", condition, row["step"], f"reader_{r}", f"read_weight_{r}_{i}",
                       row[f"reader_weight_gradient_{r}_{i}"], TRAINING.competition_lr, row["update_applied"])
    for history in histories:
        for role in ALL_ROLES:
            trajectory = history.uses[role].trajectory
            for step, weights in enumerate(trajectory):
                for i, feature in enumerate(FEATURES):
                    record("specialization", history.name, step, role, feature,
                           weights[i] - ROLE_TARGETS[role][i], TRAINING.specialization_lr, step < len(trajectory) - 1)
    return rows


def gradient_band_summary(gradients: list[dict], *, experiment: str, condition: str,
                          source: str, parameter: str, start: int, stop: int,
                          scale: float) -> dict:
    """Maximum full width = scale * mean(abs(loss gradient)) over [start, stop).

    The signed mean pressure is reported separately; width is not a signed
    net gradient and does not hide cancellation across training steps.
    """
    chosen = [r for r in gradients if r["experiment"] == experiment and r["condition"] == condition
              and r["source"] == source and r["parameter"] == parameter
              and start <= r["step"] < stop and r["update_applied"]]
    if len(chosen) != stop - start or stop <= start:
        raise ValueError("Band interval must contain every applied training step")
    # Width keeps absolute pressure even when signed contributions would cancel.
    mean_abs = float(np.mean([abs(r["loss_gradient"]) for r in chosen]))
    return dict(experiment=experiment, condition=condition, source=source, parameter=parameter,
                start_step=start, stop_step_exclusive=stop,
                mean_signed_pressure=float(np.mean([r["descent_pressure"] for r in chosen])),
                mean_positive_pressure=float(np.mean([max(0., r["descent_pressure"]) for r in chosen])),
                mean_negative_pressure=float(np.mean([max(0., -r["descent_pressure"]) for r in chosen])),
                mean_absolute_gradient=mean_abs, points_per_gradient_unit=scale,
                full_width_pt=scale * mean_abs)


def band_taper_sign(band: dict) -> int:
    """Do not assign a single taper to an interval containing opposed pressures."""
    positive, negative = band["mean_positive_pressure"], band["mean_negative_pressure"]
    if positive > 0 and negative > 0:
        raise ValueError("Mixed-sign interval requires separate positive and negative bands")
    # A nonzero wide end must exceed the fixed narrow end to preserve taper direction.
    if 0 < band["full_width_pt"] <= .85:
        raise ValueError("Increase the declared band scale: magnitude must exceed the fixed 0.85 pt narrow end")
    return 1 if positive > 0 else -1 if negative > 0 else 0


def measure_gradient_bands(gradients: list[dict], downstream: list[dict]) -> dict[str, list[dict]]:
    """Declare each figure's measurement recipe independently of rendering."""
    pathway_bands, reader_bands = [], []
    for start, stop in BANDS.competition_intervals:
        for route in range(2):
            pathway_bands.append(gradient_band_summary(
                gradients, experiment="competition", condition="warm_compatible",
                source="all_readers", parameter=f"feature_{route}",
                start=start, stop=stop, scale=BANDS.competition_scale))
    for role in ALL_ROLES:
        pathway_bands.append(gradient_band_summary(
            gradients, experiment="specialization", condition="Retention", source=role,
            parameter="flower", start=0, stop=TRAINING.specialization_steps,
            scale=BANDS.specialization_scale))
    for conflicting in (False, True):
        condition = "warm_conflicting" if conflicting else "warm_compatible"
        for feature in range(2):
            for reader in range(2):
                for parameter, scale in ((f"feature_{feature}", BANDS.feature_scale),
                                          (f"read_weight_{reader}_{feature}", BANDS.connection_scale)):
                    reader_bands.append(gradient_band_summary(
                        gradients, experiment="competition", condition=condition,
                        source=f"reader_{reader}", parameter=parameter,
                        start=BANDS.contribution_checkpoint, stop=BANDS.contribution_checkpoint+1, scale=scale))
    # Normalize reader-training records to the common band measurement schema.
    measured = [dict(experiment="downstream_training", condition=r["assignment"], source=r["reader"],
                     parameter=r["use"]+"_"+r["coordinate"], **{k:r[k] for k in
                     ("step", "loss_gradient", "descent_pressure", "update_applied")}) for r in downstream]
    for assignment, mapping in READER_ASSIGNMENTS:
        for role in ALL_ROLES:
            reader = "flower" if role in mapping.flower else "broccoli"
            for coordinate in ("green", "read_gain"):
                reader_bands.append(gradient_band_summary(
                    measured, experiment="downstream_training", condition=assignment,
                    source=reader, parameter=role+"_"+coordinate, start=BANDS.reader_checkpoint,
                    stop=BANDS.reader_checkpoint+1, scale=BANDS.reader_scale))
    for band in pathway_bands + reader_bands:
        band_taper_sign(band)
    return {"gradient_band_measurements": pathway_bands,
            "reader_gradient_band_measurements": reader_bands}


def index_bands(bands: list[dict]) -> dict[tuple, dict]:
    """Stable lookup for rendering already measured bands."""
    return {(r["experiment"], r["condition"], r["source"], r["parameter"], r["start_step"]): r
            for r in bands}


def measure_routing_bands(rows: list[dict]) -> list[dict]:
    """Applied per-reader pressures, using the same band recipe as other figures."""
    converted = []
    for row in rows:
        for parameter, pressure, applied in (
                ('flower', row['flower_pressure'], row['feature_update_applied']),
                ('routing_logit', row['routing_logit_pressure'], row['routing_update_applied'])):
            converted.append(dict(experiment='learned_routing', condition=row['condition'],
                source=row['reader'] + '_' + row['route'], parameter=parameter,
                step=row['step'], loss_gradient=-pressure, descent_pressure=pressure,
                update_applied=applied))
    bands = []
    for condition in ('intact', 'routing_frozen'):
        for reader in ('flower', 'broccoli'):
            for route in ('A', 'B'):
                for parameter, scale in (('flower', BANDS.reader_scale),
                                         ('routing_logit', BANDS.routing_logit_scale)):
                    if condition == 'routing_frozen' and parameter == 'routing_logit':
                        continue
                    band = gradient_band_summary(converted, experiment='learned_routing',
                        condition=condition, source=reader+'_'+route, parameter=parameter,
                        start=BANDS.reader_checkpoint, stop=BANDS.reader_checkpoint+1, scale=scale)
                    band_taper_sign(band)
                    bands.append(band)
    return bands
