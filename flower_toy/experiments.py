"""Experiment orchestration, assertions, and result summaries."""

from __future__ import annotations

import numpy as np
import itertools

from .routing import routing_experiment
from .development import development_experiment

from .model import (
    ALL_ROLES,
    READER_ASSIGNMENTS,
    BROCCOLI,
    FEATURES,
    FLOWER,
    GOOD_READERS,
    GREEN,
    READER_TARGETS,
    STIMULI,
    TEXTURE,
    ancestry_likelihood,
    attributed_price,
    boltzmann_choice,
    competition_trace,
    create_histories,
    create_history,
    history_arrays,
    matched_route_training,
    parent_routing_ensemble,
    precursor_interventions,
    reader_metrics,
    reader_outputs,
    response_assay,
    sustained_availability,
    train_broccoli,
    train_downstream_readers,
    train_green,
)

from .measurements import (
    gradient_measurements,
    measure_routing_bands,
    history_tables,
    measure_gradient_bands,
)

from .results import Baseline, ExperimentResult, SuiteResult
from .settings import TRAINING


def make_baseline() -> Baseline:
    # All comparisons share this pretrained checkpoint, then train independent copies.
    green = train_green(FLOWER)
    ancestors = {"A": green, "B": TEXTURE.copy()}
    return Baseline(green, ancestors, create_histories(ancestors))


def development_flow_experiment(baseline: Baseline) -> ExperimentResult:
    summary, tables = {}, {}
    green_a = baseline.green
    # Q1: retain complete developmental response and per-reader gradient traces.
    development = []
    for step in range(TRAINING.green_steps + 1):
        w = train_green(FLOWER, steps=step)
        for stimulus, response in zip(STIMULI, STIMULI @ w):
            development.append(dict(step=step, flower=int(stimulus[0]), green=int(stimulus[1]),
                                    texture=int(stimulus[2]), response=float(response)))
    competition_runs = {(warm, conflict): competition_trace(warm, conflict)
                        for warm in (False, True) for conflict in (False, True)}
    competition = [row for trace in competition_runs.values() for row in trace]
    tables["developmental_responses"] = development
    tables["gradient_pressure"] = competition
    warm_comp = competition_runs[True, False]
    cold_comp = competition_runs[False, False]
    summary["q1_development"] = dict(
        flower_before=float(FLOWER @ FLOWER), flower_after=float(green_a @ FLOWER),
        broccoli_cue_before=float(FLOWER @ BROCCOLI), broccoli_cue_after=float(green_a @ BROCCOLI),
        compatible_initial_cosine=warm_comp[0]["gradient_cosine"],
        conflicting_initial_cosine=competition_runs[True, True][0]["gradient_cosine"],
        established_final_contribution=warm_comp[-1]["established_contribution"],
        alternative_final_contribution=warm_comp[-1]["alternative_contribution"],
        cold_final_alternative_contribution=cold_comp[-1]["alternative_contribution"],
        interpretation="Cue activation precedes discrimination; learned readers route shared gradients.")

    return ExperimentResult(summary, tables)


def downstream_experiment(baseline: Baseline) -> ExperimentResult:
    summary, tables = {}, {}
    histories = baseline.histories
    downstream_trace, downstream_gradients, downstream_endpoints = [], [], []
    downstream_summary = {}
    for assignment, mapping in READER_ASSIGNMENTS:
        # Training restarts from ancestor clones, not the specialized Price endpoints.
        trace, pressure, w, gains = train_downstream_readers(histories[0], mapping)
        downstream_trace.extend(dict(assignment=assignment, **r) for r in trace)
        downstream_gradients.extend(dict(assignment=assignment, **r) for r in pressure)
        for i, role in enumerate(ALL_ROLES):
            downstream_endpoints.append(dict(assignment=assignment, use=role, read_gain=float(gains[i]),
                                             **dict(zip(FEATURES, map(float, w[i])))))
        downstream_summary[assignment] = dict(initial=trace[0], final=trace[-1],
            availability={reader:sustained_availability(trace, reader) for reader in ("flower", "broccoli")})
    tables["downstream_reader_training"] = downstream_trace
    tables["downstream_reader_gradients"] = downstream_gradients
    tables["downstream_reader_endpoints"] = downstream_endpoints
    summary["downstream_reader_training"] = dict(
        steps=TRAINING.reader_steps, learning_rate=TRAINING.reader_lr, loss="mean over readers and inputs of squared output error / 2",
        initialization="Identical ancestor copies at specialization step 0, with unit read gains; not the specialized Price endpoints.",
        interpretation="Separate end-to-end experiment; pathways and read gains train from reader losses, not role targets.",
        conditions=downstream_summary)
    return ExperimentResult(summary, tables)


def acquisition_experiment(baseline: Baseline) -> ExperimentResult:
    summary, tables = {}, {}
    green_a = baseline.green
    cold, warm = train_broccoli(0.), train_broccoli(float(green_a[1]))
    # Q2: lesion only the acquired green coefficient; preserve flower exactly.
    interventions = []
    acquisition_trace = []
    for name, weights in precursor_interventions(green_a).items():
        pretraining = 0 if name == "cold" else TRAINING.green_steps
        # Removal/rescue yield the cached cold/warm coefficients under identical updates.
        result = {"cold": cold, "warm": warm, "precursor_removed": cold, "precursor_rescued": warm}[name]
        interventions.append(dict(intervention=name, precursor_steps=pretraining,
                                  initial_flower_contrast=float(response_assay(weights)),
                                  initial_green_response=float(weights @ GREEN),
                                  initial_texture_response=float(weights @ TEXTURE),
                                  future_steps=result.steps, future_examples=4 * result.steps,
                                  total_steps=pretraining + result.steps,
                                  flower_response=result.flower,
                                  positive_score=result.broccoli_scores[-1],
                                  green_false_positive=result.green_false_positives[-1]))
        for step, (g, score, fp) in enumerate(zip(result.green_weights, result.broccoli_scores,
                                                result.green_false_positives)):
            acquisition_trace.append(dict(intervention=name, step=step, green_weight=g,
                                          broccoli_score=score, green_false_positive=fp))
    tables["precursor_interventions"] = interventions
    tables["acquisition_trajectories"] = acquisition_trace
    # Sensitivity reports all planned cells; it does not select successful cases.
    sweep = []
    for presteps, lr, confidence in itertools.product(
            TRAINING.sensitivity_green_steps, TRAINING.sensitivity_lrs, TRAINING.sensitivity_targets):
        start = float(train_green(FLOWER, steps=presteps)[1])
        try:
            result = train_broccoli(start, lr=lr, target_confidence=confidence, max_steps=TRAINING.matched_budget)
            steps, acquired = result.steps, True
        except RuntimeError:
            steps, acquired = None, False
        sweep.append(dict(precursor_steps=presteps, lr=lr, confidence=confidence,
                          future_steps=steps, acquired=acquired))
    tables["acquisition_sensitivity"] = sweep
    summary["q2_acquisition"] = interventions
    summary["acquisition_route_policy"] = dict(
        temperature=TRAINING.acquisition_temperature,
        probabilities=boltzmann_choice({"green-pretrained": warm.steps, "cold": cold.steps},
                                       temperature=TRAINING.acquisition_temperature))

    return ExperimentResult(summary, tables)


def capacity_acquisition_experiment(baseline: Baseline) -> ExperimentResult:
    """Unconstrained acquisition versus a capacity-limited shared pathway."""
    trials, traces = [], []
    for condition, green, capacity in (
            ("cold", 0., None), ("warm", float(baseline.green[1]), None),
            ("warm_capacity", float(baseline.green[1]), TRAINING.capacity)):
        result = train_broccoli(green, target_confidence=TRAINING.strict_target_score,
                                shared_capacity=capacity)
        trials.append(dict(condition=condition, target_score=TRAINING.strict_target_score,
                           capacity=capacity, subsequent_steps=result.steps,
                           flower_response=result.flower, green_response=result.g,
                           positive_score=result.broccoli_scores[-1],
                           green_false_positive=result.green_false_positives[-1]))
        for step, (flower, green, score, false_positive) in enumerate(zip(
                result.flower_weights, result.green_weights, result.broccoli_scores,
                result.green_false_positives)):
            traces.append(dict(condition=condition, step=step, flower_response=flower,
                               green_response=green, positive_score=score,
                               green_false_positive=false_positive))
    return ExperimentResult(summary={"capacity_acquisition": trials}, tables={
        "capacity_acquisition": trials, "capacity_acquisition_trajectories": traces})


def matched_world_experiment() -> ExperimentResult:
    summary, tables = {}, {}
    # Q3: identical model and update rules; checkpoint routing differs.
    shared, separate = matched_route_training(True), matched_route_training(False)
    tables["matched_worlds"] = shared + separate
    summary["q3_matched_worlds"] = dict(shared=shared[-1], separate=separate[-1],
        initial_flower_equal=bool(np.array_equal(shared[0]["flower_response"] * STIMULI[:, 0],
                                                 separate[0]["flower_response"] * STIMULI[:, 0])),
        capacity_per_route=TRAINING.capacity, target_confidence=TRAINING.target_score,
        interpretation="Both worlds implement flower=x_flower initially on all eight inputs.")
    summary["q3_matched_worlds"]["update_decomposition"] = {
        rows[0]["world"]: {key: sum(row[key] for row in rows)
                            for key in ("flower_gradient_delta", "flower_projection_delta",
                                        "flower_total_delta", "green_gradient_delta",
                                        "green_projection_delta", "green_total_delta")}
        for rows in (shared, separate)}
    world_sweep = []
    for confidence, is_shared in itertools.product(TRAINING.sensitivity_targets, (True, False)):
        rows = ((shared if is_shared else separate) if confidence == TRAINING.target_score
                else matched_route_training(is_shared, target_confidence=confidence))
        world_sweep.append(dict(confidence=confidence, **rows[-1]))
    tables["matched_world_sensitivity"] = world_sweep
    summary["q3_criterion_sensitivity"] = world_sweep
    comparison = [dict(target_score=r["confidence"], world=r["world"],
                       subsequent_steps=r["step"], flower_response=r["flower_response"],
                       flower_accuracy=r["flower_accuracy"], acquired=r["acquired"])
                  for r in world_sweep if r["confidence"] in TRAINING.comparison_targets]
    tables["acquisition_comparison"] = comparison

    return ExperimentResult(summary, tables)


def history_experiment(baseline: Baseline) -> ExperimentResult:
    summary, tables = {}, {}
    histories = baseline.histories
    # Q4-Q6: all checkpoints, both assays, both reader assignments.
    accounting, performance, dependencies, truth_tables = [], [], [], []
    for history in histories:
        base_weights = np.array([history.ancestors[p] for p in ("A", "B")])
        for step in range(TRAINING.specialization_steps + 1):
            w, credit = history_arrays(history, step)
            for assignment, mapping in READER_ASSIGNMENTS:
                outputs = reader_outputs(w, mapping)
                performance.append(dict(history=history.name, assignment=assignment,
                                        step=step, **reader_metrics(outputs)))
                # Apply the same assay to ancestors and descendants before grouping credit.
                for binary in (False, True):
                    base = response_assay(base_weights, binary)
                    later = response_assay(w, binary)
                    for group, roles in (("all", ALL_ROLES), ("flower", mapping.flower),
                                         ("broccoli", mapping.broccoli)):
                        indices = [ALL_ROLES.index(r) for r in roles]
                        accounting.append(dict(history=history.name, assignment=assignment,
                                               step=step, assay="binary" if binary else "continuous",
                                               population=group,
                                               **attributed_price(base, later[indices], credit[indices])))
                if step == TRAINING.specialization_steps:
                    for x, y in zip(STIMULI, outputs):
                        truth_tables.append(dict(history=history.name, assignment=assignment,
                                                 flower=int(x[0]), green=int(x[1]), texture=int(x[2]),
                                                 flower_output=float(y[0]), broccoli_output=float(y[1])))
                    for role in ALL_ROLES:
                        lesioned = reader_outputs(w, mapping, role)
                        delta_mse = np.mean((lesioned - READER_TARGETS)**2, axis=0) - np.mean(
                            (outputs - READER_TARGETS)**2, axis=0)
                        change = np.mean(np.abs(lesioned - outputs), axis=0)
                        dependencies.append(dict(history=history.name, assignment=assignment,
                                                 ablated_use=role, flower_effect=float(change[0]),
                                                 broccoli_effect=float(change[1]),
                                                 flower_mse_increase=float(delta_mse[0]),
                                                 broccoli_mse_increase=float(delta_mse[1])))
    tables["price_by_checkpoint"] = accounting
    # Fixed endpoint routing comparison, before any reader-driven updates.
    # Preserve the conventional transmission key in numerical exports.
    reader_accounting = [dict(comparison="fixed_specialized_endpoint", **r)
                         for r in accounting if r["step"] == TRAINING.specialization_steps and r["assay"] == "binary"]
    tables["reader_price_accounting"] = reader_accounting
    tables["reader_performance"] = performance
    tables["dependency_ablations"] = dependencies
    tables["reader_truth_tables"] = truth_tables
    summary["q4_price"] = [r for r in accounting if r["step"] == TRAINING.specialization_steps
                           and r["population"] == "all" and r["assignment"] == "coherent"]
    # Explicit equal-coordinate control: one-to-one ancestry forces selection=0.
    summary["q4_equal_coordinate_control"] = attributed_price(
        np.array([1., 0.]), np.array([.2, .9]), np.eye(2))
    summary["q5_readers"] = [r for r in performance if r["step"] == TRAINING.specialization_steps]
    endpoints = [reader_outputs(history_arrays(h)[0], GOOD_READERS) for h in histories]
    summary["q6_identity"] = dict(
        final_max_difference=float(np.max(np.abs(endpoints[0] - endpoints[1]))),
        checkpoint_max_differences=[dict(step=s, max_difference=float(np.max(np.abs(
            reader_outputs(history_arrays(histories[0], s)[0], GOOD_READERS)
            - reader_outputs(history_arrays(histories[1], s)[0], GOOD_READERS)))))
            for s in (0, 5, 20, 60, TRAINING.specialization_steps)],
        interpretation="Equivalent endpoints do not imply continuous availability during training.")
    availability = []
    for history, assignment, reader in itertools.product(
            (h.name for h in histories), ("coherent", "misrouted"), ("flower", "broccoli")):
        rows = [r for r in performance if r["history"] == history and r["assignment"] == assignment]
        measured = sustained_availability(rows, reader)
        for criterion in ("classification", "response_precision"):
            availability.append(dict(history=history, assignment=assignment, reader=reader,
                                     criterion=criterion, decision_threshold=TRAINING.decision_threshold,
                                     max_error_tolerance=measured["max_error_tolerance"],
                                     **measured[criterion]))
    tables["reader_availability"] = availability
    summary["q6_identity"]["reader_availability"] = availability

    return ExperimentResult(summary, tables)


def ancestry_experiment(baseline: Baseline) -> ExperimentResult:
    summary, tables = {}, {}
    ancestors = baseline.ancestors
    # Enumerate the entire finite ancestry ensemble, rather than just two cases.
    ensemble = parent_routing_ensemble(ancestors)
    possibilities = []
    for choices in itertools.product(("A", "B"), repeat=4):
        # One independent A/B parent choice per role gives the complete 2^4 ensemble.
        h = create_history("".join(choices), ancestors, dict(zip(ALL_ROLES, choices)))
        w, credit = history_arrays(h)
        p = attributed_price(response_assay(np.array(list(ancestors.values())), True),
                             response_assay(w, True), credit)
        possibilities.append(dict(parents="".join(choices), probability=ancestry_likelihood(h, ensemble),
                                  **p, **reader_metrics(reader_outputs(w, GOOD_READERS))))
    tables["ancestry_ensemble"] = possibilities
    summary["ensemble"] = dict(assignments=len(possibilities),
        probability_sum=sum(r["probability"] for r in possibilities),
        energy="half squared distance from parent weights to role target", temperature=TRAINING.ancestry_temperature,
        interpretation="Conditional illustrative policy; histories are interventions, not samples.")
    # A fractional, unassigned-use example exercises the footnote's full rule.
    summary["fractional_credit_example"] = attributed_price(
        np.array([1., 0., .4]), np.array([.9, .2, .7, .8]),
        np.array([[.75, .25, 0], [0, 1, 0], [.5, .5, 0], [0, 0, 0]]))
    return ExperimentResult(summary, tables)


def run_suite() -> SuiteResult:
    """Compute every experiment once, without file writes, rendering, or printing."""
    baseline = make_baseline()
    result = SuiteResult(baseline=baseline)
    result.summary = {"scope": "Synthetic instantiation of the six blog questions; declared curricula and ancestry",
        "configuration": {"input_cues": list(FEATURES), "assay": "mean matched flower contrast",
                          "binary_threshold": TRAINING.decision_threshold,
                          "specialization_steps": TRAINING.specialization_steps,
                          "reader_decision_threshold": TRAINING.decision_threshold,
                          "reader_max_error_tolerance": TRAINING.max_reader_error,
                          "specialization_lr": TRAINING.specialization_lr,
                          "green_pretraining_steps": TRAINING.green_steps,
                          "green_pretraining_lr": TRAINING.green_lr,
                          "reader_groups": {"flower": list(GOOD_READERS.flower),
                                            "broccoli": list(GOOD_READERS.broccoli)}}}
    # Table keys become export filenames; reject collisions when combining experiments.
    result.tables.update(history_tables(baseline.histories))
    for experiment in (
            routing_experiment(), development_flow_experiment(baseline), downstream_experiment(baseline),
            acquisition_experiment(baseline), capacity_acquisition_experiment(baseline),
            matched_world_experiment(),
            history_experiment(baseline), ancestry_experiment(baseline),
            development_experiment(baseline.ancestors)):
        overlap = result.tables.keys() & experiment.tables.keys()
        if overlap:
            raise ValueError(f"Duplicate experiment tables: {overlap}")
        result.summary.update(experiment.summary)
        result.tables.update(experiment.tables)
    # Measure bands before rendering so numerical outputs never depend on plotting.
    gradients = gradient_measurements(result.tables["gradient_pressure"], baseline.histories)
    result.tables["signed_gradients"] = gradients
    result.tables.update(measure_gradient_bands(gradients, result.tables["downstream_reader_gradients"]))
    result.summary["gradient_bands"] = result.tables["gradient_band_measurements"]
    result.summary["reader_gradient_bands"] = result.tables["reader_gradient_band_measurements"]
    result.tables["learned_routing_bands"] = measure_routing_bands(result.tables["learned_routing_pressure"])
    return result
