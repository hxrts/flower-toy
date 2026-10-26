"""Core learning dynamics, reader computations, and ancestry accounting."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .settings import TRAINING


# Fixed synthetic cue basis shared by every experiment.
FLOWER = np.array([1.0, 0.0, 0.0])
GREEN = np.array([0.0, 1.0, 0.0])
TEXTURE = np.array([0.0, 0.0, 1.0])
BROCCOLI = GREEN + TEXTURE
FEATURES = ("flower", "green", "texture")


# Local targets belong to the prescribed role-specialization experiment.
ROLE_TARGETS = {
    "F1": FLOWER,
    "F2": FLOWER,
    "G": GREEN,
    "T": TEXTURE,
}
ALL_ROLES = ("F1", "F2", "G", "T")


# Parameter records and declared reader wiring.
@dataclass(frozen=True)
class Acquisition:
    """Actual gradient-based training of a scalar green-feature coefficient."""
    g: float
    flower: float
    steps: int
    broccoli_scores: tuple[float, ...]
    green_false_positives: tuple[float, ...]
    green_weights: tuple[float, ...]
    flower_weights: tuple[float, ...]


@dataclass(frozen=True)
class LearnedUse:
    """A later functional use and its declared ancestral source."""
    role: str
    parent: str
    weights: np.ndarray
    trajectory: np.ndarray   # Checkpoint 0 is the clone, before any specialization update.


@dataclass(frozen=True)
class PriceResult:
    baseline: float
    selection: float
    transmission: float
    final: float
    total: float
    parent_counts: tuple[int, int]


@dataclass(frozen=True)
class ModelHistory:
    name: str
    ancestors: Mapping[str, np.ndarray]
    uses: Mapping[str, LearnedUse]
    curriculum: str


@dataclass(frozen=True)
class ConsumerMapping:
    """Ordered use assignments. Broccoli multiplies its two incoming signals."""
    flower: tuple[str, str]
    broccoli: tuple[str, str]


GOOD_READERS = ConsumerMapping(flower=("F1", "F2"), broccoli=("G", "T"))
MISROUTED_READERS = ConsumerMapping(flower=("G", "T"), broccoli=("F1", "F2"))
READER_ASSIGNMENTS = (("coherent", GOOD_READERS), ("misrouted", MISROUTED_READERS))


# Precursor learning and subsequent acquisition.
def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def flower_trait(weights: np.ndarray) -> float:
    """A common CONTINUOUS fixed-probe assay, not a changeable feature label."""
    return float(weights @ FLOWER)


def train_green(initial: np.ndarray, steps: int = TRAINING.green_steps, lr: float = TRAINING.green_lr) -> np.ndarray:
    """Learn the green cue by gradient descent on (g-1)^2/2.

    Only the green coordinate is eligible for updating in this experiment.
    The flower coefficient remains unchanged. That permission is explicit.
    """
    w = initial.astype(float).copy()
    for _ in range(steps):
        w[1] -= lr * (w[1] - 1.0)
    return w


def broccoli_readout(g: float, t: float) -> float:
    """A smooth broccoli detector using selected green and texture signals."""
    return sigmoid(8.0 * (g + t - 1.5))


def train_broccoli(
    green_start: float,
    *,
    flower_start: float = 1.0,
    target_confidence: float = TRAINING.target_score,
    max_steps: int = TRAINING.acquisition_budget,
    lr: float = TRAINING.acquisition_lr,
    shared_capacity: float | None = None,
) -> Acquisition:
    """Fit broccoli (green AND texture) with a frozen texture mechanism.

    The training data are the full {green,texture} binary square, with label
    green AND texture. The trainable scalar g is a green-sensitive response
    of a preceding pathway. The readout uses an 8x logistic transition.
    BCE loss gradient is computed exactly, not scripted as a response change.

    When shared_capacity is set, the flower and green coefficients of that
    same pathway share an L2 budget. This is a controlled toy constraint,
    not a claim about biological/transformer network normalization.
    """
    green = float(green_start)
    flower = float(flower_start)
    scores, false_positives, gs, fs = [], [], [], []
    for step in range(max_steps + 1):
        broccoli_probability = broccoli_readout(green, 1.0)
        green_only_probability = broccoli_readout(green, 0.0)
        scores.append(broccoli_probability)
        false_positives.append(green_only_probability)
        gs.append(green)
        fs.append(flower)
        if broccoli_probability >= target_confidence and green_only_probability < 0.10:
            return Acquisition(green, flower, step, tuple(scores),
                               tuple(false_positives), tuple(gs), tuple(fs))

        # Of the four examples, only green-only and broccoli have green=1.
        # d BCE(sigmoid(logit), y)/d logit = sigmoid(logit) - y.
        gradient_g = (8.0 / 4.0) * (green_only_probability + broccoli_probability - 1.0)
        green -= lr * gradient_g
        if shared_capacity is not None:
            norm = math.hypot(flower, green)
            if norm > shared_capacity:
                flower *= shared_capacity / norm
                green *= shared_capacity / norm
    raise RuntimeError("Broccoli criterion not attained; check toy hyperparameters")


# Route policies and registered developmental histories.
def boltzmann_choice(steps: Mapping[str, int], temperature: float = 7.0) -> dict[str, float]:
    """Optional contract-conditioned ensemble over eligible learning routes.

    Energy is the measured acquisition step count, *not* Price fitness.
    It illustrates a Gibbs-style policy without asserting that attention
    routing, causal learning pressure, and ancestry weights are identical.
    """
    energy = np.array([steps[name] for name in steps], dtype=float)
    logits = -energy / temperature
    weights = np.exp(logits - np.max(logits))
    probabilities = weights / weights.sum()
    return dict(zip(steps, map(float, probabilities)))


def parent_routing_ensemble(ancestors: Mapping[str, np.ndarray],
                            temperature: float = TRAINING.ancestry_temperature) -> dict[str, dict[str, float]]:
    """A Gibbs distribution over admissible parent choices for EACH role.

    Energy is the pre-training 1/2 squared distance to the role target.
    These conditional probabilities define a toy selection policy over
    candidate continuations; they do not certify ancestry or causality.
    The histories below are two possible realized outcomes of this law.
    """
    probabilities = {}
    for role, target in ROLE_TARGETS.items():
        energies = np.array([.5 * float(np.sum((ancestors[p] - target)**2))
                             for p in ("A", "B")])
        logits = -energies / temperature
        score = np.exp(logits - logits.max())
        score /= score.sum()
        probabilities[role] = {"A": float(score[0]), "B": float(score[1])}
        assert math.isclose(sum(probabilities[role].values()), 1.0, abs_tol=1e-14)
    return probabilities


def ancestry_likelihood(history: ModelHistory,
                        ensemble: Mapping[str, Mapping[str, float]]) -> float:
    """Likelihood of the declared four-use ancestry under the toy ensemble."""
    return math.prod(ensemble[role][history.uses[role].parent]
                     for role in ALL_ROLES)


def specialize(initial: np.ndarray, target: np.ndarray,
               *, steps: int = TRAINING.specialization_steps, lr: float = TRAINING.specialization_lr) -> np.ndarray:
    """Full gradient descent on 1/2 ||w-target||^2.

    Squared error over the three orthogonal probes yields gradient w-target.
    The role target is supplied by a training curriculum, not magically
    discovered by the Price equation. Full trace retained for auditing.
    """
    weights = np.asarray(initial, dtype=float).copy()
    trace = np.empty((steps + 1, 3), dtype=float)
    trace[0] = weights
    for t in range(1, steps + 1):
        weights -= lr * (weights - target)
        trace[t] = weights
    return trace


def create_history(name: str, ancestors: Mapping[str, np.ndarray],
                   parents_by_role: Mapping[str, str]) -> ModelHistory:
    assert set(parents_by_role) == set(ALL_ROLES)
    uses: dict[str, LearnedUse] = {}
    for role in ALL_ROLES:
        parent = parents_by_role[role]
        assert parent in ("A", "B")
        # Each use trains its own copy; its registered parent stays fixed.
        tr = specialize(ancestors[parent], ROLE_TARGETS[role])
        uses[role] = LearnedUse(role, parent, tr[-1].copy(), tr)
    return ModelHistory(name, ancestors, uses,
                        "Externally specified role-specific training curricula")


def create_histories(ancestors: Mapping[str, np.ndarray]) -> tuple[ModelHistory, ...]:
    """The two registered ancestries used throughout the experiments."""
    return (
        create_history("Retention", ancestors, dict(F1="A", F2="A", G="A", T="B")),
        create_history("Re-acquisition", ancestors, dict(F1="B", F2="B", G="A", T="B")),
    )


# Price accounting and common response assays.
def price(history: ModelHistory, selected_roles: tuple[str, ...] = ALL_ROLES,
          at_step: int | None = None) -> PriceResult:
    """Exact two-ancestor Price identity for any declared set of later uses.

    Let w_i=count descendants of ancestor i, z_i=its baseline flower trait,
    and delta_i=mean(descendant traits - z_i). Then:
        final - initial = cov(w,z)/mean(w) + E[w*delta]/mean(w).
    The arithmetic uses the actual trained descendant probe responses.
    """
    if not selected_roles:
        raise ValueError("Price analysis requires a nonempty selected population")
    parents = ("A", "B")
    # Reader subsets retain both ancestors and the same baseline mean.
    initial = np.array([flower_trait(history.ancestors[p]) for p in parents])
    weights = np.array([sum(history.uses[r].parent == p for r in selected_roles)
                        for p in parents], dtype=float)
    avg_weight = float(weights.mean())
    assert avg_weight > 0
    all_descendant_traits: list[float] = []
    transmission_by_parent = np.zeros(2)
    for role in selected_roles:
        use = history.uses[role]
        w = use.weights if at_step is None else use.trajectory[at_step]
        z = flower_trait(w)
        all_descendant_traits.append(z)
        parent_index = parents.index(use.parent)
        transmission_by_parent[parent_index] += z - initial[parent_index]

    baseline = float(initial.mean())
    selection = float(np.mean((weights - weights.mean()) * (initial - baseline)) /
                      avg_weight)
    transmission = float((transmission_by_parent.sum() / 2.0) / avg_weight)
    final = float(np.mean(all_descendant_traits))
    delta = final - baseline
    assert math.isclose(selection + transmission, delta, abs_tol=3e-12), (
        selection, transmission, delta)
    return PriceResult(baseline, selection, transmission, final, delta,
                       tuple(map(int, weights)))


def consumer_output(history: ModelHistory, mapping: ConsumerMapping,
                    x: np.ndarray) -> tuple[float, float]:
    """A declared downstream-reader contract.

    Flower reader aggregates two signals by their mean. Broccoli reader
    composes its two inputs by multiplication (an AND on binary cues).
    No part of this function knows or consults the ancestry of the uses.
    """
    f = float(np.mean([history.uses[r].weights @ x for r in mapping.flower]))
    b1, b2 = (float(history.uses[r].weights @ x) for r in mapping.broccoli)
    return f, b1 * b2


def observable_quotient(history: ModelHistory, mapping: ConsumerMapping) -> np.ndarray:
    """Consumer-sufficient signature on all eight declared binary inputs.

    This is an observable quotient for THESE readers only, not a proof of
    equivalence under future learning or under every possible consumer.
    """
    stimuli = (np.array(bits, dtype=float)
               for bits in itertools.product((0, 1), repeat=3))
    return np.array([consumer_output(history, mapping, x) for x in stimuli])


def attributed_price(baseline: np.ndarray, descendants: np.ndarray,
                     credit: np.ndarray) -> dict[str, float]:
    """Price accounting with equal baseline weights and fractional ancestry.

    credit[j, i] attributes later use j to ancestor i. Each included use has
    one unit of credit, split among ancestors; all-zero rows are unassigned
    and excluded. Reader groups are subsets of rows, never of ancestors.
    This is an accounting convention, not an inference of causal ancestry.
    """
    z = np.asarray(baseline, dtype=float)
    later = np.asarray(descendants, dtype=float)
    c = np.asarray(credit, dtype=float)
    if z.ndim != 1 or not len(z) or later.ndim != 1 or c.shape != (len(later), len(z)):
        raise ValueError("Expected baseline[n], descendants[m], credit[m,n]")
    if not all(np.isfinite(x).all() for x in (z, later, c)) or np.any(c < 0):
        raise ValueError("Traits must be finite and ancestry credit nonnegative")
    mass = c.sum(axis=1)
    included = mass > 0
    if not included.any() or not np.allclose(mass[included], 1, rtol=0, atol=1e-12):
        raise ValueError("Each assigned use must have exactly one total credit")
    # Rows distribute each use's credit; columns total the credit per ancestor.
    weights = c.sum(axis=0)
    total = float(weights.sum())
    initial = float(z.mean())
    selected = float(weights @ z / total)
    final = float(mass @ later / total)
    selection = selected - initial
    transmission = float(np.sum(c * (later[:, None] - z[None, :])) / total)
    if not np.isclose(selection + transmission, final - initial, atol=1e-12):
        raise ArithmeticError("Price identity failed")
    return dict(baseline=initial, selection=selection, transmission=transmission,
                final=final, delta=final - initial, descendant_credit=total)


# Exhaustive binary inputs; reader targets are flower and green AND texture.
STIMULI = np.array(list(itertools.product((0., 1.), repeat=3)))


READER_TARGETS = np.column_stack((STIMULI[:, 0], STIMULI[:, 1] * STIMULI[:, 2]))


def response_assay(weights: np.ndarray, binary: bool = False) -> np.ndarray:
    """Mean flower contrast over all four matched green/texture controls.

    The fixed binary criterion is contrast >= 0.5. This measures a response
    distinction, not selectivity against every other cue or reader accuracy.
    For these linear features the contrast equals the flower coefficient.
    """
    w = np.asarray(weights)
    responses = STIMULI @ w.T
    contrast = (responses[STIMULI[:, 0] == 1].mean(axis=0)
                - responses[STIMULI[:, 0] == 0].mean(axis=0))
    return (contrast >= TRAINING.decision_threshold).astype(float) if binary else contrast


def history_arrays(history: ModelHistory, step: int = -1) -> tuple[np.ndarray, np.ndarray]:
    weights = np.array([history.uses[r].trajectory[step] for r in ALL_ROLES])
    credit = np.array([[float(history.uses[r].parent == p) for p in ("A", "B")]
                       for r in ALL_ROLES])
    return weights, credit


# Reader computations, backpropagation, and longitudinal availability.
def reader_indices(mapping: ConsumerMapping) -> tuple[list[int], list[int]]:
    """Resolve the declared flower and broccoli inputs in pathway order."""
    return ([ALL_ROLES.index(r) for r in mapping.flower],
            [ALL_ROLES.index(r) for r in mapping.broccoli])


def read_signals(signals: np.ndarray, mapping: ConsumerMapping) -> np.ndarray:
    """Shared flower-mean and broccoli-product forward computation."""
    flower, broccoli = reader_indices(mapping)
    return np.column_stack((signals[:, flower].mean(axis=1),
                            signals[:, broccoli].prod(axis=1)))


def reader_outputs(weights: np.ndarray, mapping: ConsumerMapping,
                   ablate: str | None = None) -> np.ndarray:
    """Evaluate every input, optionally setting one pathway's signal to zero."""
    signals = STIMULI @ weights.T
    if ablate is not None:
        signals[:, ALL_ROLES.index(ablate)] = 0
    return read_signals(signals, mapping)


def reader_metrics(outputs: np.ndarray) -> dict[str, float]:
    error = outputs - READER_TARGETS
    return {f"{name}_{metric}": float(value)
            for i, name in enumerate(("flower", "broccoli"))
            for metric, value in (("mse", np.mean(error[:, i] ** 2)),
                                  ("max_error", np.max(np.abs(error[:, i]))),
                                  ("accuracy", np.mean((outputs[:, i] >= TRAINING.decision_threshold)
                                                       == READER_TARGETS[:, i])))}


def downstream_loss_gradients(weights: np.ndarray, gains: np.ndarray,
                              mapping: ConsumerMapping) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reader losses backpropagate through the actual mean/product computations.

    L = mean_readers(mean_inputs((output-target)^2 / 2)). The returned
    gradients have a leading reader axis and already include the 1/2 reader
    weighting. Sum that axis for the update. No local role targets enter.
    """
    signals = STIMULI @ weights.T
    routed = signals * gains
    f, b = reader_indices(mapping)
    outputs = read_signals(routed, mapping)
    error = outputs - READER_TARGETS
    losses = .5 * np.mean(error ** 2, axis=0)
    d_routed = np.zeros((2, len(STIMULI), 4))
    for i in f:
        d_routed[0, :, i] = error[:, 0] / (2 * len(STIMULI) * len(f))
    d_routed[1, :, b[0]] = error[:, 1] * routed[:, b[1]] / (2 * len(STIMULI))
    d_routed[1, :, b[1]] = error[:, 1] * routed[:, b[0]] / (2 * len(STIMULI))
    # Axes: reader r, input n, pathway i, and cue j.
    d_weights = np.einsum('rni,nj->rij', d_routed * gains, STIMULI)
    d_gains = np.einsum('rni,ni->ri', d_routed, signals)
    return outputs, losses, d_weights, d_gains


def train_downstream_readers(history: ModelHistory, mapping: ConsumerMapping,
                             steps: int = TRAINING.reader_steps, lr: float = TRAINING.reader_lr) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    """Joint feature/read-connection learning from reader targets alone.

    Start from registered cloned ancestors, before role specialization.
    Connectivity and ancestry are controlled; all four 3D pathways and
    their four read gains train on the full eight-input domain.
    """
    weights = history_arrays(history, 0)[0].copy()
    gains = np.ones(4)
    trace, gradients = [], []
    for step in range(steps + 1):
        outputs, losses, dw, dc = downstream_loss_gradients(weights, gains, mapping)
        trace.append(dict(step=step, loss=float(losses.mean()), flower_loss=float(losses[0]),
                          broccoli_loss=float(losses[1]), **reader_metrics(outputs)))
        for reader, name in enumerate(("flower", "broccoli")):
            for role, use in enumerate(ALL_ROLES):
                for coordinate, gradient in list(zip(FEATURES, dw[reader, role])) + [("read_gain", dc[reader, role])]:
                    gradients.append(dict(step=step, reader=name, use=use, coordinate=coordinate,
                                          loss_gradient=float(gradient), descent_pressure=float(-gradient),
                                          gradient_update=float(-lr * gradient) if step < steps else 0.,
                                          update_applied=step < steps,
                                          value=float(gains[role] if coordinate == "read_gain" else
                                                      weights[role, FEATURES.index(coordinate)])))
        if step < steps:
            # Reader contributions already include task averaging; do not average twice.
            weights -= lr * dw.sum(axis=0)
            gains -= lr * dc.sum(axis=0)
    return trace, gradients, weights, gains


def sustained_availability(rows: list[dict], reader: str,
                           max_error: float = TRAINING.max_reader_error) -> dict:
    """Availability at every recorded checkpoint, not between measurements.

    Classifying every fixed input correctly and approximating its target
    response within a tolerance are distinct reader requirements. Report
    the first checkpoint of the final uninterrupted successful interval;
    return None if even the final checkpoint fails the relevant criterion.
    """
    def summarize(passed: list[bool]) -> dict:
        # Only the final uninterrupted successful run counts as sustained availability.
        last_failure = max((i for i, ok in enumerate(passed) if not ok), default=-1)
        start = last_failure + 1
        return dict(available_at_every_checkpoint=all(passed),
                    first_sustained_step=rows[start]["step"] if start < len(rows) else None,
                    failing_checkpoints=sum(not ok for ok in passed))
    if not rows or max_error < 0 or not np.isfinite(max_error):
        raise ValueError("Require checkpoints and a finite nonnegative error tolerance")
    if any(b["step"] <= a["step"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Checkpoints must be strictly increasing")
    return dict(classification=summarize([r[f"{reader}_accuracy"] == 1. for r in rows]),
                response_precision=summarize([r[f"{reader}_max_error"] <= max_error for r in rows]),
                max_error_tolerance=max_error)


def precursor_interventions(initial: np.ndarray) -> dict[str, np.ndarray]:
    """Intervene on the acquired green coordinate, holding other weights fixed."""
    warm = np.array(initial, dtype=float, copy=True)
    removed = warm.copy()
    removed[1] = 0.
    rescued = removed.copy()
    rescued[1] = warm[1]
    return dict(cold=FLOWER.copy(), warm=warm, precursor_removed=removed,
                precursor_rescued=rescued)


# Competing routes and architecture-matched capacity experiments.
def competing_gradient(a: np.ndarray, b: np.ndarray,
                       targets: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Two trainable feature routes a and task-specific read weights b.

    Each task learns y=target*x on x in {-1,+1}. Mean squared loss on
    these balanced examples reduces exactly to 1/2 mean((b @ a-target)^2).
    Shared feature updates sum the downstream readers' actual gradients.
    """
    residual = b @ a - targets
    return (float(.5 * np.mean(residual ** 2)),
            b.T @ residual / len(targets), np.outer(residual, a) / len(targets))


def competition_trace(head_start: bool, conflicting: bool = False,
                      steps: int = TRAINING.competition_steps) -> list[dict]:
    a = np.array([.8 if head_start else .05, .05])
    b = np.full((2, 2), .05)
    targets = np.array([1., -1. if conflicting else 1.])
    trace = []
    for step in range(steps + 1):
        loss, da, db = competing_gradient(a, b, targets)
        residual = b @ a - targets
        # Keep individual contributions so a small net gradient cannot hide conflict.
        per_reader = residual[:, None] * b / len(targets)
        denom = np.linalg.norm(per_reader[0]) * np.linalg.norm(per_reader[1])
        cosine = float(per_reader[0] @ per_reader[1] / denom) if denom > 1e-30 else 0.
        trace.append(dict(step=step, head_start=head_start, conflicting=conflicting,
                          loss=loss, gradient_cosine=cosine,
                          established_contribution=float(np.mean(a[0] * b[:, 0])),
                          alternative_contribution=float(np.mean(a[1] * b[:, 1])),
                          established_pressure=float(abs(da[0])),
                          alternative_pressure=float(abs(da[1])),
                          feature_0=float(a[0]), feature_1=float(a[1]),
                          reader_00=float(b[0, 0]), reader_01=float(b[0, 1]),
                          reader_10=float(b[1, 0]), reader_11=float(b[1, 1])))
        trace[-1].update({f"feature_gradient_{i}": float(da[i]) for i in range(2)})
        trace[-1].update({f"reader_{r}_feature_gradient_{i}": float(per_reader[r, i])
                          for r in range(2) for i in range(2)})
        trace[-1].update({f"reader_weight_gradient_{r}_{i}": float(db[r, i])
                          for r in range(2) for i in range(2)})
        trace[-1]["update_applied"] = step < steps
        if step < steps:
            a -= TRAINING.competition_lr * da
            b -= TRAINING.competition_lr * db
    return trace


def matched_route_training(shared: bool, max_steps: int = TRAINING.matched_budget,
                           target_confidence: float = TRAINING.target_score) -> list[dict]:
    """Same two-route architecture, capacity, data, and optimizer in both worlds.

    Each route has flower/green coefficients under a per-route L2 budget.
    The existing flower reader uses route 0's flower channel; the broccoli
    reader uses a green channel plus fixed texture. Channel masks are part
    of the architecture, identical across worlds. Only checkpoint weights
    (including the frozen broccoli routing weights) differ. World A uses
    its warm shared route; B uses its cold independent route. The same
    projection applies to both routes in both worlds.
    """
    w = np.array([[1., .8 if shared else 0.], [0., 0.]])
    route = np.array([1., 0.]) if shared else np.array([0., 1.])
    trace = []
    x = STIMULI
    labels = READER_TARGETS[:, 1]
    for step in range(max_steps + 1):
        g = float(route @ w[:, 1])
        logits = 8 * (g * x[:, 1] + x[:, 2] - 1.5)
        predictions = 1 / (1 + np.exp(-logits))
        positive = float(predictions[labels == 1].min())
        negative = float(predictions[labels == 0].max())
        success = positive >= target_confidence and negative < .1
        gradient = float(np.mean(8 * (predictions - labels) * x[:, 1]))
        loss_gradient = np.zeros_like(w)
        loss_gradient[:, 1] = gradient * route
        applied = not success and step < max_steps
        # Projection is a separate intervention, not part of the loss derivative.
        proposed = w - TRAINING.acquisition_lr * loss_gradient if applied else w.copy()
        norms = np.linalg.norm(proposed, axis=1)
        projected = (proposed * np.minimum(1., TRAINING.capacity / np.maximum(norms, 1e-15))[:, None]
                     if applied else proposed.copy())
        gradient_delta = proposed - w
        projection_delta = projected - proposed
        trace.append(dict(world="shared" if shared else "separate", step=step,
                          flower_response=float(w[0, 0]), green_response=g,
                          flower_mse=float(np.mean((w[0, 0] * x[:, 0] - x[:, 0])**2)),
                          flower_max_error=float(np.max(np.abs(w[0, 0] * x[:, 0] - x[:, 0]))),
                          flower_accuracy=float(np.mean((w[0, 0] * x[:, 0] >= TRAINING.decision_threshold) == x[:, 0])),
                          flower_responsive=bool(w[0, 0] >= TRAINING.decision_threshold),
                          positive_min=positive, negative_max=negative,
                          loss=float(np.mean(np.logaddexp(0, logits) - labels * logits)),
                          acquired=success, update_applied=applied,
                          flower_loss_gradient=float(loss_gradient[0, 0]),
                          flower_gradient_delta=float(gradient_delta[0, 0]),
                          flower_projection_delta=float(projection_delta[0, 0]),
                          flower_total_delta=float(projected[0, 0] - w[0, 0]),
                          green_loss_gradient=gradient,
                          green_gradient_delta=float(route @ gradient_delta[:, 1]),
                          green_projection_delta=float(route @ projection_delta[:, 1]),
                          green_total_delta=float(route @ (projected - w)[:, 1])))
        if success or step == max_steps:
            break
        w = projected
    return trace
