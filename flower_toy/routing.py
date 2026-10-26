"""Reader-driven routing and weighted Price accounting on the same trajectory.

Each reader allocates unit routing mass over two persistent ancestral routes.
Softmax shares are allocation weights, not neuron counts or causal importance.
Broccoli reads a learned mixture multiplied by a fixed texture signal.
"""
import numpy as np
from .model import STIMULI, READER_TARGETS, reader_metrics
from .results import ExperimentResult
from .settings import TRAINING


def routing_experiment(steps=TRAINING.routing_steps, lr=TRAINING.routing_lr):
    trajectories, accounting, pressures = [], [], []
    endpoints = {}
    for condition in ('intact', 'green_removed', 'green_restored', 'routing_frozen'):
        initial = np.array([[1., .8, 0.], [0., 0., 0.]])
        if condition in ('green_removed', 'green_restored'):
            initial[0, 1] = 0.
        if condition == 'green_restored':
            initial[0, 1] = .8
        weights, logits = initial.copy(), np.zeros((2, 2))
        initial_trait = initial[:, 0].copy()
        previous_shares = np.full((2, 2), .5)
        previous_trait = initial_trait.copy()
        for step in range(steps + 1):
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            shares = exp / exp.sum(axis=1, keepdims=True)
            signals = STIMULI @ weights.T
            factors = np.column_stack((np.ones(len(STIMULI)), STIMULI[:, 2]))
            outputs = (signals @ shares.T) * factors
            errors = outputs - READER_TARGETS
            # Loss = mean half-squared error over both readers and all inputs.
            d_mix = errors * factors / errors.size
            per_reader_dw = np.einsum('nr,ri,nj->rij', d_mix, shares, STIMULI)
            d_share = d_mix.T @ signals
            d_logits = shares * (d_share - (d_share * shares).sum(axis=1, keepdims=True))
            trait = weights[:, 0].copy()  # Continuous matched flower contrast.
            row = dict(condition=condition, step=step,
                       a_flower=float(trait[0]), b_flower=float(trait[1]),
                       a_green=float(weights[0, 1]), b_green=float(weights[1, 1]),
                       flower_a_share=float(shares[0, 0]), broccoli_a_share=float(shares[1, 0]),
                       **reader_metrics(outputs))
            trajectories.append(row)
            for group, indices in (('flower', [0]), ('broccoli', [1]), ('all', [0, 1])):
                mass = shares[indices].mean(axis=0)
                prior_mass = previous_shares[indices].mean(axis=0)
                for reference, base_mass, base_trait in (
                        ('initial', np.full(2, .5), initial_trait),
                        ('previous', prior_mass, previous_trait)):
                    # Weighted Price: w_i = mass_i/base_mass_i. Computing means
                    # directly avoids division by small allocation probabilities.
                    selection = float((mass - base_mass) @ base_trait)
                    variation = float(mass @ (trait - base_trait))
                    baseline, final = float(base_mass @ base_trait), float(mass @ trait)
                    accounting.append(dict(condition=condition, step=step, group=group,
                        reference=reference, baseline=baseline, selection=selection,
                        transmission=variation, final=final, delta=final-baseline,
                        identity_residual=final-baseline-selection-variation))
            for r, reader in enumerate(('flower', 'broccoli')):
                for i, route in enumerate(('A', 'B')):
                    pressures.append(dict(condition=condition, step=step, reader=reader, route=route,
                        flower_pressure=float(-per_reader_dw[r, i, 0]),
                        green_pressure=float(-per_reader_dw[r, i, 1]),
                        routing_logit_pressure=float(-d_logits[r, i]),
                        routing_update_applied=condition != 'routing_frozen' and step < steps,
                        feature_update_applied=step < steps))
            previous_shares, previous_trait = shares.copy(), trait.copy()
            if step < steps:
                weights -= lr * per_reader_dw.sum(axis=0)
                if condition != 'routing_frozen':
                    logits -= lr * d_logits
        endpoints[condition] = row
    return ExperimentResult(summary={'learned_routing': dict(
        steps=steps, learning_rate=lr, endpoints=endpoints,
        interpretation='Learned allocation shares weight persistent routes. Continuous flower contrast; '
                       'equal reader mass. Initial and adjacent-checkpoint weighted Price identities. '
                       'Precursor removal/rescue and frozen routing are controlled interventions.')},
        tables={'learned_routing_trajectories': trajectories,
                'learned_routing_price': accounting, 'learned_routing_pressure': pressures})
