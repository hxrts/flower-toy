"""Functional dependence and threshold robustness beyond endpoint accounting."""
import unittest
from flower_toy.routing import routing_experiment
from flower_toy.development import handoff, handoff_thresholds


class DependencyChecks(unittest.TestCase):
    def test_zero_route_ablation_is_identity_before_training(self):
        result = routing_experiment(steps=0)
        rows = {r['removed_route']: r for r in result.tables['learned_routing_ablations']
                if r['condition'] == 'intact'}
        for key in rows['none']:
            if key not in ('condition', 'removed_route'):
                self.assertEqual(rows['none'][key], rows['B'][key])
        # Without A both outputs are zero on all eight inputs.
        self.assertEqual(rows['A']['flower_mse'], .5)
        self.assertEqual(rows['A']['broccoli_mse'], .25)

    def test_unablated_endpoint_and_selective_dependence(self):
        result = routing_experiment()
        rows = {r['removed_route']: r for r in result.tables['learned_routing_ablations']
                if r['condition'] == 'intact'}
        endpoint = result.summary['learned_routing']['endpoints']['intact']
        for reader in ('flower', 'broccoli'):
            for metric in ('mse', 'accuracy', 'max_error'):
                key = f'{reader}_{metric}'
                self.assertEqual(rows['none'][key], endpoint[key])
        self.assertEqual(rows['A']['flower_accuracy'], .5)
        self.assertEqual(rows['A']['broccoli_accuracy'], 1.)
        self.assertEqual(rows['B']['flower_accuracy'], 1.)
        self.assertEqual(rows['B']['broccoli_accuracy'], .75)

    def test_handoff_scores_match_geometric_decay(self):
        continuous, _, _ = handoff('continuous')
        interrupted, _, _ = handoff('interrupted')
        self.assertAlmostEqual(min(r['flower_score'] for r in continuous), .5)
        self.assertAlmostEqual(min(r['flower_score'] for r in interrupted), .5*.8**40)
        self.assertAlmostEqual(min(r['positive_margin'] for r in continuous), .05)
        self.assertTrue(all(r['negative_margin'] == .45 for r in continuous+interrupted))
        for row in handoff_thresholds(continuous+interrupted):
            if row['threshold'] < .5:
                if row['schedule'] == 'continuous':
                    self.assertEqual(row['unavailable_updates'], 0)
                else:
                    self.assertGreater(row['unavailable_updates'], 0)
