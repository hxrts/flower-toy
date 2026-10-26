"""Scientific invariants, counterfactuals, and numerical gradient checks."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from flower_toy import artifacts, experiments, measurements, model


class GradientTests(unittest.TestCase):
    def test_competition_gradient_against_finite_differences(self):
        a = np.array([.7, -.2])
        b = np.array([[.3, .1], [-.4, .6]])
        targets = np.array([1., -1.])
        _, da, db = model.competing_gradient(a, b, targets)
        # Central differences independently check both feature and connection derivatives.
        for param, analytic in ((a, da), (b, db)):
            for index in np.ndindex(param.shape):
                original = param[index]
                param[index] = original + 1e-6
                plus = model.competing_gradient(a, b, targets)[0]
                param[index] = original - 1e-6
                minus = model.competing_gradient(a, b, targets)[0]
                param[index] = original
                self.assertAlmostEqual((plus - minus) / 2e-6, analytic[index], places=8)

    def test_pressure_alignment_and_residual_competition(self):
        warm = model.competition_trace(True)
        cold = model.competition_trace(False)
        conflict = model.competition_trace(True, True)
        self.assertAlmostEqual(warm[0]['gradient_cosine'], 1.)
        self.assertAlmostEqual(conflict[0]['gradient_cosine'], -1.)
        self.assertLess(warm[-1]['loss'], 1e-10)
        self.assertLess(warm[-1]['alternative_contribution'], cold[-1]['alternative_contribution'])
        self.assertLess(warm[-1]['alternative_pressure'], warm[0]['alternative_pressure'])
        # Actual reader weights, not only upstream feature coefficients, learned.
        self.assertGreater(warm[-1]['reader_00'], warm[0]['reader_00'])

    def test_signed_reader_gradients_and_actual_updates(self):
        rows = model.competition_trace(True, conflicting=True)
        for current, following in zip(rows, rows[1:]):
            for i in range(2):
                parts = [current[f'reader_{r}_feature_gradient_{i}'] for r in range(2)]
                self.assertAlmostEqual(sum(parts), current[f'feature_gradient_{i}'])
                self.assertAlmostEqual(following[f'feature_{i}'] - current[f'feature_{i}'],
                                       -.15 * sum(parts))
        parts = [rows[0][f'reader_{r}_feature_gradient_0'] for r in range(2)]
        self.assertLess(abs(sum(parts)), sum(map(abs, parts)))
        self.assertFalse(rows[-1]['update_applied'])

    def test_projection_is_not_mislabeled_as_a_flower_gradient(self):
        rows = model.matched_route_training(True, target_confidence=.975)
        for current, following in zip(rows, rows[1:]):
            self.assertEqual(current['flower_loss_gradient'], 0.)
            self.assertEqual(current['flower_gradient_delta'], 0.)
            self.assertAlmostEqual(following['flower_response'] - current['flower_response'],
                                   current['flower_projection_delta'])
            self.assertAlmostEqual(current['green_total_delta'],
                                   current['green_gradient_delta'] + current['green_projection_delta'])
            self.assertAlmostEqual(following['green_response'] - current['green_response'],
                                   current['green_total_delta'])
        self.assertLess(sum(r['flower_projection_delta'] for r in rows), -.1)
        self.assertEqual(rows[-1]['flower_total_delta'], 0.)
        g = rows[0]['green_response']
        def loss(value):
            logits = 8 * (value * model.STIMULI[:, 1] + model.STIMULI[:, 2] - 1.5)
            return np.mean(np.logaddexp(0, logits) - model.READER_TARGETS[:, 1] * logits)
        self.assertAlmostEqual((loss(g+1e-6) - loss(g-1e-6)) / 2e-6,
                               rows[0]['green_loss_gradient'], places=8)

    def test_precursor_lesion_and_rescue(self):
        warm = model.train_green(model.FLOWER)
        interventions = model.precursor_interventions(warm)
        lesioned = interventions['precursor_removed']
        rescued = interventions['precursor_rescued']
        np.testing.assert_array_equal(lesioned[[0, 2]], warm[[0, 2]])
        np.testing.assert_array_equal(rescued, warm)
        self.assertFalse(np.shares_memory(lesioned, warm))
        self.assertEqual(model.flower_trait(warm), model.flower_trait(lesioned))
        cold = model.train_broccoli(0)
        self.assertEqual(model.train_broccoli(lesioned[1]).steps, cold.steps)
        self.assertEqual(model.train_broccoli(rescued[1]).steps,
                         model.train_broccoli(warm[1]).steps)
        self.assertLess(model.train_broccoli(warm[1]).steps, cold.steps)

    def test_matched_worlds_and_criterion_reversal(self):
        a, b = (model.matched_route_training(shared) for shared in (True, False))
        self.assertEqual(a[0]['flower_response'], b[0]['flower_response'])
        self.assertTrue(a[-1]['acquired'] and b[-1]['acquired'])
        self.assertLess(a[-1]['step'], b[-1]['step'])
        self.assertLess(a[-1]['flower_response'], b[-1]['flower_response'])
        for rows in (a, b):
            self.assertTrue(all(r['flower_accuracy'] == 1. for r in rows))
            self.assertTrue(all(r['flower_responsive'] for r in rows))
        strict_a, strict_b = (model.matched_route_training(s, target_confidence=.975)
                              for s in (True, False))
        self.assertGreater(strict_a[-1]['step'], strict_b[-1]['step'])
        self.assertEqual(strict_a[-1]['flower_accuracy'], 1.)
        self.assertGreater(strict_a[-1]['flower_max_error'], .1)

    def test_failure_is_reported_without_claiming_acquisition(self):
        self.assertFalse(model.matched_route_training(False, max_steps=0)[-1]['acquired'])
        with self.assertRaises(RuntimeError):
            model.train_broccoli(0., max_steps=0)


class PriceTests(unittest.TestCase):
    def test_random_fractional_credit_and_reader_reconstruction(self):
        rng = np.random.default_rng(8)
        for _ in range(40):
            baseline = rng.normal(size=5)
            later = rng.normal(size=9)
            credit = rng.dirichlet(np.ones(4), size=9)
            # Ancestor five gets zero credit; one descendant is unassigned.
            credit = np.column_stack((credit, np.zeros(9)))
            credit[-1] = 0
            p = model.attributed_price(baseline, later, credit)
            a = model.attributed_price(baseline, later[:3], credit[:3])
            b = model.attributed_price(baseline, later[3:], credit[3:])
            self.assertAlmostEqual(p['delta'], p['selection'] + p['transmission'])
            # The second group has five assigned uses; its sixth use is excluded.
            for key in ('selection', 'transmission', 'final'):
                self.assertAlmostEqual(p[key], (3 * a[key] + 5 * b[key]) / 8)
            # Changing an unassigned use cannot affect the included population.
            later[-1] += 100
            self.assertEqual(p, model.attributed_price(baseline, later, credit))

    def test_permutation_invariance_and_equal_coordinate_control(self):
        z = np.array([1., 0., .4])
        later = np.array([.8, .2, .9])
        credit = np.array([[.3, .7, 0], [1, 0, 0], [0, .5, .5]])
        p = model.attributed_price(z, later, credit)
        perm = [2, 0, 1]
        q = model.attributed_price(z[perm], later[perm], credit[perm][:, perm])
        for key in p:
            self.assertAlmostEqual(p[key], q[key])
        self.assertEqual(model.attributed_price(z, later, np.eye(3))['selection'], 0.)

    def test_invalid_credit(self):
        for credit in (np.zeros((2, 2)), np.array([[1., -1.], [0, 1]]),
                       np.ones((2, 2)), np.array([[np.nan, 0], [0, 1]])):
            with self.assertRaises(ValueError):
                model.attributed_price(np.array([1., 0.]), np.array([.4, .5]), credit)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        ancestors = dict(A=model.train_green(model.FLOWER), B=model.TEXTURE.copy())
        self.histories = [
            model.create_history('Retention', ancestors, dict(F1='A', F2='A', G='A', T='B')),
            model.create_history('Re-acquisition', ancestors, dict(F1='B', F2='B', G='A', T='B'))]

    def test_article_binary_accounting_and_all_checkpoint_identities(self):
        for history, selection in zip(self.histories, (.25, -.25)):
            for step in range(191):
                w, c = model.history_arrays(history, step)
                p = model.attributed_price(np.array([1., 0.]), model.response_assay(w, True), c)
                self.assertEqual(p['selection'], selection)
                self.assertAlmostEqual(p['selection'] + p['transmission'], p['delta'])
            self.assertEqual(p['final'], .5)
            self.assertEqual(p['transmission'], -selection)

    def test_all_inputs_reader_ablations_and_distinct_histories(self):
        endpoints = []
        for h in self.histories:
            w, _ = model.history_arrays(h)
            outputs = model.reader_outputs(w, model.GOOD_READERS)
            np.testing.assert_allclose(outputs, model.READER_TARGETS, atol=1e-14)
            endpoints.append(outputs)
            for role in ('G', 'T'):
                ablated = model.reader_outputs(w, model.GOOD_READERS, role)
                np.testing.assert_allclose(ablated[:, 0], outputs[:, 0])
                self.assertGreater(np.mean((ablated[:, 1] - outputs[:, 1])**2), .2)
            wrong = model.reader_outputs(w, model.MISROUTED_READERS)
            self.assertGreater(model.reader_metrics(wrong)['flower_mse'], .1)
            self.assertGreater(model.reader_metrics(wrong)['broccoli_mse'], .1)
        np.testing.assert_allclose(*endpoints, atol=1e-14)
        initial = [model.reader_outputs(model.history_arrays(h, 0)[0], model.GOOD_READERS)
                   for h in self.histories]
        self.assertGreater(np.max(np.abs(initial[0] - initial[1])), .5)

    def test_full_export(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            result = experiments.run_suite()
            artifacts.write_suite(result, Path(d), plots=False)
            summary = result.summary
            saved = json.loads((Path(d) / 'summary.json').read_text())
            self.assertEqual(summary, saved)
            self.assertAlmostEqual(saved['ensemble']['probability_sum'], 1.)
            self.assertEqual(saved['ensemble']['assignments'], 16)
            self.assertLess(saved['q6_identity']['final_max_difference'], 1e-12)
            with (Path(d) / 'reader_truth_tables.csv').open() as fh:
                truth = list(csv.DictReader(fh))
            self.assertEqual(len(truth), 2 * 2 * 8)
            with (Path(d) / 'price_by_checkpoint.csv').open() as fh:
                accounting = list(csv.DictReader(fh))
            self.assertEqual(len(accounting), 2 * 191 * 2 * 2 * 3)
            with (Path(d) / 'reader_price_accounting.csv').open() as fh:
                fixed = list(csv.DictReader(fh))
            self.assertEqual(len(fixed), 12)
            for assignment, expected in (
                    ('coherent', {'flower': (.5, 0., 1.), 'broccoli': (0., -.5, 0.)}),
                    ('misrouted', {'flower': (0., -.5, 0.), 'broccoli': (.5, 0., 1.)})):
                rows = {r['population']: r for r in fixed
                        if r['history'] == 'Retention' and r['assignment'] == assignment}
                for group, values in expected.items():
                    self.assertEqual(tuple(float(rows[group][k]) for k in
                                           ('selection', 'transmission', 'final')), values)
                for key in ('selection', 'transmission', 'final'):
                    self.assertEqual(float(rows['all'][key]),
                                     (float(rows['flower'][key]) + float(rows['broccoli'][key])) / 2)
            with (Path(d) / 'signed_gradients.csv').open() as fh:
                gradients = list(csv.DictReader(fh))
            for h in self.histories:
                for role in model.ALL_ROLES:
                    values = [float(r['gradient_update']) for r in gradients
                              if r['experiment'] == 'specialization' and r['condition'] == h.name
                              and r['source'] == role and r['parameter'] == 'flower']
                    trajectory = h.uses[role].trajectory
                    self.assertAlmostEqual(sum(values), trajectory[-1, 0] - trajectory[0, 0])
            bands = saved['gradient_bands']
            for row in bands:
                self.assertAlmostEqual(row['full_width_pt'],
                                       row['points_per_gradient_unit'] * row['mean_absolute_gradient'])
            price_bands = {r['source']: r for r in bands if r['experiment'] == 'specialization'}
            self.assertEqual(price_bands['F1']['full_width_pt'], 0.)
            self.assertEqual(price_bands['F2']['full_width_pt'], 0.)
            self.assertEqual(price_bands['T']['full_width_pt'], 0.)
            self.assertLess(price_bands['G']['mean_signed_pressure'], 0.)
            self.assertTrue((Path(d) / 'competition_gradients.tex').exists())

    def test_band_width_preserves_pressure_despite_temporal_cancellation(self):
        rows = [dict(experiment='test', condition='test', source='test', parameter='x',
                     step=i, update_applied=True, loss_gradient=g, descent_pressure=-g)
                for i, g in enumerate((1., -1.))]
        band = measurements.gradient_band_summary(rows, experiment='test', condition='test',
                                           source='test', parameter='x', start=0, stop=2, scale=10.)
        self.assertEqual(band['mean_signed_pressure'], 0.)
        self.assertEqual(band['full_width_pt'], 10.)
        with self.assertRaises(ValueError):
            measurements.band_taper_sign(band)

    def test_sustained_availability_does_not_confuse_recovery_with_continuity(self):
        rows = [dict(step=i, flower_accuracy=accuracy, flower_max_error=error)
                for i, (accuracy, error) in enumerate(((1., .01), (.75, .3), (1., .1), (1., .01)))]
        measured = model.sustained_availability(rows, 'flower')
        self.assertFalse(measured['classification']['available_at_every_checkpoint'])
        self.assertEqual(measured['classification']['first_sustained_step'], 2)
        self.assertEqual(measured['response_precision']['first_sustained_step'], 3)
        self.assertIsNone(model.sustained_availability(rows[:2], 'flower')
                          ['classification']['first_sustained_step'])
        self.assertTrue(model.sustained_availability(rows[:1], 'flower')
                        ['classification']['available_at_every_checkpoint'])
        with self.assertRaises(ValueError):
            model.sustained_availability([], 'flower')


class DownstreamReaderTests(unittest.TestCase):
    def test_each_reader_gradient_against_finite_differences(self):
        rng = np.random.default_rng(12)
        weights = rng.normal(.2, .3, (4, 3))
        gains = np.array([.7, 1.1, .9, .6])
        for mapping in (model.GOOD_READERS, model.MISROUTED_READERS):
            _, _, dw, dc = model.downstream_loss_gradients(weights, gains, mapping)
            for parameter, expected in ((weights, dw), (gains, dc)):
                for index in np.ndindex(parameter.shape):
                    original = parameter[index]
                    parameter[index] = original + 1e-6
                    plus = model.downstream_loss_gradients(weights, gains, mapping)[1] / 2
                    parameter[index] = original - 1e-6
                    minus = model.downstream_loss_gradients(weights, gains, mapping)[1] / 2
                    parameter[index] = original
                    np.testing.assert_allclose((plus-minus)/2e-6, expected[(slice(None),)+index],
                                               rtol=1e-6, atol=1e-9)

    def test_reader_losses_drive_actual_feature_and_connection_updates(self):
        ancestors = dict(A=model.train_green(model.FLOWER), B=model.TEXTURE.copy())
        history = model.create_history('Retention', ancestors, dict(F1='A', F2='A', G='A', T='B'))
        initial = model.history_arrays(history, 0)[0]
        for mapping in (model.GOOD_READERS, model.MISROUTED_READERS):
            trace, gradients, weights, gains = model.train_downstream_readers(history, mapping)
            self.assertLess(trace[-1]['loss'], trace[0]['loss'] / 20)
            self.assertEqual(trace[-1]['flower_accuracy'], 1.)
            self.assertEqual(trace[-1]['broccoli_accuracy'], 1.)
            for i, role in enumerate(model.ALL_ROLES):
                for j, coordinate in enumerate((*model.FEATURES, 'read_gain')):
                    total = sum(r['gradient_update'] for r in gradients
                                if r['use'] == role and r['coordinate'] == coordinate)
                    actual = gains[i]-1 if j == 3 else weights[i,j]-initial[i,j]
                    self.assertAlmostEqual(total, actual, places=10)
            if mapping == model.GOOD_READERS:
                signs = {(r['reader'],r['use'],r['coordinate']):r['descent_pressure']
                         for r in gradients if r['step']==10}
                self.assertLess(signs['flower','F1','green'], 0.)
                self.assertGreater(signs['broccoli','G','green'], 0.)


if __name__ == '__main__':
    unittest.main()
