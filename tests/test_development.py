"""Matched endpoints, reader-conditioned inheritance and continuous availability."""
import unittest
import numpy as np
from flower_toy.development import matched_inheritance, handoff

class DevelopmentTests(unittest.TestCase):
    def test_matched_global_accounting_and_readers(self):
        rows, difference = matched_inheritance()
        self.assertLess(difference, 1e-12)
        for history, expected in (
            ('Retention', {'flower': (.5, 0.), 'broccoli': (0., -.5)}),
            ('Mixed inheritance', {'flower': (0., .5), 'broccoli': (.5, -1.)})):
            groups = {r['population']: r for r in rows if r['history']==history}
            self.assertEqual(groups['all']['selection'], .25)
            self.assertEqual(groups['all']['transmission'], -.25)
            for group, (selection, variation) in expected.items():
                self.assertEqual(groups[group]['selection'], selection)
                self.assertEqual(groups[group]['transmission'], variation)
            for key in ('selection', 'transmission', 'final'):
                self.assertEqual(groups['all'][key], (groups['flower'][key]+groups['broccoli'][key])/2)

    def test_same_endpoints_different_availability(self):
        continuous, a, pa = handoff('continuous')
        interrupted, b, pb = handoff('interrupted')
        # Matching full states isolates continuity from endpoint response differences.
        np.testing.assert_array_equal(a,b)
        self.assertEqual(pa,pb)
        self.assertEqual(pa['selection'],0.)
        self.assertEqual(pa['transmission'],0.)
        self.assertTrue(all(r['flower_accuracy']==1. for r in continuous))
        self.assertEqual(interrupted[0]['flower_accuracy'],1.)
        self.assertEqual(interrupted[-1]['flower_accuracy'],1.)
        self.assertLess(min(r['flower_accuracy'] for r in interrupted),1.)
        for rows in (continuous,interrupted):
            # Each coordinate receives forty actual gradient updates.
            for coordinate in ('a_flower','b_flower'):
                changes = [(x[coordinate],y[coordinate]) for x,y in zip(rows,rows[1:])
                           if x[coordinate]!=y[coordinate]]
                self.assertEqual(len(changes),40)
                target = 0. if coordinate=='a_flower' else 1.
                for before,after in changes:
                    self.assertAlmostEqual(after,before-.2*(before-target))
