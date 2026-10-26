"""Separation between computation, measurements, and optional output."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flower_toy import artifacts, experiments, measurements
from flower_toy.development import handoff
from flower_toy.figures import write_handoff_diagram


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Scientific results must be usable without filesystem or renderer access.
        cls.stdout = io.StringIO()
        with contextlib.redirect_stdout(cls.stdout), \
                patch.object(Path, "open", side_effect=AssertionError("Experiment attempted file I/O")), \
                patch.object(Path, "write_text", side_effect=AssertionError("Experiment attempted a write")), \
                patch.object(artifacts, "render_suite", side_effect=AssertionError("Experiment attempted rendering")):
            cls.result = experiments.run_suite()

    def test_suite_provides_measurements_without_output(self):
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertTrue(self.result.tables["gradient_band_measurements"])
        self.assertTrue(self.result.tables["reader_gradient_band_measurements"])
        self.assertEqual(self.result.summary["gradient_bands"],
                         self.result.tables["gradient_band_measurements"])
        self.assertTrue(self.result.summary["capacity_acquisition"])

    def test_capacity_comparison_preserves_learning_and_exports_endpoints(self):
        trials = {r["condition"]: r for r in self.result.tables["capacity_acquisition"]}
        self.assertLess(trials["warm"]["subsequent_steps"], trials["cold"]["subsequent_steps"])
        self.assertGreater(trials["warm_capacity"]["subsequent_steps"], trials["warm"]["subsequent_steps"])
        self.assertEqual(trials["warm"]["flower_response"], 1.)
        self.assertLess(trials["warm_capacity"]["flower_response"], 1.)
        for condition, trial in trials.items():
            trace = [r for r in self.result.tables["capacity_acquisition_trajectories"]
                     if r["condition"] == condition]
            self.assertEqual(len(trace), trial["subsequent_steps"] + 1)
            self.assertGreaterEqual(trial["positive_score"], trial["target_score"])
            self.assertLess(trial["green_false_positive"], .1)
            for coordinate in ("flower_response", "green_response", "positive_score"):
                self.assertEqual(trace[-1][coordinate], trial[coordinate])
            if trial["capacity"] is not None:
                for row in trace:
                    self.assertLessEqual(row["flower_response"]**2 + row["green_response"]**2,
                                         trial["capacity"]**2 + 1e-12)

    def test_export_uses_stored_measurements_and_preserves_results(self):
        # Rendering and export must neither retrain nor mutate the reusable result.
        original = copy.deepcopy((self.result.summary, self.result.tables))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "results"
            destination = Path(directory) / "article"
            with patch.object(experiments, "run_suite", side_effect=AssertionError("Export reran suite")), \
                    patch.object(measurements, "gradient_band_summary", side_effect=AssertionError("Export remeasured bands")), \
                    patch.object(experiments, "capacity_acquisition_experiment", side_effect=AssertionError("Export reran acquisition")):
                artifacts.write_suite(self.result, out, plots=False)
                artifacts.export_blog_assets(out, destination)
                artifacts.export_blog_assets(out, out)
            self.assertFalse(list(out.glob("*.png")))
            self.assertEqual({p.name for p in destination.iterdir()}, set(artifacts.BLOG_ASSETS))
            for name in artifacts.BLOG_ASSETS:
                self.assertEqual((out / name).read_bytes(), (destination / name).read_bytes())
        self.assertEqual((self.result.summary, self.result.tables), original)

    def test_handoff_figure_uses_recorded_training_length(self):
        traces = [row for schedule in ("continuous", "interrupted")
                  for row in handoff(schedule, phase_steps=3)[0]]
        with tempfile.TemporaryDirectory() as directory:
            write_handoff_diagram(Path(directory), traces)
            source = (Path(directory) / "handoff_continuity.tex").read_text()
        self.assertIn("Update 3", source)
        self.assertIn("Update 6", source)
