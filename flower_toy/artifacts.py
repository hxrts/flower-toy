"""Write computed results and copy the article's generated assets."""

import csv
import json
import shutil
from pathlib import Path

from .figures import render_suite
from .report import format_report
from .results import SuiteResult


# Export only article inputs; standalone reports and plots stay in the output directory.
BLOG_ASSETS = (
    "acquisition_comparison.tex", "acquisition_comparison.csv",
    "competition_gradients.tex", "price_gradient_values.tex",
    "signed_gradients.csv", "gradient_band_measurements.csv",
    "reader_price_accounting.csv", "matched_reader_inheritance.csv",
    "handoff_trajectories.csv", "handoff_price.csv", "handoff_continuity.tex",
    "matched_worlds.csv", "matched_world_sensitivity.csv",
    "competition_reader_gradients.tex", "downstream_reader_gradients.tex",
    "reader_gradient_band_measurements.csv", "downstream_reader_training.csv",
    "downstream_reader_gradients.csv", "downstream_reader_endpoints.csv",
)


def write_records(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        # Each measurement table establishes its column order with its first record.
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_suite(result: SuiteResult, out: Path, *, plots: bool = True) -> str:
    """Export an in-memory suite; optional plots never control measurements."""
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in result.tables.items():
        write_records(out / f"{name}.csv", rows)
    (out / "summary.json").write_text(json.dumps(result.summary, indent=2) + "\n")
    report = format_report(result)
    (out / "results.txt").write_text(report, encoding="utf-8")
    render_suite(result, out, plots=plots)
    return report


def export_blog_assets(out: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in BLOG_ASSETS:
        source, target = out / name, destination / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
