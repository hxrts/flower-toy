"""Write computed results and copy the article's generated assets."""

import csv
import json
import shutil
from pathlib import Path

from .figures import render_suite
from .report import format_report
from .results import SuiteResult


# Site builds consume diagram sources; measurements stay in this repository.
BLOG_ASSETS = (
    "acquisition_comparison.tex",
    "competition_gradients.tex",
    "competition_reader_gradients.tex",
    "downstream_reader_gradients.tex",
    "handoff_continuity.tex",
    "learned_routing.tex",
    "price_gradient_values.tex",
)

ARTICLE_DATA = (
    "acquisition_comparison.csv",
    "downstream_reader_endpoints.csv",
    "downstream_reader_gradients.csv",
    "downstream_reader_training.csv",
    "gradient_band_measurements.csv",
    "handoff_price.csv",
    "handoff_thresholds.csv",
    "handoff_trajectories.csv",
    "learned_routing_ablations.csv",
    "learned_routing_bands.csv",
    "learned_routing_pressure.csv",
    "learned_routing_price.csv",
    "learned_routing_trajectories.csv",
    "matched_reader_inheritance.csv",
    "matched_world_sensitivity.csv",
    "matched_worlds.csv",
    "reader_gradient_band_measurements.csv",
    "reader_price_accounting.csv",
    "signed_gradients.csv",
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


def export_article_data(out: Path, destination: Path) -> None:
    """Snapshot the measurements cited by the article, without figure sources."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ARTICLE_DATA:
        source, target = out / name, destination / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
