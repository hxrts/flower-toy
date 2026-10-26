"""Command-line interface for computing, writing, and exporting the suite."""

import argparse
from pathlib import Path

from .artifacts import export_article_data, export_blog_assets, write_suite
from .experiments import run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic developmental learning experiments and measured article figures.")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Directory for generated measurements and figures (default: results/)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Run numerical demonstrations without matplotlib")
    parser.add_argument("--blog-assets", type=Path,
                        help="Copy generated TikZ sources to this site asset directory")
    parser.add_argument("--article-data", type=Path,
                        help="Snapshot article measurement CSVs in this model repository")
    args = parser.parse_args()
    # The numerical suite is independent of output paths and presentation flags.
    result = run_suite()
    report = write_suite(result, args.output, plots=not args.no_plots)
    print(report, end="")
    if not args.no_plots:
        print(f"Saved figures and numeric report in: {args.output.resolve()}")
    print("\nExperiments complete. Results: " + str(args.output.resolve()))
    worlds = result.summary["q3_matched_worlds"]
    print("Matched worlds: shared %d steps, flower %.3f; separate %d steps, flower %.3f."
          % (worlds["shared"]["step"], worlds["shared"]["flower_response"],
             worlds["separate"]["step"], worlds["separate"]["flower_response"]))
    if args.blog_assets:
        export_blog_assets(args.output, args.blog_assets)
        print(f"Updated article figures in: {args.blog_assets.resolve()}")
    if args.article_data:
        export_article_data(args.output, args.article_data)
        print(f"Updated article measurements in: {args.article_data.resolve()}")
