"""Human-readable summaries of the suite's measured experiment results."""

from .results import SuiteResult


def format_report(result: SuiteResult) -> str:
    # Format stored measurements only; reporting never reruns training or accounting.
    summary, tables = result.summary, result.tables
    flow = summary["q1_development"]
    acquisition = {r["intervention"]: r for r in summary["q2_acquisition"]}
    worlds = summary["q3_matched_worlds"]
    identity = summary["q6_identity"]
    lines = [
        "FLOWER TOY EXPERIMENTS",
        "All stimuli, curricula, routing, and ancestry are synthetic.",
        "",
        "Development and competition",
        f"  Flower response: {flow['flower_before']:.3f} → {flow['flower_after']:.3f}.",
        f"  Broccoli-cue response: {flow['broccoli_cue_before']:.3f} → {flow['broccoli_cue_after']:.3f}.",
        f"  Established/alternative route contributions: {flow['established_final_contribution']:.3f} / "
        f"{flow['alternative_final_contribution']:.3f}.",
        "",
        "Precursor acquisition",
    ]
    for name, row in acquisition.items():
        lines.append(f"  {name}: {row['future_steps']} subsequent updates; "
                     f"{row['precursor_steps']} precursor updates.")
    policy = summary["acquisition_route_policy"]
    lines += [f"  Gibbs route probability for pretraining: {policy['probabilities']['green-pretrained']:.3f} "
              f"at temperature {policy['temperature']:g}.",
              "", "Acquisition with a capacity constraint"]
    for row in summary["capacity_acquisition"]:
        lines.append(f"  {row['condition']}: {row['subsequent_steps']} updates; "
                     f"flower response {row['flower_response']:.3f} at target {row['target_score']:.3f}.")
    lines += ["", "Matched worlds"]
    for name in ("shared", "separate"):
        row = worlds[name]
        lines.append(f"  {name}: {row['step']} updates; flower response {row['flower_response']:.3f}.")
    for target in sorted({r["confidence"] for r in summary["q3_criterion_sensitivity"]}):
        rows = {r["world"]: r for r in summary["q3_criterion_sensitivity"] if r["confidence"] == target}
        lines.append(f"  Target {target:.3f}: shared/separate {rows['shared']['step']}/{rows['separate']['step']} updates.")
    lines += ["", "Ancestry and Price accounting"]
    for row in summary["q4_price"]:
        if row["assay"] == "binary":
            lines.append(f"  {row['history']}: selection {row['selection']:+.3f}; "
                         f"transmission {row['transmission']:+.3f}; final mean {row['final']:.3f}.")
    for row in tables["reader_price_accounting"]:
        if row["history"] == "Retention" and row["population"] != "all":
            lines.append(f"  {row['assignment']} {row['population']} group: selection {row['selection']:+.3f}; "
                         f"transmission {row['transmission']:+.3f}.")
    lines += ["", "Reader-driven learning"]
    for assignment, condition in summary["downstream_reader_training"]["conditions"].items():
        final = condition["final"]
        lines.append(f"  {assignment}: flower/broccoli accuracy {final['flower_accuracy']:.0%}/"
                     f"{final['broccoli_accuracy']:.0%}; MSE {final['flower_mse']:.6g}/{final['broccoli_mse']:.6g}.")
    development = summary["development_comparisons"]
    lines += ["", "Continuity",
              f"  Retention/re-acquisition endpoint maximum difference: {identity['final_max_difference']:.2g}.",
              f"  Handoff endpoint maximum difference: {development['handoff_endpoint_max_difference']:.2g}."]
    for schedule, accuracy in development["minimum_accuracy"].items():
        lines.append(f"  {schedule} handoff minimum classification accuracy: {accuracy:.0%}.")
    lines += ["", "Interpretation",
              "  Activation, classification, and uninterrupted availability are separate measurements.",
              "  Price terms are conditional on registered ancestry and attributed uses.",
              "  Gibbs probabilities describe a declared routing policy; they do not infer ancestry.",
              "  Fixed-endpoint routing and reader-driven training are separate experiments."]
    return "\n".join(lines) + "\n"
