"""Matplotlib and TikZ figures derived from experiment measurements."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from .model import (
    ALL_ROLES,
    GOOD_READERS,
    MISROUTED_READERS,
)

from .measurements import (
    band_taper_sign,
    index_bands,
)

from .settings import BANDS, TRAINING
from .results import SuiteResult


def _pyplot():
    """Load the plotting dependency only when plots are requested."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _write_tikz(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reader_header(comment: str, headings: tuple[tuple[str, str], ...]) -> list[str]:
    return [comment, r"\begin{tikzpicture}[learning]", r"\def\learningheadinglift{.6}",
            r"\tikzset{reader/.style={rectangle,draw,fill=white,minimum size=6pt,inner sep=0pt,line width=.65pt}}",
            *[fr"\stageheading{{{x}}}{{{title}}}{{{subtitle}}}"
              for x, (title, subtitle) in zip((0, 4.5, 9), headings)]]


def _reader_footer(pathway_label: str) -> list[str]:
    return [r"\coordinate (learningannotationbottom) at (0,-7.7275);",
            r"\preparelearningfooter{1.95}",
            fr"\node at (2.25,\learningaxisy) {{{pathway_label}}};",
            r"\node at (6.75,\learningaxisy) {Connection gradients};",
            r"\pressurekey{0}{\learninglegendy}{1}",
            r"\node[legendlabel] at (.7,\learninglegendy) {Increasing parameter};",
            r"\pressurekey{5}{\learninglegendy}{-1}",
            r"\node[legendlabel] at (5.7,\learninglegendy) {Decreasing parameter};",
            r"\diagramframe{\learningframebottom}", r"\end{tikzpicture}"]


def _pressure_band(target: str, start: str, first: str, last: str, band: dict) -> list[str]:
    sign = band_taper_sign(band)
    if not sign:
        return []
    command = "pressureband" if sign > 0 else "suppressband"
    return [fr"\{command}{{{target}}}{{{last}}}{{{first}}}{{{start}}}{{{band['full_width_pt']/2:.8f}}}"]


def plot_lineage(out: Path, tables: dict[str, list[dict]]) -> None:
    """Draw endpoint ancestry from the exported trajectory and accounting records."""
    plt = _pyplot()
    trajectories = tables["learning_trajectories"]
    histories = list(dict.fromkeys(r["history"] for r in trajectories))
    final_step = max(r["gradient_step"] for r in trajectories)
    endpoints = {name: {r["use"]: r for r in trajectories
                        if r["history"] == name and r["gradient_step"] == final_step}
                 for name in histories}
    prices = {r["history"]: r for r in tables["price_decompositions"]
              if r["reader_assignment"] == "coherent" and r["population"] == "all"}
    # A legible ancestry view for the two histories, deliberately independent
    # from the endpoint's consumer-facing implementation.
    fig, axarr = plt.subplots(1, 2, figsize=(14, 5.1), sharex=True, sharey=True)
    for ax, history in zip(axarr, histories):
        ax.set_xlim(0, 10)
        ax.set_ylim(-.5, 5.1)
        ax.axis("off")
        ancestor_y = {"A": 3.5, "B": 1.0}
        roles_y = {"F1": 4.1, "F2": 3.0, "G": 1.8, "T": .7}
        for parent, yy in ancestor_y.items():
            ax.scatter([.8], [yy], s=800, facecolors="white", edgecolors="black", zorder=4)
            ax.text(.8, yy, parent, ha="center", va="center", fontsize=10, zorder=5)
        for role, yy in roles_y.items():
            use = endpoints[history][role]
            ax.plot([1.1, 4.1], [ancestor_y[use["parent"]], yy],
                    lw=1.7, alpha=.7)
            is_f = use["flower_response"] > TRAINING.decision_threshold
            ax.scatter([4.3], [yy], s=590, facecolors="white" if is_f else "0.8",
                       edgecolors="black", linewidths=1.7, zorder=4)
            ax.text(4.3, yy, role, ha="center", va="center", zorder=5, fontsize=9)
            consumer = "Flower reader" if role in GOOD_READERS.flower else "Broccoli reader"
            consumer_y = 3.55 if consumer.startswith("Flower") else 1.25
            ax.plot([4.65, 7.1], [yy, consumer_y], alpha=.65, lw=1.7)
        for name, y in (("Flower reader", 3.55), ("Broccoli reader", 1.25)):
            ax.text(8.2, y, name, ha="center", va="center", fontsize=9,
                    bbox={"boxstyle": "round,pad=.4", "fc": "white", "ec": "black"})
        pp = prices[history]
        ax.text(5.0, 4.82, f"Selection {pp['selection']:+.2f}; transmission {pp['transmission']:+.2f}",
                ha="center", fontsize=10)
        ax.set_title(history)
    fig.suptitle("Same consumer-facing endpoint; different learned ancestry\n"
                 "Outlined later uses retain the flower trait", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .86))
    fig.savefig(out / "lineage_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_gradient_diagrams(out: Path, bands: list[dict], competition: list[dict]) -> None:
    """Generate measured bands; black arrows remain ancestry/continuity markers."""
    warm = {r["step"]: r for r in competition if r["head_start"] and not r["conflicting"]}
    measured = index_bands(bands)
    tex = [r"% Generated measurements: signed pressure = -dL/da; widths use mean absolute gradient.",
           r"\begin{tikzpicture}[learning]",
           fr"\stageheading{{0}}{{INITIAL}}{{step {BANDS.competition_intervals[0][0]}}}",
           fr"\stageheading{{4.5}}{{EARLY LEARNING}}{{step {BANDS.competition_intervals[0][1]}}}",
           fr"\stageheading{{9}}{{LATER LEARNING}}{{step {BANDS.competition_intervals[1][1]}}}",
           r"\foreach \x in {0,4.5,9} { \stageoval{\x,0} }"]
    for col, step in enumerate((BANDS.competition_intervals[0][0],
                                *(stop for _, stop in BANDS.competition_intervals))):
        for route in range(2):
            y = .65 if route == 0 else -.65
            tex.append(fr"\node[feature] (n{col}{route}) at ({col*4.5},{y}) {{}};")
            tex.append(fr"\abovenodelabel{{n{col}{route}}}{{${warm[step][f'feature_{route}']:.2f}$}}")
    tex.append(r"\begin{pgfonlayer}{background}")
    arrows = []
    for col, (start, stop) in enumerate(BANDS.competition_intervals):
        for route in range(2):
            band = measured["competition", "warm_compatible", "all_readers", f"feature_{route}", start]
            tex += [fr"\coordinate (start{col}{route}) at ($(n{col}{route}.center)+(.42,0)$);",
                    fr"\coordinate (cfirst{col}{route}) at ($(n{col}{route}.center)+(1.62,0)$);",
                    fr"\coordinate (clast{col}{route}) at ($(n{col+1}{route}.center)+(-1.35,0)$);"]
            tex += _pressure_band(f"n{col+1}{route}.center", f"start{col}{route}",
                                  f"cfirst{col}{route}", f"clast{col}{route}", band)
            arrows.append(fr"\draw[flow] (n{col}{route}.east) -- (n{col+1}{route}.west);")
    # Draw all fills first so crossing bands cannot cover the black signal arrows.
    tex += arrows
    tex.append(r"\end{pgfonlayer}")
    for col in range(2):
        for route in range(2):
            band = bands[2*col+route]
            x = 2.25 + 4.5 * col
            y = .15 if route == 0 else -1.15
            value = band["mean_signed_pressure"]
            tex.append(r"\node[inner sep=2pt] at (" + f"{x},{y}" + r") {$p$=\textsf{" + f"{value:+.3f}" + r"}};")
    tex += [r"\leftlabel{.65}{Established\\green route}", r"\leftlabel{-.65}{Alternative route}",
            r"\rightlabel{0}{.65}{Green coefficient}", r"\rightlabel{0}{-.65}{Green coefficient}",
            r"\coordinate (learningannotationbottom) at (0,-2.1275);",
            r"\preparelearningfooter{1.95}", r"\developmentaxis{\learningaxisy}",
            r"\pressurekey{0}{\learninglegendy}{1}",
            r"\node[legendlabel] at (.7,\learninglegendy) {Reinforcing (positive)};",
            r"\pressurekey{5}{\learninglegendy}{-1}",
            r"\node[legendlabel] at (5.7,\learninglegendy) {Suppressing (negative)};",
            r"\diagramframe{\learningframebottom}", r"\end{tikzpicture}"]
    _write_tikz(out / "competition_gradients.tex", tex)
    price = [f"% Generated retention-history flower gradients, steps 0 <= t < {TRAINING.specialization_steps}."]
    for role, macro in (("F1", "Fone"), ("F2", "Ftwo"), ("G", "Green"), ("T", "Texture")):
        band = measured["specialization", "Retention", role, "flower", 0]
        price.append(fr"\def\{macro}BandWidth{{{band['full_width_pt']:.8f}pt}}")
        price.append(fr"\def\{macro}BandHalfWidth{{{band['full_width_pt']/2:.8f}}}")
        price.append(fr"\def\{macro}BandSign{{{band_taper_sign(band)}}}")
        price.append(fr"\def\{macro}Pressure{{{band['mean_signed_pressure']:+.3f}}}")
    _write_tikz(out / "price_gradient_values.tex", price)


def tikz_measured_link(source: str, target: str, band: dict, tag: str) -> tuple[list[str], str]:
    """Return band geometry and arrow separately, so every fill precedes every arrow.

    Shared macros use opaque black!40 fill.
    Both passes occupy the background layer below the nodes and labels.
    """
    lines = [fr"\coordinate ({tag}s) at ($({source}.center)+(.42,0)$);",
             fr"\coordinate ({tag}a) at ($({source}.center)+(1.62,0)$);",
             fr"\coordinate ({tag}b) at ($({target}.center)+(-1.35,0)$);"]
    lines += _pressure_band(f"{target}.center", f"{tag}s", f"{tag}a", f"{tag}b", band)
    arrow = fr"\draw[flow] ({source}.center) .. controls ({tag}a) and ({tag}b) .. ({target}.west);"
    return lines, arrow


def write_reader_gradient_diagrams(out: Path, bands: list[dict],
                                   competition: list[dict]) -> None:
    measured = index_bands(bands)
    tex = _reader_header(f"% Generated reader-specific gradients at checkpoint {BANDS.contribution_checkpoint}.", (
        ("SHARED FEATURES", "net pressure"), ("READ WEIGHTS", "learned weights"),
        ("READERS", f"step {BANDS.contribution_checkpoint}")))
    for panel, conflicting in enumerate((False, True)):
        base = -5.6 * panel
        condition = "warm_conflicting" if conflicting else "warm_compatible"
        current = next(r for r in competition if r["head_start"] and r["conflicting"] == conflicting and r["step"] == BANDS.contribution_checkpoint)
        label = "OPPOSED" if conflicting else "ALIGNED"
        tex.append(fr"\pairedrowlabel{{{base}}}{{{label}}}")
        tex.append(fr"\foreach \x in {{0,4.5,9}} {{ \stageoval{{\x,{base}}} }}")
        for i in range(2):
            y = base + (.65 if i == 0 else -.65)
            tex.append(fr"\node[feature] (a{panel}{i}) at (0,{y}) {{}};")
            tex.append(fr"\abovenodelabel{{a{panel}{i}}}{{$a_{i}$}}")
            tex.append(fr"\leftrowlabel{{{base}}}{{{y-base}}}{{Net $p={-current[f'feature_gradient_{i}']:+.3f}$}}")
            tex.append(fr"\node[reader] (r{panel}{i}) at (9,{y}) {{}};")
            target = -1 if conflicting and i == 1 else 1
            tex.append(fr"\rightlabel{{{base}}}{{{y-base}}}{{Reader {i+1}\\Target ${target:+d}$}}")
        for i, reader in itertools.product(range(2), repeat=2):
            y = base + 1.5 - (i*2+reader)
            tex.append(fr"\node[feature] (b{panel}{reader}{i}) at (4.5,{y}) {{}};")
            tex.append(fr"\abovenodelabel{{b{panel}{reader}{i}}}{{$b_{{{reader}{i}}}$}}")
        tex.append(r"\begin{pgfonlayer}{background}")
        arrows = []
        for i, reader in itertools.product(range(2), repeat=2):
            for side, parameter in (("feature", f"feature_{i}"),
                                    ("read", f"read_weight_{reader}_{i}")):
                band = measured["competition", condition, f"reader_{reader}", parameter,
                                BANDS.contribution_checkpoint]
                src, dst = ((f"a{panel}{i}", f"b{panel}{reader}{i}") if side == "feature" else
                            (f"b{panel}{reader}{i}", f"r{panel}{reader}"))
                geometry, arrow = tikz_measured_link(src, dst, band, f"c{panel}{i}{reader}{side}")
                tex += geometry
                arrows.append(arrow)
        # Keep the same fill-then-arrow layering in both reader panels.
        tex += arrows
        tex.append(r"\end{pgfonlayer}")
    tex += _reader_footer("Feature gradients")
    _write_tikz(out / "competition_reader_gradients.tex", tex)

    tex = _reader_header(f"% Generated gradients from actual downstream reader losses, checkpoint {BANDS.reader_checkpoint}.", (
        ("PATHWAYS", "green response"), ("CONNECTIONS", "read gains"), ("READERS", f"step {BANDS.reader_checkpoint}")))
    for panel, (assignment, mapping, label) in enumerate((
            ("coherent", GOOD_READERS, "COHERENT"),
            ("misrouted", MISROUTED_READERS, "SWAPPED"))):
        base = -5.6 * panel
        tex += [fr"\pairedrowlabel{{{base}}}{{{label}}}",
                fr"\foreach \x in {{0,4.5,9}} {{ \stageoval{{\x,{base}}} }}"]
        for i, role in enumerate(ALL_ROLES):
            y = base + 1.5 - i
            tex.append(fr"\node[feature] (w{panel}{i}) at (0,{y}) {{}};")
            tex.append(fr"\node[feature] (c{panel}{i}) at (4.5,{y}) {{}};")
            ancestry_label = ("A_1", "A_2", "A_3", "B_1")[i]
            tex.append(fr"\abovenodelabel{{w{panel}{i}}}{{${ancestry_label}$}}")
            tex.append(fr"\abovenodelabel{{c{panel}{i}}}{{$c_{{{ancestry_label}}}$}}")
        for reader, y in (("flower", base+.7), ("broccoli", base-.7)):
            tex.append(fr"\node[reader] (r{panel}{reader}) at (9,{y}) {{}};")
            tex.append(fr"\rightlabel{{{base}}}{{{y-base}}}{{{reader.capitalize()}\\reader}}")
        tex.append(r"\begin{pgfonlayer}{background}")
        arrows = []
        for i, role in enumerate(ALL_ROLES):
            reader = "flower" if role in mapping.flower else "broccoli"
            for side, name in (("path", "green"), ("gain", "read_gain")):
                band = measured["downstream_training", assignment, reader, role+"_"+name,
                                BANDS.reader_checkpoint]
                src, dst = ((f"w{panel}{i}", f"c{panel}{i}") if side == "path" else
                            (f"c{panel}{i}", f"r{panel}{reader}"))
                geometry, arrow = tikz_measured_link(src, dst, band, f"d{panel}{i}{side}")
                tex += geometry
                arrows.append(arrow)
        # Keep the same fill-then-arrow layering in both reader panels.
        tex += arrows
        tex.append(r"\end{pgfonlayer}")
    tex += _reader_footer("Pathway gradients")
    _write_tikz(out / "downstream_reader_gradients.tex", tex)


def write_acquisition_tikz(path: Path, rows: list[dict]) -> None:
    """Model-derived TikZ using the article's shared learning-diagram styles.

    The site prepends tikz-learning.html. Coordinates scale the measured
    values from zero; text labels report their actual values independently.
    """
    targets = TRAINING.comparison_targets
    by_key = {(r["world"], r["target_score"]): r for r in rows}
    if len(by_key) != len(rows) or set(by_key) != set(itertools.product(("shared", "separate"), targets)):
        raise ValueError("Article comparison requires both routes at targets 0.950 and 0.975")
    if any(not r["acquired"] for r in rows):
        raise ValueError("Cannot plot an unsuccessful acquisition as measured cost")
    # Round the cost-axis ceiling upward, leaving space for numeric bar labels.
    ceiling = max(20, 20 * math.ceil(max(r["subsequent_steps"] for r in rows) / 20))
    response_ceiling = max(1., max(r["flower_response"] for r in rows))
    lines = [r"% Generated by python -m flower_toy; regenerate rather than edit.",
             r"\begin{tikzpicture}[learning]",
             r"\stageheading{.8}{LEARNING COST}{gradient steps}",
             r"\stageheading{8.5}{FLOWER RESPONSE}{at acquisition}"]
    axis_lines = []
    for origin, maximum, key, fmt in ((-1.5, ceiling, "subsequent_steps", ".0f"),
                                      (6.2, response_ceiling, "flower_response", ".3f")):
        axis_lines.append(fr"\draw[line width=.65pt] ({origin:.5f},2) -- ({origin:.5f},-1.8) -- ({origin+4.6:.5f},-1.8);")
        for i in range(5):
            value = maximum * i / 4
            y = -1.8 + 3.6 * i / 4
            label = f"{value:.0f}" if key == "subsequent_steps" else f"{value:.2g}"
            axis_lines.append(fr"\draw[line width=.65pt] ({origin-.08},{y:.5f}) -- ({origin},{y:.5f});")
            lines.append(fr"\node[anchor=east] at ({origin-.12},{y:.5f}) {{{label}}};")
        for j, target in enumerate(targets):
            center = origin + 1.2 + 2.1 * j
            lines.append(fr"\node[anchor=north] at ({center:.5f},-1.93) {{{target:.3f}}};")
            for world, offset, shade in (("shared", -.4, "black!65"), ("separate", .4, "black!15")):
                value = by_key[world, target][key]
                x = center + offset
                y = -1.8 + 3.6 * value / maximum
                lines.append(fr"\path[draw=black,line width=.65pt,fill={shade}] ({x-.32:.5f},-1.8) rectangle ({x+.32:.5f},{y:.5f});")
                lines.append(fr"\node[anchor=south,inner sep=2pt] at ({x:.5f},{y:.5f}) {{{value:{fmt}}}};")
        lines.append(fr"\node at ({origin+2.3},-2.65) {{Broccoli target score}};")
    lines += [r"\path[draw=black,line width=.65pt,fill=black!65] (1,-3.5) rectangle (1.24,-3.26);",
              r"\node[anchor=west] at (1.4,-3.38) {Shared route};",
              r"\path[draw=black,line width=.65pt,fill=black!15] (5.6,-3.5) rectangle (5.84,-3.26);",
              r"\node[anchor=west] at (6,-3.38) {Separate route};"]
    if all(r["flower_accuracy"] == 1. for r in rows):
        lines.append(r"\node[font=\sffamily\slshape\footnotesize] at (4.5,-4.15) {Flower classification: 100\% in every condition.};")
    lines += [r"\begin{pgfonlayer}{foreground}", *axis_lines,
              r"\end{pgfonlayer}", r"\diagramframe{-4.6}", r"\end{tikzpicture}"]
    _write_tikz(path, lines)


def plot_acquisition_comparison(out: Path, rows: list[dict]) -> None:
    """Article figure from the same records exported as acquisition_comparison.csv.

    Bars have a zero baseline; response differences are also labeled numerically.
    Refuse incomplete acquisitions rather than plotting budget exhaustion as cost.
    """
    plt = _pyplot()
    targets = sorted({r["target_score"] for r in rows})
    by_key = {(r["world"], r["target_score"]): r for r in rows}
    if not targets or len(by_key) != len(rows) or any(not r["acquired"] for r in rows):
        raise ValueError("Comparison requires unique, successful acquisitions")
    if set(by_key) != set(itertools.product(("shared", "separate"), targets)):
        raise ValueError("Comparison requires both routes at every target")
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none",
                         "svg.hashsalt": "flower-toy-acquisition"}):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        positions = np.arange(len(targets))
        width = .34
        for world, offset, color, hatch, label in (
                ("shared", -width / 2, "#444444", None, "Shared route"),
                ("separate", width / 2, "#bfbfbf", "///", "Separate route")):
            selected = [by_key[world, target] for target in targets]
            for ax, key, fmt in ((axes[0], "subsequent_steps", ".0f"),
                                 (axes[1], "flower_response", ".3f")):
                bars = ax.bar(positions + offset, [r[key] for r in selected], width,
                              color=color, edgecolor=color, linewidth=0,
                              hatch=hatch, label=label)
                ax.bar_label(bars, labels=[format(r[key], fmt) for r in selected], padding=4,
                             fontsize=10)
        axes[0].set(title="Subsequent learning cost", ylabel="Gradient steps",
                    ylim=(0, max(r["subsequent_steps"] for r in rows) * 1.23))
        axes[1].set(title="Final flower response", ylabel="Response on fixed assay",
                    ylim=(0, max(1., max(r["flower_response"] for r in rows)) * 1.23))
        for ax in axes:
            ax.set_xticks(positions, [f"{target:.3f}" for target in targets])
            ax.set_xlabel("Broccoli target score")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(.5, .06))
        accuracy = [r["flower_accuracy"] for r in rows]
        note = ("Flower classification remains 100% correct in every condition."
                if all(a == 1. for a in accuracy) else
                f"Flower classification accuracy ranges from {min(accuracy):.1%} to {max(accuracy):.1%}.")
        fig.text(.5, .025, note, ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .15, 1, 1), w_pad=2)
        fig.savefig(out / "acquisition_comparison.svg", metadata={"Date": None})
        fig.savefig(out / "acquisition_comparison.png", dpi=180)
        plt.close(fig)


def plot_overview(out: Path, tables: dict[str, list[dict]]) -> None:
    competition = tables["gradient_pressure"]
    acquisition = tables["acquisition_trajectories"]
    shared = [r for r in tables["matched_worlds"] if r["world"] == "shared"]
    separate = [r for r in tables["matched_worlds"] if r["world"] == "separate"]
    accounting, performance = tables["price_by_checkpoint"], tables["reader_performance"]
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    rows = [r for r in competition if r["head_start"] and not r["conflicting"]]
    for key, label in (("established_contribution", "Established route"),
                       ("alternative_contribution", "Alternative route")):
        ax.plot([r["step"] for r in rows], [r[key] for r in rows], label=label)
    ax.set(title="1. Trainable readers recruit shared routes", ylabel="Contribution")
    ax = axes[0, 1]
    for name in ("cold", "warm", "precursor_removed", "precursor_rescued"):
        rows = [r for r in acquisition if r["intervention"] == name]
        ax.plot([r["step"] for r in rows], [r["broccoli_score"] for r in rows],
                label=name.replace("_", " "), linestyle="--" if "precursor" in name else "-")
    ax.axhline(TRAINING.target_score, color="grey", linewidth=.7)
    ax.set(title="2. Precursor removal and rescue", ylabel="Broccoli score")
    ax = axes[0, 2]
    for rows in (shared, separate):
        ax.plot([r["step"] for r in rows], [r["flower_response"] for r in rows],
                label=rows[0]["world"])
    ax.text(.43, .65, "Both retain 100% flower\nclassification accuracy",
            transform=ax.transAxes, fontsize=8)
    ax.set(title="3. Weaker activation, intact classification", ylabel="Flower response")
    ax = axes[1, 0]
    rows = [r for r in accounting if r["history"] == "Retention" and r["assignment"] == "coherent"
            and r["assay"] == "continuous" and r["population"] == "all"]
    for key in ("selection", "transmission", "delta"):
        ax.plot([r["step"] for r in rows], [r[key] for r in rows], label=key)
    ax.set(title="4. Longitudinal Price accounting", ylabel="Mean response change")
    ax = axes[1, 1]
    rows = [r for r in performance if r["history"] == "Retention" and r["step"] == TRAINING.specialization_steps]
    for i, reader in enumerate(("flower", "broccoli")):
        ax.bar(np.arange(2) + i * .35, [r[reader + "_mse"] for r in rows], .35, label=reader)
    ax.set(xticks=[.175, 1.175], xticklabels=["Coherent", "Misrouted"],
           title="5. Reader errors on all eight inputs", ylabel="Mean squared error")
    ax = axes[1, 2]
    for history in ("Retention", "Re-acquisition"):
        rows = [r for r in performance if r["history"] == history and r["assignment"] == "coherent"]
        ax.plot([r["step"] for r in rows], [r["flower_max_error"] for r in rows], label=history)
    ax.axhline(TRAINING.max_reader_error, color="grey", linestyle="--", linewidth=.8,
               label=f"Precision criterion: {TRAINING.max_reader_error}")
    ax.set(title="6. When do reader responses meet the target?", ylabel="Flower reader maximum error",
           xlim=(0, 30))
    for i, ax in enumerate(axes.flat):
        ax.legend(fontsize=8)
        if i != 4:
            ax.set_xlabel("Gradient steps")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=180)
    plt.close(fig)


def write_handoff_diagram(out: Path, traces: list[dict]) -> None:
    final_step = max(r['step'] for r in traces)
    phase_step = final_step // 2
    tex = [r'\begin{tikzpicture}[learning]',
           r'\tikzset{reader/.style={rectangle,draw,fill=white,minimum size=6pt,inner sep=0pt,line width=.65pt}}']
    for x, title, step in ((0, 'INITIAL', 0), (4.5, 'HANDOFF', phase_step), (9, 'FINAL', final_step)):
        tex += [fr'\node[time] at ({x},2.0) {{{title}}};',
                fr'\node at ({x},1.6) {{Update {step}}};']
    for panel, (schedule, base, title) in enumerate((
            ('continuous', 0, r'CONTINUOUS\\HANDOFF'),
            ('interrupted', -4.3, r'INTERRUPTED\\RECOVERY'))):
        rows = [r for r in traces if r['schedule'] == schedule]
        tex.append(fr'\node[time,anchor=east,align=right] at (-1.5,{base}) {{{title}}};')
        for column, step in enumerate((0, phase_step, final_step)):
            x = column * 4.5
            row = rows[step]
            reader = f'r{panel}{column}'
            tex.append(fr'\node[reader] ({reader}) at ({x+.8},{base}) {{}};')
            for name, y, value in (('A', base+.55, row['a_flower']),
                                   ('B', base-.55, row['b_flower'])):
                node = f'n{panel}{column}{name}'
                tex += [fr'\node[feature] ({node}) at ({x-.55},{y}) {{}};',
                        fr'\node[anchor=east] at ({x-.85},{y}) {{{name}}};',
                        r'\begin{pgfonlayer}{background}',
                        fr'\draw[flow] ({node}.center) -- ({reader}.west);',
                        r'\end{pgfonlayer}']
                if value >= TRAINING.decision_threshold:
                    tex.append(fr'\responsetrait{{{node}}}')
        # Availability is sampled after every update, not inferred from trait markers.
        y = base - 1.35
        available = [r['flower_accuracy'] == 1. for r in rows]
        start = None
        # The sentinel checkpoint closes any available interval ending at the last sample.
        for step in range(len(rows)+1):
            present = step < len(rows) and available[step]
            if present and start is None:
                start = step
            if not present and start is not None:
                left = max(0, start-.5)*9/final_step
                right = min(final_step, step-.5)*9/final_step
                if step-start == 1:
                    tex.append(fr'\fill ({start*9/final_step},{y}) circle (1.2pt);')
                else:
                    tex.append(fr'\draw[line width=1.2pt] ({left},{y}) -- ({right},{y});')
                    for endpoint in (left, right):
                        tex.append(fr'\fill ({endpoint},{y}) circle (1.2pt);')
                start = None
        failed = [r['step'] for r in rows if r['flower_accuracy'] < 1.]
        label = (f'Unavailable at updates {failed[0]}--{failed[-1]}' if failed
                 else 'Available throughout')
        tex += [fr'\node[anchor=north west] at (0,{y-.22}) {{Flower classification}};',
                fr'\node[anchor=north east] at (9,{y-.22}) {{{label}}};']
    tex += [r'\node[align=center] at (9,-2.25) {Identical final parameters\\Endpoint selection 0, variation 0};',
            r'\node[feature] (keyflower) at (0,-7) {};',
            r'\responsetrait{keyflower}',
            r'\node[anchor=west] at (.3,-7) {Flower-responsive};',
            r'\node[feature] at (4,-7) {};',
            r'\node[anchor=west] at (4.3,-7) {Not flower-responsive};',
            r'\node[reader] at (9,-7) {};',
            r'\node[anchor=west] at (9.3,-7) {Flower reader};',
            r'\path[use as bounding box] (-3.45,-7.6) rectangle (12.4,2.5);',
            r'\end{tikzpicture}']
    _write_tikz(out / 'handoff_continuity.tex', tex)


def write_routing_diagram(out: Path, tables: dict[str, list[dict]]) -> None:
    """Endpoint circuits and reader-conditioned contributions on a common scale."""
    traces, prices = tables['learned_routing_trajectories'], tables['learned_routing_price']
    end = max(r['step'] for r in traces)
    tex = [r'\begin{tikzpicture}[learning]',
           r'\tikzset{reader/.style={rectangle,draw,fill=white,minimum size=6pt,inner sep=0pt,line width=.65pt}}']
    for k, (step, x, title) in enumerate(((0, 0, 'INITIAL'), (end, 4.5, 'FINAL'))):
        row = next(r for r in traces if r['condition']=='intact' and r['step']==step)
        tex += [fr'\stageoval{{{x},0}}',
                fr'\node[time] at ({x},2.7) {{{title}}};']
        for route, index, y in (('A', 1, .5), ('B', 2, -.5)):
            tex += [fr'\stacknode[feature]{{p{k}{route}}}{{{x}}}{{0}}{{2}}{{{index}}}',
                    (fr'\node[anchor=east] at ({x-.22},{y}) {{{route}}};' if k == 0 else
                     fr'\node[anchor=south] at ({x},{y+.16}) {{{route}}};')]
        for reader, y, label in (('flower', 1.45, 'Flower'), ('broccoli', -1.45, 'Broccoli')):
            name=f'r{k}{reader}'
            tex += [fr'\node[reader] ({name}) at ({x+.28},{y}) {{}};',
                    fr'\node[anchor=west] at ({x+.9},{y+.14}) {{{label}}};',
                    fr'\node[anchor=west] at ({x+.9},{y-.2}) {{A {row[reader+"_a_share"]:.0%}}};'.replace('%',r'\%'),
                    r'\begin{pgfonlayer}{background}']
            for route in ('A','B'):
                tex.append(fr'\draw[flow] (p{k}{route}.center) -- ({name}.south);' if reader=='flower'
                           else fr'\draw[flow] (p{k}{route}.center) -- ({name}.north);')
            tex.append(r'\end{pgfonlayer}')
    tex += [r'\begin{pgfonlayer}{background}',
            r'\draw[flow,shorten >=2.5pt] (p0A.center) -- (p1A.center);',
            r'\draw[flow,shorten >=2.5pt] (p0B.center) -- (p1B.center);',
            r'\end{pgfonlayer}',
            r'\node[font=\sffamily\slshape\footnotesize] at (2.25,0) {Training};',
            r'\node[time] at (7.8,2.7) {BY READER};',
            r'\node[time] at (10.9,2.7) {GLOBAL};']

    def bars(group, x, y):
        row=next(r for r in prices if r['condition']=='intact' and r['step']==end
                 and r['group']==group and r['reference']=='initial')
        tex.append(fr'\draw[line width=.4pt] ({x-1},{y}) -- ({x+1},{y});')
        for field, dx, color in (('selection',-.5,'black'),('transmission',.5,'black!40')):
            value=row[field]
            top=y+value*3.5
            tex.append(fr'\fill[{color}] ({x+dx-.2},{y}) rectangle ({x+dx+.2},{top});')
            anchor='south' if value>=0 else 'north'
            offset=.08 if value>=0 else -.08
            tex.append(fr'\node[anchor={anchor}] at ({x+dx},{top+offset}) {{{value:+.3f}}};')

    bars('flower',7.8,.5)
    bars('broccoli',7.8,-.5)
    tex += [r'\draw[->,line width=.5pt] (8.95,.5) -- (9.7,.12);',
            r'\draw[->,line width=.5pt] (8.95,-.5) -- (9.7,-.12);',
            r'\node[align=center] at (10.9,1) {Equal-reader\\average};']
    bars('all',10.9,0)
    tex += [r'\fill[black] (1.4,-3.05) rectangle (1.65,-2.8);',
            r'\node[anchor=west] at (1.85,-2.925) {Selection};',
            r'\fill[black!40] (5.5,-3.05) rectangle (5.75,-2.8);',
            r'\node[anchor=west] at (5.95,-2.925) {Variation};',
            r'\node at (4.5,-3.6) {Change in mean flower contrast};',
            r'\path[use as bounding box] (-3.45,-4) rectangle (12.4,3.15);',
            r'\end{tikzpicture}']
    _write_tikz(out / 'learned_routing.tex',tex)


def render_suite(result: SuiteResult, out: Path, *, plots: bool = True) -> None:
    """Render stored measurements without running experiments or exporting CSVs."""
    tables = result.tables
    write_gradient_diagrams(out, tables["gradient_band_measurements"], tables["gradient_pressure"])
    write_reader_gradient_diagrams(out, tables["reader_gradient_band_measurements"], tables["gradient_pressure"])
    write_acquisition_tikz(out / "acquisition_comparison.tex", tables["acquisition_comparison"])
    write_handoff_diagram(out, tables["handoff_trajectories"])
    write_routing_diagram(out, tables)
    # TikZ sources are always generated; Matplotlib is optional.
    if plots:
        plt = _pyplot()
        with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
            plot_lineage(out, tables)
            plot_overview(out, tables)
            plot_acquisition_comparison(out, tables["acquisition_comparison"])
