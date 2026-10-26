# Flower Toy Model

Deterministic synthetic experiments accompanying *Cleanliness is Next to Godliness*: acquisition costs, reader gradients, Price accounting, and developmental continuity. Three binary cues—flower, green, and texture—define eight inputs; broccoli requires green and texture. Architectures, curricula, and ancestry are controlled assumptions.

## Run

With Python 3.9+ from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m flower_toy
.venv/bin/python -m unittest -v
```

Outputs go to `results/` (ignored by Git). Use `--output PATH` to change the destination or `--no-plots` to generate measurements and TikZ sources without Matplotlib plots. No data is downloaded.

With the blog checked out alongside this repository, regenerate its figures and measurements:

```sh
.venv/bin/python -m flower_toy --no-plots \
  --blog-assets ../weboftru.st/static/images/flower-toy
```

## Experiments

| Experiment | Main comparison |
| --- | --- |
| Precursor acquisition | Green pretraining reduces subsequent broccoli acquisition from 23 to 8 updates; pretraining costs 10. Removal and rescue isolate the precursor’s contribution. |
| Capacity acquisition | At score 0.975, compare cold, green-pretrained, and capacity-limited green-pretrained acquisition. Full trajectories are exported. |
| Competing routes | Feature coefficients and reader weights learn jointly. Compatible targets reinforce shared features; opposed contributions can nearly cancel. |
| Matched worlds | Identical architectures obey `f² + g² <= 1.30²`. Shared/separate routes take 12/23 updates at broccoli score 0.95, but 67/49 at 0.975. Flower activation weakens on the shared route while classification remains correct at threshold 0.5. Negative scores must be below 0.10. Scores are not calibrated probabilities. |
| Role specialization | Clones learn prescribed local targets, supporting Price accounting under registered ancestry. |
| Reader-driven learning | Reader errors train all pathway coordinates and read gains directly. Coherent and swapped assignments start from identical ancestor copies; both classify correctly after 800 updates, with larger residual errors after the swap. |
| Continuity | Retention and re-acquisition reach equivalent endpoint functions but differ in ancestry and intermediate performance. Classification and response precision are measured separately. |

Reader-driven training uses the mean of two gained signals for flower and their product for broccoli. It minimizes mean half-squared error over readers and inputs, with learning rate 0.15.

## Accounting conventions

A acquires green while retaining flower; B starts texture-responsive. Retention assigns F1/F2/G to A and T to B; re-acquisition assigns F1/F2/T to B and G to A. Diagram labels A₁/A₂/A₃/B₁ denote F1/F2/G/T in retention.

Ancestry is registered at cloning. Each included use carries one unit of credit, optionally split among ancestors; unassigned uses are excluded. Flower responsiveness is measured by contrast against matched nonflower inputs, with binary threshold 0.5.

Retention has Price selection +¼ and transmission −¼; re-acquisition reverses these signs while preserving the final mean. The article calls transmission “variation”; numerical fields retain `transmission`. Reader groups share the original ancestor baseline, and their terms reconstruct the global decomposition when weighted by descendant credit.

The fixed endpoint comparison swaps readers while preserving responses and ancestry. Reader-driven training is a separate experiment starting from ancestor copies with unit gains, before specialization. Matching endpoint functions does not establish uninterrupted availability during training.

The Gibbs ensemble enumerates all 16 assignments. For each role, parent probabilities are proportional to `exp(-E/T)`, with energy equal to half the squared distance to its target. This is a conditional policy, not inferred ancestry.

## Gradient bands

Bands encode descent pressure (negative loss derivative): wide-left increases a parameter, wide-right decreases it. Maximum width measures absolute gradient, averaged over the selected interval; zero gradients have no band.

| Figure | Checkpoint/interval | TikZ points per gradient unit |
| --- | --- | --- |
| Competing routes | `[0,10)`, `[10,30)` | 80 |
| Reader contributions | Step 0 | Features 240; connections 48 |
| Price illustration | `[0,190)` | Flower coordinate 200 |
| Coherent/swapped readers | Step 10 | Green coordinates and gains 200 |

Opaque `black!40` bands precede arrows in the background layer; node sizes are unscaled. The blog supplies shared styles. Reader contributions include task averaging and sum to the total gradient. Terminal derivatives have no applied update. Capacity projection is recorded separately: it can weaken flower activation despite zero direct flower gradient.

## Files

`experiments.run_suite()` returns a `SuiteResult` without writing files or rendering. Its tables and summary are available directly in Python; `artifacts.write_suite(result, output_path)` writes the report, CSVs, and figures. Rendering consumes stored measurements. Scientific defaults and figure band settings are separate in `settings.py`.

| Path | Contents |
| --- | --- |
| `flower_toy/model.py` | Learning dynamics, readers, and ancestry accounting |
| `flower_toy/experiments.py` | Comparisons and summaries |
| `flower_toy/measurements.py` | Pure trajectory, accounting, and band measurements |
| `flower_toy/figures.py` | TikZ and Matplotlib rendering |
| `flower_toy/development.py` | Matched inheritance and handoff simulations |
| `flower_toy/settings.py` | Scientific defaults and independent band settings |
| `flower_toy/results.py` | In-memory experiment and suite results |
| `flower_toy/report.py`, `flower_toy/artifacts.py` | Report formatting, file writing, and article export |
| `flower_toy/cli.py` | Command-line orchestration |
| `tests/` | Numerical gradients and scientific invariants |

Start with `results/summary.json` and the single overview plot, `results/overview.png`. Acquisition comparisons use `matched_world*.csv`, `precursor_interventions.csv`, and `capacity_acquisition*.csv`; accounting uses `price_by_checkpoint.csv` and `reader_price_accounting.csv`. Reader performance, availability, and ablations are exported separately. `downstream_reader_*.csv` contains reader-driven training; `signed_gradients.csv` and `*gradient_band_measurements.csv` contain derivatives and exact band measurements. Generated article sources are `*.tex`.

## Matched inheritance and continuity

`flower_toy/development.py` adds two controlled comparisons. In the matched
comparison, Retention assigns F1/F2/G to A and T to B; Mixed inheritance assigns
F1/G/T to A and F2 to B. Both learn the same role targets and reach equivalent reader
outputs on all eight inputs. Both have global binary Price terms (+1/4, -1/4).
For flower readers the terms are (+1/2, 0) and (0, +1/2), respectively. For
broccoli they are (0, -1/2) and (+1/2, -1). Equal group weights recover the global
terms. `matched_reader_inheritance.csv` records these values.

Unit credit measures the share of attributed pathway uses, not functional
influence. Reader gains and ablation effects remain separate measurements.
Influence-weighted accounting requires an explicit nonnegative attribution rule.

The handoff comparison starts two flower channels at A=1, B=0. A fixed reader
averages them and classifies at threshold 0.45, leaving a margin below the
initial positive score of 0.5. Each channel receives forty gradient updates at
rate 0.2 on half-squared error, fitting B to one and A to zero. Acquisition-first
and loss-first schedules use identical update budgets and reach identical full
parameter states. Classification is evaluated after every update on both binary
flower inputs. It remains correct under acquisition-first training but temporarily
fails under loss-first training. These are declared schedules, not learned
handoff policies. Both have zero endpoint selection and variation under
one-to-one ancestry and a binary trait threshold of 0.5.

`handoff_trajectories.csv`, `handoff_price.csv`, and `handoff_continuity.tex`
record the comparison. Identical full endpoints imply identical future updates
under matched deterministic training. This distinguishes historical continuity
from retained state differences that could alter future acquisition costs.
