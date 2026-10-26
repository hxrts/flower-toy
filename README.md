# Flower Toy Model

Synthetic experiments for [*Learning to Be Clean*](https://weboftru.st/post/learning-to-be-clean/), with fixed inputs and initialization and no random sampling. Three binary cues—flower, green, and texture—define eight inputs; broccoli requires green and texture. Architectures, curricula, and ancestry are controlled assumptions.

## Run

With Python 3.9+ from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m flower_toy
.venv/bin/python -m unittest -v
```

Outputs go to `results/` (ignored by Git). Use `--output PATH` to choose another directory or `--no-plots` to skip Matplotlib plots while retaining measurements and TikZ sources. No data is downloaded.

With the blog checked out alongside this repository, regenerate its diagram sources and snapshot the cited measurements here:

```sh
.venv/bin/python -m flower_toy --no-plots \
  --blog-assets ../weboftru.st/static/images/flower-toy \
  --article-data data
```

The tracked [article measurements](data/) contain the CSV snapshot cited by the post, covering training trajectories, gradients, Price accounting, route ablations, and handoff threshold checks. To regenerate only this snapshot, run `.venv/bin/python -m flower_toy --no-plots --article-data data`. The blog receives only TikZ sources, including measured values required to render its figures. Full run outputs remain in ignored `results/`.

## Experiments

| Experiment | Main comparison |
| --- | --- |
| Precursor acquisition | Green pretraining reduces subsequent broccoli acquisition from 23 to 8 updates; pretraining costs 10. Removal and rescue isolate the precursor’s contribution. |
| Capacity acquisition | At score 0.975, compare cold, green-pretrained, and capacity-limited green-pretrained acquisition. Full trajectories are exported. |
| Competing routes | Feature coefficients and reader weights learn jointly. Compatible targets reinforce shared features; opposed contributions can nearly cancel. |
| Matched worlds | Under capacity `f² + g² <= 1.30²`, shared/separate routes take 12/23 updates at score 0.95, but 67/49 at 0.975. Shared flower activation weakens while classification stays correct at threshold 0.5. |
| Role specialization | Clones learn prescribed local targets, supporting Price accounting under registered ancestry. |
| Reader-driven learning | Reader errors train all pathway coordinates and read gains directly. Coherent and swapped assignments start from identical ancestor copies; both classify correctly after 800 updates, with larger residual errors after the swap. |
| Continuity | Retention and re-acquisition reach equivalent endpoint functions but differ in ancestry and intermediate performance. Classification and response precision are measured separately. |

Acquisition also requires negative scores below 0.10; scores are not calibrated probabilities. Reader-driven training averages two gained signals for flower and multiplies them for broccoli. It minimizes mean half-squared error over readers and inputs, with learning rate 0.15.

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

## Code

`experiments.run_suite()` computes an in-memory `SuiteResult`; `artifacts.write_suite(result, output_path)` writes its report, measurements, and figures. Scientific defaults and band settings live in `settings.py`.

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

## Results

Start with `results/summary.json` and `results/overview.png`. Other outputs in `results/` include:

| Files | Contents |
| --- | --- |
| `matched_world*.csv`, `precursor_interventions.csv`, `capacity_acquisition*.csv` | Acquisition costs and interventions |
| `price_by_checkpoint.csv`, `reader_price_accounting.csv` | Longitudinal and reader-group accounting |
| `reader_performance.csv`, `reader_availability.csv`, `dependency_ablations.csv` | Accuracy, continuity, and pathway dependence |
| `downstream_reader_*.csv` | Reader-driven training and endpoints |
| `signed_gradients.csv`, `*gradient_band_measurements.csv` | Gradients and exact band measurements |
| `*.tex` | Article figure sources |

## Matched inheritance and continuity

`development.py` compares two histories with the same role targets and equivalent final reader outputs. Retention assigns F1/F2/G to A and T to B; Mixed inheritance assigns F1/G/T to A and F2 to B.

Both have global binary Price terms (+¼, −¼), but their reader groups differ:

| History | Flower: selection, transmission | Broccoli: selection, transmission |
| --- | --- | --- |
| Retention | +½, 0 | 0, −½ |
| Mixed inheritance | 0, +½ | +½, −1 |

Equal group weights recover the global terms. `matched_reader_inheritance.csv` records these values. Unit credit counts attributed uses; reader gains and ablation effects measure functional influence separately.

The handoff experiment starts flower channels at A=1, B=0. A fixed mean reader classifies at threshold 0.45. Each channel receives 40 gradient updates at rate 0.2, fitting B to one and A to zero. Acquiring B before retiring A preserves classification throughout; reversing the order temporarily breaks it. Both schedules reach identical full parameter states and zero endpoint Price terms under one-to-one ancestry and binary trait threshold 0.5.

`handoff_trajectories.csv`, `handoff_price.csv`, and `handoff_continuity.tex` record this comparison. Identical full endpoints imply identical future updates under matched deterministic training, despite different histories of availability.

## Learned routing connects dynamics and Price accounting

`routing.py` trains two persistent feature routes and two readers' softmax allocations jointly from reader loss. Each reader has one unit of routing mass. The flower reader reads a mixture directly; broccoli multiplies its mixture by a fixed texture cue. A starts at flower=1, green=0.8; B starts at zero. Equal initial routing shares, 800 updates and learning rate 0.15 are shared across intact, green-removed, green-restored, and frozen-routing conditions. There are no prescribed local feature targets or descendant counts.

The continuous trait is matched flower contrast. With initial masses p and current masses q, selection is sum((q-p)*z_initial), variation is sum(q*(z_current-z_initial)). The same weighted identity is exported between adjacent checkpoints. Equal reader weights reconstruct global accounting. Allocation is not automatically functional importance, and persistent coordinates would give zero selection under unit coordinate weighting.

At step 800, intact flower-reader terms are approximately (+0.217,+0.193), broccoli terms (-0.230,-0.107), and global terms (-0.007,+0.043). Freezing routing gives zero selection and 75% broccoli classification, compared with 100% under learned routing. Removing green lowers final response error in this setup; restoring green restores the intact trajectory. A precursor's advantage therefore depends on task and reader competition, rather than following merely from its presence.

`learned_routing_trajectories.csv`, `learned_routing_price.csv`, and `learned_routing_pressure.csv` record the same runs. Pressure rows include per-reader feature and routing-logit gradients, including unapplied terminal gradients. The ancestry ensemble remains a supplementary declared policy, separate from this gradient-driven allocation experiment.

## Functional dependence and continuity checks

`learned_routing_ablations.csv` measures both readers after zeroing route A or B, holding all trained weights fixed without renormalizing routing shares. In the intact run, removing A lowers flower accuracy to 50% while broccoli stays at 100%; removing B lowers broccoli to 75% while flower stays at 100%. Errors and maximum errors are also exported for all conditions.

The handoff trace includes negative-input scores and positive/negative margins at threshold 0.45. `handoff_thresholds.csv` checks six thresholds from 0.001 to 0.50. The continuous schedule has minimum positive margin 0.05 at the declared threshold, and stays correct at every tested threshold below 0.50. Interrupted recovery fails temporarily at each of those thresholds. At 0.50 the continuous schedule reaches a zero-margin boundary and floating-point rounding triggers a final failed classification.
