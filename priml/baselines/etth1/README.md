# DLinear on ETTh1

This is a starting point for forecasting experiments in PRIML. It uses DLinear
with the past 336 hours of ETTh1 data to predict the next 96 hours across all
seven variables.

`exp000` follows the reference setup: batch size 8, Adam at `1e-4`, seed 2021,
and a limit of 10 epochs / 10,260 updates. It stops early after three epochs
without matching or improving the best validation score. The full settings
are in `experiments.py`.

## Run locally

Run these commands from the PRIML repository. Data and checkpoints go under
`/opt/scratch`, the default resource root. To use another root, override
`base_dir` when training and use the matching paths for preparation and
evaluation. Use a fresh run directory if the checkpoint path contains an older run.

Install the dependencies and download the data:

```sh
uv sync --frozen
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.prepare_data \
  --directory /opt/scratch/datasets/etth1
```

Train the baseline:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  uv --quiet run --frozen python -m priml \
  priml.baselines.etth1.experiments.exp000
```

Evaluate the checkpoint with the best validation score:

```sh
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.evaluate \
  --directory /opt/scratch/datasets/etth1 \
  --checkpoint /opt/scratch/runs/etth1/exp000/checkpoints
```

The download script checks the file's checksum. If you already have the CSV,
add `--source /path/to/ETTh1.csv` to the preparation command. If downloading
fails with a certificate error on macOS, try adding
`SSL_CERT_FILE=/etc/ssl/cert.pem` before that command.

Normalization uses only the training data. The split and batch ordering match
the reference, including dropping incomplete batches. Checkpoints save the
state needed to resume training. Older checkpoints without that state can
still be evaluated, but cannot be resumed.

## Tests and reference check

Run the baseline tests:

```sh
uv --quiet run --frozen pytest priml/baselines/etth1 -o addopts= -q
```

These tests use small fixtures, so they do not need the downloaded dataset.
They cover data loading, training, checkpoint resume, and saved reference
outputs (called goldens) that catch changes to the results.

The reference is [ql-denoising/DLinear](https://github.com/ql-denoising/DLinear)
at commit `da9e67442b95af76488b8e4e1806cc3185723dd9`. To compare against a local
checkout, run the following with its pandas, scikit-learn, and matplotlib
dependencies available:

```sh
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.verify_reference \
  --reference /path/to/DLinear --directory /opt/scratch/datasets/etth1
```

This checks the source files and compares small test runs plus three training
updates on the real data, bit for bit. It allows the NumPy 2 compatibility
change from `np.Inf` to `np.inf` in `utils/tools.py`.

The goldens were recorded from the reference after the comparison passed.
To deliberately regenerate them, use pytest so PRIML's numerical settings
are loaded before the source capture:

```sh
uv --quiet run --frozen pytest \
  priml/baselines/etth1/scripts/mint_reference_test.py -o addopts= \
  --etth1-reference /path/to/DLinear \
  --etth1-directory /opt/scratch/datasets/etth1
```

Adding `--mint` to the verifier command above runs the same pytest step.
Source details are in `testdata/source.json`.

## Local results

The CPU run on 2026-10-02 stopped after 5 epochs / 5,130 updates. The best
checkpoint was step 2,052:

| Metric | Value |
| --- | --- |
| Validation MSE | 0.6461290121 |
| Test MSE | 0.3748246133 |
| Test MAE | 0.3994735181 |

A full run of the reference matched the validation losses, saved parameters,
final Torch RNG state, and every test prediction exactly. The comparison is
recorded in `results/local_cpu.json`.

These results were measured on Apple Silicon with Python 3.12.3, PyTorch 2.11.0,
NumPy 2.5.3, and one Torch thread. GPU/reference-hardware results are still TBD.

## exp001: cosine learning-rate decay

`exp001` starts from `exp000` and replaces its halving schedule with PRIML's
cosine schedule, without restarts. The idea is to keep learning later in the
run, when the original rate has become small. The rate is still updated at
epoch boundaries, using the same ten-epoch schedule horizon.

The model, initial learning rate, data split, scoring, seed, early stopping,
and maximum training budget stay the same. Validation chooses the best
checkpoint; the test split is used only for final evaluation.

Train the experiment:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  uv --quiet run --frozen python -m priml \
  priml.baselines.etth1.experiments.exp001
```

Evaluate its best checkpoint. Select the same experiment used for training:

```sh
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.evaluate \
  --directory /opt/scratch/datasets/etth1 \
  --experiment exp001 \
  --checkpoint /opt/scratch/runs/etth1/exp001/checkpoints
```

### Local comparison

Fresh CPU runs on 2026-10-06 used the same environment listed above and seed
2021. Both stopped after 5 epochs / 5,130 updates, choosing step 2,052 by
validation score:

| Metric | exp000 | exp001 |
| --- | --- | --- |
| Best validation MSE | 0.6461290121 | 0.6464084983 |
| Test MSE | 0.3748246133 | 0.3747777343 |
| Test MAE | 0.3994735181 | 0.3994295299 |

Cosine decay did not improve validation MSE: it was about 0.043% higher.
`exp000` remains the preferred recipe from this comparison. Test scores were
measured after that decision and were not used to pick the experiment.
This is one seed on a local CPU, not evidence of a general improvement.
Reference-hardware results are still TBD. Full validation histories, settings,
and checkpoint hashes are in `results/exp001_local_cpu.json`.

## Further experiments

The search tried 45 validation-only candidates, including learning rates,
weight penalties, input lengths, averaging, and smaller projections. Trial IDs
in `results/experiment_search_cpu.json` record that search; they are not public
experiment names. The retained recipes each make one change to a named parent:

| Experiment | Parent | Change |
| --- | --- | --- |
| exp001 | exp000 | Cosine learning-rate decay |
| exp002 | exp000 | Starting learning rate `3e-4` |
| exp003 | exp000 | Starting learning rate `1e-3` |
| exp004 | exp003 | Rank-32 seasonal and trend projections |

The higher rate in `exp002` improved the first seed's validation MSE by 1.82%,
but the gain was small and inconsistent across five seeds. The compact model
in `exp004` was selected using validation scores and parameter count. Its
settings were fixed before evaluating the held-out test split.

### exp004: smaller forecast projections

Each forecast projection starts from the input mean and learns a correction
through two narrow linear layers. The final layer starts at zero, so the model
initially predicts the mean. The rank is configurable, both child layers can
be replaced, and their costs use the same config tree as the model. The unused
reference decoder is retained.

Train and evaluate it with:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  uv --quiet run --frozen python -m priml \
  priml.baselines.etth1.experiments.exp004

uv --quiet run --frozen python -m priml.baselines.etth1.scripts.evaluate \
  --experiment exp004 \
  --directory /opt/scratch/datasets/etth1 \
  --checkpoint /opt/scratch/runs/etth1/exp004/checkpoints
```

Use `--experiment exp004` when scoring these checkpoints so the evaluator
builds the compact model. The default remains `exp000`. The three rate-only
forks also work with their own experiment names.

### Five-seed CPU comparison

Both recipes used seeds 2021--2025, the same split, 336-hour history,
96-hour forecast, batch size 8, ten-epoch cap, and validation-based early
stopping. Every finished `exp004` run reproduced the selected research run's
validation history, best model weights, and test metrics exactly.

| Mean metric across five seeds | exp000 | exp004 |
| --- | --- | --- |
| Best validation MSE | 0.64887383 | 0.64470326 |
| Test MSE | 0.37299972 | 0.37695379 |
| Test MAE | 0.39705042 | 0.40350056 |

| Seed | exp000 test MSE | exp004 test MSE |
| --- | --- | --- |
| 2021 | 0.37482461 | 0.37302694 |
| 2022 | 0.37117931 | 0.37325835 |
| 2023 | 0.37500045 | 0.37942612 |
| 2024 | 0.37278536 | 0.37639061 |
| 2025 | 0.37120888 | 0.38266692 |

Validation MSE improved by 0.64% on average, with wins on four of five seeds.
Held-out MSE was 1.06% worse and MAE was 1.62% worse, with only one test win.
This is a model-size tradeoff, not a demonstrated accuracy improvement.
Keep `exp000` as the accuracy baseline.

| Model cost at batch 8 | exp000 | exp004 |
| --- | --- | --- |
| Active parameters | 64,704 | 27,840 |
| Total parameters, including unused decoder | 97,056 | 60,192 |
| Forward FLOPs | 7,730,688 | 3,650,304 |
| Forward + backward FLOPs | 23,151,552 | 10,862,016 |

The compact model has 56.97% fewer active parameters and 52.78% fewer forward
FLOPs. This does not establish a speedup: a local single-thread CPU
microbenchmark measured about 0.86 ms per baseline training update and
0.92 ms per compact update. These are warm, synthetic-batch timings, not
end-to-end training times or GPU benchmarks.

For paired runs, override `seed` and use a fresh `base_dir` for each seed.
Set `dataset.working_dir` to the prepared CSV directory so all runs use the
same data. For example, append these arguments to the training command:

```sh
--override seed=2022 \
--override base_dir=/opt/scratch/etth1-seed-2022 \
--override dataset.working_dir=/opt/scratch/datasets/etth1
```

Select each run's `best.json` checkpoint using validation only. The full
histories, scores, checkpoint hashes, environment, costs, and limitations
are in `results/exp004_local_cpu.json`. The smaller cosine comparison remains
in `results/exp001_local_cpu.json`. No canonical source goldens were changed.
GPU/reference-hardware results remain TBD.
