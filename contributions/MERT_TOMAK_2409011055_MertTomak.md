# Mert Tomak (2409011055, MertTomak)

## What I built

- **M0, repo layout.** Converted the repository to the Aralas layout: removed the old files, added the stub scripts, pinned `requirements.txt` and wrote the README table that covers every script.
- **M1, `src/metrics.py`.** Corpus-level WER and CER, `cs_wer`, `mono_wer`, `ru_word_acc` and `kz_word_acc`, per-slice scores, and bootstrap confidence intervals. The bootstrap has a pooled variant (`pooled_bootstrap_ci`, `pooled_paired_bootstrap`) that resamples utterances once and averages WER over seeds. Other members' code scores through this module.
- **M2, `src/gmm_hmm.py`.** A character-level GMM-HMM baseline: Viterbi training of 43 units with 3 states each, with Gaussian components grown 1, 2, 4, 8. Decoding uses a bigram character LM. The LM weight and insertion penalty are tuned on dev CER (`selection_metric: dev_cer`) over a fixed grid of 5 weights and 11 penalties. Model family: GMM-HMM on MFCC features, trained from scratch, so there is no pretrained checkpoint (`revision: null`).
- **`mfcc` in `src/features.py`.** 13 coefficients plus delta and delta-delta, which the GMM-HMM uses as input.
- **M3, `src/build_benchmark.py`.** Rescores every stored prediction file and writes the benchmark and ablation tables and the figures. It lists missing runs and exits with an error if a stored number disagrees with the rescored one.
- **Coordination.** I own the repository and the shared contracts (data, metrics and result formats), and I review the other members' branches (see "Reviews and `features.mfcc`" below).

## Key decisions and why

- **Tuning only on dev.** The LM weight and insertion penalty are chosen on dev CER, never on a test split. The grid is frozen in the run config (`config.lm_weight_grid`, `config.insertion_penalty_grid`).
- **Pooled bootstrap.** Seeds share the same resampled utterances, so the interval reflects the test set and not seed-to-seed noise.
- **Cut line (K10, decided 20.11).** The data froze late, so `aed` and `rnnt` run with seed 42 only. `gmm_hmm` needed no cut: all three runs used the full training set (`config.train_hours_used` is 31.079, `config.train_hours_requested` is null, 22811 utterances, 0 skipped) and all three seeds. `mhubert_ft` drops to one seed if it does not fit.
- **Reviews and `features.mfcc`.** I reviewed, tested and merged Görkem's features and scratch-models PRs. I wrote `features.mfcc` myself to unblock the GMM-HMM final runs; Görkem owns the rest of `features.py`.

## Numbers I produced

Source for every row: `results/gmm_hmm_seed13.json`, `results/gmm_hmm_seed42.json`, `results/gmm_hmm_seed2026.json`, field `splits.<split>`. Values are rounded from the files to four decimals. I did not average across seeds.

| seed | LM weight | insertion penalty | dev WER | dev CER | test WER | test CER |
|---|---|---|---|---|---|---|
| 13 | 8.0 | -10.0 | 1.0146 | 0.6363 | 1.0129 | 0.6526 |
| 42 | 8.0 | -8.0 | 1.0037 | 0.6391 | 1.0035 | 0.6575 |
| 2026 | 8.0 | -10.0 | 1.0126 | 0.6353 | 1.0109 | 0.6564 |

- `ru_word_acc` is 0.0 on dev, test, test_talkshow and test_read for all three seeds.
- Dev size: 1081 utterances and 14100 reference words. Test size: 1546 utterances and 20428 reference words (`splits.dev.n_utts`, `n_ref_words`, and the same for `test`).
- Training time: 180.05 s (seed 13), 198.70 s (seed 42), 194.98 s (seed 2026), on `hardware` "Apple M5 (10 cores)". `n_params` is 1032.
- The selected penalty sits on the grid edge (-10.0) for seeds 13 and 2026 (`config.penalty_on_grid_edge: true`) and not for seed 42 (`false`).

## How I verified the work

- Every number above was read from the three run files listed under "Numbers I produced".
- `tests/test_gmm_hmm.py` and `tests/test_metrics.py` cover the model and the metrics.
- `build_benchmark.py` exits with an error when a stored WER differs from its rescored value.

## Limitations

- **The baseline does not work as a recognizer.** WER is above 1.0 on every seed and split I report (the smallest is 1.0035, test, seed 42), meaning it makes more errors than the reference has words. CER is between 0.6353 and 0.6575 on dev and test, and `ru_word_acc` is 0.0 everywhere. It is a floor, not a competitive baseline.
- **Edge-of-grid penalty.** Seeds 13 and 2026 picked the most negative penalty on the grid, so a lower penalty might score better. The grid was frozen before the final runs and I did not widen it.
- **No `kz_word_acc` in the run files.** The three run files contain no `kz_word_acc` field, so I do not report one here.
- **Seed spread.** The three seeds disagree on the insertion penalty, and their dev WER differs between 1.0037 and 1.0146.
