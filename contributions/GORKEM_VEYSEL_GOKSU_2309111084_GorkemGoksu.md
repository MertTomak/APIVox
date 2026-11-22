# Gorkem Veysel Goksu (2309111084, GorkemGoksu)

## What I built

The acoustic features and the three baselines we train from scratch, which compare the three ways of solving the alignment between audio frames and characters: CTC, attention and the RNN transducer.

- `src/features.py`: log-mel features (25 ms Hann window, 10 ms hop, 80 mel filters, per-utterance mean and variance normalization), speed perturbation with white noise, and SpecAugment. Each formula is stated in its docstring. The `mfcc` function in the same file was written by Mert for the GMM-HMM; the CTC ablation uses it with per-utterance normalization.
- `src/models.py`: one shared encoder (two stride-2 convolutions, 10 ms to 40 ms frames, then a 4-layer bidirectional LSTM) with three heads:
  - `ctc`: a linear layer over the characters plus a blank, greedy decoding with `ctc_collapse`, which the mHuBERT CTC head also uses;
  - `aed`: a 1-layer LSTM decoder with additive (Bahdanau) attention, trained with teacher forcing and label smoothing;
  - `rnnt`: a 1-layer LSTM prediction network and a joint network trained with `torchaudio.functional.rnnt_loss`, with greedy transducer decoding.
- `src/train_scratch.py`: one training loop for all three heads with early stopping on dev WER, the two ablations (`--features mfcc`, `--no-aug`) and the `--final` output contract (predictions for every evaluated split and a run file with hardware, time, parameters and best epoch).
- `tests/test_features.py` and `tests/test_models.py`.

## Key decisions and why

- **One encoder, one loop, one output contract for all heads.** Only the alignment method changes, so differences between `ctc`, `aed` and `rnnt` can be attributed to it and not to the encoder, the data or the training budget.
- **Log-mel in numpy with our own mel filterbank.** The formulas are visible and need no extra library.
- **The ctc-mfcc ablation uses the GMM-HMM's MFCCs.** When both versions met in one file, I kept Mert's `mfcc` unchanged, because the GMM-HMM keys its feature cache on that function's source and was tuned on it; the ablation adds per-utterance normalization so that it differs from the log-mel run only in the features.
- **4x subsampling kept from the plan, with an explicit CTC length check.** Fast spontaneous Kazakh comes close to one character per 40 ms frame. Train clips that cannot hold a CTC alignment at speed 1.1 are dropped and counted in the run file (`n_train_dropped`); dev and test clips are never filtered, so the reported scores include the clips CTC cannot fit. The same check is used with `--no-aug`, so `ctc` and `ctc-noaug` see the same training clips.
- **Augmentation seeded per (seed, epoch, clip).** The result does not depend on the number or the scheduling of data-loading workers, so the same seed gives the same run.
- **CTC and RNN-T losses divided by the target length.** Both losses are on the same scale, so the same gradient clipping and learning rate apply to every head.
- **Smaller batches for the RNN-T head.** Its joint tensor grows with frames times characters; the run prints the peak GPU memory of every epoch.
- **One seed for `aed` and `rnnt`, three for `ctc`.** A team decision under the compute budget (cut-off decision K10).

## Numbers I produced

Source for every row: `results/ctc_seed13.json`, `results/ctc_seed42.json`, `results/ctc_seed2026.json`, `results/aed_seed42.json`, `results/rnnt_seed42.json`, `results/ctc-noaug_seed42.json` and `results/ctc-mfcc_seed42.json`, fields `splits.<split>.wer`, `splits.<split>.cer`, `best_epoch` and `train_time_s`. Values are rounded from the files to four decimals. I did not average across seeds. `python -m src.build_benchmark --final` rescored every run from `results/preds/` and reported that every stored number matches.

| run | seed | dev WER | dev CER | test WER | test CER | test_talkshow WER | test_read WER | best epoch | train time (s) |
|---|---|---|---|---|---|---|---|---|---|
| ctc | 13 | 0.9999 | 0.9553 | 1.0000 | 0.9580 | 0.9999 | 1.0000 | 2 | 1492.1 |
| ctc | 42 | 1.0000 | 0.9649 | 1.0000 | 0.9657 | 0.9999 | 1.0000 | 2 | 1537.7 |
| ctc | 2026 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 | 1471.9 |
| aed | 42 | 4.3106 | 5.2385 | 4.2676 | 5.2135 | 4.8598 | 5.2231 | 1 | 1021.6 |
| rnnt | 42 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 | 472.0 |
| ctc-noaug | 42 | 0.9992 | 0.6895 | 0.9996 | 0.7003 | 0.9994 | 1.0002 | 4 | 213.7 |
| ctc-mfcc | 42 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 | 2009.3 |

- Every run used `hardware` "Tesla T4". `n_params`: 6,064,171 for `ctc` and `ctc-noaug`, 6,032,683 for `ctc-mfcc`, 8,985,643 for `aed`, 7,903,531 for `rnnt`.
- Training budget (`config.max_minutes`, `config.planned_epochs`, `config.stopped_by`): `ctc` seed 42 had 30 minutes, 5 planned epochs, stopped by the time budget. `ctc` seeds 13 and 2026 and both ablations had `--epochs 5` (stopped by `max_epochs`). `aed` had 25 minutes and 2 planned epochs, and `rnnt` had 15 minutes and 1 planned epoch (`config.batch_frames` 6000 instead of 20000); both were stopped by the time budget.
- `config.n_train_dropped` is 502 for every CTC run (train clips that cannot hold a CTC alignment at speed 1.1) and 0 for `aed` and `rnnt`.
- What these numbers mean: within this budget none of the three heads learned to produce correct words. For `ctc` and `rnnt`, WER close to 1.0 means the hypotheses are almost or entirely empty; in `results/preds/rnnt_seed42_test.jsonl` every hypothesis is empty. For `aed`, WER above 1 comes from insertions: the decoder repeats itself until the length limit, so its hypotheses are much longer than the references.
- The clearest difference is at the character level. `ctc-noaug` reaches dev CER 0.6895, against 0.9553 to 1.0000 for the three `ctc` seeds, which suggests that at this budget augmentation slowed early learning. With MFCC features (`ctc-mfcc`) the dev CER is 1.0000.

## How I verified the work

- Unit tests: hand-written CTC collapse cases, the frame-count and subsampling formulas, output shapes, padding invariance of the encoder, attention weights summing to 1, decoding that stops at `<eos>`, the per-frame symbol limit of the transducer, a 440 Hz tone peaking in the expected mel band, seeded determinism of the augmentation, and a loss decrease after one step on the real fixture clips for all three heads.
- Memorization: before training on the full data, each full-size model memorized a tiny set (training loss close to its minimum, output equal to the target), which shows that the loss, the decoding and the data path are consistent.
- The `--final` path was run end to end, and the same seed run twice gave identical losses and metrics on CPU.

## Limitations

- Greedy decoding without a language model for all three heads.
- About 31 hours of training data is small for models trained from scratch; these are course baselines that show how far the alignment methods get without pre-training.
- The CUDA CTC backward pass is not deterministic, so GPU runs with the same seed can differ slightly; the difference is reported.
- `aed` and `rnnt` have one seed, so their variance across seeds is unknown.
- The attention and transducer decoders step one character at a time in Python, which makes decoding of long clips slow.
