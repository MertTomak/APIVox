# Zeynep Celik (2309111135, zeynepcelikx)

## What I built

The pre-trained models of Aralas: four baselines used without training and two models we fine-tune on our train split.

- **`src/eval_zero_shot.py`.** Scores four published Kazakh-capable checkpoints, with no training, on dev and the final splits. Whisper decodes greedily with the language fixed to Kazakh (`kk`); the CTC models take the argmax per frame. Batches are grouped by duration. The run file records, for each model, what its model card says about training on KSC or KSC2 (`config.ksc_training`).
- **`src/train_whisper.py` (`whisper_small_ft`).** Full fine-tuning of Whisper small on our train split with Kazakh transcription prefixes, fp16 autocast on CUDA and gradient accumulation. It scores dev after every epoch, keeps the best epoch by (dev WER, dev CER) and decodes with the same greedy decoder as the zero-shot run.
- **`src/train_mhubert.py` (`mhubert_ft`).** mHuBERT-147 with a new CTC head over `data_io.CHARS` (blank = 0). The convolutional feature encoder is frozen, and the new head and the Transformer use separate learning rates. Decoding is greedy CTC with Görkem's `src.models.ctc_collapse`.
- Both training scripts have `--lr-sweep` (one epoch per candidate rate on dev, writes nothing) and `--smoke` (one step on the fixture clips on CPU).
- **`tests/test_eval_zero_shot.py`.** 7 tests with stand-in models, so no checkpoint is downloaded.
- **Run files and predictions.** `results/whisper_small_zs_seed0.json`, `whisper_large_zs_seed0.json`, `xlsr_kk_zs_seed0.json`, `mms_kk_zs_seed0.json`, `mhubert_ft_seed42.json` and `whisper_small_ft_seed2026.json`, each with its files in `results/preds/`.

| Run | Architecture | Checkpoint @ revision |
|---|---|---|
| `whisper_small_zs` | Transformer encoder-decoder | `openai/whisper-small` @ `973afd24965f72e36ca33b3055d56a652f456b4d` |
| `whisper_large_zs` | Transformer encoder-decoder | `openai/whisper-large-v3` @ `06f233fe06e710322aca913c1bc4249a0d71fce1` |
| `xlsr_kk_zs` | wav2vec 2.0 XLSR-53 + CTC | `aismlv/wav2vec2-large-xlsr-kazakh` @ `b277ef85de7bbd28302d570062feb196cc4e9107` |
| `mms_kk_zs` | wav2vec 2.0 MMS-1B + `kaz` language adapter | `facebook/mms-1b-all` @ `3d33597edbdaaba14a8e858e2c8caa76e3cec0cd` |
| `mhubert_ft` | HuBERT base + CTC head (fine-tuned by us) | `utter-project/mHuBERT-147` @ `7ad3fc0bc5106c58c9c13526abccad527150d135` |
| `whisper_small_ft` | Transformer encoder-decoder (fine-tuned by us) | `openai/whisper-small` @ `973afd24965f72e36ca33b3055d56a652f456b4d` |

## Key decisions and why

- **One decoder for zero-shot and fine-tuned Whisper.** Both use greedy decoding with the language set to Kazakh, so the difference between `whisper_small_zs` and `whisper_small_ft` comes from training alone.
- **Model-card check for KSC.** `xlsr_kk_zs` was fine-tuned on the KSC v1.1 train split according to its model card, and `test_read` is the KSC test set. Its `test_read` score is therefore not a zero-shot number. The other three cards name no KSC data.
- **Learning rates chosen on dev only.** Each sweep trains one epoch per rate with seed 42 and scores dev. It writes no file, so its table is not stored. For Whisper I compared 5e-6, 1e-5 and 3e-5 and chose 3e-5, the rate with the lowest dev WER. For mHuBERT, the first sweep with a head learning rate of 1e-4 produced only blanks for every encoder rate. I raised the head rate to 1e-3, compared encoder rates of 3e-5 and 5e-5, and chose 3e-5.
- **CTC loss over each clip's true length.** The first mHuBERT version computed the loss over the padded length, which let the model align the end of every transcript to padding and stay blank. The loss now uses each clip's own frame count.
- **Gradient accumulation for Whisper.** A batch of 16 ran out of memory on a T4, so the script runs micro-batches of 8 and accumulates 2 of them. The effective batch is still 16.
- **Cut line (K10).** `mhubert_ft` runs with seed 42 only, and `whisper_large_zs` runs on `test` and `test_read` only.

## Numbers I produced

Source for every value: `results/<model>_seed<seed>.json`, field `splits.<split>`, rounded from the files to four decimals. All runs used `hardware` "Tesla T4".

WER (CER in brackets):

| Run | dev | test | test_talkshow | test_read |
|---|---|---|---|---|
| `whisper_small_zs` | 1.2069 (0.9506) | 1.1285 (0.9155) | 1.0610 (0.8452) | 0.8703 (0.3680) |
| `whisper_large_zs` | not run | 0.6271 (0.2697) | not run | 0.4318 (0.1089) |
| `xlsr_kk_zs` | 0.5948 (0.2103) | 0.5831 (0.2058) | 0.4616 (0.1530) | 0.2032 (0.0404) |
| `mms_kk_zs` | 0.5119 (0.1715) | 0.5002 (0.1607) | 0.4139 (0.1331) | 0.2690 (0.0594) |
| `mhubert_ft` seed 42 | 0.5027 (0.1538) | 0.4832 (0.1414) | 0.4148 (0.1210) | 0.4224 (0.0890) |
| `whisper_small_ft` seed 2026 | 0.3401 (0.1339) | 0.3272 (0.1263) | 0.2694 (0.1037) | 0.2993 (0.0805) |

Code-switching metrics on `test`:

| Run | cs_wer | mono_wer | ru_word_acc | kz_word_acc |
|---|---|---|---|---|
| `whisper_small_zs` | 1.3750 | 1.1087 | 0.0522 | 0.1055 |
| `whisper_large_zs` | 0.6750 | 0.6233 | 0.1791 | 0.4175 |
| `xlsr_kk_zs` | 0.6737 | 0.5758 | 0.0448 | 0.4382 |
| `mms_kk_zs` | 0.5539 | 0.4959 | 0.1418 | 0.5314 |
| `mhubert_ft` seed 42 | 0.5296 | 0.4794 | 0.1045 | 0.5642 |
| `whisper_small_ft` seed 2026 | 0.3592 | 0.3246 | 0.5746 | 0.7258 |

Training (from the same files):

- `whisper_small_ft` seed 2026: `best_epoch` 5 of 5, `train_time_s` 15124.8, `n_params` 241734912. Dev WER per epoch (`config.dev_curve`): 0.5043, 0.4127, 0.3696, 0.3452, 0.3401.
- `mhubert_ft` seed 42: `best_epoch` 10 of 10, `train_time_s` 4396.7, `n_params` 94404779. Dev WER per epoch: 0.8328, 0.6411, 0.5873, 0.5592, 0.5359, 0.5263, 0.5218, 0.5084, 0.5033, 0.5027.

## How I verified the work

- For every run file I recomputed each split with `metrics.evaluate` from the stored predictions. All values match the stored ones exactly, and `metrics.validate_preds` passes for every prediction file.
- Every run file holds the checkpoint, the full revision, the precision, the GPU name and the KSC note.
- `tests/test_eval_zero_shot.py` checks the pinned revisions, the batching, a stand-in CTC model on the fixture clips, the final run's files and that dev mode writes nothing. The full suite passes.
- Both training scripts pass `--smoke` on CPU. Before the full runs, each model learned a few fixture clips by heart and reproduced their transcripts exactly.
- `whisper_large_zs` was run twice, in two separate sessions, after the files of the first run were lost. The second run gave the same `test` and `test_read` WER and CER to four decimals.

## Limitations

- **One seed for `whisper_small_ft`.** Seeds 13 and 42 finished training, but the free Colab usage limit stopped both sessions during final decoding, before any file was written. There was no time to rerun them before submission, so I report no seed spread for this model. `mhubert_ft` also has one seed, by the cut line.
- **Best epoch is the last one** for both fine-tuned models (5 of 5 and 10 of 10). Longer training might still help.
- **Edge of the sweep grid.** Whisper's chosen rate, 3e-5, is the largest rate I tried, so a larger rate might score better.
- **`whisper_large_zs` has no dev or `test_talkshow` score**, because of the cut line.
- **`whisper_small_zs` falls into repetition loops.** Its WER is above 1.0 on dev, `test` and `test_talkshow` because some hypotheses repeat a phrase until the token limit.
- **`test_read`.** `xlsr_kk_zs` saw KSC in training, and `mms_kk_zs` (0.2690) beats `whisper_small_ft` (0.2993) on read speech, which our spontaneous train split does not contain.
- **GPU nondeterminism.** The same seed does not give exactly the same numbers twice on a T4. In two sessions with seed 42, the first-epoch dev WER of `mhubert_ft` differed in the second decimal.
- **Small code-switched slices.** `cs_wer` on `test` rests on 95 code-switched utterances (`data/processed/stats.json`).
