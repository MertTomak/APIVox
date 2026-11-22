# Error analysis

Every count in this file is printed by

```
python -m src.build_benchmark --final --errors
```

The command takes the model with the lowest mean test WER in `results/benchmark.csv` and, as the reference point, `whisper_small_zs`. It aligns each normalized hypothesis with its reference word by word (the same `jiwer` alignment `src/metrics.py` uses for WER) and gives every word error exactly one type. For each model and split, the sum over the types equals the WER numerator that `src/metrics.py` computes. Examples of the three largest types go to the system temp directory, not to the repository. Word totals per split come from `data/processed/stats.json`.

Models analysed:

- `whisper_small_ft` seed 2026 (the only seed with a run file): Transformer encoder-decoder, `openai/whisper-small` @ `973afd24965f72e36ca33b3055d56a652f456b4d`, fine-tuned on our train split. It is the best model by mean test WER (0.3272 in `results/benchmark.csv`).
- `whisper_small_zs` seed 0: the same checkpoint without training.

The two runs share the checkpoint, so the comparison shows what fine-tuning on spontaneous Kazakh changes.

## Error types

| type | definition |
|---|---|
| `ru_mix_substitution` | a reference word tagged `ru` or `mix` is replaced by another word |
| `ru_mix_deletion` | a reference word tagged `ru` or `mix` is missing from the hypothesis |
| `kz_suffix_only` | any other substituted word whose hypothesis shares a stem of at least 3 characters and differs only in the last 4 characters (the longer of the two words counts) |
| `kz_letter_confusion` | any other substituted word that becomes identical to the hypothesis after mapping қ/к, ғ/г, ң/н, ү/у, ұ/у, ө/о, ә/а, і/и, һ/х (checked before `kz_suffix_only`) |
| `other_substitution` | every remaining substitution |
| `other_deletion` | a missing reference word tagged `kz` or `amb` |
| `insertion` | a hypothesis word with no reference word |

## Counts

Share = count / all word errors of that model and split.

| type | whisper_small_ft test | share | whisper_small_ft test_talkshow | share | whisper_small_zs test | share | whisper_small_zs test_talkshow | share |
|---|---|---|---|---|---|---|---|---|
| ru_mix_substitution | 48 | 0.0072 | 183 | 0.0134 | 98 | 0.0043 | 349 | 0.0065 |
| ru_mix_deletion | 9 | 0.0013 | 24 | 0.0018 | 29 | 0.0013 | 161 | 0.0030 |
| kz_suffix_only | 1521 | 0.2276 | 3217 | 0.2357 | 1188 | 0.0515 | 3741 | 0.0696 |
| kz_letter_confusion | 75 | 0.0112 | 126 | 0.0092 | 91 | 0.0039 | 439 | 0.0082 |
| other_substitution | 3328 | 0.4979 | 6215 | 0.4554 | 12009 | 0.5209 | 27570 | 0.5129 |
| other_deletion | 807 | 0.1207 | 1339 | 0.0981 | 4852 | 0.2105 | 10750 | 0.2000 |
| insertion | 896 | 0.1341 | 2542 | 0.1863 | 4787 | 0.2076 | 10739 | 0.1998 |
| **all word errors** | 6684 | | 13646 | | 23054 | | 53749 | |
| reference words | 20428 | | 50659 | | 20428 | | 50659 | |

## Findings

1. **Fine-tuning removes most errors of every type.** On test, the errors fall from 23054 to 6684. Kazakh deletions fall from 4852 to 807 and insertions from 4787 to 896: the fine-tuned model no longer drops words or loops.
2. **Wrong suffixes are now the largest named error type.** `kz_suffix_only` is 1521 of 6684 errors on test (0.2276) and 3217 of 13646 on test_talkshow (0.2357). The model finds the right stem but writes the wrong ending. In an agglutinative language a single wrong suffix makes the whole word wrong, so WER is harsh on these errors; this is why character-level CER is also reported.
3. **Russian and mixed words improve but remain hard.** Test has 114 words tagged `ru` and 20 tagged `mix` (`stats.json`). `whisper_small_ft` substitutes 48 and deletes 9 of them (`ru_word_acc` 0.5746 in `benchmark.csv`); `whisper_small_zs` substitutes 98 and deletes 29 (`ru_word_acc` 0.0522). `cs_wer` stays above `mono_wer` (0.3592 against 0.3246 on test).
4. **Letter confusions between Kazakh-specific and Russian letters are a small part of the errors**: 75 (0.0112) on test and 126 (0.0092) on test_talkshow.
5. **Most errors fall outside the named types.** `other_substitution` is about half of all errors (0.4979 on test). These are substitutions with a different stem, or short words below the 3-character stem limit.

## Limitations

- Only one seed of `whisper_small_ft` (2026) and the two spontaneous test splits are analysed.
- The suffix rule is a spelling heuristic, not a morphological analysis: a word that differs only in its last 4 characters may also differ in its stem.
- A substitution of a `ru` or `mix` word always counts as `ru_mix_substitution`, even when the hypothesis differs only in a single letter.
