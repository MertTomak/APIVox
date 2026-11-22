# Error analysis

Every count in this file is printed by

```
python -m src.build_benchmark --final --errors
```

The command takes the model with the lowest mean test WER in `results/benchmark.csv` and, as the reference point, `whisper_small_zs`. It aligns each normalized hypothesis with its reference word by word (the same `jiwer` alignment `src/metrics.py` uses for WER) and gives every word error exactly one type. For each model and split, the sum over the types equals the WER numerator that `src/metrics.py` computes. Examples of the three largest types go to the system temp directory, not to the repository. Word totals per split come from `data/processed/stats.json`.

Models analysed:

- `mhubert_ft` seed 42: HuBERT base with a CTC head over `data_io.CHARS`, fine-tuned from `utter-project/mHuBERT-147` @ `7ad3fc0bc5106c58c9c13526abccad527150d135`. It is the best model by mean test WER (0.4832 in `results/benchmark.csv`).
- `whisper_small_zs` seed 0: Transformer encoder-decoder, `openai/whisper-small` @ `973afd24965f72e36ca33b3055d56a652f456b4d`, without training.

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

| type | mhubert_ft test | share | mhubert_ft test_talkshow | share | whisper_small_zs test | share | whisper_small_zs test_talkshow | share |
|---|---|---|---|---|---|---|---|---|
| ru_mix_substitution | 93 | 0.0094 | 402 | 0.0191 | 98 | 0.0043 | 349 | 0.0065 |
| ru_mix_deletion | 27 | 0.0027 | 67 | 0.0032 | 29 | 0.0013 | 161 | 0.0030 |
| kz_suffix_only | 2274 | 0.2304 | 5022 | 0.2390 | 1188 | 0.0515 | 3741 | 0.0696 |
| kz_letter_confusion | 109 | 0.0110 | 246 | 0.0117 | 91 | 0.0039 | 439 | 0.0082 |
| other_substitution | 5305 | 0.5375 | 10653 | 0.5070 | 12009 | 0.5209 | 27570 | 0.5129 |
| other_deletion | 1288 | 0.1305 | 2336 | 0.1112 | 4852 | 0.2105 | 10750 | 0.2000 |
| insertion | 774 | 0.0784 | 2286 | 0.1088 | 4787 | 0.2076 | 10739 | 0.1998 |
| **all word errors** | 9870 | | 21012 | | 23054 | | 53749 | |
| reference words | 20428 | | 50659 | | 20428 | | 50659 | |

## Findings

1. **For the fine-tuned model, wrong suffixes are the largest named error type.** `kz_suffix_only` is 2274 of 9870 errors on test (0.2304) and 5022 of 21012 on test_talkshow (0.2390). The model finds the right stem but writes the wrong ending. In an agglutinative language a single wrong suffix makes the whole word wrong, so WER is harsh on these errors; this is why character-level CER is also reported.
2. **Russian and mixed words are rarely recognized correctly.** Test has 114 words tagged `ru` and 20 tagged `mix` out of 20428 (`stats.json`). `mhubert_ft` substitutes 93 and deletes 27 of them (`ru_word_acc` 0.1045 in `benchmark.csv`); `whisper_small_zs` substitutes 98 and deletes 29 (`ru_word_acc` 0.0522). Because these words are rare, they make up a small share of all errors (0.0094 and 0.0027 for `mhubert_ft` on test). Even so, they explain why `cs_wer` is higher than `mono_wer` for both models (0.5296 against 0.4794 for `mhubert_ft` on test).
3. **The zero-shot model loses and invents words.** On test, `whisper_small_zs` deletes 4852 Kazakh or ambiguous words (share 0.2105) and inserts 4787 words (0.2076), against 1288 (0.1305) and 774 (0.0784) for `mhubert_ft`. Insertions are why its WER goes above 1.
4. **Letter confusions between Kazakh-specific and Russian letters are a small part of the errors**: 109 (0.0110) and 246 (0.0117) for `mhubert_ft`. The models mostly keep the Kazakh letters apart from their Russian look-alikes.
5. **Most errors fall outside the named types.** `other_substitution` is about half of all errors for both models (0.5375 and 0.5209 on test). These are substitutions with a different stem, or short words below the 3-character stem limit.

## Limitations

- Only one seed of `mhubert_ft` (42) and the two spontaneous test splits are analysed.
- The suffix rule is a spelling heuristic, not a morphological analysis: a word that differs only in its last 4 characters may also differ in its stem.
- A substitution of a `ru` or `mix` word always counts as `ru_mix_substitution`, even when the hypothesis differs only in a single letter.
