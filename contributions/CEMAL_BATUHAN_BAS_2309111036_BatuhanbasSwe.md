# Cemal Batuhan Bas (2309111036, BatuhanbasSwe)

## What I built

The data pipeline of Aralas, from the KSC2 archive to the split manifests every model reads.

- `src/data_io.py`: the data contract shared by the team: the output alphabet `CHARS`, the split names and the gate that keeps test splits behind `--final`, manifest validation, audio loading, and `normalize_text`, the single text policy applied to references and to every system's hypotheses.
- `src/codeswitch.py`: a rule-based word tagger (`kz`, `ru`, `mix`, `amb`) with a Kazakh lexicon from KSC train text and a Russian lexicon from KSC2 sources we never evaluate on; it defines the code-switching slices.
- `scripts/import_ksc2.py`: selects the spontaneous KSC2 sources and the KSC read test set, applies the program-level split plan, writes `data/raw/` (Opus audio, original transcripts, all KSC2 text, checksums) and can rebuild it with `--check`.
- `scripts/clean_dataset.py`: cleaning rules C1-C9 with a log of every removal, the split manifests with code-switching tags, the lexicons and `stats.json`; `--check` rebuilds `data/processed/` byte for byte.
- `docs/preprocessing.md`, the test fixtures (8 real radio clips) and the tests `test_data_io.py`, `test_codeswitch.py`, `test_clean.py`.

## Key decisions and why

- **Split by program, not by the official KSC2 split.** KSC2 has no speaker IDs and its paper says speaker separation between its splits is not guaranteed; the radio and talk show are single programs whose episodes are spread over its train and test. All episodes of a program now stay in one split. The podcast split was already disjoint by program, so it is used unchanged.
- **The talk show is an unseen-program test set**, because it is one program and cannot be split; the read test set measures the gap between read and spontaneous speech.
- **Speaking-rate outliers by robust z-score.** A first percentile rule removed 1 % of train by definition; the robust z-score only removes real outliers.
- **Checksums of decoded audio.** libsndfile gives each Ogg stream a random serial number, so file bytes are not reproducible but decoded samples are.
- **A conservative tagger.** Single letters, Kazakh-Russian homographs and established loans are never evidence of a switch, after I found that they caused most false alarms in the read set.

## Numbers I produced

From `data/processed/stats.json` and `data/processed/cleaning_log.csv`:

| Split | Utterances | Hours | Code-switched |
| --- | --- | --- | --- |
| train | 22811 | 31.079 | 1301 |
| dev | 1081 | 1.85 | 57 |
| test | 1546 | 2.885 | 95 |
| test_talkshow | 4368 | 6.798 | 329 |
| test_read | 3327 | 7.031 | 12 |

603 utterances removed (C1 1, C4 220, C5 378, C9 4); 0 programs and 0 episodes in more than one split; no dev or test text of at least 4 words left verbatim in train.

Code-switching tagger precision on all 57 code-switched dev utterances: 75 of 79 marked words correct (94.9 %), 53 of 57 utterances correct (93.0 %); details in `docs/preprocessing.md`.

## How I verified the work

- 130 unit tests for the data contract, the tagger and every cleaning rule, including an end-to-end build that must be identical when run twice.
- `import_ksc2.py --check` and `clean_dataset.py --check` rebuild the raw and the processed data from scratch and compare them with the committed files.
- The metrics tests of the evaluation code pass with this `data_io`.
- I read tagged sentences of the test and read sets and tightened the tagger where it was wrong, then audited every code-switched dev utterance by hand.

## Limitations

- No speaker labels: a guest may appear in two podcast programs.
- The code-switched test slices are small (95 and 329 utterances), so their confidence intervals are wide.
- The tagger is rule-based and conservative: it misses some Russian words, so the code-switched counts are a lower bound; it was audited by a member who is not a native speaker of Kazakh or Russian.
- The text policy does not remove elongated-vowel hesitations such as "ааа"; a few remain in the references (4 in test, 21 in test_talkshow).
