# Data preprocessing

This document describes how `data/raw/` and `data/processed/` are built from ISSAI KSC2. Every number below comes from `data/raw/SOURCES.json`, `data/raw/split_plan.json`, `data/processed/stats.json` or `data/processed/cleaning_log.csv`; the table at the end lists the source of each number.

## 1. Sources and licenses

| Source (KSC2 folder) | Content | Our use |
| --- | --- | --- |
| `podcasts` | 9 podcast programs, host and guest conversations | train, dev and test (official split, disjoint by program) |
| `radio` | one radio program: news, call-ins, interviews | train only |
| `talkshow` | one TV talk show | `test_talkshow` only |
| `Test/crowdsourced` | the read-speech test set of KSC v1.1 | `test_read` only |
| all `.txt` files | transcripts of every KSC2 source, text only | Kazakh and Russian word lexicons |

The corpus is ISSAI KSC2 (Mussakhojayeva, Khassanov and Varol, Interspeech 2022), distributed as `issai/Kazakh_Speech_Corpus_2` on Hugging Face. Its license statements are quoted in `data/raw/SOURCES.json`: MIT on the Hugging Face card, CC BY 4.0 in the GitHub recipe (`IS2AI/Kazakh_ASR`) and CC BY 4.0 for KSC on OpenSLR 102. All transcripts were made by native Kazakh and Russian transcribers of ISSAI; we never write or edit a reference.

## 2. Import (`scripts/import_ksc2.py`)

The archive (10 parts, listed with their sizes in `SOURCES.json`) was streamed once and only the folders above and all transcripts were extracted, outside the repository. `import_ksc2.py --plan` prints the split plan; `--write` builds `data/raw/`:

- `audio/<source>/<id>.opus`: every selected segment, decoded from FLAC and written as 16 kHz mono Opus at libsndfile compression level 0.93 (24 kbps for this speech). Segments are not cut, trimmed or resampled.
- `transcripts/<source>.jsonl`: id, program, episode, official KSC2 split, the transcript exactly as distributed (`text_orig`), duration, the sha256 of the original FLAC and a `decode_error` field.
- `text/ksc2_all_text.jsonl.gz`: 645861 transcripts of the whole archive, text only (written with a fixed gzip timestamp so that it is reproducible).
- `split_plan.json`, `SOURCES.json` and `checksums.txt`.

Ids and grouping units come from the file names: a podcast segment `09_03_415` becomes id `podcasts_09_03_415` with program `podcasts_09` and episode `podcasts_09_03`.

One FLAC file of the read test set, `Test/crowdsourced/5f5efa02744de.flac`, cannot be decoded ("flac decoder lost sync"). The importer keeps its transcript row without audio and rule C1 removes it.

`checksums.txt` holds the sha256 of every text file and, for every Opus file, the sha256 of its decoded 16-bit PCM. libsndfile gives each Ogg stream a random serial number, so the same audio encoded twice gives different bytes but identical decoded samples. `import_ksc2.py --check` rebuilds everything into a temporary folder and compares these checksums.

## 3. Splits

KSC2 does not label speakers, and its paper states that speaker separation between its splits "is not fully guaranteed". Following the course rule for corpora without speaker IDs, we group by the closest unit, the **program**: all episodes of a program, and therefore its hosts, stay in one split.

- **Podcasts**: the official KSC2 split is already disjoint by program (train 01, 03, 04, 05, 06, 08; dev 02, 07; test 09), so we use it unchanged.
- **Radio** and **talk show** are one program each; their official splits put episodes of the same show into train and test. Radio therefore goes entirely to train and the talk show entirely to `test_talkshow`.
- **Size**: whole episodes are taken in a seeded random order (seed 2026) until train reaches 32 h (radio plus the podcast train programs) and `test_talkshow` reaches 6 h; the remaining talk show episodes are not used. The selected episodes are listed in `split_plan.json`.

Statistics after cleaning (`stats.json`, `splits`):

| Split | Utterances | Hours | Programs | Episodes | Code-switched utterances |
| --- | --- | --- | --- | --- | --- |
| train | 22811 | 31.079 | 7 | 78 | 1301 |
| dev | 1081 | 1.85 | 2 | 8 | 57 |
| test | 1546 | 2.885 | 1 | 5 | 95 |
| test_talkshow | 4368 | 6.798 | 1 | 9 | 329 |
| test_read | 3327 | 7.031 | 1 | 1 | 12 |

We report programs and episodes instead of speakers because the corpus has no speaker labels. Leakage checks (`stats.json`, `leakage`): 0 programs and 0 episodes occur in more than one split. Dev or test utterances whose text occurs verbatim in train: 4 before cleaning (2 of them with at least 4 words); after cleaning 1, and none with at least 4 words.

Known limitation: a guest could appear in two different podcast programs; the corpus gives no way to check this.

Known limitation of the text policy: the hesitation list removes repeated "ы", "і", "м" and "э", but not elongated vowels such as "ааа", because a single "а" is a real word. A few such tokens remain in the references (test 4, test_talkshow 21, dev 3, train 195, test_read 0; counted from the manifests with the pattern `а{2,}|ә{2,}|о{2,}|у{2,}|е{2,}`). They are applied identically to every system, and the data is frozen at tag `data-v1`, so they stay.

## 4. Cleaning rules (`scripts/clean_dataset.py`)

Each rule logs every removed utterance to `cleaning_log.csv` with the measured value and the threshold. Fixed thresholds are in the `THRESHOLDS` dictionary at the top of the script; the C4 bounds are computed from train only and applied to every split.

| Rule | What it removes | Real problem | Removed |
| --- | --- | --- | --- |
| C1 | audio that cannot be decoded, is not 16 kHz mono, or has no transcript | broken files | 1 (test_read: the FLAC above) |
| C2 | shorter than 0.5 s or longer than 30 s | truncated segments; segments longer than the 30 s Whisper window | 0 |
| C3 | empty reference after normalization | segments that contain only hesitations | 0 |
| C4 | speaking rate (letters per second) with a robust z-score above 4 on the log scale (median and 1.4826 × MAD of train) | transcript and audio do not match: wrong segment boundaries, missing text, or a segment that is mostly music or silence | 220 (train 213, dev 1, test 1, test_talkshow 5) |
| C5 | more than 0.1 % of samples at full scale | clipping | 378 (all train) |
| C6 | energy-based speech ratio below 0.3 (librosa, top_db 40) | long silence, near-silent recordings | 0 |
| C7 | identical decoded audio; the first id in sorted order is kept | duplicated segments | 0 |
| C8 | clips rejected in the listening check (`results/human_qc.csv`) | music, cut-off or overlapped clips found by ear | 0: the data was frozen at tag `data-v1` before the listening check, so the check is reported in the README as a quality check and does not change the data |
| C9 | train utterance whose text equals a dev or test text and has at least 4 words | text leakage between splits | 4 (train) |

In total 603 utterances were removed. What the counts show:

- **C4**: all 220 removals are too slow (below 3.0014 letters per second; the upper bound, 46.2789, removed nothing), and 202 of them come from the radio program. A transcript that is short for its audio fits a segment that is mostly music, a jingle or silence. Energy alone cannot separate music from speech, so this rule, not C6, is the one that catches music.
- **C5**: all 378 removals come from one program, `podcasts_01`. A spot check of removed clips against the original FLAC files found the same share of samples at ±32767/32768 there, so the clipping is in the recording, not an effect of the Opus encoding.
- The C4 rule replaces a first design that removed the 0.5 % slowest and fastest train utterances. A percentile rule removes 1 % of train by definition, whether or not those utterances have a problem; the robust z-score removes only real outliers.

## 5. Text normalization

`src/data_io.normalize_text` is the only text policy. It is applied to references when the manifests are built and to every system's hypotheses before scoring, in this order:

1. lowercase; `ё` → `е`;
2. digit groups → Kazakh number words with `num2words(lang="kz")` (2020 → "екі мың жиырма");
3. drop spans in `()`, `[]`, `{}`; hyphens and dashes → space;
4. every character outside `CHARS` → space (`CHARS` = space, the 32 Russian letters without `ё`, the 9 Kazakh letters `әғқңөұүһі`);
5. drop hesitation tokens (`ы+`, `і+`, `м+`, `мхм`, `хм`, `э+`); "а" is kept because it is a real Kazakh and Russian word;
6. collapse spaces.

The KSC2 transcripts are already lowercase with no punctuation, digits or Latin letters, so in practice the policy removes hesitations: 12666 hesitation tokens over all selected utterances. The original transcript is kept in `text_orig`.

## 6. Code-switching tags (`src/codeswitch.py`)

Each word of a reference gets a tag `kz`, `ru`, `mix` or `amb`; an utterance is code-switched (`cs`) if it has at least one `kz` word and at least one `ru` or `mix` word. The tags never change a reference; they define the evaluation slices.

- **Kazakh lexicon**: words of the KSC train transcripts (read Kazakh), 111372 words with count at least 2.
- **Russian lexicon**: words of Russian sentences in the `parliament` and `tv_news` transcripts (no Kazakh-specific letter and at least two unambiguous Russian function words), 7559 words. We use no other source in our splits, so no evaluation text enters a lexicon.
- **Word rule**, in order: single letters and Kazakh–Russian homographs (а, о, от, там, тот, ну, том, не, да, же, …) are `amb`; a Russian stem of at least 5 letters followed by a chain of Kazakh suffixes is `mix`; a Kazakh-specific letter gives `kz`; an unambiguous Russian function word, or a word only in the Russian lexicon, gives `ru`; a word only in the Kazakh lexicon gives `kz`; a word in both is `ru` or `kz` only if it is 20 times more frequent in that language. A word (or stem) seen at least 10 times in the Kazakh lexicon is treated as an established loan ("кодекс", "офис") and is never `ru` or `mix`.

Two details of this rule were changed after looking at the tagged data. A plain "in one lexicon but not the other" rule missed words such as "уже" that also occur a few times in the Kazakh lexicon, because KSC read text quotes some Russian. And single letters, Kazakh "о" and established loans produced false alarms, mostly in the read set. 

### Precision audit

`python -m src.codeswitch --audit` lists the code-switched dev utterances with every `ru` and `mix` word marked. The audit was planned for 100 sentences, but dev has only 57 code-switched utterances, so all 57 were checked by hand with a dictionary. They contain 79 marked words:

- 75 are correct Russian words or Russian stems with Kazakh suffixes ("в общем", "уже", "может быть", "хотя бы", "двенадцать недель в году", "составтың", "школалардың").
- 4 are wrong: the surname "карпович" (twice, `podcasts_07_02_022` and `podcasts_07_02_206`), the hesitation "ааа" (`podcasts_07_03_156`) and "акиматта" (`podcasts_07_02_167`), the Russian spelling of the Kazakh administrative term "әкімдік".
- Russian words that Kazakh speakers also use, but that have a Kazakh equivalent and are rare in read Kazakh text ("план", "минимум", "характер", "штраф", "этап", "период", "тренд"), were counted as correct: choosing the Russian word over the Kazakh one is a switch.

**Word-level precision: 75 / 79 = 94.9 %.** **Utterance-level precision: 53 / 57 = 93.0 %** (the four wrong words are the only marked words of their utterances). The audit also showed missed Russian words, for example "заслуживают", "комфортный", "недель" and "друзей": the tagger is conservative, so the code-switched counts are a lower bound. The audit was done by a team member who is not a native speaker of Kazakh or Russian.

## 7. Rebuilding

```
python scripts/import_ksc2.py --ksc2 <path to ISSAI_KSC2> --plan
python scripts/import_ksc2.py --ksc2 <path to ISSAI_KSC2> --write
python scripts/clean_dataset.py
python scripts/clean_dataset.py --check
```

`clean_dataset.py --check` rebuilds `data/processed/` into a temporary folder and compares every file byte by byte. `--candidates` prints the clips for the listening check (a seeded sample of 60 kept clips plus up to 40 clips near a C4 bound), and `--fixtures` refreshes `tests/fixtures/` with 8 short train radio clips.

## Sources of the numbers

| Number | Source |
| --- | --- |
| 645861 transcripts | `data/raw/SOURCES.json`: `transcripts_in_all_text` |
| per-split utterances, hours, programs, episodes, code-switched utterances | `stats.json`: `splits` |
| leakage counts | `stats.json`: `leakage` |
| removed per rule and split, 603 in total | `stats.json`: `removed_per_rule`, `removed_total`; `cleaning_log.csv` |
| C4 bounds 3.0014 and 46.2789 | `stats.json`: `computed_thresholds` |
| 202 C4 removals from radio; 378 C5 removals from `podcasts_01` | `cleaning_log.csv` grouped by program |
| 12666 hesitation tokens | `stats.json`: `hesitation_tokens_removed_by_normalization` |
| lexicon sizes 111372 and 7559 | `stats.json`: `lexicon_sizes` |
| 57 audited utterances, 79 marked words, 75 correct, 53 utterances correct | the audit list printed by `python -m src.codeswitch --audit` and the four wrong words named in section 6 |
| leftover elongated-vowel tokens (4, 21, 3, 195, 0) | the manifests in `data/processed/`, counted with the pattern in section 3 |
