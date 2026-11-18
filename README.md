# Aralas

Speech recognition for spontaneous Kazakh with Kazakh-Russian code-switching: a podcast, radio or talk show segment is transcribed in Cyrillic, with Kazakh words in Kazakh and Russian words in Russian.

## Repository structure

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview and this table. |
| `requirements.txt` | Pinned Python dependencies. |
| `.gitignore` | Files that git must not track. |
| `.gitattributes` | Marks audio and compressed files as binary. |
| `data/` | Raw KSC2 and KSC audio and transcripts, and the cleaned split manifests (created by later tasks). |
| `docs/` | Documentation of the data pipeline (created by later tasks). |
| `results/` | Run files, predictions, benchmark tables, figures and error analysis (created by later tasks). |
| `contributions/` | One file per team member describing their work. |
| `scripts/` | Scripts that build the dataset. |
| `scripts/import_ksc2.py` | Imports the spontaneous part of KSC2 and the KSC read test set into `data/raw/`. |
| `scripts/clean_dataset.py` | Cleans the raw data and writes the split manifests, statistics and cleaning log. |
| `scripts/listen_qc.py` | Plays sampled clips for listening quality control and writes `results/human_qc.csv`. |
| `src/` | Data loading, features, models, training, evaluation and benchmark code. |
| `src/data_io.py` | Loads and validates manifests, decodes audio and normalizes text. |
| `src/codeswitch.py` | Builds the Kazakh and Russian lexicons and tags each word as kz, ru, mix or amb. |
| `src/features.py` | Computes log-mel and MFCC features and audio augmentation. |
| `src/models.py` | Defines the CTC, attention encoder-decoder and RNN-T models trained from scratch. |
| `src/metrics.py` | Computes WER, CER, code-switching slices, bootstrap confidence intervals and result files. |
| `src/gmm_hmm.py` | Trains and evaluates the GMM-HMM baseline on MFCC features. |
| `src/eval_zero_shot.py` | Evaluates pretrained models without training. |
| `src/train_scratch.py` | Trains and evaluates the models built from scratch. |
| `src/train_whisper.py` | Fine-tunes a pretrained Whisper Transformer encoder-decoder. |
| `src/train_mhubert.py` | Fine-tunes a pretrained mHuBERT encoder with a CTC head on our character set. |
| `src/build_benchmark.py` | Builds the benchmark and ablation tables and figures from stored run files. |
| `tests/` | Unit tests for the code in `src/` and `scripts/`. |
| `tests/fixtures/` | Small audio clips and a mini manifest used by the tests. |
