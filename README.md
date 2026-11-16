# APIVox

Speech recognition for spoken Python API references: a short audio clip such as "use data frame dot group by to split the data" is transcribed as "use DataFrame.groupby to split the data", with the API name in its exact code form.

## Repository structure

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview and this table. |
| `LICENSE` | License of the repository. |
| `requirements.txt` | Pinned Python dependencies. |
| `.gitignore` | Files that git must not track. |
| `.gitattributes` | Marks audio and inventory files as binary. |
| `data/` | Raw inputs, synthesized and recorded audio, and the cleaned dataset. |
| `docs/` | Written documentation of the data pipeline. |
| `scripts/` | Scripts that build the dataset: prompts, synthesis, recording and cleaning. |
| `src/` | Features, models, training, evaluation and benchmark code. |
| `tests/` | Unit tests for the code in `src/` and `scripts/`. |
| `results/` | Run files, predictions, benchmark tables, figures and error analysis. |
| `contributions/` | One file per team member describing their work. |
