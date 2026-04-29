# Exam Generator Usage

This tool has two end-user modes:

1. Generate a new exam from the question bank.
2. Process completed exam results against a marking sheet.

## Requirements

- Python 3.8+
- The dependencies from [setup.py](/Users/felixschulz/Dropbox/AdvancedHealthEconomics/Exam/bot/setup.py)
- A LaTeX PDF engine such as `pdflatex` for the default PDF backend
- A question bank Excel file with the required columns used by the bot

Install the package dependencies from the `bot` directory:

```bash
python -m pip install -e .
```

## Generate An Exam

Run the generator with the central settings file:

```bash
python main.py
```

Or point at a different YAML/JSON override file:

```bash
python main.py --config config/exam_settings.yaml
```

What happens:

- The default exam settings are loaded from `config/exam_settings.yaml`.
- If you pass another config file, only the fields you override need to be present.
- Relative paths in the config are resolved relative to the config file location.
- The question bank is read from the configured worksheet, which defaults to `v2`.
- Questions are selected with the `unified` sampler.
- The selection implementation lives in `src/selection/sampler.py`, with semantic diversity helpers in `src/selection/semantic_search.py`.
- A timestamped output folder is created unless `io.timestamp_output: false`.
- The default LaTeX backend writes student and instructor `.tex` sources plus compiled `.pdf` files, a marking sheet, a copy of the config, a selected-questions Excel export, and a log file.
- If you switch to `document.backend: word`, the bot writes `.docx` files instead.

Typical outputs appear under the configured `io.output_dir`, for example:

- `output/exam_20260423_153000/...student....tex`
- `output/exam_20260423_153000/...student....pdf`
- `output/exam_20260423_153000/...instructor....tex`
- `output/exam_20260423_153000/...instructor....pdf`
- `output/exam_20260423_153000/...marking_sheet....xlsx`
- `output/exam_20260423_153000/selected_questions_....xlsx`
- `output/exam_20260423_153000/config_....yaml`
- `output/exam_20260423_153000/exam_generator_unified.log`

## Process Results

Run the results processor with the students' answer file and the marking sheet:

```bash
python main.py \
  --results ../results/2025/Results_2025.xlsx \
  --marking-sheet output/exam_20260423_153000/Advanced_Health_Economics_-_Final_Exam_2025_marking_sheet_20260423_153000.xlsx
```

Optional output controls:

```bash
python main.py \
  --results ../results/2025/Results_2025.xlsx \
  --marking-sheet output/exam_20260423_153000/example_marking_sheet.xlsx \
  --results-output-dir ../results/2025 \
  --results-output-name 74283_AHE_2025
```

This mode writes summary files and analysis outputs into the requested results output directory.

## Config Guide

The full editable settings file lives at [config/exam_settings.yaml](/Users/felixschulz/Dropbox/AdvancedHealthEconomics/Exam/bot/config/exam_settings.yaml).
The sample override file lives at [examples/config_examples/default.yaml](/Users/felixschulz/Dropbox/AdvancedHealthEconomics/Exam/bot/examples/config_examples/default.yaml).

Important fields:

- `io.question_bank_path`: Excel source file for questions.
- `io.question_bank_sheet`: Worksheet to read from the question bank. The default is `v2`.
- `io.output_dir`: Base output folder.
- `io.timestamp_output`: If `true`, create `exam_<timestamp>` subfolders.
- `exam.title`: Title used in generated filenames and documents.
- `exam.instructions`: Instructions shown in the generated exam.
- `selection.method`: Must currently be `unified`.
- `selection.seed`: Makes question and answer randomization reproducible.
- `selection.target_points`: Total exam points to allocate exactly.
- `selection.topics`: Allowed topic pool. The sampler chooses topics from this list dynamically instead of enforcing exact topic ratios.
- `selection.type_ratios`: Exact point split across question types. Each type gets `target_points * ratio / sum(ratios)`, and every result must be an integer.
- `selection.max_per_type`: Caps the number of questions per type. The run fails if the exact ratio requires more.
- `selection.semantic.cache_dir`: Cache for sentence embeddings.
- `grading.points_per_type`: Point value per question type.
- `document.backend`: Output backend. The default is `latex`, which writes `.tex` and compiles `.pdf` with a direct LaTeX engine call.
- `document.pdf_engine`: LaTeX engine executable for the PDF backend. The default is `pdflatex`.
- `document.randomize_answers`: Randomizes answer order and carries that mapping into the marking sheet.
- `document.block_questions.image_directory`: Figure directory used for embedded media.

## Question Bank Expectations

The reader expects at least these columns:

- `Topic`
- `Type`
- `Question`
- `Answer 1`
- `Answer 2`
- `Answer 3`
- `Answer 4`
- `Correct`
- `Medium`

Optional but useful columns:

- `ID`
- `BID`
- `Check: Armando`
- `2025 Exam`

Block questions are inferred either from legacy block-style `ID` values like `22-1`, `22-2`, `22-3`, or from the newer `ID` + `BID` columns in `v2`.

## Notes

- Embeddings are cached by model and invalidated when the question bank file changes.
- The bot only supports the `unified` selection method at the moment.
- The default document backend is LaTeX/PDF. The legacy Word backend is still available via `document.backend: word`.
- The selected-questions export is a standalone sheet of the chosen rows; the source question-bank data is not annotated with a selection column.
- Selection is exact for type ratios. Topics are chosen from `selection.topics`, and the run fails if the requested type mix cannot be satisfied precisely with the available questions and block structure.
- If `--results` is used, `--marking-sheet` is required.
- If you do not pass `--config`, the bot uses `config/exam_settings.yaml`.
- Paths in the config can be relative; they are resolved from the config file folder, not the shell working directory.

## Troubleshooting

- `Configuration file not found`: check the path passed to `--config`.
- `Question bank file not found`: verify `io.question_bank_path` inside the config.
- `Unsupported selection method`: change `selection.method` to `unified`.
- `LaTeX engine is required`: install `pdflatex` or another configured TeX engine, or switch to `document.backend: word`.
- `cannot be achieved precisely`: adjust `selection.target_points`, `selection.type_ratios`, `selection.topics`, or `grading.points_per_type`.
- Slow first run: the sentence-transformer model and embeddings cache may need to be created first.
