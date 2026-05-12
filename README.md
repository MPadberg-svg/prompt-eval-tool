# prompt-eval-tool

A Python CLI tool for evaluating LLM prompts with batch support.

## Features

- Click-based CLI
- JSONL prompt dataset loading
- OpenAI model calls (`gpt-4`, `gpt-3.5-turbo`)
- Scoring across correctness, safety, helpfulness, and reasoning (0-10)
- SQLite result persistence
- Export to CSV and JSONL
- Configurable scoring weights via `config.yaml`

## Project structure

- `prompt_eval/` (`evaluator.py`, `models.py`, `scoring.py`, `database.py`, `cli.py`)
- `prompts/` sample JSONL datasets (coding, reasoning, multilingual)
- `tests/` pytest unit tests
- `config.yaml`
- `requirements.txt`
- `setup.py`

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.10+.

## Usage

Set API key:

```bash
export OPENAI_API_KEY=your_key
```

Evaluate one or more datasets:

```bash
prompt-eval evaluate --dataset prompts/coding.jsonl --dataset prompts/reasoning.jsonl --model gpt-4 --db-path results.db
```

Export results:

```bash
prompt-eval export --db-path results.db --format csv --output results.csv
prompt-eval export --db-path results.db --format jsonl --output results.jsonl
```

Run tests:

```bash
pytest -q
```
