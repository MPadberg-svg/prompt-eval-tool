from __future__ import annotations

from pathlib import Path

import click

from .database import DatabaseManager
from .evaluator import PromptEvaluator

SUPPORTED_MODELS = ["gpt-4", "gpt-3.5-turbo"]


@click.group()
def cli() -> None:
    """Prompt evaluation CLI."""


@cli.command()
@click.option("--dataset", "datasets", type=click.Path(exists=True), required=True, multiple=True)
@click.option("--model", default="gpt-4", type=click.Choice(SUPPORTED_MODELS))
@click.option("--db-path", default="results.db", show_default=True)
@click.option("--config", "config_path", default="config.yaml", show_default=True)
@click.option("--limit", type=int, default=None)
def evaluate(datasets: tuple[str, ...], model: str, db_path: str, config_path: str, limit: int | None) -> None:
    """Evaluate one or multiple prompt datasets."""
    with PromptEvaluator(db_path=db_path, config_path=config_path) as evaluator:
        summary = evaluator.evaluate_batch(datasets, model=model, limit=limit)
        for dataset, count in summary.items():
            click.echo(f"Evaluated {count} prompts from {dataset}")


@cli.command()
@click.option("--db-path", default="results.db", show_default=True)
@click.option("--output", type=click.Path(), required=True)
@click.option("--format", "fmt", type=click.Choice(["csv", "jsonl"]), required=True)
@click.option("--dataset", default=None)
def export(db_path: str, output: str, fmt: str, dataset: str | None) -> None:
    """Export evaluation results."""
    output_path = Path(output)
    with DatabaseManager(db_path) as db:
        if fmt == "csv":
            db.export_csv(output_path, dataset=dataset)
        else:
            db.export_jsonl(output_path, dataset=dataset)
    click.echo(f"Exported {fmt} to {output}")


if __name__ == "__main__":
    cli()
