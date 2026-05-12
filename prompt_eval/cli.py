from __future__ import annotations

import logging
from pathlib import Path

import click

from .database import DatabaseManager
from .evaluator import PromptEvaluator

SUPPORTED_MODELS = [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
]


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s", force=True)


@click.group()
def cli() -> None:
    """Prompt evaluation CLI."""


@cli.command()
@click.option("--dataset", "datasets", type=click.Path(exists=True), required=True, multiple=True)
@click.option("--model", default="gpt-4", type=click.Choice(SUPPORTED_MODELS))
@click.option("--db-path", default="results.db", show_default=True)
@click.option("--config", "config_path", default="config.yaml", show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("--resume", is_flag=True, help="Resume evaluation and skip existing results.")
@click.option("--cost-summary", is_flag=True, help="Show total tokens and cost after evaluation.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def evaluate(
    datasets: tuple[str, ...],
    model: str,
    db_path: str,
    config_path: str,
    limit: int | None,
    resume: bool,
    cost_summary: bool,
    verbose: bool,
) -> None:
    """Evaluate one or multiple prompt datasets."""
    _configure_logging(verbose)
    total_tokens = 0
    total_cost = 0.0
    summary: dict[str, int] = {}
    with PromptEvaluator(db_path=db_path, config_path=config_path) as evaluator:
        for dataset in datasets:
            results = evaluator.evaluate_dataset(dataset, model=model, limit=limit, resume=resume)
            summary[dataset] = len(results)
            for result in results:
                total_tokens += getattr(result, "tokens_used", 0) or 0
                total_cost += getattr(result, "cost_usd", 0.0) or 0.0
        for dataset, count in summary.items():
            click.echo(f"Evaluated {count} prompts from {dataset}")
    if cost_summary:
        click.echo(f"Total tokens: {total_tokens}")
        click.echo(f"Total cost: ${total_cost:.4f}")


@cli.command()
@click.option("--dataset", type=click.Path(exists=True), required=True)
@click.option("--models", multiple=True, required=True, type=click.Choice(SUPPORTED_MODELS))
@click.option("--db-path", default="results.db", show_default=True)
@click.option("--config", "config_path", default="config.yaml", show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("--resume", is_flag=True, help="Resume evaluation and skip existing results.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def compare(
    dataset: str,
    models: tuple[str, ...],
    db_path: str,
    config_path: str,
    limit: int | None,
    resume: bool,
    verbose: bool,
) -> None:
    """Compare models on a dataset and display average scores."""
    _configure_logging(verbose)
    with PromptEvaluator(db_path=db_path, config_path=config_path) as evaluator:
        results_by_model = evaluator.compare_models(
            dataset_path=dataset,
            models=list(models),
            limit=limit,
            resume=resume,
        )
        for model in models:
            results = results_by_model.get(model, [])
            if not results:
                click.echo(f"Average score for {model}: n/a")
                continue
            avg_score = sum(evaluator.scorer.weighted_total(r.score) for r in results) / len(results)
            click.echo(f"Average score for {model}: {avg_score:.4f}")


@cli.command()
@click.option("--db-path", default="results.db", show_default=True)
@click.option("--output", type=click.Path(), required=True)
@click.option("--format", "fmt", type=click.Choice(["csv", "jsonl"]), required=True)
@click.option("--dataset", default=None)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def export(db_path: str, output: str, fmt: str, dataset: str | None, verbose: bool) -> None:
    """Export evaluation results."""
    _configure_logging(verbose)
    output_path = Path(output)
    with DatabaseManager(db_path) as db:
        if fmt == "csv":
            db.export_csv(output_path, dataset=dataset)
        else:
            db.export_jsonl(output_path, dataset=dataset)
    click.echo(f"Exported {fmt} to {output}")


if __name__ == "__main__":
    cli()
