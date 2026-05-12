from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """Prompt evaluation CLI."""


@cli.command()
def evaluate() -> None:
    """Evaluate prompts."""
    click.echo("evaluate command is available")


@cli.command()
def export() -> None:
    """Export saved results."""
    click.echo("export command is available")


@cli.command("list-models")
def list_models() -> None:
    """List supported models."""
    click.echo("list-models command is available")


if __name__ == "__main__":
    cli()
