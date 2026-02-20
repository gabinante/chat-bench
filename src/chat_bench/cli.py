"""CLI for ChatBench."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from sentence_transformers import SentenceTransformer

from .baselines import BASELINES
from .runner import evaluate_task, load_task, print_results_table
from .schemas import EvalResult

console = Console()


@click.group()
def main():
    """ChatBench — benchmark for chat/conversational retrieval models."""
    pass


@main.command()
@click.argument("model_path")
@click.option("--tasks-dir", default="data/tasks", help="Directory containing task JSON files")
@click.option("--task", "task_name", default=None, help="Run a specific task only")
@click.option("--batch-size", default=128, type=int)
@click.option("--output", default=None, help="Save results JSON to this path")
def evaluate(model_path: str, tasks_dir: str, task_name: str | None, batch_size: int, output: str | None):
    """Evaluate a model on ChatBench tasks."""
    console.print(f"[bold]Loading model: {model_path}[/]")
    model = SentenceTransformer(model_path)

    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        console.print(f"[red]Tasks directory not found: {tasks_path}[/]")
        console.print("Run 'chat-bench build' first to create benchmark tasks.")
        return

    task_files = sorted(tasks_path.glob("*.json"))
    if task_name:
        task_files = [f for f in task_files if task_name in f.stem]

    if not task_files:
        console.print("[yellow]No task files found.[/]")
        return

    results = []
    for tf in task_files:
        task = load_task(tf)
        result = evaluate_task(model, task, model_name=model_path, batch_size=batch_size)
        results.append(result)

    print_results_table(results)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([r.model_dump() for r in results], f, indent=2)
        console.print(f"\n[green]Results saved to {out_path}[/]")


@main.command()
@click.option("--models", multiple=True, help="Model paths/IDs to compare")
@click.option("--include-baselines", is_flag=True, help="Include all baseline models")
@click.option("--tasks-dir", default="data/tasks")
@click.option("--batch-size", default=128, type=int)
def compare(models: tuple[str, ...], include_baselines: bool, tasks_dir: str, batch_size: int):
    """Compare multiple models on ChatBench."""
    model_list = list(models)

    if include_baselines:
        for name, info in BASELINES.items():
            model_list.append(info["model_id"])

    if not model_list:
        console.print("[yellow]No models specified. Use --models or --include-baselines[/]")
        return

    tasks_path = Path(tasks_dir)
    task_files = sorted(tasks_path.glob("*.json"))

    all_results: list[EvalResult] = []

    for model_path in model_list:
        console.print(f"\n[bold]{'='*60}[/]")
        console.print(f"[bold green]Model: {model_path}[/]")
        model = SentenceTransformer(model_path)

        for tf in task_files:
            task = load_task(tf)
            result = evaluate_task(model, task, model_name=model_path, batch_size=batch_size)
            all_results.append(result)

    console.print(f"\n[bold]{'='*60}[/]")
    print_results_table(all_results)


@main.command(name="list")
def list_tasks():
    """List available benchmark tasks."""
    from rich.table import Table

    table = Table(title="ChatBench Tasks")
    table.add_column("Task ID", style="cyan")
    table.add_column("Description")
    table.add_column("Use Case")

    table.add_row("thread_retrieval", "Find the correct thread for a message", "Thread search")
    table.add_row("response_retrieval", "Find the continuation of a conversation", "Temporal coherence")
    table.add_row("summary_matching", "Match a description to a conversation", "Semantic search")
    table.add_row("cross_platform", "Thread retrieval on held-out platform data", "Generalization")

    console.print(table)


if __name__ == "__main__":
    main()
