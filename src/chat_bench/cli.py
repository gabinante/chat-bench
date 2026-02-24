"""CLI for ChatBench."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from .models import MODELS, get_model_config
from .runner import evaluate_task, load_task, print_results_table
from .schemas import EvalResult

console = Console()


@click.group()
def main():
    """ChatBench — benchmark for chat/conversational retrieval models."""
    pass


@main.command()
@click.argument("model_path", required=False, default=None)
@click.option("--bm25", is_flag=True, help="Use BM25 lexical baseline instead of a neural model")
@click.option("--tasks-dir", default=None, help="Local tasks dir (auto-downloads if not set)")
@click.option("--task", "task_name", default=None, help="Run a specific task only")
@click.option("--batch-size", default=128, type=int)
@click.option("--output", default=None, help="Save results JSON to this path")
@click.option("--robustness", is_flag=True, help="Run robustness eval")
@click.option("--n-paraphrases", default=5, type=int, help="Paraphrase variants")
@click.option("--llm-paraphrase", is_flag=True, help="Use LLM paraphrasing")
@click.option("--quiet", "-q", is_flag=True, help="Suppress console output (for scripted usage)")
def evaluate(
    model_path: str | None, bm25: bool, tasks_dir: str, task_name: str | None,
    batch_size: int, output: str | None,
    robustness: bool, n_paraphrases: int, llm_paraphrase: bool,
    quiet: bool,
):
    """Evaluate a model on ChatBench tasks."""
    if quiet:
        console.quiet = True
        # Also suppress runner console output
        from . import runner
        runner.console.quiet = True

    if not bm25 and not model_path:
        console.print("[red]Provide a MODEL_PATH or use --bm25[/]")
        return

    model = None
    display_name = "bm25"
    model_config = get_model_config(model_path) if model_path else None
    if not bm25:
        console.print(f"[bold]Loading model: {model_path}[/]")
        if model_path and model_path.startswith("text-embedding-"):
            from .openai_embed import OpenAIEmbedder
            model = OpenAIEmbedder(model=model_path)
        else:
            from sentence_transformers import SentenceTransformer
            load_kwargs: dict = {}
            if model_config and model_config.get("trust_remote_code"):
                load_kwargs["trust_remote_code"] = True
            model = SentenceTransformer(model_path, **load_kwargs)
        display_name = model_path

    if tasks_dir:
        tasks_path = Path(tasks_dir)
        if not tasks_path.exists():
            console.print(f"[red]Tasks directory not found: {tasks_path}[/]")
            return
    else:
        from .data import get_tasks_dir
        tasks_path = get_tasks_dir()

    task_files = sorted(tasks_path.glob("*.json"))
    if task_name:
        task_files = [f for f in task_files if task_name in f.stem]

    if not task_files:
        console.print("[yellow]No task files found.[/]")
        return

    results = []
    for tf in task_files:
        task = load_task(tf)
        result = evaluate_task(
            model, task, model_name=display_name,
            batch_size=batch_size, use_bm25=bm25,
            model_config=model_config,
        )

        if robustness:
            from .robustness import evaluate_robustness
            console.print(f"\n[bold cyan]Robustness evaluation for {task.task_name}[/]")
            rob = evaluate_robustness(
                model, task,
                n_paraphrases=n_paraphrases,
                use_llm=llm_paraphrase,
                batch_size=batch_size,
                use_bm25=bm25,
                model_config=model_config,
            )
            result.robustness_score = rob.robustness_score
            result.metric_std_devs = rob.metric_std_devs
            result.mean_query_stability = rob.mean_query_stability
            result.n_paraphrases = rob.n_paraphrases

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
@click.option("--include-registry", is_flag=True, help="Include all registered models")
@click.option("--tasks-dir", default=None, help="Local tasks dir (auto-downloads if not set)")
@click.option("--batch-size", default=128, type=int)
@click.option("--quiet", "-q", is_flag=True, help="Suppress console output (for scripted usage)")
def compare(models: tuple[str, ...], include_registry: bool, tasks_dir: str, batch_size: int,
            quiet: bool):
    """Compare multiple models on ChatBench."""
    if quiet:
        console.quiet = True
        from . import runner
        runner.console.quiet = True

    from sentence_transformers import SentenceTransformer

    # Separate neural models from lexical baselines
    neural_models: list[str] = list(models)
    lexical_models: list[str] = []

    if include_registry:
        for name, info in MODELS.items():
            if info.get("type") == "lexical":
                lexical_models.append(info["model_id"])
            else:
                neural_models.append(info["model_id"])

    if not neural_models and not lexical_models:
        console.print("[yellow]No models specified. Use --models or --include-registry[/]")
        return

    if tasks_dir:
        tasks_path = Path(tasks_dir)
    else:
        from .data import get_tasks_dir
        tasks_path = get_tasks_dir()

    task_files = sorted(tasks_path.glob("*.json"))

    all_results: list[EvalResult] = []

    # Evaluate neural models
    for model_path in neural_models:
        console.print(f"\n[bold]{'='*60}[/]")
        console.print(f"[bold green]Model: {model_path}[/]")
        model_config = get_model_config(model_path)
        load_kwargs: dict = {}
        if model_config and model_config.get("trust_remote_code"):
            load_kwargs["trust_remote_code"] = True
        model = SentenceTransformer(model_path, **load_kwargs)

        for tf in task_files:
            task = load_task(tf)
            result = evaluate_task(
                model, task, model_name=model_path,
                batch_size=batch_size, model_config=model_config,
            )
            all_results.append(result)

    # Evaluate lexical baselines (BM25)
    for model_id in lexical_models:
        console.print(f"\n[bold]{'='*60}[/]")
        console.print(f"[bold green]Baseline: {model_id}[/]")

        for tf in task_files:
            task = load_task(tf)
            result = evaluate_task(None, task, model_name=model_id, use_bm25=True)
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
    table.add_row("conversation_similarity", "Find conversations about similar topics", "Semantic similarity")
    table.add_row("cross_platform", "Thread retrieval on held-out platform data", "Generalization")
    table.add_row("topic_retrieval", "Find conversations by topic description", "Topic search")
    table.add_row("specific_detail", "Find conversations with a specific detail", "Detail search")
    table.add_row("cross_channel", "Find related conversations across channels", "Cross-channel")
    table.add_row("thread_discrimination", "Distinguish similar conversations", "Fine-grained")

    console.print(table)


@main.command()
@click.option(
    "--phase",
    type=click.Choice(["A", "B", "C", "D", "E", "F", "all"], case_sensitive=False),
    default="all",
    help="Which generation phase to run",
)
@click.option("--model", default="claude-sonnet-4-20250514", help="Anthropic model ID")
@click.option("--resume/--no-resume", default=True, help="Resume from previous state")
def generate(phase: str, model: str, resume: bool):
    """Generate the benchmark corpus using Claude API."""
    from .generate.pipeline import run_pipeline

    try:
        phases = None if phase.lower() == "all" else [phase.upper()]
        run_pipeline(phases=phases, resume=resume, model=model)
    except ImportError as e:
        if "anthropic" in str(e):
            console.print("[red]The 'anthropic' package is required for generation.[/]")
            console.print("Install with: [bold]pip install 'chat-bench[generate]'[/]")
        else:
            raise


@main.command()
@click.option("--corpus-dir", default="data/corpus", help="Directory with conversations.jsonl")
@click.option("--queries-dir", default="data/queries", help="Directory with query JSONL files")
@click.option("--output-dir", default="data/tasks", help="Output directory for task JSON files")
@click.option("--include-disco", is_flag=True, help="Include DISCO real Discord conversations")
@click.option("--disco-max-per-channel", default=None, type=int,
              help="Max DISCO conversations per channel")
def build(corpus_dir: str, queries_dir: str, output_dir: str,
          include_disco: bool, disco_max_per_channel: int | None):
    """Build benchmark tasks from generated corpus."""
    from .build import build_all_tasks

    build_all_tasks(
        corpus_dir=corpus_dir, queries_dir=queries_dir, output_dir=output_dir,
        include_disco=include_disco, disco_max_per_channel=disco_max_per_channel,
    )


if __name__ == "__main__":
    main()
