"""The ``e2am`` command line interface.

Commands mirror the Python API and share its code paths — the CLI is a thin
orchestration layer, never a second implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

import e2am
from e2am.config.settings import ExperimentConfig
from e2am.exceptions import E2AMError

if TYPE_CHECKING:  # pragma: no cover
    from e2am.monitoring.result import MonitorResult
    from e2am.trainer.result import TrainingResult

app = typer.Typer(
    name="e2am",
    help="E2AM — Energy Efficient AI Models: automatic energy, carbon, and "
    "performance profiling for AI training.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


def _fail(message: str) -> None:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the E2AM version."""
    console.print(f"e2am {e2am.__version__}")


@app.command()
def hardware(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Detect hardware and its energy-measurement capabilities."""
    from e2am.utils.hardware import detect_hardware

    info = detect_hardware()
    if as_json:
        console.print_json(info.model_dump_json())
        return
    table = Table(title="Detected hardware", title_style="bold green")
    table.add_column("Component", style="bold")
    table.add_column("Details")
    table.add_row("Host", f"{info.hostname} · {info.os} {info.os_version}")
    table.add_row("Python", info.python_version)
    table.add_row(
        "CPU",
        f"{info.cpu.model}\n"
        f"{info.cpu.physical_cores or '?'} cores / {info.cpu.logical_cores or '?'} threads · "
        f"TDP {info.cpu.tdp_w:.0f} W (estimation basis)",
    )
    table.add_row(
        "RAM", f"{info.ram.total_gb:.1f} GB · ~{info.ram.estimated_power_w:.1f} W estimated"
    )
    for gpu in info.gpus:
        sensor = (
            "[green]live power sensor[/green]"
            if gpu.supports_power_reading
            else f"[yellow]no power sensor — estimating at {gpu.tdp_w:.0f} W × util[/yellow]"
        )
        table.add_row(
            f"GPU {gpu.index}",
            f"{gpu.name} · {gpu.total_memory_mb or 0:.0f} MB · "
            f"driver {gpu.driver_version or '?'}\n{sensor}",
        )
    if info.torch_version:
        cuda = f"CUDA {info.cuda_version}" if info.cuda_available else "CPU-only build"
        table.add_row("PyTorch", f"{info.torch_version} · {cuda}")
    else:
        table.add_row("PyTorch", "[yellow]not installed[/yellow] (monitoring still works)")
    console.print(table)


@app.command()
def train(
    script: Path = typer.Argument(..., help="Python file defining get_model() and get_loaders()."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Experiment config YAML."),
    epochs: int | None = typer.Option(None, help="Override epochs."),
    device: str | None = typer.Option(None, help="Override device (cuda/cpu)."),
    run_name: str | None = typer.Option(None, help="Override run name."),
) -> None:
    """Train a model from a script with full E2AM telemetry."""
    from e2am.cli.loader import get_factory, load_user_module
    from e2am.trainer import Trainer

    try:
        module = load_user_module(script)
        get_model = get_factory(module, "get_model")
        get_loaders = get_factory(module, "get_loaders")
        get_optimizer = get_factory(module, "get_optimizer", required=False)
        get_loss = get_factory(module, "get_loss", required=False)

        experiment = ExperimentConfig.from_yaml(config) if config else ExperimentConfig()
        model = get_model()
        loaders = get_loaders()
        train_loader, val_loader = loaders if isinstance(loaders, tuple) else (loaders, None)
        if get_optimizer is not None:
            optimizer = get_optimizer(model)
        else:
            import torch

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=get_loss() if get_loss is not None else None,
            config=experiment,
            epochs=epochs,
            device=device,
            run_name=run_name,
        )
        result = trainer.fit()
    except E2AMError as exc:
        _fail(str(exc))
        return
    console.print(
        f"[green]Run complete:[/green] {result.run_name} · status {result.status} · "
        f"{result.total_energy_wh or 0:.4g} Wh · artifacts in "
        f"{experiment.output.dir / result.run_name}"
    )


@app.command()
def benchmark(
    script: Path = typer.Argument(..., help="Python file defining get_model()."),
    input_size: str = typer.Option(
        ..., "--input-size", "-i", help="Input shape incl. batch, e.g. 8,3,224,224"
    ),
    device: str | None = typer.Option(None, help="Device (default: auto)."),
    iterations: int = typer.Option(50, help="Timed iterations."),
    warmup: int = typer.Option(10, help="Warmup iterations."),
) -> None:
    """Profile FLOPs, latency, throughput, and energy per inference."""
    import torch

    from e2am.cli.loader import get_factory, load_user_module
    from e2am.monitoring.session import MonitorSession
    from e2am.profiler import benchmark_latency, profile_model

    try:
        shape = tuple(int(part) for part in input_size.replace(" ", "").split(","))
    except ValueError:
        _fail(f"--input-size must be comma-separated integers, got {input_size!r}")
        return
    try:
        module = load_user_module(script)
        model = get_factory(module, "get_model")()
        resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(resolved)

        profile = profile_model(model, input_size=shape, device=resolved)
        session = MonitorSession(project="benchmark", run_name=script.stem)
        session.config.sampling_interval_s = 0.2
        session.start()
        latency = benchmark_latency(
            model, input_size=shape, device=resolved, warmup=warmup, iterations=iterations
        )
        monitor_result = session.stop()
    except E2AMError as exc:
        _fail(str(exc))
        return

    energy_j = monitor_result.total_energy_j
    per_inference_j = energy_j / iterations if iterations else 0.0

    table = Table(
        title=f"Benchmark · {profile.model_name} @ {list(shape)} on {latency.device}",
        title_style="bold green",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Parameters", f"{profile.params:,}")
    table.add_row("MACs / forward", f"{profile.macs:,} ({profile.gmacs:.3f} G)")
    table.add_row("FLOPs / forward", f"{profile.flops:,} ({profile.gflops:.3f} G)")
    table.add_row("Model size", f"{profile.model_size_mb:.2f} MB")
    if profile.coverage < 1.0:
        table.add_row("MAC coverage", f"[yellow]{profile.coverage * 100:.0f} %[/yellow]")
    table.add_row("Latency mean", f"{latency.mean_ms:.3f} ms")
    table.add_row("Latency p50 / p95", f"{latency.p50_ms:.3f} / {latency.p95_ms:.3f} ms")
    table.add_row("Throughput", f"{latency.throughput_samples_per_s:.1f} samples/s")
    table.add_row("Energy (whole benchmark)", f"{monitor_result.total_energy_wh:.5f} Wh")
    table.add_row("Energy / inference", f"{per_inference_j:.4f} J")
    console.print(table)


def _load_result(run_dir: Path) -> TrainingResult | MonitorResult:
    from e2am.monitoring.result import MonitorResult
    from e2am.schema import migrate
    from e2am.trainer.result import TrainingResult

    metrics = run_dir / "metrics.json"
    if not metrics.exists():
        _fail(f"No metrics.json in {run_dir}")
    data = json.loads(metrics.read_text(encoding="utf-8"))
    try:
        data = migrate(data, "result")
    except E2AMError as exc:
        _fail(str(exc))
    if "epochs_requested" in data:
        return TrainingResult.model_validate(data)
    return MonitorResult.model_validate(data)


@app.command()
def report(
    run_dir: Path = typer.Argument(..., help="A run directory containing metrics.json."),
    pdf: bool = typer.Option(False, "--pdf", help="Also generate report.pdf."),
) -> None:
    """(Re)generate plots and reports for a finished run."""
    from e2am.config.settings import OutputConfig
    from e2am.reports.generate import generate_run_artifacts

    result = _load_result(run_dir)
    written = generate_run_artifacts(result, run_dir, OutputConfig(save_pdf=pdf))
    for path in written:
        console.print(f"  [green]wrote[/green] {path}")
    console.print(f"[green]Regenerated {len(written)} artifact(s) for {result.run_name}.[/green]")


@app.command()
def optimize(
    run_dir: Path = typer.Argument(..., help="A run directory containing metrics.json."),
) -> None:
    """Suggest efficiency optimizations based on a run's measurements."""
    from e2am.optimize import analyze
    from e2am.trainer.result import TrainingResult

    result = _load_result(run_dir)
    if not isinstance(result, TrainingResult):
        _fail(
            "optimize needs a training run (metrics.json from e2am.Trainer); "
            "this directory holds a monitoring-only run."
        )
        return
    suggestions = analyze(result)
    if not suggestions:
        console.print(
            "[green]No optimization opportunities detected[/green] — this run "
            "already uses the measured hardware efficiently."
        )
        return
    table = Table(
        title=f"Optimization suggestions · {result.project} / {result.run_name}",
        title_style="bold green",
        show_lines=True,
    )
    table.add_column("Priority", style="bold", width=8)
    table.add_column("Suggestion")
    table.add_column("Estimated savings", justify="right")
    colors = {"high": "red", "medium": "yellow", "low": "cyan"}
    for s in suggestions:
        table.add_row(
            f"[{colors.get(s.priority, 'white')}]{s.priority}[/]",
            f"[bold]{s.title}[/bold]\n{s.rationale}\n[dim]{s.action}[/dim]",
            s.savings_label(),
        )
    console.print(table)
    quantified = sum(s.estimated_savings_wh or 0 for s in suggestions)
    if quantified > 0:
        console.print(
            f"[bold]Quantified savings from this run's own data: " f"~{quantified:.4g} Wh[/bold]"
        )


@app.command()
def compare(
    run_dirs: list[Path] = typer.Argument(..., help="Two or more run directories."),
) -> None:
    """Compare runs side by side on their headline metrics."""
    if len(run_dirs) < 2:
        _fail("compare needs at least two run directories.")
    flats = []
    for run_dir in run_dirs:
        result = _load_result(run_dir)
        flats.append(result.to_flat_dict())

    keys: list[str] = []
    for flat in flats:
        for key in flat:
            if key not in keys:
                keys.append(key)

    table = Table(title="Run comparison", title_style="bold green")
    table.add_column("Metric", style="bold")
    for flat in flats:
        table.add_column(str(flat.get("run_name", "?")), justify="right")
    for key in keys:
        if key == "run_name":
            continue
        values = []
        for flat in flats:
            value = flat.get(key)
            values.append("—" if value in (None, "") else str(value))
        table.add_row(key, *values)
    console.print(table)


@app.command()
def dashboard(
    results_dir: Path = typer.Argument(Path("results"), help="Results root directory."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the dashboard in the default browser."
    ),
) -> None:
    """Build a local HTML dashboard of all runs."""
    from e2am.reports.dashboard import generate_dashboard

    try:
        path = generate_dashboard(results_dir)
    except E2AMError as exc:
        _fail(str(exc))
        return
    console.print(f"[green]Dashboard written:[/green] {path}")
    if open_browser:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    app()
