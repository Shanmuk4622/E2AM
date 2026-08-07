# Changelog

All notable changes to E2AM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Groundwork for v2. Contains breaking changes, each with a migration.

### Fixed

- **Gradient clipping was skipped on a trailing partial accumulation window.**
  When `len(train_loader) % gradient_accumulation_steps != 0`, the flush step
  called `scaler.step()` without `unscale_()` + `clip_grad_norm_`, so every
  epoch took one unclipped optimizer step. Both paths now share
  `Trainer._optimizer_step()`.
- **`duration_s` was measured on the wall clock while energy was integrated on
  the monotonic clock.** A DST or NTP step mid-run desynced the two and
  corrupted `avg_total_power_w` (an injected one-hour jump made a 0.2 s run
  report "0.01 Wh over 3600.2 s"). Duration now comes from the same monotonic
  time base the integrator uses.
- **Carbon intensity used a sentinel-by-value.** "The user set it" was inferred
  from the value differing from the 475 g/kWh world average, so a user in a
  475 g/kWh grid could not pin that number — their country code silently
  overrode it.

### Changed (breaking)

- `CarbonConfig.carbon_intensity_g_per_kwh` is now `float | None` defaulting to
  `None` ("not specified") instead of `float` defaulting to `475.0`. Explicit
  values — including 475.0 — now always win over the country lookup.

### CI

- **CI was running roughly two thirds of the test suite and reporting green.**
  `pip install -e ".[dev]"` never brought PyTorch (it is deliberately a peer
  dependency), so every torch-backed test file skipped itself: 53 of 171 tests
  — the whole `Trainer`, all of `profiler/`, classification metrics, every
  plugin, the Hugging Face integration, and `e2am train`/`benchmark` — had
  never executed on a runner. The test matrix now installs CPU-only PyTorch and
  transformers, and asserts they are importable so a broken install fails loudly
  instead of silently reverting to a hollow green run.
- Added a dedicated `test-without-torch` job. The "monitoring works without
  PyTorch" promise used to hold only by accident, because every runner happened
  to lack torch; it is now covered on purpose, including an `import e2am` +
  `monitor()` + CLI smoke test.
- Added a guard that `import e2am` does not eagerly import torch, which only
  has teeth on a runner where torch is installed.
- Added pip caching and `concurrency: cancel-in-progress` to offset the added
  install time.

### Added

- **Artifact schema versioning** (`e2am.schema`). `metrics.json` and
  `config.yaml` carry a `schema_version`; artifacts written by 0.1.x are
  detected as v1 and migrated forward on load, preserving the meaning they
  were written with. The v1→v2 config migration translates the old 475.0
  sentinel back to `None`, so a legacy config with a country code still
  resolves to that country's grid. Artifacts from a *newer* E2AM are refused
  with an upgrade hint rather than silently misread.

## [0.1.0] — 2026-07-11

First public release. 🌱

### Added

**Monitoring**
- `with monitor(project=...)` context manager / decorator — zero-code-change
  energy, carbon, and utilization tracking for any code block.
- Background `MonitorSession` with per-device samplers: GPU (NVML live power
  sensor, TDP × utilization fallback for cards without one), CPU
  (TDP × utilization), RAM (~3 W per 8 GB used); live `snapshot()` totals.
- Trapezoidal power→energy integration robust to irregular intervals and
  missing readings; measured-vs-estimated flags on every device.
- Region-aware carbon estimation (27-country intensity table, user override,
  world-average fallback).

**Profiling & metrics**
- `profile_model`: hook-based MACs/FLOPs/params with an extensible counter
  registry and honest parameter-coverage reporting.
- `benchmark_latency`: warmed-up, CUDA-synchronized latency (mean/p50/p95)
  and throughput. `MemoryTracker`: exact CUDA peak + host RSS delta.
- Torch-native classification metrics (accuracy, macro/weighted P/R/F1).
- Green AI metrics: energy/carbon per sample, accuracy-per-joule,
  Green Score (`100·acc·E_ref/(E_ref+E)`), EAG (energy-accuracy gradient).

**Training**
- Drop-in `Trainer` with AMP, gradient accumulation, clipping, LR scheduler
  support, callback lifecycle, `EarlyStopping`, and automatic integration of
  monitoring + profiling + green metrics.

**Outputs**
- Automatic artifacts per run: 10 plots, self-contained `report.html`,
  `README.md`, optional `report.pdf`, `metrics.json`, `summary.yaml`,
  `config.yaml`, cross-run `leaderboard.csv`, local `dashboard.html`.

**CLI**
- `e2am hardware | train | benchmark | report | compare | optimize | dashboard`.
- `benchmark` reports energy per inference (joules) via live monitoring.
- `optimize`: rule-based efficiency suggestions with quantified Wh savings
  (wasted-epoch detection, AMP, batch size, torch.compile, checkpointing,
  quantization).

**Integrations**
- Plugins (all Trainer callbacks): Weights & Biases, MLflow, TensorBoard,
  Slack, Discord (webhooks are stdlib-only).
- Hugging Face: `E2AMCallback` for `transformers.Trainer`.

### Notes
- Python ≥ 3.10. PyTorch is a peer dependency (install the build matching
  your hardware); monitoring works without it.
