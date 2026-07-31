# Architecture

SplatThis separates optimization from deployment. A conversion produces one
target-independent `SplatScene`; an artifact backend then emits and grades the
requested browser or Office representation.

```text
ConversionRequest + immutable ConverterConfig
                    |
               RunContext
                    |
        prepare -> fit -> deploy
          |         |       |
       image      scene   backend
       analysis           emit/write/evaluate
```

## Run lifecycle

`converter.py` contains only the stable `PNG2SVGConverter` facade. The stateful
numerical implementation lives in `conversion_engine.py`; orchestration does
not belong in the public API module. `ConversionPipeline` snapshots the
constructed configuration and creates an isolated execution copy for every
call. The execution copy may adapt its time budget and refinement settings;
those mutations never ratchet the public converter's next run.

`conversion_engine.py` is itself only a composition root. Internal mixins split
the remaining stateful algorithm by reason to change:

| Module | Responsibility |
|---|---|
| `engine_configuration.py` | constructor, normalization, budgets, renderers and losses |
| `engine_initialization.py` | content-adaptive seeds and initial splat population |
| `engine_optimization.py` | Torch/MLX fitting and checkpoint evaluation |
| `engine_densification.py` | error-driven additions and residual detail passes |
| `engine_postfit.py` | target-aware post-fit, pruning and monotonic selection |
| `engine_artifacts.py` | legacy artifact strategies, fidelity and acceptance |
| `engine_guidance.py` | foreground/background guidance and coverage diagnostics |

These modules are internal implementation boundaries, not additional public
APIs. Cross-responsibility calls use the composed engine instance; deployment
continues to cross the explicit `ConversionPipeline` and artifact-backend
boundaries.

The coordinator has three explicit boundaries:

1. `prepare_input` loads the source, resolves guidance and budgets, and computes
   optional structure fields.
2. `fit_scene` initializes, optimizes, post-fits, and artifact-gates one
   `SplatScene`.
3. `emit_evaluate_and_finalize` emits the primary artifact once, writes it,
   evaluates the governing representation, and records provenance.

Process-level corpus parallelism remains the supported parallel execution
model. Torch's random state and MLX's Metal device ownership are process-wide;
threads must not be used to promise independent seeded training runs.

## Deployment backends

`artifact_backends.py` is the only registry for output targets. A backend owns:

- its media type and default training target;
- target-specific emission and persistence;
- post-fit and fidelity capabilities;
- whether a governing deployed render is required;
- evaluation provenance.

Adding an output format requires a backend and registry entry, not new
`output_format` branches in the converter. SVG, DrawingML, Canvas, CSS, and the
pixel runtime retain separate implementations because their compositors are
not interchangeable.

PPTX emission packages the DrawingML payload already produced by its backend.
It does not regenerate the shape tree during persistence.

## Evidence model

Artifact evaluation uses explicit evidence levels:

- `DEPLOYED`: the emitted artifact captured in its governing renderer;
- `PARITY_MODEL`: an exact or calibrated implementation of deployed math;
- `PROXY`: a diagnostic approximation;
- `UNAVAILABLE`: governing evidence could not be obtained.

Only eligible deployed evidence can satisfy a browser-governed acceptance
check. A proxy fallback remains visible in the manifest but cannot silently
approve an SVG, Canvas, CSS, or static pixel-runtime artifact.

## Dependency direction

The low-level dependency direction is deliberately one-way:

```text
storage <- exporters <- artifact backends <- pipeline <- conversion engine
quality <- browser evaluation -----------^
reporting / roundtrip --------------------^
public converter facade -----------------^
```

`artifact_io.py` and `io.py` are compatibility facades only. Production modules
must import `storage`, `quality`, `artifact_evaluation`, `reporting`, or
`roundtrip` directly. SVG, DrawingML, PPTX package, and report markup lives in
packaged templates.

## Stable values

- `ConversionRequest` contains values that may differ per call.
- `ConverterConfig` is a detached immutable snapshot and has a deterministic
  SHA-256 fingerprint.
- `RunContext` owns timers, RNG setup, artifact paths, and per-run timings.
- `SplatScene` owns dimensions, background, compositing space, and the final
  ordered splat population.

The configuration fingerprint is recorded in every run manifest. It is a
foundation for future phase caching, not yet a complete cache key: package,
template, renderer, and source hashes must also participate before cached
artifacts can be trusted.

## Architectural tests

`tests/unit/test_module_boundaries.py` prevents regressions in these boundaries:

- compatibility facades stay small;
- production code cannot import them;
- vector markup stays in packaged templates;
- the converter facade and engine composition root stay thin;
- configuration remains detached and immutable;
- every supported output format has a registered backend.
