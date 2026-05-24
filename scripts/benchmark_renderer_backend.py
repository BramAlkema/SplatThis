#!/usr/bin/env python
"""Benchmark SplatThis renderer backends on a fixed raw-splat JSON file."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from png2svg_gs.io import load_splats_json  # noqa: E402
from png2svg_gs.mlx_renderer import (  # noqa: E402
    MlxBatchedGaussianRenderer,
    is_mlx_available,
    splats_to_numpy_table,
)
from png2svg_gs.renderer import create_renderer, splats_to_tensor  # noqa: E402


def _parse_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_int_csv(text: str) -> List[int]:
    values = []
    for item in _parse_csv(text):
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("integer CSV values must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer value")
    return values


def _parse_background(text: str) -> List[float]:
    parts = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--background must have exactly three comma values"
        )
    return [float(np.clip(v, 0.0, 1.0)) for v in parts]


def _infer_dimensions(
    image_path: Optional[Path],
    width: Optional[int],
    height: Optional[int],
    max_edge: Optional[int],
) -> Tuple[int, int]:
    if width is not None and height is not None:
        return int(width), int(height)
    if image_path is None:
        raise SystemExit("pass --width/--height or --input")

    with Image.open(image_path) as image:
        src_w, src_h = image.size
    if max_edge is None:
        return int(width or src_w), int(height or src_h)

    scale = float(max_edge) / float(max(src_w, src_h))
    inferred_w = max(1, int(round(src_w * scale)))
    inferred_h = max(1, int(round(src_h * scale)))
    return int(width or inferred_w), int(height or inferred_h)


def _torch_device_available(device_name: str) -> Tuple[bool, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency guard.
        return False, f"torch import failed: {exc}"

    normalized = device_name.strip().lower()
    if normalized == "cpu":
        return True, ""
    if normalized == "mps":
        try:
            if torch.backends.mps.is_available():
                return True, ""
        except Exception as exc:  # pragma: no cover - defensive runtime guard.
            return False, f"MPS availability check failed: {exc}"
        return False, "torch MPS is not available"
    if normalized == "cuda":
        return (torch.cuda.is_available(), "CUDA is not available")
    try:
        torch.device(normalized)
    except Exception as exc:
        return False, f"invalid torch device: {exc}"
    return True, ""


def _sync_torch(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _summarize_times(times: Sequence[float]) -> Dict[str, Any]:
    if not times:
        return {
            "runs_s": [],
            "median_s": None,
            "mean_s": None,
            "min_s": None,
            "max_s": None,
        }
    return {
        "runs_s": [round(float(v), 6) for v in times],
        "median_s": round(float(statistics.median(times)), 6),
        "mean_s": round(float(statistics.mean(times)), 6),
        "min_s": round(float(min(times)), 6),
        "max_s": round(float(max(times)), 6),
    }


def _display_path(path: Path) -> str:
    text = str(path)
    if path.is_absolute() or text.startswith("."):
        return text
    return f"./{text}"


def _benchmark_torch(
    splats: Sequence[Any],
    *,
    backend: str,
    device_name: str,
    width: int,
    height: int,
    tile_size: int,
    batch_tile_count: Optional[int],
    blend_mode: str,
    background: Sequence[float],
    warmup: int,
    repeat: int,
    include_backward: bool,
    max_active_splats_per_tile: Optional[int],
) -> Dict[str, Any]:
    import torch

    available, reason = _torch_device_available(device_name)
    row: Dict[str, Any] = {
        "backend": backend,
        "device": device_name,
        "tile_size": tile_size,
        "batch_tile_count": batch_tile_count,
        "mode": "forward_backward" if include_backward else "forward",
    }
    if not available:
        row.update({"status": "skipped", "reason": reason})
        return row

    try:
        device = torch.device(device_name)
        tensor = splats_to_tensor(list(splats), device=device).float()
        if include_backward:
            tensor = tensor.detach().clone().requires_grad_(True)
        renderer_kwargs: Dict[str, Any] = {
            "backend": backend,
            "width": width,
            "height": height,
            "device": device,
            "tile_size": tile_size,
            "blend_mode": blend_mode,
            "background_color": background,
        }
        if batch_tile_count is not None:
            renderer_kwargs["batch_tile_count"] = batch_tile_count
        if max_active_splats_per_tile is not None:
            renderer_kwargs["max_active_splats_per_tile"] = max_active_splats_per_tile
        renderer = create_renderer(**renderer_kwargs)

        def step() -> float:
            if include_backward and tensor.grad is not None:
                tensor.grad.zero_()
            _sync_torch(device)
            started = time.perf_counter()
            image = renderer(tensor)
            if include_backward:
                loss = image.mean()
                loss.backward()
            _sync_torch(device)
            return time.perf_counter() - started

        for _ in range(warmup):
            step()
        times = [step() for _ in range(repeat)]
        row.update({"status": "ok", **_summarize_times(times)})
        return row
    except Exception as exc:
        row.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
        return row


def _benchmark_mlx(
    table_np: np.ndarray,
    *,
    width: int,
    height: int,
    tile_size: int,
    batch_tile_count: int,
    blend_mode: str,
    background: Sequence[float],
    warmup: int,
    repeat: int,
    include_backward: bool,
    max_active_splats_per_tile: Optional[int],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "backend": "mlx-batched",
        "device": "mlx-default",
        "tile_size": tile_size,
        "batch_tile_count": batch_tile_count,
        "mode": "forward_backward" if include_backward else "forward",
    }
    if not is_mlx_available():
        row.update({"status": "skipped", "reason": "MLX is not installed"})
        return row

    try:
        import mlx.core as mx

        renderer = MlxBatchedGaussianRenderer(
            width=width,
            height=height,
            tile_size=tile_size,
            batch_tile_count=batch_tile_count,
            blend_mode=blend_mode,
            background_color=background,
            max_active_splats_per_tile=max_active_splats_per_tile,
        )
        plan = renderer.build_plan(table_np)
        table = mx.array(table_np)

        if include_backward:

            def loss_fn(current_table: Any) -> Any:
                return mx.mean(renderer.render(current_table, plan=plan))

            value_and_grad = mx.value_and_grad(loss_fn)

            def step() -> float:
                started = time.perf_counter()
                loss, grad = value_and_grad(table)
                mx.eval(loss, grad)
                return time.perf_counter() - started

        else:

            def step() -> float:
                started = time.perf_counter()
                image = renderer.render(table, plan=plan)
                mx.eval(image)
                return time.perf_counter() - started

        for _ in range(warmup):
            step()
        times = [step() for _ in range(repeat)]
        row.update(
            {
                "status": "ok",
                "plan_max_active": int(plan.max_active),
                "plan_tiles": int(plan.tiles_x * plan.tiles_y),
                **_summarize_times(times),
            }
        )
        try:
            row["mlx_device"] = str(mx.default_device())
        except Exception:
            pass
        return row
    except Exception as exc:
        row.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
        return row


def _iter_benchmark_jobs(
    backends: Sequence[str],
    devices: Sequence[str],
    tile_sizes: Sequence[int],
    batch_tile_counts: Sequence[int],
) -> Iterable[Tuple[str, Optional[str], int, Optional[int]]]:
    for backend in backends:
        normalized = backend.strip().lower().replace("_", "-")
        if normalized == "torch":
            for device in devices:
                for tile_size in tile_sizes:
                    yield normalized, device, tile_size, None
        elif normalized == "torch-batched":
            for device in devices:
                for tile_size in tile_sizes:
                    for batch_tile_count in batch_tile_counts:
                        yield normalized, device, tile_size, batch_tile_count
        elif normalized == "mlx-batched":
            for tile_size in tile_sizes:
                for batch_tile_count in batch_tile_counts:
                    yield normalized, None, tile_size, batch_tile_count
        else:
            yield normalized, None, tile_sizes[0], batch_tile_counts[0]


def _print_table(rows: Sequence[Dict[str, Any]]) -> None:
    print(
        "backend        device       tile  batch  mode              status   median_ms"
    )
    print(
        "-------------  -----------  ----  -----  ----------------  -------  ---------"
    )
    for row in rows:
        median = row.get("median_s")
        median_ms = "" if median is None else f"{float(median) * 1000.0:.2f}"
        print(
            f"{row.get('backend',''):<13}  "
            f"{row.get('device',''):<11}  "
            f"{str(row.get('tile_size','')):<4}  "
            f"{str(row.get('batch_tile_count','')):<5}  "
            f"{row.get('mode',''):<16}  "
            f"{row.get('status',''):<7}  "
            f"{median_ms:>9}"
        )
        if row.get("status") not in {"ok", "skipped"} and row.get("reason"):
            print(f"  reason: {row['reason']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splats-json", required=True, type=Path)
    parser.add_argument(
        "--input", type=Path, help="Optional source image used to infer geometry."
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--max-edge",
        type=int,
        help="Resize inferred input dimensions to this max edge.",
    )
    parser.add_argument(
        "--backends",
        default="torch-batched,mlx-batched",
        help="Comma list: torch,torch-batched,mlx-batched.",
    )
    parser.add_argument(
        "--devices", default="cpu,mps", help="Comma list of torch devices."
    )
    parser.add_argument("--tile-sizes", default="16", type=_parse_int_csv)
    parser.add_argument("--batch-tile-counts", default="16,32", type=_parse_int_csv)
    parser.add_argument(
        "--blend-mode", default="alpha-over", choices=["alpha-over", "weighted"]
    )
    parser.add_argument("--background", default="0,0,0", type=_parse_background)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--include-backward", action="store_true")
    parser.add_argument("--max-active-splats-per-tile", type=int)
    parser.add_argument(
        "--out", type=Path, default=Path("./tmp/renderer_backend_benchmark.json")
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    width, height = _infer_dimensions(
        args.input, args.width, args.height, args.max_edge
    )
    splats = load_splats_json(str(args.splats_json))
    table_np = splats_to_numpy_table(splats)
    backends = _parse_csv(args.backends)
    devices = _parse_csv(args.devices)

    rows: List[Dict[str, Any]] = []
    for backend, device, tile_size, batch_tile_count in _iter_benchmark_jobs(
        backends,
        devices,
        args.tile_sizes,
        args.batch_tile_counts,
    ):
        if backend in {"torch", "torch-batched"}:
            if device is None:
                continue
            row = _benchmark_torch(
                splats,
                backend=backend,
                device_name=device,
                width=width,
                height=height,
                tile_size=tile_size,
                batch_tile_count=batch_tile_count,
                blend_mode=args.blend_mode,
                background=args.background,
                warmup=max(0, args.warmup),
                repeat=max(1, args.repeat),
                include_backward=bool(args.include_backward),
                max_active_splats_per_tile=args.max_active_splats_per_tile,
            )
        elif backend == "mlx-batched":
            row = _benchmark_mlx(
                table_np,
                width=width,
                height=height,
                tile_size=tile_size,
                batch_tile_count=int(batch_tile_count or 1),
                blend_mode=args.blend_mode,
                background=args.background,
                warmup=max(0, args.warmup),
                repeat=max(1, args.repeat),
                include_backward=bool(args.include_backward),
                max_active_splats_per_tile=args.max_active_splats_per_tile,
            )
        else:
            row = {
                "backend": backend,
                "device": device,
                "tile_size": tile_size,
                "batch_tile_count": batch_tile_count,
                "status": "skipped",
                "reason": f"unknown backend: {backend}",
            }
        rows.append(row)
        status = row.get("status", "?")
        median_s = row.get("median_s")
        if median_s is None:
            print(
                f"{backend} {device or 'mlx-default'} tile={tile_size} batch={batch_tile_count}: {status}"
            )
        else:
            print(
                f"{backend} {device or 'mlx-default'} tile={tile_size} batch={batch_tile_count}: "
                f"{float(median_s) * 1000.0:.2f} ms"
            )

    payload = {
        "splats_json": _display_path(args.splats_json),
        "num_splats": len(splats),
        "width": width,
        "height": height,
        "repeat": max(1, args.repeat),
        "warmup": max(0, args.warmup),
        "include_backward": bool(args.include_backward),
        "python": sys.version,
        "platform": platform.platform(),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _print_table(rows)
    print(f"wrote {_display_path(args.out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
