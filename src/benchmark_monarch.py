#!/usr/bin/env python3
"""Benchmark Monarch standard and paged inference modes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from inference import MonarchInference
from monarch_paging import MonarchPagingConfig


PRESETS = {
    "fast": {
        "max_new_tokens": 64,
        "temperature": 0.0,
        "repeats": 1,
        "window_size": 256,
        "max_hot_tokens": 384,
        "page_size": 16,
    },
    "accurate": {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "repeats": 3,
        "window_size": 768,
        "max_hot_tokens": 1024,
        "page_size": 32,
    },
}


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text()
    if args.prompt:
        return args.prompt
    raise ValueError("Provide either --prompt or --prompt-file")


def format_metrics(metrics: Dict[str, float]) -> str:
    ordered = sorted(metrics.items())
    return ", ".join(f"{key}={value:.4f}" for key, value in ordered)


def aggregate_runs(runs: List[Dict[str, float]]) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    keys = sorted({key for run in runs for key in run})
    for key in keys:
        values = [run[key] for run in runs if key in run]
        aggregated[key] = statistics.mean(values)
    return aggregated


def compute_delta(
    baseline: Dict[str, float],
    candidate: Dict[str, float],
) -> Dict[str, float]:
    delta: Dict[str, float] = {}
    for key in sorted(set(baseline) | set(candidate)):
        base_value = baseline.get(key, 0.0)
        candidate_value = candidate.get(key, 0.0)
        delta[f"{key}_abs"] = candidate_value - base_value
        if base_value != 0:
            delta[f"{key}_pct"] = ((candidate_value - base_value) / base_value) * 100.0
    return delta


def write_jsonl(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_mode(
    inference: MonarchInference,
    mode: str,
    prompt: str,
    paging: MonarchPagingConfig,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repeats: int,
    trace_dir: Path | None,
) -> Tuple[str, List[Dict[str, float]]]:
    response = ""
    runs: List[Dict[str, float]] = []
    for index in range(repeats):
        trace_rows: List[Dict[str, float]] = []
        if mode == "standard":
            response, metrics = inference.benchmark_standard(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        elif mode == "monarch-v3":
            response, metrics, trace_rows = inference.generate_monarch_v3_with_metrics(
                prompt=prompt,
                paging=paging,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                verbose=False,
                collect_trace=trace_dir is not None,
            )
        else:
            raise ValueError(f"Unknown benchmark mode: {mode}")

        print(f"[{mode}] run={index + 1}/{repeats} {format_metrics(metrics)}")
        if trace_dir is not None and trace_rows:
            trace_path = trace_dir / f"{mode}.run{index + 1}.jsonl"
            write_jsonl(trace_path, trace_rows)
            print(f"[{mode}] trace {trace_path}")
        runs.append(metrics)
    return response, runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Monarch inference modes")
    parser.add_argument("--model", default="models/monarch_lora", help="Path to Monarch LoRA model")
    parser.add_argument("--base-model", default=None, help="Base model override")
    parser.add_argument("--prompt", default=None, help="Prompt text to benchmark")
    parser.add_argument("--prompt-file", default=None, help="Path to a text file used as the prompt")
    parser.add_argument(
        "--preset",
        default=None,
        choices=sorted(PRESETS),
        help="Apply a named benchmark preset before command-line overrides",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["standard", "monarch-v3", "both"],
        help="Which mode to benchmark",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; 0 uses greedy decode")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--repeats", type=int, default=1, help="Number of benchmark runs per mode")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit when CUDA is available")
    parser.add_argument("--window-size", type=int, default=512, help="Hot sliding window for monarch-v3")
    parser.add_argument("--max-hot-tokens", type=int, default=768, help="Hot-token budget for monarch-v3")
    parser.add_argument("--page-size", type=int, default=16, help="Physical KV page size for monarch-v3")
    parser.add_argument(
        "--compression-mode",
        default="turboquant",
        choices=["turboquant", "legacy"],
        help="Cold-page compression scheme for monarch-v3",
    )
    parser.add_argument("--promotion-threshold", type=float, default=0.12, help="Attention threshold for promotion")
    parser.add_argument("--sticky-threshold", type=int, default=2, help="Promotion count needed to make a token sticky")
    parser.add_argument("--importance-decay", type=float, default=0.92, help="Importance decay for paging policy")
    parser.add_argument("--initial-sticky-tokens", type=int, default=16, help="Initial sticky tokens chosen from prompt attention")
    parser.add_argument("--trace-dir", default=None, help="Directory for per-step JSONL traces")
    parser.add_argument("--output", default=None, help="Path to write aggregate benchmark results as JSON")
    parser.add_argument("--json", action="store_true", help="Print aggregate metrics as JSON")

    args = parser.parse_args()
    if args.preset:
        for key, value in PRESETS[args.preset].items():
            if getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    prompt = load_prompt(args)
    paging = MonarchPagingConfig(
        window_size=args.window_size,
        max_hot_tokens=args.max_hot_tokens,
        page_size=args.page_size,
        compression_mode=args.compression_mode,
        attention_promote_threshold=args.promotion_threshold,
        sticky_threshold=args.sticky_threshold,
        importance_decay=args.importance_decay,
        initial_sticky_tokens=args.initial_sticky_tokens,
    )

    inference = MonarchInference(
        model_path=args.model,
        base_model=args.base_model,
        load_in_4bit=args.load_in_4bit,
    )

    modes = ["standard", "monarch-v3"] if args.mode == "both" else [args.mode]
    summary: Dict[str, Dict[str, float]] = {}
    trace_dir = Path(args.trace_dir) if args.trace_dir else None

    for mode in modes:
        response, runs = run_mode(
            inference=inference,
            mode=mode,
            prompt=prompt,
            paging=paging,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repeats=args.repeats,
            trace_dir=trace_dir,
        )
        aggregate = aggregate_runs(runs)
        summary[mode] = aggregate
        print(f"[{mode}] avg {format_metrics(aggregate)}")
        print(f"[{mode}] preview {response[:200].replace(chr(10), ' ')}")

    result: Dict[str, Dict[str, float] | Dict[str, str] | Dict[str, int]] = dict(summary)
    if args.mode == "both" and "standard" in summary and "monarch-v3" in summary:
        result["delta"] = compute_delta(summary["standard"], summary["monarch-v3"])

    result["config"] = {
        "mode": args.mode,
        "preset": args.preset or "",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repeats": args.repeats,
        "window_size": args.window_size,
        "max_hot_tokens": args.max_hot_tokens,
        "page_size": args.page_size,
        "compression_mode": args.compression_mode,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"[output] {output_path}")

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
