#!/usr/bin/env python3
"""Summarize Monarch benchmark traces and emit simple SVG plots."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def load_trace(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def summarize_trace(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "steps": float(len(rows)),
        "tokens_per_sec": mean(row["step_tokens_per_sec"] for row in rows),
        "latency_per_token_ms": mean(row["step_sec"] * 1000.0 for row in rows),
        "resident_hot_tokens": mean(row["resident_hot_tokens"] for row in rows),
        "desired_hot_tokens": mean(row["desired_hot_tokens"] for row in rows),
        "retention_ratio": mean(
            (row["resident_hot_tokens"] / row["sequence_length"]) if row["sequence_length"] else 0.0
            for row in rows
        ),
        "promotion_frequency": mean(row["promotions_delta"] for row in rows),
        "page_hit_rate": mean(row["page_hit"] for row in rows),
        "page_miss_rate": mean(row["page_miss"] for row in rows),
        "avg_attention_score": mean(row["avg_attention_score"] for row in rows),
        "avg_importance_ema": mean(row["avg_importance_ema"] for row in rows),
        "peak_vram_mb": max((row["peak_vram_mb"] for row in rows), default=0.0),
    }


def svg_line_plot(
    rows: List[Dict[str, float]],
    x_key: str,
    y_keys: List[str],
    labels: List[str],
    title: str,
    output_path: Path,
) -> None:
    width = 960
    height = 320
    margin = 40
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    x_values = [row[x_key] for row in rows]
    y_values = [row[key] for key in y_keys for row in rows]
    if not x_values or not y_values:
        return

    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    if max_x == min_x:
        max_x += 1.0
    if max_y == min_y:
        max_y += 1.0

    def sx(value: float) -> float:
        return margin + (value - min_x) / (max_x - min_x) * (width - 2 * margin)

    def sy(value: float) -> float:
        return height - margin - (value - min_y) / (max_y - min_y) * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="24" font-size="18" font-family="monospace">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
    ]

    for idx, key in enumerate(y_keys):
        points = " ".join(f"{sx(row[x_key]):.2f},{sy(row[key]):.2f}" for row in rows)
        parts.append(
            f'<polyline fill="none" stroke="{colors[idx % len(colors)]}" stroke-width="2" points="{points}"/>'
        )
        parts.append(
            f'<text x="{width - 220}" y="{margin + 18 * (idx + 1)}" font-size="12" font-family="monospace" fill="{colors[idx % len(colors)]}">{labels[idx]}</text>'
        )

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts))


def load_results(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def render_metrics_table(title: str, metrics: Dict[str, float]) -> str:
    rows = []
    for key, value in sorted(metrics.items()):
        rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{value:.4f}</td>"
            "</tr>"
        )
    return (
        f"<section class='card'><h2>{html.escape(title)}</h2>"
        "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def render_config_card(config: Dict[str, object]) -> str:
    compression_mode = html.escape(str(config.get("compression_mode", "unknown")))
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(key))}</td>"
        f"<td>{html.escape(str(value))}</td>"
        "</tr>"
        for key, value in sorted(config.items())
    )
    return (
        "<section class='card featured'>"
        "<h2>config</h2>"
        f"<p class='pill'>compression mode: {compression_mode}</p>"
        "<table><thead><tr><th>Field</th><th>Value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def build_dashboard(
    output_dir: Path,
    summaries: Dict[str, Dict[str, float]],
    results: Optional[Dict[str, object]],
) -> None:
    cards: List[str] = []
    if results:
        result_sections: List[str] = []
        for section in ["standard", "monarch-v3", "delta"]:
            payload = results.get(section)
            if isinstance(payload, dict):
                numeric_payload = {
                    str(key): float(value)
                    for key, value in payload.items()
                    if isinstance(value, (int, float))
                }
                result_sections.append(render_metrics_table(section, numeric_payload))
        if result_sections:
            cards.append("<section class='span-3 compare'>" + "".join(result_sections) + "</section>")
        config = results.get("config")
        if isinstance(config, dict):
            cards.append(render_config_card(config))

    aggregate = summaries.get("_aggregate", {})
    if aggregate:
        cards.append(render_metrics_table("trace aggregate", aggregate))

    plot_items: List[str] = []
    for svg_path in sorted(output_dir.glob("*.svg")):
        plot_items.append(
            "<figure class='plot'>"
            f"<img src='{html.escape(svg_path.name)}' alt='{html.escape(svg_path.name)}' />"
            f"<figcaption>{html.escape(svg_path.name)}</figcaption>"
            "</figure>"
        )

    dashboard = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Monarch Benchmark Dashboard</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --ink: #1d1a16;
      --card: #fffdf8;
      --accent: #9f3a20;
      --line: #d8cfc1;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: radial-gradient(circle at top, #fff9ef, var(--bg));
      color: var(--ink);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 64px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 2.6rem;
      letter-spacing: -0.03em;
    }}
    .lede {{
      margin: 0 0 28px;
      color: #5c554b;
      font-size: 1.05rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(29, 26, 22, 0.06);
    }}
    .featured {{
      background: linear-gradient(180deg, #fff7ea, #fffdf8);
    }}
    .span-3 {{
      grid-column: 1 / -1;
    }}
    .compare {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      background: transparent;
      border: none;
      box-shadow: none;
      padding: 0;
    }}
    .pill {{
      display: inline-block;
      margin: 0 0 12px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #f1dfcc;
      color: var(--accent);
      font-size: 0.88rem;
      font-family: "SFMono-Regular", Consolas, monospace;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 1.1rem;
      text-transform: lowercase;
      color: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      text-align: left;
      padding: 7px 0;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th:last-child, td:last-child {{
      text-align: right;
      font-family: "SFMono-Regular", Consolas, monospace;
    }}
    .plots {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .plot {{
      margin: 0;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 10px 30px rgba(29, 26, 22, 0.06);
    }}
    .plot img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 10px;
      background: white;
    }}
    .plot figcaption {{
      margin-top: 10px;
      font-size: 0.9rem;
      color: #5c554b;
      font-family: "SFMono-Regular", Consolas, monospace;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Monarch Benchmark Dashboard</h1>
    <p class="lede">Static report for standard vs paged inference, including aggregate metrics, deltas, compression mode, and per-step trace plots.</p>
    <div class="grid">{''.join(cards)}</div>
    <div class="plots">{''.join(plot_items)}</div>
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(dashboard)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Monarch trace metrics and plots")
    parser.add_argument("--trace-dir", required=True, help="Directory containing JSONL trace files")
    parser.add_argument("--output-dir", default=None, help="Directory for generated summaries and SVG plots")
    parser.add_argument("--results", default=None, help="Optional benchmark results JSON to include in the dashboard")
    parser.add_argument("--json", action="store_true", help="Print summary JSON")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir) if args.output_dir else trace_dir / "report"
    trace_files = sorted(trace_dir.glob("*.jsonl"))
    if not trace_files:
        raise FileNotFoundError(f"No JSONL traces found in {trace_dir}")

    all_summaries: Dict[str, Dict[str, float]] = {}
    aggregate_rows: List[Dict[str, float]] = []

    for trace_file in trace_files:
        rows = load_trace(trace_file)
        if not rows:
            continue
        all_summaries[trace_file.name] = summarize_trace(rows)
        aggregate_rows.extend(rows)

        stem = trace_file.stem
        svg_line_plot(
            rows,
            x_key="step",
            y_keys=["step_sec"],
            labels=["latency / token"],
            title=f"{stem}: token index vs latency",
            output_path=output_dir / f"{stem}.latency.svg",
        )
        svg_line_plot(
            rows,
            x_key="step",
            y_keys=["avg_attention_score", "avg_importance_ema"],
            labels=["avg attention", "avg importance"],
            title=f"{stem}: attention score decay",
            output_path=output_dir / f"{stem}.attention.svg",
        )
        svg_line_plot(
            rows,
            x_key="step",
            y_keys=["page_hit", "page_miss"],
            labels=["page hit", "page miss"],
            title=f"{stem}: page hits vs misses",
            output_path=output_dir / f"{stem}.paging.svg",
        )

    aggregate_summary = summarize_trace(aggregate_rows)
    all_summaries["_aggregate"] = aggregate_summary
    (output_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2, sort_keys=True))
    results_path = Path(args.results) if args.results else (trace_dir.parent / "results.json")
    results = load_results(results_path)
    build_dashboard(output_dir, all_summaries, results)

    print("tokens/sec", f"{aggregate_summary['tokens_per_sec']:.4f}")
    print("latency per token (ms)", f"{aggregate_summary['latency_per_token_ms']:.4f}")
    print(
        "memory retention stats",
        f"resident={aggregate_summary['resident_hot_tokens']:.2f}",
        f"desired={aggregate_summary['desired_hot_tokens']:.2f}",
        f"retention_ratio={aggregate_summary['retention_ratio']:.4f}",
    )
    print(
        "promotion frequency",
        f"{aggregate_summary['promotion_frequency']:.4f}",
        f"page_hit_rate={aggregate_summary['page_hit_rate']:.4f}",
        f"page_miss_rate={aggregate_summary['page_miss_rate']:.4f}",
    )
    print(f"report_dir {output_dir}")
    print(f"dashboard {output_dir / 'index.html'}")

    if args.json:
        print(json.dumps(all_summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
