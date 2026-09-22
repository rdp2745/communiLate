"""Assemble evaluation output into JSON plus a readable Markdown summary."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..config import REPO_ROOT, Config


def _plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def build_report(
    cfg: Config,
    metrics: dict[str, Sequence],
    contrastive: Sequence,
    examples: Sequence[dict],
    route_counts: dict[str, int] | None = None,
) -> dict:
    return {
        "experiment": cfg.get("experiment.name", "unnamed"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "metrics": _plain(metrics),
        "contrastive": _plain(list(contrastive)),
        "routes": route_counts or {},
        "examples": _plain(list(examples)),
    }


def to_markdown(report: dict) -> str:
    lines = [
        f"# {report['experiment']}",
        "",
        f"_generated {report['generated_at']}_",
        "",
        "## Corpus metrics",
        "",
    ]

    systems = list(report["metrics"])
    if systems:
        metric_names = sorted({m["name"] for sys in systems for m in report["metrics"][sys]})
        lines.append("| system | " + " | ".join(metric_names) + " |")
        lines.append("|---" * (len(metric_names) + 1) + "|")
        for system in systems:
            scores = {m["name"]: m["score"] for m in report["metrics"][system]}
            row = " | ".join(
                f"{scores[name]:.2f}" if name in scores and scores[name] == scores[name] else "-"
                for name in metric_names
            )
            lines.append(f"| {system} | {row} |")
    else:
        lines.append("_none computed_")

    lines += ["", "## Targeted checks", ""]
    if report["contrastive"]:
        lines.append("| check | score | n scored | n skipped |")
        lines.append("|---|---|---|---|")
        for check in report["contrastive"]:
            lines.append(
                f"| {check['name']} | {check['score']:.3f} | "
                f"{check['n_scored']} | {check.get('n_skipped', 0)} |"
            )
    else:
        lines.append("_none configured_")

    if report.get("routes"):
        lines += ["", "## Gating routes", "", "| route | count |", "|---|---|"]
        for route, count in report["routes"].items():
            lines.append(f"| {route} | {count} |")

    lines += ["", "## Frozen vs. adapted", ""]
    for i, example in enumerate(report["examples"], start=1):
        lines.append(f"**{i}.** `{example['source']}`")
        lines.append("")
        lines.append(f"- baseline: {example['baseline']}")
        lines.append(f"- adapted:  {example['adapted']}")
        if "reference" in example:
            lines.append(f"- reference: {example['reference']}")
        lines.append("")

    return "\n".join(lines)


def save_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    path = Path(output_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)

    json_path = path / "report.json"
    md_path = path / "report.md"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path
