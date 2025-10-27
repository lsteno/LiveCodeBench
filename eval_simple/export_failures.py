#!/usr/bin/env python3
"""
Extract failing LiveCodeBench generations from an eval_all JSON file.

Usage:
    python export_failures.py \
        --eval-all output/Qwen2.5-1.5B-FT/Scenario.codegeneration_10_0.2_eval_all.json \
        --output failures_ft.json \
        --limit 50 \
        --baseline-eval output/Qwen2.5-Ins-1.5B/Scenario.codegeneration_10_0.2_eval_all.json

The script gathers every instance with pass@1 == 0 (or all instances when
--include-success is set) and stores the problem statement, public tests, and
top generation (code + metadata). If --baseline-eval is provided, the matching
entry from the baseline run is attached for side-by-side comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_eval_all(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        rows = json.load(fh)
    return {row["question_id"]: row for row in rows}


def build_entry(row: dict[str, Any]) -> dict[str, Any]:
    candidate_code = row["code_list"][0] if row["code_list"] else ""
    candidate_metadata = row["metadata"][0] if row["metadata"] else {}
    candidate_grade = row["graded_list"][0] if row["graded_list"] else False
    return {
        "question_id": row["question_id"],
        "difficulty": row.get("difficulty"),
        "platform": row.get("platform"),
        "contest_id": row.get("contest_id"),
        "contest_date": row.get("contest_date"),
        "pass_at_1": row.get("pass@1"),
        "question_content": row.get("question_content"),
        "public_test_cases": row.get("public_test_cases"),
        "candidate": {
            "code": candidate_code,
            "metadata": candidate_metadata,
            "graded": candidate_grade,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export failing LiveCodeBench generations.")
    parser.add_argument("--eval-all", type=Path, required=True, help="Path to eval_all JSON.")
    parser.add_argument("--output", type=Path, required=True, help="File to write the extracted entries.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of entries to export.")
    parser.add_argument(
        "--include-success",
        action="store_true",
        help="Include successful problems as well (default: only pass@1 == 0).",
    )
    parser.add_argument(
        "--baseline-eval",
        type=Path,
        help="Optional eval_all JSON from a baseline model for comparison.",
    )
    args = parser.parse_args()

    eval_rows = load_eval_all(args.eval_all)
    baseline_rows = load_eval_all(args.baseline_eval) if args.baseline_eval else {}

    selected: list[dict[str, Any]] = []
    for row in eval_rows.values():
        pass_at_1 = bool(row.get("pass@1"))
        if not args.include_success and pass_at_1:
            continue

        entry = build_entry(row)

        baseline_row = baseline_rows.get(row["question_id"])
        if baseline_row is not None:
            entry["baseline"] = build_entry(baseline_row)

        selected.append(entry)

        if args.limit is not None and len(selected) >= args.limit:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} entries to {args.output}")


if __name__ == "__main__":
    main()
