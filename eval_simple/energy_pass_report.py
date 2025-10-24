#!/usr/bin/env python3
"""
Compare training energy and Pass@1 gains for multiple fine-tuned models.

Example:

python3 energy_pass_report.py \
  --baseline "Baseline:output/Qwen2.5-Ins-0.5B/Scenario.codegeneration_10_0.2_eval_all.json:~/GSD-finetune/lora_simple/logs/energy_baseline_zero.json" \
  --model "LoRA:output/Qwen2.5-0.5B-FT/Scenario.codegeneration_10_0.2_eval_all.json:~/GSD-finetune/lora_simple/runs/qwen2.5-0.5b-lora/energy.json" \
  --model "QLoRA:output/Qwen2.5-0.5B-QLoRA/Scenario.codegeneration_10_0.2_eval_all.json:~/GSD-finetune/qlora_simple/runs/qwen2.5-0.5b-qlora/energy.json" \
  --model "BitFit:output/Qwen2.5-0.5B-BitFit/Scenario.codegeneration_10_0.2_eval_all.json:~/GSD-finetune/bitfit_simple/runs/qwen2.5-0.5b-bitfit/energy.json"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    eval_path: Path
    energy_path: Path


def parse_spec(raw: str, flag: str) -> ModelSpec:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"{flag} expects 'name:eval_path:energy_path', got: {raw}"
        )
    name, eval_path, energy_path = parts
    return ModelSpec(name.strip(), Path(eval_path).expanduser(), Path(energy_path).expanduser())


def mean_pass_1(eval_file: Path) -> float:
    with eval_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    pass_values = [row["pass@1"] for row in data if "pass@1" in row]
    if not pass_values:
        raise ValueError(f"No pass@1 entries found in {eval_file}")
    return sum(pass_values) / len(pass_values)


def load_energy_joules(energy_file: Path) -> float:
    with energy_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if "energy_joules" not in payload:
        raise ValueError(f"'energy_joules' missing in {energy_file}")
    return float(payload["energy_joules"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report Joules per Pass@1 gain for fine-tuned models."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=lambda raw: parse_spec(raw, "--baseline"),
        help="Reference in the form 'name:eval_path:energy_path'.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        type=lambda raw: parse_spec(raw, "--model"),
        help="Fine-tuned run in the form 'name:eval_path:energy_path'. Repeat for multiple models.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline: ModelSpec = args.baseline
    models: list[ModelSpec] = args.model

    if not models:
        raise SystemExit("Provide at least one --model to compare.")

    base_pass1 = mean_pass_1(baseline.eval_path)
    base_energy = load_energy_joules(baseline.energy_path)

    header = f"{'Model':20} {'Pass@1':>10} {'ΔPass@1':>10} {'Energy (J)':>12} {'J / ΔPass@1':>15}"
    divider = "-" * len(header)

    print(header)
    print(divider)
    print(
        f"{baseline.name:20} {base_pass1:10.6f} {'-':>10} {base_energy:12.2f} {'-':>15}"
    )

    for spec in models:
        pass1 = mean_pass_1(spec.eval_path)
        energy = load_energy_joules(spec.energy_path)
        delta = pass1 - base_pass1
        if delta <= 0:
            ratio_display = "∞" if delta == 0 else "N/A"
        else:
            ratio_display = f"{energy / delta:,.2f}"

        print(
            f"{spec.name:20} {pass1:10.6f} {delta:10.6f} {energy:12.2f} {ratio_display:>15}"
        )


if __name__ == "__main__":
    main()
