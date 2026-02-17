#!/usr/bin/env python3
"""
Text command center for running project routines.

Features:
- Persistent menu loop
- Guided prompts for routine parameters
- Subprocess execution using current Python interpreter
- Report files created by each routine
- Return to menu after each run until user quits
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parent
IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


def _prompt_text(prompt: str, default: str | None = None, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        if not required:
            return ""
        print("Value required.")


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Enter a valid integer.")


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Enter a valid number.")


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    default_label = "y" if default else "n"
    while True:
        raw = input(f"{prompt} (y/n) [{default_label}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter y or n.")


def _snapshot_files(root: Path) -> Dict[str, int]:
    snapshot: Dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        base = Path(dirpath)
        for filename in filenames:
            file_path = base / filename
            try:
                rel = str(file_path.relative_to(root))
                snapshot[rel] = file_path.stat().st_mtime_ns
            except (OSError, ValueError):
                continue
    return snapshot


def _print_created_files(created: List[str]) -> None:
    print("\nFiles created:")
    if not created:
        print("  (none)")
        return
    for path in created:
        print(f"  - {path}")


def run_subprocess(
    script_and_args: List[str],
    label: str,
    output_path: Optional[str] = None
) -> int:
    cmd = [sys.executable, *script_and_args]
    print("\n" + "=" * 72)
    print(f"Running: {label}")
    print("=" * 72)
    print("Command:", " ".join(cmd))

    before = _snapshot_files(ROOT_DIR)
    output_file = None
    resolved_output_path = None
    if output_path:
        output_file = Path(output_path)
        if not output_file.is_absolute():
            output_file = ROOT_DIR / output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path = output_file
        print(f"Output will be written to: {resolved_output_path}")
        with output_file.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                cmd,
                cwd=ROOT_DIR,
                check=False,
                stdout=handle,
                stderr=subprocess.STDOUT
            )
    else:
        result = subprocess.run(cmd, cwd=ROOT_DIR, check=False)
    after = _snapshot_files(ROOT_DIR)

    created = sorted([path for path in after.keys() if path not in before])
    print("\n" + "-" * 72)
    print(f"Exit code: {result.returncode}")
    _print_created_files(created)
    print("-" * 72)
    if resolved_output_path is not None:
        print(f"Captured output: {resolved_output_path}")
    input("Press Enter to return to the menu...")
    return result.returncode


def build_calibration_args() -> List[str]:
    print("\nCalibration Inputs")
    data_path = _prompt_text("Historical data CSV path", "historical_data.csv", required=True)
    output_path = _prompt_text("Output calibration JSON", "calibration_results.json", required=True)
    start_date = _prompt_text("Start date YYYY-MM-DD (optional)")
    end_date = _prompt_text("End date YYYY-MM-DD (optional)")
    no_covid_exclusion = _prompt_yes_no("Disable COVID exclusion", False)
    quiet = _prompt_yes_no("Quiet mode", False)

    args = ["run_calibration.py", "--data", data_path, "--output", output_path]
    if start_date:
        args += ["--start-date", start_date]
    if end_date:
        args += ["--end-date", end_date]
    if no_covid_exclusion:
        args.append("--no-covid-exclusion")
    if quiet:
        args.append("--quiet")
    return args


def _prompt_output_path(default_name: str) -> Optional[str]:
    save_output = _prompt_yes_no("Pipe output to text file", False)
    if not save_output:
        return None
    return _prompt_text("Output file path", default_name, required=True)


def build_strategy_eval_args() -> Tuple[List[str], Optional[str]]:
    print("\nStrategy Evaluation Inputs")
    calibration = _prompt_text("Calibration JSON path", "calibration_results.json", required=True)
    n_paths = _prompt_int("Monte Carlo paths", 10000)
    expiration = _prompt_int("Expiration days", 45)
    min_pop = _prompt_float("Minimum POP", 0.70)
    profit_target = _prompt_float("Profit target fraction", 0.50)
    loss_multiple = _prompt_float("Loss multiple", 2.0)
    dte_close = _prompt_int("DTE close threshold", 7)
    seed = _prompt_int("Random seed", 42)
    top = _prompt_int("Top strategies to display", 10)
    spread_mode = _prompt_text("Spread type (both/put/call)", "both").lower()
    quiet = _prompt_yes_no("Quiet mode", False)
    output_path = _prompt_output_path("strategy_evaluation_output.txt")

    args = [
        "run_strategy_evaluation.py",
        "--calibration",
        calibration,
        "--n-paths",
        str(n_paths),
        "--expiration",
        str(expiration),
        "--min-pop",
        str(min_pop),
        "--profit-target",
        str(profit_target),
        "--loss-multiple",
        str(loss_multiple),
        "--dte-close",
        str(dte_close),
        "--seed",
        str(seed),
        "--top",
        str(top),
    ]
    if spread_mode == "put":
        args.append("--put-only")
    elif spread_mode == "call":
        args.append("--call-only")
    if quiet:
        args.append("--quiet")
    return args, output_path


def build_market_eval_args() -> Tuple[List[str], Optional[str]]:
    print("\nMarket Evaluation Inputs")
    calibration = _prompt_text("Calibration JSON path", "calibration_results.json", required=True)
    options_file = _prompt_text("Options CSV path", "tlt_options.csv", required=True)
    n_paths = _prompt_int("Monte Carlo paths", 10000)
    min_dte = _prompt_int("Minimum DTE", 30)
    max_dte = _prompt_int("Maximum DTE", 90)
    min_pop = _prompt_float("Minimum POP", 0.70)
    profit_target = _prompt_float("Profit target fraction", 0.50)
    loss_multiple = _prompt_float("Loss multiple", 2.0)
    dte_close = _prompt_int("DTE close threshold", 7)
    seed = _prompt_int("Random seed", 42)
    top = _prompt_int("Top strategies to display", 10)
    quiet = _prompt_yes_no("Quiet mode", False)
    output_path = _prompt_output_path("market_evaluation_output.txt")

    args = [
        "run_market_evaluation.py",
        "--calibration",
        calibration,
        "--options",
        options_file,
        "--n-paths",
        str(n_paths),
        "--min-dte",
        str(min_dte),
        "--max-dte",
        str(max_dte),
        "--min-pop",
        str(min_pop),
        "--profit-target",
        str(profit_target),
        "--loss-multiple",
        str(loss_multiple),
        "--dte-close",
        str(dte_close),
        "--seed",
        str(seed),
        "--top",
        str(top),
    ]
    if quiet:
        args.append("--quiet")
    return args, output_path


def build_bloomberg_process_args() -> List[str]:
    print("\nBloomberg Processing Inputs")
    input_file = _prompt_text("Input Bloomberg file path", required=True)
    output_file = _prompt_text("Output CSV path (optional)")
    underlying_price = _prompt_text("Underlying price (optional)")

    args = ["process_bloomberg_export.py", input_file]
    if output_file:
        args.append(output_file)
    if underlying_price:
        args.append(underlying_price)
    return args


def build_portfolio_manager_args() -> List[str]:
    print("\nPortfolio Manager Inputs")
    data_dir = _prompt_text("Data directory", ".")
    args = ["portfolio_manager.py"]
    if data_dir and data_dir != ".":
        args += ["--data-dir", data_dir]
    return args


def print_menu() -> None:
    print("\n" + "=" * 72)
    print("TLT OPTIONS WORKBENCH")
    print("=" * 72)
    print("1) Run calibration")
    print("2) Run strategy evaluation")
    print("3) Run market evaluation")
    print("4) Process Bloomberg export")
    print("5) Run portfolio manager")
    print("Q) Quit")


def main() -> int:
    while True:
        print_menu()
        choice = input("\nSelect option: ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            print("Exiting workbench.")
            return 0
        if choice == "1":
            args = build_calibration_args()
            run_subprocess(args, "Calibration")
            continue
        if choice == "2":
            args, output_path = build_strategy_eval_args()
            run_subprocess(args, "Strategy Evaluation", output_path)
            continue
        if choice == "3":
            args, output_path = build_market_eval_args()
            run_subprocess(args, "Market Evaluation", output_path)
            continue
        if choice == "4":
            args = build_bloomberg_process_args()
            run_subprocess(args, "Bloomberg Processing")
            continue
        if choice == "5":
            args = build_portfolio_manager_args()
            run_subprocess(args, "Portfolio Manager")
            continue

        print("Invalid choice. Please select a menu option.")


if __name__ == "__main__":
    sys.exit(main())
