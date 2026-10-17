"""
pipeline.py — End-to-end orchestration for the TFT stock prediction system.

Stages (all or any subset via --stages flag):
    1. features   — Run build_features.py (feature materialization from PostgreSQL)
    2. tune       — Run tune_tft.py (Optuna hyperparameter search)
    3. train      — Run train_tft.py (full training with purged CV + artifact export)
    4. predict    — Run predict.py (live inference → predictions table)
    5. drift      — Run predict.py --drift-check (KS feature drift monitoring only)

Usage examples:
    # Full pipeline from scratch:
    python pipeline.py

    # Only feature engineering + training:
    python pipeline.py --stages features train

    # Daily prediction + drift check (for a cron job):
    python pipeline.py --stages predict drift

    # Pass a specific train-end / val-end to train_tft.py:
    python pipeline.py --stages train predict --train-end 2024-07-01 --val-end 2025-07-01

Environment:
    All DB credentials are read from ../.env (same as all other scripts).
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Ensure stdout/stderr use UTF-8 on Windows
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding is None or sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))

STAGE_ORDER = ['features', 'tune', 'train', 'final', 'predict', 'drift']

STAGE_SCRIPTS = {
    'features': os.path.join(HERE, 'build_features.py'),
    'tune':     os.path.join(HERE, 'tune_tft.py'),
    'train':    os.path.join(HERE, 'train_tft.py'),
    'final':    os.path.join(HERE, 'final_model.py'),   # frozen 21-day blend (--train)
    'predict':  os.path.join(HERE, 'predict.py'),
    'drift':    os.path.join(HERE, 'predict.py'),   # same script, different flag
}


def print_banner(text: str):
    w = max(len(text) + 4, 60)
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


def run_stage(stage: str, extra_args: list[str]) -> bool:
    """
    Spawns the stage script as a subprocess.
    Returns True on success (exit code 0), False otherwise.
    """
    script = STAGE_SCRIPTS[stage]
    cmd = [sys.executable, script] + extra_args

    if stage == 'drift':
        cmd.append('--drift-check')
        cmd.append('--no-db-write')   # drift-only run; don't write new predictions

    print(f"\n  $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=HERE)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  ✓ Stage '{stage}' completed in {elapsed:.1f}s")
        return True
    else:
        print(f"  ✗ Stage '{stage}' FAILED with exit code {result.returncode} after {elapsed:.1f}s")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="TFT pipeline orchestrator — chain any subset of pipeline stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--stages', nargs='+',
        choices=STAGE_ORDER,
        default=STAGE_ORDER,
        metavar='STAGE',
        help=(
            f"Which stages to run, in order. "
            f"Choices: {', '.join(STAGE_ORDER)}. "
            f"Default: all stages."
        )
    )
    parser.add_argument(
        '--train-end', type=str, default=None,
        help='Passed to train_tft.py --train-end (e.g. 2024-07-01). Default: script default.'
    )
    parser.add_argument(
        '--val-end', type=str, default=None,
        help='Passed to train_tft.py --val-end (e.g. 2025-07-01). Default: script default.'
    )
    parser.add_argument(
        '--no-db-write', action='store_true',
        help='Pass --no-db-write to predict.py (print only, do not write to predictions table).'
    )
    parser.add_argument(
        '--stop-on-failure', action='store_true',
        help='Abort the pipeline immediately if any stage fails.'
    )
    args = parser.parse_args()

    # Deduplicate and enforce STAGE_ORDER ordering
    requested = [s for s in STAGE_ORDER if s in args.stages]

    # Build per-stage extra args
    train_args = []
    if args.train_end:
        train_args += ['--train-end', args.train_end]
    if args.val_end:
        train_args += ['--val-end', args.val_end]

    predict_args = []
    if args.no_db_write:
        predict_args.append('--no-db-write')

    stage_args = {
        'features': [],
        'tune':     [],
        'train':    train_args,
        'final':    ['--train'],
        'predict':  predict_args,
        'drift':    predict_args,   # drift uses predict.py flags
    }

    # ── Run pipeline ─────────────────────────────────────────────────────────
    started_at = datetime.now()
    print_banner(f"TFT Pipeline — {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stages: {' → '.join(requested)}")

    results: dict[str, bool] = {}

    for stage in requested:
        print_banner(f"Stage: {stage.upper()}")
        ok = run_stage(stage, stage_args[stage])
        results[stage] = ok

        if not ok and args.stop_on_failure:
            print(f"\n[PIPELINE] Stopping due to failure in stage '{stage}' (--stop-on-failure).")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = (datetime.now() - started_at).total_seconds()
    print_banner("Pipeline Summary")
    all_ok = True
    for stage in requested:
        status = '✓' if results.get(stage) else '✗'
        print(f"  {status}  {stage}")
        if not results.get(stage):
            all_ok = False

    print(f"\n  Total elapsed: {elapsed_total:.0f}s")
    if all_ok:
        print("  All stages succeeded. ✓")
    else:
        print("  One or more stages failed. Check logs above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
