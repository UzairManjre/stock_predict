import os
import sys
import glob
import argparse
import subprocess
from datetime import datetime, timedelta

# Ensure stdout/stderr use UTF-8 on Windows
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = os.path.join(HERE, 'venv', 'Scripts', 'python.exe')

# Fallback to sys.executable if virtualenv python isn't found
if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title.upper()} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def get_days_since_last_training():
    """Returns the number of days since the latest model checkpoint was modified."""
    pattern = os.path.join(HERE, 'csv_logs', '**', '*.ckpt')
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        return None  # No checkpoints found, need to train
        
    # Get the latest modified time among all checkpoints
    latest_mtime = max(os.path.getmtime(f) for f in ckpts)
    last_trained_dt = datetime.fromtimestamp(latest_mtime)
    elapsed = datetime.now() - last_trained_dt
    return elapsed.days

def main():
    parser = argparse.ArgumentParser(description="Daily stock prediction update and automated retraining orchestrator.")
    parser.add_argument('--retrain-interval', type=int, default=15, 
                        help="Retrain the model if the latest checkpoint is older than this many days (default: 15).")
    parser.add_argument('--force-retrain', action='store_true', 
                        help="Force retraining regardless of the age of the latest checkpoint.")
    args = parser.parse_args()

    print_section("starting daily update orchestrator")
    
    # 1. Fetch daily stock prices
    print_section("stage 1: fetching daily stock prices")
    fetch_script = os.path.join(HERE, 'Cleaner', 'daily_fetch.py')
    cmd_fetch = [PYTHON_EXE, fetch_script, '--days', '15', '--workers', '10']
    
    print(f"Running: {' '.join(cmd_fetch)}")
    t0 = datetime.now()
    res_fetch = subprocess.run(cmd_fetch, cwd=HERE)
    
    if res_fetch.returncode != 0:
        print(f"\n[ERROR] Daily price fetch failed with exit code {res_fetch.returncode}. Aborting.")
        sys.exit(res_fetch.returncode)
    
    print(f"Stage 1 complete. Elapsed: {(datetime.now() - t0).total_seconds():.1f}s.")
    
    # 2. Check if retraining is needed
    days_elapsed = get_days_since_last_training()
    should_retrain = False
    
    if args.force_retrain:
        print("\n[ORCHESTRATOR] Force retrain flag is set. Retraining model...")
        should_retrain = True
    elif days_elapsed is None:
        print("\n[ORCHESTRATOR] No model checkpoints found. Retraining model...")
        should_retrain = True
    elif days_elapsed >= args.retrain_interval:
        print(f"\n[ORCHESTRATOR] Latest model checkpoint was created {days_elapsed} days ago (>= retrain interval of {args.retrain_interval} days). Retraining model...")
        should_retrain = True
    else:
        print(f"\n[ORCHESTRATOR] Latest model checkpoint is {days_elapsed} days old (< retrain interval of {args.retrain_interval} days). Skipping training stage.")

    # Determine pipeline stages
    stages = ['features', 'train', 'predict', 'drift'] if should_retrain else ['features', 'predict', 'drift']
    
    # 3. Run TFT pipeline
    print_section("stage 2: running features, prediction, and drift pipeline")
    pipeline_script = os.path.join(HERE, 'Cleaner', 'pipeline.py')
    cmd_pipeline = [PYTHON_EXE, pipeline_script, '--stages'] + stages
    
    print(f"Running: {' '.join(cmd_pipeline)}")
    t1 = datetime.now()
    res_pipeline = subprocess.run(cmd_pipeline, cwd=HERE)
    
    if res_pipeline.returncode != 0:
        print(f"\n[ERROR] Downstream feature/prediction pipeline failed with exit code {res_pipeline.returncode}.")
        sys.exit(res_pipeline.returncode)
        
    print(f"Stage 2 complete. Elapsed: {(datetime.now() - t1).total_seconds():.1f}s.")

    # 3. Run accuracy checker (evaluates batches with enough elapsed time)
    print_section("stage 3: accuracy evaluation against real market data")
    acc_script = os.path.join(HERE, 'Cleaner', 'accuracy_checker.py')
    cmd_acc = [PYTHON_EXE, acc_script, '--min-days', '7']

    print(f"Running: {' '.join(cmd_acc)}")
    t2 = datetime.now()
    res_acc = subprocess.run(cmd_acc, cwd=HERE)

    if res_acc.returncode != 0:
        print(f"\n[WARN] Accuracy checker returned exit code {res_acc.returncode}. Non-critical — continuing.")
    else:
        print(f"Stage 3 complete. Elapsed: {(datetime.now() - t2).total_seconds():.1f}s.")
    
    print_section("daily update completed successfully")
    print(f"Total time elapsed: {(datetime.now() - t0).total_seconds():.1f}s.")

if __name__ == '__main__':
    main()
