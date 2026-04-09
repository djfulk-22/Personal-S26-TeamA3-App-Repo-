from pathlib import Path
import shutil
import subprocess
import sys
import time

import nbformat
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor


# ==========================================================
# Paths
# ==========================================================
SCRIPT_PATH = Path(__file__).resolve()
SCRAPERS_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = SCRAPERS_ROOT.parent

ATTENDANCE_ROOT = SCRAPERS_ROOT / "GoHeels_Attendance_Scraper"
PERFORMANCE_ROOT = SCRAPERS_ROOT / "Game_Stat_Scraper"
SCORING_ROOT = PROJECT_ROOT / "SCORING"
APP_ROOT = PROJECT_ROOT / "APP"

ATTENDANCE_NOTEBOOK = (
    ATTENDANCE_ROOT
    / "unc_baseball_scraper_final_0331_incremental_cache_project_paths_ROOTALIGNED.ipynb"
)
PERFORMANCE_NOTEBOOK = (
    PERFORMANCE_ROOT
    / "notebooks"
    / "final_game_stat_scraper_updated_combined_with_performance_features_refresh_cache_project_paths_ROOTALIGNED.ipynb"
)
SCORING_NOTEBOOK = (
    SCORING_ROOT
    / "notebooks"
    / "unified_final_midseason_scoring_PROJECT_ROOT_ALIGNED.ipynb"
)

ATTENDANCE_FINAL_CSV = ATTENDANCE_ROOT / "unc_baseball_final.csv"

PERFORMANCE_DATA_DIR = PERFORMANCE_ROOT / "data"
PERFORMANCE_EXPORTS_DIR = PERFORMANCE_ROOT / "exports"
PERFORMANCE_GOHEELS_INPUT = PERFORMANCE_DATA_DIR / "unc_baseball_final.csv"
PERFORMANCE_OUTPUT_CSV = PERFORMANCE_EXPORTS_DIR / "uncbaseball_season_performance_0331.csv"

APP_DATA_DIR = APP_ROOT / "data"

MIDSEASON_APP_FILES = [
    APP_DATA_DIR / "db_predictions_2024_2026_full_df.csv",
    APP_DATA_DIR / "db_predictions_2024_2026_tier3_raw_tier_cols.csv",
    APP_DATA_DIR / "db_predictions_2024_2026_tier3_final_feats_export.csv",
]

REFRESH_RUNS_DIR = SCRAPERS_ROOT / "refresh_runs"
REFRESH_RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Helpers
# ==========================================================
def run_command(cmd, cwd=None):
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def execute_notebook(input_path: Path, output_path: Path, cwd: Path, timeout: int = 3600):
    with input_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name="tfenv")
    ep.preprocess(nb, resources={"metadata": {"path": str(cwd)}})

    with output_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return output_path


def sanity_check_output(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing expected app output: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"{csv_path.name} is empty")

    if "season_year" not in df.columns:
        raise ValueError(f"{csv_path.name} is missing season_year")

    season_values = pd.to_numeric(df["season_year"], errors="coerce").dropna()
    if 2026 not in set(season_values.astype(int)):
        raise ValueError(f"{csv_path.name} does not contain any 2026 rows")

    print(f"{csv_path.name}: shape={df.shape}")
    print(df["season_year"].value_counts(dropna=False).sort_index())


def ensure_repo_root():
    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.exists():
        raise FileNotFoundError(f"Did not find .git at expected project root: {PROJECT_ROOT}")


def git_commit_and_push():
    # Only stage the three app data files
    rel_paths = [str(p.relative_to(PROJECT_ROOT)) for p in MIDSEASON_APP_FILES]

    run_command(["git", "add", *rel_paths], cwd=PROJECT_ROOT)

    # Check whether the staged files actually changed
    diff_result = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *rel_paths],
        cwd=PROJECT_ROOT,
    )

    if diff_result.returncode == 0:
        print("\nNo changes detected in staged app data files. Nothing to commit.")
        return

    commit_message = f"Automated midseason refresh {time.strftime('%Y-%m-%d %H:%M:%S')}"
    run_command(["git", "commit", "-m", commit_message], cwd=PROJECT_ROOT)
    run_command(["git", "push", "origin", "main"], cwd=PROJECT_ROOT)


# ==========================================================
# Main
# ==========================================================
def main():
    ensure_repo_root()

    run_stamp = time.strftime("%Y%m%d_%H%M%S")

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("SCRAPERS_ROOT:", SCRAPERS_ROOT)
    print("APP_DATA_DIR:", APP_DATA_DIR)

    # 1) Run attendance notebook
    attendance_executed = REFRESH_RUNS_DIR / f"attendance_refresh_{run_stamp}.ipynb"
    execute_notebook(
        input_path=ATTENDANCE_NOTEBOOK,
        output_path=attendance_executed,
        cwd=ATTENDANCE_NOTEBOOK.parent,
        timeout=3600,
    )

    if not ATTENDANCE_FINAL_CSV.exists():
        raise FileNotFoundError(f"Expected attendance output not found: {ATTENDANCE_FINAL_CSV}")

    print("Attendance refresh complete:", ATTENDANCE_FINAL_CSV)

    # 2) Copy refreshed GoHeels CSV into performance input
    PERFORMANCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ATTENDANCE_FINAL_CSV, PERFORMANCE_GOHEELS_INPUT)
    print("Copied refreshed GoHeels CSV ->", PERFORMANCE_GOHEELS_INPUT)

    # 3) Run performance notebook
    performance_executed = REFRESH_RUNS_DIR / f"performance_refresh_{run_stamp}.ipynb"
    execute_notebook(
        input_path=PERFORMANCE_NOTEBOOK,
        output_path=performance_executed,
        cwd=PERFORMANCE_NOTEBOOK.parent,
        timeout=7200,
    )

    if not PERFORMANCE_OUTPUT_CSV.exists():
        raise FileNotFoundError(f"Expected performance output not found: {PERFORMANCE_OUTPUT_CSV}")

    print("Performance refresh complete:", PERFORMANCE_OUTPUT_CSV)

    # 4) Run scoring notebook
    scoring_executed = REFRESH_RUNS_DIR / f"midseason_scoring_refresh_{run_stamp}.ipynb"
    execute_notebook(
        input_path=SCORING_NOTEBOOK,
        output_path=scoring_executed,
        cwd=SCORING_NOTEBOOK.parent,
        timeout=7200,
    )

    missing = [str(p) for p in MIDSEASON_APP_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError("Expected app midseason outputs not found: " + ", ".join(missing))

    print("Midseason scoring refresh complete.")
    for p in MIDSEASON_APP_FILES:
        print("  wrote:", p)

    # 5) Sanity checks
    print("\nRunning sanity checks...")
    for p in MIDSEASON_APP_FILES:
        sanity_check_output(p)

    # 6) Commit and push only app data files
    print("\nCommitting and pushing updated app data files...")
    git_commit_and_push()

    print("\nRefresh pipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)