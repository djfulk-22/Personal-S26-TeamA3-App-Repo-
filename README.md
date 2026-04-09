# UNC Baseball Attendance Forecasting App

This repository contains the full pipeline used to refresh, score, and deploy the UNC Baseball attendance forecasting app. It includes the Streamlit app, the attendance scraping notebook, the performance feature engineering notebook, the scoring notebook, and the automation scripts used to run the refresh pipeline and push updated app data back to GitHub.

## What this repository does

The project supports a forecast workflow for UNC home baseball attendance. At a high level, the repository does four things:

1. Scrapes and refreshes the latest UNC baseball schedule and attendance data
2. Builds updated game-level performance features
3. Runs the scoring pipeline to generate prediction files for the app
4. Serves those outputs through a Streamlit interface

In production, the refresh pipeline is automated on a Mac using `launchd`, which runs a local shell script that calls the Python refresh orchestrator. That orchestrator executes the notebooks in sequence, verifies the outputs, and pushes updated app-facing CSVs to GitHub only when they actually changed.

## Repository workflow

The end-to-end flow is:

1. **Attendance refresh**
   - Runs the GoHeels attendance scraper notebook
   - Produces the refreshed UNC baseball attendance file

2. **Performance refresh**
   - Copies the refreshed attendance file into the performance notebook input location
   - Runs the performance notebook to generate updated game-level performance features

3. **Scoring refresh**
   - Runs the scoring notebook
   - Writes refreshed prediction outputs into `APP/data`

4. **Sanity checks**
   - Confirms the expected app CSVs exist
   - Confirms they are non-empty
   - Confirms they include `season_year`
   - Confirms they contain 2026 rows

5. **Git update**
   - Stages only the app-facing midseason prediction files
   - Commits and pushes only if those files actually changed

## Main repository components

### `APP/`
Contains the deployed Streamlit application.

#### `APP/app.py`
This is the main Streamlit app. It loads prediction bundles from `APP/data`, prepares the data for display, and renders the user interface.

The app currently supports both:
- **midseason predictions**
- **preseason predictions**

The app includes:
- a season selector
- a prediction column selector
- a month filter
- season-level summary metrics
- a calendar view of projected attendance
- a single-game detail view
- a data explorer tab

The app now resolves its data path relative to the location of `app.py`, which makes it work both locally and on Streamlit Community Cloud.

#### `APP/data/`
This folder stores the CSV files used directly by the app.

The automated refresh pipeline currently updates these three midseason files:
- `db_predictions_2024_2026_full_df.csv`
- `db_predictions_2024_2026_tier3_raw_tier_cols.csv`
- `db_predictions_2024_2026_tier3_final_feats_export.csv`

The app is also configured to support preseason files if they are present in the same folder.

---

### `SCRAPERS/`
Contains the attendance scraper, performance notebook, and automation scripts.

#### `SCRAPERS/GoHeels_Attendance_Scraper/unc_baseball_scraper_final_0331_incremental_cache_project_paths_ROOTALIGNED.ipynb`
This notebook refreshes the UNC baseball attendance and schedule data from GoHeels and writes:

- `SCRAPERS/GoHeels_Attendance_Scraper/unc_baseball_final.csv`

This notebook is the first step in the automated pipeline.

#### `SCRAPERS/Game_Stat_Scraper/notebooks/final_game_stat_scraper_updated_combined_with_performance_features_refresh_cache_project_paths_ROOTALIGNED.ipynb`
This notebook takes the refreshed GoHeels output, refreshes performance information, engineers the game-level features used downstream, and writes:

- `SCRAPERS/Game_Stat_Scraper/exports/uncbaseball_season_performance_0331.csv`

Before this notebook runs, the automation script copies the latest GoHeels output to:

- `SCRAPERS/Game_Stat_Scraper/data/unc_baseball_final.csv`

#### `SCRAPERS/run_refresh_and_push.py`
This is the main automation orchestrator for the repository.

It:
- derives all project paths from `__file__`
- executes the three notebooks using `nbconvert` and `ExecutePreprocessor`
- uses the `tfenv` Jupyter kernel
- stores executed notebook copies in `SCRAPERS/refresh_runs/`
- verifies that the expected app outputs were created
- runs sanity checks on the app CSVs
- stages only the three app-facing midseason files
- commits and pushes only if those files changed

This is the script that should be run for a full refresh.

#### `SCRAPERS/refresh_runs/`
This folder is used to store timestamped executed copies of the notebooks from each automated refresh run. It acts as a run log / artifact trail for notebook execution.

---

### `SCORING/`
Contains the scoring notebook used to produce the app-facing prediction outputs.

#### `SCORING/notebooks/unified_final_midseason_scoring_PROJECT_ROOT_ALIGNED.ipynb`
This notebook reads the refreshed upstream data and writes the updated midseason prediction files used by the app.

The automated pipeline expects this notebook to produce the three app files in `APP/data`.

## Automation setup

### `run_refresh_launchd.sh`
This shell script is the local entrypoint used by `launchd`.

It:
- points to the local repo path
- points directly to the `tfenv` Python executable
- creates the log directory if needed
- writes refresh logs to `automation_logs/run_refresh_launchd.log`
- changes into the repo root
- runs `python SCRAPERS/run_refresh_and_push.py`

This script is intended to be called by a macOS LaunchAgent.

### `launchd` LaunchAgent
The scheduled daily run is controlled by a separate LaunchAgent plist on the local Mac. That plist is not stored in this repository.

The LaunchAgent loads the shell script above and runs it on a fixed schedule. In the current local setup, the job is configured to run once per day in the morning.

### `automation_logs/`
This folder stores log output from the automated local refresh process.

Useful log files include:
- `run_refresh_launchd.log`
- `launchd_stdout.log`
- `launchd_stderr.log`

## How to run the refresh pipeline manually

From the repository root:

```bash
conda activate tfenv
python SCRAPERS/run_refresh_and_push.py
```

If everything works correctly, the script will:

- run the attendance notebook
- copy the attendance output into the performance input location
- run the performance notebook
- run the scoring notebook
- verify the app output CSVs
- stage the three app-facing midseason files
- commit and push only if those files changed

## How the deployed app stays updated

This repository is set up so that the local scheduled refresh runs on the Mac, writes updated app CSVs into `APP/data`, and pushes those updated files to GitHub.

Once the Streamlit app is deployed from this repository, the deployed app can pick up updated data from the committed files in the repo.

The intended daily refresh mechanism is:

1. local `launchd` schedule runs the shell script
2. shell script runs `SCRAPERS/run_refresh_and_push.py`
3. the pipeline writes updated prediction CSVs into `APP/data`
4. the script pushes those updated files to GitHub if they changed
5. the deployed Streamlit app reads the refreshed files from the repository

## Important implementation notes

- The refresh pipeline depends on the local Conda environment `tfenv`
- Notebook execution is tied to the `tfenv` Jupyter kernel
- Part of the scraping process may momentarily launch a local headed browser
- The automation is designed for local Mac execution, not GitHub-hosted runners
- The current Git automation intentionally stages only the three midseason app files
- If no data changes are detected, the script exits cleanly without creating a new commit
- The app loads its data relative to `app.py`, which avoids path issues between local runs and Streamlit deployment

## Typical files produced by the pipeline

### Attendance output
- `SCRAPERS/GoHeels_Attendance_Scraper/unc_baseball_final.csv`

### Performance output
- `SCRAPERS/Game_Stat_Scraper/exports/uncbaseball_season_performance_0331.csv`

### App-facing midseason outputs
- `APP/data/db_predictions_2024_2026_full_df.csv`
- `APP/data/db_predictions_2024_2026_tier3_raw_tier_cols.csv`
- `APP/data/db_predictions_2024_2026_tier3_final_feats_export.csv`

## Local deployment and path notes

The current local shell script is configured for the author's machine and uses absolute local paths, including:

- the repository location
- the direct Python path for the `tfenv` environment

If this repository is moved to a different machine or directory, `run_refresh_launchd.sh` and the local LaunchAgent plist will need to be updated.

## Summary

This repository contains a working local-to-cloud refresh pipeline for the UNC Baseball attendance forecasting app. The notebooks generate refreshed data, the Python automation script orchestrates the full run and Git update, and the Streamlit app serves the latest committed prediction files through an interactive planning interface.





