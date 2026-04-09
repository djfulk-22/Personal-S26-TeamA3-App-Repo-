# Fork notes: `collegebaseball` (patched for UNC attendance project)

This folder contains a **fork** (or vendored copy) of the upstream `collegebaseball` package by Nathan Blumenfeld.

- Upstream repo: https://github.com/nathanblumenfeld/collegebaseball
- Upstream documentation: https://collegebaseball.readthedocs.io/en/latest/index.html

The upstream project provides an “intuitive API for stats.ncaa.org”.
In our use case, the upstream version needed a few updates to support newer seasons and to avoid features that rely on unavailable weights.

---

## Why this fork exists

We use team-level NCAA game logs to build **pre-game rolling performance features** for predicting UNC home attendance.

We needed:

1) **Season coverage extended to 2024–2026** (library originally supported 2013–2023 in our environment).
2) Ability to **disable weighted/advanced metrics** (linear weights not available for newer seasons).
3) Compatibility with NCAA site blocking: the pipeline must run via a **headed browser patch** (implemented at runtime in our notebooks).

---

## What changed vs upstream

### 1) `collegebaseball/metrics.py` — add `include_weighted` flag
- `add_batting_metrics(df, season=True, include_weighted=False)`
- `add_pitching_metrics(df, season=True, include_weighted=False)`

When `include_weighted=False`:
- weighted outputs like `wOBA`, `wRAA`, `wRC`, `FIP`, `wOBA-against` are kept as **NaN** instead of being computed from missing weights.

This lets us call `ncaa_team_game_logs(..., include_advanced=False)` in the project without breaking newer seasons.

### 2) `collegebaseball/ncaa_utils.py` — forward-extend game-log header schemas
- Added a forward-fill mechanism that copies 2023 header schemas into 2024–2026 when missing.
- Added a special case for **2024+ pitching** where NCAA appears to have dropped the leading `G` column.

### 3) Seasons table updates (data files)
The library uses `seasons.csv` and `seasons.parquet` in `collegebaseball/data/` to map:
- `season` → `season_id`, `batting_id`, `pitching_id`, `fielding_id`

We updated these to include the missing newer seasons.

> Note: the “season” argument in the library corresponds to the **end year** of the NCAA “YYYY-YY” season label (e.g., `2026` corresponds to “2025–26”). This matches how the upstream docs describe season lookups and how the NCAA pages label seasons.

---

## IMPORTANT: headed browser patch required

In our environment, direct HTTP requests to stats.ncaa.org often returned “Access Denied” (403). To run reliably, we patch the library’s `requests.Session.get()` to use Selenium and return the browser HTML.

This patch is implemented in `02_final_game_stat_scraper.ipynb` (and is required for reproducibility).

### Cache location
Caches are written to `scrapers/game_stat_scraper/vendor/collegebaseball/cache/` and are **not committed** to Git.

---


## Attribution & licensing
Upstream `collegebaseball` is MIT licensed.

