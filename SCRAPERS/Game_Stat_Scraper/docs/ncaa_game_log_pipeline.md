# NCAA game-by-game team stats pipeline (UNC home attendance project)

This folder contains the **write-ups** you can paste into your group GitHub repo to document how we got from “library doesn’t load recent seasons / gets blocked” to a working, cached game-log + feature pipeline.

## What this work enables

For each **UNC-hosted** game (Boshamer / Chapel Hill location) in the GoHeels master dataset, we can attach:

- **UNC pre-game rolling features** computed from NCAA game logs (batting/pitching/fielding) using *only games prior to the matchup* (shift(1) + rolling windows).
- **Opponent pre-game rolling features** computed from the opponent’s NCAA game logs using *only games prior to the UNC matchup*, with doubleheader ordering respected.

The NCAA game log pulls are **cached** as one file per (team, season, variant) to avoid repeated scraping.

---

## Repository organization (GitHub)

All work related to NCAA team game logs lives under:

```
scrapers/game_stat_scraper/
  notebooks/
    00_collegebaseball_Scraper_rebuild.ipynb
    01_final_game_stat_scraper.ipynb
  docs/
    ncaa_game_log_pipeline.md
    collegebaseball_fork.md
    repo_hygiene.md
  data/                      # project inputs
    unc_baseball_final_20260302_220506.csv
    ncaa_school_mapping.csv
  exports/                   # derived outputs (NOT committed)
  vendor/
    collegebaseball/         # vendored fork (MIT; keep upstream LICENSE)
      cache/                 # NOT committed (Parquet caches)
      exports/               # NOT committed
```

**Inputs** live in `scrapers/game_stat_scraper/data/` so notebooks can run without hardcoded local paths.

**Caches** live under `scrapers/game_stat_scraper/vendor/collegebaseball/cache/` and are generated locally (do not commit).


## Development timeline (how we got here)


### Notebook 1 — `collegebaseball_Scraper_rebuild.ipynb` (extend library coverage + remove advanced-weight dependencies)

Primary outcomes (the “hard” fixes):

1) **Extend supported seasons beyond 2013–2023**
- Identified that the library uses a local seasons table (CSV + Parquet) mapping “season” → (`season_id`, `batting_id`, `pitching_id`, `fielding_id`).
- Scraped the required IDs for newer seasons (e.g., 2024–2026) and wrote them into the library’s `seasons.csv` and `seasons.parquet`.

2) **Disable advanced metrics that depend on linear weights**
- Some “advanced” stats (e.g., wOBA, FIP) depend on weights that were only available for a limited year range.
- We modified the library so those “weighted” columns are **optional** and can be disabled, keeping the scrape usable for 2024+.

3) **Headed browser requirement (site blocking)**
- Observed that requests-only pulls frequently return “Access Denied” / 403.
- Confirmed that running the pipeline through a **headed browser** succeeds reliably.

4) **Schema continuity work**
- Began validating which team game-log columns are consistently available across years and which must be standardized/dropped for a stable 2013–present schema.

---

### Notebook 2 — `final_game_stat_scraper.ipynb` (end-to-end, productionized pipeline)

Primary outcomes (this is the notebook you point people to first):

1) **Selenium patch (headed browser)**
- Monkeypatch `collegebaseball.ncaa_scraper.Session.get` so all library GETs execute via Selenium and return `page_source`.

2) **Caching**
- One Parquet file per `(school_id, season, variant)` for UNC and for each opponent encountered in UNC-hosted games.
- Adds retry/backoff + “blocked/empty pull” detection to survive transient NCAA site issues.

3) **Standardization**
- Batting: canonicalize DP vs OPP DP into `opp_dp`, drop `GDP`.
- Pitching: drop `CG` and `pickoffs` (inconsistent year availability).
- Fielding: recompute `TC = PO + A + E` everywhere.

4) **Feature engineering**
- Builds game_id-keyed feature tables (shift(1) + rolling window L10) for each team-season.
- Applies your chosen feature set (HR lags for UNC only; rate features; pitching & fielding rolling metrics).
- Produces one final dataset with one row per UNC-hosted game and UNC+opponent features.

---

## Quick start (repro)

1) Install dependencies (local):
- Python 3.10+
- Chrome installed
- `pip install -r requirements.txt` (or conda env)
2) Install your forked `collegebaseball` (editable recommended)
3) Run `final_game_stat_scraper.ipynb` top-to-bottom with **headed Selenium patch enabled**
