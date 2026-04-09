# Repo hygiene checklist (to avoid clutter)

## Recommended directory structure

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


## .gitignore suggestions

__pycache__/
.ipynb_checkpoints/
.DS_Store