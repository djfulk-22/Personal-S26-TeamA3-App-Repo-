
from __future__ import annotations

import calendar
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ==========================================================
# Prediction bundle configuration
# ==========================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DEFAULT_MONTHS = [2, 3, 4, 5, 6]  # Feb-Jun

MODEL_CONFIGS = {
    "predicted_midseason": {
        "label": "Midseason",
        "full": "db_predictions_2024_2026_full_df.csv",
        "raw": "db_predictions_2024_2026_tier5_raw_tier_cols.csv",
        "final": "db_predictions_2024_2026_tier5_final_feats_export.csv",
    },
    "predicted_preseason": {
        "label": "Preseason",
        "full": "preseason_predictions_2024_2026_full_df.csv",
        "raw": "preseason_predictions_2024_2026_tier4_raw_tier_cols.csv",
        "final": "preseason_predictions_2024_2026_tier4_final_feats_export.csv",
    },
}

NON_FEATURE_COLUMNS = {
    "event_id", "season_year", "season_year_pred", "game_number", "game_number_pred",
    "date", "date_pred", "opponent", "opponent_pred", "attendance", "actual",
    "predicted", "predicted_raw", "error", "abs_error", "pct_error", "attendance_pred",
}

COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "season": ["season_year", "season", "season_year_pred"],
    "game_date": ["date", "game_date", "date_pred"],
    "opponent": ["opponent", "opponent_pred"],
    "prediction": ["predicted_midseason", "predicted_preseason", "predicted"],
    "actual": ["actual", "attendance", "attendance_pred"],
    "attendance": ["attendance", "actual", "attendance_pred"],
    "event_id": ["event_id", "game_id"],
    "start_time": ["time", "start_time"],
    "exact_time": ["exact_time"],
    "day_of_week": ["day_of_week"],
    "doubleheader": ["doubleheader"],
    "dh_game_number": ["dh_game_number"],
    "series_game_number": ["game_in_series"],
    "promotion": ["promotion"],
}


# ==========================================================
# Formatting helpers
# ==========================================================
def format_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return str(value)


def format_number(value: object, ndigits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):,.{ndigits}f}"
    except Exception:
        return str(value)


def format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{float(value):,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def normalize_season_display(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    text = str(value).strip()
    if text.endswith('.0'):
        return text[:-2]
    return text


def format_start_time(value: object) -> str:
    if value is None or pd.isna(value):
        return "Time TBA"
    text = str(value).strip()
    if text in {"", "nan", "NaT", "None"}:
        return "Time TBA"

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, format="%H:%M:%S", errors="coerce")
    if pd.isna(parsed):
        return text

    return parsed.strftime("%I:%M %p").lstrip("0")


FEATURE_LABEL_OVERRIDES = {
    "season_year": "Season",
    "season": "Season",
    "date": "Game date",
    "opponent": "Opponent",
    "time": "Start time",
    "exact_time": "Exact start time",
    "season_stage": "Season stage",
    "day_of_week": "Day of week",
    "time_bucket": "Time bucket",
    "weekend": "Weekend game",
    "month": "Month",
    "opening_day": "Opening day",
    "game_number": "Game number",
    "days_since_last_hg": "Days since last home game",
    "early_season": "Early season",
    "playoff": "Postseason game",
    "series": "Series flag",
    "game_in_series": "Game in series",
    "doubleheader": "Doubleheader",
    "dh_game_number": "Doubleheader game number",
    "acc": "ACC game",
    "rivalry": "Rivalry game",
    "opp_distance_miles": "Opponent travel distance (miles)",
    "opponent_grp": "Opponent group",
    "win": "UNC win",
    "unc_score": "UNC score",
    "opp_score": "Opponent score",
    "mbb_any": "Any UNC men's basketball conflict",
    "mbb_home": "UNC men's basketball home conflict",
    "mbb_away": "UNC men's basketball away conflict",
    "mbb_tourney": "UNC men's basketball tournament conflict",
    "venue_cap": "Venue capacity",
    "sold_out": "Sold out",
    "low_attendance": "Low attendance flag",
    "lag_series_attendance": "Previous game attendance in series",
    "prev_attendance": "Previous home game attendance",
    "att_avg_last3": "Average attendance over last 3 home games",
    "att_avg_last5": "Average attendance over last 5 home games",
    "att_avg_season_to_date": "Season-to-date average attendance",
    "prev_season_attendance": "Previous season average attendance",
    "prev_3yr_attendance": "3-year average attendance",
    "attendance_yoy_change": "Year-over-year attendance change",
    "promo_count": "Number of promotions",
    "promo_giveaway": "Giveaway promotion",
    "promo_fireworks": "Fireworks promotion",
    "promo_kids": "Kids promotion",
    "promo_bark": "Bark at the Park promotion",
    "promo_discount": "Discount promotion",
    "promo_theme": "Theme promotion",
    "promo_trivia": "Trivia promotion",
    "record_pct": "UNC win percentage",
    "runs_pg": "UNC runs per game",
    "hr_pg": "UNC home runs per game",
    "errors_pg": "UNC errors per game",
    "idp_pg": "UNC double plays per game",
    "run_diff_pg": "UNC run differential per game",
    "win_last": "Won previous game",
    "wins_last3": "UNC wins over last 3 games",
    "wins_last5": "UNC wins over last 5 games",
    "prev_szn_record_pct": "UNC previous-season win percentage",
    "runs_pg_prev_season": "UNC previous-season runs per game",
    "runs_allowed_pg_prev_season": "UNC previous-season runs allowed per game",
    "hr_pg_prev_season": "UNC previous-season home runs per game",
    "errors_pg_prev_season": "UNC previous-season errors per game",
    "k_pg_prev_season": "UNC previous-season strikeouts at plate per game",
    "so_pg_prev_season": "UNC previous-season strikeouts pitched per game",
    "hr_allowed_pg_prev_season": "UNC previous-season home runs allowed per game",
    "idp_pg_prev_season": "UNC previous-season double plays per game",
    "run_diff_pg_prev_season": "UNC previous-season run differential per game",
    "win_pct_prev_season": "UNC previous-season win percentage",
    "team_strength_prev_season": "UNC previous-season team strength",
    "opp_runs_pg": "Opponent runs per game",
    "opp_hr_pg": "Opponent home runs per game",
    "opp_errors_pg": "Opponent errors per game",
    "opp_run_diff_pg": "Opponent run differential per game",
    "opp_win_pct": "Opponent win percentage",
    "opp_wins_last3": "Opponent wins over last 3 games",
    "opp_wins_last5": "Opponent wins over last 5 games",
    "opp_runs_pg_prev_season": "Opponent previous-season runs per game",
    "opp_runs_allowed_pg_prev_season": "Opponent previous-season runs allowed per game",
    "opp_hr_pg_prev_season": "Opponent previous-season home runs per game",
    "opp_errors_pg_prev_season": "Opponent previous-season errors per game",
    "opp_k_pg_prev_season": "Opponent previous-season strikeouts at plate per game",
    "opp_so_pg_prev_season": "Opponent previous-season strikeouts pitched per game",
    "opp_hr_allowed_pg_prev_season": "Opponent previous-season home runs allowed per game",
    "opp_idp_pg_prev_season": "Opponent previous-season double plays per game",
    "opp_run_diff_pg_prev_season": "Opponent previous-season run differential per game",
    "opp_win_pct_prev_season": "Opponent previous-season win percentage",
    "opp_team_strength_prev_season": "Opponent previous-season team strength",
    "strength_diff": "Current-season team strength difference",
    "strength_diff_prev_season": "Previous-season team strength difference",
    "win_pct_diff": "Current-season win percentage difference",
    "win_pct_diff_prev_season": "Previous-season win percentage difference",
    "has_opponent_features": "Opponent feature availability",
    "momentum_index": "Momentum index",
    "excitement_index": "Excitement index",
    "wx_d1_temperature": "Weather forecast temperature (1 day out)",
    "wx_d1_humidity": "Weather forecast humidity (1 day out)",
    "wx_d1_wind_speed": "Weather forecast wind speed (1 day out)",
    "wx_d1_precipitation": "Weather forecast precipitation (1 day out)",
    "wx_d1_cloud_cover": "Weather forecast cloud cover (1 day out)",
    "wx_d1_weather_condition": "Weather forecast condition (1 day out)",
    "wx_d1_comfort_index": "Weather comfort index (1 day out)",
    "wx_d1_temp_avg_prev24h": "Average temperature over previous 24 hours",
    "wx_d1_had_rain_prev24h": "Rain in previous 24 hours",
    "acad_sesh_active": "Academic session active",
    "acad_semester": "Academic semester",
    "acad_break": "Academic break",
    "acad_exams": "Exams period",
    "acad_break_type": "Academic break type",
    "school_hours": "School hours",
    "school_night": "School night",
    "undergrad_enr": "Undergraduate enrollment",
    "graduate_enr": "Graduate enrollment",
    "professional_enr": "Professional enrollment",
    "female_enr": "Female enrollment",
    "male_enr": "Male enrollment",
    "total_enr": "Total enrollment",
    "gt_web": "Google Trends web interest",
    "gt_youtube": "Google Trends YouTube interest",
    "gt_total": "Google Trends total interest",
    "gt_web_preseason": "Preseason Google Trends web interest",
    "gt_youtube_preseason": "Preseason Google Trends YouTube interest",
    "gt_total_preseason": "Preseason Google Trends total interest",
}


def feature_display_name(column: str) -> str:
    if column in FEATURE_LABEL_OVERRIDES:
        return FEATURE_LABEL_OVERRIDES[column]

    cleaned = column.replace(".1", " duplicate")
    cleaned = cleaned.replace("_", " ").strip()
    cleaned = cleaned.replace(" pg ", " per game ")
    cleaned = cleaned.replace(" pct ", " percentage ")
    cleaned = cleaned.replace(" prev season", " previous season")
    cleaned = cleaned.replace(" prev szn", " previous season")
    cleaned = cleaned.replace(" opp ", " opponent ")
    cleaned = cleaned.replace(" dh ", " doubleheader ")
    cleaned = cleaned.replace(" gt ", " Google Trends ")
    cleaned = cleaned.replace(" wx ", " weather ")
    cleaned = cleaned.replace(" acad ", " academic ")
    cleaned = cleaned.replace(" hr ", " home runs ")
    cleaned = cleaned.replace(" so ", " strikeouts ")
    cleaned = cleaned.replace(" idp ", " double plays ")
    return cleaned.title()


# ==========================================================
# Data loading and prep
# ==========================================================
def infer_column(df: pd.DataFrame, logical_name: str) -> Optional[str]:
    for candidate in COLUMN_CANDIDATES.get(logical_name, []):
        if candidate in df.columns:
            return candidate
    return None


@st.cache_data(show_spinner=False)
def load_model_bundle(data_dir: str) -> Dict[str, Dict[str, object]]:
    base = Path(data_dir)
    bundle: Dict[str, Dict[str, object]] = {}

    for pred_alias, cfg in MODEL_CONFIGS.items():
        paths = {name: base / filename for name, filename in cfg.items() if name in {"full", "raw", "final"}}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing prediction bundle files: " + ", ".join(missing) + ". "
                "Put the six CSVs from your predictions export into the app's data/ folder."
            )

        full_df = pd.read_csv(paths["full"])
        raw_df = pd.read_csv(paths["raw"])
        final_df = pd.read_csv(paths["final"])

        bundle[pred_alias] = {
            "label": cfg["label"],
            "full": full_df,
            "raw_cols": [c for c in raw_df.columns if c not in NON_FEATURE_COLUMNS],
            "final_cols": [c for c in final_df.columns if c not in NON_FEATURE_COLUMNS],
        }
    return bundle


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col.startswith("_"):
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0 and converted.notna().sum() >= max(1, int(0.9 * df[col].notna().sum())):
                df[col] = converted
    return df


@st.cache_data(show_spinner=False)
def prep_model_data(data_dir: str) -> Dict[str, Dict[str, object]]:
    bundle = load_model_bundle(data_dir)
    prepped: Dict[str, Dict[str, object]] = {}

    for pred_alias, meta in bundle.items():
        df = meta["full"].copy()
        df = _coerce_numeric_columns(df)
        df[pred_alias] = pd.to_numeric(df["predicted"], errors="coerce")

        colmap = {name: infer_column(df, name) for name in COLUMN_CANDIDATES}
        required = ["season", "game_date", "opponent"]
        missing = [name for name in required if colmap.get(name) is None]
        if missing:
            raise ValueError(
                f"{meta['label']} data is missing required columns: " + ", ".join(missing)
            )

        df["_season"] = df[colmap["season"]].apply(normalize_season_display)
        df["_game_date"] = pd.to_datetime(df[colmap["game_date"]].astype(str).str[:10], errors="coerce").dt.normalize()
        df["_opponent"] = df[colmap["opponent"]].astype(str)
        df["_event_id"] = df[colmap["event_id"]].astype(str) if colmap.get("event_id") else (
            df["_season"]
            + "_"
            + df["_game_date"].dt.strftime("%Y%m%d").fillna("unknown")
            + "_"
            + df["_opponent"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        )

        if colmap.get("actual") and colmap["actual"] in df.columns:
            df["_actual"] = pd.to_numeric(df[colmap["actual"]], errors="coerce")
        else:
            df["_actual"] = np.nan

        if colmap.get("attendance") and colmap["attendance"] in df.columns:
            df["_attendance"] = pd.to_numeric(df[colmap["attendance"]], errors="coerce")
        else:
            df["_attendance"] = np.nan

        if colmap.get("start_time") and colmap["start_time"] in df.columns:
            df["_start_time"] = df[colmap["start_time"]].astype(str)
        else:
            df["_start_time"] = ""

        if colmap.get("exact_time") and colmap["exact_time"] in df.columns:
            exact_dt = pd.to_datetime(df[colmap["exact_time"]], errors="coerce")
        else:
            exact_dt = pd.Series(pd.NaT, index=df.index)

        if exact_dt.isna().all():
            combined = (
                df["_game_date"].dt.strftime("%Y-%m-%d").fillna("")
                + " "
                + df["_start_time"].replace({"nan": "", "NaT": ""})
            ).str.strip()
            exact_dt = pd.to_datetime(combined, errors="coerce")

        df["_exact_dt"] = exact_dt
        df["_sort_dt"] = df["_exact_dt"].fillna(df["_game_date"])

        base_id = df["_event_id"]
        dup_n = df.groupby(base_id).cumcount() + 1
        dup_size = base_id.groupby(base_id).transform("size")
        df["_row_uid"] = np.where(dup_size > 1, base_id + "__" + dup_n.astype(str), base_id)
        df["_dup_suffix"] = np.where(dup_size > 1, " entry " + dup_n.astype(str), "")

        if colmap.get("dh_game_number") and colmap["dh_game_number"] in df.columns:
            df[colmap["dh_game_number"]] = pd.to_numeric(df[colmap["dh_game_number"]], errors="coerce")

        if colmap.get("day_of_week") is None:
            df["_derived_day_of_week"] = df["_game_date"].dt.day_name()
            colmap["day_of_week"] = "_derived_day_of_week"

        sort_cols = ["_season", "_game_date", "_sort_dt"]
        if colmap.get("dh_game_number") and colmap["dh_game_number"] in df.columns:
            sort_cols.append(colmap["dh_game_number"])
        sort_cols.append("_row_uid")
        df = df.sort_values(sort_cols).reset_index(drop=True)

        prepped[pred_alias] = {
            "label": meta["label"],
            "df": df,
            "colmap": colmap,
            "raw_cols": list(meta["raw_cols"]),
            "final_cols": list(meta["final_cols"]),
        }

    return prepped


# ==========================================================
# Feature grouping
# ==========================================================
def _sort_group_columns(cols: List[str], final_cols: set[str], raw_cols: set[str]) -> List[str]:
    def key(c: str) -> Tuple[int, str]:
        if c in final_cols:
            return (0, c)
        if c in raw_cols:
            return (1, c)
        return (2, c)

    return sorted(dict.fromkeys(cols), key=key)


def build_feature_groups(df: pd.DataFrame, raw_cols: List[str], final_cols: List[str]) -> Dict[str, List[str]]:
    all_cols = set(df.columns)
    raw_set = set(raw_cols)
    final_set = set(final_cols)

    groups: Dict[str, List[str]] = {
        "Schedule and matchup": [
            c for c in [
                "season_year", "season", "date", "opponent", "time", "exact_time", "season_stage",
                "day_of_week", "time_bucket", "weekend", "month", "opening_day", "game_number",
                "days_since_last_hg", "early_season", "playoff", "series", "game_in_series",
                "doubleheader", "dh_game_number", "acc", "rivalry", "opp_distance_miles",
                "opponent_grp",
            ] if c in all_cols
        ],
        "Competition context": [
            c for c in ["win", "unc_score", "opp_score", "mbb_any", "mbb_home", "mbb_away", "mbb_tourney"]
            if c in all_cols
        ],
        "Attendance history": [
            c for c in [
                "venue_cap", "sold_out", "low_attendance", "lag_series_attendance",
                "prev_attendance", "att_avg_last3", "att_avg_last5", "att_avg_season_to_date",
                "prev_season_attendance", "prev_3yr_attendance", "attendance_yoy_change"
            ] if c in all_cols
        ],
        "Promotions": [c for c in sorted(all_cols) if c == "promo_count" or c.startswith("promo_")],
        "UNC current season": [
            c for c in [
                "record_pct", "runs_pg", "hr_pg", "errors_pg", "idp_pg", "run_diff_pg",
                "win_last", "wins_last3", "wins_last5"
            ] if c in all_cols
        ],
        "UNC previous season": [
            c for c in [
                "prev_szn_record_pct", "runs_pg_prev_season", "runs_allowed_pg_prev_season",
                "hr_pg_prev_season", "errors_pg_prev_season", "k_pg_prev_season", "so_pg_prev_season",
                "hr_allowed_pg_prev_season", "idp_pg_prev_season", "run_diff_pg_prev_season",
                "win_pct_prev_season", "team_strength_prev_season"
            ] if c in all_cols
        ],
        "Opponent current season": [
            c for c in [
                "opp_runs_pg", "opp_hr_pg", "opp_errors_pg", "opp_run_diff_pg", "opp_win_pct",
                "opp_wins_last3", "opp_wins_last5"
            ] if c in all_cols
        ],
        "Opponent previous season": [
            c for c in [
                "opp_runs_pg_prev_season", "opp_runs_allowed_pg_prev_season", "opp_hr_pg_prev_season",
                "opp_errors_pg_prev_season", "opp_k_pg_prev_season", "opp_so_pg_prev_season",
                "opp_hr_allowed_pg_prev_season", "opp_idp_pg_prev_season", "opp_run_diff_pg_prev_season",
                "opp_win_pct_prev_season", "opp_team_strength_prev_season"
            ] if c in all_cols
        ],
        "Matchup indices": [
            c for c in [
                "strength_diff", "strength_diff_prev_season",
                "win_pct_diff", "win_pct_diff_prev_season",
                "momentum_index", "excitement_index", "has_opponent_features"
            ] if c in all_cols
        ],
        "Weather": [c for c in sorted(all_cols) if c.startswith("wx_")],
        "Academic calendar": [c for c in sorted(all_cols) if c.startswith("acad_")] + [c for c in ["school_hours", "school_night"] if c in all_cols],
        "Enrollment and interest": [
            c for c in ["undergrad_enr", "graduate_enr", "professional_enr", "female_enr", "male_enr", "total_enr"] if c in all_cols
        ] + [c for c in sorted(all_cols) if c.startswith("gt_")],
    }

    clean_groups: Dict[str, List[str]] = {}
    for group_name, cols in groups.items():
        existing = [c for c in cols if c in all_cols]
        if not existing:
            continue
        existing = _sort_group_columns(existing, final_set, raw_set)
        clean_groups[group_name] = existing

    return clean_groups


# ==========================================================
# Calendar rendering
# ==========================================================
def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = [max(0, min(255, int(round(v)))) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def interpolate_color(color1: str, color2: str, t: float) -> str:
    c1 = hex_to_rgb(color1)
    c2 = hex_to_rgb(color2)
    rgb = tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(3))
    return rgb_to_hex(rgb)


def continuous_color(value: float, vmin: float, vmax: float) -> str:
    if pd.isna(value):
        return "#f3f6fb"
    if vmax <= vmin:
        return "#7bafd4"

    t = max(0.0, min(1.0, (float(value) - vmin) / (vmax - vmin)))
    low = "#eef7ff"
    mid = "#7bafd4"
    high = "#123c7a"
    if t <= 0.5:
        return interpolate_color(low, mid, t / 0.5)
    return interpolate_color(mid, high, (t - 0.5) / 0.5)


def relative_luminance(hex_color: str) -> float:
    r, g, b = [channel / 255.0 for channel in hex_to_rgb(hex_color)]

    def adjust(channel: float) -> float:
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    r_lin, g_lin, b_lin = adjust(r), adjust(g), adjust(b)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def text_color_for_bg(hex_color: str) -> str:
    return "#0f172a" if relative_luminance(hex_color) >= 0.42 else "#f8fafc"


def game_chip_style(text_color: str) -> Tuple[str, str]:
    if text_color == "#0f172a":
        return "rgba(255,255,255,0.52)", "rgba(255,255,255,0.55)"
    return "rgba(15,23,42,0.18)", "rgba(255,255,255,0.24)"


def make_gradient_legend(vmin: float, vmax: float) -> str:
    mid = (vmin + vmax) / 2 if pd.notna(vmin) and pd.notna(vmax) else np.nan
    return f"""
    <div style='margin-bottom: 1rem;'>
        <div style='font-size:0.95rem; font-weight:600; margin-bottom:0.3rem;'>Continuous calendar scale</div>
        <div style='height:16px; border-radius:999px; background: linear-gradient(90deg, #eef7ff 0%, #7bafd4 50%, #123c7a 100%); border:1px solid #d9e2ef;'></div>
        <div style='display:flex; justify-content:space-between; font-size:0.85rem; color:#44536a; margin-top:0.3rem;'>
            <span>{format_int(vmin)}</span>
            <span>{format_int(mid)}</span>
            <span>{format_int(vmax)}</span>
        </div>
    </div>
    """


def make_game_chip(row: pd.Series, pred_col: str, colmap: Dict[str, Optional[str]], parent_bg: str) -> str:
    time_val = format_start_time(row["_start_time"])
    opp = row["_opponent"]
    pred_val = format_int(row[pred_col])
    text_color = text_color_for_bg(parent_bg)
    chip_bg, chip_border = game_chip_style(text_color)

    tags = []
    dh_col = colmap.get("doubleheader")
    dh_num_col = colmap.get("dh_game_number")
    if dh_col and dh_col in row.index and pd.notna(row[dh_col]) and int(float(row[dh_col])) == 1:
        if dh_num_col and dh_num_col in row.index and pd.notna(row[dh_num_col]):
            tags.append(f"DH{int(float(row[dh_num_col]))}")
        else:
            tags.append("Doubleheader")

    series_col = colmap.get("series_game_number")
    if (
        series_col
        and series_col in row.index
        and pd.notna(row[series_col])
        and float(row[series_col]) > 0
    ):
        tags.append(f"Series {int(float(row[series_col]))}")

    suffix = row["_dup_suffix"]
    suffix_text = f"<div style='font-size:0.70rem; opacity:0.82; color:{text_color};'>Row{suffix}</div>" if suffix else ""

    tags_html = ""
    if tags:
        tags_html = "<div style='margin-top:4px; font-size:0.70rem; opacity:0.9; color:{0};'>".format(text_color) + " • ".join(tags) + "</div>"

    return f"""
    <div style='margin-top:6px; padding:7px 8px; border-radius:8px; background:{chip_bg}; border:1px solid {chip_border}; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);'>
        <div style='font-size:0.74rem; font-weight:700; letter-spacing:0.02em; color:{text_color};'>{time_val}</div>
        <div style='font-size:0.85rem; font-weight:600; line-height:1.2; color:{text_color};'>vs {opp}</div>
        <div style='font-size:0.78rem; color:{text_color};'>Pred: {pred_val}</div>
        {tags_html}
        {suffix_text}
    </div>
    """


def render_month_html(
    season_games: pd.DataFrame,
    pred_col: str,
    year: int,
    month: int,
    vmin: float,
    vmax: float,
    colmap: Dict[str, Optional[str]],
) -> str:
    cal = calendar.Calendar(firstweekday=6)
    month_weeks = cal.monthdayscalendar(year, month)
    month_games = season_games[season_games["_game_date"].dt.month == month].copy()

    daily_games = {
        day: grp.sort_values(["_sort_dt", "_row_uid"])
        for day, grp in month_games.groupby(month_games["_game_date"].dt.day)
        if int(day) > 0
    }

    header = "".join(
        f"<th style='padding:8px; font-size:0.83rem; color:#cbd5e1; background:#0f172a; border:1px solid #1e293b;'>{d}</th>"
        for d in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    )

    rows_html = []
    for week in month_weeks:
        cells = []
        for day in week:
            if day == 0:
                cells.append("<td style='background:#111827; height:150px; border:1px solid #1f2937;'></td>")
                continue

            games = daily_games.get(day)
            if games is None or games.empty:
                cells.append(
                    f"""
                    <td style='vertical-align:top; height:150px; border:1px solid #d6dee8; background:#f8fafc; padding:8px;'>
                        <div style='font-weight:700; font-size:0.9rem; margin-bottom:6px; color:#334155;'>{day}</div>
                    </td>
                    """
                )
            else:
                pred_value = games[pred_col].mean()
                bg = continuous_color(pred_value, vmin, vmax)
                text_color = text_color_for_bg(bg)
                game_lines = "".join(make_game_chip(row, pred_col, colmap, bg) for _, row in games.iterrows())
                day_summary_html = ""
                if len(games) > 1:
                    day_summary_html = (
                        f"<div style='font-size:0.72rem; opacity:0.95; text-align:right; color:{text_color};'>"
                        f"Predicted Day Average<br>{format_int(pred_value)}</div>"
                    )

                cells.append(
                    f"""
                    <td style='vertical-align:top; height:150px; max-height:220px; border:1px solid #d6dee8; background:{bg}; padding:8px; overflow-y:auto;'>
                        <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:8px;'>
                            <div style='font-weight:700; font-size:0.92rem; color:{text_color};'>{day}</div>
                            {day_summary_html}
                        </div>
                        {game_lines}
                    </td>
                    """
                )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
    <div style='margin-bottom: 1.35rem;'>
        <div style='font-size:1.15rem; font-weight:700; margin: 0.35rem 0 0.5rem 0; color:#e5e7eb;'>{calendar.month_name[month]} {year}</div>
        <table style='width:100%; border-collapse:collapse; table-layout:fixed; background:transparent;'>
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
    </div>
    """


def render_calendar_view(
    season_games: pd.DataFrame,
    pred_col: str,
    months: Iterable[int],
    colmap: Dict[str, Optional[str]],
) -> str:
    if season_games.empty:
        return "<p>No games found for the selected season.</p>"

    clean = season_games[pred_col].dropna()
    vmin = float(clean.min()) if not clean.empty else 0.0
    vmax = float(clean.max()) if not clean.empty else 1.0

    html_parts = [make_gradient_legend(vmin, vmax)]
    for month in months:
        month_games = season_games[season_games["_game_date"].dt.month == month]
        if month_games.empty:
            continue
        year_mode = int(month_games["_game_date"].dt.year.mode().iloc[0])
        html_parts.append(render_month_html(season_games, pred_col, year_mode, month, vmin, vmax, colmap))

    if len(html_parts) == 1:
        html_parts.append("<p>No games were found in the selected month range.</p>")
    return "".join(html_parts)


# ==========================================================
# Detail view helpers
# ==========================================================
def make_game_label(row: pd.Series, pred_col: str, colmap: Dict[str, Optional[str]]) -> str:
    date_str = row["_game_date"].strftime("%b %d, %Y")
    time_str = format_start_time(row["_start_time"])
    dh_col = colmap.get("dh_game_number")
    dh_text = ""
    if dh_col and dh_col in row.index and pd.notna(row[dh_col]):
        dh_text = f" | DH{int(float(row[dh_col]))}"
    pred_text = format_int(row[pred_col])
    suffix = row["_dup_suffix"]
    suffix_text = f" |{suffix.strip()}" if suffix else ""
    return f"{date_str} | {time_str} | vs {row['_opponent']}{dh_text} | pred {pred_text}{suffix_text}"


def summarize_season(season_games: pd.DataFrame, pred_col: str) -> Dict[str, str]:
    out = {
        "Games": str(len(season_games)),
        "Average prediction": format_int(season_games[pred_col].mean()),
        "Highest prediction": format_int(season_games[pred_col].max()),
        "Lowest prediction": format_int(season_games[pred_col].min()),
    }
    if season_games["_actual"].notna().any():
        actual = season_games["_actual"]
        out["Average actual"] = format_int(actual.mean())
        out["MAE"] = format_int((season_games[pred_col] - actual).abs().mean())
    return out


def _extreme_game_text(rows: pd.DataFrame, pred_col: str, colmap: Dict[str, Optional[str]]) -> str:
    labels = []
    for _, row in rows.sort_values(["_sort_dt", "_row_uid"]).iterrows():
        date_str = row["_game_date"].strftime("%b %d")
        dh_text = ""
        dh_col = colmap.get("dh_game_number")
        if dh_col and dh_col in row.index and pd.notna(row[dh_col]):
            dh_text = f" DH{int(float(row[dh_col]))}"
        labels.append(f"{date_str} vs {row['_opponent']}{dh_text}")
    return "; ".join(labels)


def describe_prediction_extremes(season_games: pd.DataFrame, pred_col: str, colmap: Dict[str, Optional[str]]) -> Dict[str, str]:
    highest_value = season_games[pred_col].max()
    lowest_value = season_games[pred_col].min()
    highest_rows = season_games.loc[season_games[pred_col] == highest_value]
    lowest_rows = season_games.loc[season_games[pred_col] == lowest_value]
    return {
        "Highest prediction game": f"{_extreme_game_text(highest_rows, pred_col, colmap)} ({format_int(highest_value)})",
        "Lowest prediction game": f"{_extreme_game_text(lowest_rows, pred_col, colmap)} ({format_int(lowest_value)})",
    }


def build_feature_frame(row: pd.Series, groups: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for group_name, cols in groups.items():
        existing = [c for c in cols if c in row.index]
        if not existing:
            continue
        sub = pd.DataFrame({
            "Feature": [feature_display_name(c) for c in existing],
            "Value": [format_value(row[c]) for c in existing],
        })
        out[group_name] = sub
    return out


# ==========================================================
# App layout
# ==========================================================
st.set_page_config(
    page_title="UNC Baseball Attendance Forecasting",
    page_icon="⚾",
    layout="wide",
)

st.title("UNC Baseball Attendance Forecasting")
st.caption("Calendar-level planning plus single-game inspection using your actual prediction file.")

try:
    model_data = prep_model_data(str(DATA_DIR))
except Exception as exc:
    st.error(str(exc))
    st.stop()

all_seasons = sorted({season for meta in model_data.values() for season in meta["df"]["_season"].dropna().unique().tolist()})
all_prediction_options = [key for key in MODEL_CONFIGS if key in model_data]
default_pred_col = "predicted_midseason" if "predicted_midseason" in all_prediction_options else all_prediction_options[0]

with st.sidebar:
    st.header("Controls")
    selected_season = st.selectbox("Season", all_seasons, index=len(all_seasons) - 1 if all_seasons else 0)
    selected_pred_col = st.selectbox(
        "Prediction column",
        all_prediction_options,
        index=all_prediction_options.index(default_pred_col),
        help="Choose which prediction column drives the calendar and summary metrics.",
    )
    show_months = st.multiselect(
        "Months to display",
        options=list(calendar.month_name)[1:],
        default=[calendar.month_name[m] for m in DEFAULT_MONTHS],
    )

selected_meta = model_data[selected_pred_col]
df = selected_meta["df"]
colmap = selected_meta["colmap"]
feature_groups = build_feature_groups(df, selected_meta["raw_cols"], selected_meta["final_cols"])

season_games = df[df["_season"] == selected_season].copy()
month_lookup = {calendar.month_name[m]: m for m in range(1, 13)}
selected_month_numbers = [month_lookup[m] for m in show_months]
season_games = season_games[season_games["_game_date"].dt.month.isin(selected_month_numbers)].copy()

if season_games.empty:
    st.warning("No rows were found for that season and month selection.")
    st.stop()

season_games = season_games.sort_values(["_sort_dt", "_row_uid"]).reset_index(drop=True)

option_map = {
    make_game_label(row, selected_pred_col, colmap): row["_row_uid"]
    for _, row in season_games.iterrows()
}
selected_label = st.selectbox("Game to inspect in detail", list(option_map.keys()), index=0)
selected_row = season_games.loc[season_games["_row_uid"] == option_map[selected_label]].iloc[0]

summary = summarize_season(season_games, selected_pred_col)
metric_cols = st.columns(len(summary))
for col, (label, value) in zip(metric_cols, summary.items()):
    col.metric(label, value)

extreme_descriptions = describe_prediction_extremes(season_games, selected_pred_col, colmap)
st.caption(f"Highest prediction game: {extreme_descriptions['Highest prediction game']}")
st.caption(f"Lowest prediction game: {extreme_descriptions['Lowest prediction game']}")

calendar_tab, detail_tab, data_tab = st.tabs(["Season calendar", "Game detail", "Data explorer"])

with calendar_tab:
    st.subheader(f"{selected_season} planning calendar")
    st.write("Each calendar cell uses a continuous color scale based on the selected prediction column. Dates with multiple games stack those games inside the same day cell.")
    st.html(render_calendar_view(season_games, selected_pred_col, selected_month_numbers, colmap))

    daily_summary = (
        season_games
        .groupby("_game_date", as_index=False)
        .agg(
            games_on_day=("_row_uid", "count"),
            opponents=("_opponent", lambda s: ", ".join(s.astype(str).tolist())),
            day_avg_prediction=(selected_pred_col, "mean"),
            day_max_prediction=(selected_pred_col, "max"),
        )
        .sort_values("day_max_prediction", ascending=False)
        .rename(columns={"_game_date": "Date", "games_on_day": "Games on date", "opponents": "Opponents", "day_avg_prediction": "Day avg prediction", "day_max_prediction": "Day max prediction"})
    )
    st.markdown("**Highest-projected calendar dates**")
    st.dataframe(daily_summary.head(8), use_container_width=True, hide_index=True)

with detail_tab:
    st.subheader("Single-game planning view")

    left, right = st.columns([1.15, 1.5])
    with left:
        st.markdown(f"### vs {selected_row['_opponent']}")
        st.write(f"**Date:** {selected_row['_game_date'].strftime('%A, %B %d, %Y')}")
        formatted_start_time = format_start_time(selected_row["_start_time"])
        if formatted_start_time != "Time TBA":
            st.write(f"**Start time:** {formatted_start_time}")
        if colmap.get("dh_game_number") and pd.notna(selected_row.get(colmap["dh_game_number"], np.nan)):
            st.write(f"**Doubleheader game:** {int(float(selected_row[colmap['dh_game_number']]))}")
        if "acc" in selected_row.index:
            st.write(f"**ACC game:** {format_value(selected_row['acc'])}")
        if "rivalry" in selected_row.index:
            st.write(f"**Rivalry flag:** {format_value(selected_row['rivalry'])}")

    with right:
        recorded_attendance = selected_row["_attendance"] if pd.notna(selected_row["_attendance"]) else selected_row["_actual"]
        c1, c2 = st.columns(2)
        c1.metric("Selected prediction", format_int(selected_row[selected_pred_col]))
        c2.metric("Recorded attendance", format_int(recorded_attendance))

        if pd.notna(selected_row["_actual"]) and pd.notna(selected_row[selected_pred_col]):
            error_abs = abs(float(selected_row[selected_pred_col]) - float(selected_row["_actual"]))
            error_pct = (error_abs / float(selected_row["_actual"]) * 100) if float(selected_row["_actual"]) != 0 else np.nan
            e1, e2 = st.columns(2)
            e1.metric("Absolute error", format_int(error_abs))
            e2.metric("Percent error", format_number(error_pct) + "%" if pd.notna(error_pct) else "—")

        other_prediction_values = {}
        for other_pred_col, other_meta in model_data.items():
            if other_pred_col == selected_pred_col:
                continue
            other_df = other_meta["df"]
            match = other_df.loc[other_df["_event_id"] == selected_row["_event_id"]]
            if not match.empty:
                other_prediction_values[other_pred_col] = match.iloc[0][other_pred_col]

        if other_prediction_values:
            st.markdown("**Other prediction columns available in file**")
            other_metrics = st.columns(len(other_prediction_values))
            for col, (pred_name, pred_val) in zip(other_metrics, other_prediction_values.items()):
                col.metric(pred_name, format_int(pred_val))

    st.markdown("### Feature snapshot")
    grouped_frames = build_feature_frame(selected_row, feature_groups)
    tabs = st.tabs(list(grouped_frames.keys()))
    for tab, (group_name, feature_df) in zip(tabs, grouped_frames.items()):
        with tab:
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Data explorer")
    default_cols = [
        c for c in [
            "season_year", "date", "opponent", selected_pred_col, "actual", "attendance",
            "time", "doubleheader", "dh_game_number", "day_of_week", "weekend",
            "promo_count", "wx_d1_temperature", "wx_d1_weather_condition", "gt_web", "gt_web_preseason"
        ] if c in season_games.columns
    ]
    selected_columns = st.multiselect(
        "Columns to show",
        options=[c for c in season_games.columns if not c.startswith("_")],
        default=default_cols,
    )
    if not selected_columns:
        st.info("Select at least one column to display.")
    else:
        sort_candidates = [c for c in ["date", "time", "opponent"] if c in season_games.columns]
        sort_cols = sort_candidates if sort_candidates else selected_columns[:1]
        st.dataframe(
            season_games[selected_columns].sort_values(sort_cols),
            use_container_width=True,
            hide_index=True,
        )

