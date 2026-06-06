"""
GRIFFEY - produce site JSON from the ratings/games CSVs.

Outputs to docs/data/:
  - current_standings.json
  - teams_index.json
  - seasons_index.json
  - goat_rs.json         (top 50 end-of-regular-season ratings)
  - goat_ps.json         (top 50 end-of-postseason ratings)
  - champions.json       (per-year WS champion + runner-up with both RS and PS ratings)
  - teams/<slug>.json    (per-team season-by-season history)
  - seasons/<year>.json  (per-season snapshots + per-team standings)

MLB-specific design choices:
  - Two GOAT lists: regular season vs postseason. Reflects the fact that in baseball
    the playoff random-walk makes "best team that season" and "champion" two valid
    answers to "who was the GOAT".
  - Era-aware league assignment: Brewers AL 1969-1997, NL 1998+. Astros NL 1962-2012,
    AL 2013+. Mirrors DUNCAN's TEAM_CONFERENCE_HISTORY pattern.
  - Era-aware display names for same-market rebrands (Devil Rays -> Rays 2008,
    Indians -> Guardians 2022, Colt .45s -> Astros 1965, Florida Marlins -> Miami 2012).
"""

import json
import os
import re
from bisect import bisect_right
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Ensure docs/data subfolders exist
os.makedirs("docs/data/teams",   exist_ok=True)
os.makedirs("docs/data/seasons", exist_ok=True)


# =========================================================
# CONFIG: TEAM METADATA
# =========================================================

# Team -> league (CURRENT, i.e. modern view). Mirrors DUNCAN's TEAM_CONFERENCE pattern -
# used as the default when no season is given, e.g. for the team-index top-level field.
TEAM_LEAGUE = {
    # American League
    "Los Angeles Angels":     "AL",
    "Baltimore Orioles":      "AL",
    "Boston Red Sox":         "AL",
    "Chicago White Sox":      "AL",
    "Cleveland Guardians":    "AL",
    "Detroit Tigers":         "AL",
    "Houston Astros":         "AL",
    "Kansas City Royals":     "AL",
    "Minnesota Twins":        "AL",
    "New York Yankees":       "AL",
    "Oakland Athletics":      "AL",
    "Las Vegas Athletics":    "AL",
    "Seattle Mariners":       "AL",
    "Tampa Bay Rays":         "AL",
    "Texas Rangers":          "AL",
    "Toronto Blue Jays":      "AL",
    "Washington Senators":    "AL",   # Expansion 1961-1971 (defunct after 1971)
    # National League
    "Arizona Diamondbacks":   "NL",
    "Atlanta Braves":         "NL",
    "Chicago Cubs":           "NL",
    "Cincinnati Reds":        "NL",
    "Colorado Rockies":       "NL",
    "Los Angeles Dodgers":    "NL",
    "Miami Marlins":          "NL",
    "Milwaukee Brewers":      "NL",
    "New York Mets":          "NL",
    "Philadelphia Phillies":  "NL",
    "Pittsburgh Pirates":     "NL",
    "San Diego Padres":       "NL",
    "San Francisco Giants":   "NL",
    "St. Louis Cardinals":    "NL",
    "Washington Nationals":   "NL",
    # Historical (defunct or pre-relocation) - current era only
    "Kansas City Athletics":  "AL",
    "Milwaukee Braves":       "NL",
    "Seattle Pilots":         "AL",
    "Montreal Expos":         "NL",
}

# Era-aware league switches. Two teams have crossed leagues in our window:
#   - Brewers: AL 1969-1997 (originally Seattle Pilots), NL 1998+
#   - Astros: NL 1962-2012, AL 2013+
# Mirrors DUNCAN's TEAM_CONFERENCE_HISTORY pattern. Lookup is by season year.
TEAM_LEAGUE_HISTORY = {
    "Milwaukee Brewers": [(1970, 1997, "AL"), (1998, 9999, "NL")],
    "Houston Astros":    [(1962, 2012, "NL"), (2013, 9999, "AL")],
}

# Division assignment. MLB went from 2-division-per-league (1969-1993) to 3-division
# (1994+). Pre-1969 had no divisions (single league standings). Era-aware lookup.
# Format: team -> list of (start_year, end_year, division)
TEAM_DIVISION_HISTORY = {
    # AL East
    "Baltimore Orioles":     [(1969, 1993, "AL East"), (1994, 9999, "AL East")],
    "Boston Red Sox":        [(1969, 1993, "AL East"), (1994, 9999, "AL East")],
    "New York Yankees":      [(1969, 1993, "AL East"), (1994, 9999, "AL East")],
    "Tampa Bay Rays":        [(1998, 9999, "AL East")],
    "Toronto Blue Jays":     [(1977, 1993, "AL East"), (1994, 9999, "AL East")],
    # AL Central (formed 1994)
    "Chicago White Sox":     [(1969, 1993, "AL West"), (1994, 9999, "AL Central")],
    "Cleveland Guardians":   [(1969, 1993, "AL East"), (1994, 9999, "AL Central")],
    "Detroit Tigers":        [(1969, 1993, "AL East"), (1994, 9999, "AL Central")],
    "Kansas City Royals":    [(1969, 1993, "AL West"), (1994, 9999, "AL Central")],
    "Minnesota Twins":       [(1969, 1993, "AL West"), (1994, 9999, "AL Central")],
    # AL West
    "Houston Astros":        [(1969, 1993, "NL West"), (1994, 2012, "NL Central"), (2013, 9999, "AL West")],
    "Los Angeles Angels":    [(1969, 1993, "AL West"), (1994, 9999, "AL West")],
    "Oakland Athletics":     [(1969, 1993, "AL West"), (1994, 2024, "AL West")],
    "Las Vegas Athletics":   [(2025, 9999, "AL West")],
    "Seattle Mariners":      [(1977, 1993, "AL West"), (1994, 9999, "AL West")],
    "Texas Rangers":         [(1972, 1993, "AL West"), (1994, 9999, "AL West")],
    # NL East
    "Atlanta Braves":        [(1969, 1993, "NL West"), (1994, 9999, "NL East")],  # Braves were NL West until 1993
    "Miami Marlins":         [(1993, 9999, "NL East")],
    "New York Mets":         [(1969, 9999, "NL East")],
    "Philadelphia Phillies": [(1969, 9999, "NL East")],
    "Washington Nationals":  [(2005, 9999, "NL East")],
    # NL Central (formed 1994)
    "Chicago Cubs":          [(1969, 1993, "NL East"), (1994, 9999, "NL Central")],
    "Cincinnati Reds":       [(1969, 1993, "NL West"), (1994, 9999, "NL Central")],
    "Milwaukee Brewers":     [(1970, 1971, "AL West"), (1972, 1993, "AL East"), (1994, 1997, "AL Central"), (1998, 9999, "NL Central")],
    "Pittsburgh Pirates":    [(1969, 1993, "NL East"), (1994, 9999, "NL Central")],
    "St. Louis Cardinals":   [(1969, 1993, "NL East"), (1994, 9999, "NL Central")],
    # NL West
    "Arizona Diamondbacks":  [(1998, 9999, "NL West")],
    "Colorado Rockies":      [(1993, 9999, "NL West")],
    "Los Angeles Dodgers":   [(1969, 9999, "NL West")],
    "San Diego Padres":      [(1969, 9999, "NL West")],
    "San Francisco Giants":  [(1969, 9999, "NL West")],
    # Defunct / pre-relocation franchises within our window
    "Kansas City Athletics": [(1961, 1967, "(pre-1969 AL)")],
    "Milwaukee Braves":      [(1961, 1965, "(pre-1969 NL)")],
    "Seattle Pilots":        [(1969, 1969, "AL West")],
    "Montreal Expos":        [(1969, 1993, "NL East"), (1994, 2004, "NL East")],
    "Washington Senators":   [(1969, 1971, "AL East")],  # Expansion Senators 1961-1971; divisions only existed 1969-1971
}


# Era-aware display names. Same-market rebrands collapse to canonical current name in
# the data layer; this dict gives back the era-appropriate label for display.
GRIFFEY_TEAM_DISPLAY_HISTORY = {
    "Cleveland Guardians":  [(1961, 2021, "Cleveland Indians"), (2022, 9999, "Cleveland Guardians")],
    "Tampa Bay Rays":       [(1998, 2007, "Tampa Bay Devil Rays"), (2008, 9999, "Tampa Bay Rays")],
    "Houston Astros":       [(1962, 1964, "Houston Colt .45s"), (1965, 9999, "Houston Astros")],
    "Miami Marlins":        [(1993, 2011, "Florida Marlins"), (2012, 9999, "Miami Marlins")],
    "Las Vegas Athletics":  [(2025, 2027, "Athletics"), (2028, 9999, "Las Vegas Athletics")],
}


def display_name(canonical, season):
    """Era-appropriate display name for the given canonical team and season."""
    history = GRIFFEY_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return canonical
    s = int(season)
    for start, end, name in history:
        if start <= s <= end:
            return name
    return canonical


def current_display_name(canonical):
    history = GRIFFEY_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return canonical
    return history[-1][2]


def historical_display_names(canonical):
    """Prior display names (most recent first), excluding the current name.
    Used to render '(formerly X / Y)' hints in the Team Summary dropdown.
    Matches DUNCAN's pattern."""
    history = GRIFFEY_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return []
    current = history[-1][2]
    seen = {current}
    out = []
    for _, _, name in reversed(history[:-1]):
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def league(team, season=None):
    """Era-aware league lookup. Pass season for historical accuracy."""
    if season is not None:
        history = TEAM_LEAGUE_HISTORY.get(team)
        if history:
            s = int(season)
            for start, end, lg in history:
                if start <= s <= end:
                    return lg
    return TEAM_LEAGUE.get(team, "Other")


def division(team, season=None):
    """Era-aware division lookup. Pre-1969 has no divisions; returns the league name."""
    if season is None or season >= 2030:
        season = 2025
    s = int(season)
    if s < 1969:
        return league(team, s)  # pre-divisional era
    history = TEAM_DIVISION_HISTORY.get(team)
    if history:
        for start, end, d in history:
            if start <= s <= end:
                return d
    return "Other"


# =========================================================
# HELPERS
# =========================================================

def clean(val):
    if pd.isna(val):
        return ""
    return str(val)


# Title-odds caches are populated later by the LR block but referenced via
# _od_fields() in lookups that fire earlier in the file (e.g. pre_ws_lookup).
# Start empty so early callers harmlessly get None until the LR block fills
# them in; per-team output blocks run after the LR block, so production
# values flow through.
_title_odds_cache = {}        # (ranking_id, team) -> float
_title_odds_rank_cache = {}   # ranking_id -> {team: rank}

def _title_odds_val(ranking_id, team):
    if ranking_id is None or team is None:
        return None
    return _title_odds_cache.get((int(ranking_id), team))

def _title_odds_rk(ranking_id, team):
    if ranking_id is None or team is None:
        return None
    rm = _title_odds_rank_cache.get(int(ranking_id))
    return rm.get(team) if rm else None


def _od_fields(r):
    """Return rating_o / rating_d / rank_o / rank_d + park_factor /
    park_factor_rank + title_odds / title_odds_rank safely from a row.
    Returns None for missing values so downstream consumers render '-'
    rather than '0'. MLB labels rating_o / rating_d as BAT / PIT in the
    UI but the underlying keys stay rating_o / rating_d for fleet-wide
    UI helper compatibility. park_factor is in raw run units (positive
    = hitter's park, negative = pitcher's park). title_odds is a
    probability (0-1) that the team wins the WS, conditional on the
    snapshot's bracket state; null for teams already eliminated."""
    rid = int(r['ranking_id']) if 'ranking_id' in r and not pd.isna(r['ranking_id']) else None
    team_name = r['name'] if 'name' in r else None
    odds = _title_odds_val(rid, team_name) if rid is not None and team_name else None
    odds_rank = _title_odds_rk(rid, team_name) if rid is not None and team_name else None
    return {
        'rating_o':         round(float(r['rating_o']), 3) if 'rating_o' in r and not pd.isna(r['rating_o']) else None,
        'rating_d':         round(float(r['rating_d']), 3) if 'rating_d' in r and not pd.isna(r['rating_d']) else None,
        'rank_o':           int(r['rank_o']) if 'rank_o' in r and not pd.isna(r['rank_o']) else None,
        'rank_d':           int(r['rank_d']) if 'rank_d' in r and not pd.isna(r['rank_d']) else None,
        'park_factor':      round(float(r['park_factor']), 3) if 'park_factor' in r and not pd.isna(r['park_factor']) else None,
        'park_factor_rank': int(r['park_factor_rank']) if 'park_factor_rank' in r and not pd.isna(r['park_factor_rank']) else None,
        'title_odds':       round(float(odds), 4) if odds is not None else None,
        'title_odds_rank':  int(odds_rank) if odds_rank is not None else None,
    }


def slug(name):
    """URL-safe slug from team name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


# =========================================================
# DATA LOAD
# =========================================================

print("Reading ratings + games...")
ratings = pd.read_csv("griffey_ratings.csv.gz")
ratings["ranking_date"] = pd.to_datetime(ratings["ranking_date"])
ratings["season"] = ratings["season"].astype(int)

games = pd.read_csv("all_mlb_games.csv", low_memory=False)
games["date_game"] = pd.to_datetime(games["date_game"])
games["season"] = games["season"].astype(int)
games = games[games["season"] >= 1961].copy()
if "gametype" not in games.columns:
    games["gametype"] = "regular"
games["gametype"] = games["gametype"].fillna("regular").astype(str).str.lower()
games["is_playoff_game_flag"] = games["gametype"] != "regular"

print(f"  Ratings: {len(ratings):,} rows, {ratings['season'].min()}-{ratings['season'].max()}")
print(f"  Games:   {len(games):,} rows, {games['season'].min()}-{games['season'].max()}")


# =========================================================
# REGULAR-SEASON END DETECTION (per season)
# =========================================================
# Approach mirrors DUNCAN's mode-with-window robust detection (fixed 2026-05-22 after
# the Boston-Marathon-cancellation bug). For each season, find the most common date
# where teams complete their threshold-th game, then take the latest date within +/-2
# days of that mode. This is robust to one-off cancellations/postponements.

# Per-season RS game count (matches griffey.py)
REGULAR_SEASON_GAMES = {
    **{y: 162 for y in range(1961, 2030)},
    1972: 156, 1981: 108, 1994: 113, 1995: 144, 2020: 60,
}


def _rs_end_date(season_games_df, threshold):
    home = season_games_df[["date_game", "home_team_name"]].rename(columns={"home_team_name": "team"})
    away = season_games_df[["date_game", "visitor_team_name"]].rename(columns={"visitor_team_name": "team"})
    g = pd.concat([home, away]).sort_values("date_game")
    g["team_game_num"] = g.groupby("team").cumcount() + 1
    totals = g.groupby("team")["team_game_num"].max()
    reached = totals[totals >= threshold].index
    if not len(reached):
        return None
    thresh_dates = (
        g[(g["team_game_num"] == threshold) & (g["team"].isin(reached))]
        .groupby("team")["date_game"].first()
    )
    mode_date = thresh_dates.mode().iloc[0]
    delta = pd.Timedelta(days=2)
    in_window = thresh_dates[(thresh_dates >= mode_date - delta) & (thresh_dates <= mode_date + delta)]
    return in_window.max()


print("\nComputing per-season RS end dates...")
# Definitive RS end = the latest date of any 'regular' game in the season.
# Falls back to the mode-with-window heuristic for any season missing gametype
# attribution (shouldn't happen post-2026-05-23 but keeps the path robust for
# historical CSVs).
rs_end_by_season = {}
rs_games = games[~games["is_playoff_game_flag"]]
for s, sub in rs_games.groupby("season"):
    # Only call a season's RS "ended" once a majority of teams have hit the
    # full RS game count. Otherwise (in-progress season), the latest date is
    # just today, and we shouldn't flag today's snapshot as "End of regular
    # season". Confirmed-complete seasons either have postseason games already
    # logged or have enough teams finished.
    th = REGULAR_SEASON_GAMES.get(int(s), 162)
    home = sub[["date_game", "home_team_name"]].rename(columns={"home_team_name": "team"})
    away = sub[["date_game", "visitor_team_name"]].rename(columns={"visitor_team_name": "team"})
    g = pd.concat([home, away])
    team_totals = g.groupby("team").size()
    n_teams_done = int((team_totals >= th).sum())
    has_ps = ((games["season"] == s) & games["is_playoff_game_flag"]).any()
    if has_ps or n_teams_done >= 15:  # >= ~half of MLB
        rs_end_by_season[int(s)] = sub["date_game"].max()
for s, sub in games.groupby("season"):
    if int(s) in rs_end_by_season:
        continue
    th = REGULAR_SEASON_GAMES.get(int(s), 162)
    d = _rs_end_date(sub, th)
    if d is not None:
        rs_end_by_season[int(s)] = d
print(f"  {len(rs_end_by_season)} seasons detected.")
print("  Sample (last 5):")
for s in sorted(rs_end_by_season)[-5:]:
    print(f"    {s}: {rs_end_by_season[s].date()}")


# =========================================================
# FLAG RS-end and PS-end ON SNAPSHOTS
# =========================================================
# For each season, find:
#   - rs_end snapshot = latest snapshot with ranking_date <= rs_end_by_season[season]
#   - ps_end snapshot = latest snapshot in that season

ratings["is_rs_end"] = 0
ratings["is_ps_end"] = 0
ratings["is_playoff_snapshot"] = 0  # any snapshot AFTER rs_end

for s, rs_end in rs_end_by_season.items():
    season_mask = ratings["season"] == s
    season_subset = ratings[season_mask]
    if season_subset.empty:
        continue

    # RS-end snapshot: highest ranking_id with ranking_date <= rs_end
    rs_candidates = season_subset[season_subset["ranking_date"] <= rs_end]
    if not rs_candidates.empty:
        rs_end_id = rs_candidates["ranking_id"].max()
        ratings.loc[season_mask & (ratings["ranking_id"] == rs_end_id), "is_rs_end"] = 1

    # PS-end snapshot: latest snapshot in season
    ps_end_id = season_subset["ranking_id"].max()
    ratings.loc[season_mask & (ratings["ranking_id"] == ps_end_id), "is_ps_end"] = 1

    # Mark playoff snapshots (ranking_date > rs_end)
    ratings.loc[season_mask & (ratings["ranking_date"] > rs_end), "is_playoff_snapshot"] = 1

print(f"\nTagged snapshots:")
print(f"  RS-end markers: {(ratings['is_rs_end']==1).sum():,} rows (one per team per season)")
print(f"  PS-end markers: {(ratings['is_ps_end']==1).sum():,} rows")
print(f"  Playoff snaps:  {(ratings['is_playoff_snapshot']==1).sum():,} rows")


# =========================================================
# PER-TEAM SEASON RECORDS (RS + PS W-L from raw games)
# =========================================================
# Compute each team's regular-season W-L and postseason W-L. Games whose date <= rs_end
# count as regular season; games after count as postseason.

print("\nComputing per-team season records...")
def team_game_view(games_df):
    """One row per team per game, with W/L flag + result string from that team's
    perspective. Used both for end-of-season totals and per-snapshot rolling records."""
    home = games_df[["season", "date_game", "home_team_name", "home_win", "is_tie", "home_result", "is_playoff_game_flag"]].rename(
        columns={"home_team_name": "team", "home_win": "won", "home_result": "result"})
    away = games_df[["season", "date_game", "visitor_team_name", "visitor_win", "is_tie", "visitor_result", "is_playoff_game_flag"]].rename(
        columns={"visitor_team_name": "team", "visitor_win": "won", "visitor_result": "result"})
    return pd.concat([home, away], ignore_index=True)

team_games = team_game_view(games)
team_games["is_playoff_game"] = team_games["is_playoff_game_flag"]

records = (
    team_games.groupby(["season", "team", "is_playoff_game"])
    .agg(W=("won", "sum"), G=("won", "size"))
    .reset_index()
)
records["L"] = records["G"] - records["W"]
rec_pivot = records.pivot_table(
    index=["season", "team"],
    columns="is_playoff_game",
    values=["W", "L"],
    fill_value=0,
).reset_index()
rec_pivot.columns = ["season", "team", "rs_L", "ps_L", "rs_W", "ps_W"]
rec_pivot["rs_record"] = rec_pivot["rs_W"].astype(int).astype(str) + "-" + rec_pivot["rs_L"].astype(int).astype(str)
rec_pivot["ps_record"] = rec_pivot.apply(
    lambda r: f"{int(r['ps_W'])}-{int(r['ps_L'])}" if (r["ps_W"] + r["ps_L"]) > 0 else "",
    axis=1,
)
rec_lookup = {(int(r["season"]), r["team"]): r for _, r in rec_pivot.iterrows()}
print(f"  {len(rec_pivot):,} (season, team) records computed.")


# =========================================================
# PER-SNAPSHOT ROLLING RECORDS + LAST MATCH
# =========================================================
# For each (team, snapshot_date), compute the team's cumulative regular-season W-L,
# playoff W-L, last completed game result, and last match date. Used to populate the
# Standings tab with W-L, Last Game, Date columns (fleet convention).

print("\nComputing per-snapshot rolling records + last-match lookups...")
team_games_sorted = team_games.sort_values(["team", "season", "date_game"]).copy()
team_games_sorted["rs_w_inc"] = (~team_games_sorted["is_playoff_game"] & (team_games_sorted["won"] == 1)).astype(int)
team_games_sorted["rs_l_inc"] = (~team_games_sorted["is_playoff_game"] & (team_games_sorted["won"] == 0)).astype(int)
team_games_sorted["ps_w_inc"] = ( team_games_sorted["is_playoff_game"] & (team_games_sorted["won"] == 1)).astype(int)
team_games_sorted["ps_l_inc"] = ( team_games_sorted["is_playoff_game"] & (team_games_sorted["won"] == 0)).astype(int)
for col in ["rs_w", "rs_l", "ps_w", "ps_l"]:
    team_games_sorted[f"cum_{col}"] = (
        team_games_sorted.groupby(["team", "season"])[f"{col}_inc"].cumsum().astype(int)
    )

# We'll do an asof-merge for each team. Result: per (team, snapshot_date), the cumulative
# record as of the latest game ON OR BEFORE that date in that season.
unique_snaps = ratings[["season", "name", "ranking_date"]].drop_duplicates().copy()
unique_snaps = unique_snaps.rename(columns={"name": "team", "ranking_date": "date_game"})

# merge_asof requires both sides sorted by the `on` key globally
unique_snaps_sorted  = unique_snaps.sort_values("date_game").reset_index(drop=True)
team_games_for_merge = team_games_sorted[
    ["team", "season", "date_game", "result", "cum_rs_w", "cum_rs_l", "cum_ps_w", "cum_ps_l"]
].copy()
# Preserve the actual game date through merge_asof (the merge's join column 'date_game'
# becomes the snapshot date on the output side, so without this we'd lose the team's
# actual last-played date).
team_games_for_merge["actual_game_date"] = team_games_for_merge["date_game"]
team_games_for_merge = team_games_for_merge.sort_values("date_game").reset_index(drop=True)

snap_records = pd.merge_asof(
    unique_snaps_sorted, team_games_for_merge,
    on="date_game", by=["team", "season"],
    direction="backward",  # latest game <= snapshot date
)

snap_records["rs_record"] = (
    snap_records["cum_rs_w"].fillna(0).astype(int).astype(str)
    + "-" + snap_records["cum_rs_l"].fillna(0).astype(int).astype(str)
)
snap_records["ps_record"] = snap_records.apply(
    lambda r: f"{int(r['cum_ps_w'])}-{int(r['cum_ps_l'])}"
              if (pd.notna(r["cum_ps_w"]) and (r["cum_ps_w"] + r["cum_ps_l"]) > 0) else "",
    axis=1,
)
snap_records["last_match"]      = snap_records["result"].fillna("")
snap_records["last_match_date"] = snap_records["actual_game_date"].dt.strftime("%Y-%m-%d").fillna("")

# Index for fast lookup: (season, team, ranking_date) -> dict
snap_records["key_date"] = snap_records["date_game"]
snap_record_lookup = {
    (int(r["season"]), r["team"], r["key_date"]): {
        "rs_record":       r["rs_record"],
        "ps_record":       r["ps_record"],
        "last_match":      r["last_match"],
        "last_match_date": r["last_match_date"],
    }
    for _, r in snap_records.iterrows()
}
print(f"  {len(snap_record_lookup):,} (season, team, snapshot) lookups built.")


# =========================================================
# WORLD SERIES CHAMPION + RUNNER-UP DETECTION
# =========================================================
# Find each season's WS champion (and runner-up) by walking the postseason structurally:
# the LAST pair of teams to play each other in the postseason is the WS, and whichever
# of them won more games is the champion. Handles all eras from 1961 (no LCS) through
# current (LDS/LCS/WS) without per-format hardcoding.

print("\nDetecting World Series champion + runner-up per season...")
ws_results = {}  # season -> {'champion': team, 'runner_up': team, 'series': '4-1'}
for s, rs_end in rs_end_by_season.items():
    s = int(s)
    season_post = games[(games["season"] == s) & (games["date_game"] > rs_end)].copy()
    if season_post.empty:
        continue
    # Latest game's two teams = WS contestants (almost always - except in 2020 if WS
    # spanned multiple days, the last game is the clinching game). Use the latest pair.
    last_dt = season_post["date_game"].max()
    # All games between those two teams in the postseason = the WS games
    last_games = season_post[season_post["date_game"] == last_dt]
    if last_games.empty:
        continue
    pair = (last_games.iloc[0]["home_team_name"], last_games.iloc[0]["visitor_team_name"])
    # Now find all PS games where these two teams played each other
    ws_games = season_post[
        ((season_post["home_team_name"] == pair[0]) & (season_post["visitor_team_name"] == pair[1])) |
        ((season_post["home_team_name"] == pair[1]) & (season_post["visitor_team_name"] == pair[0]))
    ]
    a_wins = (
        ((ws_games["home_team_name"] == pair[0]) & (ws_games["home_pts"] > ws_games["visitor_pts"])) |
        ((ws_games["visitor_team_name"] == pair[0]) & (ws_games["visitor_pts"] > ws_games["home_pts"]))
    ).sum()
    b_wins = len(ws_games) - a_wins
    if a_wins >= 4:
        champ, ru = pair[0], pair[1]
        series = f"{a_wins}-{b_wins}"
    elif b_wins >= 4:
        champ, ru = pair[1], pair[0]
        series = f"{b_wins}-{a_wins}"
    else:
        # Series incomplete (e.g., 1994 strike or in-progress current year)
        continue
    # Clinching game score (from champion's POV).
    clincher = ws_games.sort_values("date_game").iloc[-1]
    if clincher["home_team_name"] == champ:
        final_score = f"{int(clincher['home_pts'])}-{int(clincher['visitor_pts'])}"
    else:
        final_score = f"{int(clincher['visitor_pts'])}-{int(clincher['home_pts'])}"
    # Game 1 of the WS = the EARLIEST date in ws_games (the LDS/LCS games
    # already filtered out by the same-pair intersection above).
    ws_g1_dt = ws_games["date_game"].min()
    ws_results[s] = {
        "champion": champ, "runner_up": ru,
        "series": series, "final_score": final_score,
        "ws_g1_date": ws_g1_dt,
    }

print(f"  WS detected: {len(ws_results)} seasons.")


# =========================================================
# PRE-WORLD-SERIES SNAPSHOT LOOKUP
# =========================================================
# For each season's WS, find the latest rating snapshot STRICTLY BEFORE
# WS Game 1 for both the champion and runner-up. Used in the Lists
# sub-view (matchup quality / closeness / blowouts / upsets) so the
# "going-in" framing isn't contaminated by the WS result itself.
# Mirrors DUNCAN's _build_pre_finals_lookup() / DILLON's week-103 pattern.

print("\nComputing pre-WS snapshots (rating + PS W-L going into Game 1)...")

# Cumulative PS W-L going into WS Game 1: walk team_games_sorted for each
# (season, team), take the cum_ps_w/cum_ps_l of the LAST PS game strictly
# before WS Game 1 (so LDS + LCS records only). If they had no PS games
# before WS (1961-1968 pre-LCS era), record is "0-0".
def _pre_ws_ps_record(team, season, g1_dt):
    sub = team_games_sorted[
        (team_games_sorted["team"] == team)
        & (team_games_sorted["season"] == season)
        & (team_games_sorted["is_playoff_game"])
        & (team_games_sorted["date_game"] < g1_dt)
    ]
    if sub.empty:
        return "0-0"
    last = sub.iloc[-1]
    return f"{int(last['cum_ps_w'])}-{int(last['cum_ps_l'])}"


_pre_ws_lookup = {}  # (team, season) -> {rating, rank, ps_record}
for s, info in ws_results.items():
    g1 = info["ws_g1_date"]
    for team in (info["champion"], info["runner_up"]):
        season_team_rows = ratings[
            (ratings["season"] == s)
            & (ratings["name"] == team)
            & (ratings["ranking_date"] < g1)
        ]
        if season_team_rows.empty:
            continue
        # Latest ranking_id strictly before WS Game 1 for this (season, team).
        pre_id = int(season_team_rows["ranking_id"].max())
        snap_rows = ratings[
            (ratings["season"] == s)
            & (ratings["name"] == team)
            & (ratings["ranking_id"] == pre_id)
        ]
        if snap_rows.empty:
            continue
        r = snap_rows.iloc[0]
        _pre_ws_lookup[(team, int(s))] = {
            "rating":    round(float(r["rating"]), 3),
            "rank":      int(r["rank"]),
            "ps_record": _pre_ws_ps_record(team, int(s), g1),
            **_od_fields(r),
        }

print(f"  Pre-WS snapshots computed for {len(set(s for (_, s) in _pre_ws_lookup))} seasons "
      f"({len(_pre_ws_lookup)} team-entries).")


def pre_ws_fields(name, season):
    """Return the pre-WS rating/rank/PS-record block, or empty if missing."""
    p = _pre_ws_lookup.get((name, int(season)))
    if p is None:
        return {}
    return {
        "rating_pre":    p["rating"],
        "rank_pre":      p["rank"],
        "rating_o_pre": p.get("rating_o"),
        "rating_d_pre": p.get("rating_d"),
        "rank_o_pre":   p.get("rank_o"),
        "rank_d_pre":   p.get("rank_d"),
        "ps_record_pre": p["ps_record"],
    }


# =========================================================
# TITLE ODDS (logistic regression, leave-one-season-out)
# =========================================================
# Continuous-progress model mirroring DUNCAN: P(WS champ | rating, rating_o,
# rating_d, progress, 3 rating x progress interactions, series_w, series_l).
# Progress is era-aware:
#   - RS: 0 -> 0.50 linear with games_played / RS_total
#   - PS round depth-in-era 1: 0.55
#   - PS round depth-in-era N (= WS): 0.95
#   - Champion: 1.00
# Series state padded to BO7-equivalent so BO3 WCS 1-0 reads as BO7 3-2.
# Trained on 1969+ (LCS era, multi-round bracket). LOO at season level.

print("\nComputing title odds (logistic regression, leave-one-season-out)...")
from scipy.optimize import minimize

TITLE_TRAIN_FROM_SEASON = 1969  # LCS era; pre-1969 was WS-only (binary coin flip).

PHASE_RS_MAX_TO        = 0.50
PHASE_POST_RS_TO       = 0.55
PHASE_FINALS_ENTRY_TO  = 0.95
PHASE_CHAMPION_TO      = 1.00

# Era-aware RS game counts (short seasons override the default 162).
SHORT_RS_GAMES_TO = {1981: 108, 1994: 113, 1995: 144, 2020: 60}
def _to_rs_games(season):
    return SHORT_RS_GAMES_TO.get(int(season), 162)

# gametype -> absolute round (1 = WC/WCS, 2 = DS, 3 = LCS, 4 = WS).
_TO_ABS_ROUND = {'wildcard': 1, 'divisionseries': 2, 'lcs': 3, 'worldseries': 4}

def _to_total_rounds(season):
    s = int(season)
    if s >= 2012 or s == 2020: return 4  # WC/WCS + DS + LCS + WS
    if s >= 1995: return 3  # DS + LCS + WS
    if s >= 1969: return 2  # LCS + WS
    return 1  # WS only

def _to_first_round(season):
    """Absolute round number of the first PS round in this era."""
    return 5 - _to_total_rounds(season)

def _to_depth_in_era(abs_round, season):
    """Era-relative depth: 1 = first PS round in era, N = WS."""
    return abs_round - _to_first_round(season) + 1

def _to_clinch_threshold(gametype, season):
    s = int(season)
    if gametype == 'worldseries':    return 4
    if gametype == 'lcs':            return 4 if s >= 1985 else 3
    if gametype == 'divisionseries': return 3
    if gametype == 'wildcard':
        if s == 2020 or s >= 2022: return 2  # BO3 WCS
        return 1                              # 2012-2019, 2021 single-game WC
    return None

def _to_pad_to_bo7(w, l, clinch):
    pad = 4 - (clinch if clinch is not None else 4)
    return w + pad, l + pad

def _to_progress(depth_in_era, total_rounds, is_champion=False):
    if is_champion:
        return PHASE_CHAMPION_TO
    if total_rounds <= 1:
        return PHASE_POST_RS_TO
    return PHASE_POST_RS_TO + (PHASE_FINALS_ENTRY_TO - PHASE_POST_RS_TO) * \
        (depth_in_era - 1) / (total_rounds - 1)

# Walk PS games -> per-(season, team) ordered series path.
ps_games = games[games['gametype'].isin(_TO_ABS_ROUND.keys())].copy()
ps_games['date_game'] = pd.to_datetime(ps_games['date_game'])

_to_team_path = {}  # (season, team) -> [{round, gametype, opp, start, end, wins, losses, decided, won_series, clinch, state}, ...]
_series_bucket = {}  # (season, team, gametype, opp) -> [(date, team_won), ...]
for _, g in ps_games.iterrows():
    s_int = int(g['season'])
    home, vis = g['home_team_name'], g['visitor_team_name']
    home_won = bool(g['home_pts'] > g['visitor_pts'])
    for team, opp, team_won in [(home, vis, home_won), (vis, home, not home_won)]:
        _series_bucket.setdefault((s_int, team, g['gametype'], opp), []).append((g['date_game'], team_won))

for (s_int, team, gametype, opp), gs in _series_bucket.items():
    gs.sort(key=lambda x: x[0])
    wins = sum(1 for _, w in gs if w)
    losses = len(gs) - wins
    clinch = _to_clinch_threshold(gametype, s_int)
    decided = max(wins, losses) >= (clinch or 99)
    won_series = decided and wins > losses
    state = []
    w = l = 0
    for d, tw in gs:
        if tw: w += 1
        else:  l += 1
        state.append((d, w, l))
    _to_team_path.setdefault((s_int, team), []).append({
        'round':       _TO_ABS_ROUND[gametype],
        'gametype':    gametype,
        'opp':         opp,
        'start':       gs[0][0],
        'end':         gs[-1][0],
        'wins':        wins,
        'losses':      losses,
        'clinch':      clinch,
        'decided':     decided,
        'won_series':  won_series,
        'state':       state,
    })
for k in _to_team_path:
    _to_team_path[k].sort(key=lambda x: x['start'])

# Champion per season: team whose last series is WS and they won it.
_to_champion = {}
for (s_int, team), path in _to_team_path.items():
    last = path[-1]
    if last['gametype'] == 'worldseries' and last['won_series']:
        _to_champion[s_int] = team

# Playoff field per season; first elimination date per team.
_to_field = {}
_to_eliminated = {}
for (s_int, team), path in _to_team_path.items():
    _to_field.setdefault(s_int, set()).add(team)
    for s in path:
        if s['decided'] and not s['won_series']:
            _to_eliminated[(s_int, team)] = s['end']
            break

# RS games-played lookup per (season, team, snap_date).
_to_rs_log = {}
games_dt = games.copy()
games_dt['date_game'] = pd.to_datetime(games_dt['date_game'])
_rs_only = games_dt[games_dt['gametype'].isin(['regular', 'playoff'])]
for _, g in _rs_only.iterrows():
    s_int = int(g['season'])
    for t in (g['home_team_name'], g['visitor_team_name']):
        _to_rs_log.setdefault((s_int, t), []).append(g['date_game'])
for k in _to_rs_log:
    _to_rs_log[k] = sorted(_to_rs_log[k])

def _to_games_played(s_int, team, snap_date):
    log = _to_rs_log.get((s_int, team), [])
    return bisect_right(log, snap_date)

def _to_snap_state(s_int, team, snap_date):
    """Return (in_field, is_eliminated, progress, series_w_pad, series_l_pad).
    in_field=False -> not a PS team this season (RS-only snapshot).
    is_eliminated=True -> after elim date (skip in LR scoring)."""
    rs_end = rs_end_by_season.get(s_int)
    total_rounds = _to_total_rounds(s_int)

    if rs_end is None or snap_date <= rs_end:
        # RS phase: progress 0 -> 0.50
        gp = _to_games_played(s_int, team, snap_date)
        progress = PHASE_RS_MAX_TO * min(gp / _to_rs_games(s_int), 1.0)
        return (True, False, progress, 0, 0)

    in_field = (s_int in _to_field) and (team in _to_field[s_int])
    if not in_field:
        return (False, False, None, 0, 0)

    # Walk team's series chronologically, evaluating each series's STATE
    # AS-OF snap_date (not end-of-season). This is critical to:
    #   1. Correctly classify a team on a clinch day as "advanced past this
    #      round" rather than "still in this round at series_w=4". Without
    #      this, the snapshot taken on the clinching game's date carried the
    #      previous round's progress + a full series_w into the LR features,
    #      which produced spurious mid-bracket flips (2001 NYY pre-WS Game 1
    #      reading as the favorite despite ARI's higher rating).
    #   2. Detect eliminations dynamically (rather than relying on the
    #      end-of-season _to_eliminated lookup), so in-progress series in the
    #      current season are scored correctly.
    path = _to_team_path[(s_int, team)]
    current = None
    last_won = None
    for s in path:
        if s['start'] > snap_date:
            break
        # Compute series state as-of snap_date.
        w_at = l_at = 0
        for d, sw, sl in s['state']:
            if d <= snap_date:
                w_at, l_at = sw, sl
            else:
                break
        clinch = s['clinch'] if s['clinch'] is not None else 99
        decided_as_of = max(w_at, l_at) >= clinch
        team_won = decided_as_of and w_at > l_at
        team_lost = decided_as_of and l_at > w_at
        if team_lost:
            return (True, True, None, 0, 0)  # eliminated
        if team_won:
            last_won = s
            continue
        # Series is in progress as-of snap_date (or hasn't started any games
        # but s.start <= snap_date, which happens only on series-start day
        # before the first game).
        current = s
        break

    if current is not None:
        w = l = 0
        for d, sw, sl in current['state']:
            if d <= snap_date:
                w, l = sw, sl
            else:
                break
        depth = _to_depth_in_era(current['round'], s_int)
        progress = _to_progress(depth, total_rounds)
        w_pad, l_pad = _to_pad_to_bo7(w, l, current['clinch'])
        return (True, False, progress, w_pad, l_pad)

    if last_won is not None:
        # Between rounds (or just clinched a round and the next hasn't
        # started): advance to the next round's progress with empty series
        # state.
        next_depth = _to_depth_in_era(last_won['round'], s_int) + 1
        if next_depth > total_rounds:
            return (True, False, PHASE_CHAMPION_TO, 0, 0)
        return (True, False, _to_progress(next_depth, total_rounds), 0, 0)

    # In field but no series started yet (after RS-end, before round-1 G1).
    return (True, False, PHASE_POST_RS_TO, 0, 0)


# Build training rows.
_to_rows = []
rated = ratings[ratings['rating_o'].notna() & ratings['rating_d'].notna()].copy()
for _, r in rated.iterrows():
    s_int = int(r['season'])
    if s_int < TITLE_TRAIN_FROM_SEASON:
        continue
    team = r['name']
    sd = r['ranking_date']
    in_field, is_elim, progress, sw, sl = _to_snap_state(s_int, team, sd)
    if is_elim or progress is None:
        continue
    _to_rows.append({
        'season':      s_int,
        'team':        team,
        'ranking_id':  int(r['ranking_id']),
        'rating':      float(r['rating']),
        'rating_o':    float(r['rating_o']),
        'rating_d':    float(r['rating_d']),
        'progress':    float(progress),
        'series_w':    int(sw),
        'series_l':    int(sl),
        'is_champion': 1 if _to_champion.get(s_int) == team else 0,
    })

_to_train_df = pd.DataFrame(_to_rows)
print(f"  Title-odds rows: {len(_to_train_df):,} "
      f"({int(_to_train_df['is_champion'].sum())} champion-positive)")


def _to_features(d):
    p = d['progress'].values
    return np.column_stack([
        d['rating'].values, d['rating_o'].values, d['rating_d'].values, p,
        d['rating'].values * p,
        d['rating_o'].values * p,
        d['rating_d'].values * p,
        d['series_w'].values,
        d['series_l'].values,
    ])


def _to_fit_logistic(X, y, reg=1e-3):
    n, k = X.shape
    Xa = np.column_stack([np.ones(n), X])
    def nll(beta):
        z = Xa @ beta
        return float(np.sum(np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - y * z)
                     + reg * np.sum(beta[1:] ** 2))
    def grad(beta):
        z = Xa @ beta
        p_hat = 1.0 / (1.0 + np.exp(-z))
        g = Xa.T @ (p_hat - y)
        g[1:] += 2 * reg * beta[1:]
        return g
    res = minimize(nll, np.zeros(k + 1), jac=grad, method='BFGS',
                   options={'maxiter': 200, 'gtol': 1e-6})
    return res.x


def _to_predict_logistic(X, beta):
    Xa = np.column_stack([np.ones(X.shape[0]), X])
    z = Xa @ beta
    return 1.0 / (1.0 + np.exp(-z))


_completed_seasons = {s for s in _to_train_df['season'].unique() if s in _to_champion}
_title_odds_cache = {}  # (ranking_id, team) -> float

for s_int in _completed_seasons:
    train = _to_train_df[_to_train_df['season'] != s_int]
    held  = _to_train_df[_to_train_df['season'] == s_int]
    if train.empty or held.empty:
        continue
    beta = _to_fit_logistic(_to_features(train), train['is_champion'].values.astype(float))
    held = held.copy()
    held['p_raw'] = _to_predict_logistic(_to_features(held), beta)
    held['p_norm'] = held.groupby('ranking_id')['p_raw'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0.0)
    for _, r in held.iterrows():
        _title_odds_cache[(int(r['ranking_id']), r['team'])] = float(r['p_norm'])

# In-progress seasons (no champion yet): train on full eligible pool.
_in_progress = _to_train_df[~_to_train_df['season'].isin(_completed_seasons)]
if not _in_progress.empty and not _to_train_df.empty:
    beta_full = _to_fit_logistic(_to_features(_to_train_df),
                                 _to_train_df['is_champion'].values.astype(float))
    cur = _in_progress.copy()
    cur['p_raw'] = _to_predict_logistic(_to_features(cur), beta_full)
    cur['p_norm'] = cur.groupby('ranking_id')['p_raw'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0.0)
    for _, r in cur.iterrows():
        _title_odds_cache[(int(r['ranking_id']), r['team'])] = float(r['p_norm'])

# Per-snapshot rank (1 = highest odds). Reuse the cache initialized earlier.
_pairs_by_rid = {}
for (rid, team), odds in _title_odds_cache.items():
    if odds is None or odds <= 0:
        continue
    _pairs_by_rid.setdefault(rid, []).append((team, odds))
for rid, pairs in _pairs_by_rid.items():
    pairs.sort(key=lambda x: -x[1])
    rank_map = {}
    prev_odds = None
    prev_rank = 0
    for i, (team, odds) in enumerate(pairs, start=1):
        if odds != prev_odds:
            prev_rank = i
            prev_odds = odds
        rank_map[team] = prev_rank
    _title_odds_rank_cache[rid] = rank_map

print(f"  Title odds cached for {len(_title_odds_cache):,} (snapshot, team) pairs "
      f"across {len(_title_odds_rank_cache):,} snapshots.")


# =========================================================
# DIVISION WINNERS (per season, per team)
# =========================================================
# Divisions exist 1969+ in MLB. Pre-1969 had single league standings (no divisions).
# Division winner = team with best regular-season W-L pct in its division. Approximates
# the official outcome closely enough for badge purposes; rare tie-breaker games not
# resolved here.

print("\nComputing division winners...")
# 1994: players' strike on Aug 12 cancelled the rest of the season and the
# postseason. MLB officially recognized no division titles, no pennants, no WS.
NO_TITLES_SEASONS = {1994}
# Only award division titles for COMPLETED seasons. rs_end_by_season is keyed
# on seasons whose regular season has actually wrapped (>=15 teams hit 162 OR
# postseason games are logged) per the gate we added 2026-05-23.
COMPLETED_SEASONS = set(rs_end_by_season.keys())


def _resolve_division_tie(season, tied_teams, rs_games_for_season):
    """MLB tiebreaker cascade: head-to-head, then intra-division pct, then
    league-wide pct, then alphabetical fallback. Operates on regular-season
    games only. Returns the winning team name."""
    rs = rs_games_for_season
    teams_set = set(tied_teams)
    # 1. Head-to-head pct between the tied teams
    h2h = rs[
        rs["home_team_name"].isin(teams_set) & rs["visitor_team_name"].isin(teams_set)
    ]
    if not h2h.empty:
        h2h_recs = {}
        for t in tied_teams:
            tg = h2h[(h2h["home_team_name"] == t) | (h2h["visitor_team_name"] == t)]
            w = int(((tg["home_team_name"] == t) & (tg["home_win"] == 1)).sum()
                  + ((tg["visitor_team_name"] == t) & (tg["visitor_win"] == 1)).sum())
            g_count = len(tg)
            h2h_recs[t] = (w / g_count) if g_count else 0.0
        best = max(h2h_recs.values())
        winners = [t for t in tied_teams if h2h_recs[t] == best]
        if len(winners) == 1:
            return winners[0]
        tied_teams = winners
    # 2. Intra-division pct
    div_name = division(tied_teams[0], season)
    div_teams = {
        t for t in rs["home_team_name"].unique()
        if division(t, season) == div_name
    } | {
        t for t in rs["visitor_team_name"].unique()
        if division(t, season) == div_name
    }
    intra = rs[
        rs["home_team_name"].isin(div_teams) & rs["visitor_team_name"].isin(div_teams)
    ]
    if not intra.empty:
        intra_recs = {}
        for t in tied_teams:
            tg = intra[(intra["home_team_name"] == t) | (intra["visitor_team_name"] == t)]
            w = int(((tg["home_team_name"] == t) & (tg["home_win"] == 1)).sum()
                  + ((tg["visitor_team_name"] == t) & (tg["visitor_win"] == 1)).sum())
            g_count = len(tg)
            intra_recs[t] = (w / g_count) if g_count else 0.0
        best = max(intra_recs.values())
        winners = [t for t in tied_teams if intra_recs[t] == best]
        if len(winners) == 1:
            return winners[0]
        tied_teams = winners
    # 3. League-wide pct already used to flag the tie; if still tied, alphabetize
    return sorted(tied_teams)[0]


division_winners = set()  # set of (season, team) tuples
rs_only = team_games[~team_games["is_playoff_game"]].copy()
rs_games_only = games[~games["is_playoff_game_flag"]].copy()
for s, sub in rs_only.groupby("season"):
    s = int(s)
    if s < 1969 or s in NO_TITLES_SEASONS or s not in COMPLETED_SEASONS:
        continue
    by_team = sub.groupby("team").agg(W=("won", "sum"), G=("won", "size")).reset_index()
    by_team["L"]   = by_team["G"] - by_team["W"]
    by_team["pct"] = by_team["W"] / by_team["G"]
    by_team["div"] = by_team["team"].apply(lambda t: division(t, s))
    by_team = by_team[~by_team["div"].isin({"Other", "(pre-1969 AL)", "(pre-1969 NL)"})]
    if by_team.empty:
        continue
    rs_for_season = rs_games_only[rs_games_only["season"] == s]
    for div_name, div_sub in by_team.groupby("div"):
        top_pct = div_sub["pct"].max()
        tied_at_top = div_sub[div_sub["pct"] == top_pct]["team"].tolist()
        winner = (tied_at_top[0] if len(tied_at_top) == 1
                  else _resolve_division_tie(s, tied_at_top, rs_for_season))
        division_winners.add((s, winner))
print(f"  {len(division_winners)} division titles flagged.")
print(f"  Last 5: {[(s, ws_results[s]['champion'], ws_results[s]['series']) for s in sorted(ws_results)[-5:]]}")


# =========================================================
# OUTPUT: current_standings.json
# =========================================================
print("\nWriting current_standings.json...")
latest_id = int(ratings["ranking_id"].max())
latest = ratings[ratings["ranking_id"] == latest_id].sort_values("rating", ascending=False).copy()
latest_season = int(latest["season"].iloc[0])
latest_date   = str(latest["ranking_date"].iloc[0].date())

standings_data = {
    "updated": latest_date,
    "season":  latest_season,
    "teams": [
        {
            "rank":          int(r["rank"]),
            "team":          r["name"],
            "display_name":  display_name(r["name"], latest_season),
            "league":        league(r["name"], latest_season),
            "division":      division(r["name"], latest_season),
            "rating":        round(float(r["rating"]), 3),
            **_od_fields(r),
        }
        for _, r in latest.iterrows()
    ],
}
with open("docs/data/current_standings.json", "w") as f:
    json.dump(standings_data, f, separators=(",", ":"))


# =========================================================
# OUTPUT: per-season JSON files (snapshots + standings)
# =========================================================
print("Writing per-season JSON files...")

# Teams that actually played at least one game in each season. Filters out
# ghost snapshots - e.g., "Oakland Athletics" still has a rolling-window
# rating in early 2025 from their 2024 games, but they didn't exist as a
# franchise in 2025.
teams_played_by_season = {}
for s_, sub in games.groupby("season"):
    teams_played_by_season[int(s_)] = set(sub["home_team_name"]) | set(sub["visitor_team_name"])

for s in sorted(ratings["season"].unique()):
    s = int(s)
    sdf = ratings[ratings["season"] == s].copy()
    ws_info = ws_results.get(s, {})
    ws_champ = ws_info.get("champion")
    ws_ru    = ws_info.get("runner_up")
    snapshots = []
    played_this_season = teams_played_by_season.get(s, set())
    for rid, rdf in sdf.groupby("ranking_id"):
        rdf = rdf[rdf["name"].isin(played_this_season)].copy() if played_this_season else rdf
        rdf = rdf.sort_values("rank")
        snap_date_ts = rdf["ranking_date"].iloc[0]
        snap_date = str(snap_date_ts.date())
        is_rs_end = int(rdf["is_rs_end"].iloc[0])
        is_ps_end = int(rdf["is_ps_end"].iloc[0])
        is_playoff_snap = int(rdf["is_playoff_snapshot"].iloc[0])
        label = None
        if is_rs_end:
            label = "End of regular season"
        elif is_ps_end and is_playoff_snap:
            label = "End of playoffs"
        teams_snap = []
        for _, r in rdf.iterrows():
            rec = snap_record_lookup.get((s, r["name"], snap_date_ts), {})
            finals_status = 2 if r["name"] == ws_champ else (1 if r["name"] == ws_ru else 0)
            div_winner = 1 if (s, r["name"]) in division_winners else 0
            teams_snap.append({
                "rank":             int(r["rank"]) if not pd.isna(r["rank"]) else None,
                "team":             r["name"],
                "display_name":     display_name(r["name"], s),
                "league":           league(r["name"], s),
                "division":         division(r["name"], s),
                "rating":           round(float(r["rating"]), 3),
                **_od_fields(r),
                "regular_record":   rec.get("rs_record", "0-0"),
                "playoff_record":   rec.get("ps_record", ""),
                "last_match":       rec.get("last_match", ""),
                "last_match_date":  rec.get("last_match_date", ""),
                "finals_status":    finals_status,
                "division_winner":  div_winner,
            })
        snapshots.append({
            "date":            snap_date,
            "label":           label,
            "is_rs_end":       is_rs_end,
            "is_ps_end":       is_ps_end,
            "is_playoff_snap": is_playoff_snap,
            "teams":           teams_snap,
        })

    snapshots.sort(key=lambda x: x["date"])
    with open(f"docs/data/seasons/{s}.json", "w") as f:
        json.dump({"season": s, "snapshots": snapshots}, f, separators=(",", ":"))


# =========================================================
# OUTPUT: per-team JSON files (season summary + per-snapshot history)
# =========================================================
print("Writing per-team JSON files...")

teams_index = []
for team in sorted(ratings["name"].unique()):
    tdf = ratings[ratings["name"] == team].copy()
    team_slug = slug(team)
    seasons_dict = {}
    for s, sub in tdf.groupby("season"):
        s = int(s)
        if team not in teams_played_by_season.get(s, set()):
            continue  # ghost season (rolling-window only, team didn't play)
        ws_info = ws_results.get(s, {})
        finals_status = 2 if ws_info.get("champion") == team else (1 if ws_info.get("runner_up") == team else 0)
        # Per-snapshot entries (matches DUNCAN/DILLON Team Summary structure exactly:
        # one row per snapshot, with rolling record + last match + season_flag).
        # season_flag follows DUNCAN/DILLON convention: 0 = mid-season, 1 = end of
        # regular season, 2 = end of postseason.
        entries = []
        div_winner = 1 if (s, team) in division_winners else 0
        for _, r in sub.sort_values("ranking_date").iterrows():
            snap_date_ts = r["ranking_date"]
            rec = snap_record_lookup.get((s, team, snap_date_ts), {})
            sf = 2 if int(r["is_ps_end"]) == 1 else (1 if int(r["is_rs_end"]) == 1 else 0)
            entries.append({
                "date":             str(snap_date_ts.date()),
                "rank":             int(r["rank"]) if not pd.isna(r["rank"]) else None,
                "rating":           round(float(r["rating"]), 3),
                **_od_fields(r),
                "display_name":     display_name(team, s),
                "league":           league(team, s),
                "division":        division(team, s),
                "regular_record":   rec.get("rs_record", "0-0"),
                "playoff_record":   rec.get("ps_record", ""),
                "last_match":       rec.get("last_match", ""),
                "last_match_date":  rec.get("last_match_date", ""),
                "finals_status":    finals_status,
                "division_winner":  div_winner,
                "season_flag":      sf,
            })
        seasons_dict[str(s)] = entries
    with open(f"docs/data/teams/{team_slug}.json", "w") as f:
        json.dump({
            "team":         team,
            "display_name": current_display_name(team),
            "league":       league(team, ratings["season"].max()),
            "seasons":      seasons_dict,
        }, f, separators=(",", ":"))

    teams_index.append({
        "team":              team,
        "display_name":      current_display_name(team),
        "league":            league(team, ratings["season"].max()),
        "historical_names":  historical_display_names(team),
        "slug":              team_slug,
    })

teams_index.sort(key=lambda x: x["team"])
with open("docs/data/teams_index.json", "w") as f:
    json.dump(teams_index, f, separators=(",", ":"))


# =========================================================
# OUTPUT: champions.json
# =========================================================
print("Writing champions.json...")
champs_list = []
for s in sorted(ws_results.keys(), reverse=True):
    info = ws_results[s]
    ch_rs = ratings[(ratings["season"]==s) & (ratings["name"]==info["champion"]) & (ratings["is_rs_end"]==1)]
    ch_ps = ratings[(ratings["season"]==s) & (ratings["name"]==info["champion"]) & (ratings["is_ps_end"]==1)]
    ru_rs = ratings[(ratings["season"]==s) & (ratings["name"]==info["runner_up"]) & (ratings["is_rs_end"]==1)]
    ru_ps = ratings[(ratings["season"]==s) & (ratings["name"]==info["runner_up"]) & (ratings["is_ps_end"]==1)]
    rec_ch = rec_lookup.get((s, info["champion"]))
    rec_ru = rec_lookup.get((s, info["runner_up"]))
    pre_rated = ch_rs.empty and ch_ps.empty  # season was consumed by warm-up window
    champs_list.append({
        "season":      s,
        "series":      info["series"],
        "final_score": info.get("final_score", ""),
        "pre_rated":   pre_rated,
        "champion": {
            "team":         info["champion"],
            "display_name": display_name(info["champion"], s),
            "league":       league(info["champion"], s),
            "rs_record":    rec_ch["rs_record"] if rec_ch is not None else "",
            "ps_record":    rec_ch["ps_record"] if rec_ch is not None else "",
            "rs_end_rating":  round(float(ch_rs["rating"].iloc[0]), 3) if not ch_rs.empty else None,
            "rs_end_rank":    int(ch_rs["rank"].iloc[0]) if not ch_rs.empty else None,
            **({f"rs_end_{k}": v for k, v in _od_fields(ch_rs.iloc[0]).items()} if not ch_rs.empty else {}),
            "ps_end_rating":  round(float(ch_ps["rating"].iloc[0]), 3) if not ch_ps.empty else None,
            "ps_end_rank":    int(ch_ps["rank"].iloc[0]) if not ch_ps.empty else None,
            **({f"ps_end_{k}": v for k, v in _od_fields(ch_ps.iloc[0]).items()} if not ch_ps.empty else {}),
            **pre_ws_fields(info["champion"], s),
        },
        "runner_up": {
            "team":         info["runner_up"],
            "display_name": display_name(info["runner_up"], s),
            "league":       league(info["runner_up"], s),
            "rs_record":    rec_ru["rs_record"] if rec_ru is not None else "",
            "ps_record":    rec_ru["ps_record"] if rec_ru is not None else "",
            "rs_end_rating":  round(float(ru_rs["rating"].iloc[0]), 3) if not ru_rs.empty else None,
            "rs_end_rank":    int(ru_rs["rank"].iloc[0]) if not ru_rs.empty else None,
            **({f"rs_end_{k}": v for k, v in _od_fields(ru_rs.iloc[0]).items()} if not ru_rs.empty else {}),
            "ps_end_rating":  round(float(ru_ps["rating"].iloc[0]), 3) if not ru_ps.empty else None,
            "ps_end_rank":    int(ru_ps["rank"].iloc[0]) if not ru_ps.empty else None,
            **({f"ps_end_{k}": v for k, v in _od_fields(ru_ps.iloc[0]).items()} if not ru_ps.empty else {}),
            **pre_ws_fields(info["runner_up"], s),
        },
    })

# Insert a synthetic 1994 entry - players' strike on Aug 12 cancelled the
# postseason and no World Series was played. ws_results omits 1994 since the
# WS-walker can't find a champion, so the season would otherwise vanish from
# the Champions tab. Flag it as `no_series` so the UI can render a callout.
if not any(e["season"] == 1994 for e in champs_list):
    champs_list.append({
        "season":    1994,
        "series":    "",
        "no_series": True,
        "pre_rated": False,
        "champion":  None,
        "runner_up": None,
    })
    champs_list.sort(key=lambda e: -e["season"])

# Pre-rated World Series titles (1903-1960).
# Our rating data starts at season 1961 (expansion era anchor), so WS from 1903-1960
# aren't otherwise captured. Counts only titles won under the franchise's CURRENT
# city/name per fleet relocation policy: e.g., the Brooklyn Dodgers' 1955 title is
# NOT carried into the LA Dodgers count; only LA-era titles (1959 here) carry.
# Defunct/relocated franchises (Brooklyn Dodgers, NY Giants, Boston Braves,
# Philadelphia Athletics, original Washington Senators, St. Louis Browns) are
# omitted entirely since they're not in our data window.
PRE_1961_CHAMPIONSHIPS = {
    "Boston Red Sox":         5,   # 1903, 1912, 1915, 1916, 1918
    "Chicago Cubs":           2,   # 1907, 1908
    "Chicago White Sox":      2,   # 1906, 1917
    "Cincinnati Reds":        2,   # 1919, 1940
    "Cleveland Guardians":    2,   # 1920, 1948 (as Indians)
    "Detroit Tigers":         2,   # 1935, 1945
    "Los Angeles Dodgers":    1,   # 1959 (LA era only; Brooklyn era not carried)
    "Milwaukee Braves":       1,   # 1957 (Milwaukee era; defunct franchise in our window)
    "New York Yankees":      18,   # 1923, 1927, 1928, 1932, 1936, 1937, 1938, 1939, 1941, 1943, 1947, 1949, 1950, 1951, 1952, 1953, 1956, 1958
    "Pittsburgh Pirates":     3,   # 1909, 1925, 1960
    "St. Louis Cardinals":    6,   # 1926, 1931, 1934, 1942, 1944, 1946
}

PRE_1961_RUNNER_UPS = {
    "Boston Red Sox":         1,   # 1946
    "Chicago Cubs":           8,   # 1906, 1910, 1918, 1929, 1932, 1935, 1938, 1945
    "Chicago White Sox":      2,   # 1919, 1959
    "Cincinnati Reds":        1,   # 1939
    "Cleveland Guardians":    1,   # 1954 (as Indians)
    "Detroit Tigers":         5,   # 1907, 1908, 1909, 1934, 1940
    "Milwaukee Braves":       1,   # 1958
    "New York Yankees":       7,   # 1921, 1922, 1926, 1942, 1955, 1957, 1960
    "Philadelphia Phillies":  2,   # 1915, 1950
    "Pittsburgh Pirates":     2,   # 1903, 1927
    "St. Louis Cardinals":    3,   # 1928, 1930, 1943
}

# Running counts: walk chronologically (oldest first), seeded with pre-1961 totals.
# Matches DUNCAN/DILLON pattern exactly: each rated entry gets title_count and
# runner_up_count attached for display as "(N 👑)" / "(N 🥈)" badges in the UI.
_champ_count = dict(PRE_1961_CHAMPIONSHIPS)
_ru_count    = dict(PRE_1961_RUNNER_UPS)
for entry in reversed(champs_list):  # champs_list is newest-first; walk oldest-first
    if entry.get("no_series"):
        continue  # 1994 strike - no champion/runner-up to count
    ct = entry["champion"]["team"]
    rt = entry["runner_up"]["team"]
    _champ_count[ct] = _champ_count.get(ct, 0) + 1
    _ru_count[rt]    = _ru_count.get(rt, 0) + 1
    entry["champion"]["title_count"]      = _champ_count[ct]
    entry["runner_up"]["runner_up_count"] = _ru_count[rt]

with open("docs/data/champions.json", "w") as f:
    json.dump({"MLB": champs_list}, f, separators=(",", ":"))


# =========================================================
# OUTPUT: goat_rs.json and goat_ps.json (top 50 each)
# =========================================================
# Filter to "completed" seasons only (need a verified WS to know champion status,
# and need stable end-of-RS rating). Also exclude in-progress current year.
print("Writing GOAT lists (RS + PS)...")

GOAT_TOP_N = 50
ws_seasons = set(ws_results.keys())

# Short / disrupted seasons - flagged on GOAT/Champions/Standings/TeamSummary
# rows so the UI can tag them inline + footnote. Categories drive colour
# (labor = amber, covid = yellow, cancelled = red).
SHORT_SEASONS = {
    1981: {
        'tag': 'strike split',
        'category': 'labor',
        'note': "The 1981 regular season was interrupted by a players' strike from June 12 to August 9 (~50 days lost). The season was played in a split format with separate first-half and second-half division winners.",
    },
    1994: {
        'tag': 'strike 113g',
        'category': 'labor',
        'note': "A players' strike on August 12, 1994 ended the season early at ~113 games per team and cancelled the entire postseason. No division titles, pennants, or World Series.",
    },
    1995: {
        'tag': 'strike-delayed 144g',
        'category': 'labor',
        'note': "The 1995 season was shortened to 144 games; spring training and the start of the season were delayed by the residual 1994-95 strike.",
    },
    2020: {
        'tag': 'COVID 60g',
        'category': 'covid',
        'note': "The 2020 season was shortened to 60 games with regional-only play; postseason expanded to 16 teams and played in a bubble.",
    },
}


def build_goat(flag_col, require_ws=False, sort_col="rating"):
    rows = ratings[(ratings[flag_col] == 1) & (ratings["season"].isin(ws_seasons))].copy()

    # Filter ghost snapshots (team in window but didn't actually play that season).
    rows = rows[rows.apply(
        lambda r: r["name"] in teams_played_by_season.get(int(r["season"]), set()), axis=1
    )]

    # For GOAT-PS, require the team to have actually played in the World Series.
    # Matches the fleet convention (DUNCAN top 10 = NBA Finals teams, DILLON = SB teams,
    # LOBO = WNBA Finals teams). Also filters out the post-merge PS-end inflation
    # problem for non-WS teams.
    if require_ws:
        ws_teams = {(s, info["champion"]) for s, info in ws_results.items()} | \
                   {(s, info["runner_up"]) for s, info in ws_results.items()}
        rows = rows[rows.apply(lambda r: (int(r["season"]), r["name"]) in ws_teams, axis=1)]

    rows = rows[rows[sort_col].notna()]
    rows = rows.sort_values(sort_col, ascending=False).reset_index(drop=True)
    top = rows.head(GOAT_TOP_N)
    out = []
    for i, r in top.iterrows():
        s = int(r["season"])
        rec = rec_lookup.get((s, r["name"]))
        info = ws_results.get(s, {})
        finals_status = 2 if info.get("champion") == r["name"] else (1 if info.get("runner_up") == r["name"] else 0)
        out.append({
            "rank":         i + 1,
            "team":         r["name"],
            "display_name": display_name(r["name"], s),
            "league":       league(r["name"], s),
            "division":     division(r["name"], s),
            "division_winner": 1 if (s, r["name"]) in division_winners else 0,
            "season":       s,
            "short_season":          s in SHORT_SEASONS,
            "short_season_tag":      SHORT_SEASONS.get(s, {}).get("tag", "")      if s in SHORT_SEASONS else "",
            "short_season_category": SHORT_SEASONS.get(s, {}).get("category", "") if s in SHORT_SEASONS else "",
            "short_season_note":     SHORT_SEASONS.get(s, {}).get("note", "")     if s in SHORT_SEASONS else "",
            "rating":       round(float(r["rating"]), 3),
            **_od_fields(r),
            "regular_record": rec["rs_record"] if rec is not None else "",
            "playoff_record": rec["ps_record"] if rec is not None else "",
            "finals_status":  finals_status,  # 0=neither, 1=runner-up, 2=champion
        })
    return out

# Six GOAT files: {Rating, BAT, PIT} × {RS-end, PS-end}. PS-end variants
# require a WS appearance (champion or runner-up). Mirrors DUNCAN/LOBO.
goat_files = [
    ("goat_rs.json",   "is_rs_end", False, "rating"),
    ("goat_ps.json",   "is_ps_end", True,  "rating"),
    ("goat_rs_o.json", "is_rs_end", False, "rating_o"),
    ("goat_rs_d.json", "is_rs_end", False, "rating_d"),
    ("goat_ps_o.json", "is_ps_end", True,  "rating_o"),
    ("goat_ps_d.json", "is_ps_end", True,  "rating_d"),
]
for fname, flag_col, require_ws, sort_col in goat_files:
    payload = build_goat(flag_col=flag_col, require_ws=require_ws, sort_col=sort_col)
    with open(f"docs/data/{fname}", "w") as f:
        json.dump(payload, f, separators=(",", ":"))


# =========================================================
# OUTPUT: seasons_index.json
# =========================================================
seasons_index = sorted({int(s) for s in ratings["season"].unique()}, reverse=True)
with open("docs/data/seasons_index.json", "w") as f:
    json.dump({
        "seasons":      seasons_index,
        "first_date":   str(games["date_game"].min().date()),
        "last_date":    str(games["date_game"].max().date()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disrupted_seasons": {
            str(year): {"tag": info["tag"], "category": info["category"], "note": info["note"]}
            for year, info in SHORT_SEASONS.items()
        },
    }, f, separators=(",", ":"))


# Reload the canonical Rating-sorted GOAT lists for the diagnostic print.
_goat_rs_top = json.load(open("docs/data/goat_rs.json"))
_goat_ps_top = json.load(open("docs/data/goat_ps.json"))
print(f"\nDone. {len(teams_index)} teams, {len(seasons_index)} seasons, "
      f"{len(ws_results)} WS detected, {len(_goat_rs_top)} GOAT-RS, {len(_goat_ps_top)} GOAT-PS.")
print(f"\n=== GOAT-RS top 10 ===")
for t in _goat_rs_top[:10]:
    marker = " ★" if t["finals_status"] == 2 else " (RU)" if t["finals_status"] == 1 else ""
    print(f"  #{t['rank']:2d}. {t['display_name']} {t['season']}  ({t['regular_record']}, {t['rating']:+.2f}){marker}")
print(f"\n=== GOAT-PS top 10 ===")
for t in _goat_ps_top[:10]:
    marker = " ★" if t["finals_status"] == 2 else " (RU)" if t["finals_status"] == 1 else ""
    print(f"  #{t['rank']:2d}. {t['display_name']} {t['season']}  ({t['regular_record']}, {t['rating']:+.2f}){marker}")
