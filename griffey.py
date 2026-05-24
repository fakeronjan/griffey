"""
GRIFFEY - MLB power ratings via WLS Massey solver.

Named after Ken Griffey Jr., Hall-of-Famer who played for Cincinnati 2000-2008.

Model stack mirrors the WLS-template fleet (DUNCAN/LOBO/DILLON/SALAAM):
  - Homebrew WLS Massey solver (no rankit dependency)
  - Margin cap to suppress blowouts
  - Per-game HCA on raw run margin
  - Season-aware rolling window (game-days)
  - Linear recency decay across the window

Data source: Retrosheet gameinfo.csv (https://www.retrosheet.org/downloads/gameinfo.zip).
Covers every MLB game from 1898 to present, updated annually.
"""

import io
import json
import os
import zipfile
from datetime import datetime
from urllib.request import urlopen

import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION
# =========================================================

MIN_SEASON = 1961             # Expansion era anchor (162-game season + AL/NL post-expansion)

# Season-aware rolling window: window (game-days) = WINDOW_MULTIPLIER * games-per-team-per-season.
# At 0.75, a 162-game MLB season gets a 121-day window (~67% of season). Shortened seasons
# (lockouts, COVID) get proportionally smaller windows automatically.
WINDOW_MULTIPLIER = 1.25

HOME_COURT_ADJUSTMENT = 0.1   # raw-run home advantage (modern-era empirical: ~0.09-0.13 runs)

# Margin transform: cap at 10 runs. Captures ~98% of MLB games uncapped - only the most
# extreme blowouts (often garbage-time bullpen scenarios) get trimmed.
MARGIN_TRANSFORM = "cap"
MARGIN_CAP = 8

# WLS observation weights: recency × tier × match-type. MLB has no tier weights (all games
# are full competitive games; no friendlies). Just recency decay across the window.
WEIGHTING_MODE = "wls"

# Re-process the most recent N game-days on every run so late-arriving data is absorbed.
# MLB games are usually finalized within a few hours, but September call-ups + spring
# training + postseason can push edits.
RECOMPUTE_TAIL_DAYS = 7

# Regular-season games per team per season. MLB has been 162 since 1961-1962, with these
# documented shortenings:
#   1972: ~155-156 (player strike)
#   1981: ~108 (mid-season strike, split season)
#   1994: ~113 (August strike, no postseason)
#   1995: 144 (continued strike, delayed start)
#   2020: 60 (COVID)
REGULAR_SEASON_GAMES = {
    **{y: 162 for y in range(1961, 2030)},
    1972: 156,
    1981: 108,
    1994: 113,
    1995: 144,
    2020: 60,
}

# Retrosheet download URL (single zip with one CSV inside). Annually updated; we use
# it as the historical backbone (1961 through end of last completed season).
RETROSHEET_URL = "https://retrosheet.org/downloads/gameinfo.zip"

# MLB Stats API endpoint. Used as a "tail fetcher" for current-year in-season games
# that Retrosheet hasn't published yet. Free, no auth, official MLB data.
STATSAPI_URL = "https://statsapi.mlb.com/api/v1/schedule"

LOADED_GAMES_CSV = "loaded_mlb_games.csv"
ALL_GAMES_CSV    = "all_mlb_games.csv"

# Retrosheet uses 3-letter team codes. Map to canonical full names. Franchises that
# relocated keep separate identities (per fleet policy: move = lose history).
# Same-market rebrands collapse to canonical current name.
RETROSHEET_TEAM = {
    # American League
    "ANA": "Los Angeles Angels",     # Anaheim Angels 1997-2004 (same-market with LA Angels)
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CAL": "Los Angeles Angels",     # California Angels 1966-1996 (same-market)
    "CHA": "Chicago White Sox",
    "CLE": "Cleveland Guardians",    # Same-market rebrand: Indians 1915-2021 -> Guardians 2022+
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",         # AL since 2013; Colt .45s 1962-64 -> Astros 1965+ (same-market)
    "KCA": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "MIN": "Minnesota Twins",
    "NYA": "New York Yankees",
    "OAK": "Oakland Athletics",      # 1968-2024 (separate franchise; relocated)
    "ATH": "Las Vegas Athletics",    # 2025+ (in Sacramento transitionally pending Vegas park)
    "SEA": "Seattle Mariners",
    "TBA": "Tampa Bay Rays",         # Devil Rays 1998-2007 -> Rays 2008+ (same-market)
    "TEX": "Texas Rangers",          # 1972+ (separate franchise from expansion Senators)
    "TOR": "Toronto Blue Jays",
    "WS2": "Washington Senators",    # Expansion 1961-1971 (separate franchise per fleet 'move = lose history' policy)
    # National League
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "CHN": "Chicago Cubs",
    "CIN": "Cincinnati Reds",
    "COL": "Colorado Rockies",
    "FLO": "Miami Marlins",          # Florida Marlins 1993-2011 -> Miami Marlins (same-market)
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",      # NL since 1998
    "LAN": "Los Angeles Dodgers",
    "MON": "Montreal Expos",         # 1969-2004; relocated to DC, kept SEPARATE per fleet policy
    "NYN": "New York Mets",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDN": "San Diego Padres",
    "SFN": "San Francisco Giants",
    "SLN": "St. Louis Cardinals",
    "WAS": "Washington Nationals",
    # Historical franchises within our 1961+ window (kept separate per fleet relocation policy)
    "KC1": "Kansas City Athletics",  # Athletics 1955-1967 (post-Philadelphia, pre-Oakland)
    "MLN": "Milwaukee Braves",       # Braves 1953-1965 (post-Boston, pre-Atlanta)
    "SE1": "Seattle Pilots",         # 1969 only, became Brewers in 1970
}

# Confederation/division mapping: AL = American League, NL = National League. Era-aware
# since some teams switched (Brewers NL since 1998, Astros AL since 2013).
TEAM_LEAGUE = {
    # AL teams (always AL)
    "Los Angeles Angels": "AL", "Baltimore Orioles": "AL", "Boston Red Sox": "AL",
    "Chicago White Sox": "AL", "Cleveland Guardians": "AL", "Detroit Tigers": "AL",
    "Kansas City Royals": "AL", "Minnesota Twins": "AL", "New York Yankees": "AL",
    "Oakland Athletics": "AL", "Las Vegas Athletics": "AL",
    "Seattle Mariners": "AL", "Tampa Bay Rays": "AL",
    "Texas Rangers": "AL", "Toronto Blue Jays": "AL",
    "Washington Senators": "AL",   # Expansion 1961-1971 (defunct after 1971; separate from Rangers)
    # NL teams (always NL)
    "Arizona Diamondbacks": "NL", "Atlanta Braves": "NL", "Chicago Cubs": "NL",
    "Cincinnati Reds": "NL", "Colorado Rockies": "NL", "Miami Marlins": "NL",
    "Los Angeles Dodgers": "NL", "New York Mets": "NL", "Philadelphia Phillies": "NL",
    "Pittsburgh Pirates": "NL", "San Diego Padres": "NL", "San Francisco Giants": "NL",
    "St. Louis Cardinals": "NL", "Washington Nationals": "NL",
    # Switchers - Brewers AL 1969-1997 then NL; Astros NL 1962-2012 then AL
    # This dict gives the CURRENT-era league. Era-aware lookups (used by the
    # WLS solver's cross-league mean anchor) consult TEAM_LEAGUE_HISTORY below.
    "Milwaukee Brewers": "NL",
    "Houston Astros": "AL",
    # Historical franchises in our window (kept separate from successors per relocation policy)
    "Kansas City Athletics": "AL",
    "Milwaukee Braves": "NL",
    "Seattle Pilots": "AL",
    "Montreal Expos": "NL",
}

# Era-aware league overrides for switchers. Lookup by season. Lookups for any team
# not in this dict fall back to TEAM_LEAGUE (the modern-era default).
TEAM_LEAGUE_HISTORY = {
    "Milwaukee Brewers": [(1970, 1997, "AL"), (1998, 9999, "NL")],
    "Houston Astros":    [(1962, 2012, "NL"), (2013, 9999, "AL")],
}


def _team_league(team, season):
    """Era-aware league lookup. Returns 'AL' or 'NL' (or 'Other' if unknown)."""
    history = TEAM_LEAGUE_HISTORY.get(team)
    if history:
        s = int(season)
        for start, end, lg in history:
            if start <= s <= end:
                return lg
    return TEAM_LEAGUE.get(team, "Other")


# =========================================================
# DATA ACQUISITION
# =========================================================

def fetch_retrosheet():
    """Download the Retrosheet gameinfo zip and return a DataFrame of all games."""
    print(f"Fetching Retrosheet game info from {RETROSHEET_URL}...")
    with urlopen(RETROSHEET_URL) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open("gameinfo.csv") as f:
            df = pd.read_csv(f, low_memory=False)
    print(f"  Fetched {len(df):,} rows.")
    return df


def _name_to_retrosheet_code():
    """Build a canonical-name -> CURRENT Retrosheet code map.
    RETROSHEET_TEAM maps multiple historical codes (CAL/ANA/LAA) to the same canonical
    name, so we need to pick ONE code per name for the reverse lookup. MLB Stats API
    returns current team names, so use the modern Retrosheet code for each."""
    current_code = {
        "Los Angeles Angels":     "LAA",
        "Baltimore Orioles":      "BAL",
        "Boston Red Sox":         "BOS",
        "Chicago White Sox":      "CHA",
        "Cleveland Guardians":    "CLE",
        "Detroit Tigers":         "DET",
        "Houston Astros":         "HOU",
        "Kansas City Royals":     "KCA",
        "Minnesota Twins":        "MIN",
        "New York Yankees":       "NYA",
        "Oakland Athletics":      "OAK",
        # Athletics 2025+ are a separate franchise (left Oakland; Vegas-bound; Sacramento
        # is a transitional venue). MLB API name 'Athletics' → 'ATH' Retrosheet code so
        # gids match Retrosheet's own coding for these games.
        "Athletics":              "ATH",
        "Las Vegas Athletics":    "ATH",
        "Seattle Mariners":       "SEA",
        "Tampa Bay Rays":         "TBA",
        "Texas Rangers":          "TEX",
        "Toronto Blue Jays":      "TOR",
        "Arizona Diamondbacks":   "ARI",
        "Atlanta Braves":         "ATL",
        "Chicago Cubs":           "CHN",
        "Cincinnati Reds":        "CIN",
        "Colorado Rockies":       "COL",
        "Los Angeles Dodgers":    "LAN",
        "Miami Marlins":          "MIA",
        "Milwaukee Brewers":      "MIL",
        "New York Mets":          "NYN",
        "Philadelphia Phillies":  "PHI",
        "Pittsburgh Pirates":     "PIT",
        "San Diego Padres":       "SDN",
        "San Francisco Giants":   "SFN",
        "St. Louis Cardinals":    "SLN",
        "Washington Nationals":   "WAS",
    }
    return current_code


def fetch_mlb_stats_api(year, since_date=None):
    """Pull current-year games from MLB Stats API. Returns a DataFrame in the same shape
    as Retrosheet's gameinfo.csv so it can merge cleanly. Filters to Final games only.
    All gameTypes included (R = regular season, F/D/L/W = playoff rounds).

    If `since_date` is given (str 'YYYY-MM-DD'), pulls from that date onward; otherwise
    pulls the full year from March 1.
    """
    start = since_date or f"{year}-03-01"
    end   = f"{year}-12-31"
    url = f"{STATSAPI_URL}?sportId=1&startDate={start}&endDate={end}"
    print(f"Fetching MLB Stats API current-year games from {start} to {end}...")
    with urlopen(url) as resp:
        data = json.load(resp)

    name_to_code = _name_to_retrosheet_code()

    rows = []
    unmapped = set()
    for date_obj in data.get("dates", []):
        for g in date_obj.get("games", []):
            if g.get("status", {}).get("detailedState") != "Final":
                continue
            if g.get("gameType") not in {"R", "F", "D", "L", "W"}:
                continue  # exclude spring training, exhibitions, All-Star, etc.
            home_name  = g["teams"]["home"]["team"]["name"]
            away_name  = g["teams"]["away"]["team"]["name"]
            home_score = g["teams"]["home"].get("score")
            away_score = g["teams"]["away"].get("score")
            if home_score is None or away_score is None:
                continue
            home_code = name_to_code.get(home_name)
            away_code = name_to_code.get(away_name)
            if not home_code or not away_code:
                unmapped.add(home_name if not home_code else away_name)
                continue
            # Date format: YYYYMMDD (matches Retrosheet's `date` column)
            date_str = g["officialDate"].replace("-", "")
            # Doubleheader: Retrosheet uses 0 for single, 1/2 for doubleheader games.
            # MLB Stats API uses 1 by default and 2 for second game.
            game_no  = int(g.get("gameNumber", 1))
            retro_dh = 0 if game_no == 1 and not g.get("doubleHeader", "N") in ("Y", "S") else game_no
            # gid format mirrors Retrosheet so dedup-by-gid works if Retrosheet ever
            # publishes the same game later.
            gid = f"{home_code}{date_str}{retro_dh}"
            api_to_retro_gametype = {
                "R": "regular",
                "F": "wildcard",
                "D": "divisionseries",
                "L": "lcs",
                "W": "worldseries",
            }
            rows.append({
                "gid": gid,
                "date": int(date_str),
                "hometeam": home_code,
                "visteam": away_code,
                "hruns": int(home_score),
                "vruns": int(away_score),
                "number": retro_dh,
                "gametype": api_to_retro_gametype.get(g.get("gameType"), "regular"),
            })

    if unmapped:
        print(f"  WARN: unmapped team names from MLB Stats API: {sorted(unmapped)}")
    df = pd.DataFrame(rows)
    print(f"  Fetched {len(df)} Final {year} games.")
    return df


def merge_game_sources(retrosheet_df, current_df):
    """Concat Retrosheet historical + MLB Stats API current-year. Dedupes by gid so
    repeated games are kept once. Retrosheet wins for the overlap (richer metadata)."""
    if current_df is None or current_df.empty:
        return retrosheet_df
    # Drop current-year gids from Retrosheet (let MLB Stats API supply them) so we don't
    # carry stale partial-season data if Retrosheet has a few early games but not the rest.
    current_year = pd.to_datetime(current_df["date"].astype(str), format="%Y%m%d").dt.year.max()
    retro_year = pd.to_datetime(retrosheet_df["date"].astype(str), format="%Y%m%d", errors="coerce").dt.year
    retro_clean = retrosheet_df[retro_year != current_year].copy()
    combined = pd.concat([retro_clean, current_df], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["gid"], keep="first")
    return combined


def prepare_game_data(raw_df):
    """Convert Retrosheet gameinfo rows into our master game frame."""
    df = raw_df.copy()
    # Parse date YYYYMMDD -> datetime
    df["date_game"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["year"] = df["date_game"].dt.year
    df["season"] = df["year"]  # MLB season = calendar year (no cross-year wraparound)

    # Filter to MIN_SEASON+ and games with valid scores
    df = df[df["year"] >= MIN_SEASON].copy()
    df["hruns"] = pd.to_numeric(df["hruns"], errors="coerce")
    df["vruns"] = pd.to_numeric(df["vruns"], errors="coerce")
    df = df.dropna(subset=["hruns", "vruns", "date_game"])
    df["hruns"] = df["hruns"].astype(int)
    df["vruns"] = df["vruns"].astype(int)

    # gametype: 'regular' / 'wildcard' / 'divisionseries' / 'lcs' / 'worldseries' / 'allstar'.
    # Drop All-Star + any unrecognized values; keep only games that count.
    if "gametype" not in df.columns:
        df["gametype"] = "regular"
    df["gametype"] = df["gametype"].fillna("regular").astype(str).str.lower()
    keep_types = {"regular", "wildcard", "divisionseries", "lcs", "worldseries", "playoff"}
    pre = len(df)
    df = df[df["gametype"].isin(keep_types)].copy()
    dropped = pre - len(df)
    if dropped:
        print(f"  Dropped {dropped} non-counting games (All-Star etc.).")

    # Map team codes -> canonical full names. Any unmapped code = surface a warning.
    df["home_team_name"]    = df["hometeam"].map(RETROSHEET_TEAM)
    df["visitor_team_name"] = df["visteam"].map(RETROSHEET_TEAM)
    unmapped_home = df.loc[df["home_team_name"].isna(), "hometeam"].unique()
    unmapped_away = df.loc[df["visitor_team_name"].isna(), "visteam"].unique()
    unmapped = sorted(set(unmapped_home) | set(unmapped_away))
    if unmapped:
        print(f"WARN: unmapped Retrosheet team codes (dropping {df['home_team_name'].isna().sum() + df['visitor_team_name'].isna().sum()} rows): {unmapped}")
    df = df.dropna(subset=["home_team_name", "visitor_team_name"]).copy()

    # Margins (home-team perspective)
    df["home_pts"] = df["hruns"]
    df["visitor_pts"] = df["vruns"]
    df["home_margin"] = df["hruns"] - df["vruns"]
    df["visitor_margin"] = -df["home_margin"]

    # Win flags
    df["home_win"] = (df["home_margin"] > 0).astype(int)
    df["visitor_win"] = (df["home_margin"] < 0).astype(int)
    # MLB has no ties in modern era - extra innings always determine a winner. (Pre-1969
    # there were occasional ties before suspended-game rules tightened.)
    df["is_tie"] = (df["home_margin"] == 0).astype(int)

    # Sort + dedupe
    df = df.sort_values("date_game").drop_duplicates(subset=["gid"], keep="first")

    # Date IDs
    df["grouped_date_id"] = df.groupby("date_game").ngroup() + 1
    df["unique_game_id"]  = np.arange(1, len(df) + 1)

    # Result strings for last-game display
    df["home_wl"]      = np.where(df["home_win"] == 1, "W", np.where(df["is_tie"] == 1, "T", "L"))
    df["visitor_wl"]   = np.where(df["visitor_win"] == 1, "W", np.where(df["is_tie"] == 1, "T", "L"))
    df["home_result"] = (
        df["home_wl"] + " " + df["home_pts"].astype(str) + "-" + df["visitor_pts"].astype(str)
        + " vs. " + df["visitor_team_name"]
    )
    df["visitor_result"] = (
        df["visitor_wl"] + " " + df["visitor_pts"].astype(str) + "-" + df["home_pts"].astype(str)
        + " @ " + df["home_team_name"]
    )

    out_cols = [
        "gid", "season", "year", "date_game",
        "visitor_team_name", "home_team_name",
        "visitor_pts", "home_pts",
        "visitor_margin", "home_margin",
        "visitor_win", "home_win", "is_tie",
        "grouped_date_id", "unique_game_id",
        "home_wl", "visitor_wl",
        "home_result", "visitor_result",
        "gametype",
    ]
    df = df[out_cols]
    df.to_csv(ALL_GAMES_CSV, index=False)
    print(f"  Prepared {len(df):,} games, {df['season'].min()}-{df['season'].max()}.")
    return df


# =========================================================
# WLS MASSEY SOLVER
# =========================================================

def _apply_margin_transform(margin, transform, cap):
    """Sign-preserving transform applied to (raw_margin - hca)."""
    m = np.asarray(margin, dtype=float)
    if transform == "raw":
        return m
    if transform == "cap":
        return np.clip(m, -cap, cap)
    raise ValueError(f"Unknown MARGIN_TRANSFORM: {transform}")


def _connected_components(teams, edges):
    """Find connected components in the team-game graph.
    teams: list of team names. edges: iterable of (home_name, away_name) tuples.
    Returns dict mapping team_name -> component_id (0-indexed)."""
    parent = {t: t for t in teams}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for h, a in edges:
        union(h, a)

    # Map roots to component IDs
    roots = {}
    out = {}
    for t in teams:
        r = find(t)
        if r not in roots:
            roots[r] = len(roots)
        out[t] = roots[r]
    return out


def _solve_massey(window_df, hca, weighting_mode, margin_transform, margin_cap, season):
    """
    Solve for team Massey ratings on a single rolling window.

    Zero-sum constraint is applied PER (connected component, league) — so AL and NL
    are independently centered at zero within each component. This prevents the small
    sample of inter-league games (6 WS games pre-1997, ~250 interleague + WS post-1997)
    from sloshing the cross-league baseline around. WS / interleague games still
    influence INDIVIDUAL ratings via the regression, just not league means.

    Components also handle MLB's 2020 COVID regional schedule cleanly.

    WLS via row-scaling: multiplying both X and y by sqrt(w_i) turns the weighted problem
    into an ordinary lstsq.
    """
    teams = sorted(set(window_df["home_team_name"]) | set(window_df["visitor_team_name"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    home_pts = window_df["home_pts"].to_numpy(dtype=float)
    visitor_pts = window_df["visitor_pts"].to_numpy(dtype=float)
    weights = window_df["date_weight"].to_numpy(dtype=float)
    home_names = window_df["home_team_name"].to_numpy()
    visitor_names = window_df["visitor_team_name"].to_numpy()

    # Connected components × era-aware league → one zero-sum anchor each
    comp_map = _connected_components(teams, zip(home_names, visitor_names))
    league_lookup = {t: _team_league(t, season) for t in teams}
    anchor_groups = {}  # (comp_id, league) -> [teams]
    for t in teams:
        anchor_groups.setdefault((comp_map[t], league_lookup[t]), []).append(t)
    anchor_keys = sorted(anchor_groups.keys())

    n_rows = n_games + len(anchor_keys)
    X = np.zeros((n_rows, n_teams))
    y = np.zeros(n_rows)
    w = np.zeros(n_rows)

    raw_margin = home_pts - visitor_pts - hca
    transformed = _apply_margin_transform(raw_margin, margin_transform, margin_cap)

    for i in range(n_games):
        X[i, team_idx[home_names[i]]] = 1.0
        X[i, team_idx[visitor_names[i]]] = -1.0

    if weighting_mode == "wls":
        y[:n_games] = transformed
        w[:n_games] = weights
    else:
        raise ValueError(f"Unknown WEIGHTING_MODE: {weighting_mode}")

    # One zero-sum row per (component, league) — AL and NL each centered at zero.
    for k, key in enumerate(anchor_keys):
        row = n_games + k
        for t in anchor_groups[key]:
            X[row, team_idx[t]] = 1.0
        y[row] = 0.0
        w[row] = 1.0e8

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = pd.DataFrame({"name": teams, "rating": r, "component": [comp_map[t] for t in teams]})
    out["rank"] = out["rating"].rank(ascending=False, method="min").astype(int)
    return out


def _window_for_season(season):
    """Fixed 139-game-day rolling window across all seasons.

    Why 139: ~75% of MLB's calendar regular season (~185 game-days). This
    matches DUNCAN's calibration ratio (123 game-days = 75% of NBA's ~165
    game-day RS), which is the fleet's best-calibrated model. The previous
    202-game-day window covered ~100% of RS + ~17 game-days of prior PS,
    which created an asymmetric carryover for teams that recently made
    playoffs. 139 stays entirely within the current season at RS-end.

    Why fixed not variable: short seasons (1981 strike, 1994 strike, 1995
    catch-up, 2020 COVID) used to get a proportionally shrunk window, which
    inflated tiny-sample ratings for whoever ran hot in those years.
    """
    return 250


_MIN_WINDOW = min(_window_for_season(s) for s in REGULAR_SEASON_GAMES)


# =========================================================
# RATING LOOP
# =========================================================

def compute_ratings(master_df, existing_ratings_df):
    """
    Compute per-game-day Massey power ratings using a season-aware rolling window
    (WINDOW_MULTIPLIER × games-per-team-this-season). Skips dates already present in
    existing_ratings_df. Re-processes the most recent RECOMPUTE_TAIL_DAYS ranking_ids
    each run to absorb late-arriving data.
    """
    max_date_id = int(master_df["grouped_date_id"].max())
    min_date_id = _MIN_WINDOW

    if len(existing_ratings_df):
        all_ids = sorted(existing_ratings_df["ranking_id"].unique())
        if len(all_ids) > RECOMPUTE_TAIL_DAYS:
            tail_threshold = all_ids[-RECOMPUTE_TAIL_DAYS]
            n_dropped = int((existing_ratings_df["ranking_id"] >= tail_threshold).sum())
            existing_ratings_df = existing_ratings_df[
                existing_ratings_df["ranking_id"] < tail_threshold
            ].copy()
            print(f"  Re-processing tail {RECOMPUTE_TAIL_DAYS} game-days "
                  f"({n_dropped:,} rows dropped from ratings cache).")
        max_ranked = int(existing_ratings_df["ranking_id"].max()) if len(existing_ratings_df) else -1
        min_ranked = int(existing_ratings_df["ranking_id"].min()) if len(existing_ratings_df) else -1
    else:
        max_ranked, min_ranked = -1, -1

    print("Running GRIFFEY ratings for new data...")
    new_frames = []

    # Per-game-day -> season lookup for window sizing
    rid_to_season = (
        master_df.sort_values("grouped_date_id")
                 .drop_duplicates("grouped_date_id", keep="last")
                 .set_index("grouped_date_id")["season"]
                 .to_dict()
    )

    last_printed_ym = None
    for i in range(min_date_id, max_date_id + 1):
        if min_ranked <= i <= max_ranked:
            continue

        season_for_window = rid_to_season.get(i)
        if season_for_window is None:
            prior_ids = [k for k in rid_to_season if k < i]
            season_for_window = rid_to_season[max(prior_ids)] if prior_ids else MIN_SEASON
        window_size = _window_for_season(season_for_window)

        # Per-season window-fill gate - don't publish until this season's window is full.
        if i < window_size:
            continue

        window = master_df[
            (master_df["grouped_date_id"] >= i - (window_size - 1)) &
            (master_df["grouped_date_id"] <= i)
        ].copy()

        window["date_weight"] = (window["grouped_date_id"] - i + window_size) / window_size

        current_date = window["date_game"].max()
        season = window["season"].max()

        # Throttled progress: print once per month rather than per game-day
        current_ym = (current_date.year, current_date.month)
        if current_ym != last_printed_ym:
            pct = (i - min_date_id) / max(1, max_date_id - min_date_id) * 100
            print(f"  Ratings: {current_date.strftime('%B %Y')} ({pct:.0f}% complete)")
            last_printed_ym = current_ym

        try:
            ranked = _solve_massey(
                window,
                hca=HOME_COURT_ADJUSTMENT,
                weighting_mode=WEIGHTING_MODE,
                margin_transform=MARGIN_TRANSFORM,
                margin_cap=MARGIN_CAP,
                season=season,
            )
        except Exception as e:
            print(f"  [skip] grouped_date_id {i} ({current_date.date()}): {e}")
            continue

        ranked["ranking_date"] = current_date
        ranked["ranking_id"]   = i
        ranked["season"]       = season
        new_frames.append(ranked)

    if new_frames:
        ratings_df = pd.concat([existing_ratings_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    else:
        ratings_df = existing_ratings_df.copy()
    ratings_df.sort_values(["ranking_id", "name"], inplace=True)
    ratings_df.drop_duplicates(keep="first", inplace=True)
    ratings_df["ranking_date"] = pd.to_datetime(ratings_df["ranking_date"]).dt.date

    # MLB cache is large but compressible - use gzip like MESSI
    ratings_df.to_csv("griffey_ratings.csv.gz", index=False, compression="gzip")
    print(f"griffey_ratings.csv.gz saved ({len(ratings_df):,} rows)")
    return ratings_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    # Load Retrosheet historical data (cached locally to avoid hammering each run)
    if os.path.exists(LOADED_GAMES_CSV):
        raw = pd.read_csv(LOADED_GAMES_CSV, low_memory=False)
        print(f"Loaded {len(raw):,} games from local cache.")
    else:
        raw = fetch_retrosheet()
        raw.to_csv(LOADED_GAMES_CSV, index=False)
        print(f"Cached raw games to {LOADED_GAMES_CSV}")

    # Add current-year data from MLB Stats API (Retrosheet only updates annually).
    current_year = datetime.now().year
    try:
        current = fetch_mlb_stats_api(current_year)
        raw = merge_game_sources(raw, current)
        print(f"  Combined dataset: {len(raw):,} games.")
    except Exception as e:
        print(f"  WARN: MLB Stats API fetch failed ({e}). Continuing with Retrosheet only.")

    master = prepare_game_data(raw)
    print(f"\nMargin sanity: mean home margin = {master['home_margin'].mean():+.3f}, home win rate = {master['home_win'].mean()*100:.1f}%")

    # Ratings (incremental - load existing cache if present)
    try:
        existing_ratings = pd.read_csv("griffey_ratings.csv.gz")
    except FileNotFoundError:
        existing_ratings = pd.DataFrame(columns=["ranking_id", "ranking_date", "season", "name", "rating", "rank"])
    ratings = compute_ratings(master, existing_ratings)

    # Quick smoke: top 10 at latest snapshot
    latest_id = ratings["ranking_id"].max()
    latest = ratings[ratings["ranking_id"] == latest_id].sort_values("rating", ascending=False)
    print(f"\nTop 10 at latest snapshot ({ratings[ratings['ranking_id']==latest_id]['ranking_date'].iloc[0]}):")
    print(latest.head(10)[["rank", "name", "rating"]].to_string(index=False))
