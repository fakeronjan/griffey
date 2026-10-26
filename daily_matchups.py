"""Daily Matchups: every game of a day, previewed from the ratings going into
it - win probability, GRIFFEY's moneyline and O/U (projected total runs), and
the stakes (each team's postseason and title odds with a win vs with a loss).
After the games: the final score and whether GRIFFEY's pick was right. The
MLB port of DUNCAN's Daily Matchups (duncan/daily_matchups.py).

Stakes come from the season sim (playoff_sim.py): one run of 10k simulations
from the last snapshot before the day. A day with no snapshot yet this
season (opening day) previews from last season's final ratings. Postseason
games use the series' next game in the sim. Game 163 tiebreakers settle
standings ties outside the sim: Quality only, no stakes. So are tie games
(the sim leaves them out) and 1994 (no postseason).

Moneyline: the fair (no-vig) American odds for the sim's win probability,
rounded to 5. O/U (fit on 1962-2026 games, leave-one-season-out; MAE 3.48
runs vs 3.52 league average only):
    total = 2 * mu + TOTAL_B * (both BAT - both PIT) + PARK_C * home park factor
            + PS_OFFSET in the postseason and tiebreakers
        mu = league runs per team-game this season so far, shrunk toward
        last season's with MU_PRIOR_GAMES pseudo team-games
        park factor = 0 at neutral sites
        PS_OFFSET: October scoring runs lower (aces, top bullpens); without
        it postseason totals ran 0.74 high (postseason MAE 3.36 -> 3.24)
First pitch: Retrosheet's start time (local, converted by park) or the MLB
Stats API's for the current season (griffey.py); none listed for many games
before 1990.
"""
import hashlib
import multiprocessing as _mp
import os
import pickle

import numpy as np
import pandas as pd
from scipy.special import ndtr

import playoff_sim

N_SIMS = 10_000                   # fleet standard since 2026-10-02
TOTAL_B = 0.425
PARK_C = 1.108
MU_PRIOR_GAMES = 300
PS_OFFSET = -0.737
POSTSEASON = ('wildcard', 'divisionseries', 'lcs', 'worldseries')


def _record(w, l):
    return f"{w}-{l}"


def _moneyline(p):
    """Fair American odds for win probability p, rounded to 5 (-145 = bet
    145 to win 100; +145 = bet 100 to win 145)."""
    p = min(max(p, 1e-4), 1 - 1e-4)
    v = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    return int(5 * round(v / 5))


def _win_value(y, wins, ctrl):
    """(mean of y, change in y per extra win) for outcome y (per-sim bool)
    and the team's final regular-season wins in each sim.

    Splitting the sims by this one game's result (as DILLON does) also sorts
    them by how strong the drift draw made each team, which inflated early-
    season stakes ~3.5x over late-season ones on DUNCAN. Holding the drift
    fixed removes that, but one game's own result is then too noisy for
    title odds at 10k sims. So: regress y on the team's final win total plus
    both teams' drift offsets. Given the drift, the win total varies only by
    luck across all remaining games, so the slope is the value of one more
    win, with much less noise."""
    y = y.astype(float)
    m = float(y.mean())
    X = np.column_stack([np.ones(len(y)), wins] + ([ctrl] if ctrl is not None else []))
    if np.ptp(wins) == 0:
        return m, 0.0
    return m, max(float(np.linalg.lstsq(X, y, rcond=None)[0][1]), 0.0)


def _raw_split(y, won):
    """Postseason games: no drift left, so split the sims by the game's result."""
    y = y.astype(float)
    return (float(y[won].mean()) if won.any() else None,
            float(y[~won].mean()) if (~won).any() else None)


def _around(m, b, pw):
    """Split mean m into (with a win, with a loss) for a +b per-win effect."""
    clip = lambda v: min(max(v, 0.0), 1.0)
    return clip(m + b * (1 - pw)), clip(m - b * pw)


def _mu_by_day(season_games, prev_mu):
    """League runs per team-game going into each day: this season so far,
    shrunk toward last season's mean."""
    played = season_games[season_games['home_pts'].notna()]
    per_day = played.groupby('date').apply(
        lambda x: (x['home_pts'].sum() + x['visitor_pts'].sum(), 2 * len(x)))
    out, s, n = {}, 0.0, 0
    for d in sorted(set(season_games['date'])):
        out[d] = (s + MU_PRIOR_GAMES * prev_mu) / (n + MU_PRIOR_GAMES)
        if d in per_day.index:
            s += per_day[d][0]; n += per_day[d][1]
    return out


def build_season(season, games, ratings, league_of, div_of, prev_mu, n_sims=N_SIMS, log=print):
    """games: the season's games (season, date, home, away, home_pts,
    visitor_pts, gametype, is_neutral, game_number, first_pitch; NaN pts =
    scheduled). ratings: (season, date, name, rating, rank, rating_o,
    rating_d, park_factor) for this season and the last. Returns the
    season's days (list of dicts)."""
    g = games[games['season'] == season].sort_values(
        ['date', 'home', 'away', 'game_number'], kind='stable').reset_index(drop=True)
    snaps = {d: x.set_index('name') for d, x in ratings.groupby('date')}
    cur = sorted(d for d in snaps if ratings.loc[ratings['date'] == d, 'season'].iloc[0] == season)
    prev = sorted(d for d in snaps if d not in set(cur) and d < (cur[0] if cur else pd.Timestamp.max))
    sim_ratings = {d: dict(zip(snaps[d].index, snaps[d]['rating'])) for d in cur}
    # Scheduled (unplayed) rows are the sim's remaining schedule only in the
    # regular season; an unplayed postseason row would read as a played game.
    sim_g = g[(g['gametype'] == 'regular') | g['home_pts'].notna()]
    sim = playoff_sim.SeasonSim(season, sim_g, league_of, div_of, sim_ratings)
    rounds = playoff_sim.round_names(season)[0]
    rs_keys = set(zip(sim.rs['date'], sim.rs['home'], sim.rs['away']))
    mu = _mu_by_day(g, prev_mu)
    no_ps = season in playoff_sim.NO_POSTSEASON

    days = sorted(set(g['date']))
    unplayed_days = sorted(set(g.loc[g['home_pts'].isna(), 'date']))
    preview_day = unplayed_days[0] if unplayed_days else None

    days_out, previewed = [], False
    for day in days:
        if previewed:
            break                                     # live season: preview only the next game day
        before = [d for d in cur if d < day]
        if before:
            snap_d = before[-1]
        elif prev:
            snap_d = prev[-1]                         # opening day: last season's final ratings
            sim.ratings[snap_d] = dict(zip(snaps[snap_d].index, snaps[snap_d]['rating']))
        else:
            continue                                  # 1961: no ratings yet
        rt = snaps[snap_d]
        cap = {}
        sim.odds_at(snap_d, n_sims=n_sims, capture=cap)
        idx = {t: i for i, t in enumerate(cap['teams'])}
        made, champ = cap['made'], cap['champ']
        rest = cap.get('rest')
        E, Wsim = cap.get('E'), cap.get('W')
        value = {}                                    # team -> win values, once per day

        def win_value(t, opp):
            if t not in value:
                it = idx[t]
                ctrl = None if E is None else E[:, [it, idx[opp]]]
                value[t] = (_win_value(made[:, it], Wsim[:, it], ctrl),
                            _win_value(champ == it, Wsim[:, it], ctrl))
            return value[t]
        ps_next = {frozenset((a, b)): (a, won, bo, rnd) for a, b, won, bo, rnd in cap['ps_next']}

        # Regular-season records going into the day; series scores for postseason games.
        played = g[(g['date'] < day) & g['home_pts'].notna()]
        rec, series = {}, {}
        for x in played.itertuples(index=False):
            if x.home_pts == x.visitor_pts:
                continue                              # tie games don't count
            win = x.home if x.home_pts > x.visitor_pts else x.away
            if x.gametype == 'regular':
                for t in (x.home, x.away):
                    w, l = rec.get(t, (0, 0))
                    rec[t] = (w + (t == win), l + (t != win))
            elif x.gametype in POSTSEASON:
                k = frozenset((x.home, x.away))
                series.setdefault(k, {}).setdefault(win, 0)
                series[k][win] += 1

        day_g = g[g['date'] == day]
        # Doubleheaders: the same two teams twice in a day.
        n_pair = day_g.groupby(['home', 'away']).size().to_dict()
        seen = {}
        games_out = []
        for x in day_g.itertuples(index=False):
            h, a = x.home, x.away
            k_dh = seen.get((h, a), 0)
            seen[(h, a)] = k_dh + 1
            neutral = int(x.is_neutral or 0) == 1
            hp = 0.0 if neutral else sim.hp
            blank = pd.Series({'rating': 0.0, 'rating_o': 0.0, 'rating_d': 0.0, 'park_factor': 0.0, 'rank': np.nan})
            rh = rt.loc[h] if h in rt.index else blank
            ra = rt.loc[a] if a in rt.index else blank
            is_tie = not pd.isna(x.home_pts) and x.home_pts == x.visitor_pts
            stakes, ser = None, None
            if x.gametype == 'regular' and not is_tie and not no_ps and rest is not None \
                    and (x.date, h, a) in rs_keys and h in idx and a in idx:
                j = rest.index[(rest['home'] == h) & (rest['away'] == a) & (rest['date'] == day)]
                if not len(j):
                    continue
                hw = cap['hw'][:, j[min(k_dh, len(j) - 1)]]
                p_home = float(hw.mean())
                stakes = {}
                for t, opp, pw in ((h, a, p_home), (a, h, 1 - p_home)):
                    (po_m, po_b), (ti_m, ti_b) = win_value(t, opp)
                    po_w, po_l = _around(po_m, po_b, pw)
                    ti_w, ti_l = _around(ti_m, ti_b, pw)
                    stakes['home' if t == h else 'away'] = {
                        'po_win': po_w, 'po_loss': po_l, 'title_win': ti_w, 'title_loss': ti_l}
            elif x.gametype in POSTSEASON:
                hit = ps_next.get(frozenset((h, a)))
                if hit is None:
                    continue                          # an unplayed game after the series ended
                ta, won_a, best_of, rnd = hit
                sw = series.get(frozenset((h, a)), {})
                hw = won_a if ta == h else ~won_a
                p_home = float(hw.mean())
                stakes = {}
                for t, won in ((h, hw), (a, ~hw)):
                    ti_w, ti_l = _raw_split(champ == idx[t], won)
                    stakes['home' if t == h else 'away'] = {
                        'po_win': None, 'po_loss': None, 'title_win': ti_w, 'title_loss': ti_l}
                ser = {'home_wins': sw.get(h, 0), 'away_wins': sw.get(a, 0), 'best_of': int(best_of),
                       'round': rounds[rnd - 1]}
            else:
                # Game 163 tiebreakers, tie games, 1994: Quality only.
                p_home = float(ndtr(sim.A * (rh['rating'] - ra['rating'] + hp)))
            park = 0.0 if neutral else float(rh['park_factor'] if not pd.isna(rh['park_factor']) else 0.0)
            total = 2 * mu.get(day, prev_mu) + TOTAL_B * (
                (rh['rating_o'] + ra['rating_o']) - (rh['rating_d'] + ra['rating_d'])) + PARK_C * park + (
                PS_OFFSET if x.gametype != 'regular' else 0.0)
            game = {
                'home': h, 'away': a, 'neutral': bool(neutral),
                'home_record': _record(*rec.get(h, (0, 0))), 'away_record': _record(*rec.get(a, (0, 0))),
                'home_rank': None if pd.isna(rh['rank']) else int(rh['rank']),
                'away_rank': None if pd.isna(ra['rank']) else int(ra['rank']),
                'p_home': round(p_home, 4),
                'moneyline': _moneyline(p_home),            # home's fair moneyline
                'total': round(float(total) * 2) / 2,       # GRIFFEY O/U: projected combined runs
                'kickoff': x.first_pitch if isinstance(x.first_pitch, str) else None,
                'stakes': None if stakes is None else
                          {k: {kk: (None if vv is None else round(vv, 4)) for kk, vv in v.items()} for k, v in stakes.items()},
                'result': None,
            }
            if n_pair.get((h, a), 1) > 1:
                game['dh'] = k_dh + 1                       # doubleheader game 1 / 2
            if ser is not None:
                game['series'] = ser
            if x.gametype == 'playoff':
                game['tiebreaker'] = True
            if not pd.isna(x.home_pts):
                game['result'] = {'home': int(x.home_pts), 'away': int(x.visitor_pts)}
            game['_q'] = (min(rh['rating'], ra['rating']), max(rh['rating'], ra['rating']))
            games_out.append(game)
        # The next game day with a game left on it: an unplayed game after
        # its series ended doesn't count.
        previewed = preview_day is not None and day >= preview_day and bool(games_out)
        if games_out:
            days_out.append({'date': day.strftime('%Y-%m-%d'),
                             'playoffs': any('series' in x for x in games_out), 'games': games_out})
        log(f'  {season} {day.date()}: {len(games_out)} games')
    return days_out


# ── Juice: Quality x Stakes ──────────────────────────────────────────────────
# As DUNCAN, after DILLON (weekly_matchups.py): Quality = 35% the worse
# team's rating, 35% the better team's, 30% closeness; Stakes = both teams'
# postseason-odds swing + K x their title-odds swing, K ramping 4 -> 8 over
# the regular season (postseason games: title swing only, K = 8). Each is
# ranked 0-100 and Juice = sqrt(Quality x Stakes).
#
# The ranking pool is COMPLETED seasons only: ranking against the live
# season too would move every historical score every day and rewrite all 66
# season files nightly. Live-season games are placed within the completed
# seasons' distribution instead, so history only re-ranks once a year, when a
# season finishes.
TITLE_K_START, TITLE_K_END = 4.0, 8.0


def _pct(ref_sorted, x):
    """Percentile (0-1) of x within ref, matching pandas rank(pct=True)'s
    average rank for values already in ref."""
    lo = np.searchsorted(ref_sorted, x, side='left')
    hi = np.searchsorted(ref_sorted, x, side='right')
    return ((lo + hi + 1) / 2) / len(ref_sorted)


def add_juice(seasons, pool_seasons):
    """seasons: {season: days (build_season output)}; pool_seasons: the
    completed seasons to rank against. Adds 'quality', 'stakes_score' and
    'juice' (0-100) to every game and drops the internal '_q'."""
    rows = []
    for season, days in seasons.items():
        rs_days = [d['date'] for d in days if not d['playoffs']]
        first, last = (rs_days[0], rs_days[-1]) if rs_days else (None, None)
        span = (pd.Timestamp(last) - pd.Timestamp(first)).days if rs_days else 1
        for d in days:
            if d['playoffs']:
                k = TITLE_K_END
            else:
                frac = (pd.Timestamp(d['date']) - pd.Timestamp(first)).days / max(span, 1)
                k = TITLE_K_START + (TITLE_K_END - TITLE_K_START) * min(frac, 1.0)
            for g in d['games']:
                qmin, qmax = g.pop('_q')
                close = 1 - abs(2 * g['p_home'] - 1)
                stake = None
                if g['stakes'] is not None:
                    st = list(g['stakes'].values())
                    po = sum((s['po_win'] or 0) - (s['po_loss'] or 0) for s in st)
                    ti = sum((s['title_win'] or 0) - (s['title_loss'] or 0) for s in st)
                    stake = po + k * ti
                rows.append((g, season in pool_seasons, qmin, qmax, close, stake))
    if not rows:
        return
    df = pd.DataFrame([r[1:] for r in rows], columns=['pool', 'qmin', 'qmax', 'close', 'stake'])
    ref = df[df['pool']]
    parts = {c: np.sort(ref[c].to_numpy()) for c in ('qmin', 'qmax', 'close')}
    q_raw = sum(w * _pct(parts[c], df[c].to_numpy()) for c, w in (('qmin', .35), ('qmax', .35), ('close', .3)))
    q = _pct(np.sort(q_raw[df['pool'].to_numpy()]), q_raw) * 100
    st_ref = np.sort(ref['stake'].dropna().to_numpy())
    for (g, *_), qq, ss in zip(rows, q, df['stake']):
        g['quality'] = int(round(min(qq, 100)))
        if pd.isna(ss):
            g['stakes_score'] = g['juice'] = None
        else:
            s_pct = min(_pct(st_ref, ss) * 100, 100)
            g['stakes_score'] = int(round(s_pct))
            g['juice'] = int(round(np.sqrt(min(qq, 100) * s_pct)))


# ── Per-season cache ─────────────────────────────────────────────────────────
# Finished seasons never change unless the engine or their inputs do: cache
# each season under a fingerprint of this file + the sim's engine files + the
# season's games (and last season's, for opening day and the scoring
# baseline) + ratings (rounded to 3dp; the ratings engine isn't
# bit-reproducible) + real seeds.
CACHE_DIR = 'matchups_cache'
_HERE = os.path.dirname(os.path.abspath(__file__))
_JOB = {}


def _fingerprint(season, J):
    h = hashlib.sha256()
    for f in ('daily_matchups.py',) + playoff_sim._ENGINE_FILES:
        p = os.path.join(_HERE, f)
        if os.path.exists(p):
            h.update(open(p, 'rb').read())
    g = J['games'][J['games']['season'].isin([season - 1, season])]
    h.update(g.sort_values(list(g.columns), kind='stable', na_position='last').to_csv(index=False).encode())
    r = J['ratings'][J['ratings']['season'].isin([season - 1, season])].sort_values(['date', 'name']).copy()
    for c in ('rating', 'rating_o', 'rating_d', 'park_factor'):
        r[c] = r[c].round(3)
    h.update(r.to_csv(index=False).encode())
    h.update(repr((playoff_sim.REAL_SEEDS.get(season), J['prev_mu'].get(season))).encode())
    return h.hexdigest()


def _one(season):
    J = _JOB
    r = J['ratings']
    days = build_season(season, J['games'], r[r['season'].isin([season - 1, season])],
                        J['league_of'], J['div_of'], J['prev_mu'][season], log=lambda *_: None)
    return season, days


def build_cached(seasons, games, ratings, league_of, div_of, workers=None, log=print):
    """{season: days} for every season, reusing cached ones. ratings needs
    season, date, name, rating, rank, rating_o, rating_d, park_factor."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    played = games[games['home_pts'].notna()]
    team_pts = pd.concat([played[['season', 'home_pts']].rename(columns={'home_pts': 'p'}),
                          played[['season', 'visitor_pts']].rename(columns={'visitor_pts': 'p'})])
    season_mu = team_pts.groupby('season')['p'].mean().to_dict()
    first = min(season_mu)
    prev_mu = {s: float(season_mu.get(s - 1, season_mu.get(s, season_mu[first]))) for s in seasons}
    _JOB.update(games=games, ratings=ratings, league_of=league_of, div_of=div_of, prev_mu=prev_mu)
    out, todo, sigs = {}, [], {}
    for s in seasons:
        sigs[s] = _fingerprint(s, _JOB)
        path = os.path.join(CACHE_DIR, f'{s}.pkl')
        if os.path.exists(path):
            try:
                old_sig, days = pickle.load(open(path, 'rb'))
                if old_sig == sigs[s]:
                    out[s] = days
                    continue
            except Exception:
                pass
        todo.append(s)
    log(f'  daily matchups: {len(seasons) - len(todo)} seasons from cache, computing {len(todo)}')
    if todo:
        # Most recent first: the live season isn't stuck behind 60 others.
        with _mp.get_context('fork').Pool(workers or os.cpu_count()) as pool:
            for s, days in pool.imap_unordered(_one, sorted(todo, reverse=True)):
                out[s] = days
                pickle.dump((sigs[s], days), open(os.path.join(CACHE_DIR, f'{s}.pkl'), 'wb'))
    return {s: out[s] for s in seasons}
