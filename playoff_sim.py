"""GRIFFEY title odds: Monte Carlo of the rest of the MLB season + postseason.

Port of LOBO/DUNCAN's playoff_sim.py. For every rating snapshot, simulate
the remaining regular-season games, qualify and seed each league under that
season's format, then play the bracket. Anything already played is fixed.

Game model (probit on 140,343 MLB games 1961-2026, pre-game snapshot
ratings, fit by log loss per era):
    P(home win) = Phi(A * (rating_home - rating_away + home_runs))
Neutral-site games (Tokyo/London/Mexico City series, 2020 Division Series
onward) get no home edge.

Output per (snapshot, team): probability of making the postseason, reaching
each later round, and the title (the Title odds column).
"""
import hashlib
import json as _json
import multiprocessing as _mp
import os as _os
import pickle

import numpy as np
import pandas as pd
from scipy.special import ndtr

# Simulation counts (fleet standard): regular-season dates 10k; once the
# regular season is over, 100k.
N_SIMS = 10_000
N_SIMS_PLAYOFFS = 100_000

# (first season, A, home-field runs), fit per era.
ERA_PARAMS = [
    (1961, 0.2038, 0.475),
    (1980, 0.1662, 0.582),
    (2000, 0.1741, 0.616),
    (2010, 0.1801, 0.504),
    (2021, 0.1829, 0.433),
]
# Ratings aren't fixed for the rest of the season: each simulation gives
# every team a random rating offset for the remaining games, SD =
# DRIFT_SD0 * (share of regular season left)**DRIFT_K, fit to how far this
# league's ratings actually moved from each date to the end of the regular
# season. Zero once the regular season is over. (Same fix as DILLON: fixed
# ratings made early-season odds overconfident.)
DRIFT_SD0, DRIFT_K = 0.633, 0.46
NO_POSTSEASON = {1994}                 # strike
# 1995-97 Division Series matchups followed a fixed rotation, not seeding;
# the wild card's actual opponent in the years that broke the usual rule
# (best division winner unless it's the wild card's own division).
WC_OPPONENT = {1995: {'AL': 'Seattle Mariners'}, 1997: {'AL': 'Cleveland Guardians'}}
SPLIT_1981 = pd.Timestamp('1981-08-10')  # second half of the split season

_TB = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'mlb_tiebreak_orders.json')
# Standings ties the league broke differently from our rules (tiebreaker-game
# results, then head-to-head, then run differential). Per season, tied teams
# in the order they were actually seeded; earlier wins.
TIEBREAK_WINNERS = ({int(k): v for k, v in _json.load(open(_TB)).items()}
                    if _os.path.exists(_TB) else {})


def era_params(season):
    a, h = ERA_PARAMS[0][1:]
    for start, aa, hh in ERA_PARAMS:
        if season >= start:
            a, h = aa, hh
    return a, h


def fmt(season):
    if season <= 1968:
        return 'pennant'
    if season == 1981:
        return 'split'
    if season <= 1993:
        return 'lcs'
    if season <= 2011:
        return 'wc1'
    if season == 2020:
        return 'sixteen'
    if season <= 2021:
        return 'wcgame'
    return 'twelve'


def round_names(season):
    """(full, short) names for every round, first to last."""
    f = fmt(season)
    if f == 'pennant':
        return ['World Series'], ['WS']
    if f == 'lcs':
        return ['League Championship Series', 'World Series'], ['LCS', 'WS']
    if f in ('split', 'wc1'):
        return ['Division Series', 'League Championship Series', 'World Series'], ['DS', 'LCS', 'WS']
    wc = 'Wild Card Game' if f == 'wcgame' else 'Wild Card Series'
    return [wc, 'Division Series', 'League Championship Series', 'World Series'], ['WC', 'DS', 'LCS', 'WS']


def n_seeds(season):
    return {'pennant': 1, 'lcs': 2, 'split': 4, 'wc1': 4, 'wcgame': 5, 'sixteen': 8, 'twelve': 6}[fmt(season)]


def entry_rounds(season):
    """Seed label ('A1', 'N5', ...) -> round that seed enters (later = bye)."""
    f = fmt(season)
    out = {}
    for lg in 'AN':
        for k in range(1, n_seeds(season) + 1):
            if f == 'wcgame':
                r = 1 if k >= 4 else 2
            elif f == 'twelve':
                r = 1 if k >= 3 else 2
            else:
                r = 1
            out[f'{lg}{k}'] = r
    return out


def host_pattern(best_of, kind):
    """Per game: True = better record hosts."""
    if kind == 'wcs':          # Wild Card Series: every game at the higher seed
        return [True] * best_of
    if best_of == 1:
        return [True]
    if best_of == 3:
        return [True, False, True]
    if best_of == 5:
        return [True, True, False, False, True]
    return [True, True, False, False, False, True, True]   # 2-3-2


class SeasonSim:
    def __init__(self, season, games, league_of, div_of, ratings, schedule=None):
        """games: the season's games (date, home, away, home_pts, visitor_pts,
        gametype). Regular season = gametype 'regular' (+ scheduled rows);
        'playoff' = tiebreaker games (settle standings ties); the rest are the
        postseason."""
        self.season = season
        g = games.sort_values('date', kind='stable').reset_index(drop=True)
        if 'is_neutral' not in g.columns:
            g = g.assign(is_neutral=0)
        rs = g[g['gametype'] == 'regular'][['date', 'home', 'away', 'home_pts', 'visitor_pts', 'is_neutral']]
        if schedule is not None and len(schedule):
            rs = pd.concat([rs, schedule.assign(home_pts=np.nan, visitor_pts=np.nan)], ignore_index=True)
        rs = rs.assign(is_neutral=rs['is_neutral'].fillna(0).astype(int))
        rs = rs[~((rs['home_pts'] == rs['visitor_pts']) & rs['home_pts'].notna())]  # tie games don't count
        self.teams = sorted(set(rs['home']) | set(rs['away']))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.lg = np.array([league_of(t, season) for t in self.teams])
        self.div = np.array([div_of(t, season) for t in self.teams])
        self.A, self.hp = era_params(season)
        rs = rs.assign(h=rs['home'].map(self.idx), a=rs['away'].map(self.idx))
        self.rs = rs.sort_values('date', kind='stable')
        tb = g[g['gametype'] == 'playoff']
        self.tb_wins = {}
        for x in tb.itertuples(index=False):
            w = x.home if x.home_pts > x.visitor_pts else x.away
            self.tb_wins[w] = self.tb_wins.get(w, 0) + 1
        ps = g[g['gametype'].isin(['wildcard', 'divisionseries', 'lcs', 'worldseries'])].copy()
        ps['winner'] = np.where(ps['home_pts'] > ps['visitor_pts'], ps['home'], ps['away'])
        self.ps = ps
        self.ratings = ratings
        self.n_rounds = len(round_names(season)[0])

    def rs_over(self, d):
        return not ((self.rs['date'] > d) | self.rs['home_pts'].isna()).any()

    def _static_tiebreak(self, done):
        """Deterministic tiebreak once the regular season is over: tiebreaker-
        game wins (the Game 163s), then head-to-head within the tie group, then
        run differential; recorded outcomes override. Higher = better; only
        compared within a group tied on win%."""
        T = len(self.teams)
        w = np.zeros(T); gp = np.zeros(T); rd = np.zeros(T)
        for h, a, hp, vp in done[['h', 'a', 'home_pts', 'visitor_pts']].itertuples(index=False):
            gp[h] += 1; gp[a] += 1; rd[h] += hp - vp; rd[a] += vp - hp
            w[h if hp > vp else a] += 1
        pct = w / np.maximum(gp, 1)
        score = np.zeros(T)
        for p in np.unique(pct):
            grp = np.where(pct == p)[0]
            if len(grp) < 2:
                continue
            gs = set(grp)
            sub = done[done['h'].isin(gs) & done['a'].isin(gs)]
            hw = np.zeros(T); hg = np.zeros(T)
            for h, a, hp, vp in sub[['h', 'a', 'home_pts', 'visitor_pts']].itertuples(index=False):
                hg[h] += 1; hg[a] += 1; hw[h if hp > vp else a] += 1
            h2h = np.where(hg > 0, hw / np.maximum(hg, 1), 0.5)
            key = sorted(grp, key=lambda t: (self.tb_wins.get(self.teams[t], 0), h2h[t], rd[t]), reverse=True)
            for rank, t in enumerate(key):
                score[t] = len(key) - rank
        order = TIEBREAK_WINNERS.get(self.season, [])
        for k, t in enumerate(order):
            if t in self.idx:
                score[self.idx[t]] += 1e6 * (len(order) - k)
        return score

    def odds_at(self, d, n_sims=N_SIMS):
        season = self.season
        f = fmt(season)
        T = len(self.teams)
        rng = np.random.default_rng(int(pd.Timestamp(d).strftime('%Y%m%d')))
        rt = self.ratings.get(d, {})
        R = np.array([rt.get(t, 0.0) for t in self.teams])
        A, hp = self.A, self.hp

        played = self.rs['home_pts'].notna() & (self.rs['date'] <= d)
        done, rest = self.rs[played], self.rs[~played]
        frac_left = len(rest) / max(len(self.rs), 1)
        sd = DRIFT_SD0 * frac_left ** DRIFT_K if frac_left > 0 else 0.0
        E = rng.normal(0.0, sd, (n_sims, T)) if sd > 0 else None   # per-sim rating offsets

        def standings(sub):
            w = np.zeros(T); gp = np.zeros(T)
            for h, a, hpt, vpt in sub[['h', 'a', 'home_pts', 'visitor_pts']].itertuples(index=False):
                gp[h] += 1; gp[a] += 1; w[h if hpt > vpt else a] += 1
            return w, gp

        w0, g0 = standings(done)
        W = np.tile(w0, (n_sims, 1)); G = np.tile(g0, (n_sims, 1))
        hw_all = None
        if len(rest):
            h = rest['h'].to_numpy(); a = rest['a'].to_numpy()
            hp_g = np.where(rest['is_neutral'].to_numpy() == 1, 0.0, hp)   # neutral site: no edge
            ph = ndtr(A * (R[h] - R[a] + hp_g + (0.0 if E is None else E[:, h] - E[:, a])))
            hw_all = (rng.random((n_sims, len(rest))) < ph).astype(np.float32)
            Hm = np.zeros((len(rest), T), np.float32); Hm[np.arange(len(rest)), h] = 1
            Am = np.zeros((len(rest), T), np.float32); Am[np.arange(len(rest)), a] = 1
            W += hw_all @ Hm + (1 - hw_all) @ Am
            G += (Hm + Am).sum(0)
        pct = W / np.maximum(G, 1)
        static = self._static_tiebreak(done) if rest.empty else np.zeros(T)
        noise = rng.random((n_sims, T))   # drawn either way: keeps the RNG stream fixed
        sim_ix = np.arange(n_sims)
        S = 1 if rest.empty else n_sims   # final table: seed once, broadcast
        srank = np.unique(static, return_inverse=True)[1].astype(float)
        tie_term = (srank[None, :] + noise[:S]) * 1e-8

        def ranked(members, pct_rows, bonus=None):
            m = np.array(members)
            key = pct_rows[:, m] + tie_term[:, m]
            if bonus is not None:
                key = key + bonus * 10.0
            return m[np.argsort(-key, axis=1, kind='stable')]

        pct_s = pct[:S]
        six = np.arange(S)

        def div_winners(lg, pct_rows):
            """(S, n_div) division winners of league lg, plus a mask of them."""
            m = np.where(self.lg == lg)[0]
            order = ranked(m, pct_rows)
            pos = np.empty((S, T), dtype=int)
            pos[six[:, None], order] = np.arange(len(m))[None, :]
            wins = []
            for dv in np.unique(self.div[m]):
                mem = m[self.div[m] == dv]
                wins.append(mem[np.argmin(pos[:, mem], axis=1)])
            return m, order, np.stack(wins, 1)

        # ── qualify + seed each league: seeds[lg] = (S, k) team idx by seed;
        #    brk[lg] = bracket-position arrays (may differ from seed order) ──
        seeds = {}
        for lg in ('AL', 'NL'):
            m = np.where(self.lg == lg)[0]
            if f == 'pennant':
                seeds[lg] = ranked(m, pct_s)[:, :1]
            elif f == 'split':
                # First-half winners (games before the strike) vs second-half
                # winners; if a team won both halves it plays the second-half
                # runner-up.
                def half(second):
                    part = done[(done['date'] >= SPLIT_1981) == second]
                    wd, gd = standings(part)
                    wh = np.tile(wd, (n_sims, 1)); gh = np.tile(gd, (n_sims, 1))
                    if len(rest):
                        msk = ((rest['date'] >= SPLIT_1981) == second).to_numpy()
                        if msk.any():
                            wh += hw_all[:, msk] @ Hm[msk] + (1 - hw_all[:, msk]) @ Am[msk]
                            gh += (Hm[msk] + Am[msk]).sum(0)
                    return (wh / np.maximum(gh, 1))[:S]
                p1, p2 = half(False), half(True)
                cols = []
                for dv in np.unique(self.div[m]):
                    mem = m[self.div[m] == dv]
                    first = ranked(mem, p1)[:, 0]
                    o2 = ranked(mem, p2)
                    second = np.where(o2[:, 0] == first, o2[:, 1], o2[:, 0])
                    cols += [first, second]
                seeds[lg] = np.stack(cols, 1)       # [E1st, E2nd, W1st, W2nd]
            else:
                _, order, dw = div_winners(lg, pct_s)
                is_dw = np.zeros((S, T), bool); is_dw[six[:, None], dw] = True
                # division winners by record, then everyone else by record
                bonus = is_dw[:, m].astype(float)
                full = ranked(m, pct_s, bonus)
                if f == 'lcs':
                    seeds[lg] = full[:, :2]
                elif f == 'wc1':
                    seeds[lg] = full[:, :4]
                elif f == 'wcgame':
                    seeds[lg] = full[:, :5]
                elif f == 'twelve':
                    seeds[lg] = full[:, :6]
                elif f == 'sixteen':
                    # 1-3 division winners, 4-6 division runners-up, 7-8 next best.
                    pos = np.empty((S, T), dtype=int)
                    pos[six[:, None], order] = np.arange(len(m))[None, :]
                    ru = []
                    for dv in np.unique(self.div[m]):
                        mem = m[self.div[m] == dv]
                        pm = np.where(is_dw[:, mem], 10**6, pos[:, mem])
                        ru.append(mem[np.argmin(pm, axis=1)])
                    ru = np.stack(ru, 1)
                    is_ru = np.zeros((S, T), bool); is_ru[six[:, None], ru] = True
                    tier = is_dw[:, m] * 2.0 + is_ru[:, m] * 1.0
                    seeds[lg] = ranked(m, pct_s, tier)[:, :8]
        if S == 1:
            seeds = {k: np.broadcast_to(v, (n_sims, v.shape[1])) for k, v in seeds.items()}
        # overall record rank per team (home field)
        allrank = ranked(range(T), pct_s)
        lg_rank = np.empty((S, T), dtype=int)
        lg_rank[six[:, None], allrank] = np.arange(T)[None, :]
        if S == 1:
            lg_rank = np.broadcast_to(lg_rank, (n_sims, T))

        ps_by_pair = {}
        for r in self.ps[self.ps['date'] <= d].itertuples(index=False):
            ps_by_pair.setdefault(frozenset((r.home, r.away)), []).append(r.winner)

        self.used_actual = 0
        self.rs_complete = rest.empty
        self.seeds = {}
        if self.rs_complete and season not in NO_POSTSEASON:
            for lg, arr in seeds.items():
                for k, t in enumerate(arr[0]):
                    self.seeds[self.teams[t]] = f"{lg[0]}{k + 1}"
        self.matchups = []
        reach = np.zeros((self.n_rounds + 2, T))
        if season in NO_POSTSEASON:
            cols = ['playoffs'] + [f'r{k}' for k in range(2, self.n_rounds + 1)] + ['champ']
            return pd.DataFrame(np.zeros((T, len(cols))), index=self.teams, columns=cols)
        entered = np.zeros((n_sims, T), dtype=bool)

        def play(a, b, rnd, bo, kind='series', neutral=False):
            for t in (a, b):
                new = ~entered[sim_ix, t]
                entered[sim_ix, t] = True
                np.add.at(reach[0], t[new], 1)
                for k in range(2, rnd):
                    np.add.at(reach[k], t[new], 1)
                np.add.at(reach[rnd], t, 1)
            a_better = lg_rank[sim_ix, a] < lg_rank[sim_ix, b]
            fixed = np.all(a == a[0]) and np.all(b == b[0])
            actual = ps_by_pair.get(frozenset((self.teams[a[0]], self.teams[b[0]])), []) if fixed else []
            need = bo // 2 + 1
            edge_h = 0.0 if neutral else hp
            wa = np.zeros(n_sims, dtype=int); wb = np.zeros(n_sims, dtype=int)
            for gi, better_hosts in enumerate(host_pattern(bo, kind)):
                if gi < len(actual):
                    won = np.full(n_sims, actual[gi] == self.teams[a[0]]); self.used_actual += 1
                else:
                    a_home = a_better if better_hosts else ~a_better
                    off = 0.0 if E is None else E[sim_ix, a] - E[sim_ix, b]
                    won = rng.random(n_sims) < ndtr(A * (R[a] - R[b] + off + np.where(a_home, edge_h, -edge_h)))
                live = (wa < need) & (wb < need)
                wa += won & live; wb += ~won & live
            a_wins = wa >= need
            if fixed and self.rs_complete:
                ta, tb = self.teams[a[0]], self.teams[b[0]]
                gm = list(actual[:bo])
                na, nb = gm.count(ta), gm.count(tb)
                decided = ta if na >= need else (tb if nb >= need else None)
                self.matchups.append((rnd, bo, ta, tb, gm, decided))
            return np.where(a_wins, a, b)

        champs = {}
        neutral_late = season == 2020
        for lg in ('AL', 'NL'):
            sd = seeds[lg]
            s = lambda k: sd[:, k - 1]
            if f == 'pennant':
                champs[lg] = s(1)
            elif f == 'lcs':
                champs[lg] = play(s(1), s(2), 1, 5 if season <= 1984 else 7)
            elif f == 'split':
                e = play(s(1), s(2), 1, 5)
                wst = play(s(3), s(4), 1, 5)
                champs[lg] = play(e, wst, 2, 5)
            elif f == 'wc1':
                # The wild card plays the best division winner unless they're
                # division rivals; then it plays the second-best one.
                wc = s(4)
                forced = WC_OPPONENT.get(season, {}).get(lg)
                if forced in self.idx:
                    t = self.idx[forced]
                    dws = np.stack([s(1), s(2), s(3)], 1)
                    hit = dws == t
                    # opponent = the forced team when it's a division winner in this sim
                    k = np.where(hit.any(1), hit.argmax(1), np.where(self.div[wc] == self.div[s(1)], 1, 0))
                else:
                    k = np.where(self.div[wc] == self.div[s(1)], 1, 0)
                dws = np.stack([s(1), s(2), s(3)], 1)
                top = dws[sim_ix, k]
                rest2 = np.sort(np.stack([(k + 1) % 3, (k + 2) % 3], 1), axis=1)  # keep seed order
                other1, other2 = dws[sim_ix, rest2[:, 0]], dws[sim_ix, rest2[:, 1]]
                x = play(top, wc, 1, 5)
                y = play(other1, other2, 1, 5)
                champs[lg] = play(x, y, 2, 7)
            elif f == 'wcgame':
                wc = play(s(4), s(5), 1, 1)
                x = play(s(1), wc, 2, 5)
                y = play(s(2), s(3), 2, 5)
                champs[lg] = play(x, y, 3, 7)
            elif f == 'twelve':
                p45 = play(s(4), s(5), 1, 3, kind='wcs')
                p36 = play(s(3), s(6), 1, 3, kind='wcs')
                x = play(s(1), p45, 2, 5)
                y = play(s(2), p36, 2, 5)
                champs[lg] = play(x, y, 3, 7)
            elif f == 'sixteen':
                a18 = play(s(1), s(8), 1, 3, kind='wcs')
                a45 = play(s(4), s(5), 1, 3, kind='wcs')
                a27 = play(s(2), s(7), 1, 3, kind='wcs')
                a36 = play(s(3), s(6), 1, 3, kind='wcs')
                x = play(a18, a45, 2, 5, neutral=True)
                y = play(a27, a36, 2, 5, neutral=True)
                champs[lg] = play(x, y, 3, 7, neutral=True)
        champ = play(champs['AL'], champs['NL'], self.n_rounds, 7, neutral=neutral_late)
        np.add.at(reach[-1], champ, 1)
        reach /= n_sims
        cols = ['playoffs'] + [f'r{k}' for k in range(2, self.n_rounds + 1)] + ['champ']
        rows = np.vstack([reach[0]] + [reach[k] for k in range(2, self.n_rounds + 1)] + [reach[-1]])
        return pd.DataFrame(rows.T, index=self.teams, columns=cols)


def compute(games, ratings_df, league_of, div_of, current_season, schedule=None,
            seasons=None, log=print):
    """games: MLB games (season, date, home, away, home_pts, visitor_pts,
    gametype). ratings_df: (season, date, name, rating). Returns (odds,
    brackets) like DUNCAN's."""
    out, brackets = [], {}
    for season, g in games.groupby('season'):
        season = int(season)
        if seasons is not None and season not in seasons:
            continue
        rsub = ratings_df[ratings_df['season'] == season]
        ratings = {d: dict(zip(x['name'], x['rating'])) for d, x in rsub.groupby('date')}
        if not ratings:
            continue
        sim = SeasonSim(season, g, league_of, div_of, ratings,
                        schedule if season == current_season else None)
        for d in sorted(ratings):
            n = N_SIMS_PLAYOFFS if sim.rs_over(d) else N_SIMS
            o = sim.odds_at(d, n_sims=n)
            if sim.rs_complete and sim.seeds:
                brackets.setdefault(season, {})[d] = (dict(sim.seeds), list(sim.matchups), n)
            o.index.name = 'team'
            o = o.reset_index()
            o['season'] = season
            o['date'] = d
            out.append(o)
        log(f"  {season}: {len(ratings)} snapshots")
    return pd.concat(out, ignore_index=True), brackets


# ── Cached, parallel driver (as DUNCAN) ──────────────────────────────────────
_ENGINE_FILES = ('playoff_sim.py', 'mlb_tiebreak_orders.json', 'mlb_divisions.csv')
_JOB = {}


def _fingerprint(season, games, ratings_df, schedule, current_season):
    h = hashlib.sha256()
    here = _os.path.dirname(_os.path.abspath(__file__))
    for fn in _ENGINE_FILES:
        p = _os.path.join(here, fn)
        if _os.path.exists(p):
            h.update(open(p, 'rb').read())
    g = games[games['season'] == season].sort_values(['date', 'home', 'away']).copy()
    h.update(g.to_csv(index=False).encode())
    r = ratings_df[ratings_df['season'] == season].sort_values(['date', 'name']).copy()
    r['rating'] = r['rating'].round(3)
    h.update(r.to_csv(index=False).encode())
    if season == current_season and schedule is not None:
        h.update(schedule.sort_values(['date', 'home']).to_csv(index=False).encode())
    return h.hexdigest()


def _one(season):
    j = _JOB
    return season, compute(j['games'], j['ratings'], j['league_of'], j['div_of'], j['current'],
                           j['schedule'], seasons={season}, log=lambda *_: None)


def compute_cached(games, ratings_df, league_of, div_of, current_season, schedule=None,
                   cache_dir='title_odds_cache', workers=None, log=print):
    _os.makedirs(cache_dir, exist_ok=True)
    seasons = sorted(int(s) for s in games['season'].unique() if (ratings_df['season'] == s).any())
    results, todo, sigs = {}, [], {}
    for s in seasons:
        sigs[s] = _fingerprint(s, games, ratings_df, schedule, current_season)
        path = _os.path.join(cache_dir, f'{s}.pkl')
        if _os.path.exists(path):
            try:
                sig, payload = pickle.load(open(path, 'rb'))
                if sig == sigs[s]:
                    results[s] = payload
                    continue
            except Exception:
                pass
        todo.append(s)
    log(f"  {len(results)} seasons from cache, computing {len(todo)}: {todo}")
    if todo:
        _JOB.update(games=games, ratings=ratings_df, league_of=league_of, div_of=div_of,
                    current=current_season, schedule=schedule)
        ctx = _mp.get_context('fork')
        with ctx.Pool(workers or _os.cpu_count()) as pool:
            for s, payload in pool.imap_unordered(_one, todo):
                results[s] = payload
                pickle.dump((sigs[s], payload), open(_os.path.join(cache_dir, f'{s}.pkl'), 'wb'))
                log(f"  {s} done")
    odds = pd.concat([results[s][0] for s in seasons], ignore_index=True)
    brackets = {}
    for s in seasons:
        brackets.update(results[s][1])
    return odds, brackets
