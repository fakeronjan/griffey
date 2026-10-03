"""Real MLB postseason seeds for the current season -> mlb_playoff_seeds.json
(read by playoff_sim.REAL_SEEDS). Runs daily; does nothing until the regular
season is over.

Source: the season's Wikipedia postseason bracket (plain-text team names,
American League in the first half of each round), checked against the
standings and the real postseason games before anything is written.
"""
# ── Shared core (same in every fleet site's playoff_seeds.py) ─────────────────
# Once a regular season is over, the playoff sims seed from the real seeds
# (playoff_sim.REAL_SEEDS), never their own tiebreak estimate. run() fetches
# them, checks them and writes them; seeds missing or failing a check give a
# warning for GRACE_DAYS after the regular season, then the run fails.
#
# Checks: every seed filled once and every team known; within each seeding
# tier, seeds follow the standings (a source can only differ from our own
# order where records are level); and once real playoff games exist, every
# series between two seeded teams of the same group was opened at the
# better seed.
import json
import os
import re
import urllib.parse
import urllib.request

import pandas as pd

GRACE_DAYS = 2
# Wikipedia asks for a descriptive agent with contact details; ESPN and the
# league APIs refuse agents with an email in them, so they get a plain one.
_UA_WIKI = {'User-Agent': 'fakeronjan-sports/1.0 (rjsikdar@gmail.com)'}
_UA = {'User-Agent': 'Mozilla/5.0'}


def get_json(url):
    ua = _UA_WIKI if 'wikipedia.org' in url else _UA
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=30))


def wikitext(title):
    d = get_json('https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(
        {'action': 'parse', 'page': title, 'prop': 'wikitext', 'format': 'json', 'formatversion': 2,
         'redirects': 1}))
    return d['parse']['wikitext'] if 'parse' in d else None


def bracket_links(text):
    """{linked page name: seed label} from a Wikipedia bracket template's
    RDn-seedXX / RDn-teamXX pairs (first label seen per team)."""
    seeds = {}
    sd = {(m.group(1), int(m.group(2))): m.group(3)
          for m in re.finditer(r'\|\s*RD(\d+)-seed0*(\d+)\s*=\s*([^\n|]*)', text)}
    for m in re.finditer(r'\|\s*RD(\d+)-team0*(\d+)\s*=\s*([^\n]*)', text):
        lab = re.sub(r'[^\w]', '', sd.get((m.group(1), int(m.group(2))), ''))
        link = re.search(r'\[\[([^\]|]+)', m.group(3))
        if not lab or not link:
            continue
        name = re.sub(r'^\d{4}(?:[–-]\d{2,4})?\s+', '', link.group(1).strip())
        seeds.setdefault(re.sub(r'\s+season$', '', name), lab)
    return seeds


def _problem(seeds, teams, tiers, rec, ps_games):
    """seeds: {group: [team, ...]} best first. tiers: [(group, [seed numbers])].
    rec: {team: standings value, higher = better}. ps_games: [(home, away,
    ...)] in date order. Returns a description of the first failed check, or None."""
    seen = [t for lst in seeds.values() for t in lst]
    if any(t not in teams for t in seen) or len(set(seen)) != len(seen):
        return f"unknown or repeated teams: {seen}"
    for g, nums in tiers:
        lst = seeds.get(g, [])
        if len(lst) < max(nums):
            return f"{g} has {len(lst)} seeds, expected {max(nums)}"
        vals = [rec.get(lst[k - 1], 0) for k in nums]
        if any(a < b - 1e-9 for a, b in zip(vals, vals[1:])):
            return f"{g} seeds {nums} don't follow the standings: {[lst[k - 1] for k in nums]} {vals}"
    rank = {t: (g, k) for g, lst in seeds.items() for k, t in enumerate(lst)}
    first = {}
    for x in ps_games:
        first.setdefault(frozenset(x[:2]), x[0])
    for pair, host in first.items():
        x, y = sorted(pair)
        if x in rank and y in rank and rank[x][0] == rank[y][0]:
            if host != min(pair, key=lambda t: rank[t][1]):
                return f"{' vs '.join(pair)} opened at {host}, not the better seed"
    return None


def run(path, season, rs_end, fetch, teams, tiers, rec, ps_games, today=None, label=''):
    """rs_end: last regular-season date (None while it's still going).
    fetch(stored) -> seeds dict (may refine stored ones, e.g. from play-in
    games) or None when the source doesn't have them yet."""
    if rs_end is None:
        return None
    data = json.load(open(path)) if os.path.exists(path) else {}
    stored = data.get(str(season))
    problem = None
    try:
        seeds = fetch(stored)
        if seeds is None:
            problem = 'the source has no seeds yet'
        else:
            problem = _problem(seeds, teams, tiers, rec, ps_games)
    except Exception as e:                       # network, parsing
        seeds, problem = None, f'{type(e).__name__}: {e}'
    if problem is None:
        if seeds != stored:
            data[str(season)] = seeds
            json.dump(dict(sorted(data.items())), open(path, 'w'), indent=1)
            print(f"  {season} {label}playoff seeds -> {path}: {seeds}")
        return seeds
    if stored is not None and _problem(stored, teams, tiers, rec, ps_games) is None:
        print(f"::warning::{season} {label}seed refresh failed ({problem}); keeping the stored seeds")
        return stored
    days = ((today or pd.Timestamp.now()).normalize() - pd.Timestamp(rs_end).normalize()).days
    msg = f"{season} {label}playoff seeds not usable yet: {problem}"
    if days > GRACE_DAYS:
        raise RuntimeError(msg + f" ({days} days after the regular season)")
    print(f"::warning::{msg}")
    return None


# ── MLB ──────────────────────────────────────────────────────────────────────
SEEDS_JSON = 'mlb_playoff_seeds.json'
N_SEEDS = 6                       # per league since 2022
POSTSEASON = ('wildcard', 'divisionseries', 'lcs', 'worldseries')
# Wikipedia's short names -> the start of ours
SUBS = [('ny ', 'new york '), ('la ', 'los angeles '), ('chi ', 'chicago '),
        ('anaheim', 'los angeles angels'), ('florida', 'miami')]


def season_state(games_csv='all_mlb_games.csv', sched_csv='mlb_schedule.csv'):
    """(season, rs_end or None, teams, win% by team, postseason games as
    (home, away, winner) in date order)."""
    g = pd.read_csv(games_csv, low_memory=False)
    season = int(g['season'].max())
    g = g[(g['season'] == season) & g['home_pts'].notna()]
    g = g.assign(date=pd.to_datetime(g['date_game'], format='mixed')).sort_values('date', kind='stable')
    rs = g[g['gametype'] == 'regular']
    rs = rs[rs['home_pts'] != rs['visitor_pts']]           # tie games don't count
    sched = pd.read_csv(sched_csv)
    left = sched[pd.to_datetime(sched['date_game']).dt.year == season] if len(sched) else sched
    teams = set(rs['home_team_name']) | set(rs['visitor_team_name'])
    rs_end = rs['date'].max() if len(rs) and not len(left) else None
    played = pd.concat([rs['home_team_name'], rs['visitor_team_name']]).value_counts()
    wins = pd.concat([rs.loc[rs['home_pts'] > rs['visitor_pts'], 'home_team_name'],
                      rs.loc[rs['visitor_pts'] > rs['home_pts'], 'visitor_team_name']]).value_counts()
    rec = {t: wins.get(t, 0) / max(played.get(t, 1), 1) for t in teams}
    po = g[g['gametype'].isin(POSTSEASON)]
    ps = [(h, a, h if hp > vp else a) for h, a, hp, vp in
          po[['home_team_name', 'visitor_team_name', 'home_pts', 'visitor_pts']].itertuples(index=False)]
    return season, rs_end, teams, rec, ps


def _plain(v):
    v = re.sub(r'\{\{nowrap\|([^}]*)\}\}', r'\1', v)
    v = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r'\1', v)
    return re.sub(r"'''?|''|<[^>]*>|\{\{[^}]*\}\}|\*", '', v).strip()


def bracket_plain(text):
    """[(league, seed, team text)] from the bracket template, World Series
    left out; each round's entries are AL then NL."""
    i = text.find('TeamBracket')
    if i < 0:
        return []
    j = text.find('}}<section end', i)
    body = text[i: j if j > 0 else i + 20000]
    sd = {(int(m.group(1)), int(m.group(2))): _plain(m.group(3))
          for m in re.finditer(r'\|\s*RD(\d+)-seed0*(\d+)\s*=\s*([^\n|]*)', body)}
    tm = {(int(m.group(1)), int(m.group(2))): _plain(m.group(3))
          for m in re.finditer(r'\|\s*RD(\d+)-team0*(\d+)\s*=\s*([^\n]*)', body)}
    last = max((r for r, _ in tm), default=0)
    out = []
    for r in sorted({r for r, _ in tm}):
        if r == last:
            continue
        ks = sorted(k for rr, k in tm if rr == r)
        for n, k in enumerate(ks):
            lab = sd.get((r, k), '')
            if lab.isdigit() and tm[(r, k)]:
                out.append(('AL' if n < len(ks) / 2 else 'NL', int(lab), tm[(r, k)]))
    return out


def _norm(s):
    return re.sub(r'[^a-z ]', '', s.lower().replace('.', '')).strip()


def wiki_seeds(season, teams, league):
    text = wikitext(f'{season} Major League Baseball postseason')
    if not text:
        return None
    by = {'AL': {}, 'NL': {}}
    for lg, sd, txt in bracket_plain(text):
        t = _norm(txt)
        for a, b in SUBS:
            if t.startswith(a):
                t = b + t[len(a):]
                break
        hit = [n for n in teams if league.get(n) == lg
               and (_norm(n).startswith(t) or set(t.split()) <= set(_norm(n).split()))]
        if len(hit) != 1:
            raise ValueError(f"can't match {txt!r} to one {lg} team: {hit}")
        if by[lg].setdefault(sd, hit[0]) != hit[0]:
            raise ValueError(f"{lg} seed {sd} is both {by[lg][sd]} and {hit[0]}")
    if any(sorted(v) != list(range(1, N_SEEDS + 1)) for v in by.values()):
        return None                                     # bracket not filled in yet
    return {lg: [v[k] for k in sorted(v)] for lg, v in by.items()}


def main(today=None):
    season, rs_end, teams, rec, ps = season_state()
    dv = pd.read_csv('mlb_divisions.csv')
    dv = dv[dv['season'] == (season if (dv['season'] == season).any() else dv['season'].max())]
    league = dict(zip(dv['team'], dv['league']))
    tiers = [(lg, nums) for lg in ('AL', 'NL') for nums in ([1, 2, 3], [4, 5, 6])]   # division winners, wild cards
    run(SEEDS_JSON, season, rs_end, lambda stored: wiki_seeds(season, teams, league), teams,
        tiers, rec, ps, today=today, label='MLB ')


if __name__ == '__main__':
    main()
