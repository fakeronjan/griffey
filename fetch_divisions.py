"""One-off: MLB league + division per team-season (1969+) from the MLB Stats
API -> mlb_divisions.csv, keyed by GRIFFEY's franchise names.

The API names teams as they were that season (Cleveland Indians, Montreal
Expos); GRIFFEY names each franchise once. Match on exact name first, then
on the API's franchise id via any season where the name matched. Only used
for playoff seeding in the title-odds sim. Rerun to add a new season, or let
generate_data.py fall back to the latest season's alignment.
"""
import time
import pandas as pd
import requests

games = pd.read_csv('all_mlb_games.csv', usecols=['season', 'home_team_name', 'visitor_team_name'])
names_by_season = {s: set(g.home_team_name) | set(g.visitor_team_name) for s, g in games.groupby('season')}
api = {}
for s in range(1969, int(games.season.max()) + 1):
    teams = requests.get('https://statsapi.mlb.com/api/v1/teams',
                         params={'sportId': 1, 'season': s}, timeout=30).json()['teams']
    api[s] = [(t['id'], t['name'], t['league']['name'], t['division']['name']) for t in teams
              if t.get('league', {}).get('name') and t.get('division', {}).get('name')]
    time.sleep(0.3)
# franchise id -> every GRIFFEY name it ever matched exactly (the A's are
# 'Oakland Athletics' and later 'Las Vegas Athletics')
id_to_names = {}
for s, rows in api.items():
    for tid, name, *_ in rows:
        if name in names_by_season.get(s, ()):
            id_to_names.setdefault(tid, set()).add(name)
all_names = set().union(*names_by_season.values())
out, missing = [], []
for s, rows in api.items():
    for tid, name, lg, div in rows:
        here = names_by_season.get(s, set())
        cands = id_to_names.get(tid, set()) & here
        if name in here:
            team = name
        elif cands:
            team = next(iter(cands))
        else:  # franchise renamed with no exact match anywhere: GRIFFEY names absent from the API that season
            spare = sorted(here - {n for r in rows for n in [r[1]]} - set().union(*[id_to_names.get(r[0], set()) for r in rows if r[0] != tid]))
            team = spare[0] if len(spare) == 1 else None
        if team is None:
            missing.append((s, name)); continue
        out.append((s, team, 'AL' if lg.startswith('American') else 'NL', div.split(' League ')[-1]))
df = pd.DataFrame(out, columns=['season', 'team', 'league', 'division'])
df.to_csv('mlb_divisions.csv', index=False)
gaps = [(s, sorted(n - set(df[df.season == s].team))) for s, n in names_by_season.items() if s >= 1969]
print(len(df), 'team-seasons; API names unmatched:', missing[:5], '; GRIFFEY teams without a division:', [x for x in gaps if x[1]][:5])
