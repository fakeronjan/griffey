"""One-off: recover how past MLB standings ties were actually broken.

For each season whose simulated bracket doesn't reproduce the real playoff
games, try every ordering of the teams tied on win% at the end of the
regular season (within tie groups only) and report the first ordering whose
bracket consumes every real playoff game and crowns the real champion.
Writes mlb_tiebreak_orders.json (read by playoff_sim.TIEBREAK_WINNERS).
"""
import itertools, json, sys
import numpy as np
exec(open(sys.argv[1]).read().split("if __name__ == '__main__':")[0])
out = dict(PS.TIEBREAK_WINNERS)
for s in [int(x) for x in sys.argv[2].split(',')]:
    rsub = r[r.season == s]
    ratings = {d: dict(zip(x.name, x.rating)) for d, x in rsub.groupby('date')}
    sim = PS.SeasonSim(s, g[g.season == s], _team_league, div_of, ratings)
    last = sorted(ratings)[-1]
    done = sim.rs[sim.rs.home_pts.notna()]
    w = np.zeros(len(sim.teams)); gp = np.zeros(len(sim.teams))
    for x in done.itertuples():
        gp[x.h] += 1; gp[x.a] += 1; w[x.h if x.home_pts > x.visitor_pts else x.a] += 1
    pct = w / gp
    groups = [[sim.teams[i] for i in np.where(pct == p)[0]] for p in np.unique(pct) if (pct == p).sum() > 1]
    # Only ties among plausible playoff teams matter: keep groups within the top 12 by pct.
    cut = np.sort(pct)[::-1][min(len(pct) - 1, 17)]
    groups = [grp for grp in groups if pct[sim.idx[grp[0]]] >= cut]
    options = [list(itertools.permutations(grp)) for grp in groups]
    n = int(np.prod([len(o) for o in options])) if options else 1
    found = None
    for combo in itertools.product(*options):
        PS.TIEBREAK_WINNERS[s] = [t for grp in combo for t in grp]
        o = sim.odds_at(last, n_sims=20)
        if sim.used_actual == len(sim.ps) and o.champ.max() == 1:
            found = PS.TIEBREAK_WINNERS[s]
            break
    PS.TIEBREAK_WINNERS.pop(s, None)
    if found:
        out[s] = found
    print(f"{s}: {len(groups)} tie groups, {n} orderings -> {'found' if found else 'NONE'}", flush=True)
json.dump({str(k): v for k, v in sorted(out.items())}, open('mlb_tiebreak_orders.json', 'w'), indent=1)
