"""Ejecuta R réplicas de un escenario en paralelo y guarda los resultados."""
import os, sys, json, time
import numpy as np
from joblib import Parallel, delayed
from scenarios import SCEN
from truth import population
from gen import sample_calls
from est import estimate

C_STAR_CACHE = "c_star.json"


def c_star_for(name):
    try:
        cache = json.load(open(C_STAR_CACHE))
    except FileNotFoundError:
        cache = {}
    if name not in cache:
        p, _, _ = SCEN[name]
        cache[name] = float(population(p, n=1_500_000, seed=11)["c_star"])
        json.dump(cache, open(C_STAR_CACHE, "w"), indent=1)
    return cache[name]


def one(name, seed, cs):
    p, useC, n = SCEN[name]
    d = sample_calls(n, p, cs, np.random.default_rng(seed))
    return estimate(d, useC, spec=(name in ("base", "n_pequeno")))


if __name__ == "__main__":
    name, R = sys.argv[1], int(sys.argv[2])
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    cs = c_star_for(name)
    t = time.time()
    res = Parallel(n_jobs=int(os.environ.get("N_JOBS", "2")))(delayed(one)(name, seed0 + r, cs) for r in range(R))
    res = [{k: float(v) for k, v in r.items()} for r in res]
    json.dump(res, open(f"res_{name}_{R}.json", "w"))
    print(f"{name}: {R} réplicas en {time.time()-t:.0f} s")
