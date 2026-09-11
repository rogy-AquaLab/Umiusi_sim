"""Gain search for the classical controller — the tuning that had never been done.

The baseline in tools/classical_control.py beat the learned policy on every condition with gains
that were guessed once and never touched (kp=2.2, kd=0.45, k_v=1.2, no integral). Its one loss was
under domain randomization. So the question this tool answers is narrow and worth answering before
any further RL work:

    how much of the DR gap is just untuned gains?

Every candidate is scored on THREE regimes at once, because they trade against each other and a
gain set that only wins on one is not a controller:

    nom-hold    DR off, v_cmd = 0     — the clean-model attitude number
    dr-hold     DR on,  v_cmd = 0     — model mismatch, the regime that lost
    dr-cruise   DR on,  v_cmd sampled — mismatch WHILE moving (the observer loop is live here)

Seeds are shared across candidates (env.reset(seed=5000+ep)), so the comparison is paired and a
few episodes already separate configs that differ.

    python tools/classical_tune.py --stage ki        # the integral sweep (start here)
    python tools/classical_tune.py --stage pd        # kp/kd around the incumbent
    python tools/classical_tune.py --stage kv        # observer-feedback trust
    python tools/classical_tune.py --stage confirm --episodes 16   # re-measure the finalists
"""

import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from fault_compare import run_classical  # noqa: E402

# (label, dr, cruise, disturb). dist-hold is where the learned policy actually wins
# (ori 0.206 vs 0.332), and hold is where the azimuth singularity bites, so it has to be in here.
REGIMES = [("nom-hold", False, False, False), ("dr-hold", True, False, False),
           ("dist-hold", True, False, True), ("dr-cruise", True, True, False)]

BASE = {"kp": 2.2, "kd": 0.45, "ki": 0.0, "k_v": 1.2}


def _one(job):
    gains, dr, cruise, disturb, episodes = job
    alloc = {k: gains[k] for k in ("prefer_deg", "w_move") if k in gains}
    g = {k: v for k, v in gains.items() if k not in ("prefer_deg", "w_move")}
    return run_classical(episodes, None, 0.0, True, dr, disturb, gains=g, cruise=cruise,
                         alloc=alloc or None)


def evaluate(cands, episodes, workers):
    """Every candidate on every regime. Returns {label: {regime: metrics}}."""
    jobs = [(g, dr, cr, ds, episodes) for g in cands.values() for _, dr, cr, ds in REGIMES]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_one, jobs))
    n = len(REGIMES)
    return {name: {REGIMES[i][0]: out[k * n + i] for i in range(n)}
            for k, name in enumerate(cands)}


def _sweep(name, axis, values, base, extra=None):
    """Candidates varying `axis` over `values`, everything else at `base`."""
    c = {}
    for v in values:
        g = dict(base, **{axis: v}, **(extra or {}))
        c[f"{name}={v:g}"] = g
    return c


def report(res, key="ori"):
    cols = [r[0] for r in REGIMES]
    print(f"{'候補':<18}" + "".join(f"{c:>24}" for c in cols))
    print(f"{'':<18}" + "".join(f"{'ori   横流れ    esc':>24}" for _ in cols))
    for name, per in res.items():
        line = f"{name:<18}"
        for c in cols:
            m = per[c]
            line += f"{m['ori']:>9.3f}{m['drift']:>8.4f}{m['esc']:>7.3f}"
        print(line)
    # ranking by the regime that was losing, with the clean regime shown so a win that only
    # exists under mismatch is visible as such
    order = sorted(res, key=lambda n: res[n]["dr-hold"][key])
    print(f"\n  dr-hold {key} 昇順: " + ", ".join(f"{n}({res[n]['dr-hold'][key]:.3f})" for n in order))


STAGES = {
    "ki":      ("ki", [0.0, 0.3, 0.8, 1.5, 3.0]),
    "pd":      (None, None),   # 2-D, handled below
    "fine":    (None, None),   # 2-D, the survivors of `pd` at more episodes
    "kv":      ("k_v", [0.0, 0.4, 0.8, 1.2, 1.8]),
}

PD_GRID = {"pd": ([0.4, 0.7, 1.0, 1.4, 2.2], [0.3, 0.45, 0.7]),
           "fine": ([0.7, 1.0, 1.4], [0.25, 0.35, 0.45])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="ki", choices=[*STAGES, "confirm", "alloc"])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--kp", type=float, default=BASE["kp"])
    ap.add_argument("--kd", type=float, default=BASE["kd"])
    ap.add_argument("--ki", type=float, default=BASE["ki"])
    ap.add_argument("--k-v", type=float, default=BASE["k_v"])
    args = ap.parse_args()
    base = {"kp": args.kp, "kd": args.kd, "ki": args.ki, "k_v": args.k_v}

    if args.stage in PD_GRID:
        cands = {}
        # Deliberately spanning BELOW the incumbent kp=2.2: the first pass was monotone in kp
        # (2.2 > 3.5 > 5.0 on every regime). That is the signature of a loop whose gain margin is
        # already spent — here on the thrust-curve exponent, which tools/dr_ablation.py measured as
        # the single biggest DR cost (3.5x) and which enters as a MULTIPLICATIVE gain error.
        for kp, kd in itertools.product(*PD_GRID[args.stage]):
            cands[f"kp={kp:g},kd={kd:g}"] = dict(base, kp=kp, kd=kd)
    elif args.stage == "alloc":
        # does killing the azimuth-singularity chatter (tools/alloc_rate_audit.py) actually buy
        # attitude, or only smoother servos?
        cands = {"現行(minnorm)": dict(base)}
        for pref in (45.0, 60.0, 75.0):
            for wm in (1.0, 3.0):
                cands[f"回避{pref:g} w{wm:g}"] = dict(base, prefer_deg=pref, w_move=wm)
    elif args.stage == "confirm":
        # the incumbent is the ORIGINAL controller: PD only, gains in mode units (no cap norm)
        cands = {"incumbent": dict(BASE, cap_norm=False), "tuned": base}
    else:
        axis, values = STAGES[args.stage]
        cands = _sweep(axis, axis, values, base)

    print(f"stage={args.stage}  episodes={args.episodes}  base={base}")
    report(evaluate(cands, args.episodes, args.workers))


if __name__ == "__main__":
    main()
