"""Calibrate, validate and apply the gravity model; writes results/results.json and figures/validation.png.

Two specifications are compared:
  v1 all flows: intrazonal trips included, with an approximate intrazonal distance.
  v2 inter-zonal: intrazonal trips held at observed values; only flows between districts are modelled.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from model import deterrence, furness, mean_trip_length, r2, srmse

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, FIG = ROOT / "data" / "processed", ROOT / "results", ROOT / "figures"
CORE = ["Liverpool", "Manchester", "Leeds", "Bradford", "Sheffield",
        "Kingston upon Hull, City of", "Newcastle upon Tyne", "York"]
CUT = 0.20  # scenario: 20% cut in effective distance between core-city pairs
BRACKETS = {"exp": (0.01, 0.4), "power": (0.1, 6.0)}


def fit(T, D, form, mask):
    """Calibrate beta so modelled mean trip length (over masked cells) matches observed."""
    Tm = T * mask
    O, Dj = Tm.sum(1), Tm.sum(0)
    c_obs = mean_trip_length(Tm, D)

    def model(beta, DD=D):
        return furness(O, Dj, deterrence(DD, beta, form) * mask)[0]

    beta = brentq(lambda b: mean_trip_length(model(b), D) - c_obs, *BRACKETS[form], xtol=1e-8)
    return beta, model


def tld(T, D, bins):
    h, _ = np.histogram(D.ravel(), bins=bins, weights=T.ravel())
    return h / h.sum()


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    names = pd.read_csv(PROC / "zones.csv").LAD21NM.tolist()
    T, D = np.load(PROC / "T_obs.npy"), np.load(PROC / "D_km.npy")
    n = len(T)
    off = ~np.eye(n, dtype=bool)
    out = {"zones": n, "commuters": float(T.sum()), "intrazonal_share": float(T.trace() / T.sum()),
           "obs_mean_km_interzonal": mean_trip_length(T * off, D)}

    specs = {"v1_all_flows": np.ones((n, n), dtype=bool), "v2_interzonal": off}
    fits, models = {}, {}
    for spec, mask in specs.items():
        for form in ["exp", "power"]:
            beta, model = fit(T, D, form, mask)
            P = model(beta)
            key = f"{spec}/{form}"
            fits[key] = dict(beta=beta, r2_interzonal=r2(T[off], P[off]), srmse_interzonal=srmse(T[off], P[off]))
            models[key] = (beta, model)
    out["fits"] = fits
    best = max((k for k in fits if k.startswith("v2")), key=lambda k: fits[k]["r2_interzonal"])
    out["chosen"] = best
    beta, model = models[best]
    base = model(beta)

    bins = [0, 10, 20, 30, 40, 60, 80, 120, 250]
    out["tld_bins_km"] = bins
    out["tld_obs"] = tld(T * off, D, bins).round(4).tolist()
    out["tld_mod"] = tld(base, D, bins).round(4).tolist()

    resid = sorted(((names[i], names[j], int(T[i, j]), int(base[i, j])) for i in range(n) for j in range(n) if i != j),
                   key=lambda r: -abs(r[2] - r[3]))
    out["largest_residuals"] = resid[:8]

    core = [names.index(c) for c in CORE]
    D2 = D.copy()
    for i in core:
        for j in core:
            if i != j:
                D2[i, j] *= 1 - CUT
    scen = model(beta, D2)
    pairs = [(i, j) for i in core for j in core if i != j]
    cb, cs = sum(base[p] for p in pairs), sum(scen[p] for p in pairs)
    L, M = names.index("Leeds"), names.index("Manchester")
    out["scenario"] = dict(
        cut=CUT, core_cities=CORE, core_pairs_obs=float(sum(T[p] for p in pairs)),
        core_pairs_base=cb, core_pairs_scen=cs, core_pairs_pct=100 * (cs / cb - 1),
        leeds_to_manchester_obs=float(T[L, M]), leeds_to_manchester_base=float(base[L, M]),
        leeds_to_manchester_scen=float(scen[L, M]),
        interzonal_total_unchanged=bool(np.isclose(scen.sum(), base.sum())),
        trips_diverted_from_other_pairs=float((base - scen)[base > scen].sum()),
    )
    # same scenario under the rejected v1 specification, to show how the choice changes the answer
    b1, m1 = models["v1_all_flows/" + best.split("/")[1]]
    s1b, s1s = m1(b1), m1(b1, D2)
    out["scenario"]["v1_core_pairs_pct"] = 100 * (sum(s1s[p] for p in pairs) / sum(s1b[p] for p in pairs) - 1)
    (RES / "results.json").write_text(json.dumps(out, indent=2, default=float))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(T[off] + 1, base[off] + 1, ".", ms=3, alpha=0.5)
    lim = [1, T[off].max() * 1.5]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlabel="Observed commuters + 1", ylabel="Modelled commuters + 1",
              title=f"Flows between districts (R² = {fits[best]['r2_interzonal']:.2f})")
    x = np.arange(len(bins) - 1)
    ax[1].bar(x - 0.2, out["tld_obs"], 0.4, label="Observed")
    ax[1].bar(x + 0.2, out["tld_mod"], 0.4, label="Modelled")
    ax[1].set_xticks(x, [f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])], rotation=45)
    ax[1].set(xlabel="Distance band (km)", ylabel="Share of commuters", title="Trip length distribution")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "validation.png", dpi=150)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
