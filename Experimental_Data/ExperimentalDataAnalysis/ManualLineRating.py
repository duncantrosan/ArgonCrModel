# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:43:29 2026

@author: dptro
"""

# -*- coding: utf-8 -*-
"""
Manual line-rating tool for the Ar/N2 analysis.

Shows each measured line one at a time (zoomed to +-half_window_nm) and asks you
for a 1-5 score in the console.  Scores are written to manual_ratings.csv next to
results.pkl and can be read back by the main script as an independent second check
against the automatic contamination call.

Suggested scale (it's just an int you store):
    5 = clean, confident real line      2 = probably contaminated / blended
    4 = good, minor doubt               1 = spurious / junk
    3 = unsure

The rating is deliberately BLIND to the automatic contamination flag so your
judgement stays independent.  (Set show_auto=True in CONFIG if you'd rather see it.)

Spyder: figures must pop up in a window for interactive rating, so run
    %matplotlib qt
once in the console first, then press F5.  Per-line console commands:
    1-5    rating (optionally add a note:  '2 N2 forest under it')
    Enter  skip (leave unrated)     b  back     q  save & quit
"""
import os, csv, importlib
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt

# Project root (ArgonCrModel folder), found relative to this file
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CONFIG = dict(
    module="Argon_Nitrogen_Mix_Analysis_V2",   # main analysis module (for unpickle + load_results)
    results=os.path.join(os.path.join(ROOT_DIR, "Experimental_Data", "Output", "ArN2_out"), "results.pkl"),
    out_csv=None,                    # None -> manual_ratings.csv beside results.pkl
    species="Ar I",                   # "Ar I", "N I", or "all"
    statuses=("ok", "weak", "blend", "saturated"),
    half_window_nm=2.0,
    show_pure=False,                 # overlay pure-Ar reference in the wide panel
    show_auto=False,                 # show the automatic contamination flag (breaks blindness)
    resume=True,                     # skip lines already rated in the csv
)


# --- stable identity so the main script can join ratings back onto res ------
def rating_key(species, lower, upper, wl_air):
    return f"{species}|{lower}->{upper}|{float(wl_air):.3f}"


def _default_csv(cfg):
    return cfg["out_csv"] or os.path.join(os.path.dirname(cfg["results"]), "manual_ratings.csv")


def load_manual_ratings(path):
    """csv -> {key: {'rating': int|None, 'note': str}}.  Import this in the main script."""
    out = {}
    if path and os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                r = row.get("rating", "")
                out[row["key"]] = dict(rating=int(r) if str(r).strip() else None,
                                       note=row.get("note", ""))
    return out


def _write_csv(path, ratings, meta):
    """ratings: {key:{rating,note,rated_utc}}; meta: {key:{species,wl_air,lower,upper}}."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "species", "wl_air", "lower", "upper", "rating", "note", "rated_utc"])
        for k, v in ratings.items():
            m = meta.get(k, {})
            w.writerow([k, m.get("species", ""), m.get("wl_air", ""), m.get("lower", ""),
                        m.get("upper", ""), "" if v["rating"] is None else v["rating"],
                        v.get("note", ""), v.get("rated_utc", "")])


def _draw(ax_pair, r, sp_mix, sp_pure, hw, show_pure, show_auto):
    axm, axr = ax_pair
    for ax in (axm, axr):
        ax.clear()
    c = r.centre_fit if np.isfinite(r.centre_fit) else r.wl_air
    lo, hi = c - hw, c + hw
    xm, ym = sp_mix.window(lo, hi)
    axm.plot(xm, ym, color="crimson", lw=0.9, label="mix")
    if show_pure:
        xp, yp = sp_pure.window(lo, hi)
        axm.plot(xp, yp, color="0.5", lw=0.8, label="pure")
        axm.legend(fontsize=7, loc="upper right")
    axm.axvline(r.wl_air, color="k", ls=":", lw=0.7)         # catalogue position
    axm.axvline(c, color="tab:blue", ls="--", lw=0.7)        # fitted centre
    axm.set_xlim(lo, hi); axm.set_ylabel("intensity")
    auto = f"   [{getattr(r,'contamination','')}]" if show_auto else ""
    axm.set_title(f"{r.wl_air:.2f} nm   {r.lower}->{r.upper}   {r.status}, SNR {r.snr:.0f}{auto}",
                  fontsize=10)
    cl, ch = c - 0.15, c + 0.15                              # tight core for shape/asymmetry
    xz, yz = sp_mix.window(cl, ch)
    axr.plot(xz, yz, color="crimson", lw=1.0, marker=".", ms=3)
    axr.axvline(c, color="tab:blue", ls="--", lw=0.7)
    axr.set_xlim(cl, ch); axr.set_title("core (+-0.15 nm)", fontsize=9)


def rate_lines(cfg=CONFIG):
    mod = importlib.import_module(cfg["module"])   # provides load_results + Spectrum for unpickle
    R = mod.load_results(cfg["results"])
    res, sp_mix, sp_pure = R["res_mix"], R["sp_mix"], R["sp_pure"]
    hw = cfg["half_window_nm"]

    sel = res[res.status.isin(cfg["statuses"])]
    if cfg["species"] != "all":
        sel = sel[sel.species == cfg["species"]]
    sel = sel.sort_values("wl_air").reset_index(drop=True)

    csv_path = _default_csv(cfg)
    ratings = load_manual_ratings(csv_path)
    meta = {}
    for _, r in sel.iterrows():
        k = rating_key(r.species, r.lower, r.upper, r.wl_air)
        meta[k] = dict(species=r.species, wl_air=round(float(r.wl_air), 3), lower=r.lower, upper=r.upper)

    order = list(range(len(sel)))
    if cfg["resume"]:
        order = [i for i in order
                 if ratings.get(rating_key(sel.iloc[i].species, sel.iloc[i].lower,
                                           sel.iloc[i].upper, sel.iloc[i].wl_air), {}).get("rating") is None]
    print(f"{len(sel)} {cfg['species']} lines, {len(order)} to rate "
          f"({len(sel)-len(order)} already scored).  1-5 / Enter skip / b back / q save+quit")

    plt.close("all")
    fig, ax_pair = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[3, 1]))
    plt.show(block=False)

    pos = 0
    while pos < len(order):
        i = order[pos]
        r = sel.iloc[i]
        k = rating_key(r.species, r.lower, r.upper, r.wl_air)
        _draw(ax_pair, r, sp_mix, sp_pure, hw, cfg["show_pure"], cfg["show_auto"])
        fig.suptitle(f"[{pos+1}/{len(order)}]  {r.wl_air:.2f} nm", fontsize=11)
        fig.tight_layout(); fig.canvas.draw_idle(); plt.pause(0.05)

        prev = ratings.get(k, {}).get("rating")
        ans = input(f"[{pos+1}/{len(order)}] {r.wl_air:.2f} {r.lower}->{r.upper}"
                    f"{'' if prev is None else f' (was {prev})'}  1-5/Enter/b/q: ").strip()
        if ans == "":
            pos += 1; continue
        if ans.lower() == "q":
            break
        if ans.lower() == "b":
            pos = max(0, pos - 1); continue
        tok = ans.split(maxsplit=1)
        if tok[0] in ("1", "2", "3", "4", "5"):
            ratings[k] = dict(rating=int(tok[0]), note=(tok[1] if len(tok) > 1 else ""),
                              rated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            _write_csv(csv_path, ratings, meta)   # save after every entry (crash-safe)
            pos += 1
        else:
            print("  ? enter 1-5, or Enter / b / q")

    _write_csv(csv_path, ratings, meta)
    done = sum(v["rating"] is not None for v in ratings.values())
    print(f"saved {done} ratings -> {csv_path}")
    return csv_path


if __name__ == "__main__":
    rate_lines(CONFIG)