# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:37:35 2026

@author: dptro
"""

# -*- coding: utf-8 -*-
"""
Post-processing / eyeball tool for the Ar-N2 line analysis.
Reads the pickled RESULTS and shows each measured line zoomed to +-2 nm so the
fitted centre / status can be checked against the spectrum by eye.
Nothing is saved.  We'll build on this later.

Run from Spyder: press F5.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import Argon_Nitrogen_Mix_Analysis_V2 as arn        # needed so the Spectrum dataclass unpickles
from Argon_Nitrogen_Mix_Analysis_V2 import load_results


# Project root (ArgonCrModel folder), found relative to this file
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MainPath = os.path.join(ROOT_DIR, "Experimental_Data", "Output", "ArN2_out")
CONFIG = dict(
    
    results=os.path.join(MainPath, "results.pkl"),
    species="Ar I",                     # which lines to show
    statuses=("ok", "weak", "blend", "saturated"),   # 'measured' = detected
    half_window_nm=2.0,                 # +- this around the fitted centre
    per_page=12,                        # panels per figure (rows*cols)
    ncols=4,
    show_pure=False,                     # overlay the pure-Ar reference
)

# def ActinometryModel:
#     R = load_results(cfg["results"])
#     res, sp_mix, sp_pure = R["res_mix"], R["sp_mix"], R["sp_pure"]
#     hw = cfg["half_window_nm"]
    
    
    
def show_measured_lines(cfg=CONFIG):
    R = load_results(cfg["results"])
    res, sp_mix, sp_pure = R["res_mix"], R["sp_mix"], R["sp_pure"]
    hw = cfg["half_window_nm"]

    sel = res[(res.species == cfg["species"]) & res.status.isin(cfg["statuses"])]
    sel = sel.sort_values("wl_air").reset_index(drop=True)
    print(f"{len(sel)} measured {cfg['species']} lines "
          f"({', '.join(f'{s}:{(sel.status==s).sum()}' for s in cfg['statuses'])})")

    ncols = cfg["ncols"]; nrows = int(np.ceil(cfg["per_page"] / ncols))
    for p0 in range(0, len(sel), cfg["per_page"]):
        chunk = sel.iloc[p0:p0 + cfg["per_page"]]
        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows))
        for ax, (_, r) in zip(axs.flat, chunk.iterrows()):
            c = r.centre_fit if np.isfinite(r.centre_fit) else r.wl_air
            lo, hi = c - hw, c + hw
            xm, ym = sp_mix.window(lo, hi)
            ax.plot(xm, ym, color="crimson", lw=0.8, label="mix")
            if cfg["show_pure"]:
                xp, yp = sp_pure.window(lo, hi)
                ax.plot(xp, yp, color="0.5", lw=0.7, label="pure")
            ax.axvline(r.wl_air, color="k", ls=":", lw=0.6)        # catalogue
            ax.axvline(c, color="tab:blue", ls="--", lw=0.6)       # fitted centre
            ax.set_xlim(lo, hi)
            ax.set_title(f"{r.wl_air:.2f}  {r.lower}->{r.upper}\n"
                         f"{r.status}, SNR {r.snr:.0f}, {getattr(r,'contamination','')}",
                         fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axs.flat[len(chunk):]:
            ax.axis("off")
        axs.flat[0].legend(fontsize=6, loc="upper right")
        fig.suptitle(f"{cfg['species']} measured lines  {p0+1}-{p0+len(chunk)} of {len(sel)}",
                     fontsize=10)
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_measured_lines(CONFIG)