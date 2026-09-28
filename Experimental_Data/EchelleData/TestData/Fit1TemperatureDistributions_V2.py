import math
from pathlib import Path
import re
import time
import typing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MolecularFitting.Fit_One_Temp_V2 import MolecularFitter
from MolecularFitting.Boltzmann_Fitter import BoltzmannAnalyzer


# ==============================================================================
# LOGGING & TERMINAL FORMATTING ENGINE
# ==============================================================================
class Log:
    CYAN, GREEN, YELLOW, RED = "\033[96m", "\033[92m", "\033[93m", "\033[91m"
    BOLD, GRAY, BLUE, RESET = "\033[1m", "\033[90m", "\033[94m", "\033[0m"

    @classmethod
    def header(cls, text: str):
        print(f"\n{cls.BOLD}{cls.CYAN}{'═'*75}\n {text}\n{'═'*75}{cls.RESET}", flush=True)

    @classmethod
    def subheader(cls, text: str):
        print(f"\n{cls.BOLD}{cls.BLUE}─── {text} ───{cls.RESET}", flush=True)

    @classmethod
    def info(cls, text: str):
        print(f"  {cls.GRAY}ℹ {text}{cls.RESET}", flush=True)

    @classmethod
    def status(cls, band: str, status: str, detail: str):
        badge = {"PASS": f"{cls.GREEN}[PASS]", "SKIP": f"{cls.YELLOW}[SKIP]", "FAIL": f"{cls.RED}[FAIL]"}.get(status, status)
        print(f"  {cls.BOLD}{badge}{cls.RESET} {cls.BOLD}{band:<18}{cls.RESET} │ {detail}", flush=True)


# ==============================================================================
# CONFIGURATION & SETUP
# ==============================================================================
BASE_DIR = Path(__file__).parent.resolve()
RESULTS_BASE_DIR = BASE_DIR / "results_fitting_one_temp"
SPREADS_DIR = RESULTS_BASE_DIR / "fit_spreads_waterfall_3x3"
TRENDS_DIR = RESULTS_BASE_DIR / "temperature_trends_3x3"
BOLTZMANN_DIR = RESULTS_BASE_DIR / "boltzmann_plots_3x3"
SUMMARY_CSV_PATH = RESULTS_BASE_DIR / "fit_summary_report.csv"

RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
SPREADS_DIR.mkdir(parents=True, exist_ok=True)
TRENDS_DIR.mkdir(parents=True, exist_ok=True)
BOLTZMANN_DIR.mkdir(parents=True, exist_ok=True)

SLM_FOLDERS = ["0.5SLM", "1SLM", "2SLM"]
N2_FOLDERS = ["0%N2", "0.25%N2", "0.5%N2"]
N2_LABELS = ["0% N2", "0.25% N2", "0.5% N2"]
GAS_LABELS = ["Pure He", "Pure Ar", "He/Ar Mix"]

SLM_CONFIG = {
    "0.5SLM": {"offset": 0.0, "color": "#1f77b4"},
    "1SLM": {"offset": 0.8, "color": "#2ca02c"},
    "2SLM": {"offset": 1.6, "color": "#d62728"},
}

MIN_RAW_PEAK_COUNTS, MIN_SNR, MAX_SHIFT_NM = 1500.0, 8.0, 0.20
R2_MIN, EDGE_TOL_FRAC = 0.50, 0.02

# TARGET_BANDS format: (Species Library Key, Display Name, ROI Range (nm), Fit_Tvib)
TARGET_BANDS = [
    # NO Gamma Individual Sub-Bands (1D Fit: Trot only)
    ("14N-16O", "NO Gamma (1-0)", (212.0, 215.7), False),
    ("14N-16O", "NO Gamma (0-0)", (224.0, 227.5), False),
    ("14N-16O", "NO Gamma (0-1)", (234.5, 238.0), False),
    ("14N-16O", "NO Gamma (0-2)", (244.5, 248.5), False),
    ("14N-16O", "NO Gamma (0-3)", (256.5, 260.5), False),
    # NO Gamma Full System Envelope (2D Fit: Trot + Tvib across full range)
    ("14N-16O", "NO Gamma System", (220.0, 262.0), True),

    # N2 Second Positive Individual Sub-Bands (1D Fit: Trot only)
    ("14N2", "N2(C-B) (1-0)", (314.0, 316.5), False),
    ("14N2", "N2(C-B) (0-0)", (335.0, 337.5), False),
    ("14N2", "N2(C-B) (0-1)", (355.0, 358.5), False),
    ("14N2", "N2(C-B) (0-2)", (378.0, 381.0), False),
    ("14N2", "N2(C-B) (0-3)", (403.0, 406.5), False),
    # N2 Second Positive Full System Envelope (2D Fit: Trot + Tvib across full range)
    ("14N2", "N2(C-B) System", (330.0, 385.0), True),

    # Isolated Diatomics (1D Fit: Trot only)
    ("16O-1H", "OH(A-X) (0-0)", (304.0, 308.0), False),
    ("N2+(B-X)", "N2+(B-X) (0-0)", (388.0, 392.0), False),
]

# ==============================================================================
# HELPERS & FILE READERS
# ==============================================================================
def compute_o2_mole_fraction(slm_str: str, n2_str: str, o2_str: str) -> typing.Tuple[float, float]:
    slm_val = float(re.sub(r"[^\d\.]", "", slm_str))
    q_carrier_sccm = slm_val * 1000.0

    o2_match = re.search(r"([\d\.]+)SCCMO2", o2_str, re.IGNORECASE)
    q_o2_sccm = float(o2_match.group(1)) if o2_match else 0.0

    n2_match = re.search(r"([\d\.]+)%N2", n2_str, re.IGNORECASE)
    q_n2_sccm = (float(n2_match.group(1)) / 100.0) * q_carrier_sccm if n2_match else 0.0

    q_total_sccm = q_carrier_sccm + q_n2_sccm + q_o2_sccm
    x_o2 = q_o2_sccm / q_total_sccm if q_total_sccm > 0 else 0.0
    return x_o2, x_o2 * 100.0

def read_spa(filepath: Path) -> typing.Tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(filepath, sep=r"\s+", skiprows=1, header=None, engine="c", usecols=[0, 1])
        return df[0].to_numpy(dtype=np.float64), df[1].to_numpy(dtype=np.float64)
    except Exception:
        data = np.genfromtxt(filepath, skip_header=1, encoding="utf-8")
        return data[:, 0], data[:, 1]

def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.replace("\xa0", " ").strip())
    return re.sub(r"_+", "_", cleaned)

def resolve_library_key(target_folder: str, loaded_keys: typing.List[str]) -> typing.Optional[str]:
    clean_target = re.sub(r"[^a-zA-Z0-9]", "", target_folder).lower()
    for key in loaded_keys:
        clean_key = re.sub(r"[^a-zA-Z0-9]", "", key).lower()
        if clean_target in clean_key or clean_key in clean_target:
            return key
    synonyms = {"16o1h": ["oh"], "14n16o": ["no"], "14n14n": ["14n2", "n2"], "n2+bx": ["n2+", "n2+(b-x)", "14n2+"]}
    target_syns = synonyms.get(clean_target, [])
    for key in loaded_keys:
        if any(syn in re.sub(r"[^a-zA-Z0-9]", "", key).lower() for syn in target_syns):
            return key
    return None

def save_fig_safe(fig: plt.Figure, out_path: Path, max_retries: int = 5, delay: float = 0.5):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"  {Log.GREEN}✔ Saved Plot:{Log.RESET} {out_path.name}", flush=True)
            return
        except OSError:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                plt.close(fig)

# ==============================================================================
# EVALUATION & SUMMARY ENGINES
# ==============================================================================
def evaluate_signal_quality(wl: np.ndarray, I_raw: np.ndarray, roi: typing.Tuple[float, float]):
    mask = (wl >= roi[0]) & (wl <= roi[1])
    if np.count_nonzero(mask) < 10:
        return False, "Too few ROI points", 0.0, 0.0, np.array([]), np.array([])

    wl_crop, I_roi_raw = wl[mask], I_raw[mask]
    I_sub = np.maximum(I_roi_raw - np.percentile(I_roi_raw, 5), 0.0)
    peak_raw = float(np.max(I_sub))

    diffs_raw = np.diff(I_roi_raw)
    mad_noise_raw = max(float(1.4826 * np.median(np.abs(diffs_raw - np.median(diffs_raw))) / 1.414), 1.0)
    snr = float(peak_raw / mad_noise_raw)

    if peak_raw < MIN_RAW_PEAK_COUNTS: return False, "Low Intensity", peak_raw, snr, wl_crop, np.zeros_like(wl_crop)
    if snr < MIN_SNR: return False, "Low SNR", peak_raw, snr, wl_crop, np.zeros_like(wl_crop)

    I_crop_norm = I_sub / peak_raw if peak_raw > 0 else I_roi_raw / np.max(I_roi_raw)
    return True, "PRESENT", peak_raw, snr, wl_crop, I_crop_norm

def evaluate_fit_quality(fit_res: dict, trot_bounds: typing.Tuple[float, float], max_shift_nm: float):
    reasons = []
    y_exp, y_fit = fit_res["I_exp_norm"], fit_res["I_fit_norm"]
    ss_res, ss_tot = np.sum((y_exp - y_fit) ** 2), np.sum((y_exp - np.mean(y_exp)) ** 2)
    r2_score = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    if r2_score < R2_MIN:
        reasons.append(f"Low R² ({r2_score:.2f})")
    
    t_rot, (tr_lo, tr_hi) = fit_res["T_rot"], trot_bounds
    if t_rot <= tr_lo + EDGE_TOL_FRAC * (tr_hi - tr_lo):
        reasons.append("Trot pinned at FLOOR")
    elif t_rot >= tr_hi - EDGE_TOL_FRAC * (tr_hi - tr_lo):
        reasons.append("Trot pinned at CEILING")

    shift = fit_res["shift_nm"]
    if abs(shift) >= (1.0 - EDGE_TOL_FRAC) * max_shift_nm:
        reasons.append("Shift pinned at BOUND")

    return (len(reasons) == 0), r2_score, reasons

def generate_summary_sheet(fit_records: typing.List[dict], output_csv: Path):
    if not fit_records: return
    df_summary = pd.DataFrame([{
        "File": r.get("TargetFile", ""), "FlowRate": r.get("FlowRate", ""), 
        "GasType": r.get("GasType", ""), "O2_Mole_Pct": r.get("O2_mole_pct", np.nan),
        "Band Target": r.get("Species", ""), "Fit Status": r.get("Status", ""),
        "Trot [K]": r.get("fit_res", {}).get("T_rot", np.nan),
        "Tvib [K]": r.get("fit_res", {}).get("T_vib", np.nan),
        "T_boltzmann [K]": r.get("boltzmann_data", {}).get("T_boltzmann_K", np.nan) if r.get("boltzmann_data") else np.nan,
        "R2": r.get("r2_score", np.nan)
    } for r in fit_records])
    df_summary.to_csv(output_csv, index=False)
    Log.info(f"Summary sheet generated: {output_csv.name}")

# ==============================================================================
# PLOTTING ENGINES
# ==============================================================================
def generate_3x3_fit_spreads(fit_records: typing.List[dict], save_folder: Path):
    if not fit_records: return
    save_folder.mkdir(parents=True, exist_ok=True)
    grouped = pd.DataFrame(fit_records).groupby(["Species", "GasType"])
    Log.header(f"GENERATING 3x3 FIT OVERLAY SPREADS ({len(grouped)} GRIDS)")

    for (species, gas_type), group in grouped:
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=False, sharey=False, dpi=200)
        for r_n2, n2_label in enumerate(N2_LABELS):
            for c_o2 in range(3):
                ax = axes[r_n2, c_o2]
                cell_fits = group[(group["N2_idx"] == r_n2) & (group["O2_idx"] == c_o2)]

                if cell_fits.empty:
                    ax.text(0.5, 0.5, "No Signal", transform=ax.transAxes, ha="center", color="#999999", style="italic")
                else:
                    for _, row in cell_fits.iterrows():
                        slm, fit, status = row["FlowRate"], row["fit_res"], row["Status"]
                        cfg = SLM_CONFIG.get(slm, {"offset": 0.0, "color": "#1f77b4"})
                        wl = fit["wl_crop"] + fit["shift_nm"]
                        y_exp = fit["I_exp_norm"] + cfg["offset"]
                        y_fit = fit["I_fit_norm"] + cfg["offset"]

                        ax.plot(wl, y_exp, color=cfg["color"], alpha=0.85 if status == "VALID" else 0.4, lw=1.2)
                        ax.plot(wl, y_fit, color="black", ls="-" if status == "VALID" else ":", lw=1.0, alpha=0.9 if status == "VALID" else 0.5)

                        tvib_val = fit.get("T_vib", np.nan)
                        tvib_str = f" Tvib={tvib_val:.0f}K" if not np.isnan(tvib_val) else ""
                        lbl = f"{slm}: Trot={fit['T_rot']:.0f}K{tvib_str}"
                        ax.text(0.03, 0.90 - (0.12 * list(SLM_CONFIG.keys()).index(slm)), lbl, transform=ax.transAxes, fontsize=8, color=cfg["color"])

                ax.grid(False)
                if r_n2 == 0: ax.set_title(["Low O2 (0%)", "Mid O2 (~0.25%)", "High O2 (~0.5%)"][c_o2], fontsize=10, fontweight="bold")
                if c_o2 == 0: ax.set_ylabel(f"{n2_label}\nNorm. Intensity", fontsize=9, fontweight="bold")
                if r_n2 == 2: ax.set_xlabel("Wavelength (nm)", fontsize=9, fontweight="bold")
                for spine in ["top", "right"]: ax.spines[spine].set_visible(False)

        fig.suptitle(f"Spectral Fit Overlays: {species} [{gas_type}]", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_fig_safe(fig, save_folder / f"FitSpread3x3_{sanitize_filename(str(species))}_{sanitize_filename(str(gas_type))}.png")

# ==============================================================================
# SAFE DYNAMIC GROUPED BAR GRAPH TEMPERATURE TRENDS (MATPLOTLIB 3.9+ COMPATIBLE)
# ==============================================================================
def generate_3x3_temperature_bar_charts(fit_records: typing.List[dict], save_folder: Path):
    if not fit_records: return
    save_folder.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([r for r in fit_records if r["Status"] == "VALID"])
    if df.empty: return

    Log.header("GENERATING 3x3 GROUPED BAR CHART TREND PLOTS")

    # Collect all species defined in TARGET_BANDS present in valid records
    all_species = [b[1] for b in TARGET_BANDS if b[1] in df["Species"].unique()]
    if not all_species:
        all_species = list(df["Species"].unique())

    # Compatible colormap retrieval across all Matplotlib versions
    cmap = plt.get_cmap("tab20")
    SPECIES_COLORS = {spec: cmap(i % 20) for i, spec in enumerate(all_species)}

    for temp_type, temp_key in [("T_rot", "T_rot"), ("T_vib", "T_vib")]:
        def extract_temp(row):
            fit_res = row.get("fit_res", {})
            if isinstance(fit_res, dict):
                val = fit_res.get(temp_key, np.nan)
                if val is not None and np.isfinite(val) and val > 0:
                    return float(val)
            return np.nan

        df_temp = df.copy()
        df_temp["temp_val"] = df_temp.apply(extract_temp, axis=1)

        if df_temp["temp_val"].isna().all():
            Log.info(f"Skipping {temp_type} bar charts: No valid numerical values found.")
            continue

        active_species = [
            s for s in all_species 
            if not df_temp[df_temp["Species"] == s]["temp_val"].isna().all()
        ]

        if not active_species:
            continue

        fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True, dpi=200)

        for r_n2, n2_label in enumerate(N2_LABELS):
            for c_gas, gas_label in enumerate(GAS_LABELS):
                ax = axes[r_n2, c_gas]
                cell_data = df_temp[(df_temp["N2_idx"] == r_n2) & (df_temp["GasType"] == gas_label)]

                if not cell_data.empty:
                    o2_conditions = sorted(cell_data["O2_mole_pct"].unique())
                    n_species = len(active_species)
                    
                    if n_species > 0 and len(o2_conditions) > 0:
                        bar_width = 0.85 / n_species
                        x_indices = np.arange(len(o2_conditions))

                        for i, species_name in enumerate(active_species):
                            spec_data = cell_data[cell_data["Species"] == species_name]
                            y_vals = []
                            
                            for o2_val in o2_conditions:
                                match = spec_data[spec_data["O2_mole_pct"] == o2_val]
                                if not match.empty:
                                    val = match.iloc[0]["temp_val"]
                                    y_vals.append(val if (np.isfinite(val) and val > 0) else 0.0)
                                else:
                                    y_vals.append(0.0)

                            offsets = x_indices + (i - (n_species - 1) / 2.0) * bar_width
                            color = SPECIES_COLORS.get(species_name, "#7f7f7f")
                            
                            ax.bar(
                                offsets, y_vals, width=bar_width * 0.9, 
                                color=color, edgecolor="black", linewidth=0.5,
                                label=species_name if (r_n2 == 0 and c_gas == 0) else ""
                            )

                        ax.set_xticks(x_indices)
                        ax.set_xticklabels([f"{val:.2f}%" for val in o2_conditions], fontsize=8)

                ax.grid(False)
                if r_n2 == 0: ax.set_title(gas_label, fontsize=10, fontweight="bold")
                if c_gas == 0: ax.set_ylabel(f"{n2_label}\n{temp_type} (K)", fontsize=9, fontweight="bold")
                if r_n2 == 2: ax.set_xlabel("O₂ Mole Fraction", fontsize=9, fontweight="bold")
                for spine in ["top", "right"]: ax.spines[spine].set_visible(False)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
                ncol=min(len(labels), 5), frameon=False, fontsize=8
            )

        plt.tight_layout()
        save_fig_safe(fig, save_folder / f"Trends_BarChart_3x3_{temp_type}.png")

def generate_3x3_boltzmann_grids(fit_records: typing.List[dict], save_folder: Path):
    if not fit_records: return
    save_folder.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([r for r in fit_records if r["Status"] == "VALID"])
    if df.empty: return

    Log.header("GENERATING 3x3 BOLTZMANN PLOT GRIDS")

    for (species, gas_type), group in df.groupby(["Species", "GasType"]):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True, dpi=200)
        has_data = False

        for r_n2, n2_label in enumerate(N2_LABELS):
            for c_o2 in range(3):
                ax = axes[r_n2, c_o2]
                cell_fits = group[(group["N2_idx"] == r_n2) & (group["O2_idx"] == c_o2)]

                if cell_fits.empty:
                    ax.text(0.5, 0.5, "No Signal", transform=ax.transAxes, ha="center", color="#999999", style="italic")
                else:
                    for _, row in cell_fits.iterrows():
                        b_data = row.get("boltzmann_data")
                        slm = row["FlowRate"]
                        cfg = SLM_CONFIG.get(slm, {"color": "#1f77b4"})

                        if b_data and "X_energies_cm1" in b_data:
                            has_data = True
                            X = b_data["X_energies_cm1"]
                            Y = b_data["Y_ln_population"]
                            Y_fit = b_data["Y_fit"]
                            tb = b_data["T_boltzmann_K"]
                            r2 = b_data["R_squared"]

                            ax.scatter(X, Y, color=cfg["color"], s=25, alpha=0.8, label=f"{slm}: T={tb:.0f}K (R²={r2:.2f})")
                            ax.plot(X, Y_fit, color="black", ls="--", lw=1.0)
                        else:
                            ax.text(0.5, 0.5, "Insufficient Peaks", transform=ax.transAxes, ha="center", color="#999999", style="italic")

                ax.grid(False)
                if cell_fits.size > 0 and ax.get_legend_handles_labels()[0]:
                    ax.legend(frameon=False, fontsize=7, loc="upper right")

                if r_n2 == 0: ax.set_title(["Low O₂ (0%)", "Mid O₂ (~0.25%)", "High O₂ (~0.5%)"][c_o2], fontsize=10, fontweight="bold")
                if c_o2 == 0: ax.set_ylabel(f"{n2_label}\nln(Population)", fontsize=9, fontweight="bold")
                if r_n2 == 2: ax.set_xlabel("Energy E' (cm⁻¹)", fontsize=9, fontweight="bold")
                for spine in ["top", "right"]: ax.spines[spine].set_visible(False)

        if has_data:
            fig.suptitle(f"Boltzmann Distribution Grid: {species} [{gas_type}]", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            save_fig_safe(fig, save_folder / f"BoltzmannGrid3x3_{sanitize_filename(str(species))}_{sanitize_filename(str(gas_type))}.png")
        else:
            plt.close(fig)

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
if __name__ == "__main__":
    Log.header("EXOMOL MOLECULAR FITTING & BOLTZMANN ANALYSIS PIPELINE")

    fitter = MolecularFitter()
    loaded_keys = list(fitter.loaded_libraries.keys())
    
    file_queue, fit_records, stats = [], [], {"valid_fits": 0, "rejected_fits": 0, "skipped_bands": 0}

    # Queue target file paths based on standard experimental matrix
    for slm in SLM_FOLDERS:
        o2_levels = ["0SCCMO2", "1.25SCCMO2", "2.5SCCMO2"] if slm == "0.5SLM" else ["0SCCMO2", "2.5SCCMO2", "5SCCMO2"]
        if slm == "2SLM": o2_levels = ["0SCCMO2", "5SCCMO2", "10SCCMO2"]

        for c_n2, n2_folder in enumerate(N2_FOLDERS):
            n2_dir = BASE_DIR / n2_folder / slm
            if not n2_dir.exists(): n2_dir = BASE_DIR / slm / n2_folder
            if not n2_dir.exists(): continue

            dir_files = [p for p in n2_dir.glob("*") if p.suffix.lower() == ".spa"]
            for r_o2, o2_lvl in enumerate(o2_levels):
                x_o2, o2_mole_pct = compute_o2_mole_fraction(slm, n2_folder, o2_lvl)
                for g_gas, gas_label in enumerate(GAS_LABELS):
                    target_file = next((f for f in dir_files if o2_lvl.lower() in f.name.lower() and 
                                       ((g_gas == 0 and "he" in f.name.lower() and "hear" not in f.name.lower()) or 
                                        (g_gas == 1 and "ar" in f.name.lower() and "hear" not in f.name.lower()) or 
                                        (g_gas == 2 and "hear" in f.name.lower()))), None)
                    if target_file:
                        file_queue.append({
                            "target_file": target_file, "slm": slm, "n2_folder": n2_folder, 
                            "c_n2": c_n2, "r_o2": r_o2, "gas_label": gas_label, "o2_mole_pct": o2_mole_pct
                        })

    for file_idx, item in enumerate(file_queue, 1):
        target_file = item["target_file"]
        Log.subheader(f"[{file_idx}/{len(file_queue)}] {target_file.name}")
        wl, I_raw = read_spa(target_file)

        for radical_folder, display_name, roi, fit_tvib in TARGET_BANDS:
            matched_key = resolve_library_key(radical_folder, loaded_keys)
            if matched_key is None: continue

            # Evaluate signal quality over the target ROI
            is_present, presence_reason, raw_peak, snr, wl_crop, I_crop_norm = evaluate_signal_quality(wl, I_raw, roi)
            if not is_present:
                Log.status(display_name, "SKIP", presence_reason)
                stats["skipped_bands"] += 1
                continue

            # Execute model fit (fit_tvib dynamically toggles 1D vs 2D optimization)
            fit_res = fitter.fit_spectrum(
                wl_crop=wl_crop, 
                I_crop_norm=I_crop_norm, 
                target_band=matched_key,
                fit_baseline=True, 
                max_shift_nm=MAX_SHIFT_NM, 
                fit_tvib=fit_tvib, 
                default_tvib=4500.0
            )

            lib_info = fitter.loaded_libraries[matched_key]
            is_valid_fit, r2_score, qc_reasons = evaluate_fit_quality(
                fit_res, 
                trot_bounds=(float(lib_info["T_rot_grid"][0]), float(lib_info["T_rot_grid"][-1])),
                max_shift_nm=MAX_SHIFT_NM
            )

            # Compute Boltzmann population diagnostics using BoltzmannAnalyzer
            boltzmann_data = BoltzmannAnalyzer.extract_populations(
                fit_res["wl_crop"], 
                fit_res["I_exp_norm"], 
                fit_res["I_fit_norm"], 
                fit_res["T_rot"]
            )

            status = "VALID" if is_valid_fit else "FIT_REJECTED"
            if status == "VALID":
                stats["valid_fits"] += 1
                tvib_str = f" │ T_vib={fit_res['T_vib']:.0f}K" if fit_tvib and "T_vib" in fit_res else ""
                tboltz_str = f" │ T_boltz={boltzmann_data['T_boltzmann_K']:.0f}K" if boltzmann_data else ""
                Log.status(display_name, "PASS", f"T_rot={fit_res['T_rot']:.0f}K{tvib_str}{tboltz_str} │ R²={r2_score:.3f}")
            else:
                stats["rejected_fits"] += 1
                Log.status(display_name, "FAIL", f"R²={r2_score:.3f} │ {', '.join(qc_reasons)}")

            fit_records.append({
                "TargetFile": target_file.name, "FlowRate": item["slm"], "N2_idx": item["c_n2"], 
                "O2_idx": item["r_o2"], "GasType": item["gas_label"], "Species": display_name, 
                "RadicalKey": radical_folder, "O2_mole_pct": item["o2_mole_pct"], 
                "raw_peak_counts": raw_peak, "fit_res": fit_res, "r2_score": r2_score, 
                "boltzmann_data": boltzmann_data, "Status": status
            })

    # Generate 3x3 diagnostic output plots & reports
    generate_3x3_fit_spreads(fit_records, SPREADS_DIR)
    generate_3x3_temperature_bar_charts(fit_records, TRENDS_DIR)
    generate_3x3_boltzmann_grids(fit_records, BOLTZMANN_DIR)
    generate_summary_sheet(fit_records, SUMMARY_CSV_PATH)

    Log.header("ANALYSIS COMPLETE")