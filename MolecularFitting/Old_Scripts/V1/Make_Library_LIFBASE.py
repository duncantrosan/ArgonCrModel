from pathlib import Path
import traceback
import matplotlib

# Set non-interactive backend BEFORE importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
RADICAL = "NO(A-X)(0-1)"  # Subfolder under LIFBASE

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

INPUT_DIR = BASE_DIR / "LIFBASE" / RADICAL
OUTPUT_DIR = BASE_DIR / "Raw_Library" / RADICAL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Wavelength Bounds in Nanometers (nm) - Set to None for auto-detection
WL_MIN_NM = None
WL_MAX_NM = None

# High-Resolution Temperature Grids (Kelvin)
T_ROT_MIN, T_ROT_MAX, T_ROT_STEP = 200, 5000, 10
T_VIB_MIN, T_VIB_MAX, T_VIB_STEP = 200, 25000, 50

HC_K = 1.4387779  # cm*K


# ==============================================================================
# DYNAMIC SPECTROSCOPIC ESTIMATORS (STRICTLY FROM LIFBASE DATA)
# ==============================================================================
def estimate_b_rot(N_levels: np.ndarray, pos_matrix_angstrom: np.ndarray) -> float:
    """Calculates B_rot (cm^-1) dynamically from line spacing in the LIFBASE input matrix."""
    b_estimates = []
    for col_idx in range(pos_matrix_angstrom.shape[1]):
        branch_pos = pos_matrix_angstrom[:, col_idx]
        valid = (N_levels >= 0) & (branch_pos > 0)
        if np.sum(valid) >= 3:
            n_vals, pos_ang = N_levels[valid], branch_pos[valid]
            sort_idx = np.argsort(n_vals)
            wn = 1e8 / pos_ang[sort_idx]  # Convert Angstrom to cm^-1
            d_wn = np.abs(np.diff(wn))
            d_n = np.diff(n_vals[sort_idx])

            valid_diff = d_n > 0
            if np.any(valid_diff):
                b_branch = np.median(d_wn[valid_diff] / d_n[valid_diff]) / 2.0
                if 0.1 <= b_branch <= 50.0:
                    b_estimates.append(b_branch)

    if len(b_estimates) > 0:
        b_est = float(np.median(b_estimates))
        print(f"[AUTO-DETECT] Calculated Dynamic B_rot = {b_est:.4f} cm^-1")
        return b_est
    else:
        raise ValueError("Unable to calculate B_rot dynamically from the LIFBASE input matrix.")


# ==============================================================================
# 2x2 DIAGNOSTIC PLOTTING ROUTINE (EXOMOL ALIGNED)
# ==============================================================================
def plot_raw_sweeps_and_boltzmann(npz_file_path: Path):
    if not npz_file_path.exists():
        print(f"Cannot plot: {npz_file_path} does not exist.")
        return

    data = np.load(npz_file_path)
    line_positions_nm = data["line_positions_nm"]
    line_probabilities = data["line_probabilities"]
    T_rot_grid = data["T_rot"]
    T_vib_grid = data["T_vib"]
    rot_factors = data["rot_factors"]
    vib_factors = data["vib_factors"]
    E_rot_lines = data["E_rot"]
    E_vib_lines = data["E_vib"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=120)
    fig.suptitle(
        f"Raw Library Diagnostics ({RADICAL}): Stick Spectra & Boltzmann Distributions",
        fontsize=14,
        fontweight="bold",
    )

    # Plot indices ordered hottest to coldest so hot features draw first
    trot_indices = np.linspace(len(T_rot_grid) - 1, 0, 5, dtype=int)
    tvib_indices = np.linspace(len(T_vib_grid) - 1, 0, 5, dtype=int)

    def get_rot_color(t_val):
        norm = (t_val - np.min(T_rot_grid)) / (np.max(T_rot_grid) - np.min(T_rot_grid)) if np.max(T_rot_grid) != np.min(T_rot_grid) else 0.5
        return plt.cm.plasma(0.15 + 0.7 * norm)

    def get_vib_color(t_val):
        norm = (t_val - np.min(T_vib_grid)) / (np.max(T_vib_grid) - np.min(T_vib_grid)) if np.max(T_vib_grid) != np.min(T_vib_grid) else 0.5
        return plt.cm.viridis(0.15 + 0.7 * norm)

    i_rot_central = len(T_rot_grid) // 2
    j_vib_central = len(T_vib_grid) // 2

    rot_sort_idx = np.argsort(E_rot_lines)
    vib_sort_idx = np.argsort(E_vib_lines)

    # 1. TOP-LEFT: Rotational Spectrum Sweep (Hottest First)
    for i in trot_indices:
        t_rot = T_rot_grid[i]
        v_fac = vib_factors[j_vib_central, :] if vib_factors.ndim == 2 else vib_factors[j_vib_central]
        stick_intensities = line_probabilities * rot_factors[i, :] * v_fac
        max_val = np.max(stick_intensities) if np.max(stick_intensities) > 0 else 1.0
        axes[0, 0].vlines(
            line_positions_nm,
            0,
            stick_intensities / max_val,
            color=get_rot_color(t_rot),
            alpha=0.6,
            linewidth=1.0,
            zorder=2,
        )

    axes[0, 0].set_title(
        f"Rotational Sweep: Stick Spectrum ($T_{{vib}}$ = {T_vib_grid[j_vib_central]:.0f} K)",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("Wavelength (nm)")
    axes[0, 0].set_ylabel("Normalized Line Intensity")
    axes[0, 0].set_ylim(0, 1.15)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # 2. TOP-RIGHT: Rotational Boltzmann Distribution (Hottest First)
    for i in trot_indices:
        t_rot = T_rot_grid[i]
        pop_rot = rot_factors[i, :]
        norm_pop = pop_rot / np.max(pop_rot) if np.max(pop_rot) > 0 else pop_rot
        axes[0, 1].scatter(
            E_rot_lines[rot_sort_idx],
            norm_pop[rot_sort_idx],
            color=get_rot_color(t_rot),
            s=8,
            alpha=0.6,
            label=f"$T_{{rot}}$ = {t_rot:.0f} K",
        )

    axes[0, 1].set_title("Rotational Boltzmann Population ($E_{rot}$)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Rotational Energy $E_{rot}$ (cm$^{-1}$)")
    axes[0, 1].set_ylabel("Relative Rotational Factor")
    axes[0, 1].set_ylim(0, 1.15)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(frameon=True, facecolor="white", loc="upper right")

    # 3. BOTTOM-LEFT: Vibrational Spectrum Sweep (Hottest First)
    for j in tvib_indices:
        t_vib = T_vib_grid[j]
        v_fac = vib_factors[j, :] if vib_factors.ndim == 2 else vib_factors[j]
        stick_intensities = line_probabilities * rot_factors[i_rot_central, :] * v_fac
        max_val = np.max(stick_intensities) if np.max(stick_intensities) > 0 else 1.0
        axes[1, 0].vlines(
            line_positions_nm,
            0,
            stick_intensities / max_val,
            color=get_vib_color(t_vib),
            alpha=0.6,
            linewidth=1.0,
            zorder=2,
        )

    axes[1, 0].set_title(
        f"Vibrational Sweep: Stick Spectrum ($T_{{rot}}$ = {T_rot_grid[i_rot_central]:.0f} K)",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Wavelength (nm)")
    axes[1, 0].set_ylabel("Normalized Line Intensity")
    axes[1, 0].set_ylim(0, 1.15)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # 4. BOTTOM-RIGHT: Vibrational Boltzmann Distribution (Hottest First)
    for j in tvib_indices:
        t_vib = T_vib_grid[j]
        v_fac = vib_factors[j, :] if vib_factors.ndim == 2 else vib_factors[j]
        norm_pop = v_fac / np.max(v_fac) if np.max(v_fac) > 0 else v_fac
        axes[1, 1].scatter(
            E_vib_lines[vib_sort_idx],
            norm_pop[vib_sort_idx],
            color=get_vib_color(t_vib),
            s=8,
            alpha=0.6,
            label=f"$T_{{vib}}$ = {t_vib:.0f} K",
        )

    axes[1, 1].set_title("Vibrational Boltzmann Population ($E_{vib}$)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Vibrational Energy $E_{vib}$ (cm$^{-1}$)")
    axes[1, 1].set_ylabel("Relative Vibrational Factor")
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(frameon=True, facecolor="white", loc="upper right")

    plt.tight_layout()
    save_file = OUTPUT_DIR / f"{RADICAL.replace('(', '_').replace(')', '_')}_raw_diagnostics.png"
    plt.savefig(save_file, dpi=120)
    plt.close()
    print(f"[SUCCESS] Diagnostics plot saved directly to: {save_file.resolve()}")


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    try:
        print(f"Loading LIFBASE CSVs from: {INPUT_DIR.resolve()}")
        rot_prob = pd.read_csv(INPUT_DIR / "Rotational_Transition_Probabilities.csv", skiprows=4, header=None)
        rot_pos = pd.read_csv(INPUT_DIR / "Rotational_Line_Positions.csv", skiprows=4, header=None)

        rot_pos_num = rot_pos.apply(pd.to_numeric, errors="coerce")
        rot_prob_num = rot_prob.apply(pd.to_numeric, errors="coerce")

        N_levels = np.nan_to_num(rot_pos_num.iloc[:, 0].to_numpy(dtype=float), nan=-1.0)
        pos_matrix_angstrom = np.nan_to_num(rot_pos_num.iloc[:, 1:].to_numpy(dtype=float), nan=0.0)
        prob_matrix = np.nan_to_num(rot_prob_num.iloc[:, 1:].to_numpy(dtype=float), nan=0.0)

        # Dynamic spectroscopic calculations
        B_ROT = estimate_b_rot(N_levels, pos_matrix_angstrom)

        N_matrix = np.tile(N_levels[:, None], (1, pos_matrix_angstrom.shape[1]))
        pos_vals_nm = pos_matrix_angstrom.flatten() / 10.0
        prob_vals = prob_matrix.flatten()
        N_vals = N_matrix.flatten()

        N_valid = np.maximum(N_vals, 0)
        E_rot_cm1 = B_ROT * N_valid * (N_valid + 1.0)

        nonzero_mask = (pos_vals_nm > 0) & (prob_vals > 0)
        if WL_MIN_NM is None:
            WL_MIN_NM = np.min(pos_vals_nm[nonzero_mask])
        if WL_MAX_NM is None:
            WL_MAX_NM = np.max(pos_vals_nm[nonzero_mask])

        valid_mask = nonzero_mask & (pos_vals_nm >= WL_MIN_NM) & (pos_vals_nm <= WL_MAX_NM)
        line_positions_nm = pos_vals_nm[valid_mask]
        line_probabilities = prob_vals[valid_mask]
        E_rot_lines = E_rot_cm1[valid_mask]
        N_lines = N_vals[valid_mask]

        # Deconstruct wavenumbers into upper vibrational energy E_vib
        wn_lines = 1e7 / line_positions_nm
        E_vib_raw = wn_lines - E_rot_lines
        E_vib_lines = np.maximum(0.0, E_vib_raw - np.min(E_vib_raw))

        deg_multiplier = 2.0 * np.maximum(N_lines, 0) + 1.0

        T_rot_grid = np.arange(T_ROT_MIN, T_ROT_MAX + T_ROT_STEP, T_ROT_STEP)
        T_vib_grid = np.arange(T_VIB_MIN, T_VIB_MAX + T_VIB_STEP, T_VIB_STEP)

        print("Computing factor matrices...")
        inv_trot = (1.0 / T_rot_grid)[:, np.newaxis]
        rot_factors = deg_multiplier[np.newaxis, :] * np.exp(-HC_K * inv_trot * E_rot_lines[np.newaxis, :])

        inv_tvib = (1.0 / T_vib_grid)[:, np.newaxis]
        vib_factors = np.exp(-HC_K * inv_tvib * E_vib_lines[np.newaxis, :])

        output_npz_path = OUTPUT_DIR / f"{RADICAL.replace('(', '_').replace(')', '_')}_raw_library.npz"
        np.savez_compressed(
            output_npz_path,
            line_positions_nm=line_positions_nm,
            line_probabilities=line_probabilities,
            T_rot=T_rot_grid,
            T_vib=T_vib_grid,
            rot_factors=rot_factors,
            vib_factors=vib_factors,
            E_rot=E_rot_lines,
            E_vib=E_vib_lines,
            N_lines=N_lines,
        )
        print(f"[SUCCESS] Saved factorized library to: {output_npz_path.resolve()}")

        # Run updated plotting routine
        plot_raw_sweeps_and_boltzmann(output_npz_path)

    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Script crashed:\n")
        traceback.print_exc()