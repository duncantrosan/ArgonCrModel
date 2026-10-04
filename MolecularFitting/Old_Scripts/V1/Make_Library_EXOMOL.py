from pathlib import Path
import traceback
import matplotlib

# Set non-interactive backend BEFORE importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
MOLECULE = "14N2"

# Safe BASE_DIR resolution for interactive environments / Jupyter
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

INPUT_DIR = BASE_DIR / "EXOMOL" / MOLECULE
if not INPUT_DIR.exists():
    INPUT_DIR = BASE_DIR / MOLECULE

OUTPUT_DIR = BASE_DIR / "Raw_Library" / MOLECULE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Wavelength Bounds (nm) - OH/N2 = 280-450, N0 = 180-350
WL_MIN_NM = 180.0
WL_MAX_NM = 350.0

# Intensity Threshold
A_MIN_THRESHOLD = 1e3

# Temperature Grids (Kelvin)
T_ROT_MIN, T_ROT_MAX, T_ROT_STEP = 200, 5000, 10
T_VIB_MIN, T_VIB_MAX, T_VIB_STEP = 200, 25000, 50

HC_K = 1.4387779  # cm*K


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def find_exomol_files(input_dir: Path):
    """Locates .states and .trans files."""
    states_files = list(input_dir.glob("*.states*")) + list(input_dir.glob("*.lines*"))
    trans_files = list(input_dir.glob("*.trans*"))

    if not states_files or not trans_files:
        raise FileNotFoundError(f"Could not find ExoMol files in {input_dir.resolve()}.")
    return states_files[0], trans_files


def estimate_b_rot(j_vals: np.ndarray, e_vals: np.ndarray, max_j: float = 15.0) -> float:
    """Dynamically computes B_rot (cm^-1) by fitting E = E_0 + B * J * (J + 1) on min energy levels."""
    df_temp = pd.DataFrame({"J": j_vals, "E": e_vals})
    df_low = df_temp[df_temp["J"] <= max_j]
    min_e = df_low.groupby("J")["E"].min().reset_index()

    if len(min_e) < 2:
        raise ValueError("Insufficient J levels found to estimate B_rot dynamically.")

    x = (min_e["J"] * (min_e["J"] + 1.0)).values
    y = min_e["E"].values
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


# ==============================================================================
# 2x2 DIAGNOSTIC PLOTTING ROUTINE
# ==============================================================================
def plot_raw_sweeps_and_boltzmann(
    line_positions_nm: np.ndarray,
    line_probabilities: np.ndarray,
    rot_factors: np.ndarray,
    vib_factors: np.ndarray,
    E_rot_lines: np.ndarray,
    E_vib_lines: np.ndarray,
    T_rot_grid: np.ndarray,
    T_vib_grid: np.ndarray,
    molecule_name: str,
    output_fig_path: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=120)
    fig.suptitle(
        f"Raw Library Diagnostics ({molecule_name}): Stick Spectra & Boltzmann Distributions",
        fontsize=14,
        fontweight="bold",
    )

    # Reversed order (highest index to lowest) so hottest temperatures plot first
    trot_indices = np.linspace(len(T_rot_grid) - 1, 0, 5, dtype=int)
    tvib_indices = np.linspace(len(T_vib_grid) - 1, 0, 5, dtype=int)

    def get_rot_color(t_val):
        norm = (t_val - np.min(T_rot_grid)) / (np.max(T_rot_grid) - np.min(T_rot_grid))
        return plt.cm.plasma(0.15 + 0.7 * norm)

    def get_vib_color(t_val):
        norm = (t_val - np.min(T_vib_grid)) / (np.max(T_vib_grid) - np.min(T_vib_grid))
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
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig_path, dpi=120)
    plt.close()

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        print(f"Loading ExoMol files from: {INPUT_DIR.resolve()}")
        states_file, trans_files = find_exomol_files(INPUT_DIR)

        print(f"Parsing states file: {states_file.name} ...")
        df_states = pd.read_csv(
            states_file,
            sep=r"\s+",
            header=None,
            comment="#",
            usecols=[0, 1, 2, 3],
            names=["state_id", "e_cm1", "g_tot", "j_val"],
            dtype={"state_id": np.int32, "e_cm1": np.float64, "g_tot": np.float32, "j_val": np.float32},
            engine="c",
        )

        trans_dfs = []
        for tf in trans_files:
            print(f"Parsing transitions file: {tf.name} ...")
            tdf = pd.read_csv(
                tf,
                sep=r"\s+",
                header=None,
                comment="#",
                usecols=[0, 1, 2],
                names=["upper_id", "lower_id", "a_coeff"],
                dtype={"upper_id": np.int32, "lower_id": np.int32, "a_coeff": np.float32},
                engine="c",
            )
            trans_dfs.append(tdf)

        df_trans = pd.concat(trans_dfs, ignore_index=True)

        print("Joining transition and state matrices ...")
        merged = df_trans.merge(df_states, left_on="upper_id", right_on="state_id", how="inner")
        merged.rename(columns={"e_cm1": "e_upper", "j_val": "j_upper", "g_tot": "g_upper"}, inplace=True)

        merged = merged.merge(
            df_states[["state_id", "e_cm1"]],
            left_on="lower_id",
            right_on="state_id",
            how="inner",
            suffixes=("", "_lower"),
        )
        merged.rename(columns={"e_cm1": "e_lower"}, inplace=True)

        wavenumbers = (merged["e_upper"] - merged["e_lower"]).values
        a_coeffs = merged["a_coeff"].values
        j_upper_arr = merged["j_upper"].values
        g_upper_arr = merged["g_upper"].values
        e_upper_arr = merged["e_upper"].values

        valid_wn = wavenumbers > 0
        wavenumbers = wavenumbers[valid_wn]
        a_coeffs = a_coeffs[valid_wn]
        j_upper_arr = j_upper_arr[valid_wn]
        g_upper_arr = g_upper_arr[valid_wn]
        e_upper_arr = e_upper_arr[valid_wn]

        pos_vals_nm = 1e7 / wavenumbers

        # Dynamically compute B_rot from state energies
        b_rot_c_state = estimate_b_rot(j_upper_arr, e_upper_arr)
        print(f"Dynamically calculated B_rot constant: {b_rot_c_state:.6f} cm^-1")

        J_valid = np.maximum(j_upper_arr, 0)
        E_rot_cm1 = b_rot_c_state * J_valid * (J_valid + 1.0)
        E_vib_cm1 = np.maximum(0.0, e_upper_arr - E_rot_cm1)

        valid_mask = (pos_vals_nm >= WL_MIN_NM) & (pos_vals_nm <= WL_MAX_NM) & (a_coeffs >= A_MIN_THRESHOLD)

        line_positions_nm = pos_vals_nm[valid_mask]
        line_probabilities = a_coeffs[valid_mask]
        E_rot_lines = E_rot_cm1[valid_mask]
        E_vib_lines = E_vib_cm1[valid_mask]
        N_lines = j_upper_arr[valid_mask]
        G_lines = g_upper_arr[valid_mask]

        deg_multiplier = np.where(G_lines > 0, G_lines, 2.0 * N_lines + 1.0)

        T_rot_grid = np.arange(T_ROT_MIN, T_ROT_MAX + T_ROT_STEP, T_ROT_STEP)
        T_vib_grid = np.arange(T_VIB_MIN, T_VIB_MAX + T_VIB_STEP, T_VIB_STEP)

        print(f"Filtered to {len(line_positions_nm)} lines between {WL_MIN_NM}-{WL_MAX_NM} nm.")

        inv_trot = (1.0 / T_rot_grid)[:, np.newaxis]
        rot_factors = deg_multiplier[np.newaxis, :] * np.exp(-HC_K * inv_trot * E_rot_lines[np.newaxis, :])

        inv_tvib = (1.0 / T_vib_grid)[:, np.newaxis]
        vib_factors = np.exp(-HC_K * inv_tvib * E_vib_lines[np.newaxis, :])

        output_npz_path = OUTPUT_DIR / f"{MOLECULE}_raw_library.npz"
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
        print(f"[SUCCESS] Saved raw library to: {output_npz_path.resolve()}")

        # Plot generation
        fig_path = OUTPUT_DIR / f"{MOLECULE}_raw_diagnostics.png"
        print(f"Generating diagnostic plot at: {fig_path.resolve()}")

        plot_raw_sweeps_and_boltzmann(
            line_positions_nm=line_positions_nm,
            line_probabilities=line_probabilities,
            rot_factors=rot_factors,
            vib_factors=vib_factors,
            E_rot_lines=E_rot_lines,
            E_vib_lines=E_vib_lines,
            T_rot_grid=T_rot_grid,
            T_vib_grid=T_vib_grid,
            molecule_name=MOLECULE,
            output_fig_path=fig_path,
        )

        if fig_path.exists():
            print(f"[SUCCESS] Diagnostics image created successfully at:\n  -> {fig_path.resolve()}")
        else:
            print("[ERROR] Matplotlib completed without raising errors, but file was not written.")

    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Pipeline failed prior to completing image output:\n")
        traceback.print_exc()