from pathlib import Path
import sys
import traceback
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
DEFAULT_MOLECULE = "14N2"

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Molecule Wavelength Profiles (nm)
SPECTRAL_WINDOWS = {
    "14N2": (280.0, 450.0),
    "14N-16O": (180.0, 350.0),
    "16O-1H": (280.0, 450.0),
}

# Einstein-A Intensity Threshold (s^-1)
A_MIN_THRESHOLD = 0

# Temperature Grids (Kelvin)
T_ROT_MIN, T_ROT_MAX, T_ROT_STEP = 200, 15000, 10
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
    """Dynamically computes B_rot (cm^-1) by fitting E = E_0 + B * J * (J + 1) on upper-state levels."""
    df_temp = pd.DataFrame({"J": j_vals, "E": e_vals})
    df_low = df_temp[df_temp["J"] <= max_j]
    min_e = df_low.groupby("J")["E"].min().reset_index()

    if len(min_e) < 2:
        return 2.0  # Physical fallback default for standard diatomics if levels are missing

    x = (min_e["J"] * (min_e["J"] + 1.0)).values
    y = min_e["E"].values
    slope, _ = np.polyfit(x, y, 1)
    return max(0.1, float(slope))


# ==============================================================================
# DIAGNOSTIC PLOTTING ROUTINE
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

    # 1. TOP-LEFT: Rotational Spectrum Sweep
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

    # 2. TOP-RIGHT: Rotational Boltzmann Distribution
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

    # 3. BOTTOM-LEFT: Vibrational Spectrum Sweep
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

    # 4. BOTTOM-RIGHT: Vibrational State Population (Aggregated)
    unique_evib = np.unique(np.round(E_vib_lines, 2))
    for j in tvib_indices:
        t_vib = T_vib_grid[j]
        v_fac_vals = np.exp(-HC_K * (1.0 / t_vib) * unique_evib)
        axes[1, 1].scatter(
            unique_evib,
            v_fac_vals,
            color=get_vib_color(t_vib),
            s=20,
            alpha=0.8,
            label=f"$T_{{vib}}$ = {t_vib:.0f} K",
        )

    axes[1, 1].set_title("Vibrational State Population ($E_{vib}$)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Relative Vibrational Energy $E_{vib}$ (cm$^{-1}$)")
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
def process_exomol(molecule_name: str):
    input_dir = BASE_DIR / "EXOMOL" / molecule_name
    if not input_dir.exists():
        input_dir = BASE_DIR / molecule_name

    output_dir = BASE_DIR / "Raw_Library" / molecule_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ExoMol files from: {input_dir.resolve()}")
    states_file, trans_files = find_exomol_files(input_dir)

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

    # Resolve spectral bounds for active molecule
    wl_min, wl_max = SPECTRAL_WINDOWS.get(molecule_name, (np.min(pos_vals_nm), np.max(pos_vals_nm)))

    # Apply line selection criteria FIRST
    valid_mask = (pos_vals_nm >= wl_min) & (pos_vals_nm <= wl_max) & (a_coeffs >= A_MIN_THRESHOLD)

    line_positions_nm = pos_vals_nm[valid_mask]
    line_probabilities = a_coeffs[valid_mask]
    j_upper_valid = j_upper_arr[valid_mask]
    g_upper_valid = g_upper_arr[valid_mask]
    e_upper_valid = e_upper_arr[valid_mask]

    # Calculate upper-state B_rot dynamically
    b_rot_c_state = estimate_b_rot(j_upper_valid, e_upper_valid)
    print(f"[AUTO-DETECT] Calculated Upper-State B_rot = {b_rot_c_state:.6f} cm^-1")

    J_valid = np.maximum(j_upper_valid, 0)
    E_rot_lines = b_rot_c_state * J_valid * (J_valid + 1.0)

    # ZERO-REFERENCE VIBRATIONAL ENERGY FIX:
    # Isolate active electronic state manifold (C^3Pi_u state near ~80,000+ cm^-1 for N2 SPS)
    E_vib_raw = np.maximum(0.0, e_upper_valid - E_rot_lines)

    if molecule_name == "14N2":
        system_mask = E_vib_raw >= 80000.0
        if np.any(system_mask):
            line_positions_nm = line_positions_nm[system_mask]
            line_probabilities = line_probabilities[system_mask]
            E_rot_lines = E_rot_lines[system_mask]
            j_upper_valid = j_upper_valid[system_mask]
            g_upper_valid = g_upper_valid[system_mask]
            E_vib_raw = E_vib_raw[system_mask]

    # Reference E_vib to v'=0 of the target state
    E_vib_lines = E_vib_raw - np.min(E_vib_raw)
    N_lines = j_upper_valid
    G_lines = g_upper_valid

    deg_multiplier = np.where(G_lines > 0, G_lines, 2.0 * N_lines + 1.0)

    T_rot_grid = np.arange(T_ROT_MIN, T_ROT_MAX + T_ROT_STEP, T_ROT_STEP)
    T_vib_grid = np.arange(T_VIB_MIN, T_VIB_MAX + T_VIB_STEP, T_VIB_STEP)

    print(f"Filtered to {len(line_positions_nm)} lines between {wl_min:.1f}-{wl_max:.1f} nm.")

    inv_trot = (1.0 / T_rot_grid)[:, np.newaxis]
    rot_factors = deg_multiplier[np.newaxis, :] * np.exp(-HC_K * inv_trot * E_rot_lines[np.newaxis, :])

    inv_tvib = (1.0 / T_vib_grid)[:, np.newaxis]
    vib_factors = np.exp(-HC_K * inv_tvib * E_vib_lines[np.newaxis, :])

    # Save to HDF5 (.h5) with Chunking and Gzip Compression
    output_h5_path = output_dir / f"{molecule_name}_raw_library.h5"
    num_lines = len(line_positions_nm)

    with h5py.File(output_h5_path, "w") as h5f:
        # Save 1D Grids
        h5f.create_dataset("T_rot", data=T_rot_grid)
        h5f.create_dataset("T_vib", data=T_vib_grid)

        # Save 1D Line Properties with Compression
        h5f.create_dataset("line_positions_nm", data=line_positions_nm, compression="gzip", compression_opts=4)
        h5f.create_dataset("line_probabilities", data=line_probabilities, compression="gzip", compression_opts=4)
        h5f.create_dataset("E_rot", data=E_rot_lines, compression="gzip", compression_opts=4)
        h5f.create_dataset("E_vib", data=E_vib_lines, compression="gzip", compression_opts=4)
        h5f.create_dataset("N_lines", data=N_lines, compression="gzip", compression_opts=4)

        # Save 2D Factor Matrices Chunked Per Temperature Row
        h5f.create_dataset(
            "rot_factors",
            data=rot_factors,
            compression="gzip",
            compression_opts=4,
            chunks=(1, num_lines),
        )
        h5f.create_dataset(
            "vib_factors",
            data=vib_factors,
            compression="gzip",
            compression_opts=4,
            chunks=(1, num_lines),
        )

    print(f"[SUCCESS] Saved raw HDF5 library to: {output_h5_path.resolve()}")

    fig_path = output_dir / f"{molecule_name}_raw_diagnostics.png"
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
        molecule_name=molecule_name,
        output_fig_path=fig_path,
    )


if __name__ == "__main__":
    try:
        target_molecule = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MOLECULE
        process_exomol(target_molecule)
    except Exception:
        print("\n[CRITICAL FAILURE] Pipeline failed prior to completing image output:\n")
        traceback.print_exc()