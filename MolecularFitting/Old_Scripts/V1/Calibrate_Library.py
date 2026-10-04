from pathlib import Path
import matplotlib

# Set non-interactive backend BEFORE importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
TARGET_SLIT_FILE = "09_04_2026.txt"  # Active slit file inside 'Slit_Functions/'
GRID_STEP_NM = 0.005  # Resolution of continuous wavelength grid (nm)

# Target temperature grid limits to keep 3D tensor RAM manageable (~300-500 MB)
MAX_TROT_POINTS = 100
MAX_TVIB_POINTS = 100

BASE_DIR = Path(__file__).parent
RAW_NPZ_DIR = BASE_DIR / "Raw_Library"
SLIT_DIR = BASE_DIR / "Slit_Functions"
CALIB_BASE_DIR = BASE_DIR / "Calibrated_Library"


# ==============================================================================
# SLIT KERNEL LOADER & RESAMPLER
# ==============================================================================
def load_and_resample_slit(slit_path: Path, grid_step: float) -> np.ndarray:
    """Loads a 2-column slit function [offset_nm, response] and resamples it to grid_step."""
    if not slit_path.exists():
        raise FileNotFoundError(f"Slit function file not found: {slit_path}")

    slit_raw = np.loadtxt(slit_path)
    if slit_raw.ndim != 2 or slit_raw.shape[1] < 2:
        raise ValueError("Slit function file must have 2 columns: [Wavelength_Offset_nm, Response]")

    offsets = slit_raw[:, 0]
    response = slit_raw[:, 1]

    min_off, max_off = np.min(offsets), np.max(offsets)
    kernel_offsets = np.arange(min_off, max_off + grid_step, grid_step)
    resampled_response = np.interp(kernel_offsets, offsets, response, left=0.0, right=0.0)

    kernel_sum = np.sum(resampled_response)
    if kernel_sum > 0:
        resampled_response /= kernel_sum

    return resampled_response


# ==============================================================================
# STICK TO CONTINUOUS GRID MAPPING
# ==============================================================================
def sticks_to_continuous(
    line_positions: np.ndarray,
    line_intensities: np.ndarray,
    wl_grid: np.ndarray,
    grid_step: float
) -> np.ndarray:
    """Bins discrete stick line intensities onto a uniform continuous wavelength grid."""
    continuous_spectrum = np.zeros_like(wl_grid, dtype=np.float64)
    wl_min = wl_grid[0]
    
    indices = np.round((line_positions - wl_min) / grid_step).astype(int)
    valid_mask = (indices >= 0) & (indices < len(wl_grid))
    
    np.add.at(continuous_spectrum, indices[valid_mask], line_intensities[valid_mask])
    return continuous_spectrum


# ==============================================================================
# 2x2 PLOTTING ROUTINE (LHS: RAW STICKS | RHS: CONVOLVED SPECTRA)
# ==============================================================================
def plot_convolved_sweeps(
    lines_nm: np.ndarray,
    raw_a_coeffs: np.ndarray,
    raw_r_factors: np.ndarray,
    raw_v_factors: np.ndarray,
    wl_grid: np.ndarray,
    convolved_3d_matrix: np.ndarray,
    trot_grid: np.ndarray,
    tvib_grid: np.ndarray,
    radical_name: str,
    output_fig_path: Path
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=120)
    fig.suptitle(f"Calibrated 3D {radical_name} Model: Raw Sticks vs. Convolved Spectra", fontsize=14, fontweight="bold")

    trot_idx_desc = np.linspace(len(trot_grid) - 1, 0, 5, dtype=int)
    trot_idx_asc = np.linspace(0, len(trot_grid) - 1, 5, dtype=int)
    tvib_idx_asc = np.linspace(0, len(tvib_grid) - 1, 5, dtype=int)

    def get_rot_color(t_val):
        norm = (t_val - np.min(trot_grid)) / (np.max(trot_grid) - np.min(trot_grid))
        return plt.cm.plasma(0.15 + 0.7 * norm)

    def get_vib_color(t_val):
        norm = (t_val - np.min(tvib_grid)) / (np.max(tvib_grid) - np.min(tvib_grid))
        return plt.cm.viridis(0.15 + 0.7 * norm)

    j_vib_fixed = len(tvib_grid) // 2
    i_rot_fixed = len(trot_grid) // 2

    # 1. TOP-LEFT: Rotational Sweep - Stick Line Strengths
    for i in trot_idx_desc:
        v_fac = raw_v_factors[j_vib_fixed, :] if raw_v_factors.ndim == 2 else raw_v_factors[j_vib_fixed]
        raw_lines = raw_a_coeffs * raw_r_factors[i, :] * v_fac
        max_val = np.max(raw_lines) if np.max(raw_lines) > 0 else 1.0
        axes[0, 0].vlines(lines_nm, 0, raw_lines / max_val, color=get_rot_color(trot_grid[i]), alpha=0.6, linewidth=1.1)

    axes[0, 0].set_title("Rotational Sweep: Raw Line Strengths", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Wavelength (nm)")
    axes[0, 0].set_ylabel("Normalized Line Strength")
    axes[0, 0].set_ylim(0, 1.15)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # 2. TOP-RIGHT: Rotational Sweep - Convolved Continuous Spectra
    for i in trot_idx_asc:
        t_val = trot_grid[i]
        conv_spec = convolved_3d_matrix[j_vib_fixed, i, :]
        norm_spec = conv_spec / np.max(conv_spec) if np.max(conv_spec) > 0 else conv_spec
        axes[0, 1].plot(wl_grid, norm_spec, color=get_rot_color(t_val), label=f"$T_{{rot}}$ = {t_val:.0f} K", linewidth=1.4)

    axes[0, 1].set_title("Rotational Sweep: Convolved Spectra", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Wavelength (nm)")
    axes[0, 1].set_ylabel("Normalized Intensity")
    axes[0, 1].set_ylim(0, 1.15)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(frameon=True, facecolor="white", loc="upper right")

    # 3. BOTTOM-LEFT: Vibrational Sweep - Stick Line Strengths
    for j in tvib_idx_asc:
        v_fac = raw_v_factors[j, :] if raw_v_factors.ndim == 2 else raw_v_factors[j]
        raw_lines = raw_a_coeffs * raw_r_factors[i_rot_fixed, :] * v_fac
        max_val = np.max(raw_lines) if np.max(raw_lines) > 0 else 1.0
        axes[1, 0].vlines(lines_nm, 0, raw_lines / max_val, color=get_vib_color(tvib_grid[j]), alpha=0.7, linewidth=1.1)

    axes[1, 0].set_title("Vibrational Sweep: Raw Line Strengths", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Wavelength (nm)")
    axes[1, 0].set_ylabel("Scaled Line Intensity")
    axes[1, 0].set_ylim(0, 1.15)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # 4. BOTTOM-RIGHT: Vibrational Sweep - Convolved Continuous Spectra
    for j in tvib_idx_asc:
        t_val = tvib_grid[j]
        conv_spec = convolved_3d_matrix[j, i_rot_fixed, :]
        norm_spec = conv_spec / np.max(conv_spec) if np.max(conv_spec) > 0 else conv_spec
        axes[1, 1].plot(wl_grid, norm_spec, color=get_vib_color(t_val), label=f"$T_{{vib}}$ = {t_val:.0f} K", linewidth=1.4)

    axes[1, 1].set_title("Vibrational Sweep: Convolved Spectra", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Wavelength (nm)")
    axes[1, 1].set_ylabel("Scaled Intensity")
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(frameon=True, facecolor="white", loc="upper right")

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=120)
    plt.close()
    print(f"[SUCCESS] Saved 2x2 comparison figure to: {output_fig_path}")


# ==============================================================================
# MAIN CALIBRATION PIPELINE
# ==============================================================================
def build_calibrated_database():
    slit_path = SLIT_DIR / TARGET_SLIT_FILE
    if not slit_path.exists():
        print(f"[ERROR] Slit function file not found at '{slit_path}'")
        return

    print(f"Loading Slit Function: {slit_path.name}")
    kernel = load_and_resample_slit(slit_path, GRID_STEP_NM)
    slit_stem = slit_path.stem

    raw_files = list(RAW_NPZ_DIR.rglob("*_raw_library.npz"))
    if not raw_files:
        print(f"[ERROR] No raw library NPZ files found in '{RAW_NPZ_DIR}'")
        return

    for raw_path in raw_files:
        radical_folder = raw_path.parent.name
        calib_radical_dir = CALIB_BASE_DIR / slit_stem / radical_folder
        calib_radical_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing raw database: {raw_path.relative_to(RAW_NPZ_DIR)}")
        data = np.load(raw_path)

        lines_nm = data["line_positions_nm"]
        a_coeffs = data["line_probabilities"]
        trot_grid_raw = data["T_rot"]
        tvib_grid_raw = data["T_vib"]
        rot_factors_raw = data["rot_factors"]
        vib_factors_raw = data["vib_factors"]

        # Downsample grids if too dense to prevent RAM exhaustion
        rot_step_idx = max(1, len(trot_grid_raw) // MAX_TROT_POINTS)
        vib_step_idx = max(1, len(tvib_grid_raw) // MAX_TVIB_POINTS)

        trot_grid = trot_grid_raw[::rot_step_idx]
        tvib_grid = tvib_grid_raw[::vib_step_idx]
        rot_factors = rot_factors_raw[::rot_step_idx, :]
        vib_factors = vib_factors_raw[::vib_step_idx, :] if vib_factors_raw.ndim == 2 else vib_factors_raw[::vib_step_idx]

        wl_min = np.floor(np.min(lines_nm)) - 1.0
        wl_max = np.ceil(np.max(lines_nm)) + 1.0
        wl_grid = np.arange(wl_min, wl_max + GRID_STEP_NM, GRID_STEP_NM)

        num_tvib = len(tvib_grid)
        num_trot = len(trot_grid)
        num_wl = len(wl_grid)

        # Build 3D Tensor: shape (num_tvib, num_trot, num_wl)
        print(f" Generating 3D spectral tensor ({num_tvib} Tvib x {num_trot} Trot x {num_wl} Wavelengths)...")
        convolved_3d_matrix = np.zeros((num_tvib, num_trot, num_wl), dtype=np.float32)

        for j in range(num_tvib):
            v_fac = vib_factors[j, :] if vib_factors.ndim == 2 else vib_factors[j]
            for i in range(num_trot):
                stick_intensities = a_coeffs * rot_factors[i, :] * v_fac
                continuous_stick = sticks_to_continuous(lines_nm, stick_intensities, wl_grid, GRID_STEP_NM)
                convolved_3d_matrix[j, i, :] = convolve(continuous_stick, kernel, mode="same")

        radical_slug = raw_path.name.replace("_raw_library.npz", "")
        calib_npz_path = calib_radical_dir / f"{radical_slug}_calibrated_library.npz"

        np.savez_compressed(
            calib_npz_path,
            wavelength_nm=wl_grid,
            T_rot=trot_grid,
            T_vib=tvib_grid,
            convolved_rot_matrix=convolved_3d_matrix,
            grid_step_nm=GRID_STEP_NM,
        )
        print(f"[SUCCESS] Saved 3D calibrated library NPZ: {calib_npz_path}")

        fig_path = calib_radical_dir / f"{radical_slug}_sweeps_comparison.png"
        plot_convolved_sweeps(
            lines_nm=lines_nm,
            raw_a_coeffs=a_coeffs,
            raw_r_factors=rot_factors,
            raw_v_factors=vib_factors,
            wl_grid=wl_grid,
            convolved_3d_matrix=convolved_3d_matrix,
            trot_grid=trot_grid,
            tvib_grid=tvib_grid,
            radical_name=radical_folder,
            output_fig_path=fig_path,
        )


if __name__ == "__main__":
    build_calibrated_database()