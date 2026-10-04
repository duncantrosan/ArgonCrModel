from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
TARGET_SLIT_FILE = "09_04_2026.txt"
GRID_STEP_NM = 0.005

ENABLE_ECHELLE_SCALING = True
SLIT_REF_WAVELENGTH_NM = 435.833

MAX_TROT_POINTS = 100
MAX_TVIB_POINTS = 100

BASE_DIR = Path(__file__).parent
RAW_NPZ_DIR = BASE_DIR / "Raw_Library"
SLIT_DIR = BASE_DIR / "Slit_Functions"
CALIB_BASE_DIR = BASE_DIR / "Calibrated_Library"


# ==============================================================================
# ECHELLE CONVOLUTION OPERATOR BUILDER
# ==============================================================================
def build_echelle_convolution_matrix_fast(
    slit_path: Path,
    wl_grid: np.ndarray,
    grid_step: float,
    ref_wl_nm: float,
    enable_echelle: bool = True,
) -> csr_matrix:
    if not slit_path.exists():
        raise FileNotFoundError(f"Slit function file not found: {slit_path}")

    slit_raw = np.loadtxt(slit_path)
    if slit_raw.ndim != 2 or slit_raw.shape[1] < 2:
        raise ValueError("Slit function file must have 2 columns: [Offset_nm, Response]")

    raw_offsets = slit_raw[:, 0]
    raw_response = slit_raw[:, 1]
    raw_offsets -= raw_offsets[np.argmax(raw_response)]  # Center peak at 0.0

    num_wl = len(wl_grid)
    max_offset_nm = max(abs(np.min(raw_offsets)), abs(np.max(raw_offsets)))

    scale_factors = (wl_grid / ref_wl_nm) if enable_echelle else np.ones_like(wl_grid)
    max_half_idx = int(np.ceil((max_offset_nm * np.max(scale_factors)) / grid_step)) + 2

    idx_offsets = np.arange(-max_half_idx, max_half_idx + 1)
    delta_nm = idx_offsets * grid_step

    # Vectorized 2D offset calculation: (num_wl x num_offsets)
    delta_ref = -delta_nm[None, :] / scale_factors[:, None]
    resp_2d = np.interp(delta_ref.ravel(), raw_offsets, raw_response, left=0.0, right=0.0).reshape(
        num_wl, len(idx_offsets)
    )

    row_sums = resp_2d.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    resp_2d /= row_sums

    rows = np.repeat(np.arange(num_wl, dtype=np.int32), len(idx_offsets))
    cols = (np.arange(num_wl, dtype=np.int32)[:, None] + idx_offsets[None, :]).ravel()
    data = resp_2d.ravel().astype(np.float32)

    valid = (cols >= 0) & (cols < num_wl) & (data > 0)
    return coo_matrix((data[valid], (rows[valid], cols[valid])), shape=(num_wl, num_wl)).tocsr()


# ==============================================================================
# CHUNKED HIGH-SPEED TENSOR BUILDER
# ==============================================================================
def build_3d_tensor_chunked(
    lines_nm: np.ndarray,
    a_coeffs: np.ndarray,
    rot_factors: np.ndarray,  # shape: (num_trot, num_lines)
    vib_factors: np.ndarray,  # shape: (num_tvib, num_lines) or (num_tvib,)
    wl_grid: np.ndarray,
    grid_step: float,
    conv_operator: csr_matrix,
) -> np.ndarray:
    num_tvib = len(vib_factors) if vib_factors.ndim == 1 else vib_factors.shape[0]
    num_trot = rot_factors.shape[0]
    num_lines = len(lines_nm)
    num_wl = len(wl_grid)

    wl_min = wl_grid[0]
    line_indices = np.round((lines_nm - wl_min) / grid_step).astype(np.int32)
    valid_mask = (line_indices >= 0) & (line_indices < num_wl)

    valid_line_idx = np.where(valid_mask)[0].astype(np.int32)
    valid_grid_idx = line_indices[valid_mask]
    ones_data = np.ones(len(valid_line_idx), dtype=np.float32)

    # Build transposed mapping operator: (num_wl, num_lines)
    map_matrix_T = coo_matrix(
        (ones_data, (valid_grid_idx, valid_line_idx)), shape=(num_wl, num_lines)
    ).tocsr()

    convolved_3d = np.zeros((num_tvib, num_trot, num_wl), dtype=np.float32)

    for j in range(num_tvib):
        v_fac = vib_factors[j, :] if vib_factors.ndim == 2 else vib_factors[j]
        effective_A = (a_coeffs * v_fac).astype(np.float32)

        # W shape: (num_trot, num_lines)
        W = (rot_factors * effective_A[None, :]).astype(np.float32)

        # CSR (num_wl, num_lines) @ Dense (num_lines, num_trot) -> (num_trot, num_wl)
        unconvolved_2d = (map_matrix_T @ W.T).T

        # CSR (num_wl, num_wl) @ Dense (num_wl, num_trot) -> (num_trot, num_wl)
        convolved_3d[j, :, :] = (conv_operator @ unconvolved_2d.T).T

    return convolved_3d


# ==============================================================================
# PLOTTING ROUTINE
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
    output_fig_path: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=120)
    title_suffix = (
        f" [Echelle R-Scaled @ {SLIT_REF_WAVELENGTH_NM}nm Ref]"
        if ENABLE_ECHELLE_SCALING
        else " [Uniform Slit Profile]"
    )
    fig.suptitle(
        f"Calibrated 3D {radical_name} Model: Raw Sticks vs. Convolved Spectra{title_suffix}",
        fontsize=13,
        fontweight="bold",
    )

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
        v_fac = (
            raw_v_factors[j_vib_fixed, :]
            if raw_v_factors.ndim == 2
            else raw_v_factors[j_vib_fixed]
        )
        raw_lines = raw_a_coeffs * raw_r_factors[i, :] * v_fac
        max_val = np.max(raw_lines) if np.max(raw_lines) > 0 else 1.0
        axes[0, 0].vlines(
            lines_nm, 0, raw_lines / max_val, color=get_rot_color(trot_grid[i]), alpha=0.6, linewidth=1.1
        )

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
        axes[0, 1].plot(
            wl_grid, norm_spec, color=get_rot_color(t_val), label=f"$T_{{rot}}$ = {t_val:.0f} K", linewidth=1.4
        )

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
        axes[1, 0].vlines(
            lines_nm, 0, raw_lines / max_val, color=get_vib_color(tvib_grid[j]), alpha=0.7, linewidth=1.1
        )

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
        axes[1, 1].plot(
            wl_grid, norm_spec, color=get_vib_color(t_val), label=f"$T_{{vib}}$ = {t_val:.0f} K", linewidth=1.4
        )

    axes[1, 1].set_title("Vibrational Sweep: Convolved Spectra", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Wavelength (nm)")
    axes[1, 1].set_ylabel("Scaled Intensity")
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(frameon=True, facecolor="white", loc="upper right")

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=120)
    plt.close()
    print(f"[SUCCESS] Saved comparison figure to: {output_fig_path}")


# ==============================================================================
# MAIN CALIBRATION PIPELINE
# ==============================================================================
def build_calibrated_database():
    slit_path = SLIT_DIR / TARGET_SLIT_FILE
    if not slit_path.exists():
        print(f"[ERROR] Slit function file not found at '{slit_path}'")
        return

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

        rot_step_idx = max(1, len(trot_grid_raw) // MAX_TROT_POINTS)
        vib_step_idx = max(1, len(tvib_grid_raw) // MAX_TVIB_POINTS)

        trot_grid = trot_grid_raw[::rot_step_idx]
        tvib_grid = tvib_grid_raw[::vib_step_idx]
        rot_factors = rot_factors_raw[::rot_step_idx, :]
        vib_factors = (
            vib_factors_raw[::vib_step_idx, :]
            if vib_factors_raw.ndim == 2
            else vib_factors_raw[::vib_step_idx]
        )

        # Robust grid bounds definition with float endpoint buffer
        wl_min = np.floor(np.min(lines_nm)) - 1.0
        wl_max = np.ceil(np.max(lines_nm)) + 1.0
        wl_grid = np.arange(wl_min, wl_max + 0.5 * GRID_STEP_NM, GRID_STEP_NM)

        print(f" Constructing Echelle Sparse Operator ({len(wl_grid)} grid points)...")
        conv_operator = build_echelle_convolution_matrix_fast(
            slit_path=slit_path,
            wl_grid=wl_grid,
            grid_step=GRID_STEP_NM,
            ref_wl_nm=SLIT_REF_WAVELENGTH_NM,
            enable_echelle=ENABLE_ECHELLE_SCALING,
        )

        print(f" Generating 3D spectral tensor ({len(tvib_grid)} Tvib x {len(trot_grid)} Trot x {len(wl_grid)} Wavelengths)...")
        convolved_3d_matrix = build_3d_tensor_chunked(
            lines_nm=lines_nm,
            a_coeffs=a_coeffs,
            rot_factors=rot_factors,
            vib_factors=vib_factors,
            wl_grid=wl_grid,
            grid_step=GRID_STEP_NM,
            conv_operator=conv_operator,
        )

        radical_slug = raw_path.name.replace("_raw_library.npz", "")
        calib_npz_path = calib_radical_dir / f"{radical_slug}_calibrated_library.npz"

        # Includes dual matrix keys for backward and forward compatibility
        np.savez_compressed(
            calib_npz_path,
            wavelength_nm=wl_grid,
            T_rot=trot_grid,
            T_vib=tvib_grid,
            convolved_rot_matrix=convolved_3d_matrix,
            convolved_3d_matrix=convolved_3d_matrix,
            grid_step_nm=GRID_STEP_NM,
            echelle_scaling=ENABLE_ECHELLE_SCALING,
            ref_wavelength_nm=SLIT_REF_WAVELENGTH_NM,
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