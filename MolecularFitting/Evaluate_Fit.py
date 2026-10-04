"""
MolecularFitting/Evaluate_Fit.py
Centralized evaluation, pre-alignment, baseline reconstitution, and quality control module.
"""

import typing
import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import lsq_linear


# ==============================================================================
# 1. ADVANCED BASELINE & SIGNAL QUALITY ENGINES
# ==============================================================================
def is_masked(wl: np.ndarray, masks: typing.List[typing.Tuple[float, float]]) -> np.ndarray:
    """Returns boolean array indicating whether wavelengths fall inside exclusion masks."""
    in_mask = np.zeros(wl.shape, dtype=bool)
    for low, high in masks:
        in_mask |= (wl >= low) & (wl <= high)
    return in_mask


def estimate_airpls_baseline(y: np.ndarray, lam: float = 1e5, max_iter: int = 15) -> np.ndarray:
    """
    Adaptive Iteratively Reweighted Penalized Least Squares (airPLS).
    Control baseline fitting without manual asymmetry thresholding.
    """
    L = len(y)
    if L < 5:
        return np.full_like(y, np.min(y) if L > 0 else 0.0)

    # Scale lambda with ROI length to preserve stiffness across variable window sizes
    lam_eff = lam * ((L / 500.0) ** 3)
    D = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    H = lam_eff * (D @ D.T)

    w = np.ones(L)
    z = y.copy()

    for i in range(1, max_iter + 1):
        W = diags(w, 0, shape=(L, L)).tocsc()
        try:
            z = spsolve((W + H).tocsc(), w * y)
        except Exception:
            break
        
        d = y - z
        d_negative = d[d < 0]
        if len(d_negative) == 0:
            break
            
        mean_neg = np.mean(d_negative)
        std_neg = np.std(d_negative) + 1e-12

        # Sigmoid weight function for robust convergence
        w = np.zeros(L)
        mask = d < 0
        w[mask] = 1.0 / (1.0 + np.exp((d[mask] - mean_neg) / std_neg))
        w[~mask] = 0.0  # Zero weight for positive peak residuals

    return z


def estimate_als_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares (ALS) with length-adaptive stiffness scaling."""
    L = len(y)
    if L < 5:
        return np.full_like(y, np.min(y) if L > 0 else 0.0)

    lam_eff = lam * ((L / 500.0) ** 3)
    D = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    H = lam_eff * (D @ D.T)
    w = np.ones(L)

    z = y.copy()
    for _ in range(n_iter):
        W = diags(w, 0, shape=(L, L)).tocsc()
        try:
            z = spsolve((W + H).tocsc(), w * y)
        except Exception:
            break
        w = p * (y > z) + (1.0 - p) * (y < z)

    return z


def estimate_linear_baseline(
    wl: np.ndarray,
    y: np.ndarray,
    p: float = 0.05,
    n_iter: int = 10,
    slope_bounds: typing.Optional[typing.Tuple[float, float]] = None,
) -> np.ndarray:
    """Asymmetric-weighted straight-line baseline with slope bounds."""
    L = len(y)
    if L < 3:
        return np.full_like(y, np.min(y) if L > 0 else 0.0)

    wl_c = wl - np.mean(wl)
    A = np.column_stack([wl_c, np.ones(L)])

    lb = np.array([slope_bounds[0], -np.inf]) if slope_bounds else np.array([-np.inf, -np.inf])
    ub = np.array([slope_bounds[1], np.inf]) if slope_bounds else np.array([np.inf, np.inf])

    w = np.ones(L)
    z = np.full(L, np.percentile(y, 5))

    for _ in range(n_iter):
        w_sqrt = np.sqrt(w)
        try:
            res = lsq_linear(A * w_sqrt[:, None], y * w_sqrt, bounds=(lb, ub))
            coeffs = res.x
            z = A @ coeffs
        except Exception:
            break
        w = p * (y > z) + (1.0 - p) * (y < z)

    return z


def evaluate_signal_quality(
    wl: np.ndarray,
    I_raw: np.ndarray,
    roi: typing.Tuple[float, float],
    masks: typing.Optional[typing.List[typing.Tuple[float, float]]] = None,
    min_raw_peak: float = 300.0,
    min_snr: float = 0.0,
    background_method: str = "percentile",
    als_kwargs: typing.Optional[dict] = None,
    linear_kwargs: typing.Optional[dict] = None,
    airpls_kwargs: typing.Optional[dict] = None,
) -> typing.Tuple[bool, str, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates raw signal strength, noise level, and applies region masks and baseline subtraction."""
    mask_roi = (wl >= roi[0]) & (wl <= roi[1])
    if np.count_nonzero(mask_roi) < 10:
        return False, "Too few ROI points", 0.0, 0.0, np.array([]), np.array([]), np.array([]), np.array([])

    wl_roi, I_roi_raw = wl[mask_roi], I_raw[mask_roi]

    method = background_method.lower()
    if method == "airpls":
        baseline = estimate_airpls_baseline(I_roi_raw, **(airpls_kwargs or {}))
        I_sub = I_roi_raw - baseline
    elif method == "als":
        baseline = estimate_als_baseline(I_roi_raw, **(als_kwargs or {}))
        I_sub = I_roi_raw - baseline
    elif method == "linear":
        baseline = estimate_linear_baseline(wl_roi, I_roi_raw, **(linear_kwargs or {}))
        I_sub = I_roi_raw - baseline
    else:
        baseline = np.percentile(I_roi_raw, 5)
        I_sub = I_roi_raw - baseline

    # Noise floor stabilization instead of hard zero truncation
    noise_floor = np.median(np.abs(np.diff(I_sub))) / 1.414
    I_sub_bounded = np.where(I_sub < -3 * noise_floor, 0.0, np.maximum(I_sub, 0.0))

    roi_peak = float(np.max(I_sub_bounded)) if len(I_sub_bounded) > 0 else 1.0
    I_roi_norm = I_sub_bounded / roi_peak if roi_peak > 0 else I_sub_bounded

    if masks:
        in_bandhead = is_masked(wl_roi, masks)
        wl_fit = wl_roi[~in_bandhead]
        I_fit_sub = I_sub_bounded[~in_bandhead]
    else:
        wl_fit = wl_roi
        I_fit_sub = I_sub_bounded

    if len(wl_fit) < 5:
        return False, "Too few unmasked fit points", 0.0, 0.0, np.array([]), np.array([]), wl_roi, I_roi_norm

    peak_raw = float(np.max(I_fit_sub)) if len(I_fit_sub) > 0 else 0.0
    diffs_raw = np.diff(I_roi_raw)
    mad_noise_raw = max(float(1.4826 * np.median(np.abs(diffs_raw - np.median(diffs_raw))) / 1.414), 1.0)
    snr = float(peak_raw / mad_noise_raw)

    if peak_raw < min_raw_peak:
        return False, f"Low Intensity ({peak_raw:.0f} < {min_raw_peak:.0f})", peak_raw, snr, wl_fit, np.zeros_like(wl_fit), wl_roi, I_roi_norm
    if snr < min_snr:
        return False, f"Low SNR ({snr:.1f} < {min_snr:.1f})", peak_raw, snr, wl_fit, np.zeros_like(wl_fit), wl_roi, I_roi_norm

    I_fit_norm = I_fit_sub / peak_raw if peak_raw > 0 else I_fit_sub / np.max(I_fit_sub)
    return True, "PRESENT", peak_raw, snr, wl_fit, I_fit_norm, wl_roi, I_roi_norm


# ==============================================================================
# 2. COARSE PRE-ALIGNMENT SEARCH
# ==============================================================================
def find_coarse_initial_guesses(
    fitter,
    wl_crop: np.ndarray,
    I_crop_norm: np.ndarray,
    species_configs: typing.List[dict],
    max_shift: float = 0.30,
) -> typing.Tuple[float, float]:
    """Finds starting shift_nm and T_rot to prevent non-linear optimizer divergence."""
    best_r2 = -np.inf
    best_shift = 0.0
    best_trot = 500.0

    shifts_to_test = np.linspace(-max_shift, max_shift, 11)
    trots_to_test = [300.0, 500.0, 800.0, 1000.0]

    sp_name = species_configs[0]["name"]
    lib = fitter.loaded_libraries.get(sp_name)
    if not lib:
        return 0.0, 500.0

    for s in shifts_to_test:
        for t in trots_to_test:
            I_candidate = fitter._interpolate_spectrum(
                wl_crop + s, t, 3000.0, lib, interp_mode="log_linear"
            )
            c_max = np.max(I_candidate)
            if c_max > 0:
                I_candidate = I_candidate / c_max

            res = np.sum((I_crop_norm - I_candidate) ** 2)
            tot = np.sum((I_crop_norm - np.mean(I_crop_norm)) ** 2)
            r2 = 1.0 - (res / tot) if tot > 0 else -1.0

            if r2 > best_r2:
                best_r2 = r2
                best_shift = s
                best_trot = t

    return float(best_shift), float(best_trot)


# ==============================================================================
# 3. SPECTRUM RECONSTRUCTION & TWO-STAGE SCORING
# ==============================================================================
def compute_full_theoretical_spectrum(
    fitter,
    fit_res: dict,
    resolved_species_configs: typing.List[dict],
    wl_roi: np.ndarray,
    wl_crop: np.ndarray,
    interp_mode: str = "log_linear",
) -> np.ndarray:
    """Generates continuous pure theoretical line profile across full ROI region."""
    shift = fit_res.get("shift_nm", 0.0)
    baseline_slope = fit_res.get("baseline_slope", 0.0)
    baseline_offset = fit_res.get("baseline_offset", 0.0)
    sp_results = fit_res.get("species_results", {})

    I_fit_roi = np.zeros_like(wl_roi, dtype=np.float64)

    for cfg in resolved_species_configs:
        sp_name = cfg["name"]
        if sp_name not in fitter.loaded_libraries:
            continue

        lib = fitter._crop_library_view(
            fitter.loaded_libraries[sp_name],
            float(np.min(wl_roi)) - abs(shift) - 0.5,
            float(np.max(wl_roi)) + abs(shift) + 0.5,
        )

        res_info = sp_results.get(sp_name, {})
        two_trot = res_info.get("two_trot", False)
        sp_shift = res_info.get("shift_nm", shift)
        wl_crop_shifted = wl_crop + sp_shift
        wl_roi_shifted = wl_roi + sp_shift

        if two_trot:
            tr1, tr2 = res_info.get("T_rot_1", 300.0), res_info.get("T_rot_2", 300.0)
            tv = res_info.get("T_vib", 3000.0)
            a1, a2 = res_info.get("amplitude_1", 0.0), res_info.get("amplitude_2", 0.0)

            sh1_crop = fitter._interpolate_spectrum(wl_crop_shifted, tr1, tv, lib, interp_mode=interp_mode)
            max1 = np.max(sh1_crop)
            if max1 > 0:
                sh1_roi = fitter._interpolate_spectrum(wl_roi_shifted, tr1, tv, lib, interp_mode=interp_mode)
                I_fit_roi += a1 * (sh1_roi / max1)

            sh2_crop = fitter._interpolate_spectrum(wl_crop_shifted, tr2, tv, lib, interp_mode=interp_mode)
            max2 = np.max(sh2_crop)
            if max2 > 0:
                sh2_roi = fitter._interpolate_spectrum(wl_roi_shifted, tr2, tv, lib, interp_mode=interp_mode)
                I_fit_roi += a2 * (sh2_roi / max2)
        else:
            tr = res_info.get("T_rot", 300.0)
            tv = res_info.get("T_vib", 3000.0)
            amp = res_info.get("amplitude", 0.0)

            sh_crop = fitter._interpolate_spectrum(wl_crop_shifted, tr, tv, lib, interp_mode=interp_mode)
            max_val = np.max(sh_crop)
            if max_val > 0:
                sh_roi = fitter._interpolate_spectrum(wl_roi_shifted, tr, tv, lib, interp_mode=interp_mode)
                I_fit_roi += amp * (sh_roi / max_val)

    wl_centered_roi = wl_roi - np.mean(wl_crop)
    baseline_roi = baseline_slope * wl_centered_roi + baseline_offset
    return I_fit_roi + baseline_roi


def compute_reconstituted_r2(
    I_exp_norm: np.ndarray, I_fit_pure_peaks: np.ndarray, wl_roi: np.ndarray
) -> typing.Tuple[float, np.ndarray]:
    """Fits linear baseline back onto pure peaks to evaluate overall full model R2."""
    wl_centered = wl_roi - np.mean(wl_roi)
    A = np.column_stack([I_fit_pure_peaks, wl_centered, np.ones_like(wl_centered)])

    try:
        scale, slope, offset = np.linalg.lstsq(A, I_exp_norm, rcond=None)[0]
    except Exception:
        scale, slope, offset = 1.0, 0.0, 0.0

    I_full_model = max(scale, 0.0) * I_fit_pure_peaks + (slope * wl_centered + offset)

    ss_res = np.sum((I_exp_norm - I_full_model) ** 2)
    ss_tot = np.sum((I_exp_norm - np.mean(I_exp_norm)) ** 2)

    r2_full = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return r2_full, I_full_model


def evaluate_fit_quality(
    fit_res: dict,
    r2_score: float,
    trot_bounds: typing.Optional[typing.Tuple[float, float]] = None,
    max_shift_nm: float = 0.20,
    r2_min: float = 0.35,
    edge_tol_frac: float = 0.005,
) -> typing.Tuple[bool, typing.List[str]]:
    """Checks R2 threshold and checks for parameter boundary pinning."""
    reasons = []

    if r2_score < r2_min:
        reasons.append(f"Low R² ({r2_score:.2f} < {r2_min:.2f})")

    t_rot = fit_res.get("T_rot", np.nan)
    if not np.isnan(t_rot) and trot_bounds is not None:
        tr_lo, tr_hi = trot_bounds
        if t_rot >= tr_hi - edge_tol_frac * (tr_hi - tr_lo):
            reasons.append("Trot pinned at CEILING")
        elif t_rot <= tr_lo + edge_tol_frac * (tr_hi - tr_lo):
            reasons.append("Trot pinned at FLOOR")

    shift = fit_res.get("shift_nm", 0.0)
    if abs(shift) >= (1.0 - edge_tol_frac) * max_shift_nm:
        reasons.append("Shift pinned at BOUND")

    return (len(reasons) == 0), reasons