import time
import typing
import numpy as np


def calc_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Coefficient of Determination (R^2)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0


def run_bootstrap_fitting(
    wl_ref: np.ndarray,
    I_flux: np.ndarray,
    dc_vals: np.ndarray,
    fitter: typing.Any,
    sps_key: str,
    n2cb_range: typing.Tuple[float, float] = (331.0, 339.0),
    bandhead_mask_range: typing.Optional[typing.Tuple[float, float]] = (337.05, 337.25),
    max_shift_nm: float = 0.12,
    n_boot: int = 100,
    seed: int = 42,
) -> typing.Dict[str, np.ndarray]:
    """
    Resamples triplicate spectra with replacement N times per Duty Cycle (DC)
    and fits each synthetic composite to quantify confidence intervals.
    """
    np.random.seed(seed)
    n_dc = len(dc_vals)
    n_reps = I_flux.shape[1]

    idx_n2cb = (wl_ref >= n2cb_range[0]) & (wl_ref <= n2cb_range[1])
    wl_fit = wl_ref[idx_n2cb]

    if bandhead_mask_range:
        fit_mask = (wl_fit < bandhead_mask_range[0]) | (wl_fit > bandhead_mask_range[1])
    else:
        fit_mask = np.ones_like(wl_fit, dtype=bool)

    # Raw bootstrap iterations storage shape: (n_boot, n_dc)
    boot_tr = np.zeros((n_boot, n_dc))
    boot_tv = np.zeros((n_boot, n_dc))
    boot_sh = np.zeros((n_boot, n_dc))
    boot_r2 = np.zeros((n_boot, n_dc))

    print(f"-> Running Bootstrap Fits ({n_boot} resamples/DC across {n_dc} Duty Cycles)...")

    for d_idx in range(n_dc):
        t_dc = time.time()
        reps_dc = I_flux[idx_n2cb, :, d_idx]

        for b in range(n_boot):
            # Resample replicate indices (0, 1, 2) with replacement
            boot_indices = np.random.choice(n_reps, size=n_reps, replace=True)
            exp_roi = np.mean(reps_dc[:, boot_indices], axis=1)

            # Baseline floor subtraction and normalization
            bg_floor = np.percentile(exp_roi, 5)
            exp_sub = np.maximum(exp_roi - bg_floor, 0.0)
            exp_norm = exp_sub / np.max(exp_sub) if np.max(exp_sub) > 0 else exp_roi / np.max(exp_roi)

            # Perform spectroscopic fit on included mask region
            fit_res = fitter.fit_spectrum(
                wl_crop=wl_fit[fit_mask],
                I_crop_norm=exp_norm[fit_mask],
                target_band=sps_key,
                fit_baseline=True,
                max_shift_nm=max_shift_nm,
                fit_tvib=True,
            )

            boot_tr[b, d_idx] = fit_res["T_rot"]
            boot_tv[b, d_idx] = fit_res.get("T_vib", 3000.0)
            boot_sh[b, d_idx] = fit_res["shift_nm"]

            if "I_fit_norm" in fit_res:
                boot_r2[b, d_idx] = calc_r_squared(exp_norm[fit_mask], fit_res["I_fit_norm"])

        print(
            f"   [DC {d_idx+1}/{n_dc} = {dc_vals[d_idx]:5.2f}] "
            f"Trot = {np.mean(boot_tr[:, d_idx]):4.0f} ± {np.std(boot_tr[:, d_idx], ddof=1):2.1f} K | "
            f"Tvib = {np.mean(boot_tv[:, d_idx]):4.0f} ± {np.std(boot_tv[:, d_idx], ddof=1):2.1f} K | "
            f"R² = {np.mean(boot_r2[:, d_idx]):.4f} ({time.time() - t_dc:.2f}s)"
        )

    # Compute summary statistics
    return {
        "boot_trot": boot_tr,
        "boot_tvib": boot_tv,
        "boot_shift": boot_sh,
        "boot_r2": boot_r2,
        "trot_mean": np.mean(boot_tr, axis=0),
        "trot_std": np.std(boot_tr, axis=0, ddof=1),
        "trot_ci95_low": np.percentile(boot_tr, 2.5, axis=0),
        "trot_ci95_high": np.percentile(boot_tr, 97.5, axis=0),
        "tvib_mean": np.mean(boot_tv, axis=0),
        "tvib_std": np.std(boot_tv, axis=0, ddof=1),
        "tvib_ci95_low": np.percentile(boot_tv, 2.5, axis=0),
        "tvib_ci95_high": np.percentile(boot_tv, 97.5, axis=0),
        "shift_mean": np.mean(boot_sh, axis=0),
        "shift_std": np.std(boot_sh, axis=0, ddof=1),
        "r2_mean": np.mean(boot_r2, axis=0),
        "r2_std": np.std(boot_r2, axis=0, ddof=1),
        "wl_fit": wl_fit,
        "idx_n2cb": idx_n2cb,
    }