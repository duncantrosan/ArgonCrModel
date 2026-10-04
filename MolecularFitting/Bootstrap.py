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
    method: str = "residual",  # 'residual' or 'sem_gaussian'
) -> typing.Dict[str, np.ndarray]:
    """
    Runs continuous Residual or Monte Carlo Gaussian Bootstrapping per Duty Cycle (DC)
    to eliminate discrete distribution spikes caused by low-replicate sampling.
    
    Parameters:
        method: 'residual' samples baseline fit residuals with replacement.
                'sem_gaussian' injects Gaussian noise based on pixel-wise Standard Error of the Mean.
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

    boot_tr = np.zeros((n_boot, n_dc))
    boot_tv = np.zeros((n_boot, n_dc))
    boot_sh = np.zeros((n_boot, n_dc))
    boot_r2 = np.zeros((n_boot, n_dc))

    print(f"-> Running {method.upper()} Bootstrap ({n_boot} resamples/DC across {n_dc} Duty Cycles)...")

    for d_idx in range(n_dc):
        t_dc = time.time()
        reps_dc = I_flux[idx_n2cb, :, d_idx]  # Shape: (n_wl_roi, n_reps)

        # Calculate mean spectrum and Standard Error of the Mean (SEM)
        exp_mean = np.mean(reps_dc, axis=1)
        exp_std = np.std(reps_dc, axis=1, ddof=1) if n_reps > 1 else np.zeros_like(exp_mean)
        sem = exp_std / np.sqrt(n_reps)

        # Baseline fit on mean spectrum to extract initial model fit and residuals
        bg_floor_mean = np.percentile(exp_mean, 5)
        exp_sub_mean = np.maximum(exp_mean - bg_floor_mean, 0.0)
        peak_mean = np.max(exp_sub_mean) if np.max(exp_sub_mean) > 0 else 1.0
        exp_norm_mean = exp_sub_mean / peak_mean

        base_fit = fitter.fit_spectrum(
            wl_crop=wl_fit[fit_mask],
            I_crop_norm=exp_norm_mean[fit_mask],
            target_band=sps_key,
            fit_baseline=True,
            max_shift_nm=max_shift_nm,
            fit_tvib=True,
        )

        # Compute fit residuals on masked region
        if "I_fit_norm" in base_fit:
            base_fit_norm = base_fit["I_fit_norm"]
            residuals = exp_norm_mean[fit_mask] - base_fit_norm
        else:
            base_fit_norm = exp_norm_mean[fit_mask]
            residuals = np.zeros_like(wl_fit[fit_mask])

        for b in range(n_boot):
            if method == "residual" and len(residuals) > 0 and not np.all(residuals == 0):
                # Continuous Residual Bootstrap: Resample fit residuals with replacement
                res_sample = np.random.choice(residuals, size=len(residuals), replace=True)
                exp_fit_norm = np.maximum(base_fit_norm + res_sample, 0.0)
            else:
                # Continuous Monte Carlo Noise Injection using SEM
                noise = np.random.normal(0, np.maximum(sem, 1e-12))
                exp_roi = np.maximum(exp_mean + noise, 0.0)
                bg_floor = np.percentile(exp_roi, 5)
                exp_sub = np.maximum(exp_roi - bg_floor, 0.0)
                peak = np.max(exp_sub)
                exp_norm = exp_sub / peak if peak > 0 else exp_roi
                exp_fit_norm = exp_norm[fit_mask]

            # Fit synthetic continuous spectrum
            fit_res = fitter.fit_spectrum(
                wl_crop=wl_fit[fit_mask],
                I_crop_norm=exp_fit_norm,
                target_band=sps_key,
                fit_baseline=True,
                max_shift_nm=max_shift_nm,
                fit_tvib=True,
            )

            boot_tr[b, d_idx] = fit_res["T_rot"]
            boot_tv[b, d_idx] = fit_res.get("T_vib", 3000.0)
            boot_sh[b, d_idx] = fit_res["shift_nm"]

            if "I_fit_norm" in fit_res:
                boot_r2[b, d_idx] = calc_r_squared(exp_fit_norm, fit_res["I_fit_norm"])

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