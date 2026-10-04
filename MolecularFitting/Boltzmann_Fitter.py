import typing
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

HC_OVER_KB = 1.438777  # hc/kB in cm*K


class BoltzmannAnalyzer:
    """Standalone analyzer to extract Boltzmann populations and plot 3-panel diagnostics."""

    @staticmethod
    def _fit_temperature_from_slope(slope: float, T_rot: float) -> float:
        """Converts model-ratio slope d(ln_ratio)/dE into temperature in Kelvin."""
        if slope != 0 and T_rot > 0:
            inv_T_exp = (1.0 / T_rot) - (slope / HC_OVER_KB)
            return float(1.0 / inv_T_exp) if inv_T_exp > 0 else T_rot
        return T_rot

    @classmethod
    def extract_populations(
        cls,
        wl_crop: np.ndarray,
        I_exp: np.ndarray,
        I_fit: np.ndarray,
        T_rot: float,
        min_points_for_segmented: int = 6,
    ) -> typing.Optional[dict]:
        """Extracts Boltzmann populations using model-ratio normalization.
        Automatically evaluates both single-temperature and bi-linear (2-Trot) fits.
        """
        try:
            peaks, _ = find_peaks(I_fit, height=0.05, distance=4, prominence=0.02)
            if len(peaks) < 4:
                return None

            peak_wl = wl_crop[peaks]
            peak_I_exp = I_exp[peaks]
            peak_I_fit = I_fit[peaks]

            valid = (peak_I_exp > 0.02) & (peak_I_fit > 0.02)
            if np.count_nonzero(valid) < 4:
                return None

            peak_wl = peak_wl[valid]
            peak_I_exp = peak_I_exp[valid]
            peak_I_fit = peak_I_fit[valid]

            ref_idx = np.argmax(peak_I_fit)
            lambda_ref = peak_wl[ref_idx]
            energies_cm1 = np.abs((1e7 / peak_wl) - (1e7 / lambda_ref))
            Y_ratio = np.log(peak_I_exp / peak_I_fit)

            sort_idx = np.argsort(energies_cm1)
            X = energies_cm1[sort_idx]
            Y = Y_ratio[sort_idx]
            N = len(X)

            # Single linear fit
            slope_s, int_s, r_s, _, _ = linregress(X, Y)
            rss_single = np.sum((Y - (slope_s * X + int_s)) ** 2)
            t_boltz_single = cls._fit_temperature_from_slope(slope_s, T_rot)

            is_segmented = False
            best_break_idx = None
            seg_results = None

            # Test segmented bi-linear fit if enough data points exist
            if N >= min_points_for_segmented:
                best_rss_seg = float("inf")

                for k in range(3, N - 2):
                    X1, Y1 = X[:k], Y[:k]
                    X2, Y2 = X[k:], Y[k:]

                    m1, c1, r1, _, _ = linregress(X1, Y1)
                    m2, c2, r2, _, _ = linregress(X2, Y2)

                    rss1 = np.sum((Y1 - (m1 * X1 + c1)) ** 2)
                    rss2 = np.sum((Y2 - (m2 * X2 + c2)) ** 2)
                    total_rss_seg = rss1 + rss2

                    if total_rss_seg < best_rss_seg:
                        best_rss_seg = total_rss_seg
                        best_break_idx = k
                        seg_results = {
                            "m1": m1, "c1": c1, "r1_sq": r1**2,
                            "m2": m2, "c2": c2, "r2_sq": r2**2,
                        }

                # Accept segmented fit if variance reduces by > 25%
                if seg_results and (rss_single - best_rss_seg) / (rss_single + 1e-12) > 0.25:
                    is_segmented = True

            if is_segmented and seg_results and best_break_idx is not None:
                m1, c1 = seg_results["m1"], seg_results["c1"]
                m2, c2 = seg_results["m2"], seg_results["c2"]

                X1, X2 = X[:best_break_idx], X[best_break_idx:]
                Y_fit1 = m1 * X1 + c1
                Y_fit2 = m2 * X2 + c2

                return {
                    "is_segmented": True,
                    "X_energies_cm1": X,
                    "Y_ln_population": Y,
                    "break_X": float(X[best_break_idx]),
                    "X1": X1, "Y_fit1": Y_fit1,
                    "X2": X2, "Y_fit2": Y_fit2,
                    "T_boltzmann_1": cls._fit_temperature_from_slope(m1, T_rot),
                    "T_boltzmann_2": cls._fit_temperature_from_slope(m2, T_rot),
                    "R_squared_1": float(seg_results["r1_sq"]),
                    "R_squared_2": float(seg_results["r2_sq"]),
                }

            return {
                "is_segmented": False,
                "X_energies_cm1": X,
                "Y_ln_population": Y,
                "Y_fit": slope_s * X + int_s,
                "T_boltzmann_K": t_boltz_single,
                "R_squared": float(r_s**2),
            }
        except Exception:
            return None

    @staticmethod
    def generate_3panel_diagnostic(
        fit_res: dict,
        species_name: str,
        file_name: str,
        output_path: Path,
        boltzmann_data: typing.Optional[dict] = None,
    ):
        """Generates 3-panel diagnostic with support for single and bi-linear Boltzmann slopes."""
        fig = plt.figure(figsize=(10, 10), dpi=200)
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1.5])

        ax_spec = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax_spec)
        ax_boltz = fig.add_subplot(gs[2])

        wl = fit_res["wl_crop"] + fit_res["shift_nm"]
        y_exp = fit_res["I_exp_norm"]
        y_fit = fit_res["I_fit_norm"]
        residuals = y_exp - y_fit
        t_rot = fit_res["T_rot"]
        t_vib = fit_res.get("T_vib", np.nan)

        # Panel 1: Overlay
        ax_spec.plot(wl, y_exp, color="#1f77b4", lw=1.2, alpha=0.8, label="Experiment")
        ax_spec.plot(wl, y_fit, color="black", ls="--", lw=1.0, label="Synthetic Model")
        ax_spec.set_ylabel("Normalized Intensity", fontsize=10, fontweight="bold")
        ax_spec.set_title(
            f"Spectral Fit: {species_name} ({file_name})\n"
            f"$T_{{rot}}$ = {t_rot:.0f} K | $T_{{vib}}$ = {t_vib:.0f} K",
            fontsize=11, fontweight="bold"
        )
        ax_spec.legend(loc="upper right", fontsize=9)
        ax_spec.grid(True, linestyle=":", alpha=0.6)

        # Panel 2: Residuals
        ax_res.plot(wl, residuals, color="#d62728", lw=0.8)
        ax_res.axhline(0, color="black", ls=":", lw=1.0)
        ax_res.set_xlabel("Wavelength (nm)", fontsize=10, fontweight="bold")
        ax_res.set_ylabel("Residuals", fontsize=10, fontweight="bold")
        ax_res.grid(True, linestyle=":", alpha=0.6)

        # Panel 3: Boltzmann Diagnostics
        if boltzmann_data is not None and "X_energies_cm1" in boltzmann_data:
            X = boltzmann_data["X_energies_cm1"]
            Y = boltzmann_data["Y_ln_population"]
            ax_boltz.scatter(X, Y, color="#2ca02c", s=25, alpha=0.8, label="Model-Ratio Peaks")

            if boltzmann_data.get("is_segmented", False):
                tb1 = boltzmann_data["T_boltzmann_1"]
                tb2 = boltzmann_data["T_boltzmann_2"]
                r1 = boltzmann_data["R_squared_1"]
                r2 = boltzmann_data["R_squared_2"]

                ax_boltz.plot(
                    boltzmann_data["X1"], boltzmann_data["Y_fit1"], color="#1f77b4", ls="--", lw=1.4,
                    label=f"Cold: $T_{{b1}}$ = {tb1:.0f} K (R²={r1:.2f})"
                )
                ax_boltz.plot(
                    boltzmann_data["X2"], boltzmann_data["Y_fit2"], color="#d62728", ls="--", lw=1.4,
                    label=f"Hot: $T_{{b2}}$ = {tb2:.0f} K (R²={r2:.2f})"
                )
                ax_boltz.axvline(
                    boltzmann_data["break_X"], color="gray", ls=":", alpha=0.7, label="Thermal Knee"
                )
            else:
                tb = boltzmann_data["T_boltzmann_K"]
                r2 = boltzmann_data["R_squared"]
                ax_boltz.plot(
                    X, boltzmann_data["Y_fit"], color="black", ls="--", lw=1.2,
                    label=f"$T_{{boltz}}$ = {tb:.0f} K (R²={r2:.2f})"
                )

            ax_boltz.set_xlabel("Relative Energy $\\Delta\\tilde{\\nu}$ (cm⁻¹)", fontsize=10, fontweight="bold")
            ax_boltz.set_ylabel("$\\ln(I_{exp} / I_{fit})$", fontsize=10, fontweight="bold")
            ax_boltz.set_title("Model-Ratio Boltzmann Population Diagnostic", fontsize=11, fontweight="bold")
            ax_boltz.legend(fontsize=9, loc="best")
            ax_boltz.grid(True, linestyle=":", alpha=0.6)
        else:
            ax_boltz.text(
                0.5, 0.5, "Insufficient Peaks for Boltzmann Plot",
                transform=ax_boltz.transAxes, ha="center", va="center",
                color="#999999", fontsize=10, fontweight="bold", style="italic"
            )
            ax_boltz.axis("off")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)