from pathlib import Path
import typing
import numpy as np
from scipy.optimize import differential_evolution


class MolecularFitter:
    """General spectroscopic fitting engine supporting both 2D and 3D NPZ calibrated libraries."""

    def __init__(self, calib_lib_dir: typing.Optional[Path] = None, slit_stem: str = "09_04_2026"):
        if calib_lib_dir is None:
            calib_lib_dir = Path(__file__).parent / "Calibrated_Library"

        self.slit_dir = calib_lib_dir / slit_stem
        self.loaded_libraries: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        self._load_all_calibrated_libraries()

    def _load_all_calibrated_libraries(self):
        """Scans slit_dir and dynamically registers every .npz library found."""
        if not self.slit_dir.exists():
            raise FileNotFoundError(f"Calibrated library directory not found: {self.slit_dir}")

        for npz_path in self.slit_dir.rglob("*.npz"):
            band_key = (
                npz_path.parent.name
                if npz_path.parent != self.slit_dir
                else npz_path.stem.replace("_calibrated_library", "")
            )

            with np.load(npz_path) as data:
                keys = list(data.keys())
                wl_key = "wavelength_nm" if "wavelength_nm" in keys else "wl"
                trot_key = "T_rot" if "T_rot" in keys else "Trot"
                tvib_key = "T_vib" if "T_vib" in keys else "Tvib"

                if "convolved_rot_matrix" in keys:
                    intensity_data = np.ascontiguousarray(data["convolved_rot_matrix"], dtype=np.float64)
                elif "intensity" in keys:
                    intensity_data = np.ascontiguousarray(data["intensity"], dtype=np.float64)
                else:
                    continue

                self.loaded_libraries[band_key] = {
                    "wl_grid": np.ascontiguousarray(data[wl_key], dtype=np.float64).flatten(),
                    "T_rot_grid": np.ascontiguousarray(data[trot_key], dtype=np.float64).flatten(),
                    "T_vib_grid": np.ascontiguousarray(data[tvib_key], dtype=np.float64).flatten() if tvib_key in keys else np.array([3000.0]),
                    "intensity": intensity_data,
                    "is_3d": intensity_data.ndim == 3,
                }

    def _interpolate_spectrum(self, wl_eval: np.ndarray, T_rot: float, T_vib: float, target_band: str) -> np.ndarray:
        """Evaluates model shape at (T_vib, T_rot) and interpolates onto wl_eval."""
        lib = self.loaded_libraries[target_band]
        wl_grid = lib["wl_grid"]
        rot_grid = lib["T_rot_grid"]
        vib_grid = lib["T_vib_grid"]
        data = lib["intensity"]

        # Rotational index & bilinear weight
        r_val = float(np.clip(T_rot, rot_grid[0], rot_grid[-1]))
        r_idx = int(np.searchsorted(rot_grid, r_val) - 1)
        r_idx = max(0, min(r_idx, len(rot_grid) - 2))
        w_r = (r_val - rot_grid[r_idx]) / (rot_grid[r_idx + 1] - rot_grid[r_idx]) if rot_grid[r_idx + 1] > rot_grid[r_idx] else 0.0

        if lib["is_3d"] and len(vib_grid) > 1:
            # Vibrational index & bilinear weight
            v_val = float(np.clip(T_vib, vib_grid[0], vib_grid[-1]))
            v_idx = int(np.searchsorted(vib_grid, v_val) - 1)
            v_idx = max(0, min(v_idx, len(vib_grid) - 2))
            w_v = (v_val - vib_grid[v_idx]) / (vib_grid[v_idx + 1] - vib_grid[v_idx]) if vib_grid[v_idx + 1] > vib_grid[v_idx] else 0.0

            # 3D bilinear blend across (Tvib, Trot, wavelength)
            spec_1d = (
                (1.0 - w_v) * (1.0 - w_r) * data[v_idx, r_idx, :]
                + (1.0 - w_v) * w_r * data[v_idx, r_idx + 1, :]
                + w_v * (1.0 - w_r) * data[v_idx + 1, r_idx, :]
                + w_v * w_r * data[v_idx + 1, r_idx + 1, :]
            )
        else:
            slice_2d = data[0, :, :] if data.ndim == 3 else data
            spec_1d = (1.0 - w_r) * slice_2d[r_idx, :] + w_r * slice_2d[r_idx + 1, :]

        return np.interp(wl_eval, wl_grid, spec_1d, left=0.0, right=0.0)

    @staticmethod
    def _solve_amplitude_baseline_weighted(
        shape: np.ndarray,
        wl_crop: np.ndarray,
        I_exp: np.ndarray,
        fit_baseline: bool,
        weight_power: float = 1.0,  # 1.0 = Linear shape weight, 0.5 = Square-root weight
    ) -> typing.Tuple[float, float, float, np.ndarray, float]:
        """Weighted Linear Least Squares solver prioritizing bandheads over background tail noise."""
        if np.all(shape == 0.0):
            base = float(np.mean(I_exp)) if fit_baseline else 0.0
            return 0.0, 0.0, base, np.full_like(I_exp, base), 1.0

        # Define intensity weights based on model shape (bandheads receive peak weights)
        # 0.01 floor ensures background points maintain numerical stability
        weights = (np.maximum(shape, 0.0) + 0.01) ** weight_power
        w_sqrt = np.sqrt(weights)

        wl_centered = wl_crop - np.mean(wl_crop)

        if fit_baseline:
            design_matrix = np.column_stack([shape, wl_centered, np.ones_like(shape)])
        else:
            design_matrix = shape[:, None]

        # Transform linear system: W * A * x = W * y
        A_weighted = design_matrix * w_sqrt[:, None]
        y_weighted = I_exp * w_sqrt

        coeffs, *_ = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)

        if fit_baseline:
            amp, slope, base = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        else:
            amp, slope, base = float(coeffs[0]), 0.0, 0.0

        # Physical constraint: Prevent spectrum inversion
        if amp < 0.0:
            amp = 0.0
            if fit_baseline:
                base_matrix = np.column_stack([wl_centered, np.ones_like(shape)])
                base_weighted = base_matrix * w_sqrt[:, None]
                c_base, *_ = np.linalg.lstsq(base_weighted, y_weighted, rcond=None)
                slope, base = float(c_base[0]), float(c_base[1])
            else:
                slope, base = 0.0, 0.0

        if fit_baseline:
            synth = amp * shape + slope * wl_centered + base
        else:
            synth = amp * shape

        # Compute Weighted RMSE for Differential Evolution objective
        residuals = I_exp - synth
        weighted_rmse = float(np.sqrt(np.sum(weights * (residuals**2)) / np.sum(weights)))

        return amp, slope, base, synth, weighted_rmse

    def fit_spectrum(
        self,
        wl_crop: np.ndarray,
        I_crop_norm: np.ndarray,
        target_band: str,
        fit_baseline: bool = True,
        max_shift_nm: float = 0.35,
        fit_tvib: bool = True,
    ) -> typing.Dict[str, typing.Any]:
        """Fits T_rot, T_vib, shift_nm, amplitude, baseline slope, and offset."""
        lib = self.loaded_libraries.get(target_band)
        if lib is None:
            raise KeyError(f"Target band '{target_band}' not found in loaded libraries. Loaded: {list(self.loaded_libraries.keys())}")

        rot_bounds = (float(lib["T_rot_grid"][0]), float(lib["T_rot_grid"][-1]))
        shift_bounds = (-max_shift_nm, max_shift_nm)
        use_3d_tvib = fit_tvib and lib["is_3d"] and len(lib["T_vib_grid"]) > 1

        if use_3d_tvib:
            vib_bounds = (float(lib["T_vib_grid"][0]), float(lib["T_vib_grid"][-1]))
            bounds = [rot_bounds, vib_bounds, shift_bounds]
        else:
            bounds = [rot_bounds, shift_bounds]

        def evaluate_step(params):
            if use_3d_tvib:
                T_rot, T_vib, shift_nm = params
            else:
                T_rot, shift_nm = params
                T_vib = float(lib["T_vib_grid"][0])

            wl_shifted = wl_crop + shift_nm
            shape = self._interpolate_spectrum(wl_shifted, T_rot, T_vib, target_band)

            # Solves baseline and computes bandhead-weighted RMSE
            amp, slope, base, synth, w_rmse = self._solve_amplitude_baseline_weighted(
                shape, wl_crop, I_crop_norm, fit_baseline, weight_power=1.0
            )
            return w_rmse, T_rot, T_vib, shift_nm, amp, slope, base, synth

        def objective(params):
            return evaluate_step(params)[0]

        res = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=150,
            popsize=20,
            tol=1e-5,
            seed=42,
        )

        rmse, best_rot, best_vib, best_shift, best_amp, best_slope, best_base, I_fit_norm = evaluate_step(res.x)

        return {
            "T_rot": float(best_rot),
            "T_vib": float(best_vib) if use_3d_tvib else np.nan,
            "shift_nm": float(best_shift),
            "amplitude": float(best_amp),
            "baseline_slope": float(best_slope),
            "baseline_offset": float(best_base),
            "rmse": float(rmse),
            "wl_crop": wl_crop,
            "I_exp_norm": I_crop_norm,
            "I_fit_norm": I_fit_norm,
        }