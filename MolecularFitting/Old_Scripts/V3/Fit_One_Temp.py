from pathlib import Path
import typing
import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, lsq_linear


class MolecularFitter:
    """General spectroscopic fitting engine supporting N-species concurrent fitting,
    2-Trot dual-temperature distributions per species, and non-linear spectral grid interpolation.
    """

    def __init__(
        self,
        calib_lib_dir: typing.Optional[Path] = None,
        slit_stem: str = "09_04_2026",
        interp_mode: str = "log_linear",
    ):
        if calib_lib_dir is None:
            calib_lib_dir = Path(__file__).parent / "Calibrated_Library"

        self.slit_dir = calib_lib_dir / slit_stem
        self.interp_mode = interp_mode.lower()
        self.loaded_libraries: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        self._load_all_calibrated_libraries()

    def _load_all_calibrated_libraries(self):
        """Loads both compressed HDF5 (.h5) and legacy NPZ (.npz) calibrated libraries."""
        if not self.slit_dir.exists():
            raise FileNotFoundError(f"Calibrated library directory not found: {self.slit_dir}")

        for h5_path in self.slit_dir.rglob("*.h5"):
            band_key = (
                h5_path.parent.name
                if h5_path.parent != self.slit_dir
                else h5_path.stem.replace("_calibrated_library", "")
            )

            with h5py.File(h5_path, "r") as h5f:
                keys = list(h5f.keys())
                wl_key = "wavelength_nm" if "wavelength_nm" in keys else "wl"
                trot_key = "T_rot" if "T_rot" in keys else "Trot"
                tvib_key = "T_vib" if "T_vib" in keys else "Tvib"

                if "convolved_3d_matrix" in keys:
                    intensity_data = np.ascontiguousarray(h5f["convolved_3d_matrix"][:], dtype=np.float64)
                elif "convolved_rot_matrix" in keys:
                    intensity_data = np.ascontiguousarray(h5f["convolved_rot_matrix"][:], dtype=np.float64)
                elif "intensity" in keys:
                    intensity_data = np.ascontiguousarray(h5f["intensity"][:], dtype=np.float64)
                else:
                    continue

                self.loaded_libraries[band_key] = {
                    "wl_grid": np.ascontiguousarray(h5f[wl_key][:], dtype=np.float64).flatten(),
                    "T_rot_grid": np.ascontiguousarray(h5f[trot_key][:], dtype=np.float64).flatten(),
                    "T_vib_grid": np.ascontiguousarray(h5f[tvib_key][:], dtype=np.float64).flatten()
                    if tvib_key in keys
                    else np.array([3000.0]),
                    "intensity": intensity_data,
                    "is_3d": intensity_data.ndim == 3 and intensity_data.shape[0] > 1,
                }

        for npz_path in self.slit_dir.rglob("*.npz"):
            band_key = (
                npz_path.parent.name
                if npz_path.parent != self.slit_dir
                else npz_path.stem.replace("_calibrated_library", "")
            )

            if band_key in self.loaded_libraries:
                continue

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
                    "T_vib_grid": np.ascontiguousarray(data[tvib_key], dtype=np.float64).flatten()
                    if tvib_key in keys
                    else np.array([3000.0]),
                    "intensity": intensity_data,
                    "is_3d": intensity_data.ndim == 3 and intensity_data.shape[0] > 1,
                }

    def _interpolate_spectrum(
        self,
        wl_eval: np.ndarray,
        T_rot: float,
        T_vib: float,
        target_band: str,
        interp_mode: typing.Optional[str] = None,
    ) -> np.ndarray:
        """Interpolates synthetic spectrum across temperature grid using linear, log-linear, or cubic spline mode."""
        mode = (interp_mode or self.interp_mode).lower()
        lib = self.loaded_libraries[target_band]
        wl_grid = lib["wl_grid"]
        rot_grid = lib["T_rot_grid"]
        vib_grid = lib["T_vib_grid"]
        data = lib["intensity"]

        r_val = float(np.clip(T_rot, rot_grid[0], rot_grid[-1]))
        r_idx = int(np.searchsorted(rot_grid, r_val) - 1)
        r_idx = max(0, min(r_idx, len(rot_grid) - 2))

        # --- CUBIC SPLINE INTERPOLATION ---
        if mode == "cubic" and len(rot_grid) >= 4:
            # Select 4-point window for local cubic spline
            r_start = max(0, min(r_idx - 1, len(rot_grid) - 4))
            r_win = slice(r_start, r_start + 4)
            rot_sub = rot_grid[r_win]

            if lib["is_3d"] and len(vib_grid) >= 4:
                v_val = float(np.clip(T_vib, vib_grid[0], vib_grid[-1]))
                v_idx = int(np.searchsorted(vib_grid, v_val) - 1)
                v_start = max(0, min(v_idx - 1, len(vib_grid) - 4))
                v_win = slice(v_start, v_start + 4)
                vib_sub = vib_grid[v_win]

                # 2D local cubic spline (Vib then Rot)
                data_sub = data[v_win, r_win, :]  # Shape: (4, 4, N_wl)
                cs_vib = CubicSpline(vib_sub, data_sub, axis=0)(v_val)  # Shape: (4, N_wl)
                spec_1d = CubicSpline(rot_sub, cs_vib, axis=0)(r_val)   # Shape: (N_wl,)
            else:
                slice_2d = data[0, r_win, :] if data.ndim == 3 else data[r_win, :]
                spec_1d = CubicSpline(rot_sub, slice_2d, axis=0)(r_val)

            spec_1d = np.maximum(spec_1d, 0.0)

        # --- LOG-LINEAR OR STANDARD LINEAR INTERPOLATION ---
        else:
            w_r = (
                (r_val - rot_grid[r_idx]) / (rot_grid[r_idx + 1] - rot_grid[r_idx])
                if rot_grid[r_idx + 1] > rot_grid[r_idx]
                else 0.0
            )

            if lib["is_3d"] and len(vib_grid) > 1:
                v_val = float(np.clip(T_vib, vib_grid[0], vib_grid[-1]))
                v_idx = int(np.searchsorted(vib_grid, v_val) - 1)
                v_idx = max(0, min(v_idx, len(vib_grid) - 2))
                w_v = (
                    (v_val - vib_grid[v_idx]) / (vib_grid[v_idx + 1] - vib_grid[v_idx])
                    if vib_grid[v_idx + 1] > vib_grid[v_idx]
                    else 0.0
                )

                nodes = [
                    data[v_idx, r_idx, :],
                    data[v_idx, r_idx + 1, :],
                    data[v_idx + 1, r_idx, :],
                    data[v_idx + 1, r_idx + 1, :],
                ]
                weights = [(1.0 - w_v) * (1.0 - w_r), (1.0 - w_v) * w_r, w_v * (1.0 - w_r), w_v * w_r]
            else:
                slice_2d = data[0, :, :] if data.ndim == 3 else data
                nodes = [slice_2d[r_idx, :], slice_2d[r_idx + 1, :]]
                weights = [1.0 - w_r, w_r]

            if mode == "log_linear":
                # Interpolate in log-intensity domain (Boltzmann-consistent)
                eps = 1e-12
                log_sum = np.zeros_like(nodes[0], dtype=np.float64)
                for w, node in zip(weights, nodes):
                    log_sum += w * np.log(np.maximum(node, eps))
                spec_1d = np.exp(log_sum)
            else:
                # Standard linear blend
                spec_1d = np.zeros_like(nodes[0], dtype=np.float64)
                for w, node in zip(weights, nodes):
                    spec_1d += w * node

        return np.interp(wl_eval, wl_grid, spec_1d, left=0.0, right=0.0)

    @staticmethod
    def _solve_multi_component_linear(
        basis_vectors: typing.List[np.ndarray],
        wl_crop: np.ndarray,
        I_exp: np.ndarray,
        fit_baseline: bool = True,
        weight_power: float = 1.0,
    ) -> typing.Tuple[np.ndarray, float, float, np.ndarray, float]:
        """Solves Non-Negative Least Squares (NNLS) for N basis shapes + linear baseline."""
        num_pts = len(I_exp)
        num_comp = len(basis_vectors)

        if num_comp == 0 or all(np.all(b == 0.0) for b in basis_vectors):
            base = float(np.mean(I_exp)) if fit_baseline else 0.0
            return np.zeros(num_comp), 0.0, base, np.full_like(I_exp, base), 1.0

        total_shape = np.sum([np.maximum(b, 0.0) for b in basis_vectors], axis=0)
        weights = (total_shape + 0.01) ** weight_power
        w_sqrt = np.sqrt(weights)
        wl_centered = wl_crop - np.mean(wl_crop)

        cols = [b for b in basis_vectors]
        if fit_baseline:
            cols.append(wl_centered)
            cols.append(np.ones(num_pts, dtype=np.float64))

        design_matrix = np.column_stack(cols)
        A_weighted = design_matrix * w_sqrt[:, None]
        y_weighted = I_exp * w_sqrt

        lb = np.zeros(design_matrix.shape[1], dtype=np.float64)
        ub = np.full(design_matrix.shape[1], np.inf, dtype=np.float64)
        if fit_baseline:
            lb[-2:] = -np.inf

        res = lsq_linear(A_weighted, y_weighted, bounds=(lb, ub), method="trf")
        coeffs = res.x

        amps = coeffs[:num_comp]
        slope, base = (float(coeffs[-2]), float(coeffs[-1])) if fit_baseline else (0.0, 0.0)

        synth = np.zeros(num_pts, dtype=np.float64)
        for i, amp in enumerate(amps):
            synth += amp * basis_vectors[i]

        if fit_baseline:
            synth += slope * wl_centered + base

        residuals = I_exp - synth
        weighted_rmse = float(np.sqrt(np.sum(weights * (residuals**2)) / np.sum(weights)))
        return amps, slope, base, synth, weighted_rmse

    def fit_spectrum(
        self,
        wl_crop: np.ndarray,
        I_crop_norm: np.ndarray,
        species_configs: typing.Union[str, typing.List[typing.Dict[str, typing.Any]]],
        fit_baseline: bool = True,
        global_max_shift_nm: typing.Optional[float] = None,
        fit_global_shift: bool = True,
        interp_mode: typing.Optional[str] = None,
        seed: int = 42,
    ) -> typing.Dict[str, typing.Any]:
        if isinstance(species_configs, str):
            configs = [{"name": species_configs}]
        else:
            configs = species_configs

        for cfg in configs:
            name = cfg["name"]
            if name not in self.loaded_libraries:
                raise KeyError(f"Target species band '{name}' not found in loaded libraries.")

        bounds = []
        param_mapping = []

        for idx, cfg in enumerate(configs):
            lib = self.loaded_libraries[cfg["name"]]
            two_trot = cfg.get("two_trot", False)
            fit_tvib = cfg.get("fit_tvib", True) and lib["is_3d"]
            max_shift = cfg.get("max_shift_nm", global_max_shift_nm)

            entry = {
                "species_idx": idx,
                "name": cfg["name"],
                "two_trot": two_trot,
                "has_tvib": fit_tvib,
                "has_shift": max_shift is not None and not fit_global_shift,
            }

            rot_b = (float(lib["T_rot_grid"][0]), float(lib["T_rot_grid"][-1]))
            bounds.append(rot_b)
            if two_trot:
                bounds.append(rot_b)

            if fit_tvib:
                vib_b = (float(lib["T_vib_grid"][0]), float(lib["T_vib_grid"][-1]))
                bounds.append(vib_b)

            if entry["has_shift"]:
                bounds.append((-max_shift, max_shift))

            param_mapping.append(entry)

        if fit_global_shift:
            global_shift_bound = (
                -global_max_shift_nm if global_max_shift_nm is not None else -0.35,
                global_max_shift_nm if global_max_shift_nm is not None else 0.35,
            )
            bounds.append(global_shift_bound)

        def evaluate_step(params):
            param_idx = 0
            basis_vectors = []
            basis_meta = []
            global_shift = params[-1] if fit_global_shift else 0.0

            for cfg, entry in zip(configs, param_mapping):
                lib = self.loaded_libraries[entry["name"]]

                if entry["two_trot"]:
                    T_rot1, T_rot2 = params[param_idx], params[param_idx + 1]
                    param_idx += 2
                else:
                    T_rot1, T_rot2 = params[param_idx], None
                    param_idx += 1

                if entry["has_tvib"]:
                    T_vib = params[param_idx]
                    param_idx += 1
                else:
                    T_vib = cfg.get("fixed_tvib", float(lib["T_vib_grid"][0]))

                shift_nm = params[param_idx] if entry["has_shift"] else global_shift
                if entry["has_shift"]:
                    param_idx += 1

                wl_shifted = wl_crop + shift_nm

                shape1 = self._interpolate_spectrum(
                    wl_shifted, T_rot1, T_vib, entry["name"], interp_mode=interp_mode
                )
                max1 = np.max(shape1)
                if max1 > 0:
                    shape1 /= max1
                basis_vectors.append(shape1)
                basis_meta.append(
                    {
                        "species": entry["name"],
                        "component": "cold/single" if entry["two_trot"] else "single",
                        "T_rot": T_rot1,
                        "T_vib": T_vib,
                        "shift_nm": shift_nm,
                    }
                )

                if entry["two_trot"]:
                    shape2 = self._interpolate_spectrum(
                        wl_shifted, T_rot2, T_vib, entry["name"], interp_mode=interp_mode
                    )
                    max2 = np.max(shape2)
                    if max2 > 0:
                        shape2 /= max2
                    basis_vectors.append(shape2)
                    basis_meta.append(
                        {
                            "species": entry["name"],
                            "component": "hot",
                            "T_rot": T_rot2,
                            "T_vib": T_vib,
                            "shift_nm": shift_nm,
                        }
                    )

            amps, slope, base, synth, w_rmse = self._solve_multi_component_linear(
                basis_vectors, wl_crop, I_crop_norm, fit_baseline=fit_baseline
            )

            return w_rmse, amps, slope, base, synth, basis_meta

        res = differential_evolution(
            lambda p: evaluate_step(p)[0],
            bounds=bounds,
            maxiter=150,
            popsize=20,
            tol=1e-5,
            seed=seed,
        )

        w_rmse, best_amps, best_slope, best_base, I_fit_norm, basis_meta = evaluate_step(res.x)

        species_results = {}
        meta_idx = 0
        for cfg in configs:
            name = cfg["name"]
            if cfg.get("two_trot", False):
                m1, m2 = basis_meta[meta_idx], basis_meta[meta_idx + 1]
                a1, a2 = float(best_amps[meta_idx]), float(best_amps[meta_idx + 1])
                meta_idx += 2

                species_results[name] = {
                    "two_trot": True,
                    "T_rot_1": float(m1["T_rot"]),
                    "T_rot_2": float(m2["T_rot"]),
                    "T_vib": float(m1["T_vib"]),
                    "shift_nm": float(m1["shift_nm"]),
                    "amplitude_1": a1,
                    "amplitude_2": a2,
                    "total_amplitude": a1 + a2,
                    "ratio_1": a1 / (a1 + a2) if (a1 + a2) > 0 else 0.0,
                }
            else:
                m = basis_meta[meta_idx]
                amp = float(best_amps[meta_idx])
                meta_idx += 1

                species_results[name] = {
                    "two_trot": False,
                    "T_rot": float(m["T_rot"]),
                    "T_vib": float(m["T_vib"]),
                    "shift_nm": float(m["shift_nm"]),
                    "amplitude": amp,
                }

        return {
            "species_results": species_results,
            "baseline_slope": float(best_slope),
            "baseline_offset": float(best_base),
            "rmse": float(w_rmse),
            "wl_crop": wl_crop,
            "I_exp_norm": I_crop_norm,
            "I_fit_norm": I_fit_norm,
        }