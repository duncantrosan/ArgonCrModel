from pathlib import Path
import typing
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, lsq_linear

try:
    import h5py
except ImportError:
    h5py = None

trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))


class MolecularFitter:
    """General spectroscopic fitting engine incorporating mathematical and algorithmic safeguards."""

    def __init__(
        self,
        calib_lib_dir: typing.Optional[Path] = None,
        slit_stem: str = "09_04_2026",
        interp_mode: str = "linear",
    ):
        if calib_lib_dir is None:
            candidates = [
                Path(__file__).parent / "Calibrated_Library",
                Path(__file__).parent.parent / "Calibrated_Library",
                Path.cwd() / "Calibrated_Library",
            ]
            for cand in candidates:
                if cand.exists():
                    calib_lib_dir = cand
                    break
            if calib_lib_dir is None:
                calib_lib_dir = Path(__file__).parent / "Calibrated_Library"

        self.slit_dir = calib_lib_dir / slit_stem
        self.interp_mode = interp_mode.lower()
        self.loaded_libraries: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        self._load_all_calibrated_libraries()

    def _load_all_calibrated_libraries(self):
        """Loads both compressed HDF5 (.h5) and legacy NPZ (.npz) calibrated libraries."""
        if not self.slit_dir.exists():
            return

        if h5py is not None:
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

    @staticmethod
    def _crop_library_view(lib: typing.Dict[str, typing.Any], wl_min: float, wl_max: float) -> typing.Dict[str, typing.Any]:
        wl_grid = lib["wl_grid"]
        idx0 = int(np.searchsorted(wl_grid, wl_min)) - 2
        idx1 = int(np.searchsorted(wl_grid, wl_max)) + 2
        idx0 = max(0, idx0)
        idx1 = min(len(wl_grid), idx1)
        if idx1 <= idx0:
            idx0, idx1 = 0, len(wl_grid)

        data = lib["intensity"]
        if lib["is_3d"]:
            sliced_intensity = np.ascontiguousarray(data[:, :, idx0:idx1])
        elif data.ndim == 2:
            sliced_intensity = np.ascontiguousarray(data[:, idx0:idx1])
        else:
            sliced_intensity = np.ascontiguousarray(data[idx0:idx1])

        return {
            "wl_grid": np.ascontiguousarray(wl_grid[idx0:idx1]),
            "T_rot_grid": lib["T_rot_grid"],
            "T_vib_grid": lib["T_vib_grid"],
            "intensity": sliced_intensity,
            "is_3d": lib["is_3d"],
        }

    def _interpolate_spectrum(
        self,
        wl_eval: np.ndarray,
        T_rot: float,
        T_vib: float,
        lib: typing.Dict[str, typing.Any],
        interp_mode: typing.Optional[str] = None,
    ) -> np.ndarray:
        mode = (interp_mode or self.interp_mode).lower()
        wl_grid = lib["wl_grid"]
        rot_grid = lib["T_rot_grid"]
        vib_grid = lib["T_vib_grid"]
        data = lib["intensity"]

        r_val = float(np.clip(T_rot, rot_grid[0], rot_grid[-1]))
        r_idx = int(np.searchsorted(rot_grid, r_val) - 1)
        r_idx = max(0, min(r_idx, len(rot_grid) - 2)) if len(rot_grid) > 1 else 0

        if mode == "cubic" and len(rot_grid) >= 4:
            r_start = max(0, min(r_idx - 1, len(rot_grid) - 4))
            r_win = slice(r_start, r_start + 4)
            rot_sub = rot_grid[r_win]

            if lib["is_3d"] and len(vib_grid) >= 4:
                v_val = float(np.clip(T_vib, vib_grid[0], vib_grid[-1]))
                v_idx = int(np.searchsorted(vib_grid, v_val) - 1)
                v_start = max(0, min(v_idx - 1, len(vib_grid) - 4))
                v_win = slice(v_start, v_start + 4)
                vib_sub = vib_grid[v_win]

                data_sub = data[v_win, r_win, :]
                cs_vib = CubicSpline(vib_sub, data_sub, axis=0)(v_val)
                spec_1d = CubicSpline(rot_sub, cs_vib, axis=0)(r_val)
            else:
                slice_2d = data[0, r_win, :] if data.ndim == 3 else data[r_win, :]
                spec_1d = CubicSpline(rot_sub, slice_2d, axis=0)(r_val)

            spec_1d = np.maximum(spec_1d, 0.0)
        else:
            w_r = (
                (r_val - rot_grid[r_idx]) / (rot_grid[r_idx + 1] - rot_grid[r_idx])
                if len(rot_grid) > 1 and rot_grid[r_idx + 1] > rot_grid[r_idx]
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

                w00 = (1.0 - w_v) * (1.0 - w_r)
                w01 = (1.0 - w_v) * w_r
                w10 = w_v * (1.0 - w_r)
                w11 = w_v * w_r

                n00, n01 = data[v_idx, r_idx, :], data[v_idx, r_idx + 1, :]
                n10, n11 = data[v_idx + 1, r_idx, :], data[v_idx + 1, r_idx + 1, :]

                if mode == "log_linear":
                    eps = 1e-12
                    log_sum = (
                        w00 * np.log(np.maximum(n00, eps))
                        + w01 * np.log(np.maximum(n01, eps))
                        + w10 * np.log(np.maximum(n10, eps))
                        + w11 * np.log(np.maximum(n11, eps))
                    )
                    spec_1d = np.exp(log_sum)
                else:
                    spec_1d = w00 * n00 + w01 * n01 + w10 * n10 + w11 * n11
            else:
                slice_2d = data[0, :, :] if data.ndim == 3 else data
                if len(rot_grid) > 1:
                    n0, n1 = slice_2d[r_idx, :], slice_2d[r_idx + 1, :]
                    w0, w1 = 1.0 - w_r, w_r
                else:
                    n0, n1 = slice_2d[0, :], slice_2d[0, :]
                    w0, w1 = 1.0, 0.0

                if mode == "log_linear":
                    eps = 1e-12
                    spec_1d = np.exp(w0 * np.log(np.maximum(n0, eps)) + w1 * np.log(np.maximum(n1, eps)))
                else:
                    spec_1d = w0 * n0 + w1 * n1

        return np.interp(wl_eval, wl_grid, spec_1d, left=0.0, right=0.0)

    @staticmethod
    def _solve_multi_component_linear(
        basis_vectors: typing.List[np.ndarray],
        wl_crop: np.ndarray,
        I_exp: np.ndarray,
        fit_baseline: bool = True,
        weight_power: float = 1.0,
        custom_weights: typing.Optional[np.ndarray] = None,
        use_robust_huber: bool = False,
        huber_c: float = 1.345,
        max_irls_iter: int = 3,
        wl_centered: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[np.ndarray, float, float, np.ndarray, float]:
        num_pts = len(I_exp)
        num_comp = len(basis_vectors)

        if num_comp == 0 or all(np.all(b == 0.0) for b in basis_vectors):
            base = float(np.mean(I_exp)) if fit_baseline else 0.0
            return np.zeros(num_comp), 0.0, base, np.full_like(I_exp, base), 1.0

        total_shape = np.zeros(num_pts, dtype=np.float64)
        for b in basis_vectors:
            total_shape += np.maximum(b, 0.0)
        base_weights = (total_shape + 0.01) ** weight_power

        if custom_weights is not None:
            base_weights *= custom_weights

        num_cols = num_comp + (2 if fit_baseline else 0)
        design_matrix = np.empty((num_pts, num_cols), dtype=np.float64)
        for i in range(num_comp):
            design_matrix[:, i] = basis_vectors[i]

        if fit_baseline:
            if wl_centered is None:
                wl_centered = wl_crop - np.mean(wl_crop)
            design_matrix[:, -2] = wl_centered
            design_matrix[:, -1] = 1.0

        # BOUNDS FIX: Constrain baseline slope and non-negative baseline offset
        lb = np.zeros(num_cols, dtype=np.float64)
        ub = np.full(num_cols, np.inf, dtype=np.float64)
        if fit_baseline:
            lb[-2] = -0.05  # Prevent steep baseline tilting
            ub[-2] =  0.05
            lb[-1] =  0.0   # Baseline offset cannot be negative

        current_weights = base_weights.copy()
        coeffs = np.zeros(num_cols, dtype=np.float64)

        for irls_iter in range(max_irls_iter if use_robust_huber else 1):
            w_sqrt = np.sqrt(current_weights)
            A_weighted = design_matrix * w_sqrt[:, None]
            y_weighted = I_exp * w_sqrt

            res = lsq_linear(A_weighted, y_weighted, bounds=(lb, ub), method="bvls")
            coeffs = res.x

            if not use_robust_huber or irls_iter == max_irls_iter - 1:
                break

            synth_temp = design_matrix @ coeffs
            raw_residuals = I_exp - synth_temp
            mad = float(np.median(np.abs(raw_residuals - np.median(raw_residuals))))
            sigma = 1.4826 * mad + 1e-8

            norm_res = np.abs(raw_residuals) / (huber_c * sigma)
            huber_weights = np.where(norm_res <= 1.0, 1.0, 1.0 / norm_res)
            current_weights = base_weights * huber_weights

        amps = coeffs[:num_comp]
        slope, base = (float(coeffs[-2]), float(coeffs[-1])) if fit_baseline else (0.0, 0.0)

        synth = design_matrix @ coeffs
        residuals = I_exp - synth
        weighted_rmse = float(np.sqrt(np.sum(current_weights * (residuals**2)) / np.sum(current_weights)))

        return amps, slope, base, synth, weighted_rmse

    def _prealign_shift(
        self,
        wl_crop: np.ndarray,
        I_crop_norm: np.ndarray,
        configs: typing.List[typing.Dict[str, typing.Any]],
        cropped_libs: typing.Dict[str, typing.Dict[str, typing.Any]],
        max_shift_nm: float,
        custom_weights: typing.Optional[np.ndarray] = None,
        weight_power: float = 1.0,
        interp_mode: typing.Optional[str] = None,
        num_grid_pts: int = 51,
        wl_centered: typing.Optional[np.ndarray] = None,
    ) -> float:
        shift_grid = np.linspace(-max_shift_nm, max_shift_nm, num_grid_pts)
        best_shift = 0.0
        min_rmse = float("inf")

        for s_test in shift_grid:
            wl_shifted = wl_crop + s_test
            basis_vectors = []

            for cfg in configs:
                lib = cropped_libs[cfg["name"]]
                t_rot = cfg.get("fixed_trot") if cfg.get("fixed_trot") is not None else float(np.mean(lib["T_rot_grid"]))
                t_vib = cfg.get("default_tvib") if cfg.get("default_tvib") is not None else float(lib["T_vib_grid"][0])

                shape = self._interpolate_spectrum(
                    wl_shifted, t_rot, t_vib, lib, interp_mode=interp_mode
                )
                
                shape_max = np.max(shape)
                if shape_max > 0:
                    shape = shape / shape_max
                basis_vectors.append(shape)

            _, _, _, _, rmse = self._solve_multi_component_linear(
                basis_vectors,
                wl_crop,
                I_crop_norm,
                fit_baseline=True,
                weight_power=weight_power,
                custom_weights=custom_weights,
                use_robust_huber=False,
                wl_centered=wl_centered,
            )

            if rmse < min_rmse:
                min_rmse = rmse
                best_shift = s_test

        return float(best_shift)

    def fit_spectrum(
        self,
        wl_crop: np.ndarray,
        I_crop_norm: np.ndarray,
        target_band: typing.Optional[str] = None,
        species_configs: typing.Optional[typing.Union[str, typing.List[typing.Dict[str, typing.Any]]]] = None,
        fit_baseline: bool = True,
        max_shift_nm: typing.Optional[float] = 0.35,
        fit_tvib: bool = True,
        fixed_trot: typing.Optional[float] = None,
        default_tvib: typing.Optional[float] = 4500.0,
        global_max_shift_nm: typing.Optional[float] = None,
        fit_global_shift: bool = True,
        weight_regions: typing.Optional[typing.List[typing.Tuple[float, float, float]]] = None,
        prealign_shift: bool = True,
        weight_power: float = 1.0,
        popsize: int = 20,
        maxiter: int = 150,
        interp_mode: typing.Optional[str] = None,
        seed: int = 42,
        # --- Convergence-conditioning toggles (default = V0-strength search) ---
        de_updating: str = "immediate",   # "immediate" converges harder per-gen when serial (no workers=); "deferred" only pays off with workers>1
        de_tol: float = 1e-6,             # tighter than V1's 1e-3; stops DE later, closer to V0's search depth
        shift_window_nm: typing.Optional[float] = None,  # None = search full (-shift_bound_max, +shift_bound_max), matching V0; set e.g. 0.15 to re-enable V1's narrowed/faster window once a band is validated
    ) -> typing.Dict[str, typing.Any]:
        if species_configs is None:
            if target_band is None:
                raise ValueError("Must specify either target_band or species_configs.")
            configs = [{
                "name": target_band,
                "fit_tvib": fit_tvib,
                "fixed_trot": fixed_trot,
                "default_tvib": default_tvib,
            }]
        elif isinstance(species_configs, str):
            configs = [{
                "name": species_configs,
                "fit_tvib": fit_tvib,
                "fixed_trot": fixed_trot,
                "default_tvib": default_tvib,
            }]
        else:
            configs = species_configs

        shift_bound_max = global_max_shift_nm if global_max_shift_nm is not None else max_shift_nm
        if shift_bound_max is None:
            shift_bound_max = 0.35

        cropped_libs = {}
        for cfg in configs:
            sp_name = cfg["name"]
            if sp_name not in self.loaded_libraries:
                raise KeyError(f"Target species band '{sp_name}' not found in loaded libraries.")
            cropped_libs[sp_name] = self._crop_library_view(
                self.loaded_libraries[sp_name],
                float(np.min(wl_crop)) - shift_bound_max - 0.5,
                float(np.max(wl_crop)) + shift_bound_max + 0.5,
            )

        custom_weights = np.ones_like(wl_crop, dtype=np.float64)
        if weight_regions:
            for wl_min, wl_max, weight_val in weight_regions:
                mask = (wl_crop >= wl_min) & (wl_crop <= wl_max)
                custom_weights[mask] *= weight_val

        wl_centered = wl_crop - np.mean(wl_crop) if fit_baseline else None

        init_shift = 0.0
        if fit_global_shift and prealign_shift:
            init_shift = self._prealign_shift(
                wl_crop, I_crop_norm, configs, cropped_libs, shift_bound_max, custom_weights, weight_power, interp_mode, wl_centered=wl_centered
            )

        bounds = []
        param_mapping = []

        for idx, cfg in enumerate(configs):
            lib = cropped_libs[cfg["name"]]
            two_trot = cfg.get("two_trot", False)
            do_fit_tvib = cfg.get("fit_tvib", fit_tvib) and lib["is_3d"] and len(lib["T_vib_grid"]) > 1
            sp_fixed_trot = cfg.get("fixed_trot", fixed_trot)
            sp_max_shift = cfg.get("max_shift_nm", shift_bound_max)

            entry = {
                "species_idx": idx,
                "name": cfg["name"],
                "two_trot": two_trot,
                "has_tvib": do_fit_tvib,
                "default_tvib": cfg.get("default_tvib", default_tvib),
                "has_shift": sp_max_shift is not None and not fit_global_shift,
            }

            if sp_fixed_trot is not None:
                tr_min = max(sp_fixed_trot * 0.90, float(lib["T_rot_grid"][0]))
                tr_max = min(sp_fixed_trot * 1.10, float(lib["T_rot_grid"][-1]))
                rot_b = (tr_min, tr_max)
            else:
                rot_b = (float(lib["T_rot_grid"][0]), float(lib["T_rot_grid"][-1]))

            bounds.append(rot_b)
            if two_trot:
                bounds.append(rot_b)

            if do_fit_tvib:
                vib_b = (float(lib["T_vib_grid"][0]), float(lib["T_vib_grid"][-1]))
                bounds.append(vib_b)

            if entry["has_shift"]:
                bounds.append((-sp_max_shift, sp_max_shift))

            param_mapping.append(entry)

        if fit_global_shift:
            if shift_window_nm is None:
                # Full-range search (V0 behavior) -- robust when prealign's fixed-temperature
                # coarse grid picks the wrong local shift for temperature-sensitive band shapes.
                s_min, s_max = -shift_bound_max, shift_bound_max
            else:
                # Narrowed window around the prealign estimate (V1 speed behavior, opt-in).
                s_min = max(-shift_bound_max, init_shift - shift_window_nm)
                s_max = min(shift_bound_max, init_shift + shift_window_nm)
            bounds.append((s_min, s_max))

        def evaluate_step(params, use_robust_huber=False):
            param_idx = 0
            basis_vectors = []
            basis_meta = []
            global_shift = params[-1] if fit_global_shift else 0.0

            for entry in param_mapping:
                lib = cropped_libs[entry["name"]]

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
                    def_tv = entry["default_tvib"]
                    T_vib = def_tv if def_tv is not None else float(lib["T_vib_grid"][0])

                shift_nm = params[param_idx] if entry["has_shift"] else global_shift
                if entry["has_shift"]:
                    param_idx += 1

                wl_shifted = wl_crop + shift_nm

                shape1 = self._interpolate_spectrum(
                    wl_shifted, T_rot1, T_vib, lib, interp_mode=interp_mode
                )
                
                shape1_max = np.max(shape1)
                if shape1_max > 0:
                    shape1 = shape1 / shape1_max

                basis_vectors.append(shape1)
                basis_meta.append({
                    "species": entry["name"],
                    "component": "cold/single" if entry["two_trot"] else "single",
                    "T_rot": T_rot1,
                    "T_vib": T_vib,
                    "shift_nm": shift_nm,
                })

                if entry["two_trot"]:
                    shape2 = self._interpolate_spectrum(
                        wl_shifted, T_rot2, T_vib, lib, interp_mode=interp_mode
                    )
                    
                    shape2_max = np.max(shape2)
                    if shape2_max > 0:
                        shape2 = shape2 / shape2_max

                    basis_vectors.append(shape2)
                    basis_meta.append({
                        "species": entry["name"],
                        "component": "hot",
                        "T_rot": T_rot2,
                        "T_vib": T_vib,
                        "shift_nm": shift_nm,
                    })

            amps, slope, base, synth, w_rmse = self._solve_multi_component_linear(
                basis_vectors,
                wl_crop,
                I_crop_norm,
                fit_baseline=fit_baseline,
                weight_power=weight_power,
                custom_weights=custom_weights,
                use_robust_huber=use_robust_huber,
                wl_centered=wl_centered,
            )

            return w_rmse, amps, slope, base, synth, basis_meta, basis_vectors

        res = differential_evolution(
            lambda p: evaluate_step(p, use_robust_huber=False)[0],
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=de_tol,
            updating=de_updating,
            seed=seed,
        )

        w_rmse, best_amps, best_slope, best_base, I_fit_norm, basis_meta, basis_vecs = evaluate_step(
            res.x, use_robust_huber=True
        )

        species_results = {}
        meta_idx = 0
        for cfg in configs:
            name = cfg["name"]
            if cfg.get("two_trot", False):
                m1, m2 = basis_meta[meta_idx], basis_meta[meta_idx + 1]
                a1, a2 = float(best_amps[meta_idx]), float(best_amps[meta_idx + 1])
                
                comp1 = a1 * basis_vecs[meta_idx]
                comp2 = a2 * basis_vecs[meta_idx + 1]
                meta_idx += 2

                area1 = float(trapz_fn(comp1, wl_crop)) if trapz_fn else 0.0
                area2 = float(trapz_fn(comp2, wl_crop)) if trapz_fn else 0.0

                species_results[name] = {
                    "two_trot": True,
                    "T_rot_1": float(m1["T_rot"]),
                    "T_rot_2": float(m2["T_rot"]),
                    "T_vib": float(m1["T_vib"]),
                    "shift_nm": float(m1["shift_nm"]),
                    "amplitude_1": a1,
                    "amplitude_2": a2,
                    "total_amplitude": a1 + a2,
                    "integrated_area_1": area1,
                    "integrated_area_2": area2,
                    "total_integrated_area": area1 + area2,
                    "ratio_1": a1 / (a1 + a2) if (a1 + a2) > 0 else 0.0,
                }
            else:
                m = basis_meta[meta_idx]
                amp = float(best_amps[meta_idx])
                comp = amp * basis_vecs[meta_idx]
                meta_idx += 1

                area = float(trapz_fn(comp, wl_crop)) if trapz_fn else 0.0

                species_results[name] = {
                    "two_trot": False,
                    "T_rot": float(m["T_rot"]),
                    "T_vib": float(m["T_vib"]),
                    "shift_nm": float(m["shift_nm"]),
                    "amplitude": amp,
                    "integrated_area": area,
                }

        primary_sp = list(species_results.values())[0] if species_results else {}
        primary_rot = primary_sp.get("T_rot", primary_sp.get("T_rot_1", np.nan))
        primary_vib = primary_sp.get("T_vib", np.nan)
        primary_shift = primary_sp.get("shift_nm", 0.0)
        primary_amp = primary_sp.get("total_amplitude", primary_sp.get("amplitude", 0.0))

        return {
            "T_rot": float(primary_rot),
            "T_vib": float(primary_vib),
            "shift_nm": float(primary_shift),
            "amplitude": float(primary_amp),
            "baseline_slope": float(best_slope),
            "baseline_offset": float(best_base),
            "rmse": float(w_rmse),
            "wl_crop": wl_crop,
            "I_exp_norm": I_crop_norm,
            "I_fit_norm": I_fit_norm,
            "species_results": species_results,
        }