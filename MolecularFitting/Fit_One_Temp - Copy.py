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


def compute_relative_scale_constants(
    fit_res: typing.Dict[str, typing.Any],
    ref_species_keyword: typing.Optional[str] = "14N2",
) -> typing.Dict[str, typing.Any]:
    """
    Generalized engine to compute N-1 relative scale constants (C_X/Y = A_X / A_Y) 
    and integrated area ratios for joint multi-species co-fits.
    """
    sp_res = fit_res.get("species_results", {})
    if len(sp_res) < 2:
        return {
            "ref_species": None,
            "scale_constants": {},
            "area_scale_constants": {},
        }

    amps = {}
    areas = {}
    for sp, details in sp_res.items():
        amps[sp] = details.get("total_amplitude", details.get("amplitude", np.nan))
        areas[sp] = details.get("total_integrated_area", details.get("integrated_area", np.nan))

    ref_sp = None
    if ref_species_keyword:
        ref_sp = next((sp for sp in amps if ref_species_keyword.lower() in sp.lower()), None)
    if ref_sp is None:
        ref_sp = list(amps.keys())[0]

    ref_amp = amps.get(ref_sp, np.nan)
    ref_area = areas.get(ref_sp, np.nan)

    scale_consts = {}
    area_scale_consts = {}

    if not np.isnan(ref_amp) and ref_amp > 0:
        for sp, amp in amps.items():
            if sp != ref_sp:
                ratio_key = f"C_{sp}_over_{ref_sp}"
                scale_consts[ratio_key] = float(amp / ref_amp) if not np.isnan(amp) else np.nan

    if not np.isnan(ref_area) and ref_area > 0:
        for sp, area in areas.items():
            if sp != ref_sp:
                ratio_key = f"C_area_{sp}_over_{ref_sp}"
                area_scale_consts[ratio_key] = float(area / ref_area) if not np.isnan(area) else np.nan

    return {
        "ref_species": ref_sp,
        "ref_amplitude": float(ref_amp) if not np.isnan(ref_amp) else np.nan,
        "ref_area": float(ref_area) if not np.isnan(ref_area) else np.nan,
        "scale_constants": scale_consts,
        "area_scale_constants": area_scale_consts,
    }


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

    @staticmethod
    def compute_relative_scale_constants(
        fit_res: typing.Dict[str, typing.Any],
        ref_species_keyword: typing.Optional[str] = "14N2",
    ) -> typing.Dict[str, typing.Any]:
        return compute_relative_scale_constants(fit_res, ref_species_keyword)

    def _load_all_calibrated_libraries(self):
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
        baseline_quad_bounds: typing.Tuple[float, float] = (-0.002, 0.002),
        baseline_slope_bounds: typing.Tuple[float, float] = (-0.005, 0.005),
        baseline_offset_bounds: typing.Tuple[float, float] = (-0.05, 0.05),
    ) -> typing.Tuple[np.ndarray, float, float, float, np.ndarray, float]:
        num_pts = len(I_exp)
        num_comp = len(basis_vectors)

        if num_comp == 0 or all(np.all(b == 0.0) for b in basis_vectors):
            base = float(np.percentile(I_exp, 5))
            return np.zeros(num_comp), 0.0, 0.0, base, np.full(num_pts, base), 1e6

        if wl_centered is None:
            wl_centered = wl_crop - np.mean(wl_crop)

        # Build Design Matrix A
        cols = list(basis_vectors)
        if fit_baseline:
            cols.append(wl_centered)
            cols.append(np.ones(num_pts))

        A = np.column_stack(cols)
        n_cols = A.shape[1]

        # Bounds: Non-negative amplitudes for spectra, strict bounds for baseline
        lb = np.zeros(n_cols)
        ub = np.full(n_cols, np.inf)

        if fit_baseline:
            lb[-2], ub[-2] = baseline_slope_bounds
            lb[-1], ub[-1] = baseline_offset_bounds

        # Setup Weighting
        if custom_weights is not None:
            w_vec = np.ascontiguousarray(custom_weights, dtype=np.float64)
        else:
            w_vec = np.ones(num_pts)

        if weight_power != 1.0:
            w_vec = w_vec ** weight_power

        weights = np.sqrt(w_vec)
        A_w = A * weights[:, None]
        y_w = I_exp * weights

        # Bounded Linear Least Squares Solve
        res = lsq_linear(A_w, y_w, bounds=(lb, ub))
        coeffs = res.x

        if use_robust_huber:
            for _ in range(max_irls_iter):
                y_pred = A @ coeffs
                resid = np.abs(I_exp - y_pred)
                sigma = 1.4826 * np.median(resid) + 1e-12
                r_norm = resid / sigma
                huber_w = np.where(r_norm <= huber_c, 1.0, huber_c / (r_norm + 1e-12))
                
                w_curr = np.sqrt(w_vec * huber_w)
                res = lsq_linear(A * w_curr[:, None], I_exp * w_curr, bounds=(lb, ub))
                coeffs = res.x

        amplitudes = coeffs[:num_comp]
        slope = coeffs[-2] if fit_baseline else 0.0
        offset = coeffs[-1] if fit_baseline else 0.0

        y_fit = A @ coeffs
        ss_res = float(np.sum((I_exp - y_fit) ** 2))

        return amplitudes, 0.0, float(slope), float(offset), y_fit, ss_res

    def fit_spectrum(
        self,
        wl_crop: np.ndarray,
        I_crop_norm: np.ndarray,
        species_configs: typing.List[dict],
        fit_baseline: bool = True,
        global_max_shift_nm: float = 0.30,
        fit_global_shift: bool = True,
        interp_mode: typing.Optional[str] = None,
        weight_regions: typing.Optional[typing.List[typing.Tuple[float, float, float]]] = None,
        weight_power: float = 1.0,
        maxiter: int = 200,
    ) -> typing.Dict[str, typing.Any]:
        """Main non-linear fitting execution engine."""
        wl_centered = wl_crop - np.mean(wl_crop)
        num_pts = len(wl_crop)

        # Build custom pixel weight vector
        custom_weights = np.ones(num_pts, dtype=np.float64)
        if weight_regions:
            for w_lo, w_hi, w_fact in weight_regions:
                m = (wl_crop >= w_lo) & (wl_crop <= w_hi)
                custom_weights[m] *= w_fact

        # Prepare species optimization bounds
        bounds = []
        if fit_global_shift:
            bounds.append((-global_max_shift_nm, global_max_shift_nm))

        for cfg in species_configs:
            sp_name = cfg["name"]
            lib = self.loaded_libraries.get(sp_name)
            if not lib:
                continue

            r_grid = lib["T_rot_grid"]
            v_grid = lib["T_vib_grid"]

            if cfg.get("two_trot", False):
                bounds.append((r_grid[0], r_grid[-1]))  # Trot1
                bounds.append((r_grid[0], r_grid[-1]))  # Trot2
            elif "fixed_trot" not in cfg:
                bounds.append((r_grid[0], r_grid[-1]))  # Trot

            if cfg.get("fit_tvib", False) and lib["is_3d"]:
                bounds.append((v_grid[0], v_grid[-1]))  # Tvib

        def _objective(params):
            p_idx = 0
            g_shift = params[p_idx] if fit_global_shift else 0.0
            if fit_global_shift:
                p_idx += 1

            basis_vectors = []
            for cfg in species_configs:
                sp_name = cfg["name"]
                lib = self.loaded_libraries.get(sp_name)
                if not lib:
                    continue

                if cfg.get("two_trot", False):
                    tr1, tr2 = params[p_idx], params[p_idx + 1]
                    p_idx += 2
                    tv = params[p_idx] if (cfg.get("fit_tvib", False) and lib["is_3d"]) else 3000.0
                    if cfg.get("fit_tvib", False) and lib["is_3d"]:
                        p_idx += 1

                    b1 = self._interpolate_spectrum(wl_crop + g_shift, tr1, tv, lib, interp_mode)
                    b2 = self._interpolate_spectrum(wl_crop + g_shift, tr2, tv, lib, interp_mode)
                    m1, m2 = np.max(b1), np.max(b2)
                    basis_vectors.append(b1 / m1 if m1 > 0 else b1)
                    basis_vectors.append(b2 / m2 if m2 > 0 else b2)
                else:
                    tr = cfg.get("fixed_trot")
                    if tr is None:
                        tr = params[p_idx]
                        p_idx += 1

                    tv = params[p_idx] if (cfg.get("fit_tvib", False) and lib["is_3d"]) else 3000.0
                    if cfg.get("fit_tvib", False) and lib["is_3d"]:
                        p_idx += 1

                    b = self._interpolate_spectrum(wl_crop + g_shift, tr, tv, lib, interp_mode)
                    mb = np.max(b)
                    basis_vectors.append(b / mb if mb > 0 else b)

            _, _, _, _, _, ss_res = self._solve_multi_component_linear(
                basis_vectors, wl_crop, I_crop_norm, fit_baseline=fit_baseline,
                weight_power=weight_power, custom_weights=custom_weights,
                wl_centered=wl_centered,
            )
            return ss_res

        # Global Optimization
        opt_res = differential_evolution(_objective, bounds=bounds, maxiter=maxiter, popsize=12, polish=True)

        # Final Extraction
        best_p = opt_res.x
        p_idx = 0
        g_shift = best_p[p_idx] if fit_global_shift else 0.0
        if fit_global_shift:
            p_idx += 1

        basis_vectors = []
        species_results = {}

        for cfg in species_configs:
            sp_name = cfg["name"]
            lib = self.loaded_libraries.get(sp_name)
            if not lib:
                continue

            if cfg.get("two_trot", False):
                tr1, tr2 = best_p[p_idx], best_p[p_idx + 1]
                p_idx += 2
                tv = best_p[p_idx] if (cfg.get("fit_tvib", False) and lib["is_3d"]) else 3000.0
                if cfg.get("fit_tvib", False) and lib["is_3d"]:
                    p_idx += 1

                b1 = self._interpolate_spectrum(wl_crop + g_shift, tr1, tv, lib, interp_mode)
                b2 = self._interpolate_spectrum(wl_crop + g_shift, tr2, tv, lib, interp_mode)
                m1, m2 = np.max(b1), np.max(b2)

                basis_vectors.append(b1 / m1 if m1 > 0 else b1)
                basis_vectors.append(b2 / m2 if m2 > 0 else b2)

                species_results[sp_name] = {
                    "two_trot": True, "T_rot_1": float(tr1), "T_rot_2": float(tr2),
                    "T_vib": float(tv), "shift_nm": float(g_shift),
                }
            else:
                tr = cfg.get("fixed_trot", best_p[p_idx] if "fixed_trot" not in cfg else cfg["fixed_trot"])
                if "fixed_trot" not in cfg:
                    p_idx += 1

                tv = best_p[p_idx] if (cfg.get("fit_tvib", False) and lib["is_3d"]) else 3000.0
                if cfg.get("fit_tvib", False) and lib["is_3d"]:
                    p_idx += 1

                b = self._interpolate_spectrum(wl_crop + g_shift, tr, tv, lib, interp_mode)
                mb = np.max(b)
                basis_vectors.append(b / mb if mb > 0 else b)

                species_results[sp_name] = {
                    "two_trot": False, "T_rot": float(tr), "T_vib": float(tv),
                    "shift_nm": float(g_shift),
                }

        amps, quad, slope, offset, I_fit_norm, ss_res = self._solve_multi_component_linear(
            basis_vectors, wl_crop, I_crop_norm, fit_baseline=fit_baseline,
            weight_power=weight_power, custom_weights=custom_weights,
            wl_centered=wl_centered,
        )

        # Unpack Amplitudes and Compute Integrated Areas
        amp_idx = 0
        for sp_name, details in species_results.items():
            if details["two_trot"]:
                a1, a2 = amps[amp_idx], amps[amp_idx + 1]
                amp_idx += 2
                tot_a = a1 + a2
                details["amplitude_1"] = float(a1)
                details["amplitude_2"] = float(a2)
                details["total_amplitude"] = float(tot_a)
                details["ratio_1"] = float(a1 / tot_a) if tot_a > 0 else 0.5

                b1 = basis_vectors[amp_idx - 2]
                b2 = basis_vectors[amp_idx - 1]
                area1 = float(trapz_fn(a1 * b1, wl_crop))
                area2 = float(trapz_fn(a2 * b2, wl_crop))
                details["area_1"] = area1
                details["area_2"] = area2
                details["total_integrated_area"] = area1 + area2
            else:
                a = amps[amp_idx]
                amp_idx += 1
                details["amplitude"] = float(a)
                details["total_amplitude"] = float(a)
                b = basis_vectors[amp_idx - 1]
                details["integrated_area"] = float(trapz_fn(a * b, wl_crop))
                details["total_integrated_area"] = details["integrated_area"]

        fit_res = {
            "wl_crop": wl_crop,
            "I_exp_norm": I_crop_norm,
            "I_fit_norm": I_fit_norm,
            "shift_nm": float(g_shift),
            "baseline_slope": float(slope),
            "baseline_offset": float(offset),
            "species_results": species_results,
            "ss_res": float(ss_res),
        }

        fit_res["relative_scale_constants"] = compute_relative_scale_constants(fit_res)
        return fit_res