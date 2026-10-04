import typing
import numpy as np
from scipy.optimize import minimize, nnls

try:
    from .Fit_One_Temp import MolecularFitter
except ImportError:
    try:
        from MolecularFitting.Fit_One_Temp import MolecularFitter
    except ImportError:
        from Fit_One_Temp import MolecularFitter

trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))


class MultiComponentFitSolver:
    """
    Generalized framework for deconvolving overlapping OH and N2 emissions in ROI.
    Optimizes Trot and Tvib jointly for both species alongside a global shift.
    """

    def __init__(self, fitter: typing.Optional[MolecularFitter] = None):
        self.fitter = fitter if fitter is not None else MolecularFitter()

    def _resolve_band_key(self, key: str) -> str:
        if key in self.fitter.loaded_libraries:
            return key
        aliases = {"14N2": "14N-14N", "14N-14N": "14N2", "16O-1H": "16O-1H"}
        if key in aliases and aliases[key] in self.fitter.loaded_libraries:
            return aliases[key]
        for lib_key in self.fitter.loaded_libraries.keys():
            if key.lower() in lib_key.lower():
                return lib_key
        raise KeyError(f"Target band '{key}' not in libraries: {list(self.fitter.loaded_libraries.keys())}")

    def synthesize_band_shape(
        self, target_band: str, T_rot: float, T_vib: float, wl_eval: np.ndarray
    ) -> np.ndarray:
        resolved_key = self._resolve_band_key(target_band)
        shape = self.fitter._interpolate_spectrum(
            wl_eval=wl_eval, T_rot=T_rot, T_vib=T_vib, target_band=resolved_key
        )
        max_val = np.max(shape)
        return shape / max_val if max_val > 0 else shape

    def fit_merged_spectrum(
        self,
        wl_roi: np.ndarray,
        I_exp_roi: np.ndarray,
        species_params: typing.Dict[str, typing.Dict[str, typing.Any]],
        shift_range: typing.Tuple[float, float] = (-1.0, 1.0),
    ) -> typing.Dict[str, typing.Any]:
        species_keys = list(species_params.keys())
        if not species_keys:
            raise ValueError("species_params dictionary cannot be empty.")

        I_max = np.max(I_exp_roi)
        I_norm = (I_exp_roi / I_max) if I_max > 0 else I_exp_roi

        # Feature-focused regional weighting to prevent zeroing out under high quenching
        weights = np.ones_like(wl_roi)
        weights[(wl_roi >= 305.5) & (wl_roi <= 310.5)] = 3.0
        if any(k in species_keys for k in ["14N2", "14N-14N"]):
            weights[(wl_roi >= 314.0) & (wl_roi <= 317.0)] = 3.0

        baseline = np.ones_like(wl_roi)

        p_temp_init = []
        bounds = []
        for key in species_keys:
            cfg = species_params[key]
            p_temp_init.extend([cfg.get("Trot", 1500.0), cfg.get("Tvib", 3000.0)])
            rot_bnd = cfg.get("Trot_bounds", (250.0, 6000.0))
            vib_bnd = cfg.get("Tvib_bounds", (300.0, 25000.0))
            bounds.extend([rot_bnd, vib_bnd])

        # Step 1: Broad Coarse Pre-Alignment for Wavelength Shift
        shift_grid = np.linspace(shift_range[0], shift_range[1], 201)
        best_shift = 0.0
        min_rnorm = float("inf")

        for s_test in shift_grid:
            wl_shifted = wl_roi - s_test
            design_cols = []
            for idx, key in enumerate(species_keys):
                t_rot, t_vib = p_temp_init[2 * idx], p_temp_init[2 * idx + 1]
                cols_shape = self.synthesize_band_shape(key, t_rot, t_vib, wl_shifted)
                design_cols.append(cols_shape * weights)
            design_cols.append(baseline * weights)
            A_grid = np.column_stack(design_cols)
            _, rnorm = nnls(A_grid, I_norm * weights)
            if rnorm < min_rnorm:
                min_rnorm = rnorm
                best_shift = s_test

        p_init = p_temp_init + [best_shift]
        bounds.append(shift_range)

        # Step 2: Joint Nelder-Mead Optimization for Temperatures + Shift
        def objective(params):
            shift = params[-1]
            wl_shifted = wl_roi - shift
            design_cols = []

            for idx, key in enumerate(species_keys):
                t_rot, t_vib = params[2 * idx], params[2 * idx + 1]
                shape = self.synthesize_band_shape(key, t_rot, t_vib, wl_shifted)
                design_cols.append(shape * weights)

            design_cols.append(baseline * weights)
            A = np.column_stack(design_cols)
            _, rnorm = nnls(A, I_norm * weights)
            return rnorm

        opt_res = minimize(
            objective,
            p_init,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 800, "xatol": 1e-2, "fatol": 1e-3},
        )
        p_opt = opt_res.x
        opt_shift = p_opt[-1]

        # Step 3: Physical Reconstruction using Joint Optimal Parameters
        wl_final = wl_roi - opt_shift
        design_cols_opt = []
        templates_opt = {}
        temperatures_opt = {}

        for idx, key in enumerate(species_keys):
            t_rot_opt, t_vib_opt = p_opt[2 * idx], p_opt[2 * idx + 1]
            temperatures_opt[key] = {"Trot": t_rot_opt, "Tvib": t_vib_opt}

            shape = self.synthesize_band_shape(key, t_rot_opt, t_vib_opt, wl_final)
            design_cols_opt.append(shape)
            templates_opt[key] = shape

        design_cols_opt.append(baseline)
        A_opt = np.column_stack(design_cols_opt)
        coeffs, _ = nnls(A_opt, I_norm)

        amplitudes = {}
        components = {}
        integrals = {}
        I_fit_total = np.zeros_like(wl_roi)

        for idx, key in enumerate(species_keys):
            amp = coeffs[idx] * I_max
            comp = amp * templates_opt[key]

            amplitudes[key] = amp
            components[key] = comp
            integrals[key] = float(trapz_fn(comp, wl_roi))
            I_fit_total += comp

        c_base = coeffs[-1] * I_max
        I_fit_total += c_base

        ss_res = float(np.sum((I_exp_roi - I_fit_total) ** 2))
        ss_tot = float(np.sum((I_exp_roi - np.mean(I_exp_roi)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "amplitudes": amplitudes,
            "temperatures": temperatures_opt,
            "components": components,
            "integrals": integrals,
            "baseline": c_base,
            "shift_nm": opt_shift,
            "r2": r2,
            "I_fit_total": I_fit_total,
        }


# Alias for backward compatibility
JointFitSolver = MultiComponentFitSolver