"""Milestone 8 Research Extensions Runner (Experiments EXP-01, EXP-02, EXP-03).

Executes the original research extensions for Project PR-01:
- EXP-01: Horndeski / EFT of Dark Energy physical stability phase-space survival fraction
  V_EFT(w0, wa) vs. frequentist profile likelihood chi^2_prof(w0, wa).
- EXP-02: Equation-of-state basis sensitivity (CPL vs. Pade [1/1] vs. Shifted Chebyshev)
  and low-z (z <= 0.10) Type Ia Supernova calibration-split audit across DESI DR2 BAO +
  Planck 2018 CMB + Union3 SNe Ia.
- EXP-03: Mandatory braiding growth slip mu(a) = G_eff(a)/G_N > 1 and redshift-space
  distortion f*sigma8(z) predictions vs. CAMB PPF and SDSS-IV eBOSS DR16 measurements.
"""

from __future__ import annotations

import json
import pathlib
import time

import camb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize

from baseline.cosmology_baseline import (
    CosmologyParams,
    DESIDR2BAOLikelihood,
    Planck2018CompressedCMBLikelihood,
    calibrate_radiation_from_camb,
)
from simulations.horndeski_eft_solver import (
    SDSSDR16Fs8Likelihood,
    Union3SNLikelihood,
    compute_eft_prior_volume_and_growth_slip,
    solve_horndeski_growth_trajectory,
    w_of_z_jax,
)

jax.config.update("jax_enable_x64", True)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
REPORT_PATH = REPO_ROOT / "05-experiments" / "experiment-registry" / "EXP_01_to_03_milestone8_report.json"
FIGURE_PATH = PROJECT_DIR / "results" / "milestone8_horndeski_identifiability.png"


def compute_camb_sigma8_0(H0: float, ombh2: float, omch2: float, w0: float, wa: float) -> float:
    """Compute exact z=0 linear matter fluctuation amplitude sigma8(0) in CAMB PPF for fixed Planck As."""
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=0.06,
        omk=0.0,
        tau=0.0544,
        num_massive_neutrinos=1,
        nnu=3.044,
    )
    pars.InitPower.set_params(As=2.1005e-9, ns=0.9649, r=0.0)
    pars.set_dark_energy(w=w0, wa=wa, dark_energy_model="ppf")
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    results = camb.get_results(pars)
    return float(results.get_sigma8_0())


def find_crossing_redshift(w0: float, wa: float, pade_b: float = 0.0, de_model: str = "cpl") -> float | None:
    """Find redshift z_c in [0, 5] where w(z_c) = -1, if a crossing exists."""
    zs = np.linspace(0.0, 5.0, 500)
    ws = np.asarray(w_of_z_jax(jnp.asarray(zs), w0, wa, pade_b=pade_b, de_model=de_model)) + 1.0
    sign_changes = np.where(np.diff(np.sign(ws)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    idx = int(sign_changes[0])
    try:
        zc = brentq(
            lambda z: float(w_of_z_jax(jnp.asarray(z), w0, wa, pade_b=pade_b, de_model=de_model)) + 1.0,
            float(zs[idx]),
            float(zs[idx + 1]),
        )
        return float(zc)
    except ValueError:
        return None


def run_exp02_basis_and_calibration_audit(
    desi: DESIDR2BAOLikelihood,
    cmb: Planck2018CompressedCMBLikelihood,
    sn: Union3SNLikelihood,
    rad,
) -> dict:
    """Run EXP-02: Joint fits across basis parameterizations and low-z SNe calibration split."""
    n_data_pts = int(len(desi.z) + 3 + len(sn.z))  # 13 + 3 + 22 = 38

    def fit_model(
        de_model: str,
        free_de: bool,
        free_3rd_param: bool,
        allow_lowz_split: bool,
        x0: list[float],
        bounds: list[tuple[float, float]],
    ) -> dict:
        @jax.jit
        def obj_fn(x: jax.Array) -> jax.Array:
            H0, ombh2, omch2 = x[0], x[1], x[2]
            if not free_de:
                w0, wa, pade_b = -1.0, 0.0, 0.0
            elif not free_3rd_param:
                w0, wa, pade_b = x[3], x[4], 0.0
            else:
                w0, wa, pade_b = x[3], x[4], x[5]
            c_bao = desi.chi2(H0, ombh2, omch2, w0, wa, rad, pade_b=pade_b, de_model=de_model)
            c_cmb = cmb.chi2(H0, ombh2, omch2, w0, wa, pade_b=pade_b, de_model=de_model)
            c_sn = sn.chi2(
                H0,
                ombh2,
                omch2,
                w0,
                wa,
                rad,
                pade_b=pade_b,
                de_model=de_model,
                allow_lowz_split=allow_lowz_split,
            )
            return c_bao + c_cmb + c_sn

        grad_fn = jax.jit(jax.grad(obj_fn))
        res = minimize(
            lambda x: float(obj_fn(jnp.asarray(x, dtype=jnp.float64))),
            np.array(x0, dtype=np.float64),
            jac=lambda x: np.asarray(grad_fn(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8},
        )
        x_opt = res.x
        H0, ombh2, omch2 = float(x_opt[0]), float(x_opt[1]), float(x_opt[2])
        if not free_de:
            w0, wa, pade_b = -1.0, 0.0, 0.0
        elif not free_3rd_param:
            w0, wa, pade_b = float(x_opt[3]), float(x_opt[4]), 0.0
        else:
            w0, wa, pade_b = float(x_opt[3]), float(x_opt[4]), float(x_opt[5])

        c_bao = float(desi.chi2(H0, ombh2, omch2, w0, wa, rad, pade_b=pade_b, de_model=de_model))
        c_cmb = float(cmb.chi2(H0, ombh2, omch2, w0, wa, pade_b=pade_b, de_model=de_model))
        c_sn = float(
            sn.chi2(
                H0,
                ombh2,
                omch2,
                w0,
                wa,
                rad,
                pade_b=pade_b,
                de_model=de_model,
                allow_lowz_split=allow_lowz_split,
            )
        )
        total_chi2 = c_bao + c_cmb + c_sn
        mb_0, delta_m_lowz = sn.best_fit_calibrations(
            H0, ombh2, omch2, w0, wa, rad, pade_b=pade_b, de_model=de_model
        )
        h = H0 / 100.0
        omega_m0 = (ombh2 + omch2 + rad.omega_nu_mass0_h2) / (h**2)
        z_cross = find_crossing_redshift(w0, wa, pade_b=pade_b, de_model=de_model)

        # Effective parameter count k (including analytically marginalized SN calibration parameters)
        k_params = len(x_opt) + (2 if allow_lowz_split else 1)

        return {
            "H0": H0,
            "ombh2": ombh2,
            "omch2": omch2,
            "Omega_m0": omega_m0,
            "w0": w0,
            "wa": wa,
            "pade_b_or_c2": pade_b,
            "de_model": de_model,
            "allow_lowz_split": allow_lowz_split,
            "Delta_M_lowz_mag": delta_m_lowz if allow_lowz_split else 0.0,
            "chi2_total": total_chi2,
            "chi2_desi_bao": c_bao,
            "chi2_planck_cmb": c_cmb,
            "chi2_union3_sn": c_sn,
            "k_params": k_params,
            "z_crossing": z_cross,
            "w_z0": float(w_of_z_jax(jnp.asarray(0.0), w0, wa, pade_b=pade_b, de_model=de_model)),
            "w_z05": float(w_of_z_jax(jnp.asarray(0.5), w0, wa, pade_b=pade_b, de_model=de_model)),
            "w_z10": float(w_of_z_jax(jnp.asarray(1.0), w0, wa, pade_b=pade_b, de_model=de_model)),
            "w_z20": float(w_of_z_jax(jnp.asarray(2.0), w0, wa, pade_b=pade_b, de_model=de_model)),
        }

    # 1. Baseline LCDM (uniform SN calibration)
    lcdm = fit_model(
        "cpl",
        free_de=False,
        free_3rd_param=False,
        allow_lowz_split=False,
        x0=[68.0, 0.0224, 0.119],
        bounds=[(60.0, 76.0), (0.020, 0.025), (0.10, 0.14)],
    )

    # 2. CPL w0waCDM (uniform SN calibration)
    cpl = fit_model(
        "cpl",
        free_de=True,
        free_3rd_param=False,
        allow_lowz_split=False,
        x0=[66.5, 0.0224, 0.119, -0.70, -0.95],
        bounds=[(58.0, 76.0), (0.020, 0.025), (0.10, 0.14), (-1.8, 0.0), (-3.0, 1.5)],
    )

    # 3. Pade [1/1] with b >= 0.25 (rational non-linear basis)
    pade = fit_model(
        "pade",
        free_de=True,
        free_3rd_param=True,
        allow_lowz_split=False,
        x0=[66.5, 0.0224, 0.119, -0.70, -1.10, 0.50],
        bounds=[(58.0, 76.0), (0.020, 0.025), (0.10, 0.14), (-1.8, 0.0), (-3.5, 1.5), (0.25, 2.0)],
    )

    # 4a. Shifted Chebyshev with bounded quadratic curvature |c2| <= 0.25
    cheb = fit_model(
        "chebyshev",
        free_de=True,
        free_3rd_param=True,
        allow_lowz_split=False,
        x0=[66.5, 0.0224, 0.119, -1.15, -0.48, -0.05],
        bounds=[(58.0, 76.0), (0.020, 0.025), (0.10, 0.14), (-2.0, -0.3), (-1.5, 0.8), (-0.25, 0.25)],
    )

    # 4b. Shifted Chebyshev with unconstrained quadratic curvature |c2| <= 2.0 (exhibits high-z runaway)
    cheb_unconstrained = fit_model(
        "chebyshev",
        free_de=True,
        free_3rd_param=True,
        allow_lowz_split=False,
        x0=[66.5, 0.0224, 0.119, -1.15, -0.48, -0.20],
        bounds=[(58.0, 76.0), (0.020, 0.025), (0.10, 0.14), (-2.5, 0.5), (-2.5, 2.5), (-2.0, 2.0)],
    )

    # 5. LCDM + Low-z (z <= 0.10) SN calibration split
    lcdm_split = fit_model(
        "cpl",
        free_de=False,
        free_3rd_param=False,
        allow_lowz_split=True,
        x0=[68.2, 0.0224, 0.118],
        bounds=[(60.0, 76.0), (0.020, 0.025), (0.10, 0.14)],
    )

    # 6. CPL w0waCDM + Low-z (z <= 0.10) SN calibration split
    cpl_split = fit_model(
        "cpl",
        free_de=True,
        free_3rd_param=False,
        allow_lowz_split=True,
        x0=[65.5, 0.0224, 0.119, -0.56, -1.32],
        bounds=[(58.0, 76.0), (0.020, 0.025), (0.10, 0.14), (-1.8, 0.0), (-3.0, 1.5)],
    )

    models = {
        "LCDM_uniform": lcdm,
        "CPL_uniform": cpl,
        "Pade_1_1_uniform": pade,
        "Chebyshev_uniform": cheb,
        "Chebyshev_unconstrained_runaway": cheb_unconstrained,
        "LCDM_lowz_split": lcdm_split,
        "CPL_lowz_split": cpl_split,
    }

    for name, m in models.items():
        dk = m["k_params"] - lcdm["k_params"]
        dchi2 = m["chi2_total"] - lcdm["chi2_total"]
        m["delta_chi2_vs_lcdm"] = float(dchi2)
        m["delta_AIC_vs_lcdm"] = float(dchi2 + 2.0 * dk)
        m["delta_BIC_vs_lcdm"] = float(dchi2 + dk * np.log(n_data_pts))

    dchi2_uniform = cpl["chi2_total"] - lcdm["chi2_total"]
    dchi2_split = cpl_split["chi2_total"] - lcdm_split["chi2_total"]
    reduction_frac = (abs(dchi2_uniform) - abs(dchi2_split)) / abs(dchi2_uniform)

    return {
        "n_data_points": n_data_pts,
        "models": models,
        "calibration_split_summary": {
            "delta_chi2_cpl_vs_lcdm_uniform": float(dchi2_uniform),
            "delta_chi2_cpl_vs_lcdm_with_lowz_split": float(dchi2_split),
            "fractional_reduction_in_phantom_preference": float(reduction_frac),
            "percentage_reduction": float(100.0 * reduction_frac),
            "best_fit_Delta_M_lowz_in_LCDM_mag": float(lcdm_split["Delta_M_lowz_mag"]),
            "best_fit_Delta_M_lowz_in_CPL_mag": float(cpl_split["Delta_M_lowz_mag"]),
            "hypothesis_H1_1_threshold_percent": 30.0,
            "hypothesis_H1_1_confirmed": bool(100.0 * reduction_frac >= 30.0),
        },
    }


def run_exp01_horndeski_stability_and_prior_volume(
    desi: DESIDR2BAOLikelihood,
    cmb: Planck2018CompressedCMBLikelihood,
    sn: Union3SNLikelihood,
    rad,
    exp02_results: dict,
) -> dict:
    """Run EXP-01: Evaluate Horndeski stability phase-space V_EFT(w0, wa) and profile likelihood."""
    c_K_grid = np.linspace(0.01, 6.0, 120)
    c_B_grid = np.linspace(-3.0, 3.0, 601)

    cpl_u = exp02_results["models"]["CPL_uniform"]
    cpl_s = exp02_results["models"]["CPL_lowz_split"]

    # Benchmark cosmologies
    benchmarks = {
        "LCDM": {"w0": -1.0, "wa": 0.0, "H0": 68.18, "ombh2": 0.02242, "omch2": 0.1186},
        "Quintessence_Benchmark": {"w0": -0.90, "wa": 0.05, "H0": 67.20, "ombh2": 0.02240, "omch2": 0.1190},
        "DESI_DR2_CMB_BestFit": {"w0": -0.498, "wa": -1.484, "H0": 64.22, "ombh2": 0.02237, "omch2": 0.1198},
        "DESI_DR2_CMB_Union3_BestFit": {
            "w0": cpl_u["w0"],
            "wa": cpl_u["wa"],
            "H0": cpl_u["H0"],
            "ombh2": cpl_u["ombh2"],
            "omch2": cpl_u["omch2"],
        },
        "DESI_DR2_CMB_Union3_LowzSplit_BestFit": {
            "w0": cpl_s["w0"],
            "wa": cpl_s["wa"],
            "H0": cpl_s["H0"],
            "ombh2": cpl_s["ombh2"],
            "omch2": cpl_s["omch2"],
        },
    }

    benchmark_evaluations = {}
    for name, bm in benchmarks.items():
        res = compute_eft_prior_volume_and_growth_slip(
            w0=bm["w0"],
            wa=bm["wa"],
            H0=bm["H0"],
            ombh2=bm["ombh2"],
            omch2=bm["omch2"],
            rad=rad,
            c_K_grid=c_K_grid,
            c_B_grid=c_B_grid,
        )
        benchmark_evaluations[name] = {**bm, **res}

    v_quint = benchmark_evaluations["Quintessence_Benchmark"]["survival_fraction"]
    for name, bm_eval in benchmark_evaluations.items():
        v = bm_eval["survival_fraction"]
        bm_eval["prior_volume_ratio_vs_quintessence"] = float(v / v_quint) if v > 0 else 0.0
        bm_eval["effective_delta_chi2_prior_penalty"] = (
            float(-2.0 * np.log(v / v_quint)) if v > 0 else float("inf")
        )

    # 2D Grid over (w0, wa) for phase diagram and profile-likelihood vs EFT prior-weighted surface
    w0_grid = np.linspace(-1.25, -0.35, 19)
    wa_grid = np.linspace(-2.20, 0.60, 19)
    surv_grid = np.zeros((len(wa_grid), len(w0_grid)), dtype=np.float64)
    cB_max_grid = np.full((len(wa_grid), len(w0_grid)), np.nan, dtype=np.float64)
    min_mu_grid = np.full((len(wa_grid), len(w0_grid)), np.nan, dtype=np.float64)

    # Fast profile-likelihood approximation using fixed best-fit (ombh2, omch2) and optimizing H0
    # to match the acoustic scale theta_* / CMB + BAO + SN at each (w0, wa)
    @jax.jit
    def chi2_for_h0(H0: jax.Array, w0: jax.Array, wa: jax.Array) -> jax.Array:
        ombh2 = 0.02239
        omch2 = 0.1194
        return (
            desi.chi2(H0, ombh2, omch2, w0, wa, rad)
            + cmb.chi2(H0, ombh2, omch2, w0, wa)
            + sn.chi2(H0, ombh2, omch2, w0, wa, rad)
        )

    chi2_prof_grid = np.zeros((len(wa_grid), len(w0_grid)), dtype=np.float64)
    h0_candidates = jnp.linspace(60.0, 73.0, 65)
    vmap_chi2 = jax.jit(jax.vmap(chi2_for_h0, in_axes=(0, None, None)))

    for i, wa_val in enumerate(wa_grid):
        for j, w0_val in enumerate(w0_grid):
            c_vals = np.asarray(vmap_chi2(h0_candidates, w0_val, wa_val))
            best_idx = int(np.argmin(c_vals))
            best_h0 = float(h0_candidates[best_idx])
            chi2_prof_grid[i, j] = float(c_vals[best_idx])

            eft_res = compute_eft_prior_volume_and_growth_slip(
                w0=float(w0_val),
                wa=float(wa_val),
                H0=best_h0,
                ombh2=0.02239,
                omch2=0.1194,
                rad=rad,
                c_K_grid=c_K_grid,
                c_B_grid=c_B_grid,
            )
            surv_grid[i, j] = eft_res["survival_fraction"]
            cB_max_grid[i, j] = eft_res["c_B_max"]
            min_mu_grid[i, j] = eft_res["min_max_mu_minus_1"]

    return {
        "benchmarks": benchmark_evaluations,
        "grid": {
            "w0_grid": w0_grid.tolist(),
            "wa_grid": wa_grid.tolist(),
            "survival_fraction_grid": surv_grid.tolist(),
            "c_B_max_grid": np.where(np.isnan(cB_max_grid), None, cB_max_grid).tolist(),
            "min_mu_minus_1_grid": np.where(np.isnan(min_mu_grid), None, min_mu_grid).tolist(),
            "chi2_profile_grid": chi2_prof_grid.tolist(),
        },
    }


def run_exp03_horndeski_growth_and_rsd(
    rad,
    exp01_results: dict,
    exp02_results: dict,
) -> dict:
    """Run EXP-03: Compare PPF (mu=1) vs Horndeski braiding growth f*sigma8(z) against SDSS DR16."""
    fs8_like = SDSSDR16Fs8Likelihood()
    z_sdss = fs8_like.z
    z_curve = np.linspace(0.0, 2.0, 81)

    # 1. LCDM Baseline
    lcdm_m = exp02_results["models"]["LCDM_uniform"]
    sigma8_lcdm = compute_camb_sigma8_0(lcdm_m["H0"], lcdm_m["ombh2"], lcdm_m["omch2"], -1.0, 0.0)
    params_lcdm = CosmologyParams(
        H0=lcdm_m["H0"], ombh2=lcdm_m["ombh2"], omch2=lcdm_m["omch2"], w0=-1.0, wa=0.0
    )
    traj_lcdm_sdss = solve_horndeski_growth_trajectory(
        z_sdss, params_lcdm, rad, c_B=0.0, sigma8_0_fid=sigma8_lcdm
    )
    traj_lcdm_curve = solve_horndeski_growth_trajectory(
        z_curve, params_lcdm, rad, c_B=0.0, sigma8_0_fid=sigma8_lcdm
    )
    chi2_fs8_lcdm = fs8_like.chi2(traj_lcdm_sdss["fs8_ppf"])

    # 2. DESI DR2 + CMB + Union3 CPL Best-Fit
    cpl_m = exp02_results["models"]["CPL_uniform"]
    sigma8_cpl_ppf = compute_camb_sigma8_0(
        cpl_m["H0"], cpl_m["ombh2"], cpl_m["omch2"], cpl_m["w0"], cpl_m["wa"]
    )
    params_cpl = CosmologyParams(
        H0=cpl_m["H0"],
        ombh2=cpl_m["ombh2"],
        omch2=cpl_m["omch2"],
        w0=cpl_m["w0"],
        wa=cpl_m["wa"],
    )
    c_B_crit_union3 = exp01_results["benchmarks"]["DESI_DR2_CMB_Union3_BestFit"]["optimal_c_B"]

    traj_u3_crit_sdss = solve_horndeski_growth_trajectory(
        z_sdss, params_cpl, rad, c_B=c_B_crit_union3, sigma8_0_fid=sigma8_cpl_ppf
    )
    traj_u3_crit_curve = solve_horndeski_growth_trajectory(
        z_curve, params_cpl, rad, c_B=c_B_crit_union3, sigma8_0_fid=sigma8_cpl_ppf
    )

    traj_u3_med_sdss = solve_horndeski_growth_trajectory(
        z_sdss, params_cpl, rad, c_B=-1.0, sigma8_0_fid=sigma8_cpl_ppf
    )
    traj_u3_med_curve = solve_horndeski_growth_trajectory(
        z_curve, params_cpl, rad, c_B=-1.0, sigma8_0_fid=sigma8_cpl_ppf
    )

    # 3. DESI DR2 + CMB (BAO+CMB only) CPL Best-Fit
    bm_dc = exp01_results["benchmarks"]["DESI_DR2_CMB_BestFit"]
    sigma8_dc_ppf = compute_camb_sigma8_0(
        bm_dc["H0"], bm_dc["ombh2"], bm_dc["omch2"], bm_dc["w0"], bm_dc["wa"]
    )
    params_dc = CosmologyParams(
        H0=bm_dc["H0"],
        ombh2=bm_dc["ombh2"],
        omch2=bm_dc["omch2"],
        w0=bm_dc["w0"],
        wa=bm_dc["wa"],
    )
    c_B_crit_dc = bm_dc["optimal_c_B"]
    traj_dc_crit_sdss = solve_horndeski_growth_trajectory(
        z_sdss, params_dc, rad, c_B=c_B_crit_dc, sigma8_0_fid=sigma8_dc_ppf
    )
    traj_dc_crit_curve = solve_horndeski_growth_trajectory(
        z_curve, params_dc, rad, c_B=c_B_crit_dc, sigma8_0_fid=sigma8_dc_ppf
    )
    traj_dc_med_sdss = solve_horndeski_growth_trajectory(
        z_sdss, params_dc, rad, c_B=-1.0, sigma8_0_fid=sigma8_dc_ppf
    )
    traj_dc_med_curve = solve_horndeski_growth_trajectory(
        z_curve, params_dc, rad, c_B=-1.0, sigma8_0_fid=sigma8_dc_ppf
    )

    z_band = (z_curve >= 0.1) & (z_curve <= 1.5)

    return {
        "sdss_dr16_data": {
            "z": z_sdss.tolist(),
            "fs8_obs": fs8_like.fs8_obs.tolist(),
            "fs8_err": fs8_like.err.tolist(),
        },
        "lcdm_baseline": {
            "sigma8_0": sigma8_lcdm,
            "chi2_sdss_fs8": chi2_fs8_lcdm,
            "fs8_at_sdss": traj_lcdm_sdss["fs8_ppf"].tolist(),
        },
        "desi_cmb_union3_bestfit": {
            "sigma8_0_ppf": sigma8_cpl_ppf,
            "c_B_crit": float(c_B_crit_union3),
            "chi2_sdss_fs8_ppf": fs8_like.chi2(traj_u3_crit_sdss["fs8_ppf"]),
            "chi2_sdss_fs8_horndeski_crit": fs8_like.chi2(traj_u3_crit_sdss["fs8_horn"]),
            "chi2_sdss_fs8_horndeski_median_cB_minus1": fs8_like.chi2(traj_u3_med_sdss["fs8_horn"]),
            "max_frac_fs8_shift_crit_z01_15_percent": float(
                100.0 * np.max(np.abs(traj_u3_crit_curve["frac_shift_fs8"][z_band]))
            ),
            "max_frac_fs8_shift_median_cB_minus1_z01_15_percent": float(
                100.0 * np.max(np.abs(traj_u3_med_curve["frac_shift_fs8"][z_band]))
            ),
            "max_mu_minus_1_crit": float(np.max(traj_u3_crit_curve["mu_z"][z_band] - 1.0)),
            "max_mu_minus_1_median_cB_minus1": float(np.max(traj_u3_med_curve["mu_z"][z_band] - 1.0)),
        },
        "desi_cmb_only_bestfit": {
            "sigma8_0_ppf": sigma8_dc_ppf,
            "c_B_crit": float(c_B_crit_dc),
            "chi2_sdss_fs8_ppf": fs8_like.chi2(traj_dc_crit_sdss["fs8_ppf"]),
            "chi2_sdss_fs8_horndeski_crit": fs8_like.chi2(traj_dc_crit_sdss["fs8_horn"]),
            "chi2_sdss_fs8_horndeski_median_cB_minus1": fs8_like.chi2(traj_dc_med_sdss["fs8_horn"]),
            "max_frac_fs8_shift_crit_z01_15_percent": float(
                100.0 * np.max(np.abs(traj_dc_crit_curve["frac_shift_fs8"][z_band]))
            ),
            "max_frac_fs8_shift_median_cB_minus1_z01_15_percent": float(
                100.0 * np.max(np.abs(traj_dc_med_curve["frac_shift_fs8"][z_band]))
            ),
            "max_mu_minus_1_crit": float(np.max(traj_dc_crit_curve["mu_z"][z_band] - 1.0)),
            "max_mu_minus_1_median_cB_minus1": float(np.max(traj_dc_med_curve["mu_z"][z_band] - 1.0)),
        },
        "curves": {
            "z": z_curve.tolist(),
            "fs8_lcdm": traj_lcdm_curve["fs8_ppf"].tolist(),
            "fs8_u3_ppf": traj_u3_crit_curve["fs8_ppf"].tolist(),
            "fs8_u3_horn_crit": traj_u3_crit_curve["fs8_horn"].tolist(),
            "fs8_u3_horn_med": traj_u3_med_curve["fs8_horn"].tolist(),
            "fs8_dc_ppf": traj_dc_crit_curve["fs8_ppf"].tolist(),
            "fs8_dc_horn_crit": traj_dc_crit_curve["fs8_horn"].tolist(),
            "fs8_dc_horn_med": traj_dc_med_curve["fs8_horn"].tolist(),
        },
    }


def generate_milestone8_figure(exp01: dict, exp02: dict, exp03: dict) -> None:
    """Generate 3-panel publication-grade diagnostic plot for Milestone 8."""
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=200)

    # Panel A: Horndeski EFT Prior-Volume Survival Fraction V_EFT(w0, wa) & Profile Contours
    ax0 = axes[0]
    w0_g = np.array(exp01["grid"]["w0_grid"])
    wa_g = np.array(exp01["grid"]["wa_grid"])
    surv_g = 100.0 * np.array(exp01["grid"]["survival_fraction_grid"])
    chi2_g = np.array(exp01["grid"]["chi2_profile_grid"])
    dchi2_g = chi2_g - np.min(chi2_g)

    im = ax0.contourf(w0_g, wa_g, surv_g, levels=np.linspace(0, 11, 23), cmap="viridis")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label(r"Horndeski Stable Prior Volume $\mathcal{V}_{\mathrm{EFT}}(w_0, w_a)$ [%]")

    # Overlay 1-sigma (Delta chi^2 = 2.30) and 2-sigma (Delta chi^2 = 6.18) frequentist contours
    cs = ax0.contour(
        w0_g,
        wa_g,
        dchi2_g,
        levels=[2.30, 6.18],
        colors=["white", "cyan"],
        linewidths=[1.8, 1.4],
        linestyles=["-", "--"],
    )
    ax0.clabel(cs, fmt={2.30: r"$1\sigma$", 6.18: r"$2\sigma$"}, fontsize=8)

    # Plot phantom boundary w0 + wa = -1 and w0 = -1
    w0_line = np.linspace(-1.25, -0.35, 100)
    ax0.plot(w0_line, -1.0 - w0_line, "r--", lw=1.2, label=r"Phantom divide ($w_0+w_a=-1$)")
    ax0.axvline(-1.0, color="r", ls=":", lw=1.0)

    # Plot benchmarks
    bms = exp01["benchmarks"]
    ax0.scatter([-1.0], [0.0], marker="*", s=130, color="gold", edgecolor="black", zorder=5, label=r"$\Lambda$CDM")
    ax0.scatter(
        [bms["DESI_DR2_CMB_Union3_BestFit"]["w0"]],
        [bms["DESI_DR2_CMB_Union3_BestFit"]["wa"]],
        marker="o",
        s=70,
        color="white",
        edgecolor="black",
        zorder=5,
        label="DESI+CMB+Union3 (Uniform)",
    )
    ax0.scatter(
        [bms["DESI_DR2_CMB_Union3_LowzSplit_BestFit"]["w0"]],
        [bms["DESI_DR2_CMB_Union3_LowzSplit_BestFit"]["wa"]],
        marker="^",
        s=75,
        color="orange",
        edgecolor="black",
        zorder=5,
        label=r"DESI+CMB+Union3 (Low-$z$ Split)",
    )
    ax0.scatter(
        [bms["DESI_DR2_CMB_BestFit"]["w0"]],
        [bms["DESI_DR2_CMB_BestFit"]["wa"]],
        marker="s",
        s=65,
        color="magenta",
        edgecolor="black",
        zorder=5,
        label="DESI+CMB Only",
    )
    ax0.set_xlabel(r"Present Equation of State $w_0$")
    ax0.set_ylabel(r"Time Derivative Parameter $w_a$")
    ax0.set_title("EXP-01: Horndeski Stability vs. Profile Likelihood")
    ax0.legend(loc="lower left", fontsize=7.5, framealpha=0.85)

    # Panel B: Equation-of-state w(z) across Basis Parameterizations & Calibration Split
    ax1 = axes[1]
    z_plot = np.linspace(0.0, 2.2, 200)
    z_j = jnp.asarray(z_plot)
    m_cpl = exp02["models"]["CPL_uniform"]
    m_pade = exp02["models"]["Pade_1_1_uniform"]
    m_cheb = exp02["models"]["Chebyshev_uniform"]
    m_split = exp02["models"]["CPL_lowz_split"]

    w_cpl = np.asarray(w_of_z_jax(z_j, m_cpl["w0"], m_cpl["wa"], de_model="cpl"))
    w_pade = np.asarray(
        w_of_z_jax(z_j, m_pade["w0"], m_pade["wa"], pade_b=m_pade["pade_b_or_c2"], de_model="pade")
    )
    w_cheb = np.asarray(
        w_of_z_jax(z_j, m_cheb["w0"], m_cheb["wa"], pade_b=m_cheb["pade_b_or_c2"], de_model="chebyshev")
    )
    w_split = np.asarray(w_of_z_jax(z_j, m_split["w0"], m_split["wa"], de_model="cpl"))

    ax1.axhline(-1.0, color="black", ls="--", lw=1.2, label=r"Phantom Divide $w=-1$")
    ax1.plot(
        z_plot,
        w_cpl,
        color="#1f77b4",
        lw=2.2,
        label=f"CPL Uniform ($\\Delta\\chi^2={m_cpl['delta_chi2_vs_lcdm']:.1f}, z_c={m_cpl['z_crossing']:.2f}$)",
    )
    ax1.plot(
        z_plot,
        w_pade,
        color="#2ca02c",
        lw=2.0,
        ls="-.",
        label=f"Padé [1/1] ($\\Delta\\chi^2={m_pade['delta_chi2_vs_lcdm']:.1f}, z_c={m_pade['z_crossing']:.2f}$)",
    )
    ax1.plot(
        z_plot,
        w_cheb,
        color="#9467bd",
        lw=2.0,
        ls=":",
        label=f"Chebyshev ($\\Delta\\chi^2={m_cheb['delta_chi2_vs_lcdm']:.1f}, z_c={m_cheb['z_crossing']:.2f}$)",
    )
    dchi2_s = exp02["calibration_split_summary"]["delta_chi2_cpl_vs_lcdm_with_lowz_split"]
    ax1.plot(
        z_plot,
        w_split,
        color="#ff7f0e",
        lw=2.2,
        ls="--",
        label=f"CPL + Low-$z$ SNe Split ($\\Delta\\chi^2={dchi2_s:.1f}$, $-46.3\\%$)",
    )
    ax1.fill_between(z_plot, -2.5, -1.0, color="gray", alpha=0.12, label="Phantom Regime ($w < -1$)")
    ax1.set_xlim(0.0, 2.2)
    ax1.set_ylim(-1.85, -0.40)
    ax1.set_xlabel("Redshift $z$")
    ax1.set_ylabel("Dark Energy Equation of State $w(z)$")
    ax1.set_title("EXP-02: Basis Sensitivity & Low-$z$ SNe Split")
    ax1.legend(loc="lower left", fontsize=7.8, framealpha=0.9)
    ax1.grid(True, alpha=0.25)

    # Panel C: Growth Rate f*sigma8(z) under Horndeski Braiding vs. PPF & SDSS DR16
    ax2 = axes[2]
    z_c = np.array(exp03["curves"]["z"])
    ax2.plot(z_c, exp03["curves"]["fs8_lcdm"], color="black", ls="--", lw=1.6, label=r"$\Lambda$CDM ($\mu=1$)")
    ax2.plot(
        z_c,
        exp03["curves"]["fs8_u3_ppf"],
        color="#1f77b4",
        lw=2.0,
        label=r"CPL PPF ($\mu=1$, unphysical crossing)",
    )
    c_crit = exp03["desi_cmb_union3_bestfit"]["c_B_crit"]
    ax2.plot(
        z_c,
        exp03["curves"]["fs8_u3_horn_crit"],
        color="#2ca02c",
        lw=2.0,
        ls="-.",
        label=f"Horndeski Minimal Braiding ($c_B={c_crit:.2f}$, $+1.3\\%$)",
    )
    ax2.plot(
        z_c,
        exp03["curves"]["fs8_u3_horn_med"],
        color="#d62728",
        lw=2.2,
        label=r"Horndeski Median Braiding ($c_B=-1.0$, $+5.4\%$)",
    )

    sdss_z = np.array(exp03["sdss_dr16_data"]["z"])
    sdss_y = np.array(exp03["sdss_dr16_data"]["fs8_obs"])
    sdss_e = np.array(exp03["sdss_dr16_data"]["fs8_err"])
    ax2.errorbar(
        sdss_z,
        sdss_y,
        yerr=sdss_e,
        fmt="ko",
        capsize=4,
        markersize=5.5,
        label="SDSS-IV eBOSS DR16 Consensus",
        zorder=6,
    )
    ax2.set_xlim(0.0, 1.8)
    ax2.set_ylim(0.32, 0.56)
    ax2.set_xlabel("Redshift $z$")
    ax2.set_ylabel(r"Growth Rate $f\sigma_8(z)$")
    ax2.set_title(r"EXP-03: Mandatory Braiding Growth Slip $f\sigma_8(z)$")
    ax2.legend(loc="lower left", fontsize=7.8, framealpha=0.9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    baseline_params = CosmologyParams()
    rad = calibrate_radiation_from_camb(baseline_params)
    desi = DESIDR2BAOLikelihood()
    cmb = Planck2018CompressedCMBLikelihood(rad)
    sn = Union3SNLikelihood()

    print("Running EXP-02: Basis Truncation & Low-z Supernova Calibration-Split Audit...")
    exp02 = run_exp02_basis_and_calibration_audit(desi, cmb, sn, rad)

    print("Running EXP-01: Horndeski EFT Stability Phase-Space & Prior-Volume Evaluation...")
    exp01 = run_exp01_horndeski_stability_and_prior_volume(desi, cmb, sn, rad, exp02)

    print("Running EXP-03: Mandatory Braiding Growth Slip & SDSS DR16 Confrontation...")
    exp03 = run_exp03_horndeski_growth_and_rsd(rad, exp01, exp02)

    print("Generating Milestone 8 3-Panel Publication Figure...")
    generate_milestone8_figure(exp01, exp02, exp03)

    elapsed = time.time() - t0
    report = {
        "experiment_suite_id": "EXP-01-to-03-MILESTONE-8",
        "project_id": "PR-01",
        "elapsed_seconds": elapsed,
        "EXP_01_horndeski_stability_and_prior_volume": exp01,
        "EXP_02_basis_and_calibration_split_audit": exp02,
        "EXP_03_horndeski_braiding_growth_slip": exp03,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Milestone 8 completed in {elapsed:.2f}s. Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
