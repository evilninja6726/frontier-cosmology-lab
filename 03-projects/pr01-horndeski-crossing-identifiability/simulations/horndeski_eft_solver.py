"""Horndeski / EFT of Dark Energy Stability, Prior-Volume, Basis-Sensitivity, and Growth Solver.

Implements the mathematical and numerical core for Milestone 8 (Experiments EXP-01, EXP-02, EXP-03):
1. Union3 Type Ia Supernova likelihood (Rubin et al. 2023, arXiv:2311.12098) with exact
   analytical marginalization over absolute magnitude M_B, plus optional low-z (z <= 0.10)
   calibration zero-point split Delta_M_lowz.
2. Exact scale-invariant Horndeski / EFT of Dark Energy physical stability evaluation:
   - No-ghost condition: D(z)/Omega_DE(z) = c_K + 1.5 * c_B^2 * Omega_DE(z) > 0
   - Gradient & sub-luminal stability: 0 < c_s^2(z) <= 1 across z in [0, 1100]
   - Prior-volume survival fraction V_EFT(w0, wa) over physical EFT parameter space (c_K, c_B).
3. Exact critical braiding threshold c_B_crit(w0, wa), minimum gravitational coupling
   enhancement mu_min(a) = G_eff(a)/G_N >= 1, and induced redshift-space distortion
   growth rate f*sigma8(z) vs. SDSS DR16 LRG + QSO measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d

from baseline.cosmology_baseline import (
    CosmologyParams,
    RadiationCalibration,
    comoving_distance_dm_jax,
    e_of_z_jax,
    f_de_chebyshev_jax,
    f_de_cpl_jax,
    f_de_pade_jax,
    solve_subhorizon_growth,
)

jax.config.update("jax_enable_x64", True)

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "baseline" / "data"


class Union3SNLikelihood:
    """Official Union3 binned Type Ia Supernova likelihood (22 redshift bins, z in [0.05, 2.26]).

    Reference: Rubin et al. (2023), arXiv:2311.12098.
    Implements exact analytical marginalization over the nuisance offset M_B (and optional
    low-redshift z <= 0.10 split offset Delta_M_lowz).
    """

    def __init__(self) -> None:
        lc_file = DATA_DIR / "union3_lcparam_full.txt"
        cov_file = DATA_DIR / "union3_mag_covmat.txt"

        zs, mbs = [], []
        for line in lc_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            zs.append(float(parts[1]))
            mbs.append(float(parts[4]))

        self.z = jnp.asarray(zs, dtype=jnp.float64)
        self.mb_obs = jnp.asarray(mbs, dtype=jnp.float64)

        cov_lines = [
            float(x.strip())
            for x in cov_file.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        n_bins = int(cov_lines[0])
        cov = np.array(cov_lines[1 : 1 + n_bins * n_bins], dtype=np.float64).reshape((n_bins, n_bins))
        inv_cov = np.linalg.inv(cov)
        self.cov = jnp.asarray(cov, dtype=jnp.float64)
        self.inv_cov = jnp.asarray(inv_cov, dtype=jnp.float64)

        ones = np.ones((n_bins, 1), dtype=np.float64)
        c_inv_1 = inv_cov @ ones
        denom = float((ones.T @ c_inv_1)[0, 0])
        proj_single = inv_cov - (c_inv_1 @ c_inv_1.T) / denom
        self.proj_single = jnp.asarray(proj_single, dtype=jnp.float64)

        lowz_indicator = (np.array(zs) <= 0.105).astype(np.float64).reshape((n_bins, 1))
        x_mat = np.hstack([ones, lowz_indicator])
        xt_cinv = x_mat.T @ inv_cov
        gram_inv = np.linalg.inv(xt_cinv @ x_mat)
        proj_split = inv_cov - (xt_cinv.T @ gram_inv @ xt_cinv)
        self.proj_split = jnp.asarray(proj_split, dtype=jnp.float64)
        self.x_mat = x_mat
        self.gram_inv = gram_inv
        self.xt_cinv = xt_cinv

    def chi2(
        self,
        H0: float | jax.Array,
        ombh2: float | jax.Array,
        omch2: float | jax.Array,
        w0: float | jax.Array,
        wa: float | jax.Array,
        rad: RadiationCalibration,
        pade_b: float | jax.Array = 0.0,
        de_model: str = "cpl",
        allow_lowz_split: bool = False,
    ) -> jax.Array:
        """Compute analytically marginalized Union3 SN Ia chi^2."""
        dm = comoving_distance_dm_jax(
            self.z, H0, ombh2, omch2, w0, wa, rad, pade_b=pade_b, de_model=de_model
        )
        dl = (1.0 + self.z) * dm
        mu_th = 5.0 * jnp.log10(dl) + 25.0
        res = self.mb_obs - mu_th
        proj = self.proj_split if allow_lowz_split else self.proj_single
        return res @ (proj @ res)

    def best_fit_calibrations(
        self,
        H0: float,
        ombh2: float,
        omch2: float,
        w0: float,
        wa: float,
        rad: RadiationCalibration,
        pade_b: float = 0.0,
        de_model: str = "cpl",
    ) -> tuple[float, float]:
        """Return GLS best-fit (M_B, Delta_M_lowz) in magnitudes for the low-z split model."""
        dm = np.asarray(
            comoving_distance_dm_jax(
                self.z, H0, ombh2, omch2, w0, wa, rad, pade_b=pade_b, de_model=de_model
            )
        )
        dl = (1.0 + np.asarray(self.z)) * dm
        mu_th = 5.0 * np.log10(dl) + 25.0
        res = np.asarray(self.mb_obs) - mu_th
        beta = self.gram_inv @ (self.xt_cinv @ res)
        return float(beta[0]), float(beta[1])


class SDSSDR16Fs8Likelihood:
    """Official SDSS-IV eBOSS DR16 consensus f*sigma8(z) measurements at z = 0.38, 0.51, 0.698, 1.48."""

    def __init__(self) -> None:
        lrg_dat = DATA_DIR / "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat"
        lrg_cov = np.loadtxt(DATA_DIR / "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8_covtot.txt")
        qso_dat = DATA_DIR / "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat"
        qso_cov = np.loadtxt(DATA_DIR / "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8_covtot.txt")

        # Extract f_sigma8 rows (indices 2, 5, 8 in LRG and index 2 in QSO)
        lrg_lines = [l.split() for l in lrg_dat.read_text().splitlines() if l.strip()]
        qso_lines = [l.split() for l in qso_dat.read_text().splitlines() if l.strip()]

        fs8_indices_lrg = [2, 5, 8]
        zs = [float(lrg_lines[i][0]) for i in fs8_indices_lrg] + [float(qso_lines[2][0])]
        vals = [float(lrg_lines[i][1]) for i in fs8_indices_lrg] + [float(qso_lines[2][1])]

        cov_sub_lrg = lrg_cov[np.ix_(fs8_indices_lrg, fs8_indices_lrg)]
        cov_qso_fs8 = float(qso_cov[2, 2])
        cov = np.zeros((4, 4), dtype=np.float64)
        cov[:3, :3] = cov_sub_lrg
        cov[3, 3] = cov_qso_fs8

        self.z = np.array(zs, dtype=np.float64)
        self.fs8_obs = np.array(vals, dtype=np.float64)
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)
        self.err = np.sqrt(np.diag(cov))

    def chi2(self, fs8_pred: np.ndarray) -> float:
        res = fs8_pred - self.fs8_obs
        return float(res @ (self.inv_cov @ res))


def f_de_model_jax(
    z: jax.Array,
    w0: float | jax.Array,
    wa: float | jax.Array,
    pade_b: float | jax.Array = 0.0,
    de_model: str = "cpl",
) -> jax.Array:
    """Evaluate normalized dark energy density f_DE(z) = rho_DE(z)/rho_DE(0) for any model."""
    if de_model == "cpl":
        return f_de_cpl_jax(z, w0, wa)
    if de_model == "pade":
        return f_de_pade_jax(z, w0, wa, pade_b)
    if de_model == "chebyshev":
        return f_de_chebyshev_jax(z, w0, wa, pade_b)
    raise ValueError(f"Unknown de_model: {de_model}")


def w_of_z_jax(
    z: jax.Array,
    w0: float | jax.Array,
    wa: float | jax.Array,
    pade_b: float | jax.Array = 0.0,
    de_model: str = "cpl",
) -> jax.Array:
    """Evaluate dark energy equation of state w(z) for CPL, Pade [1/1], or Chebyshev."""
    a = 1.0 / (1.0 + z)
    x = 1.0 - a
    if de_model == "cpl":
        return w0 + wa * x
    if de_model == "pade":
        return (w0 + wa * x) / (1.0 + pade_b * x)
    if de_model == "chebyshev":
        return w0 + wa * (2.0 * x - 1.0) + pade_b * (8.0 * (x**2) - 8.0 * x + 1.0)
    raise ValueError(f"Unknown de_model: {de_model}")


def compute_eft_prior_volume_and_growth_slip(
    w0: float,
    wa: float,
    H0: float,
    ombh2: float,
    omch2: float,
    rad: RadiationCalibration,
    c_K_grid: np.ndarray,
    c_B_grid: np.ndarray,
    z_check: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute scale-invariant Horndeski EFT stability fraction and gravitational slip bounds.

    Because alpha_K(a) = c_K * Omega_DE(a) and alpha_B(a) = c_B * Omega_DE(a) scale with
    Omega_DE(a) > 0, positivity of D(a) and N_s(a) = D(a)*c_s^2(a) across z in [0, 1100]
    is evaluated using the exact scale-invariant ratios D(a)/Omega_DE(a) and N_s(a)/Omega_DE(a).
    """
    if z_check is None:
        z_check = np.concatenate([np.linspace(0.0, 3.0, 80), np.geomspace(3.1, 1100.0, 40)])

    z_arr = np.asarray(z_check, dtype=np.float64)
    zp1 = 1.0 + z_arr
    a_arr = 1.0 / zp1
    h = H0 / 100.0
    h2 = h**2
    omega_cb0 = (ombh2 + omch2) / h2
    omega_g0 = rad.omega_g_h2 / h2
    omega_nu_rel0 = rad.omega_nu_rel_h2 / h2
    omega_nu_m0 = rad.omega_nu_mass0_h2 / h2
    omega_de0 = 1.0 - omega_cb0 - omega_g0 - omega_nu_rel0 - omega_nu_m0

    f_de = (zp1 ** (3.0 * (1.0 + w0 + wa))) * np.exp(-3.0 * wa * z_arr / zp1)
    w_z = w0 + wa * (1.0 - a_arr)
    e2 = (
        (omega_cb0 + omega_nu_m0) * (zp1**3)
        + (omega_g0 + omega_nu_rel0) * (zp1**4)
        + omega_de0 * f_de
    )

    ode = omega_de0 * f_de / e2
    om_m = (omega_cb0 + omega_nu_m0) * (zp1**3) / e2
    om_r = np.maximum(0.0, 1.0 - om_m - ode)

    dlnH = -1.5 * (om_m + (4.0 / 3.0) * om_r + ode * (1.0 + w_z))
    dlnOde = -3.0 * (1.0 + w_z) - 2.0 * dlnH

    # Notice N_s / ode depends ONLY on c_B, not c_K!
    cB = c_B_grid[:, None]  # shape (n_B, 1)
    aB = cB * ode[None, :]  # shape (n_B, n_z)
    Ns_over_ode = (
        3.0 * (1.0 + w_z[None, :])
        + (1.0 - 0.5 * aB) * cB * (dlnH[None, :] - 1.0)
        - cB * dlnOde[None, :]
    )
    valid_num_B = np.all(Ns_over_ode > 1e-8, axis=-1) & np.all(np.abs(aB) < 1.95, axis=-1)

    if not np.any(valid_num_B):
        return {
            "survival_fraction": 0.0,
            "c_B_min": float("nan"),
            "c_B_max": float("nan"),
            "min_max_mu_minus_1": float("nan"),
            "median_max_mu_minus_1": float("nan"),
            "optimal_c_B": float("nan"),
        }

    # Analytical integration over c_K in [c_K_min_prior, c_K_max_prior]:
    # Since D(z)/ode(z) = c_K + 1.5 * c_B^2 * ode(z) is linear in c_K > 0,
    # sub-luminality Ns_over_ode <= D_over_ode holds across all z iff:
    # c_K >= max_z [ Ns_over_ode(c_B, z) - 1.5 * c_B^2 * ode(z) ].
    c_K_low = float(np.min(c_K_grid))
    c_K_high = float(np.max(c_K_grid))
    c_K_req = np.max(Ns_over_ode - 1.5 * (cB**2) * ode[None, :], axis=-1)  # shape (n_B,)
    frac_K_per_B = np.where(
        valid_num_B,
        np.clip((c_K_high - np.maximum(c_K_low, c_K_req)) / (c_K_high - c_K_low), 0.0, 1.0),
        0.0,
    )
    survival_frac = float(np.mean(frac_K_per_B))

    valid_B_with_K = valid_num_B & (frac_K_per_B > 0.0)
    if not np.any(valid_B_with_K):
        return {
            "survival_fraction": 0.0,
            "c_B_min": float("nan"),
            "c_B_max": float("nan"),
            "min_max_mu_minus_1": float("nan"),
            "median_max_mu_minus_1": float("nan"),
            "optimal_c_B": float("nan"),
        }

    valid_cBs = c_B_grid[valid_B_with_K]
    lowz = z_check <= 1.5
    # Recall mu(z) - 1 = c_B^2 * ode / (2 * Ns_over_ode) is INDEPENDENT of c_K!
    mu_m1_lowz = (valid_cBs[:, None] ** 2) * ode[None, lowz] / (2.0 * Ns_over_ode[valid_B_with_K][:, lowz])
    max_mu_per_B = np.max(mu_m1_lowz, axis=-1)
    best_b_idx = int(np.argmin(max_mu_per_B))

    return {
        "survival_fraction": survival_frac,
        "c_B_min": float(np.min(valid_cBs)),
        "c_B_max": float(np.max(valid_cBs)),
        "min_max_mu_minus_1": float(max_mu_per_B[best_b_idx]),
        "median_max_mu_minus_1": float(np.median(max_mu_per_B)),
        "optimal_c_B": float(valid_cBs[best_b_idx]),
    }


def solve_horndeski_growth_trajectory(
    z_eval: np.ndarray,
    params: CosmologyParams,
    rad: RadiationCalibration,
    c_B: float,
    sigma8_0_fid: float = 0.8111,
) -> dict[str, np.ndarray]:
    """Solve sub-horizon linear growth ODE for both PPF (mu=1) and Horndeski braiding c_B."""
    z_dense = np.concatenate([[0.0], np.geomspace(1e-4, 1100.0, 600)])
    a_dense = 1.0 / (1.0 + z_dense)
    z_j = jnp.asarray(z_dense, dtype=jnp.float64)
    zp1 = 1.0 + z_j
    h = params.H0 / 100.0
    h2 = h**2
    omega_cb0 = (params.ombh2 + params.omch2) / h2
    omega_g0 = rad.omega_g_h2 / h2
    omega_nu_rel0 = rad.omega_nu_rel_h2 / h2
    omega_nu_m0 = rad.omega_nu_mass0_h2 / h2
    omega_de0 = 1.0 - omega_cb0 - omega_g0 - omega_nu_rel0 - omega_nu_m0

    e_z = np.asarray(
        e_of_z_jax(
            z_j,
            params.H0,
            params.ombh2,
            params.omch2,
            params.w0,
            params.wa,
            rad,
            pade_b=params.pade_b,
            de_model=params.de_model,
        )
    )
    e2 = e_z**2
    f_de = np.asarray(f_de_model_jax(z_j, params.w0, params.wa, pade_b=params.pade_b, de_model=params.de_model))
    w_z = np.asarray(w_of_z_jax(z_j, params.w0, params.wa, pade_b=params.pade_b, de_model=params.de_model))
    ode = omega_de0 * f_de / e2
    om_m = (omega_cb0 + omega_nu_m0) * (zp1**3) / e2
    om_r = np.maximum(0.0, 1.0 - om_m - ode)

    dlnH = -1.5 * (om_m + (4.0 / 3.0) * om_r + ode * (1.0 + w_z))
    dlnOde = -3.0 * (1.0 + w_z) - 2.0 * dlnH
    aB = c_B * ode
    Ns_over_ode = np.maximum(
        3.0 * (1.0 + w_z) + (1.0 - 0.5 * aB) * c_B * (dlnH - 1.0) - c_B * dlnOde,
        1e-3,
    )
    mu_dense = 1.0 + (c_B**2) * ode / (2.0 * Ns_over_ode)

    # Build monotonic linear interpolator in scale factor a (sorted ascending)
    order = np.argsort(a_dense)
    a_sorted = a_dense[order]
    mu_sorted = mu_dense[order]

    def mu_interp(a_val: float | np.ndarray) -> float | np.ndarray:
        return np.interp(a_val, a_sorted, mu_sorted)

    # Note: In both PPF and Horndeski, initial conditions at z_ini = 1000 (a_ini = 1e-3)
    # are identical (since Omega_DE(1000) ~ 1e-9, mu(1e-3) = 1 to < 1e-10).
    # Therefore, starting from the same primordial CMB amplitude A_s at a_ini = 1e-3,
    # enhanced gravity mu(a) > 1 increases the total growth factor delta_m(a)/a_ini!
    # Let us compute the exact unnormalized growing mode delta_m(a) from the same initial condition
    # at a_ini = 1e-3 to capture both the cumulative sigma8(z) enhancement and f(z) enhancement!
    d_ppf, f_ppf = solve_subhorizon_growth(z_eval, params, rad, mu_func=None)
    d_horn, f_horn = solve_subhorizon_growth(z_eval, params, rad, mu_func=mu_interp)

    # Also compute the cumulative amplitude boost delta_horn(a=1) / delta_ppf(a=1) from fixed CMB A_s
    z_anchor = np.array([500.0, 0.0])
    d_ppf_anc, _ = solve_subhorizon_growth(z_anchor, params, rad, mu_func=None)
    d_horn_anc, _ = solve_subhorizon_growth(z_anchor, params, rad, mu_func=mu_interp)
    # Since d_ppf_anc[0] = delta(a=1e-3)/delta(a=1), the ratio delta_horn(1)/delta_ppf(1) for fixed delta(1e-3) is:
    sigma8_boost_0 = float(d_ppf_anc[0] / d_horn_anc[0])

    fs8_ppf = f_ppf * sigma8_0_fid * d_ppf
    fs8_horn = f_horn * (sigma8_0_fid * sigma8_boost_0) * d_horn
    mu_eval = mu_interp(1.0 / (1.0 + z_eval))

    return {
        "z": z_eval,
        "mu_z": mu_eval,
        "f_ppf": f_ppf,
        "f_horn": f_horn,
        "D_ppf": d_ppf,
        "D_horn": d_horn * sigma8_boost_0,
        "sigma8_boost_0": np.array(sigma8_boost_0),
        "fs8_ppf": fs8_ppf,
        "fs8_horn": fs8_horn,
        "frac_shift_fs8": (fs8_horn - fs8_ppf) / fs8_ppf,
    }
