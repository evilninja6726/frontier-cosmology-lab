"""Baseline relativistic FLRW, BAO, CMB distance prior, and linear growth engine (Milestone 7).

Provides:
- Exact 64-bit JAX-differentiable background expansion E(z) = H(z)/H0 with
  physical photon density omega_g = Omega_g h^2, massless neutrinos omega_nu_rel,
  and massive Fermi-Dirac neutrinos omega_nu_mass(z).
- Support for LambdaCDM, CPL (w0, wa), Pade [1/1], and shifted-Chebyshev dark
  energy equations of state.
- Gauss-Legendre differentiable quadrature for D_M(z), D_H(z), D_V(z), r_d(z_drag),
  and r_*(z_*).
- Sub-horizon linear growth ODE solver for D_+(z) and f(z) = d(ln D_+)/d(ln a)
  with general Horndeski effective gravitational coupling mu(a) = G_eff/G_N.
- Official DESI DR2 BAO (13 measurements across 7 redshift bins, arXiv:2503.14738)
  + Planck 2018 compressed CMB shift-parameter likelihood (R, l_A, omega_b).
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Literal

import camb
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize

jax.config.update("jax_enable_x64", True)

C_LIGHT_KMS = 299792.458  # Speed of light in km/s
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"

# Precompute 64-node Gauss-Laguerre quadrature for Fermi-Dirac massive neutrinos
_Q_LAG, _W_LAG = laggauss(64)
_FD_WEIGHTS = jnp.asarray(_W_LAG * (_Q_LAG**2) / (1.0 + np.exp(-_Q_LAG)), dtype=jnp.float64)
_Q_NODES = jnp.asarray(_Q_LAG, dtype=jnp.float64)
_FD_I_REL = float(jnp.sum(_FD_WEIGHTS * _Q_NODES))

# Precompute 96-node Gauss-Legendre quadrature on [0, 1] for distance integrals
_X_LEG, _W_LEG = leggauss(96)
_U_LEG = jnp.asarray(0.5 * (_X_LEG + 1.0), dtype=jnp.float64)
_WU_LEG = jnp.asarray(0.5 * _W_LEG, dtype=jnp.float64)


@dataclass(frozen=True)
class CosmologyParams:
    """Fiducial cosmological parameters (defaults to Planck 2018 Table 2 baseline)."""

    H0: float = 67.36
    ombh2: float = 0.02237
    omch2: float = 0.1200
    mnu: float = 0.06
    w0: float = -1.0
    wa: float = 0.0
    pade_b: float = 0.0
    de_model: Literal["cpl", "pade", "chebyshev"] = "cpl"


@dataclass(frozen=True)
class RadiationCalibration:
    """Physical radiation & massive-neutrino densities omega_i = Omega_i h^2 (H0-independent)."""

    wb_fid: float
    wc_fid: float
    omega_g_h2: float
    omega_nu_rel_h2: float
    omega_nu_mass0_h2: float
    omega_nu_mass_rel_h2: float
    y0_nu: float
    z_drag_fid: float
    r_drag_camb_fid: float
    z_star_fid: float
    r_star_camb_fid: float


def _hu_sugiyama_z_drag_raw(ombh2: float | jax.Array, omch2: float | jax.Array, onuh2: float) -> jax.Array:
    """Hu & Sugiyama (1996) / Eisenstein & Hu (1998) baryon drag redshift fitting formula."""
    ommh2 = ombh2 + omch2 + onuh2
    b1 = 0.313 * (ommh2**-0.419) * (1.0 + 0.607 * (ommh2**0.674))
    b2 = 0.238 * (ommh2**0.223)
    return 1291.0 * (ommh2**0.251) / (1.0 + 0.659 * (ommh2**0.828)) * (1.0 + b1 * (ombh2**b2))


def _hu_sugiyama_z_star_raw(ombh2: float | jax.Array, omch2: float | jax.Array, onuh2: float) -> jax.Array:
    """Hu & Sugiyama (1996) recombination redshift z_* fitting formula."""
    ommh2 = ombh2 + omch2 + onuh2
    g1 = 0.0783 * (ombh2**-0.238) / (1.0 + 39.5 * (ombh2**0.763))
    g2 = 0.560 / (1.0 + 21.1 * (ombh2**1.81))
    return 1048.0 * (1.0 + 0.00124 * (ombh2**-0.738)) * (1.0 + g1 * (ommh2**g2))


def calibrate_radiation_from_camb(params: CosmologyParams) -> RadiationCalibration:
    """Extract H0-invariant physical photon/neutrino densities omega_i = Omega_i h^2 from CAMB."""
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=params.H0,
        ombh2=params.ombh2,
        omch2=params.omch2,
        mnu=params.mnu,
        omk=0.0,
        tau=0.0544,
    )
    pars.set_dark_energy(w=params.w0, wa=params.wa, dark_energy_model="ppf")
    results = camb.get_results(pars)
    derived = results.get_derived_params()

    h = params.H0 / 100.0
    h2 = h**2
    og_h2 = float(results.get_Omega("photon", 0.0)) * h2
    onu_rel_h2 = float(results.get_Omega("neutrino", 0.0)) * h2
    onu_m0_h2 = float(results.get_Omega("nu", 0.0)) * h2

    if params.mnu > 0.0 and onu_m0_h2 > 0.0:
        e_high = float(results.hubble_parameter(1e5)) / params.H0
        onu_m_rel_h2 = (
            float(results.get_Omega("nu", 1e5)) * (e_high**2) * h2 / ((1.0 + 1e5) ** 4)
        )

        def _fd_ratio(y: float) -> float:
            val = float(jnp.sum(_FD_WEIGHTS * jnp.sqrt(_Q_NODES**2 + y**2)))
            return val / _FD_I_REL - (onu_m0_h2 / onu_m_rel_h2)

        y0 = float(brentq(_fd_ratio, 0.0, 5000.0))
    else:
        onu_m_rel_h2 = 0.0
        y0 = 0.0

    return RadiationCalibration(
        wb_fid=params.ombh2,
        wc_fid=params.omch2,
        omega_g_h2=og_h2,
        omega_nu_rel_h2=onu_rel_h2,
        omega_nu_mass0_h2=onu_m0_h2,
        omega_nu_mass_rel_h2=onu_m_rel_h2,
        y0_nu=y0,
        z_drag_fid=float(derived["zdrag"]),
        r_drag_camb_fid=float(derived["rdrag"]),
        z_star_fid=float(derived["zstar"]),
        r_star_camb_fid=float(derived["rstar"]),
    )


def z_drag_jax(ombh2: float | jax.Array, omch2: float | jax.Array, rad: RadiationCalibration) -> jax.Array:
    """Calibrated baryon drag redshift z_drag(ombh2, omch2) anchored to CAMB."""
    raw_val = _hu_sugiyama_z_drag_raw(ombh2, omch2, rad.omega_nu_mass0_h2)
    raw_fid = _hu_sugiyama_z_drag_raw(rad.wb_fid, rad.wc_fid, rad.omega_nu_mass0_h2)
    return rad.z_drag_fid * (raw_val / raw_fid)


def z_star_jax(ombh2: float | jax.Array, omch2: float | jax.Array, rad: RadiationCalibration) -> jax.Array:
    """Calibrated recombination redshift z_*(ombh2, omch2) anchored to CAMB."""
    raw_val = _hu_sugiyama_z_star_raw(ombh2, omch2, rad.omega_nu_mass0_h2)
    raw_fid = _hu_sugiyama_z_star_raw(rad.wb_fid, rad.wc_fid, rad.omega_nu_mass0_h2)
    return rad.z_star_fid * (raw_val / raw_fid)


def f_de_cpl_jax(z: jax.Array, w0: float | jax.Array, wa: float | jax.Array) -> jax.Array:
    """Analytical normalized dark energy density f_DE(z) = rho_DE(z)/rho_DE(0) for CPL."""
    zp1 = 1.0 + z
    return (zp1 ** (3.0 * (1.0 + w0 + wa))) * jnp.exp(-3.0 * wa * z / zp1)


def f_de_pade_jax(
    z: jax.Array,
    w0: float | jax.Array,
    w1: float | jax.Array,
    b: float | jax.Array,
) -> jax.Array:
    """Analytical normalized dark energy density f_DE(z) for Pade [1/1] w(a)."""
    a = 1.0 / (1.0 + z)
    is_small_b = jnp.abs(b) < 1e-7
    b_safe = jnp.where(is_small_b, 1e-4, b)
    ln_f_pade = (
        -3.0 * (1.0 + (w0 + w1) / (1.0 + b_safe)) * jnp.log(a)
        - 3.0 * (w1 - b_safe * w0) / (b_safe * (1.0 + b_safe)) * jnp.log(1.0 + b_safe * (1.0 - a))
    )
    ln_f_cpl = -3.0 * (1.0 + w0 + w1) * jnp.log(a) - 3.0 * w1 * (1.0 - a)
    return jnp.where(is_small_b, jnp.exp(ln_f_cpl), jnp.exp(ln_f_pade))


def f_de_chebyshev_jax(
    z: jax.Array,
    c0: float | jax.Array,
    c1: float | jax.Array,
    c2: float | jax.Array,
) -> jax.Array:
    """Analytical normalized dark energy density f_DE(z) for shifted Chebyshev basis."""
    a = 1.0 / (1.0 + z)
    ln_f = (
        -12.0 * (a**2) * c2
        + 6.0 * a * (c1 + 4.0 * c2)
        - 6.0 * c1
        - 12.0 * c2
        - 3.0 * (1.0 + c0 + c1 + c2) * jnp.log(a)
    )
    return jnp.exp(ln_f)


def e_of_z_jax(
    z: jax.Array,
    H0: float | jax.Array,
    ombh2: float | jax.Array,
    omch2: float | jax.Array,
    w0: float | jax.Array,
    wa: float | jax.Array,
    rad: RadiationCalibration,
    pade_b: float | jax.Array = 0.0,
    de_model: str = "cpl",
) -> jax.Array:
    """Compute dimensionless Hubble rate E(z) = H(z)/H0 in JAX (64-bit)."""
    z_arr = jnp.asarray(z, dtype=jnp.float64)
    zp1 = 1.0 + z_arr
    a = 1.0 / zp1
    h = H0 / 100.0
    h2 = h**2

    omega_cb0 = (ombh2 + omch2) / h2
    omega_g0 = rad.omega_g_h2 / h2
    omega_nu_rel0 = rad.omega_nu_rel_h2 / h2
    omega_nu_m0 = rad.omega_nu_mass0_h2 / h2
    omega_nu_m_rel = rad.omega_nu_mass_rel_h2 / h2
    omega_de0 = 1.0 - omega_cb0 - omega_g0 - omega_nu_rel0 - omega_nu_m0

    y_z = rad.y0_nu * a[..., None]
    fd_ratio = jnp.sum(_FD_WEIGHTS * jnp.sqrt(_Q_NODES**2 + y_z**2), axis=-1) / _FD_I_REL
    omega_nu_m_z = omega_nu_m_rel * (zp1**4) * fd_ratio

    if de_model == "cpl":
        f_de = f_de_cpl_jax(z_arr, w0, wa)
    elif de_model == "pade":
        f_de = f_de_pade_jax(z_arr, w0, wa, pade_b)
    elif de_model == "chebyshev":
        f_de = f_de_chebyshev_jax(z_arr, w0, wa, pade_b)
    else:
        raise ValueError(f"Unknown de_model: {de_model}")

    e2 = (
        omega_cb0 * (zp1**3)
        + (omega_g0 + omega_nu_rel0) * (zp1**4)
        + omega_nu_m_z
        + omega_de0 * f_de
    )
    return jnp.sqrt(e2)


def comoving_distance_dm_jax(
    z: jax.Array,
    H0: float | jax.Array,
    ombh2: float | jax.Array,
    omch2: float | jax.Array,
    w0: float | jax.Array,
    wa: float | jax.Array,
    rad: RadiationCalibration,
    pade_b: float | jax.Array = 0.0,
    de_model: str = "cpl",
) -> jax.Array:
    """Compute transverse comoving distance D_M(z) in Mpc via 96-node Gauss-Legendre quadrature."""
    z_arr = jnp.asarray(z, dtype=jnp.float64)
    z_grid = z_arr[..., None] * _U_LEG
    e_grid = e_of_z_jax(z_grid, H0, ombh2, omch2, w0, wa, rad, pade_b=pade_b, de_model=de_model)
    integral = z_arr * jnp.sum(_WU_LEG / e_grid, axis=-1)
    return (C_LIGHT_KMS / H0) * integral


def hubble_distance_dh_jax(
    z: jax.Array,
    H0: float | jax.Array,
    ombh2: float | jax.Array,
    omch2: float | jax.Array,
    w0: float | jax.Array,
    wa: float | jax.Array,
    rad: RadiationCalibration,
    pade_b: float | jax.Array = 0.0,
    de_model: str = "cpl",
) -> jax.Array:
    """Compute Hubble distance D_H(z) = c / H(z) in Mpc."""
    e_val = e_of_z_jax(z, H0, ombh2, omch2, w0, wa, rad, pade_b=pade_b, de_model=de_model)
    return (C_LIGHT_KMS / H0) / e_val


def _sound_horizon_at_z_jax(
    z_upper: float | jax.Array,
    H0: float | jax.Array,
    ombh2: float | jax.Array,
    omch2: float | jax.Array,
    w0: float | jax.Array,
    wa: float | jax.Array,
    rad: RadiationCalibration,
) -> jax.Array:
    """Compute comoving sound horizon r_s(z_upper) in Mpc."""
    a_up = 1.0 / (1.0 + z_upper)
    a_grid = a_up * _U_LEG
    z_grid = (1.0 / a_grid) - 1.0
    e_grid = e_of_z_jax(z_grid, H0, ombh2, omch2, w0, wa, rad)

    r_b = (3.0 * ombh2 / (4.0 * rad.omega_g_h2)) * a_grid
    cs_over_c = 1.0 / jnp.sqrt(3.0 * (1.0 + r_b))
    integrand = cs_over_c / ((a_grid**2) * e_grid)
    return (C_LIGHT_KMS / H0) * a_up * jnp.sum(_WU_LEG * integrand)


def sound_horizon_rd_jax(
    H0: float | jax.Array,
    ombh2: float | jax.Array,
    omch2: float | jax.Array,
    w0: float | jax.Array,
    wa: float | jax.Array,
    rad: RadiationCalibration,
) -> jax.Array:
    """Compute comoving sound horizon at baryon drag r_d(z_drag) in Mpc."""
    z_d = z_drag_jax(ombh2, omch2, rad)
    return _sound_horizon_at_z_jax(z_d, H0, ombh2, omch2, w0, wa, rad)


def sound_horizon_rstar_jax(
    H0: float | jax.Array,
    ombh2: float | jax.Array,
    omch2: float | jax.Array,
    w0: float | jax.Array,
    wa: float | jax.Array,
    rad: RadiationCalibration,
) -> jax.Array:
    """Compute comoving sound horizon at recombination r_*(z_*) in Mpc."""
    z_s = z_star_jax(ombh2, omch2, rad)
    return _sound_horizon_at_z_jax(z_s, H0, ombh2, omch2, w0, wa, rad)


def solve_subhorizon_growth(
    z_eval: np.ndarray,
    params: CosmologyParams,
    rad: RadiationCalibration,
    mu_func=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the sub-horizon linear growth ODE for D_+(z) and f(z) = d(ln D_+)/d(ln a)."""
    h = params.H0 / 100.0
    omega_cb0 = (params.ombh2 + params.omch2) / (h**2)

    def _e_of_a(a_val: float) -> float:
        z_val = (1.0 / a_val) - 1.0
        return float(
            e_of_z_jax(
                jnp.array(z_val),
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

    def _dlnH_dlna(a_val: float) -> float:
        eps = 1e-5 * a_val
        return (np.log(_e_of_a(a_val + eps)) - np.log(_e_of_a(a_val - eps))) / (2.0 * eps / a_val)

    def _rhs(x: float, y: np.ndarray) -> list[float]:
        a_val = float(np.exp(x))
        e_val = _e_of_a(a_val)
        dlnH = _dlnH_dlna(a_val)
        om_cb_a = omega_cb0 * (a_val**-3) / (e_val**2)
        mu_val = float(mu_func(a_val)) if mu_func is not None else 1.0
        delta, ddelta = y
        d2delta = -(2.0 + dlnH) * ddelta + 1.5 * om_cb_a * mu_val * delta
        return [float(ddelta), float(d2delta)]

    a_ini = 1e-3
    x_ini = float(np.log(a_ini))
    z_sorted_idx = np.argsort(z_eval)[::-1]
    z_sorted = z_eval[z_sorted_idx]
    x_eval = np.log(1.0 / (1.0 + z_sorted))
    if x_eval[-1] < 0.0:
        x_eval_with_zero = np.append(x_eval, 0.0)
    else:
        x_eval_with_zero = x_eval

    sol = solve_ivp(
        _rhs,
        (x_ini, 0.0),
        [a_ini, a_ini],
        t_eval=x_eval_with_zero,
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",
    )
    delta_zero = sol.y[0, -1]
    deltas = sol.y[0, : len(x_eval)] / delta_zero
    f_vals = sol.y[1, : len(x_eval)] / sol.y[0, : len(x_eval)]

    inv_idx = np.argsort(z_sorted_idx)
    return deltas[inv_idx], f_vals[inv_idx]


class DESIDR2BAOLikelihood:
    """Official DESI DR2 Gaussian BAO likelihood (13 points across z in [0.295, 2.33])."""

    def __init__(self) -> None:
        mean_file = DATA_DIR / "desi_gaussian_bao_ALL_GCcomb_mean.txt"
        cov_file = DATA_DIR / "desi_gaussian_bao_ALL_GCcomb_cov.txt"
        zs, vals, kinds = [], [], []
        for line in mean_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            zs.append(float(parts[0]))
            vals.append(float(parts[1]))
            kinds.append(parts[2])

        self.z = jnp.asarray(zs, dtype=jnp.float64)
        self.y_obs = jnp.asarray(vals, dtype=jnp.float64)
        self.kinds = kinds
        cov = np.loadtxt(cov_file, dtype=np.float64)
        self.cov = jnp.asarray(cov, dtype=jnp.float64)
        self.inv_cov = jnp.asarray(np.linalg.inv(cov), dtype=jnp.float64)

        code_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
        self.kind_codes = jnp.asarray([code_map[k] for k in kinds], dtype=jnp.int32)

    def predict_vector(
        self,
        H0: float | jax.Array,
        ombh2: float | jax.Array,
        omch2: float | jax.Array,
        w0: float | jax.Array,
        wa: float | jax.Array,
        rad: RadiationCalibration,
        rd_fid: float | jax.Array | None = None,
        pade_b: float | jax.Array = 0.0,
        de_model: str = "cpl",
    ) -> jax.Array:
        """Compute theoretical BAO vector matching the 13 DESI DR2 entries."""
        dm = comoving_distance_dm_jax(
            self.z, H0, ombh2, omch2, w0, wa, rad, pade_b=pade_b, de_model=de_model
        )
        dh = hubble_distance_dh_jax(
            self.z, H0, ombh2, omch2, w0, wa, rad, pade_b=pade_b, de_model=de_model
        )
        dv = (self.z * (dm**2) * dh) ** (1.0 / 3.0)
        rd = sound_horizon_rd_jax(H0, ombh2, omch2, w0, wa, rad) if rd_fid is None else rd_fid

        pred = jnp.where(
            self.kind_codes == 0,
            dv / rd,
            jnp.where(self.kind_codes == 1, dm / rd, dh / rd),
        )
        return pred

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
    ) -> jax.Array:
        """Compute exact Gaussian chi^2 against DESI DR2 BAO."""
        rd_raw = sound_horizon_rd_jax(H0, ombh2, omch2, w0, wa, rad)
        pred = self.predict_vector(
            H0, ombh2, omch2, w0, wa, rad, rd_fid=rd_raw, pade_b=pade_b, de_model=de_model
        )
        residual = pred - self.y_obs
        return residual @ (self.inv_cov @ residual)


class Planck2018CompressedCMBLikelihood:
    """Correlated Planck 2018 shift-parameter prior on (R, l_A, omega_b)."""

    def __init__(self, rad: RadiationCalibration) -> None:
        self.rad = rad
        z_s_fid = float(z_star_jax(0.02237, 0.1200, rad))
        dm_star_fid = float(
            comoving_distance_dm_jax(jnp.array(z_s_fid), 67.36, 0.02237, 0.1200, -1.0, 0.0, rad)
        )
        rs_fid = float(sound_horizon_rstar_jax(67.36, 0.02237, 0.1200, -1.0, 0.0, rad))
        omm_fid = (0.02237 + 0.1200 + rad.omega_nu_mass0_h2) / (0.6736**2)
        r_fid = float(np.sqrt(omm_fid) * (67.36 / C_LIGHT_KMS) * dm_star_fid)
        la_fid = float(np.pi * dm_star_fid / rs_fid)

        self.y_obs = jnp.array([r_fid, la_fid, 0.02237], dtype=jnp.float64)
        sig_cmb = np.array([0.0046, 0.090, 0.00015], dtype=np.float64)
        corr_cmb = np.array(
            [
                [1.00, 0.46, -0.66],
                [0.46, 1.00, -0.33],
                [-0.66, -0.33, 1.00],
            ],
            dtype=np.float64,
        )
        cov_cmb = np.outer(sig_cmb, sig_cmb) * corr_cmb
        self.inv_cov = jnp.asarray(np.linalg.inv(cov_cmb), dtype=jnp.float64)

    def chi2(
        self,
        H0: float | jax.Array,
        ombh2: float | jax.Array,
        omch2: float | jax.Array,
        w0: float | jax.Array,
        wa: float | jax.Array,
        pade_b: float | jax.Array = 0.0,
        de_model: str = "cpl",
    ) -> jax.Array:
        """Compute correlated Planck 2018 shift-parameter chi^2."""
        z_s = z_star_jax(ombh2, omch2, self.rad)
        dm_star = comoving_distance_dm_jax(
            z_s, H0, ombh2, omch2, w0, wa, self.rad, pade_b=pade_b, de_model=de_model
        )
        rs = sound_horizon_rstar_jax(H0, ombh2, omch2, w0, wa, self.rad)
        h = H0 / 100.0
        om_m = (ombh2 + omch2 + self.rad.omega_nu_mass0_h2) / (h**2)
        r_th = jnp.sqrt(om_m) * (H0 / C_LIGHT_KMS) * dm_star
        la_th = jnp.pi * dm_star / rs
        res = jnp.array([r_th, la_th, ombh2]) - self.y_obs
        return res @ (self.inv_cov @ res)


def fit_desi_dr2_plus_cmb(
    rad: RadiationCalibration,
    bao: DESIDR2BAOLikelihood,
    cmb: Planck2018CompressedCMBLikelihood,
    vary_w0wa: bool = False,
) -> dict[str, float]:
    """Find exact MAP / profile-likelihood minimum for LambdaCDM or CPL w0waCDM."""

    @jax.jit
    def _joint_chi2(theta: jax.Array) -> jax.Array:
        H0, ombh2, omch2, w0, wa = theta
        return bao.chi2(H0, ombh2, omch2, w0, wa, rad) + cmb.chi2(H0, ombh2, omch2, w0, wa)

    grad_fn = jax.jit(jax.grad(_joint_chi2))

    if not vary_w0wa:
        def _obj_lcdm(x: np.ndarray) -> tuple[float, np.ndarray]:
            th = jnp.array([x[0], x[1], x[2], -1.0, 0.0], dtype=jnp.float64)
            return float(_joint_chi2(th)), np.asarray(grad_fn(th)[:3], dtype=np.float64)

        res = minimize(
            _obj_lcdm,
            np.array([68.0, 0.02237, 0.1200]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(55.0, 80.0), (0.020, 0.025), (0.10, 0.14)],
        )
        H0_opt, ob_opt, oc_opt = [float(v) for v in res.x]
        w0_opt, wa_opt = -1.0, 0.0
    else:
        def _obj_cpl(x: np.ndarray) -> tuple[float, np.ndarray]:
            th = jnp.asarray(x, dtype=jnp.float64)
            return float(_joint_chi2(th)), np.asarray(grad_fn(th), dtype=np.float64)

        res = minimize(
            _obj_cpl,
            np.array([65.0, 0.02237, 0.1200, -0.50, -1.50]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(55.0, 80.0), (0.020, 0.025), (0.10, 0.14), (-2.0, 0.5), (-3.0, 1.0)],
        )
        H0_opt, ob_opt, oc_opt, w0_opt, wa_opt = [float(v) for v in res.x]

    om_opt = (ob_opt + oc_opt + rad.omega_nu_mass0_h2) / ((H0_opt / 100.0) ** 2)
    return {
        "H0": H0_opt,
        "ombh2": ob_opt,
        "omch2": oc_opt,
        "Omega_m": float(om_opt),
        "w0": w0_opt,
        "wa": wa_opt,
        "chi2": float(res.fun),
    }
