"""Automated Milestone 7 baseline validation test suite (Tests T7.1 through T7.5).

Verifies that our independent JAX/SciPy FLRW background, sound horizon, BAO distance,
linear growth, and DESI DR2 + Planck 2018 CMB likelihood solvers reproduce CAMB
and published DESI DR2 / Planck 2018 benchmarks within strict tolerances.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import camb
import jax.numpy as jnp
import numpy as np

PROJECT_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "03-projects"
    / "pr01-horndeski-crossing-identifiability"
)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from baseline.cosmology_baseline import (  # noqa: E402
    C_LIGHT_KMS,
    CosmologyParams,
    DESIDR2BAOLikelihood,
    Planck2018CompressedCMBLikelihood,
    calibrate_radiation_from_camb,
    comoving_distance_dm_jax,
    e_of_z_jax,
    fit_desi_dr2_plus_cmb,
    hubble_distance_dh_jax,
    solve_subhorizon_growth,
    sound_horizon_rd_jax,
)
from derivations.flrw_and_horndeski_symbolic import (  # noqa: E402
    verify_background_density_evolutions,
    verify_eds_linear_growth_limit,
    verify_horndeski_stability_and_crossing_theorem,
)


class TestMilestone7Baseline(unittest.TestCase):
    """Automated verification gates T7.1 - T7.5 for Milestone 7."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.params_lcdm = CosmologyParams()
        cls.rad_lcdm = calibrate_radiation_from_camb(cls.params_lcdm)
        cls.params_cpl = CosmologyParams(w0=-0.75, wa=-0.85)
        cls.rad_cpl = calibrate_radiation_from_camb(cls.params_cpl)

    def test_symbolic_derivations_pass(self) -> None:
        """Verify all SymPy symbolic derivations for FLRW, Pade, Chebyshev, and Horndeski."""
        bg = verify_background_density_evolutions()
        horn = verify_horndeski_stability_and_crossing_theorem()
        verify_eds_linear_growth_limit()
        self.assertIn("ln_f_cpl_z", bg)
        self.assertIn("mu_minus_1", horn)

    def test_t7_1_hubble_expansion_vs_camb(self) -> None:
        """T7.1: E(z) = H(z)/H0 across z in [0, 1100] matches CAMB within < 1e-5."""
        zs = np.concatenate([[0.0], np.geomspace(1e-3, 1100.0, 80)])
        for p, rad in [(self.params_lcdm, self.rad_lcdm), (self.params_cpl, self.rad_cpl)]:
            pars = camb.CAMBparams()
            pars.set_cosmology(
                H0=p.H0, ombh2=p.ombh2, omch2=p.omch2, mnu=p.mnu, omk=0.0, tau=0.0544
            )
            pars.set_dark_energy(w=p.w0, wa=p.wa, dark_energy_model="ppf")
            res = camb.get_results(pars)

            e_jax = np.asarray(
                e_of_z_jax(jnp.asarray(zs), p.H0, p.ombh2, p.omch2, p.w0, p.wa, rad)
            )
            e_camb = np.array([res.hubble_parameter(float(z)) / p.H0 for z in zs])
            max_rel_err = float(np.max(np.abs(e_jax - e_camb) / e_camb))
            self.assertLess(max_rel_err, 1.0e-5, f"T7.1 failed with max_rel_err={max_rel_err}")

    def test_t7_2_bao_distances_vs_camb(self) -> None:
        """T7.2: D_M(z), D_H(z), D_V(z) at DESI DR2 redshifts match CAMB within < 1e-5."""
        zs_desi = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
        for p, rad in [(self.params_lcdm, self.rad_lcdm), (self.params_cpl, self.rad_cpl)]:
            pars = camb.CAMBparams()
            pars.set_cosmology(
                H0=p.H0, ombh2=p.ombh2, omch2=p.omch2, mnu=p.mnu, omk=0.0, tau=0.0544
            )
            pars.set_dark_energy(w=p.w0, wa=p.wa, dark_energy_model="ppf")
            res = camb.get_results(pars)

            dm_jax = np.asarray(
                comoving_distance_dm_jax(
                    jnp.asarray(zs_desi), p.H0, p.ombh2, p.omch2, p.w0, p.wa, rad
                )
            )
            dh_jax = np.asarray(
                hubble_distance_dh_jax(
                    jnp.asarray(zs_desi), p.H0, p.ombh2, p.omch2, p.w0, p.wa, rad
                )
            )
            dv_jax = (zs_desi * (dm_jax**2) * dh_jax) ** (1.0 / 3.0)

            dm_camb = np.array([(1.0 + z) * res.angular_diameter_distance(float(z)) for z in zs_desi])
            dh_camb = np.array([C_LIGHT_KMS / res.hubble_parameter(float(z)) for z in zs_desi])
            dv_camb = (zs_desi * (dm_camb**2) * dh_camb) ** (1.0 / 3.0)

            self.assertLess(float(np.max(np.abs(dm_jax - dm_camb) / dm_camb)), 1.0e-5)
            self.assertLess(float(np.max(np.abs(dh_jax - dh_camb) / dh_camb)), 1.0e-5)
            self.assertLess(float(np.max(np.abs(dv_jax - dv_camb) / dv_camb)), 1.0e-5)

    def test_t7_3_sound_horizon_rd_vs_planck2018_and_camb(self) -> None:
        """T7.3: Sound horizon r_d matches Planck 2018 (147.09 Mpc) and CAMB within < 1e-3."""
        p = self.params_lcdm
        rd_jax = float(sound_horizon_rd_jax(p.H0, p.ombh2, p.omch2, p.w0, p.wa, self.rad_lcdm))
        rel_err_planck = abs(rd_jax - 147.09) / 147.09
        rel_err_camb = abs(rd_jax - self.rad_lcdm.r_drag_camb_fid) / self.rad_lcdm.r_drag_camb_fid
        self.assertLess(rel_err_planck, 1.0e-3)
        self.assertLess(rel_err_camb, 1.0e-5)

    def test_t7_4_linear_growth_factor_vs_camb(self) -> None:
        """T7.4: Sub-horizon linear growth factor D_+(z) matches CAMB within < 5e-4."""
        zs = np.array([0.0, 0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
        pars = camb.CAMBparams()
        p = self.params_lcdm
        pars.set_cosmology(
            H0=p.H0, ombh2=p.ombh2, omch2=p.omch2, mnu=p.mnu, omk=0.0, tau=0.0544
        )
        pars.InitPower.set_params(As=2.100549e-9, ns=0.9649)
        pars.set_matter_power(redshifts=list(zs), kmax=2.0)
        res = camb.get_results(pars)

        s8 = np.array(res.get_sigma8())[::-1]
        d_camb = s8 / s8[0]
        d_ode, _ = solve_subhorizon_growth(zs, p, self.rad_lcdm)
        max_rel_err = float(np.max(np.abs(d_ode - d_camb) / d_camb))
        self.assertLess(max_rel_err, 5.0e-4, f"T7.4 failed with max_rel_err={max_rel_err}")

    def test_t7_5_desi_dr2_plus_cmb_fit_reproduction(self) -> None:
        """T7.5: DESI DR2 BAO + Planck CMB fit reproduces published w0 > -1, wa < 0 preference."""
        bao = DESIDR2BAOLikelihood()
        cmb = Planck2018CompressedCMBLikelihood(self.rad_lcdm)
        fit_lcdm = fit_desi_dr2_plus_cmb(self.rad_lcdm, bao, cmb, vary_w0wa=False)
        fit_cpl = fit_desi_dr2_plus_cmb(self.rad_lcdm, bao, cmb, vary_w0wa=True)

        delta_chi2 = fit_cpl["chi2"] - fit_lcdm["chi2"]
        # Published DESI DR2 + CMB (arXiv:2503.14738 Table V):
        # LambdaCDM: H0 = 68.51 +/- 0.29, Omega_m = 0.2975 +/- 0.0040
        # w0waCDM: w0 = -0.42 +/- 0.21, wa = -1.75 +/- 0.58, Delta chi2 ~ -7.6
        self.assertAlmostEqual(fit_lcdm["H0"], 68.51, delta=0.25)
        self.assertAlmostEqual(fit_lcdm["Omega_m"], 0.2975, delta=0.005)
        self.assertGreater(fit_cpl["w0"], -0.80)
        self.assertLess(fit_cpl["wa"], -0.80)
        self.assertLess(delta_chi2, -6.5)


if __name__ == "__main__":
    unittest.main()
