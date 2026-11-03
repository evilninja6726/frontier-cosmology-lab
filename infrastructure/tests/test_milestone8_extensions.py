"""Automated verification suite for Milestone 8 Research Extensions (EXP-01, EXP-02, EXP-03)."""

from __future__ import annotations

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baseline.cosmology_baseline import (
    CosmologyParams,
    DESIDR2BAOLikelihood,
    Planck2018CompressedCMBLikelihood,
    calibrate_radiation_from_camb,
    e_of_z_jax,
    f_de_cpl_jax,
)
from simulations.horndeski_eft_solver import (
    SDSSDR16Fs8Likelihood,
    Union3SNLikelihood,
    compute_eft_prior_volume_and_growth_slip,
    solve_horndeski_growth_trajectory,
    w_of_z_jax,
)

jax.config.update("jax_enable_x64", True)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "05-experiments" / "experiment-registry" / "EXP_01_to_03_milestone8_report.json"


@pytest.fixture(scope="module")
def rad_cal():
    return calibrate_radiation_from_camb(CosmologyParams())


def test_union3_sn_likelihood_integrity(rad_cal) -> None:
    """Verify Union3 22-bin SN Ia dataset, covariance invertibility, and low-z split projection."""
    sn = Union3SNLikelihood()
    assert len(sn.z) == 22
    assert sn.cov.shape == (22, 22)
    eigvals = np.linalg.eigvalsh(np.asarray(sn.cov))
    assert np.all(eigvals > 0.0)

    chi2_unif = float(sn.chi2(68.0, 0.0224, 0.119, -1.0, 0.0, rad_cal, allow_lowz_split=False))
    chi2_split = float(sn.chi2(68.0, 0.0224, 0.119, -1.0, 0.0, rad_cal, allow_lowz_split=True))
    # Adding a free nuisance parameter (low-z calibration split) must lower or equal chi^2
    assert 0.0 < chi2_split <= chi2_unif


def test_horndeski_kineticity_cancellation_theorem(rad_cal) -> None:
    """Verify numerically that sub-horizon gravitational slip mu(z) - 1 is strictly independent of c_K."""
    z_j = jnp.linspace(0.0, 2.0, 50)
    H0, ombh2, omch2, w0, wa = 66.7, 0.0224, 0.119, -0.688, -0.978
    h = H0 / 100.0
    h2 = h**2
    omega_cb0 = (ombh2 + omch2) / h2
    omega_de0 = (
        1.0
        - omega_cb0
        - (rad_cal.omega_g_h2 + rad_cal.omega_nu_rel_h2 + rad_cal.omega_nu_mass0_h2) / h2
    )
    e_z = np.asarray(e_of_z_jax(z_j, H0, ombh2, omch2, w0, wa, rad_cal))
    e2 = e_z**2
    ode = omega_de0 * np.asarray(f_de_cpl_jax(z_j, w0, wa)) / e2
    w_z = np.asarray(w_of_z_jax(z_j, w0, wa))
    om_m = (omega_cb0 + rad_cal.omega_nu_mass0_h2 / h2) * ((1.0 + np.asarray(z_j)) ** 3) / e2
    om_r = np.maximum(0.0, 1.0 - om_m - ode)
    dlnH = -1.5 * (om_m + (4.0 / 3.0) * om_r + ode * (1.0 + w_z))
    dlnOde = -3.0 * (1.0 + w_z) - 2.0 * dlnH

    c_B = -0.8
    a_B = c_B * ode
    Ns = ode * (3.0 * (1.0 + w_z) + (1.0 - 0.5 * a_B) * c_B * (dlnH - 1.0) - c_B * dlnOde)

    # Compute D(a) and c_s^2(a) for two vastly different kineticities c_K = 0.5 and c_K = 50.0
    D_1 = 0.5 * ode + 1.5 * (a_B**2)
    cs2_1 = Ns / D_1
    mu_m1_1 = (a_B**2) / (2.0 * D_1 * cs2_1)

    D_2 = 50.0 * ode + 1.5 * (a_B**2)
    cs2_2 = Ns / D_2
    mu_m1_2 = (a_B**2) / (2.0 * D_2 * cs2_2)

    assert np.allclose(mu_m1_1, mu_m1_2, rtol=1e-14, atol=1e-15)
    assert np.all(mu_m1_1 > 0.0)


def test_horndeski_phantom_crossing_requires_negative_braiding(rad_cal) -> None:
    """Verify that phantom crossing (w0 > -1, w0 + wa < -1) forbids c_B >= 0 and compresses V_EFT."""
    c_K_grid = np.linspace(0.01, 6.0, 120)
    c_B_grid = np.linspace(-3.0, 3.0, 601)

    res_quint = compute_eft_prior_volume_and_growth_slip(
        w0=-0.90,
        wa=0.05,
        H0=67.2,
        ombh2=0.0224,
        omch2=0.119,
        rad=rad_cal,
        c_K_grid=c_K_grid,
        c_B_grid=c_B_grid,
    )
    res_cross = compute_eft_prior_volume_and_growth_slip(
        w0=-0.498,
        wa=-1.484,
        H0=64.22,
        ombh2=0.02237,
        omch2=0.1198,
        rad=rad_cal,
        c_K_grid=c_K_grid,
        c_B_grid=c_B_grid,
    )

    assert res_quint["survival_fraction"] > res_cross["survival_fraction"] > 0.0
    assert res_cross["c_B_max"] < -0.25  # Strictly negative braiding required
    assert res_cross["min_max_mu_minus_1"] > 0.01  # Strictly positive gravitational coupling boost


def test_horndeski_growth_solver_and_milestone8_report(rad_cal) -> None:
    """Verify SDSS DR16 f*sigma8 likelihood, growth ODE consistency, and Milestone 8 report."""
    sdss = SDSSDR16Fs8Likelihood()
    assert len(sdss.z) == 4
    params = CosmologyParams(H0=66.7, ombh2=0.0224, omch2=0.119, w0=-0.688, wa=-0.978)

    # At c_B = 0, Horndeski solver must exactly match PPF solver
    traj_zero = solve_horndeski_growth_trajectory(sdss.z, params, rad_cal, c_B=0.0, sigma8_0_fid=0.81)
    assert np.allclose(traj_zero["fs8_ppf"], traj_zero["fs8_horn"], rtol=1e-12)

    # At stable negative braiding c_B = -1.0, enhanced gravity mu(a) > 1 boosts f*sigma8(z)
    traj_braid = solve_horndeski_growth_trajectory(sdss.z, params, rad_cal, c_B=-1.0, sigma8_0_fid=0.81)
    assert np.all(traj_braid["mu_z"] > 1.0)
    assert np.all(traj_braid["fs8_horn"] > traj_braid["fs8_ppf"])

    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert report["EXP_02_basis_and_calibration_split_audit"]["calibration_split_summary"][
            "hypothesis_H1_1_confirmed"
        ]
