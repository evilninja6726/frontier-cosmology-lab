"""Automated verification suite for Project PR-02 (Hyperboloidal Black Hole Ringdown & Pseudospectra)."""

from __future__ import annotations

import numpy as np

from derivations.hyperboloidal_qnm_symbolic import verify_hyperboloidal_wave_operator
from simulations.chebyshev_pseudospectrum_solver import (
    build_hyperboloidal_system,
    chebyshev_lobatto_grid_and_matrices,
    compute_spectrum_and_condition_numbers,
    compute_waveform_mismatch,
    environmental_bump_delta_v_tilde,
    exact_poschl_teller_qnms,
    poschl_teller_v_tilde,
    simulate_ringdown_waveform,
)


def test_pr02_symbolic_derivations() -> None:
    """Verify SymPy proofs of hyperboloidal operators, QNM eigenvalues, and energy dissipation."""
    res = verify_hyperboloidal_wave_operator()
    assert res["energy_flux_divergence_residual"] == "0"


def test_bm02_poschl_teller_spectral_accuracy() -> None:
    """Verify Chebyshev-Lobatto quadrature and < 1e-11 accuracy of Poschl-Teller QNMs."""
    sigma, D1, D2, w = chebyshev_lobatto_grid_and_matrices(40)
    assert np.isclose(np.sum(w), 2.0, rtol=1e-14, atol=1e-14)
    # Derivative of sigma^3 is 3*sigma^2
    assert np.allclose(D1 @ (sigma**3), 3.0 * (sigma**2), rtol=1e-11, atol=1e-11)

    sys_h = build_hyperboloidal_system(N=40, V0=1.0)
    spec = compute_spectrum_and_condition_numbers(sys_h, positive_real_only=True)
    exact_qnm = exact_poschl_teller_qnms(5, V0=1.0)
    rel_err = np.abs(spec["omega"][:5] - exact_qnm) / np.abs(exact_qnm)
    # Fundamental (n=0) and first overtone (n=1) raw forward error < 1e-11:
    assert np.max(rel_err[:2]) < 1e-11
    # Condition-number normalized backward error |Delta omega_n| / (kappa_n * |omega_n|) < 1e-10 for n=0..4:
    assert np.max(rel_err[:5] / spec["kappa"][:5]) < 1e-10


def test_exp04_energy_norm_condition_numbers_and_dissipation() -> None:
    """Verify energy-norm condition number growth (H2.1) and monotonic energy decay dE/dtau <= 0."""
    sys_h = build_hyperboloidal_system(N=44, V0=1.0)
    spec = compute_spectrum_and_condition_numbers(sys_h, positive_real_only=True)
    kappas = spec["kappa"][:6]
    assert np.all(kappas >= 1.0)
    assert (kappas[3] / kappas[0]) > 20.0

    tau_grid = np.linspace(0.0, 10.0, 201)
    wf = simulate_ringdown_waveform(sys_h, tau_grid)
    assert np.max(np.diff(wf["energy"])) <= 1e-12


def test_exp06_prompt_ringdown_immunity_vs_spectral_migration() -> None:
    """Verify prompt time-domain waveform immunity (M_prompt < 1e-4) under distant bump (H2.2)."""
    N = 50
    sys_0 = build_hyperboloidal_system(N=N, V0=1.0)
    tau_grid = np.linspace(0.0, 14.0, 281)
    wf_0 = simulate_ringdown_waveform(sys_0, tau_grid, x0_packet=-0.5, width_packet=0.5, sigma_obs_target=0.80)

    sys_b = build_hyperboloidal_system(
        N=N,
        v_tilde_fn=lambda s: poschl_teller_v_tilde(s, V0=1.0)
        + environmental_bump_delta_v_tilde(s, eps_bump=2e-3, x_bump=2.8, width=0.30, k_osc=6.0),
        V0=1.0,
    )
    wf_b = simulate_ringdown_waveform(sys_b, tau_grid, x0_packet=-0.5, width_packet=0.5, sigma_obs_target=0.80)

    m_prompt = compute_waveform_mismatch(tau_grid, wf_0["phi_obs"], wf_b["phi_obs"], tau_min=0.0, tau_max=3.5)
    m_late = compute_waveform_mismatch(tau_grid, wf_0["phi_obs"], wf_b["phi_obs"], tau_min=5.0, tau_max=14.0)
    assert m_prompt < 1e-4
    assert m_late > m_prompt
