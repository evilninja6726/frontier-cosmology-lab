"""Project PR-02 Benchmark (BM-02) and Research Extensions Runner (EXP-04, EXP-05, EXP-06).

Executes:
- BM-02: Exact Poschl-Teller hyperboloidal QNM spectral convergence against analytical
  formula omega_n = sqrt(V0 - 1/4) + i*(n + 1/2) and energy-dissipation verification.
- EXP-04: Energy-norm Bauer-Fike condition numbers kappa_n and 2D SVD epsilon-pseudospectrum
  sigma_eps(L) reproducing Jaramillo, Panosso Macedo, & Al Sheikh (2021, PRX 11, 031003).
- EXP-05: Wavelength sensitivity of QNM migration under high-wavenumber ("flea") vs.
  low-wavenumber ("adiabatic halo") potential perturbations.
- EXP-06: Prompt time-domain ringdown waveform mismatch M_prompt vs. post-echo mismatch
  M_echo across bump distances x_bump, testing Hypothesis H2.2.
"""

from __future__ import annotations

import json
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from simulations.chebyshev_pseudospectrum_solver import (
    build_hyperboloidal_system,
    compute_pseudospectrum_grid,
    compute_spectrum_and_condition_numbers,
    compute_waveform_mismatch,
    environmental_bump_delta_v_tilde,
    exact_poschl_teller_qnms,
    poschl_teller_v_tilde,
    simulate_ringdown_waveform,
    sinusoidal_v_tilde_perturbation,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
REPORT_PATH = REPO_ROOT / "05-experiments" / "experiment-registry" / "PR02_BM02_and_EXP04_to_06_report.json"
FIGURE_PATH = PROJECT_DIR / "results" / "pr02_ringdown_pseudospectra_and_timedomain.png"


def run_bm02_poschl_teller_validation(V0: float = 1.0) -> dict:
    """Run BM-02: Validate Chebyshev hyperboloidal QNM eigenvalues against exact analytical spectrum."""
    exact_qnm = exact_poschl_teller_qnms(6, V0=V0)
    resolutions = [20, 40, 60]
    convergence = {}

    for N in resolutions:
        sys_h = build_hyperboloidal_system(N=N, V0=V0)
        spec = compute_spectrum_and_condition_numbers(sys_h, positive_real_only=True)
        omega_num = spec["omega"][:6]
        rel_errs = np.abs(omega_num - exact_qnm) / np.abs(exact_qnm)
        convergence[f"N_{N}"] = {
            "max_rel_err_n0_to_n4": float(np.max(rel_errs[:5])),
            "rel_err_per_mode": [float(x) for x in rel_errs[:5]],
            "omega_real": [float(np.real(w)) for w in omega_num[:5]],
            "omega_imag": [float(np.imag(w)) for w in omega_num[:5]],
            "kappa_condition_numbers": [float(k) for k in spec["kappa"][:5]],
        }

    # Verify monotonic energy dissipation dE/dtau <= 0 in time-domain evolution at N=50
    sys_50 = build_hyperboloidal_system(N=50, V0=V0)
    tau_grid = np.linspace(0.0, 12.0, 241)
    wf = simulate_ringdown_waveform(sys_50, tau_grid)
    dE = np.diff(wf["energy"])
    max_positive_energy_drift = float(np.max(dE))

    return {
        "benchmark_id": "BM-02",
        "V0": V0,
        "exact_omega_real": float(np.real(exact_qnm[0])),
        "exact_omega_imag_n0_to_n4": [float(np.imag(w)) for w in exact_qnm[:5]],
        "convergence": convergence,
        "max_positive_energy_drift_dE": max_positive_energy_drift,
        "energy_monotonic_dissipation_verified": bool(max_positive_energy_drift <= 1e-12),
        "bm02_passed": bool(convergence["N_60"]["max_rel_err_n0_to_n4"] < 1e-11),
    }


def run_exp04_pseudospectrum_and_condition_numbers(N: int = 50, V0: float = 1.0) -> dict:
    """Run EXP-04: Compute Bauer-Fike condition numbers kappa_n and 2D SVD pseudospectrum."""
    sys_h = build_hyperboloidal_system(N=N, V0=V0)
    spec = compute_spectrum_and_condition_numbers(sys_h, positive_real_only=True)
    omega_pos = spec["omega"][:7]
    kappa_pos = spec["kappa"][:7]

    re_grid = np.linspace(-2.2, 2.2, 55)
    im_grid = np.linspace(0.1, 5.2, 52)
    log10_smin = compute_pseudospectrum_grid(sys_h, re_grid, im_grid)

    kappa_ratio_n3_n0 = float(kappa_pos[3] / kappa_pos[0])
    kappa_ratio_n5_n0 = float(kappa_pos[5] / kappa_pos[0])

    return {
        "N": N,
        "modes_n0_to_n6": {
            "n": list(range(7)),
            "omega_real": [float(np.real(w)) for w in omega_pos],
            "omega_imag": [float(np.imag(w)) for w in omega_pos],
            "kappa": [float(k) for k in kappa_pos],
        },
        "kappa_ratio_n3_over_n0": kappa_ratio_n3_n0,
        "kappa_ratio_n5_over_n0": kappa_ratio_n5_over_n0,
        "hypothesis_H2_1_condition_growth_confirmed": bool(kappa_ratio_n3_n0 > 20.0),
        "pseudospectrum_grid": {
            "re_grid": re_grid.tolist(),
            "im_grid": im_grid.tolist(),
            "log10_smin": log10_smin.tolist(),
        },
    }


def run_exp05_wavelength_and_flea_perturbation_audit(N: int = 50, V0: float = 1.0) -> dict:
    """Run EXP-05: Compare QNM shifts across perturbation wavenumbers k_mode and overtone index n."""
    sys_0 = build_hyperboloidal_system(N=N, V0=V0)
    spec_0 = compute_spectrum_and_condition_numbers(sys_0, positive_real_only=True)
    omega_0 = spec_0["omega"][:5]

    eps = 5e-3
    k_modes = [1, 2, 4, 8, 12]
    wavenumber_scan = {}

    for k_m in k_modes:
        sys_k = build_hyperboloidal_system(
            N=N,
            v_tilde_fn=lambda s, km=k_m: poschl_teller_v_tilde(s, V0=V0)
            + sinusoidal_v_tilde_perturbation(s, eps=eps, k_mode=km),
            V0=V0,
        )
        spec_k = compute_spectrum_and_condition_numbers(sys_k, positive_real_only=True)
        omega_k = spec_k["omega"][:5]

        # Match each unperturbed mode n=0..3 to nearest perturbed eigenvalue
        rel_shifts = []
        perturbed_matched = []
        for w_unpert in omega_0[:4]:
            idx = int(np.argmin(np.abs(spec_k["omega"] - w_unpert)))
            w_match = spec_k["omega"][idx]
            rel_shifts.append(float(np.abs(w_match - w_unpert) / np.abs(w_unpert)))
            perturbed_matched.append([float(np.real(w_match)), float(np.imag(w_match))])

        wavenumber_scan[f"k_{k_m}"] = {
            "k_mode": k_m,
            "eps": eps,
            "rel_shift_n0_to_n3": rel_shifts,
            "rel_shift_percent_n0_to_n3": [100.0 * x for x in rel_shifts],
            "matched_omega_n0_to_n3": perturbed_matched,
            "all_positive_omega_real": [float(np.real(w)) for w in spec_k["omega"][:8]],
            "all_positive_omega_imag": [float(np.imag(w)) for w in spec_k["omega"][:8]],
        }

    return {
        "eps": eps,
        "unperturbed_omega_real": [float(np.real(w)) for w in omega_0[:4]],
        "unperturbed_omega_imag": [float(np.imag(w)) for w in omega_0[:4]],
        "wavenumber_scan": wavenumber_scan,
    }


def run_exp06_timedomain_observability_vs_echo_delay(N: int = 60, V0: float = 1.0) -> dict:
    """Run EXP-06: Compare spectral migration against prompt vs. late time-domain waveform mismatch."""
    sys_0 = build_hyperboloidal_system(N=N, V0=V0)
    tau_grid = np.linspace(0.0, 16.0, 481)
    wf_0 = simulate_ringdown_waveform(sys_0, tau_grid, x0_packet=-0.5, width_packet=0.5, sigma_obs_target=0.80)
    h0 = wf_0["phi_obs"]

    # Scan environmental bump positions x_bump in tortoise coordinate
    # Note: on N=60 Chebyshev-Lobatto grid, max tortoise coordinate is x_max ~ artanh(cos(pi/60)) ~ 3.65,
    # or for localized perturbation directly in sigma_bump near +1 (null infinity) / -1 (horizon),
    # let us place a smooth localized bump in tortoise coordinate x_bump in [0.5, 1.2, 1.8, 2.4, 3.0]
    # with amplitude eps_bump = 2e-3 (in V(x)) and evaluate both prompt window [0, tau_split]
    # and late echo tail window [tau_split, 16.0]!
    eps_bump = 2e-3
    x_bumps = [0.4, 1.0, 1.6, 2.2, 2.8]
    bump_results = []
    sample_waveforms = {"tau": tau_grid.tolist(), "h0_unperturbed": h0.tolist()}

    for xb in x_bumps:
        sys_b = build_hyperboloidal_system(
            N=N,
            v_tilde_fn=lambda s, x_b=xb: poschl_teller_v_tilde(s, V0=V0)
            + environmental_bump_delta_v_tilde(s, eps_bump=eps_bump, x_bump=x_b, width=0.30, k_osc=6.0),
            V0=V0,
        )
        spec_b = compute_spectrum_and_condition_numbers(sys_b, positive_real_only=True)
        wf_b = simulate_ringdown_waveform(
            sys_b, tau_grid, x0_packet=-0.5, width_packet=0.5, sigma_obs_target=0.80
        )
        hb = wf_b["phi_obs"]

        # Prompt ringdown window [0, 3.5] vs Late/Echo tail window [5.0, 15.0]
        m_prompt = compute_waveform_mismatch(tau_grid, h0, hb, tau_min=0.0, tau_max=3.5)
        m_late = compute_waveform_mismatch(tau_grid, h0, hb, tau_min=5.0, tau_max=15.0)

        # Eigenvalue shift of overtone n=2 and n=3
        exact_qnm = exact_poschl_teller_qnms(5, V0=V0)
        shift_n2 = float(
            np.min(np.abs(spec_b["omega"] - exact_qnm[2])) / np.abs(exact_qnm[2])
        )
        shift_n3 = float(
            np.min(np.abs(spec_b["omega"] - exact_qnm[3])) / np.abs(exact_qnm[3])
        )

        # SNR required to distinguish prompt waveform: rho_crit = 1 / sqrt(2 * M_prompt)
        snr_crit_prompt = float(1.0 / np.sqrt(2.0 * max(m_prompt, 1e-16)))

        bump_results.append(
            {
                "x_bump": xb,
                "eps_bump": eps_bump,
                "overtone_n2_rel_shift_percent": 100.0 * shift_n2,
                "overtone_n3_rel_shift_percent": 100.0 * shift_n3,
                "mismatch_prompt_window": m_prompt,
                "mismatch_late_echo_window": m_late,
                "late_over_prompt_mismatch_ratio": float(m_late / max(m_prompt, 1e-16)),
                "critical_gw_snr_for_prompt_detection": snr_crit_prompt,
            }
        )
        if xb in (0.4, 2.2):
            sample_waveforms[f"h_pert_xb_{xb}"] = hb.tolist()

    distant_bump = bump_results[-1]
    return {
        "eps_bump": eps_bump,
        "observer_x_tortoise": float(wf_0["x_obs"]),
        "bump_scan": bump_results,
        "hypothesis_H2_2_prompt_mismatch_below_1e4": bool(distant_bump["mismatch_prompt_window"] < 1e-4),
        "sample_waveforms": sample_waveforms,
    }


def generate_pr02_figure(exp04: dict, exp05: dict, exp06: dict) -> None:
    """Generate 3-panel publication figure for Project PR-02."""
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=200)

    # Panel A: 2D Hyperboloidal Energy-Norm Pseudospectrum & Perturbed "Flea" QNMs
    ax0 = axes[0]
    re_g = np.array(exp04["pseudospectrum_grid"]["re_grid"])
    im_g = np.array(exp04["pseudospectrum_grid"]["im_grid"])
    smin_g = np.array(exp04["pseudospectrum_grid"]["log10_smin"])

    im = ax0.contourf(re_g, im_g, smin_g, levels=np.linspace(-4.5, 0.5, 21), cmap="magma")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10} s_{\min}(\tilde{\mathbf{L}} - \omega \mathbf{I})$ [Energy Norm]")

    ax0.contour(
        re_g,
        im_g,
        smin_g,
        levels=[-3.0, -2.0, -1.0],
        colors=["cyan", "white", "yellow"],
        linewidths=[1.2, 1.4, 1.2],
        linestyles=["--", "-", ":"],
    )

    # Plot unperturbed exact Poschl-Teller QNMs (+/- Re, Im)
    w_re = np.array(exp04["modes_n0_to_n6"]["omega_real"][:5])
    w_im = np.array(exp04["modes_n0_to_n6"]["omega_imag"][:5])
    ax0.scatter(w_re, w_im, marker="o", s=65, color="lime", edgecolor="black", zorder=5, label="Exact Unperturbed QNMs")
    ax0.scatter(-w_re, w_im, marker="o", s=65, color="lime", edgecolor="black", zorder=5)

    # Overlay perturbed QNMs for high-wavenumber perturbation k=8 (EXP-05)
    k8 = exp05["wavenumber_scan"]["k_8"]
    p_re = np.array(k8["all_positive_omega_real"][:6])
    p_im = np.array(k8["all_positive_omega_imag"][:6])
    ax0.scatter(
        p_re,
        p_im,
        marker="^",
        s=70,
        color="cyan",
        edgecolor="black",
        zorder=6,
        label=r"Perturbed QNMs ($\epsilon=5\times 10^{-3}, k=8$)",
    )
    ax0.scatter(-p_re, p_im, marker="^", s=70, color="cyan", edgecolor="black", zorder=6)

    ax0.set_xlim(-2.1, 2.1)
    ax0.set_ylim(5.1, 0.1)  # Inverted y-axis per QNM convention (fundamental mode n=0 at top)
    ax0.set_xlabel(r"Real Frequency $\mathrm{Re}(\omega)$")
    ax0.set_ylabel(r"Damping Rate $\mathrm{Im}(\omega)$")
    ax0.set_title("EXP-04 & 05: Energy-Norm Pseudospectrum & QNM Migration")
    ax0.legend(loc="lower left", fontsize=7.8, framealpha=0.9)

    # Panel B: Exponential Growth of Condition Number kappa_n & Wavelength Sensitivity
    ax1 = axes[1]
    ns = np.array(exp04["modes_n0_to_n6"]["n"][:6])
    kappas = np.array(exp04["modes_n0_to_n6"]["kappa"][:6])
    ax1.semilogy(
        ns,
        kappas,
        "ko-",
        lw=2.2,
        markersize=6,
        label=r"Energy-Norm Condition Number $\kappa_n$",
    )

    ax1_twin = ax1.twinx()
    for km, color, marker in [(1, "#1f77b4", "s"), (4, "#ff7f0e", "^"), (8, "#d62728", "D")]:
        shifts = exp05["wavenumber_scan"][f"k_{km}"]["rel_shift_percent_n0_to_n3"]
        ax1_twin.semilogy(
            ns[:4],
            shifts,
            color=color,
            marker=marker,
            ls="--",
            lw=1.8,
            label=f"Relative Shift $|\\Delta\\omega_n|/|\\omega_n|$ [%] ($k={km}$)",
        )

    ax1.set_xlabel("Overtone Index $n$")
    ax1.set_ylabel(r"Bauer–Fike Condition Number $\kappa_n$ (solid)")
    ax1_twin.set_ylabel(r"Eigenvalue Migration $|\Delta\omega_n|/|\omega_n|$ [%] (dashed)")
    ax1.set_title(r"EXP-04 & 05: Non-Normality $\kappa_n$ vs. Overtone Instability")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7.5, framealpha=0.9)
    ax1.grid(True, alpha=0.25)

    # Panel C: Time-Domain Ringdown Observability vs. Bump Distance (EXP-06)
    ax2 = axes[2]
    tau = np.array(exp06["sample_waveforms"]["tau"])
    h0 = np.abs(np.array(exp06["sample_waveforms"]["h0_unperturbed"])) + 1e-12
    h_near = np.abs(np.array(exp06["sample_waveforms"]["h_pert_xb_0.4"])) + 1e-12
    h_far = np.abs(np.array(exp06["sample_waveforms"]["h_pert_xb_2.2"])) + 1e-12

    ax2.semilogy(tau, h0, "k-", lw=2.0, label="Unperturbed Black Hole Ringdown $|h_0(\\tau)|$")
    ax2.semilogy(
        tau,
        np.abs(h_near - h0) + 1e-12,
        color="#d62728",
        ls="--",
        lw=1.8,
        label=r"Waveform Residual $|\Delta h(\tau)|$ (Near Bump $x_b=0.4$)",
    )
    ax2.semilogy(
        tau,
        np.abs(h_far - h0) + 1e-12,
        color="#2ca02c",
        ls="-.",
        lw=2.0,
        label=r"Waveform Residual $|\Delta h(\tau)|$ (Distant Bump $x_b=2.2$)",
    )
    ax2.axvspan(0.0, 3.5, color="gray", alpha=0.12, label="Prompt Ringdown Window $\\tau \\in [0, 3.5]$")
    ax2.set_xlim(0.0, 15.0)
    ax2.set_ylim(1e-7, 2.0)
    ax2.set_xlabel(r"Hyperboloidal Time $\tau$")
    ax2.set_ylabel(r"Ringdown Amplitude $|h(\tau)|$ & Residual $|\Delta h(\tau)|$")
    ax2.set_title("EXP-06: Prompt Ringdown Immunity vs. Late Echo Residual")
    ax2.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    print("Running BM-02: Poschl-Teller Hyperboloidal Spectral Convergence & Energy Dissipation...")
    bm02 = run_bm02_poschl_teller_validation(V0=1.0)

    print("Running EXP-04: Energy-Norm Condition Numbers & 2D Pseudospectrum Grid...")
    exp04 = run_exp04_pseudospectrum_and_condition_numbers(N=50, V0=1.0)

    print("Running EXP-05: High-Wavenumber 'Flea' vs. Adiabatic Perturbation Scan...")
    exp05 = run_exp05_wavelength_and_flea_perturbation_audit(N=50, V0=1.0)

    print("Running EXP-06: Time-Domain Ringdown Waveform Mismatch & Echo Delay Audit...")
    exp06 = run_exp06_timedomain_observability_vs_echo_delay(N=60, V0=1.0)

    print("Generating PR-02 3-Panel Publication Figure...")
    generate_pr02_figure(exp04, exp05, exp06)

    elapsed = time.time() - t0
    report = {
        "project_id": "PR-02",
        "experiment_suite_id": "BM-02-and-EXP-04-to-06",
        "elapsed_seconds": elapsed,
        "BM_02_poschl_teller_validation": bm02,
        "EXP_04_pseudospectrum_and_condition_numbers": exp04,
        "EXP_05_wavelength_and_flea_perturbation_audit": exp05,
        "EXP_06_timedomain_observability_vs_echo_delay": exp06,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PR-02 suite completed in {elapsed:.2f}s. Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
