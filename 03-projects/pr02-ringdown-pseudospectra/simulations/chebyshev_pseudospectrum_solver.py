"""Chebyshev Hyperboloidal Spectral QNM, Energy-Norm Pseudospectrum, and Time-Domain Ringdown Engine.

Implements the numerical core for Project PR-02 (Benchmark BM-02 and Experiments EXP-04, EXP-05, EXP-06):
1. Chebyshev-Gauss-Lobatto collocation grids, spectral differentiation matrices D_1, D_2,
   and exact Clenshaw-Curtis quadrature weights on sigma in [-1, 1].
2. Hyperboloidal evolution operator L and physical Energy-Norm Gram matrix G_E = W^T W
   following Jaramillo, Panosso Macedo, & Al Sheikh (2021, Phys. Rev. X 11, 031003).
3. Exact Bauer-Fike condition numbers kappa_n = ||v_n||_E ||w_n||_E / |<w_n, v_n>_E|
   and SVD epsilon-pseudospectrum sigma_eps(L).
4. Exact matrix-exponential propagator U(dt) = exp(i * dt * L) for time-domain ringdown
   waveforms, prompt vs. echo window mismatch M(h_0, h_pert), and GW SNR thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import cholesky, eig, expm, inv, svdvals


@dataclass(frozen=True)
class HyperboloidalSpectralSystem:
    """Container for discretized hyperboloidal wave operator and physical energy norm."""

    N: int
    sigma: np.ndarray
    x_tortoise: np.ndarray
    weights: np.ndarray
    D1: np.ndarray
    D2: np.ndarray
    v_tilde: np.ndarray
    L: np.ndarray
    G_E: np.ndarray
    W: np.ndarray
    W_inv: np.ndarray
    L_tilde: np.ndarray


def chebyshev_lobatto_grid_and_matrices(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct N+1 Chebyshev-Gauss-Lobatto nodes, D1, D2, and Clenshaw-Curtis weights on [-1, 1]."""
    if N < 4:
        raise ValueError("Chebyshev resolution N must be >= 4.")

    j = np.arange(N + 1, dtype=np.float64)
    sigma = np.cos(np.pi * j / N)

    c = np.ones(N + 1, dtype=np.float64)
    c[0] = 2.0
    c[N] = 2.0
    c = c * ((-1.0) ** j)

    X = np.tile(sigma[:, None], (1, N + 1))
    dX = X - X.T
    D1 = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    D1 -= np.diag(np.sum(D1, axis=1))
    D2 = D1 @ D1

    # Clenshaw-Curtis quadrature weights on [-1, 1] (Trefethen 2000, Chapter 12)
    theta = np.pi * j / N
    w = np.zeros(N + 1, dtype=np.float64)
    ii = np.arange(1, N)
    v = np.ones(N - 1, dtype=np.float64)
    if N % 2 == 0:
        w[0] = 1.0 / (N**2 - 1.0)
        w[N] = w[0]
        for k in range(1, N // 2):
            v -= 2.0 * np.cos(2.0 * k * theta[1:N]) / (4.0 * (k**2) - 1.0)
        v -= np.cos(N * theta[1:N]) / (N**2 - 1.0)
    else:
        w[0] = 1.0 / (N**2)
        w[N] = w[0]
        for k in range(1, (N + 1) // 2):
            v -= 2.0 * np.cos(2.0 * k * theta[1:N]) / (4.0 * (k**2) - 1.0)
    w[1:N] = 2.0 * v / N

    return sigma, D1, D2, w


def poschl_teller_v_tilde(sigma: np.ndarray, V0: float = 1.0) -> np.ndarray:
    """Rescaled Poschl-Teller potential V_tilde(sigma) = V(x)/(1 - sigma^2) = V0."""
    return np.full_like(sigma, float(V0), dtype=np.float64)


def exact_poschl_teller_qnms(n_max: int, V0: float = 1.0) -> np.ndarray:
    """Return exact analytical Poschl-Teller QNMs omega_n = +sqrt(V0 - 1/4) + i*(n + 1/2)."""
    re_part = float(np.sqrt(max(0.0, V0 - 0.25)))
    ns = np.arange(n_max, dtype=np.float64)
    return re_part + 1j * (ns + 0.5)


def environmental_bump_delta_v_tilde(
    sigma: np.ndarray,
    eps_bump: float,
    x_bump: float,
    width: float = 0.5,
    k_osc: float = 0.0,
) -> np.ndarray:
    """Localized environmental bump or oscillatory perturbation delta_V_tilde(sigma).

    In tortoise coordinate x = artanh(sigma), a physical bump delta_V(x) = eps * sech^2((x - x0)/w)
    corresponds to rescaled hyperboloidal potential delta_V_tilde(sigma) = delta_V(x) / (1 - sigma^2).
    """
    # Clip sigma slightly inside (-1, 1) to evaluate tortoise coordinate x = artanh(sigma)
    s_safe = np.clip(sigma, -1.0 + 1e-12, 1.0 - 1e-12)
    x = np.arctanh(s_safe)
    one_minus_s2 = np.maximum(1.0 - sigma**2, 1e-12)
    envelope = 1.0 / (np.cosh((x - x_bump) / width) ** 2)
    modulation = np.cos(k_osc * x) if k_osc > 0.0 else 1.0
    # Multiply by (1 - sigma^2) if specifying perturbation directly in V_tilde, or divide if in V(x);
    # To ensure bounded support near x_bump where sech^2((x - x_bump)/w) decays faster than (1 - sigma^2),
    # delta_V(x) / (1 - sigma^2) is smooth and localized around x_bump!
    delta_v_t = eps_bump * envelope * modulation / one_minus_s2
    delta_v_t[0] = 0.0
    delta_v_t[-1] = 0.0
    return delta_v_t


def sinusoidal_v_tilde_perturbation(
    sigma: np.ndarray,
    eps: float,
    k_mode: int,
) -> np.ndarray:
    """Deterministic deterministic wavenumber perturbation delta_V_tilde(sigma) = eps * cos(k * pi * sigma)."""
    return eps * np.cos(k_mode * np.pi * sigma)


def build_hyperboloidal_system(
    N: int,
    v_tilde_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    V0: float = 1.0,
) -> HyperboloidalSpectralSystem:
    """Assemble hyperboloidal operator L and Cholesky energy-norm similarity matrix L_tilde."""
    sigma, D1, D2, w = chebyshev_lobatto_grid_and_matrices(N)
    s_safe = np.clip(sigma, -1.0 + 1e-12, 1.0 - 1e-12)
    x_tortoise = np.arctanh(s_safe)

    if v_tilde_fn is None:
        v_tilde = poschl_teller_v_tilde(sigma, V0=V0)
    else:
        v_tilde = np.asarray(v_tilde_fn(sigma), dtype=np.float64)

    p_sig = 1.0 - sigma**2
    L1 = np.diag(p_sig) @ D2 - np.diag(2.0 * sigma) @ D1 - np.diag(v_tilde)
    L2 = -(np.diag(2.0 * sigma) @ D1 + np.eye(N + 1))

    zeros = np.zeros((N + 1, N + 1), dtype=np.float64)
    eye = np.eye(N + 1, dtype=np.float64)
    block_real = np.block([[zeros, eye], [L1, L2]])
    L = -1j * block_real

    # Physical Energy Norm Gram Matrix G_E:
    # <u1, u2>_E = 1/2 int_{-1}^1 [ psi1^* psi2 + (1 - sigma^2) (d_sigma phi1)^* (d_sigma phi2) + V_tilde_0 phi1^* phi2 ] d_sigma
    # Note: We always anchor the energy norm G_E to the positive background potential max(v_tilde, V0)
    # so G_E is strictly positive-definite even if a perturbation has oscillatory sign.
    v_norm_weight = np.maximum(v_tilde, 0.25)
    G_phi = 0.5 * (D1.T @ np.diag(w * p_sig) @ D1 + np.diag(w * v_norm_weight))
    G_phi = 0.5 * (G_phi + G_phi.T)  # enforce exact symmetry
    G_psi = 0.5 * np.diag(w)
    G_E = np.block([[G_phi, zeros], [zeros, G_psi]])

    # Cholesky factorization G_E = W^T @ W (upper triangular W)
    W = cholesky(G_E, lower=False)
    W_inv = inv(W)
    L_tilde = W @ L @ W_inv

    return HyperboloidalSpectralSystem(
        N=N,
        sigma=sigma,
        x_tortoise=x_tortoise,
        weights=w,
        D1=D1,
        D2=D2,
        v_tilde=v_tilde,
        L=L,
        G_E=G_E,
        W=W,
        W_inv=W_inv,
        L_tilde=L_tilde,
    )


def compute_spectrum_and_condition_numbers(
    sys_h: HyperboloidalSpectralSystem,
    positive_real_only: bool = True,
) -> dict[str, np.ndarray]:
    """Compute eigenvalues omega_n and physical energy-norm Bauer-Fike condition numbers kappa_n."""
    eigvals, V_right = eig(sys_h.L_tilde)
    # Normalize columns of V_right to unit 2-norm (which equals unit energy norm)
    col_norms = np.linalg.norm(V_right, axis=0)
    V_norm = V_right / col_norms[None, :]
    V_inv = inv(V_norm)
    # Row norms of V_inv equal ||w_n||_E / |<w_n, v_n>_E| when ||v_n||_E = 1
    kappas = np.linalg.norm(V_inv, axis=1)

    if positive_real_only:
        mask = np.real(eigvals) > 1e-6
        eigvals = eigvals[mask]
        kappas = kappas[mask]

    # Sort by imaginary part ascending (fundamental mode n=0 has smallest damping Im(omega))
    order = np.argsort(np.imag(eigvals))
    return {
        "omega": eigvals[order],
        "kappa": kappas[order],
    }


def compute_pseudospectrum_grid(
    sys_h: HyperboloidalSpectralSystem,
    re_grid: np.ndarray,
    im_grid: np.ndarray,
) -> np.ndarray:
    """Compute log10(s_min(L_tilde - omega * I)) across a 2D complex grid (re_grid, im_grid)."""
    n_im = len(im_grid)
    n_re = len(re_grid)
    log10_smin = np.zeros((n_im, n_re), dtype=np.float64)
    eye = np.eye(sys_h.L_tilde.shape[0], dtype=np.complex128)

    for i, im_val in enumerate(im_grid):
        for j, re_val in enumerate(re_grid):
            omega = float(re_val) + 1j * float(im_val)
            s_vals = svdvals(sys_h.L_tilde - omega * eye)
            log10_smin[i, j] = np.log10(max(float(s_vals[-1]), 1e-16))

    return log10_smin


def simulate_ringdown_waveform(
    sys_h: HyperboloidalSpectralSystem,
    tau_grid: np.ndarray,
    x0_packet: float = -1.0,
    width_packet: float = 0.6,
    sigma_obs_target: float = 0.90,
) -> dict[str, np.ndarray | int | float]:
    """Evolve an initial Gaussian wavepacket in hyperboloidal time tau using exact matrix exponential."""
    N = sys_h.N
    obs_idx = int(np.argmin(np.abs(sys_h.sigma - sigma_obs_target)))
    sigma_obs = float(sys_h.sigma[obs_idx])
    x_obs = float(sys_h.x_tortoise[obs_idx])

    # Smooth outgoing Gaussian wavepacket in tortoise coordinate x
    phi_init = np.exp(-((sys_h.x_tortoise - x0_packet) ** 2) / (2.0 * (width_packet**2)))
    phi_init[0] = 0.0
    phi_init[-1] = 0.0
    psi_init = -(sys_h.D1 @ phi_init) * (1.0 - sys_h.sigma**2)
    u0 = np.concatenate([phi_init, psi_init]).astype(np.complex128)

    dt = float(tau_grid[1] - tau_grid[0])
    # Exact single-step propagator U = exp(i * dt * L)
    prop = expm(1j * dt * sys_h.L)

    n_steps = len(tau_grid)
    phi_obs = np.zeros(n_steps, dtype=np.float64)
    energy_tau = np.zeros(n_steps, dtype=np.float64)

    u_curr = u0.copy()
    for k in range(n_steps):
        phi_obs[k] = float(np.real(u_curr[obs_idx]))
        # Energy E(tau) = u^\dagger G_E u = ||W u||_2^2
        wu = sys_h.W @ u_curr
        energy_tau[k] = float(np.real(np.vdot(wu, wu)))
        if k + 1 < n_steps:
            u_curr = prop @ u_curr

    return {
        "tau": tau_grid,
        "phi_obs": phi_obs,
        "energy": energy_tau,
        "sigma_obs": sigma_obs,
        "x_obs": x_obs,
        "obs_idx": obs_idx,
    }


def compute_waveform_mismatch(
    tau: np.ndarray,
    h0: np.ndarray,
    h_pert: np.ndarray,
    tau_min: float,
    tau_max: float,
) -> float:
    """Compute time-domain gravitational-wave mismatch M in [0, 1] over window [tau_min, tau_max]."""
    mask = (tau >= tau_min) & (tau <= tau_max)
    s0 = h0[mask]
    s1 = h_pert[mask]
    norm0 = float(np.sqrt(np.trapezoid(s0**2, tau[mask])))
    norm1 = float(np.sqrt(np.trapezoid(s1**2, tau[mask])))
    if norm0 <= 0.0 or norm1 <= 0.0:
        return 0.0
    inner = float(np.abs(np.trapezoid(s0 * s1, tau[mask])))
    overlap = min(1.0, inner / (norm0 * norm1))
    return float(1.0 - overlap)
