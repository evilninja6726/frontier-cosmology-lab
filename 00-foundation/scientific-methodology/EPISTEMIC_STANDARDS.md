# EPISTEMIC STANDARDS & VALIDATION GATES — Project Frontier

## 1. Mandatory Pre-Implementation Mathematical Checklist
Every equation implemented in `04-scientific-software/` or `03-projects/` must document:
1. **Origin & Action Principle:** Exact Lagrangian or conservation identity from which it is derived.
2. **Physical Interpretation:** Role of each term (e.g., Hubble friction, pressure gradient, kinetic braiding).
3. **Dimensional & Unit Table:** Explicit verification in SI/astrophysical units ($\text{Mpc}, \text{km s}^{-1}\text{ Mpc}^{-1}$) and natural units ($c = \hbar = 1, M_{\mathrm{Pl}} = (8\pi G_N)^{-1/2}$).
4. **Limiting Cases:** Exact analytical reduction in known limits ($\Lambda$CDM limit $w_0 \to -1, w_a \to 0$; GR limit $\alpha_K, \alpha_B, \alpha_M, \alpha_T \to 0$; early matter domination $a \ll 1 \implies \delta_m(a) \propto a$).
5. **Stability & Well-Posedness:** Verification of hyperbolicity, positivity of kinetic matrix $Q_s > 0$, and real propagation speeds $0 < c_s^2 \le 1$.

---

## 2. Numerical Convergence & Testing Standards
1. **Step-Size & Quadrature Convergence:** Every numerical integral or ODE solution must pass a Richardson extrapolation / step-halving test demonstrating that discretization error is at least $10^2\times$ smaller than observational uncertainty.
2. **Dual-Solver Cross-Check:** Key baseline quantities must be computed by two independent implementations (e.g., `CAMB` Fortran kernel vs. local `JAX`/`SciPy` Dormand-Prince 8(5,3) integrator) and asserted equal via `pytest` within documented tolerances ($< 10^{-4}$).
3. **Deterministic Provenance:** Every experiment execution logs the exact Git commit SHA, Python/package versions, random seed, configuration YAML, and runtime hash in `05-experiments/`.
