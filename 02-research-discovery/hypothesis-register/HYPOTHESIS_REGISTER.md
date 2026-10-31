# PREREGISTERED HYPOTHESIS REGISTER — Project Frontier

All scientific hypotheses are explicitly registered here **prior to executing original numerical experiments** (Milestone 8), together with their exact falsification criteria.

---

## Primary Project (`PR-01`): Horndeski Phantom-Crossing Identifiability & Prior-Volume Audit

### Hypothesis `H1.1` (Basis & Calibration Sensitivity of Phantom Crossing)
- **Statement:** When the linear CPL expansion $w(a) = w_0 + w_a(1-a)$ is replaced by a bounded Padé rational approximation $w(a) = \frac{w_0 + w_1(1-a)}{1 + w_2(1-a)}$ or orthogonal shifted-Chebyshev polynomials $T_n(1-a)$ on $a \in [a_{\mathrm{min}}, 1]$, AND when the low-redshift Type Ia supernova absolute magnitude offset $\Delta M_B$ ($z < 0.1$) is profiled independently of $z > 0.1$ Hubble-flow supernovae, the frequentist profile-likelihood preference $\Delta \chi^2_{\mathrm{prof}}$ for actual phantom-divide crossing ($\exists z \in [0, 2.5] : w(z) < -1 \text{ and } w(0) > -1$) drops by $\ge 35\%$ compared to the standard CPL fit.
- **Null Hypothesis (`H1.1_0`):** The profile-likelihood preference $\Delta \chi^2_{\mathrm{prof}}$ for phantom crossing is invariant ($< 10\%$ change) across Chebyshev/Padé bases and low-$z$ SNe split calibrations, proving the crossing is data-driven rather than a Taylor-truncation artifact.
- **Falsification Criterion:** Direct computation of $\Delta \chi^2_{\mathrm{prof}}$ via `JAX` L-BFGS-B / Newton-CG optimization over DESI DR2 BAO + Planck compressed CMB + Pantheon+/Union3/DES-SN5YR covariance matrices.

### Hypothesis `H1.2` (Horndeski Braiding Floor & Growth-Rate Slip)
- **Statement:** Any background trajectory $(w_0, w_a)$ within the $95\%$ confidence region of DESI DR2 + CMB + Pantheon+ that crosses $w = -1$ at $z_c \in [0.3, 0.8]$ requires a non-zero kinetic braiding parameter $\alpha_B(a) \ge \alpha_{B,\min}(w_0, w_a) > 0$ (assuming luminal tensor speed $\alpha_T = 0$ and minimal coupling $\alpha_M = 0$) to maintain $Q_s(a) > 0$ and $0 < c_s^2(a) \le 1$. This mandatory braiding induces a scale-independent sub-horizon modification to the effective gravitational constant $\mu(a) - 1 = \frac{\alpha_B^2}{2 c_s^2 D} > 0$, producing a $> 1.5\%$ enhancement in $f\sigma_8(z \in [0.3, 1.0])$ relative to `CAMB`'s standard `PPF` fluid approximation ($\mu \equiv 1$).
- **Null Hypothesis (`H1.2_0`):** There exist ghost-free, gradient-stable Horndeski trajectories ($\alpha_T = 0, \alpha_M = 0, D > 0, 0 < c_s^2 \le 1$) matching the DESI DR2 best-fit $w(z)$ whose induced growth shift $|\Delta f\sigma_8 / f\sigma_8|$ is $< 0.2\%$ across all $z \in [0, 2]$, making `PPF` indistinguishable from exact Horndeski perturbation theory.
- **Falsification Criterion:** Analytical minimization of $\mu(a) - 1 = \frac{\alpha_B^2}{2 c_s^2 D}$ subject to $D > 0, 0 < c_s^2 \le 1$ across the $(w_0, w_a)$ crossing region, verified by integrating the linear growth ODE for $\delta_m(a)$.
