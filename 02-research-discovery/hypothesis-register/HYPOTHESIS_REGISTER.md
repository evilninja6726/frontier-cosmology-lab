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

---

## Project 2 (`PR-02`): Environmental vs. Intrinsic Perturbations in Black Hole Ringdown Pseudospectra

### Hypothesis `H2.1` (Exponential Non-Normality & Wavelength-Dependent Spectral Instability)
- **Statement:** In the physical energy norm $\|\cdot\|_E$ on hyperboloidal Chebyshev spectral grids ($\sigma \in [-1, 1]$), the Bauer–Fike condition number $\kappa_n \equiv \|v_n\|_E \|w_n\|_E / |\langle w_n, v_n \rangle_E|$ of the $n$-th Quasi-Normal Mode (QNM) grows rapidly with overtone index $n$ ($\kappa_3 / \kappa_0 > 20$). Consequently, a small potential perturbation $\delta V(\sigma)$ with energy norm $\epsilon = \|\delta V\|_E \le 10^{-2}$ destabilizes overtones $n \ge 2$ by $|\Delta\omega_n|/|\omega_n| > 5\%$, whereas the fundamental mode ($n = 0$) exhibits a sharp wavelength threshold: it is exponentially sensitive to high-wavenumber perturbations ($k_{\mathrm{pert}} \ge 4$) when $\kappa_0$ is large, or remains stable ($|\Delta\omega_0|/|\omega_0| < 1\%$) under smooth adiabatic halo shifts unless the perturbation overlaps the photon-sphere peak.
- **Null Hypothesis (`H2.1_0`):** The hyperboloidal QNM evolution operator $L$ is nearly normal ($\kappa_n \sim \mathcal{O}(1)$ for all $n \le 4$), so eigenvalue shifts satisfy $|\Delta\omega_n| \le \|\delta L\|_E$ uniformly across overtones and spatial wavenumbers.
- **Falsification Criterion:** Direct computation of the energy-norm Cholesky similarity matrix $\tilde{\mathbf{L}} = \mathbf{W}\mathbf{L}\mathbf{W}^{-1}$, left/right eigenvectors $(w_n, v_n)$, condition numbers $\kappa_n$, and SVD $\epsilon$-pseudospectrum $\sigma_\epsilon(L) = \{\omega \in \mathbb{C} : s_{\min}(\tilde{\mathbf{L}} - \omega\mathbf{I}) < \epsilon\}$.

### Hypothesis `H2.2` (Prompt Time-Domain Ringdown Immunity vs. Causal Echo Delay)
- **Statement:** For a localized environmental bump $\delta V(x) = \epsilon_{\mathrm{bump}} \operatorname{sech}^2[(x - x_{\mathrm{bump}})/w]$ of amplitude $\epsilon_{\mathrm{bump}} \le 5 \times 10^{-3}$ located at tortoise coordinate $x_{\mathrm{bump}} \ge 6$ outside the potential peak ($x = 0$), even when the perturbed eigenvalue spectrum $\{\tilde{\omega}_n\}$ migrates substantially from the unperturbed QNMs $\{\omega_n\}$, the prompt time-domain ringdown waveform $\phi(\tau, \sigma_{\mathrm{obs}})$ over $\tau \in [0, \tau_{\mathrm{echo}})$ (where $\tau_{\mathrm{echo}} \approx 2 x_{\mathrm{bump}}$ is the causal round-trip travel time) has a waveform mismatch $\mathcal{M}_{\mathrm{prompt}} \equiv 1 - \frac{|\langle h_0, h_{\mathrm{pert}}\rangle|}{\|h_0\|\|h_{\mathrm{pert}}\|} < 10^{-4}$ against the unperturbed black hole ringdown, rendering the spectral migration unobservable in the prompt ringdown up to gravitational-wave SNR $\rho \le 70$.
- **Null Hypothesis (`H2.2_0`):** Spectral migration of the fundamental or overtone QNM eigenvalues immediately alters the prompt ringdown decay rate and frequency at $\tau \in [0, x_{\mathrm{bump}}]$, producing $\mathcal{M}_{\mathrm{prompt}} > 10^{-2}$ prior to the causal echo arrival time.
- **Falsification Criterion:** Direct comparison of perturbed QNM eigenvalues $\{\tilde{\omega}_n\}$ against high-precision time-domain evolution $u(\tau) = \exp(i\tau\mathbf{L})u(0)$ across $\tau \in [0, 25]$ for varying bump distances $x_{\mathrm{bump}} \in [2, 10]$.

