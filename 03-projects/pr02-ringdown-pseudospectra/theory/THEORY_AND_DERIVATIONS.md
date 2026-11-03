# Project `PR-02`: Theoretical Framework & 10-Point Mathematical Audit — Hyperboloidal Black Hole Ringdown Pseudospectra & Time-Domain Observability

**Project ID:** `PR-02`  
**Title:** *Environmental vs. Intrinsic Perturbations in Black Hole Ringdown: Quantitative Pseudospectrum Bounds and Time-Domain Observability with Chebyshev Spectral Methods*  
**Primary References:** Jaramillo, Panosso Macedo, & Al Sheikh (2021), [`arXiv:2004.06434`](https://arxiv.org/abs/2004.06434) (*Phys. Rev. X* 11, 031003); Cheung, Destounis, Panosso Macedo, Berti, & Cardoso (2022), [`arXiv:2112.07679`](https://arxiv.org/abs/2112.07679) (*Phys. Rev. Lett.* 128, 111103).

---

## 1. Software-Engineering Intuition: Why Black Hole Ringdown Operators Are Non-Normal

In standard quantum mechanics, a bound-state Hamiltonian $H = -\partial_x^2 + V(x)$ on a Hilbert space with vanishing boundary conditions is **self-adjoint (Hermitian)**: $H^\dagger = H$. Its eigenvectors are orthogonal ($\langle v_n, v_m \rangle = \delta_{nm}$), its eigenvalues $\lambda_n \in \mathbb{R}$ are purely real, and by the Bauer–Fike theorem a small perturbation $\delta H$ shifts the eigenvalues by at most the operator norm of the perturbation:
$$|\Delta\lambda_n| \le \|\delta H\|_2 \qquad (\text{Normal / Hermitian Operator: } \kappa_n = 1)$$

Aperturbed black hole is fundamentally different because it is an **open, dissipative radiative system**: gravitational waves fall irreversibly into the event horizon ($x \to -\infty$) and radiate out to future null infinity $\mathscr{I}^+$ ($x \to +\infty$). Because energy leaks out through the boundaries, the time-evolution generator $L$ is **non-Hermitian and non-normal** ($L^\dagger L \neq L L^\dagger$). Its right eigenvectors $v_n$ (Quasi-Normal Modes, QNMs) and left eigenvectors $w_n$ are strongly non-orthogonal, resulting in condition numbers:
$$\kappa_n \equiv \frac{\|v_n\|_E\,\|w_n\|_E}{|\langle w_n, v_n \rangle_E|} \gg 1$$
Under a small environmental or ultraviolet perturbation $\delta L$ of energy norm $\|\delta L\|_E = \epsilon$, the first-order eigenvalue shift is amplified by $\kappa_n$:
$$|\Delta\omega_n| \le \kappa_n \|\delta L\|_E + \mathcal{O}(\epsilon^2)$$
This amplification is captured globally in the complex frequency plane by the **$\epsilon$-pseudospectrum** $\sigma_\epsilon(L)$.

---

## Equation 1: Hyperboloidal Compactification of the Master Black Hole Wave Equation

$$-\frac{\partial^2\phi}{\partial t^2} + \frac{\partial^2\phi}{\partial x^2} - V(x)\phi = 0, \qquad x \in (-\infty, +\infty)$$

Under the compactified hyperboloidal coordinate transformation $(\tau, \sigma) \in \mathbb{R} \times [-1, 1]$ defined by:
$$\sigma = \tanh(x) \in (-1, 1), \qquad \tau = t - \ln(\cosh x), \qquad V(x) = (1 - \sigma^2)\tilde{V}(\sigma)$$
and introducing the first-order velocity variable $\psi \equiv \partial_\tau\phi$, the wave equation becomes the regular first-order linear evolution system:

$$\partial_\tau \begin{pmatrix} \phi \\ \psi \end{pmatrix} = i L \begin{pmatrix} \phi \\ \psi \end{pmatrix}, \qquad L = \frac{1}{i}\begin{pmatrix} 0 & 1 \\ L_1 & L_2 \end{pmatrix}$$

$$L_1 = \partial_\sigma\left[(1 - \sigma^2)\partial_\sigma\right] - \tilde{V}(\sigma) = (1 - \sigma^2)\partial_\sigma^2 - 2\sigma\partial_\sigma - \tilde{V}(\sigma), \qquad L_2 = -\left(2\sigma\partial_\sigma + 1\right)$$

### 10-Point Mathematical Audit
1. **Origin:** Hyperboloidal foliation of $1+1$D black hole perturbation equations (Regge–Wheeler, Zerilli, and Pöschl–Teller barriers) introduced by Zenginoğlu (2008) and formulated for spectral pseudospectrum analysis by Jaramillo, Panosso Macedo, & Al Sheikh (2021, [`arXiv:2004.06434`](https://arxiv.org/abs/2004.06434), Eqs. 15–22).
2. **Physical Interpretation:** Slices of constant $t$ intersect the bifurcation sphere and spatial infinity $i^0$, where outgoing QNM eigenfunctions $e^{-i\omega(t \mp x)}$ diverge exponentially as $e^{+\operatorname{Im}(\omega)|x|} \to \infty$ for $\operatorname{Im}(\omega) < 0$ (damped modes in the convention $\partial_\tau \sim i\omega$ with $\operatorname{Im}(\omega) > 0$, or $e^{i\omega\tau}$ where $\operatorname{Im}(\omega) > 0$ denotes exponential decay $e^{-\operatorname{Im}(\omega)\tau}$). Hyperboloidal slices $\tau = t - \ln(\cosh x)$ bend upward to intersect the future event horizon $\mathcal{H}^+$ ($\sigma = -1$) and future null infinity $\mathscr{I}^+$ ($\sigma = +1$), factoring out the oscillatory/divergent phase so QNM eigenfunctions become smooth polynomials on $\sigma \in [-1, 1]$.
3. **Variables and Parameters:**
   - $x \in (-\infty, \infty)$: Dimensionless tortoise coordinate ($r_*/(2M)$ or $\kappa r_*$).
   - $\sigma = \tanh x \in [-1, 1]$: Compactified spatial coordinate ($\sigma = -1$ at $\mathcal{H}^+$, $\sigma = +1$ at $\mathscr{I}^+$).
   - $\tau = t - \ln(\cosh x)$: Hyperboloidal time coordinate.
   - $\tilde{V}(\sigma) \equiv V(x(\sigma)) / (1 - \sigma^2)$: Rescaled effective potential barrier. For the Pöschl–Teller potential $V_{\mathrm{PT}}(x) = V_0 \operatorname{sech}^2(x) = V_0(1 - \sigma^2)$, the rescaled potential is the exact constant $\tilde{V}_{\mathrm{PT}}(\sigma) = V_0$. For the Schwarzschild Regge–Wheeler barrier ($\ell=2, s=2$), $V_{\mathrm{RW}}(r) = \left(1 - \frac{1}{r}\right)\left(\frac{\ell(\ell+1)}{r^2} - \frac{3}{r^3}\right)$ falls off as $e^{x} \sim 1 - \sigma$ at the horizon and $x^{-2}$ at infinity, remaining bounded on $[-1, 1]$.
4. **Units and Dimensional Consistency:** All coordinates $(t, x, \tau, \sigma)$ and frequencies $\omega$ are rendered dimensionless by scaling with the characteristic barrier scale (or $2M = 1$), so $[L] = [\omega] = [1]$.
5. **Assumptions:** Linearized metric/scalar perturbations on a stationary, spherically symmetric background spacetime.
6. **Boundary and Initial Conditions (Automatic Outgoing Radiation):** Because the principal second-derivative coefficient $(1 - \sigma^2)$ vanishes at $\sigma = \pm 1$, the boundaries are outflow characteristic surfaces. Any $C^1([-1, 1])$ function on the Chebyshev grid automatically satisfies outgoing radiation boundary conditions at both $\mathcal{H}^+$ and $\mathscr{I}^+$ without imposing ad-hoc Dirichlet/Robin boundary rows!
7. **Symmetries and Parity:** When $\tilde{V}(-\sigma) = \tilde{V}(\sigma)$ (symmetric barrier), $L$ commutes with parity $(\sigma \to -\sigma)$, splitting the QNM spectrum into even-parity and odd-parity branches symmetric across the imaginary axis ($\omega_n^{\pm} = \pm |\operatorname{Re}\omega_n| + i\operatorname{Im}\omega_n$).
8. **Exact Analytical Benchmark Limit (Pöschl–Teller Spectrum):**
   For $\tilde{V}_{\mathrm{PT}}(\sigma) = V_0 > \frac{1}{4}$, the eigenvalues of $L_1 \phi + i\omega L_2 \phi + \omega^2 \phi = 0$ on $C^\infty([-1, 1])$ are Legendre/Gegenbauer polynomials of degree $n = 0, 1, 2, \dots$ with exact eigenvalues:
   $$\omega_n^{\pm} = \pm\sqrt{V_0 - \frac{1}{4}} + i\left(n + \frac{1}{2}\right), \qquad n = 0, 1, 2, \dots$$
   *(where $u(\tau, \sigma) \propto e^{i\omega_n\tau} = e^{\pm i\sqrt{V_0 - 1/4}\,\tau} e^{-(n + 1/2)\tau}$ decays exponentially in $\tau$ for $\operatorname{Im}(\omega_n) = n + \frac{1}{2} > 0$).*
9. **Mathematical Stability:** Verified symbolically in [`hyperboloidal_qnm_symbolic.py`](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr02-ringdown-pseudospectra/derivations/hyperboloidal_qnm_symbolic.py) that for $\tilde{V}(\sigma) \ge 0$, all eigenvalues lie strictly in the upper half-plane $\operatorname{Im}(\omega) > 0$ (exponential decay in $\tau$).
10. **Known Limitations:** For algebraic long-range tails $V(x) \sim \ell(\ell+1)/x^2$ as $x \to +\infty$, $\tilde{V}(\sigma) = V(x)/(1-\sigma^2)$ has a logarithmic/algebraic branch cut at $\sigma = +1$ that produces the Price power-law tail $\phi(\tau) \sim \tau^{-(2\ell+3)}$ along the branch cut $\operatorname{Re}(\omega) = 0$.

---

## Equation 2: Physical Energy Norm, Dissipation Identity, and the $\epsilon$-Pseudospectrum

$$\langle u_1, u_2 \rangle_E \equiv \frac{1}{2}\int_{-1}^{1} \left[ \psi_1^*(\sigma)\psi_2(\sigma) + (1 - \sigma^2)\partial_\sigma\phi_1^*(\sigma)\partial_\sigma\phi_2(\sigma) + \tilde{V}(\sigma)\phi_1^*(\sigma)\phi_2(\sigma) \right] d\sigma = u_1^\dagger \mathbf{G}_E u_2$$

$$\frac{dE[u]}{d\tau} = -\left|\psi(\tau, +1)\right|^2 - \left|\psi(\tau, -1)\right|^2 \le 0$$

$$\sigma_\epsilon(L) \equiv \left\{ \omega \in \mathbb{C} : \left\| (L - \omega\mathbf{I})^{-1} \right\|_E > \epsilon^{-1} \right\} = \left\{ \omega \in \mathbb{C} : s_{\min}\!\left(\mathbf{W}\mathbf{L}\mathbf{W}^{-1} - \omega\mathbf{I}\right) < \epsilon \right\}$$

### 10-Point Mathematical Audit
1. **Origin:** Physical energy-norm inner product on hyperboloidal slices and Cholesky-weighted singular value decomposition (`Jaramillo:2020pseudospectrum`, Eqs. 28–39; Trefethen & Embree 2005, *Spectra and Pseudospectra*).
2. **Physical Interpretation:** Measuring perturbation size $\|\delta L\|_E$ and resolvent amplification $\|(L - \omega\mathbf{I})^{-1}\|_E$ requires a physically meaningful norm rather than the coordinate-dependent Euclidean $L^2$ vector norm on grid values. The energy norm $\|u\|_E^2 = \langle u, u \rangle_E$ measures the total conserved/dissipated wave energy on the hyperboloidal slice.
3. **Variables and Parameters:**
   - $\mathbf{G}_E = \mathbf{W}^\dagger\mathbf{W}$: Hermitian positive-definite $(2N+2)\times(2N+2)$ Gram matrix assembled via Clenshaw–Curtis quadrature weights $w_j$ on the $N+1$ Chebyshev–Gauss–Lobatto nodes $\sigma_j = \cos(j\pi/N)$.
   - $\tilde{\mathbf{L}} \equiv \mathbf{W}\mathbf{L}\mathbf{W}^{-1}$: Energy-norm similarity-transformed evolution matrix, whose standard Euclidean 2-norm equals the exact physical energy norm of $\mathbf{L}$ ($\|\mathbf{L}\|_E = \|\tilde{\mathbf{L}}\|_2$).
   - $s_{\min}(\tilde{\mathbf{L}} - \omega\mathbf{I})$: Smallest singular value of $\tilde{\mathbf{L}} - \omega\mathbf{I}$ at complex frequency $\omega \in \mathbb{C}$.
   - $\kappa_n \equiv \frac{\|\tilde{v}_n\|_2 \|\tilde{w}_n\|_2}{|\tilde{w}_n^\dagger \tilde{v}_n|} \ge 1$: Condition number of the $n$-th QNM eigenvalue.
4. **Units and Dimensional Consistency:** $\mathbf{G}_E$, $\mathbf{W}$, and $s_{\min}$ carry dimensionless wave-energy and frequency units.
5. **Assumptions:** Positive potential $\tilde{V}(\sigma) > 0$ on $(-1, 1)$ so that $\mathbf{G}_E$ is strictly positive-definite (eliminating the constant-shift zero mode $\phi = \text{const}, \psi = 0$).
6. **Boundary Dissipation Law:** As proved symbolically in [`hyperboloidal_qnm_symbolic.py`](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr02-ringdown-pseudospectra/derivations/hyperboloidal_qnm_symbolic.py), integrating $\frac{dE}{d\tau}$ by parts cancels all interior bulk terms identically, leaving only the boundary outgoing flux $-\left(|\psi(\tau, +1)|^2 + |\psi(\tau, -1)|^2\right) \le 0$.
7. **Symmetries:** Because $\mathbf{G}_E$ is real symmetric and $i\mathbf{L}$ is real, $\sigma_\epsilon(L)$ is strictly symmetric under reflection across the imaginary axis ($\omega \to -\omega^*$).
8. **Limiting Cases:** As $\epsilon \to 0$, $\sigma_\epsilon(L)$ shrinks to circular disks of radius $r_n(\epsilon) \simeq \epsilon / \kappa_n$ centered at the exact discrete QNM eigenvalues $\omega_n$. For a normal operator ($\kappa_n = 1$), $r_n(\epsilon) = \epsilon$; for black hole overtones ($\kappa_n \gg 1$), the $\epsilon$-contours merge into open parabolic/logarithmic branches spanning the complex plane long before $\epsilon$ reaches $10^{-2}$.
9. **Mathematical Stability:** Cholesky factorization $\mathbf{G}_E = \mathbf{W}^T\mathbf{W}$ is numerically well-conditioned for $N \le 120$ when Clenshaw–Curtis weights are used.
10. **Known Limitations:** High overtones $n \gtrsim N/3$ become unresolved by an $N$-node Chebyshev grid; all physical conclusions must be verified for convergence across multiple grid resolutions (e.g., $N = 40, 60, 80$).

---

## Equation 3: Environmental Perturbations vs. Prompt Time-Domain Waveform Mismatch

$$\tilde{V}_{\mathrm{pert}}(\sigma) = \tilde{V}_0(\sigma) + \delta\tilde{V}(\sigma), \qquad \mathcal{M}\!\left(h_0, h_{\mathrm{pert}}; [0, \tau_{\max}]\right) \equiv 1 - \frac{\left|\int_0^{\tau_{\max}} h_0^*(\tau)\,h_{\mathrm{pert}}(\tau)\,d\tau\right|}{\sqrt{\int_0^{\tau_{\max}} |h_0(\tau)|^2\,d\tau \int_0^{\tau_{\max}} |h_{\mathrm{pert}}(\tau)|^2\,d\tau}}$$

### 10-Point Mathematical Audit
1. **Origin:** Time-domain ringdown observability and gravitational-wave matched-filtering mismatch metric (`Cheung:2021fleas`, [`arXiv:2112.07679`](https://arxiv.org/abs/2112.07679); Boyanov et al. 2026).
2. **Physical Interpretation:** Resolves the central paradox of black hole spectroscopy: while a tiny environmental bump $\delta V(x) = \epsilon_{\mathrm{bump}} \operatorname{sech}^2[(x - x_{\mathrm{bump}})/w]$ at distance $x_{\mathrm{bump}} \gg 1$ creates a new resonance cavity between the black hole photon sphere ($x = 0$) and the bump ($x = x_{\mathrm{bump}}$) that replaces the unperturbed QNMs $\omega_n$ with densely spaced cavity modes $\tilde{\omega}_k$ ($\Delta\operatorname{Re}\omega \sim \pi / x_{\mathrm{bump}}$), causality forbids any scattered wave from reaching the observer before the round-trip light-crossing delay $\tau_{\mathrm{echo}} \simeq 2 x_{\mathrm{bump}}$.
3. **Observability Threshold:** Two gravitational-wave ringdown signals with mismatch $\mathcal{M}$ are indistinguishable at matched-filter Signal-to-Noise Ratio $\rho$ if:
   $$\mathcal{M} < \frac{1}{2\rho^2}$$
   For LIGO/Virgo/KAGRA ringdown SNR $\rho = 20$ ($\mathcal{M}_{\mathrm{crit}} = 1.25 \times 10^{-3}$) and Einstein Telescope / LISA SNR $\rho = 70$ ($\mathcal{M}_{\mathrm{crit}} = 1.02 \times 10^{-4}$), any environmental perturbation yielding $\mathcal{M}_{\mathrm{prompt}} < 10^{-4}$ is observationally indistinguishable from a vacuum black hole during the prompt ringdown!
