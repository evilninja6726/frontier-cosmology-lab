# SCIENTIFIC GLOSSARY & CONVENTIONS — Project Frontier

## Metric, Units, and Fundamental Conventions
- **Metric Signature:** Mostly plus $(-,+,+,+)$ throughout General Relativity and Cosmology modules.
- **Natural Units:** $c = \hbar = k_B = 1$ unless explicitly noted; reduced Planck mass $M_{\mathrm{Pl}} \equiv (8\pi G_N)^{-1/2} \approx 2.435 \times 10^{18}\text{ GeV}$.
- **Scale Factor & Redshift:** $a(t_0) = 1$, $1 + z = a^{-1}$, conformal time $d\tau = dt/a(t)$, cosmic Hubble rate $H \equiv \dot{a}/a$, conformal Hubble rate $\mathcal{H} \equiv a'/a = aH$.

---

## Core Cosmological & Gravitational Terms

| Symbol / Term | Definition | Physical Meaning & Units |
| :--- | :--- | :--- |
| **$D_M(z)$** | $\int_0^z \frac{c\,dz'}{H(z')}$ (in flat FLRW) | Transverse comoving distance ($\text{Mpc}$). |
| **$D_H(z)$** | $c / H(z)$ | Hubble distance scale at redshift $z$ ($\text{Mpc}$). |
| **$D_V(z)$** | $\left[z D_M^2(z) D_H(z)\right]^{1/3}$ | Spherically averaged BAO dilation distance ($\text{Mpc}$). |
| **$r_d \equiv r_s(z_{\mathrm{drag}})$** | $\int_{z_{\mathrm{drag}}}^\infty \frac{c_s(z)}{H(z)}\,dz$ | Comoving sound horizon at the baryon drag epoch ($\approx 147.09\text{ Mpc}$ in Planck $\Lambda$CDM). |
| **CPL Parametrization** | $w(a) = w_0 + w_a(1 - a)$ | First-order Taylor expansion of dark energy equation of state around $a = 1$. |
| **Phantom Divide** | $w(z) = -1$ | Boundary between quintessence-like ($\dot{\rho}_{\mathrm{DE}} < 0, w > -1$) and phantom ($\dot{\rho}_{\mathrm{DE}} > 0, w < -1$) regimes. |
| **Horndeski Gravity** | Most general 4D Lorentz-invariant scalar-tensor theory with second-order Euler-Lagrange equations | Parameterized at the linear perturbation level by the 4 Bellini-Sawicki functions $\{\alpha_K, \alpha_B, \alpha_M, \alpha_T\}$. |
| **$\alpha_K(a)$ (Kineticity)** | Dimensionless kinetic coefficient of scalar field perturbations in unitary gauge | Controls the scalar no-ghost condition via $D = \alpha_K + \frac{3}{2}\alpha_B^2 > 0$. |
| **$\alpha_B(a)$ (Braiding)** | Dimensionless kinetic mixing between scalar field $\phi$ and extrinsic curvature $K$ | Enables ghost-free, gradient-stable crossing of $w = -1$. |
| **$\mu(k,a) \equiv G_{\mathrm{eff}}/G_N$** | Ratio of effective gravitational coupling in the Poisson equation to Newton's constant | Modifies the linear growth rate $f\sigma_8(z)$ of matter perturbations. |
| **$\sigma_\epsilon(L)$ (Pseudospectrum)** | $\{\omega \in \mathbb{C} : \|(L - \omega I)^{-1}\|_E > \epsilon^{-1}\}$ | Measures spectral sensitivity of a non-normal operator $L$ (such as black hole QNMs) to perturbations of norm $\epsilon$. |
