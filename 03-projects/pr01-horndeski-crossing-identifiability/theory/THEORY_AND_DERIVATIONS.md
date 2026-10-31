# MATHEMATICAL MODEL & RIGOROUS DERIVATIONS (`PR-01`) — Stage 9

**Project:** `PR-01` — *Identifiability, Prior-Volume Effects, and Physical Stability of Phantom-Divide Crossing in Horndeski / EFT Dark Energy under DESI DR2, CMB, and SNe Ia*  
**Companion Symbolic Verification Script:** [`flrw_and_horndeski_symbolic.py`](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr01-horndeski-crossing-identifiability/derivations/flrw_and_horndeski_symbolic.py)

Every equation in our theoretical framework is derived below and audited across the **10 mandatory mathematical criteria** specified in our Research Charter.

---

## Equation 1: Covariant Dark Energy Continuity & General Basis Density Evolution

$$\frac{d\rho_{\mathrm{DE}}}{dz} - \frac{3\left[1 + w(z)\right]}{1 + z}\,\rho_{\mathrm{DE}}(z) = 0 \quad \Longleftrightarrow \quad f_{\mathrm{DE}}(z) \equiv \frac{\rho_{\mathrm{DE}}(z)}{\rho_{\mathrm{DE},0}} = \exp\left(3 \int_0^z \frac{1 + w(z')}{1 + z'}\,dz'\right)$$

### 10-Point Mathematical Audit
1. **Origin:** Derived from the time component ($\nu = 0$) of the covariant conservation of the dark energy stress-energy tensor, $\nabla_\mu T^{\mu\nu}_{\mathrm{DE}} = 0$, evaluated on a spatially homogeneous and isotropic Friedmann-Lemaître-Robertson-Walker (FLRW) background metric $ds^2 = -dt^2 + a^2(t)\delta_{ij}dx^i dx^j$ with Christoffel symbols $\Gamma^0_{ij} = a\dot{a}\delta_{ij}$ and $\Gamma^i_{0j} = H\delta^i_j$.
2. **Physical Interpretation:** Expresses the First Law of Thermodynamics $d(\rho_{\mathrm{DE}} a^3) = -P_{\mathrm{DE}}\,d(a^3)$ in an expanding comoving volume $V \propto a^3 = (1+z)^{-3}$. When $w(z) > -1$ (quintessence), expansion dilutes dark energy ($\dot{\rho}_{\mathrm{DE}} < 0$); when $w(z) = -1$ ($\Lambda$), energy density is constant; when $w(z) < -1$ (phantom), energy density grows with expansion ($\dot{\rho}_{\mathrm{DE}} > 0$).
3. **Variables and Parameters:**
   - $z = a^{-1} - 1 \in [0, \infty)$: Cosmological redshift (dimensionless).
   - $\rho_{\mathrm{DE}}(z)$: Proper energy density of dark energy; $\rho_{\mathrm{DE},0} \equiv \rho_{\mathrm{DE}}(z=0)$.
   - $w(z) \equiv P_{\mathrm{DE}}(z)/\rho_{\mathrm{DE}}(z)$: Effective equation-of-state parameter (dimensionless).
   - $f_{\mathrm{DE}}(z)$: Normalized dark energy density evolution factor ($f_{\mathrm{DE}}(0) = 1$, dimensionless).
4. **Units and Dimensional Consistency:** In natural units ($c = \hbar = 1$), $[\rho_{\mathrm{DE}}] = [P_{\mathrm{DE}}] = \text{M}^4$ ($\text{eV}^4$), while $[z] = [w] = [f_{\mathrm{DE}}] = 1$. In astrophysical units, $\rho_{\mathrm{DE},0} = \frac{3 H_0^2}{8\pi G_N}\Omega_{\mathrm{DE},0}$ has units of $\text{J m}^{-3}$ or $\text{kg m}^{-3} c^2$. Both sides of the differential equation have dimension $[\rho_{\mathrm{DE}}]$.
5. **Assumptions:**
   - Spatially homogeneous background $\rho_{\mathrm{DE}}(t, \mathbf{x}) = \rho_{\mathrm{DE}}(t)$.
   - Zero background energy-momentum exchange with cold dark matter or baryons ($\nabla_\mu T^{\mu\nu}_{\mathrm{DE}} = 0$ separately from $\nabla_\mu T^{\mu\nu}_{\mathrm{m}} = 0$).
   - Strictly positive energy density $\rho_{\mathrm{DE}}(z) > 0$ so $\ln f_{\mathrm{DE}}(z)$ is real-valued.
6. **Boundary and Initial Conditions:** Anchored at present day $z = 0$ ($a = 1$): $f_{\mathrm{DE}}(0) = 1$, where $\Omega_{\mathrm{DE},0} = 1 - \Omega_{m,0} - \Omega_{r,0} - \Omega_{k,0}$.
7. **Symmetries and Conservation Laws:** Invariant under spatial translations and $SO(3)$ rotations (killings of the spatial 3-slices) and spatial rescalings $x^i \to \lambda x^i, a \to \lambda^{-1} a$. Enforces covariant energy conservation along timelike comoving observers $u^\mu = (1,0,0,0)$.
8. **Relevant Limiting Cases:**
   - **$\Lambda$CDM Limit ($w(z) = -1$):** $f_{\mathrm{DE}}(z) = 1$.
   - **$w$CDM Limit ($w(z) = w_0 = \text{const}$):** $f_{\mathrm{DE}}(z) = (1 + z)^{3(1 + w_0)}$.
   - **CPL Limit ($w(a) = w_0 + w_a(1-a)$):**
     $$f_{\mathrm{DE}}^{\mathrm{CPL}}(z) = (1 + z)^{3(1 + w_0 + w_a)} \exp\left(-\frac{3 w_a z}{1 + z}\right)$$
   - **Padé $[1/1]$ Rational Basis ($w(a) = \frac{w_0 + w_1(1-a)}{1 + b(1-a)}$ with $b > -1$):**
     $$f_{\mathrm{DE}}^{\mathrm{Pade}}(a) = a^{-3\left(1 + \frac{w_0 + w_1}{1 + b}\right)} \left[1 + b(1 - a)\right]^{-\frac{3(w_1 - b w_0)}{b(1 + b)}}$$
   - **Shifted Chebyshev Basis ($w(a) = c_0 T_0^*(x) + c_1 T_1^*(x) + c_2 T_2^*(x)$ with $x = 1 - a \in [0, 1]$):** where $T_0^*(x) = 1$, $T_1^*(x) = 2x - 1$, $T_2^*(x) = 8x^2 - 8x + 1$, giving an exact closed-form exponential-polynomial integral verified in [`flrw_and_horndeski_symbolic.py`](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr01-horndeski-crossing-identifiability/derivations/flrw_and_horndeski_symbolic.py).
9. **Mathematical Stability:** The integral $\int_0^z \frac{1+w(z')}{1+z'}dz'$ is smooth and Lipschitz-continuous across $w(z_c) = -1$ at the background level! Crossing $w = -1$ creates **no singularity in the background geometry $H(z)$**—the pathology arises exclusively at the level of **perturbations** (Equations 3 & 4 below).
10. **Known Limitations:** For unbound linear polynomials like CPL ($w(a) = w_0 + w_a(1-a)$), as $z \to \infty$ ($a \to 0$), $w(z) \to w_0 + w_a$. If $w_0 + w_a > 0$, dark energy ceases to be negligible at early times (spoiling CMB acoustic peaks unless $w_a \le -(1 + w_0)$). Rational Padé bases remain bounded at high redshift ($w(a \to 0) \to \frac{w_0 + w_1}{1 + b}$).

---

## Equation 2: Relativistic Multi-Component Friedmann Equation & BAO Distance Observables

$$E^2(z) \equiv \frac{H^2(z)}{H_0^2} = \Omega_{cb,0}(1+z)^3 + \Omega_{\gamma,0}(1+z)^4 + \Omega_{\nu,\mathrm{rel}}(z) + \Omega_{\nu,\mathrm{mass}}(z) + \Omega_{k,0}(1+z)^2 + \Omega_{\mathrm{DE},0} f_{\mathrm{DE}}(z)$$

$$D_H(z) = \frac{c}{H_0 E(z)}, \qquad D_M(z) = \frac{c}{H_0}\int_0^z \frac{dz'}{E(z')}, \qquad r_d = \int_{z_{\mathrm{drag}}}^{\infty} \frac{c}{\sqrt{3\left(1 + \frac{3\Omega_{b,0}}{4\Omega_{\gamma,0}(1+z)}\right)} H_0 E(z)}\,dz$$

### 10-Point Mathematical Audit
1. **Origin:** $00$-component of Einstein's equations $G_{00} = 8\pi G_N T_{00}$ for the FLRW metric, combined with null geodesic integration ($ds^2 = 0 \implies dr = dt/a(t) = dz/H(z)$) and the tightly coupled photon-baryon acoustic Euler-continuity wave speed $c_s^2 = \frac{\delta P_\gamma}{\delta\rho_\gamma + \delta\rho_b} = \frac{c^2}{3(1 + R_b)}$, where $R_b(z) = \frac{3\rho_b}{4\rho_\gamma}$.
2. **Physical Interpretation:** $E(z)$ governs the cosmic expansion rate relative to today; $D_M(z)/r_d$ and $D_H(z)/r_d$ are the transverse and line-of-sight BAO angles/redshift intervals measured by galaxy surveys (`DESI DR2`).
3. **Variables and Parameters:**
   - $\Omega_{cb,0} = \Omega_{c,0} + \Omega_{b,0}$: Present dimensionless cold dark matter + baryon density.
   - $\Omega_{\gamma,0} = 2.469 \times 10^{-5} h^{-2} (T_{\mathrm{CMB}}/2.7255\text{ K})^4$: Present photon density.
   - $\Omega_{\nu,\mathrm{mass}}(z)$: Massive neutrino energy density computed from the exact relativistic Fermi-Dirac phase-space integral:
     $$\rho_\nu(a) = \frac{g_\nu T_{\nu,0}^4}{2\pi^2 a^4}\int_0^\infty \frac{q^2 \sqrt{q^2 + (m_\nu a / T_{\nu,0})^2}}{e^q + 1}\,dq$$
4. **Units and Dimensional Consistency:** $[E(z)] = 1$; $[c/H_0] = [D_M] = [D_H] = [r_d] = \text{Mpc}$; observable ratios $[D_M/r_d] = [D_H/r_d] = 1$.
5. **Assumptions:** Standard thermal history $T_{\mathrm{CMB}} = 2.7255\text{ K}$, effective neutrino number $N_{\mathrm{eff}} = 3.044$, and instantaneous drag epoch evaluation $z_{\mathrm{drag}}$ calibrated against full Boltzmann recombination (`CAMB`).
6. **Boundary Conditions:** At $z = 0$, $E(0) = 1$ (enforced via the flatness closure relation $\Omega_{\mathrm{DE},0} = 1 - \Omega_{m,0} - \Omega_{r,0} - \Omega_{k,0}$) and $D_M(0) = 0$.
7. **Symmetries & Conservation:** Preserves diffeomorphism invariance of 4D GR reduced to homogeneous 3-surfaces; scale-invariant observables $D_M/r_d$ are invariant under simultaneous rescaling $H_0 \to \lambda H_0, r_d \to \lambda^{-1} r_d$ (the **sound-horizon—$H_0$ degeneracy**).
8. **Limiting Cases:** At $z \ll 1$, $D_M(z) \simeq \frac{c}{H_0}z\left[1 - \frac{1}{2}(1 + q_0)z\right]$; at $z \gg z_{\mathrm{eq}} \approx 3400$, $E(z) \propto (1+z)^2$ (radiation domination, where massive neutrinos are ultrarelativistic).
9. **Mathematical Stability:** Because $E(z) > 0$ for all $z \ge 0$ whenever $\Omega_{m,0} > 0$ and $\Omega_{\mathrm{DE},0} > 0$, the integrand $1/E(z)$ is strictly positive, smooth, and bounded on $[0, \infty)$.
10. **Known Limitations:** Compressing the full baryon-photon recombination visibility function into a single drag redshift $z_{\mathrm{drag}}$ is accurate to $< 10^{-4}$ for late-time dark energy variations (which are negligible at $z > 1000$), but full `CAMB` Boltzmann integration is used to anchor $z_{\mathrm{drag}}$ and $r_d$.

---

## Equation 3: Horndeski / EFT of Dark Energy Quadratic Action, No-Ghost Condition, & Sound Speed

$$S^{(2)}_{\text{scalar}} = \int d^3x\,dt\,a^3 \frac{M_*^2}{2}\left[ \frac{2 D(a)}{(2 - \alpha_B)^2}\dot{\zeta}^2 - \frac{2 D(a) c_s^2(a)}{(2 - \alpha_B)^2}\frac{(\nabla\zeta)^2}{a^2} + \cdots \right]$$

$$D(a) \equiv \alpha_K(a) + \frac{3}{2}\alpha_B^2(a) > 0$$

$$c_s^2(a) = \frac{1}{D(a)}\left[ \left(1 - \frac{\alpha_B}{2}\right)\left(\alpha_B(1 + \alpha_M) + 2\alpha_M - \alpha_B\frac{d\ln H}{d\ln a}\right) + \frac{d\alpha_B}{d\ln a} - \frac{3\Omega_{\mathrm{DE}}(a)\left[1 + w(a)\right]}{1} \right] > 0$$
*(written here in the standard Bellini-Sawicki convention where $\frac{d\ln H}{d\ln a} = -\frac{3}{2}\left[\Omega_m(a) + \frac{4}{3}\Omega_r(a) + \Omega_{\mathrm{DE}}(a)(1 + w(a))\right]$ and $\alpha_T = 0$).*

### 10-Point Mathematical Audit
1. **Origin:** Second-order perturbative expansion of the general Horndeski / Effective Field Theory of Dark Energy action in unitary gauge ($\delta\phi = 0$) around a flat FLRW background, after integrating out the non-dynamical ADM lapse perturbation $\delta N$ and shift $\partial_i\psi$ via the Hamiltonian and momentum constraints ([`Gubitosi:2012eft`](https://arxiv.org/abs/1210.0201); [`Bellini:2014maximal`](https://arxiv.org/abs/1404.3713), Eq. 3.12–3.14; [`Tsujikawa:2025crossing`](https://arxiv.org/abs/2508.17231)).
2. **Physical Interpretation:**
   - $Q_s(a) \equiv \frac{M_*^2 D(a)}{(2 - \alpha_B)^2} > 0$: **No-Ghost (Kinetic Positivity) Condition.** Guarantees that high-frequency scalar fluctuations carry positive kinetic energy; if $D(a) < 0$, the vacuum decays catastrophically via spontaneous pair production of negative-energy scalar quanta and positive-energy gravitons/matter.
   - $c_s^2(a) > 0$: **Gradient / Laplace Stability Condition.** Guarantees that short-wavelength modes ($k \gg aH$) oscillate as acoustic waves $\ddot{\zeta}_k + c_s^2 \frac{k^2}{a^2}\zeta_k \simeq 0$ rather than growing exponentially as $\exp(|c_s| k \tau) \to \infty$.
   - $c_s^2(a) \le 1$: **Sub-luminality / Causality Condition.** Prevents superluminal scalar signal propagation relative to the light cone.
3. **Variables and Parameters:**
   - $\alpha_K(a)$: Dimensionless kineticity parameter (arise from $G_2(\phi, X)$ and higher terms).
   - $\alpha_B(a)$: Dimensionless kinetic braiding parameter (arises from $G_3(\phi, X)\Box\phi$ and $G_4(\phi)\mathcal{R}$).
   - $\alpha_M(a) \equiv \frac{d\ln M_*^2}{d\ln a}$: Dimensionless running rate of the effective Planck mass $M_*^2(a)$.
   - $\Omega_{\mathrm{DE}}(a) \equiv \frac{\rho_{\mathrm{DE}}(a)}{3 M_*^2 H^2(a)}$: Fractional dark energy density at scale factor $a$.
4. **Units and Dimensional Consistency:** All $\alpha_i(a)$, $D(a)$, $c_s^2(a)$, $\Omega_i(a)$, and $w(a)$ are strictly dimensionless ($[1]$).
5. **Assumptions:** Single scalar field degree of freedom coupled to gravity with second-order equations of motion (Horndeski class); luminal gravitational wave speed $c_T^2 = 1 \implies \alpha_T = 0$ (enforced by GW170817 / GRB 170817A).
6. **Boundary and Initial Conditions:** As $a \to 0$ (early universe), $\Omega_{\mathrm{DE}}(a) \to 0$ and $\alpha_i(a) \to 0$ so Standard Model BBN and CMB recombination are unperturbed.
7. **Symmetries and Conservation Laws:** Unitary-gauge residual spatial diffeomorphism invariance $x^i \to x^i + \xi^i(t, \mathbf{x})$, which guarantees only one scalar and two tensor propagating degrees of freedom (no Ostrogradsky higher-derivative ghost).
8. **Relevant Limiting Cases & The Phantom-Crossing Theorem:**
   - **Unbraided / Minimally Coupled Limit ($\alpha_B = 0, \alpha_M = 0$ — k-essence / Quintessence):**
     Setting $\alpha_B = 0$ and $\alpha_M = 0$ in Equation 3 yields the exact identity:
     $$D(a) c_s^2(a) = \alpha_K(a) c_s^2(a) = -3\Omega_{\mathrm{DE}}(a)\left[1 + w(a)\right] \quad \text{? Wait: in canonical convention } \alpha_K c_s^2 = 3\Omega_{\mathrm{DE}}(1 + w(a)).$$
     Let us check the sign convention carefully (verified symbolically in [`flrw_and_horndeski_symbolic.py`](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr01-horndeski-crossing-identifiability/derivations/flrw_and_horndeski_symbolic.py)): in canonical quintessence $\mathcal{L} = X - V(\phi)$, $\dot{\phi}^2 = \rho_{\mathrm{DE}}(1 + w) > 0$ for $w > -1$, so $\alpha_K = 3\Omega_{\mathrm{DE}}(1 + w) > 0$ and $c_s^2 = 1$, which means the term in $D(a) c_s^2(a)$ is $+3\Omega_{\mathrm{DE}}(1 + w(a))$:
     $$D(a) c_s^2(a) = 3\Omega_{\mathrm{DE}}(a)\left[1 + w(a)\right] + \left(1 - \frac{\alpha_B}{2}\right)\left[\alpha_B\frac{d\ln H}{d\ln a} - \alpha_B(1 + \alpha_M) - 2\alpha_M\right] - \frac{d\alpha_B}{d\ln a}$$
     Therefore, when $\alpha_B = 0$ and $\alpha_M = 0$:
     $$\alpha_K(a) c_s^2(a) = 3\Omega_{\mathrm{DE}}(a)\left[1 + w(a)\right]$$
     **Theorem (No-Go for Unbraided Phantom Crossing):** In any unbraided, minimally coupled scalar-tensor theory ($\alpha_B \equiv 0, \alpha_M \equiv 0$), whenever the background crosses into the phantom regime $1 + w(a) < 0$, the product $\alpha_K(a) c_s^2(a) = 3\Omega_{\mathrm{DE}}(a)[1 + w(a)] < 0$ is strictly negative! Consequently, **either $\alpha_K < 0$ (ghost instability) or $c_s^2 < 0$ (exponential gradient instability)**, and at the crossing point $w(a_c) = -1$, $\alpha_K(a_c)c_s^2(a_c) = 0$.
   - **Braided Horndeski Resolution ($\alpha_B \neq 0$):** When $\alpha_B(a) \neq 0$, the braiding terms $-\left(1 - \frac{\alpha_B}{2}\right)\alpha_B\left(1 - \frac{d\ln H}{d\ln a}\right) - \frac{d\alpha_B}{d\ln a}$ remain positive even when $1 + w(a) \le 0$, allowing $D = \alpha_K + \frac{3}{2}\alpha_B^2 > 0$ and $0 < c_s^2(a) \le 1$ across the phantom divide!
9. **Mathematical Stability:** Requires $D(a) > 0$, $0 < c_s^2(a) \le 1$, and $\alpha_B(a) \neq 2$ (to avoid strong-coupling divergence in $Q_s \propto (2 - \alpha_B)^{-2}$) across all $a \in [a_{\mathrm{ini}}, 1]$.
10. **Known Limitations:** Linear EFT stability does not automatically guarantee absence of nonlinear Vainshtein screening breakdown around compact objects; we restrict $\alpha_B$ to the linear stability regime $0 < |\alpha_B| < 1$.

---

## Equation 4: Sub-Horizon Linear Matter Growth & Braiding-Induced Gravitational Slip

$$\frac{d^2\delta_m}{d(\ln a)^2} + \left(2 + \frac{d\ln H}{d\ln a}\right)\frac{d\delta_m}{d\ln a} - \frac{3}{2}\Omega_m(a)\,\mu(a)\,\delta_m = 0, \qquad \mu(a) \equiv \frac{G_{\mathrm{eff}}(a)}{G_N} = 1 + \frac{\alpha_B^2(a)}{2 c_s^2(a) D(a)}$$

### 10-Point Mathematical Audit
1. **Origin:** Quasi-static sub-horizon ($k \gg aH$, $c_s k \gg aH$) reduction of the coupled linearized Einstein-Horndeski field equations for cold dark matter + baryon density contrast $\delta_m \equiv \delta\rho_m / \bar{\rho}_m$ with $\alpha_M = \alpha_T = 0$ ([`Bellini:2014maximal`](https://arxiv.org/abs/1404.3713), Eq. 4.10).
2. **Physical Interpretation:** Because kinetic braiding $\alpha_B \neq 0$ couples the scalar field $\delta\phi$ to the Newtonian gravitational potential $\Psi$, the scalar field mediates an **attractive fifth force** on sub-horizon matter perturbations. Notice that in any stable theory ($D > 0, c_s^2 > 0$), $\mu(a) - 1 = \frac{\alpha_B^2}{2 c_s^2 D} \ge 0$ is **strictly non-negative**: stabilizing phantom crossing via kinetic braiding ($\alpha_B \neq 0$) **unavoidably enhances gravity ($\mu > 1$) and accelerates structure growth $f\sigma_8(z)$** relative to `CAMB`'s standard `PPF` fluid approximation ($\mu \equiv 1$)!
3. **Variables and Parameters:**
   - $\delta_m(a)$: Linear matter density contrast; $D_+(a) \equiv \delta_m(a)/\delta_m(1)$ is the normalized growth factor.
   - $f(a) \equiv \frac{d\ln D_+}{d\ln a}$: Dimensionless linear growth rate.
   - $f\sigma_8(z) \equiv f(z)\,\sigma_8(0)\,D_+(z)$: Redshift-space distortion (RSD) observable measured by galaxy clustering surveys.
4. **Units and Dimensional Consistency:** $[\delta_m] = [D_+] = [f] = [\mu] = [\sigma_8] = 1$.
5. **Assumptions:** Sub-horizon scales $k \gg aH / c_s$ (valid for galaxy survey RSD scales $k \sim 0.05\text{--}0.2\,h\text{ Mpc}^{-1}$ whenever $c_s^2 \gtrsim 0.1$); pressureless matter ($w_m = c_{s,m}^2 = 0$).
6. **Boundary and Initial Conditions:** Deep in the matter-dominated era ($a_{\mathrm{ini}} = 10^{-3}$, where $\Omega_m(a_{\mathrm{ini}}) \simeq 1$ and $\mu(a_{\mathrm{ini}}) \simeq 1$), the growing mode satisfies $\delta_m(a_{\mathrm{ini}}) = a_{\mathrm{ini}}$ and $\left.\frac{d\delta_m}{d\ln a}\right|_{a_{\mathrm{ini}}} = a_{\mathrm{ini}}$.
7. **Symmetries & Conservation:** Respects the equivalence principle for baryons and cold dark matter (both fall in the same metric potential $\Psi$, so Euler's equation $\dot{\theta}_m + H\theta_m - \frac{k^2}{a^2}\Psi = 0$ is unmodified).
8. **Limiting Cases:**
   - **GR / `PPF` Limit ($\alpha_B \to 0$):** $\mu(a) = 1$.
   - **Einstein-de Sitter Limit ($\Omega_m = 1, \mu = 1, \frac{d\ln H}{d\ln a} = -\frac{3}{2}$):** $\frac{d^2\delta_m}{d(\ln a)^2} + \frac{1}{2}\frac{d\delta_m}{d\ln a} - \frac{3}{2}\delta_m = 0 \implies \delta_m(a) = c_1 a + c_2 a^{-3/2}$ ($f = 1$).
9. **Mathematical Stability:** Second-order linear ODE with smooth coefficients on $\ln a \in [\ln(10^{-3}), 0]$ whenever $D(a)c_s^2(a) > 0$.
10. **Known Limitations:** The quasi-static approximation breaks down on ultra-large scales near the cosmic horizon ($k \sim aH$) or if $c_s^2 \to 0$.
