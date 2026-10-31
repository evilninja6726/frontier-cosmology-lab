# TARGETED PHYSICS & MATHEMATICS LEARNING ROADMAP (`PR-01`) — Milestone 6

**Audience:** Senior Software Engineer (strong algorithmic, systems, calculus, linear algebra, and numerical computing foundation; accelerating straight to the exact theoretical & mathematical physics required to derive, simulate, and validate **Project `PR-01`**).  
**Scope:** Strictly pruned to the **4 prerequisite modules** needed for `PR-01`. No extraneous detours into quantum field theory loop diagrams, string theory, or 3+1 AMR numerical relativity.

---

## Module 1: Relativistic FLRW Cosmology, Distance Measures, and the BAO Sound Horizon

### 1.1 Intuition (Software / Systems Analogy)
Think of the spatially flat Friedmann-Lemaître-Robertson-Walker (FLRW) metric as a **time-dependent global coordinate scaling factor** $a(t)$ applied to a 3D Euclidean spatial mesh, normalized so that today $a(t_0) = 1$, linked to redshift $z$ via $a = (1 + z)^{-1}$. Photons and acoustic waves propagate along characteristics (null or sound-cone geodesics) of this expanding grid.

### 1.2 Mathematical Formulation
With metric signature $(-,+,+,+)$ and natural units $c = 1$, the spatially flat FLRW line element is:
$$ds^2 = g_{\mu\nu}dx^\mu dx^\nu = -dt^2 + a^2(t)\left(dr^2 + r^2 d\Omega^2\right) = a^2(\tau)\left(-d\tau^2 + dr^2 + r^2 d\Omega^2\right)$$
where $d\tau = dt/a(t)$ is conformal time. Inserting $g_{\mu\nu}$ and a homogeneous perfect-fluid stress-energy tensor $T^\mu_{\ \nu} = \mathrm{diag}(-\rho, P, P, P)$ into Einstein's field equations $G_{\mu\nu} = 8\pi G_N T_{\mu\nu} = M_{\mathrm{Pl}}^{-2} T_{\mu\nu}$ yields the two **Friedmann equations** and the **covariant continuity equation** ($\nabla_\mu T^{\mu 0} = 0$):
$$H^2(a) \equiv \left(\frac{\dot{a}}{a}\right)^2 = \frac{1}{3 M_{\mathrm{Pl}}^2}\sum_i \rho_i(a), \qquad \frac{\ddot{a}}{a} = -\frac{1}{6 M_{\mathrm{Pl}}^2}\sum_i \left(\rho_i + 3P_i\right), \qquad \dot{\rho}_i + 3H(\rho_i + P_i) = 0$$
For dark energy with time-varying equation of state $w(a) \equiv P_{\mathrm{DE}}(a)/\rho_{\mathrm{DE}}(a)$, integrating the continuity equation gives:
$$\rho_{\mathrm{DE}}(z) = \rho_{\mathrm{DE},0} \exp\left(3 \int_0^z \frac{1 + w(z')}{1 + z'} dz'\right)$$
Under the **Chevallier-Polarski-Linder (CPL)** parametrization $w(a) = w_0 + w_a(1 - a) = w_0 + w_a \frac{z}{1+z}$, this integral evaluates analytically to:
$$f_{\mathrm{DE}}(z) \equiv \frac{\rho_{\mathrm{DE}}(z)}{\rho_{\mathrm{DE},0}} = (1 + z)^{3(1 + w_0 + w_a)} \exp\left(-\frac{3 w_a z}{1 + z}\right)$$

### 1.3 Physical Significance & Observable Distances
DESI BAO does not measure $H(z)$ in isolation; it measures the ratio of the **transverse comoving distance** $D_M(z)$ and the **Hubble distance** $D_H(z)$ to the **comoving sound horizon at the baryon drag epoch** $r_d = r_s(z_{\mathrm{drag}})$:
$$D_H(z) = \frac{c}{H(z)}, \qquad D_M(z) = \int_0^z \frac{c\,dz'}{H(z')}, \qquad D_V(z) = \left[z D_M^2(z) D_H(z)\right]^{1/3}, \qquad r_d = \int_{z_{\mathrm{drag}}}^{\infty} \frac{c_s(z)}{H(z)}\,dz$$
where $c_s(z) = c / \sqrt{3\left(1 + \frac{3\rho_b(z)}{4\rho_\gamma(z)}\right)}$ is the photon-baryon acoustic sound speed prior to decoupling ($z_{\mathrm{drag}} \approx 1060$).

### 1.4 Symbolic & Numerical Example (Python / SymPy)
```python
import sympy as sp

z, w0, wa = sp.symbols('z w0 wa', real=True)
a = 1 / (1 + z)
w_cpl = w0 + wa * (1 - a)
# Verify that the analytical CPL density evolution satisfies d(ln rho_DE)/dz = 3*(1 + w(z))/(1 + z)
ln_f_de = 3 * (1 + w0 + wa) * sp.log(1 + z) - 3 * wa * z / (1 + z)
residual = sp.simplify(sp.diff(ln_f_de, z) - 3 * (1 + w_cpl) / (1 + z))
assert residual == 0, f"Mismatch: {residual}"
```

### 1.5 How It Appears in `PR-01` & Exercises
- **Role in `PR-01`:** Forms the exact likelihood core for DESI DR2 BAO ($D_M/r_d, D_H/r_d$) and Pantheon+ Type Ia Supernovae ($\mu_{\mathrm{SN}}(z) = 5\log_{10}[(1+z)D_M(z)/10\text{ pc}] + M_B$).
- **Exercise 1.1:** Derive the analytical CPL density factor $f_{\mathrm{DE}}(z)$ when $w(a)$ is replaced by the [1/1] Padé approximant $w(a) = \frac{w_0 + w_1(1-a)}{1 + w_2(1-a)}$.
- **References:** Dodelson & Schmidt, *Modern Cosmology* (2nd ed., 2020), Ch. 1–2; Baumann, *Cosmology* (Cambridge Univ. Press, 2022), Ch. 1–3; [`DESI:2025dr2`](https://arxiv.org/abs/2503.14738).

---

## Module 2: Linear Cosmological Perturbations, Fluid Pathology at $w = -1$, and the `PPF` Approximation

### 2.1 Intuition
A background expansion $H(z)$ only tells us how the homogeneous grid stretches; **linear perturbation theory** tells us how small density ripples $\delta_i(\mathbf{x}, t) \equiv (\rho_i - \bar{\rho}_i)/\bar{\rho}_i$ grow under gravity vs. pressure support. Crucially, if you treat dark energy as a simple fluid and let $w(z)$ cross $-1$, the naive fluid pressure-perturbation equation divides by $(1 + w) = 0$ and **blows up to infinity**!

### 2.2 Mathematical Formulation
In Newtonian gauge, scalar metric perturbations are characterized by two gravitational potentials $(\Psi, \Phi)$:
$$ds^2 = a^2(\tau)\left[-(1 + 2\Psi)d\tau^2 + (1 - 2\Phi)\delta_{ij}dx^i dx^j\right]$$
For a dark energy fluid with rest-frame sound speed $c_s^2$, the linear continuity and Euler equations in Fourier space ($k$) contain the adiabatic sound speed $c_a^2 = w - \frac{\dot{w}}{3H(1+w)}$:
$$\dot{\delta}_{\mathrm{DE}} + 3H(c_s^2 - w)\delta_{\mathrm{DE}} + (1 + w)\left(\theta_{\mathrm{DE}} - 3\dot{\Phi}\right) + 9H^2(1 + w)(c_s^2 - c_a^2)\frac{\theta_{\mathrm{DE}}}{k^2} = 0$$
Notice the factor $(1 + w)$ in the denominator of $c_a^2$: when $\dot{w} \neq 0$ and $w \to -1$, the product $(1+w)c_a^2 = w(1+w) - \frac{\dot{w}}{3H}$ stays finite, **but** solving for the velocity divergence $\theta_{\mathrm{DE}}$ in a single-field scalar Lagrangian with $c_s^2 = 1$ forces the kinetic energy $\dot{\phi}^2 = \rho_{\mathrm{DE}}(1+w)$ to change sign, flipping the sign of the kinetic term in the action (a **ghost instability**).
To bypass this in `CAMB` without specifying a modified-gravity Lagrangian, Fang, Hu, & Lewis (2008) introduced the **Parameterized Post-Friedmann (`PPF`)** approximation, which artificially enforces $\Phi = \Psi$ ($G_{\mathrm{eff}} = G_N$) and damps dark energy clustering on sub-horizon scales.

### 2.3 Physical Significance & Role in `PR-01`
Because `CAMB`'s `PPF` module hardcodes $G_{\mathrm{eff}} = G_N$ ($\mu(k,a) = 1$) across $w = -1$, it hides the physical gravitational coupling modification required by an actual ghost-free scalar-tensor theory!
- **References:** Ma & Bertschinger (1995), `astro-ph/9506072`; Fang, Hu, & Lewis (2008), `arXiv:0808.3125`.

---

## Module 3: Horndeski Gravity, Effective Field Theory of Dark Energy (EFTofDE), and the Bellini-Sawicki $\alpha$-Basis

### 3.1 Intuition
Instead of guessing dozens of arbitrary scalar-tensor Lagrangians, **Effective Field Theory of Dark Energy** ([`Gubitosi:2012eft`](https://arxiv.org/abs/1210.0201)) chooses the "unitary gauge" where time slices coincide with uniform scalar field surfaces ($\delta\phi = 0$). In this gauge, **every possible single-field Horndeski theory of linear perturbations** is completely compressed into 4 dimensionless time-dependent functions $\{\alpha_M(a), \alpha_K(a), \alpha_B(a), \alpha_T(a)\}$ plus the background $H(a)$ ([`Bellini:2014maximal`](https://arxiv.org/abs/1404.3713)).

### 3.2 Mathematical Formulation
The four Bellini-Sawicki functions have direct physical interpretations:
1. **$\alpha_K(a)$ (Kineticity):** Standard scalar kinetic energy contribution; in canonical quintessence, $\alpha_K = \frac{3\Omega_{\mathrm{DE}}(a)(1 + w(a))}{1}$.
2. **$\alpha_B(a)$ (Kinetic Braiding):** Mixing between the scalar field derivative $\dot{\phi}$ and the metric extrinsic curvature $K = 3H$. **This is the key term that allows $w(a)$ to cross $-1$ without a ghost!**
3. **$\alpha_M(a) \equiv \frac{d\ln M_*^2}{d\ln a}$ (Planck-Mass Run Rate):** Time variation of the effective gravitational coupling $M_*^2(a)$.
4. **$\alpha_T(a) \equiv c_T^2 - 1$ (Tensor Speed Excess):** Constrained by GW170817 / GRB 170817A to $|\alpha_T| < 10^{-15}$ at late times, so we set $\alpha_T = 0$.

#### Exact Physical Stability Conditions (No-Ghost & Gradient Stability)
For scalar perturbations to avoid quantum ghost instabilities (negative kinetic energy, $Q_s > 0$) and gradient/Laplace instabilities (exponential growth at high $k$, $c_s^2 > 0$), the parameters **must** satisfy at all times $a$:
$$D(a) \equiv \alpha_K(a) + \frac{3}{2}\alpha_B^2(a) > 0$$
$$c_s^2(a) = \frac{1}{D(a)}\left[ (2 - \alpha_B)\left(\hat{\alpha} - \frac{3\Omega_m(a)}{2}\right) - \frac{3\Omega_{\mathrm{DE}}(a)(1 + w(a))}{1} + \frac{d\alpha_B}{d\ln a} \right] > 0$$
(where $\hat{\alpha} \equiv \frac{1}{2}\alpha_B(1 + \alpha_M) + \alpha_M - \frac{\dot{H}}{H^2}\frac{\alpha_B}{2}$).
Look closely at what happens when $\alpha_B = 0$ (no braiding) and $\alpha_M = 0$:
$$D(a) = \alpha_K(a), \qquad D(a) c_s^2(a) = -3\Omega_{\mathrm{DE}}(a)(1 + w(a)) \quad \text{? No: in canonical quintessence } \alpha_K = 3\Omega_{\mathrm{DE}}(1+w) \text{ and } c_s^2 = 1.$$
When $(1 + w(a))$ crosses zero into the phantom regime $1 + w(a) < 0$, if $\alpha_B = 0$, either $\alpha_K < 0$ (violating $D > 0$, a **ghost**) or $c_s^2 < 0$ (violating **gradient stability**)! Only when $\alpha_B(a) \neq 0$ can $D = \alpha_K + \frac{3}{2}\alpha_B^2 > 0$ and $c_s^2 > 0$ hold simultaneously across $w = -1$.

#### Sub-Horizon Growth Modification
On sub-horizon scales ($k \gg aH$) with $\alpha_T = \alpha_M = 0$, the Poisson equation for matter perturbations $\delta_m$ acquires an enhanced effective Newton constant:
$$\ddot{\delta}_m + 2H\dot{\delta}_m - \frac{3}{2}H^2 \Omega_m(a) \mu(a) \delta_m = 0, \qquad \mu(a) \equiv \frac{G_{\mathrm{eff}}(a)}{G_N} = 1 + \frac{\alpha_B^2(a)}{2 c_s^2(a) D(a)} \ge 1$$

### 3.3 References
- Gubitosi, Piazza, & Vernizzi (2013), [`arXiv:1210.0201`](https://arxiv.org/abs/1210.0201).
- Bellini & Sawicki (2014), [`arXiv:1404.3713`](https://arxiv.org/abs/1404.3713).
- Tsujikawa (2025), [`arXiv:2508.17231`](https://arxiv.org/abs/2508.17231).

---

## Module 4: Bayesian Posterior Inference vs. Frequentist Profile Likelihoods & Prior-Volume Geometry

### 4.1 Intuition
In software engineering terms, **Bayesian marginalization** integrates a probability density over nuisance dimensions: $P(\theta_1 | d) = \int P(\theta_1, \theta_{\mathrm{nuis}} | d)\,\pi(\theta_1, \theta_{\mathrm{nuis}})\,d\theta_{\mathrm{nuis}}$. If a nonlinear parameter constraint (such as Horndeski stability $D > 0, c_s^2 > 0$) carves out a narrow wedge in nuisance space near $\theta_1$, the **volume of that wedge** multiplies the likelihood—shifting the marginalized posterior peak even when the data likelihood $\mathcal{L}(d|\theta)$ is completely flat!
By contrast, the **frequentist profile likelihood**:
$$\mathcal{L}_{\mathrm{prof}}(\theta_1) = \max_{\theta_{\mathrm{nuis}}} \mathcal{L}(d | \theta_1, \theta_{\mathrm{nuis}}), \qquad \Delta \chi^2_{\mathrm{prof}}(\theta_1) = \min_{\theta_{\mathrm{nuis}}} \chi^2(\theta_1, \theta_{\mathrm{nuis}}) - \chi^2_{\min}$$
is **strictly invariant under reparametrizations and immune to prior-volume projection effects**. Using `JAX` automatic differentiation ($\nabla_\theta \chi^2, \nabla^2_\theta \chi^2$), we can compute both exact profile likelihoods and Bayesian posteriors to separate true data preference from prior-volume illusions.
