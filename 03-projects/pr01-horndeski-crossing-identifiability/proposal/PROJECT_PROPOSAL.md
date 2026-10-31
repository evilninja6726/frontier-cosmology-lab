# PROJECT PROPOSAL (`PR-01`): Identifiability, Prior-Volume Effects, and Physical Stability of Phantom-Divide Crossing in Horndeski / EFT Dark Energy

**Project ID:** `PR-01`  
**Status:** Selected Primary Project (Gated on Milestone 7 Baseline Reproduction)  
**Target Classification:** Improved Computational Methodology + New Mathematical Result + Observational Constraint

---

## 1. Executive Summary
Joint analyses of DESI DR1/DR2 Baryon Acoustic Oscillations ([`DESI:2025dr2`](https://arxiv.org/abs/2503.14738)), Cosmic Microwave Background (Planck PR4 + ACT DR6), and Type Ia Supernovae (Pantheon+, Union3, DES-SN5YR) report a $2.8\sigma\text{--}4.2\sigma$ preference for a time-varying dark energy equation of state $w(a) = w_0 + w_a(1-a)$ with $w_0 > -1$ and $w_a < 0$. Such trajectories cross the phantom divide ($w = -1$) at $z_c \approx 0.4\text{--}0.6$.

However, three fundamental questions remain unresolved at the intersection of gravitational theory and statistical inference:
1. **Basis Truncation vs. Data Signal:** Because $w(a) = w_0 + w_a(1-a)$ is a strict first-order Taylor expansion around $a = 1$, fitting a linear ramp across both low-redshift SNe ($z < 0.8$) and high-redshift Lyman-$\alpha$ BAO ($z \approx 2.33$) + CMB ($z_* \approx 1090$) can artificially extrapolate $w(z) < -1$ at $z > 0.5$ even if the true physical energy density $\rho_{\mathrm{DE}}(z)$ is monotonic or experiences a mild transition.
2. **Physical Horndeski Stability vs. Phenomenological `PPF` Fluid:** Standard analyses evolve perturbations across $w = -1$ using `CAMB`'s phenomenological Parameterized Post-Friedmann (`PPF`) fluid module, which hardcodes $G_{\mathrm{eff}} = G_N$ and imposes no Lagrangian stability bounds. In actual single-field scalar-tensor gravity (Horndeski / EFT of Dark Energy), crossing $w = -1$ without ghost ($D = \alpha_K + \frac{3}{2}\alpha_B^2 > 0$) or gradient ($c_s^2 > 0$) instabilities requires non-zero kinetic braiding $\alpha_B(a) \neq 0$ ([`Bellini:2014maximal`](https://arxiv.org/abs/1404.3713); [`Tsujikawa:2025crossing`](https://arxiv.org/abs/2508.17231)).
3. **Prior-Volume Distortion vs. Frequentist Profile Likelihood:** Enforcing physical EFT stability conditions ($D(z) > 0, 0 < c_s^2(z) \le 1$) over a nuisance space $(\alpha_{K,0}, \alpha_{B,0})$ induces a non-uniform effective prior volume in $(w_0, w_a)$ during Bayesian MCMC marginalization.

---

## 2. Mathematical Architecture of the Investigation

### Phase A: Analytical Derivation & `SymPy` Verification
1. Derive the exact stability region $\mathcal{S} = \{(w_0, w_a, \alpha_K, \alpha_B, \alpha_M) : D(z) > 0 \land 0 < c_s^2(z) \le 1 \ \forall z \in [0, z_*]\}$ for canonical time parameterizations of $\alpha_i(a)$ (e.g., $\alpha_i(a) = c_i \Omega_{\mathrm{DE}}(a)$ and scaling power laws).
2. Prove the lower bound $\alpha_{B,\min}(w_0, w_a)$ required to cross $w = -1$ at redshift $z_c$ with $c_s^2 \le 1$, and derive the resulting minimum enhancement in the sub-horizon effective gravitational coupling:
   $$\mu_{\min}(a) - 1 = \min_{\alpha_K, \alpha_B \in \mathcal{S}} \frac{\alpha_B^2(a)}{2 c_s^2(a) D(a)}$$

### Phase B: Differentiable (`JAX`) Profile Likelihood vs. Bayesian MCMC Engine
1. Implement a `JAX`-differentiable background + linear perturbation solver for:
   - Standard CPL $w(a) = w_0 + w_a(1-a)$
   - Orthogonal shifted-Chebyshev basis $w(a) = \sum_{n=0}^N c_n T_n^*(1-a)$
   - Rational Padé $[1/1]$ basis $w(a) = \frac{w_0 + w_1(1-a)}{1 + w_2(1-a)}$
2. Compute both **Bayesian marginal posteriors** (under uniform $(w_0, w_a)$ priors vs. uniform physical EFT $(\alpha_K, \alpha_B)$ priors) and **frequentist profile likelihoods** $\Delta \chi^2_{\mathrm{prof}}$ (using exact `JAX` Newton-CG / L-BFGS-B optimization over nuisance parameters).

### Phase C: Observational Confrontation & SNe Calibration Split
Confront the models against:
1. **DESI DR1 & DR2 BAO:** 7 effective redshift bins ($z = 0.295$ to $2.33$) with full covariance matrix ([`DESI:2025dr2`](https://arxiv.org/abs/2503.14738)).
2. **Compressed CMB Distance Priors:** Planck 2018 / PR4 + ACT DR6 acoustic scale $(R, l_a, \omega_b)$ covariance matrix ([`Planck:2018vi`](https://arxiv.org/abs/1807.06209)).
3. **Type Ia Supernovae:** Pantheon+ ([`Brout:2022pantheon`](https://arxiv.org/abs/2202.04077)), testing sensitivity to splitting the low-$z$ ($z < 0.08$) vs. Hubble-flow zero-point calibration $\Delta M_B$.
4. **Growth Rate $f\sigma_8(z)$:** Confronting the mandatory braided Horndeski growth enhancement $\mu(a) > 1$ against redshift-space distortion (RSD) measurements.
