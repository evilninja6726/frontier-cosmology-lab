# NOVELTY ASSESSMENT: Primary (`PR-01`) & Backup (`PR-02`) Projects — Milestone 5

**Principle:** *A failed novelty search does not prove originality. We must explicitly map the closest existing papers and define our exact non-duplicative boundary.*

---

## 1. Novelty Audit for Primary Project (`PR-01`)

### Closest Existing Literature (Through October 2026)
1. **DESI Collaboration (2025), `arXiv:2503.14738`:** Establishes the $2.8\sigma\text{--}4.2\sigma$ preference for $(w_0 > -1, w_a < 0)$ using `CAMB`'s Parameterized Post-Friedmann (`PPF`) fluid module and standard uniform priors on $(w_0 \in [-3, 1], w_a \in [-3, 2])$. Does not compute the Horndeski EFT stability prior Jacobian or frequentist profile likelihoods across non-CPL bases.
2. **Tsujikawa (2025), `arXiv:2508.17231` (*Crossing the phantom divide in scalar-tensor and vector-tensor theories*):** Derives analytical conditions for phantom-divide crossing in shift-symmetric and generalized Horndeski/vector-tensor theories, showing that cubic Galileon/kinetic braiding ($G_3(\phi, X) \neq 0$, i.e. $\alpha_B \neq 0$) or non-minimal coupling ($G_4(\phi) \neq \text{const}$) is mandatory to avoid ghost instabilities ($Q_s < 0$). Does not perform a statistical prior-volume vs. profile-likelihood decomposition on DESI DR2 + SNe Ia compilations.
3. **Cataneo & Koyama (2025), `arXiv:2512.13691` (*Non-parametric exploration of minimally coupled gravity with phantom crossing*):** Explores minimally coupled ($\alpha_M = 0$) dark energy trajectories crossing $w = -1$ via non-parametric reconstructions.
4. **Ye, Lin, & Pan (2026), `arXiv:2603.01662` (*$\mathcal{H}$-EFTCAMB*):** Introduces a Python-wrapped EFTCAMB module for covariant Horndeski gravity in Cobaya. Focuses on code architecture and Bayesian MCMC sampling rather than frequentist profile likelihood vs. Bayesian prior-volume decoupling.
5. **Sun, Li, & Du (2026), `arXiv:2610.31044` (*Assessing the consistency of the CPL dark-energy model with cosmological data*):** Uses simulation-calibrated neural posteriors to test dataset consistency within CPL, without Horndeski EFT stability constraints.

### Exact Non-Duplicative Scope of `PR-01`
Our investigation focuses on a specific three-way methodological synthesis that has **not** been unified in the literature above:
1. **Analytical & Numerical Prior-Volume Jacobian:** Explicitly computing the volume compression factor $\det \left|\frac{\partial(\alpha_K, \alpha_B)}{\partial(w_0, w_a)}\right|$ induced when restricting the $(w_0, w_a)$ plane to trajectories that satisfy Horndeski no-ghost ($D = \alpha_K + \frac{3}{2}\alpha_B^2 > 0$) and sub-luminal gradient stability ($0 < c_s^2(z) \le 1$) at all redshifts $z \in [0, 1100]$.
2. **Frequentist Profile Likelihood vs. Bayesian Posterior Decoupling (via `JAX` Auto-Diff):** Because `JAX` provides exact first and second derivatives ($\nabla_\theta \chi^2$, $\nabla^2_\theta \chi^2$), we can compute exact **frequentist profile likelihoods** $\Delta \chi^2_{\mathrm{prof}}(w_0, w_a) = \min_{\theta_{\mathrm{nuis}}} \chi^2(w_0, w_a, \theta_{\mathrm{nuis}})$ alongside Bayesian marginal posteriors under both CPL and orthogonal Chebyshev/Padé rational bases, cleanly isolating how much of the phantom-crossing preference comes from likelihood data versus prior-volume and Taylor-truncation artifacts.
3. **Minimum Braiding Growth Slip ($\mu(a) - 1$):** Quantifying the minimum kinetic braiding $\alpha_{B,\min}(z)$ required to stabilize each point in the $1\sigma/2\sigma$ DESI DR2 phantom-crossing contour and predicting the resulting irreducible deviation in the linear growth rate $f\sigma_8(z)$ relative to `CAMB`'s default `PPF` approximation (which assumes $G_{\mathrm{eff}} = G_N$).

---

## 2. Novelty Audit for Backup Project (`PR-02`)

### Closest Existing Literature (Through October 2026)
1. **Jaramillo, Panosso Macedo, & Al Sheikh (2021), `arXiv:2004.06434`:** Foundational discovery of black hole QNM spectral instability using hyperboloidal slicing and Chebyshev pseudospectra.
2. **Miyachi, Namba, & Omiya (2025), `arXiv:2512.18631`:** Exact WKB analysis of high-overtone spectral instability.
3. **Gurtas Dogan, Hassanabadi, & Guvendi (2026), `arXiv:2610.37501` & Rachh (2026), `arXiv:2608.16525`:** Compute QNM shifts of Schwarzschild black holes surrounded by King, Dehnen, and DC14 dark matter halos using higher-order WKB approximations and Padé approximants.

### Exact Non-Duplicative Scope of `PR-02`
Recent 2026 papers (`arXiv:2610.37501`, `arXiv:2608.16525`) rely on single-barrier WKB approximations to compute QNM shifts for extended dark matter halo potentials. However, WKB is a local expansion around the photon-sphere peak ($r \approx 3M$) that **cannot capture non-local backscattering or non-normal pseudospectral branch migration** induced by extended halo potentials at $r \gg 3M$. `PR-02` directly tests—and where appropriate, falsifies—WKB halo QNM predictions against exact Chebyshev hyperboloidal spectral eigenvalues, energy-norm pseudospectra, and time-domain 1+1D characteristic/method-of-lines wave evolutions.
