# FIVE RANKED CANDIDATE RESEARCH PROJECTS — Milestone 5

Each candidate project has been evaluated across the **10 mandatory scoring axes** (1–10 scale, maximum 100 points) and subjected to an adversarial counterargument audit.

---

## 1. Quantitative Comparison Matrix (Scored 1–10 per Criterion)

| Criterion | **PR-01:** Horndeski / EFT Phantom-Crossing Identifiability & Prior-Volume Audit (DESI DR2) | **PR-02:** Black Hole QNM Pseudospectrum vs. Time-Domain Ringdown Observability | **PR-03:** Tomographic Cosmic Birefringence ($EB$) & ALP Background Coupling (ACT DR6 + DESI DR2) | **PR-04:** Differentiable (`JAX`) Boltzmann Tight-Coupling & Hierarchy Truncation Error Analysis | **PR-05:** Non-Instantaneous Reheating & BBN Bounds on PTA Supercooled Phase Transitions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Scientific Significance** | **9.5** | **9.0** | **8.5** | **7.5** | **8.5** |
| **2. Literature-Supported Gap** | **9.0** | **9.0** | **8.5** | **8.5** | **8.0** |
| **3. Mathematical Tractability** | **9.5** | **9.5** | **8.0** | **8.5** | **7.0** |
| **4. Computational Feasibility (Apple M5)** | **9.5** | **9.5** | **8.0** | **8.5** | **7.5** |
| **5. Accessibility of Public Data** | **10.0** | **8.5** | **7.5** | **9.5** | **8.0** |
| **6. Independent Validation Path** | **9.5** | **9.5** | **8.0** | **9.5** | **7.5** |
| **7. Defensible Contribution Potential** | **9.0** | **9.0** | **8.0** | **8.5** | **7.5** |
| **8. Required Physics Ramp-Up Speed** | **9.0** | **8.5** | **7.5** | **7.5** | **6.5** |
| **9. Time & Infrastructure Fit (16 GB RAM)** | **9.5** | **9.5** | **8.0** | **8.5** | **8.0** |
| **10. Low Risk of Duplication (10 = Lowest Risk)** | **7.5** | **8.5** | **7.0** | **8.5** | **7.0** |
| **TOTAL SCORE (/ 100)** | **92.0 / 100** | **90.5 / 100** | **79.0 / 100** | **85.0 / 100** | **75.5 / 100** |
| **RANK** | **#1 (PRIMARY)** | **#2 (BACKUP)** | **#4** | **#3** | **#5** |

---

## 2. Detailed Profiles & Strongest Counterarguments

### Rank #1 (PRIMARY): `PR-01` — Identifiability, Prior-Volume Effects, and Physical Stability of Phantom-Divide Crossing in Horndeski / EFT Dark Energy under DESI DR2, CMB, and SNe Ia
- **Research Question:** Does the $2.8\sigma\text{--}4.2\sigma$ preference for phantom-crossing dynamical dark energy ($w_0 > -1, w_a < 0$) in DESI DR2 + CMB + SNe Ia survive when: (1) replacing the linear CPL Taylor polynomial with orthogonal Chebyshev / Padé rational bases, (2) comparing prior-independent frequentist profile likelihoods against Bayesian marginal posteriors under physical Horndeski no-ghost ($Q_s > 0$) and gradient stability ($c_s^2 > 0$) priors, and (3) accounting for the minimum kinetic braiding $\alpha_B(a)$ required to cross $w = -1$ without ghost instabilities and its induced signature on linear growth $f\sigma_8(z)$?
- **Contribution Type:** Improved computational methodology + New mathematical result (explicit Jacobian and stability boundary mapping between EFT $(\alpha_K, \alpha_B)$ and multipole/Padé $w(z)$) + Observational constraint.
- **Strongest Counterargument Against Pursuing `PR-01`:**
  > **Counterargument:** The DESI dark energy anomaly is the single most crowded topic in observational cosmology in 2025–2026 (with dozens of papers appearing monthly on arXiv, including [`Cataneo:2025nonparametric`](https://arxiv.org/abs/2512.13691), [`Tsujikawa:2025crossing`](https://arxiv.org/abs/2508.17231), [`Ye:2026heftcamb`](https://arxiv.org/abs/2603.01662), and [`Sun:2026cpl`](https://arxiv.org/abs/2610.31044)). A generic "we fit another $w(z)$ parametrization to DESI BAO" paper has zero scientific value and high scoop risk.
  > **Mitigation:** We explicitly **reject** writing a generic $w(z)$ curve-fitting paper. Instead, our contribution targets the exact **mathematical & statistical interface** between (i) the singular Jacobian of the Horndeski no-ghost/gradient-stability boundary ($D = \alpha_K + \frac{3}{2}\alpha_B^2 > 0, c_s^2 > 0$) at $w = -1$, (ii) **frequentist profile likelihoods** (which eliminate Bayesian prior-volume effects entirely) implemented via exact `JAX` automatic differentiation, and (iii) the joint growth-rate slip $\mu(k,a) = G_{\mathrm{eff}}/G_N$ forced by the minimum braiding $\alpha_B$ needed to stabilize phantom crossing.

---

### Rank #2 (BACKUP): `PR-02` — Environmental vs. Intrinsic Perturbations in Black Hole Ringdown: Quantitative Pseudospectrum Bounds and Time-Domain Observability
- **Research Question:** Several 2025–2026 papers (e.g., [`GurtasDogan:2026qnm`](https://arxiv.org/abs/2610.37501), [`Rachh:2026qnm`](https://arxiv.org/abs/2608.16525)) use higher-order WKB approximations to claim detectable Quasinormal Mode (QNM) frequency shifts from astrophysical dark-matter halos around Schwarzschild black holes. However, Jaramillo et al. ([`Jaramillo:2020pseudospectrum`](https://arxiv.org/abs/2004.06434)) proved that the Regge-Wheeler/Zerilli operator is non-normal and spectrally unstable in the frequency domain while remaining stable in prompt time-domain ringdown. What is the exact quantitative relationship between the Chebyshev energy-norm pseudospectrum $\sigma_\epsilon(L)$, the breakdown of WKB approximations, and the actual time-domain LIGO/LISA waveform mismatch $\mathcal{M}(h_{\mathrm{halo}}, h_{\mathrm{vac}})$ for realistic dark-matter density profiles?
- **Contribution Type:** Improved computational methodology (demonstrating where WKB papers produce spurious observability claims vs. exact Chebyshev spectral + time-domain PDE integration) + New mathematical/numerical bound via the Kreiss Matrix Theorem.
- **Strongest Counterargument Against Pursuing `PR-02`:**
  > **Counterargument:** For realistic astrophysical dark matter halo densities ($\rho_{\mathrm{DM}} \sim 10^{-20}\text{--}10^{-12}\text{ g cm}^{-3}$ outside spikes), the time-domain ringdown mismatch during the first $\sim 50M$ after merger is physically tiny ($\mathcal{M} \ll 10^{-6}$), meaning the true result is a **null / falsification result** showing that recent WKB literature overestimates observability unless an extreme relativistic dark matter spike is present.
  > **Mitigation:** In fundamental physics, a rigorous, quantitative methodological paper proving that 6th/13th-order WKB approximations mispredict overtone shifts in non-normal black-hole potentials—and establishing the exact Kreiss/pseudospectrum threshold required for time-domain detectability—is a highly valuable, citeable peer-review contribution.

---

### Rank #3: `PR-04` — Truncated Hierarchy Reflections vs. Tight-Coupling Switching Artifacts in Differentiable (`JAX`) Einstein-Boltzmann Solvers
- **Research Question:** When computing exact parameter gradients $\partial C_\ell / \partial \theta_i$ in differentiable Boltzmann solvers (`DISCO-EB`, `HIcosmo` [`arXiv:2606.28175`]), how do artificial reflections from hierarchy truncation at $\ell_{\max}$ and stiff implicit Runge-Kutta/Kvaerno tolerances bias Fisher matrices compared to finite-difference `CLASS`/`CAMB`?
- **Strongest Counterargument Against Pursuing `PR-04`:**
  > **Counterargument:** While valuable to computational cosmologists building `JAX` emulators, this is primarily a numerical analysis / software engineering paper rather than a fundamental physics investigation.

---

### Rank #4: `PR-03` — Joint Tomographic Cosmic Birefringence ($EB$) and Background Expansion Constraints on Reionization-Era Axion-Like Particles (ACT DR6 + DESI DR2)
- **Research Question:** Can an ultralight axion-like particle (ALP) rolling between recombination and reionization simultaneously explain the Planck/ACT isotropic birefringence angle $\beta \approx 0.25^\circ$ and reduce the CMB–DESI BAO $\Omega_m$ tension without violating ACT DR6 tomographic reionization $EB$ constraints (`arXiv:2610.19049`)?
- **Strongest Counterargument Against Pursuing `PR-03`:**
  > **Counterargument:** Simultaneously on-boarding polarized CMB Boltzmann modifications ($C_\ell^{EB}$ parity-violating source integration in `CLASS`/`CAMB`) AND ACT DR6 polarization systematic calibration angles ($\Delta \psi_{\mathrm{cal}}$) has a steep learning and implementation curve for a first project, and directly overlaps [`Sirletti:2026birefringence`](https://arxiv.org/abs/2610.29764) posted on `2026-10-21`.

---

### Rank #5: `PR-05` — Non-Instantaneous Reheating and Neutrino Spectral Distortions in Supercooled First-Order Phase Transitions Confronting NANOGrav 15-yr and BBN
- **Research Question:** How does finite-duration percolation and non-instantaneous decay of a MeV-scale scalar sector into electromagnetic vs. neutrino baths shift $\Delta N_{\mathrm{eff}}$ and primordial $^4\mathrm{He}$ / $\mathrm{D/H}$ abundances for phase transitions explaining the NANOGrav 15-yr SGWB?
- **Strongest Counterargument Against Pursuing `PR-05`:**
  > **Counterargument:** Requires coupling finite-temperature effective potential bounce solvers (`CosmoTransitions`), relativistic bubble wall hydrodynamics, and momentum-dependent neutrino Boltzmann collision integrals (`NUDEC_BSM` / `PRyMordial`), creating high multi-physics complexity and a longer ramp-up before producing verified results.
