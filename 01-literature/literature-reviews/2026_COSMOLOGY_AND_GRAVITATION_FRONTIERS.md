# SYSTEMATIC LITERATURE REVIEW: 2026 Cosmology & Gravitation Frontiers — Milestone 4

**Review Date:** 2026-10-31  
**Primary Sources Audited:** arXiv (`astro-ph.CO`, `gr-qc`, `hep-th`, `hep-ph`), INSPIRE-HEP, Crossref, and DESI / Planck / ACT / Pantheon+ / LVK / NANOGrav collaboration releases.  
**Traceable Bibliography:** [`01-literature/citation-database/references.bib`](file:///Users/rajat/personal/research/frontier-cosmology-lab/01-literature/citation-database/references.bib)

---

## Domain 1: Late-Time Dynamical Dark Energy & Phantom-Divide Crossing (DESI DR1 & DR2)

### 1.1 What Is Scientifically Established?
- **Standard $\Lambda$CDM Baseline:** For over two decades, cosmic acceleration has been modeled by a cosmological constant $\Lambda$ with constant equation of state $w(z) \equiv P_{\mathrm{DE}}/\rho_{\mathrm{DE}} = -1$ ([`Planck:2018vi`](https://arxiv.org/abs/1807.06209), DOI: `10.1051/0004-6361/201833910`).
- **DESI DR1 (2024) & DR2 (2025–2026) BAO Measurements:** The Dark Energy Spectroscopic Instrument (DESI) measured baryon acoustic oscillation (BAO) transverse ($D_M(z)/r_d$) and radial ($D_H(z)/r_d$) distance scales across $>14\text{ million}$ galaxies, quasars, and Lyman-$\alpha$ forest sightlines over $0.1 < z < 4.2$ ([`DESI:2024vi`](https://arxiv.org/abs/2404.03002), DOI: `10.1088/1475-7516/2025/02/021`; [`DESI:2025dr2`](https://arxiv.org/abs/2503.14738), DOI: `10.1103/tr6y-kpc6`).
- **Observed Deviation in CPL Parameter Space:** When parameterized via the Chevallier-Polarski-Linder (CPL) linear Taylor expansion $w(a) = w_0 + w_a(1 - a)$, joint analyses of DESI DR2 BAO + CMB (Planck PR4 + ACT DR6 lensing) + Type Ia Supernovae (Pantheon+, Union3, or DES-SN5YR) yield $w_0 > -1$ and $w_a < 0$, disfavoring $\Lambda$CDM ($w_0 = -1, w_a = 0$) at $2.8\sigma$ to $4.2\sigma$ depending on the supernova compilation ([`DESI:2025dr2`](https://arxiv.org/abs/2503.14738)).

### 1.2 What Remains Unresolved & Competing Explanations
1. **Physical Phantom Crossing vs. Parametrization Artifact:** A trajectory with $w_0 > -1$ and $w_a < 0$ crosses the "phantom divide" $w(z_c) = -1$ at $z_c \approx 0.4\text{--}0.6$. In a minimally coupled canonical scalar field (quintessence, $\mathcal{L} = -\frac{1}{2}g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi - V(\phi)$), $w \ge -1$ strictly; crossing $w = -1$ requires either a ghost ($Q_s < 0$, violating quantum unitarity/vacuum stability) or higher-derivative **kinetic braiding** ($\alpha_B \neq 0$) / non-minimal coupling ($\alpha_M \neq 0$) in Horndeski gravity ([`Gubitosi:2012eft`](https://arxiv.org/abs/1210.0201); [`Bellini:2014maximal`](https://arxiv.org/abs/1404.3713); [`Tsujikawa:2025crossing`](https://arxiv.org/abs/2508.17231)).
2. **Prior-Volume & Basis Sensitivity:** Recent 2025–2026 analyses ([`Cataneo:2025nonparametric`](https://arxiv.org/abs/2512.13691); [`Sun:2026cpl`](https://arxiv.org/abs/2610.31044); [`Wen:2026glitch`](https://arxiv.org/abs/2609.40176)) show that because BAO and SNe Ia constrain integrals of $w(z)$ primarily at $z < 1.5$, a linear polynomial in $1-a$ forces a correlated tilt ($w_a < 0$) when fitting high-$z$ CMB distance priors alongside low-$z$ SNe calibration offsets.
3. **What Can Be Investigated Independently:** Public DESI DR1/DR2 BAO covariance matrices, Pantheon+ covariance matrices, and Planck/ACT compressed likelihoods are 100% open and small ($<50\text{ MB}$). Mapping the exact jacobian transformation and physical stability boundary (`no-ghost` $Q_s > 0$ and `gradient stability` $c_s^2 > 0$) between Horndeski EFT $(\alpha_K, \alpha_B, \alpha_M)$ and $(w_0, w_a)$—and comparing frequentist profile likelihoods against Bayesian posteriors across non-CPL bases—is computationally tractable on an Apple M5 workstation.

---

## Domain 2: Black Hole Ringdown Spectroscopy & Pseudospectrum Instability

### 2.1 What Is Scientifically Established?
- **Quasinormal Mode (QNM) Ringdown:** Perturbations of Schwarzschild and Kerr black holes obey the Regge-Wheeler / Zerilli / Teukolsky master wave equations with dissipative boundary conditions ( purely ingoing at the horizon $r_* \to -\infty$, purely outgoing at spatial infinity $r_* \to +\infty$). Because the system radiates energy, the time-evolution generator $L$ is **non-self-adjoint** (non-normal, $[L, L^\dagger] \neq 0$).
- **Spectral Instability Discovery:** Jaramillo, Panosso Macedo, & Al Sheikh ([`Jaramillo:2020pseudospectrum`](https://arxiv.org/abs/2004.06434), DOI: `10.1103/PhysRevX.11.031003`) and Destounis & Duque ([`Destounis:2023qnm`](https://arxiv.org/abs/2308.16227)) demonstrated via the $\epsilon$-pseudospectrum $\sigma_\epsilon(L) = \{\omega \in \mathbb{C} : \|(L - \omega I)^{-1}\|_E > \epsilon^{-1}\}$ that high-overtone QNM eigenvalues ($n \ge 1$) are exponentially sensitive to tiny ultraviolet or environmental perturbations $\delta V(r)$ in the effective potential.

### 2.2 What Remains Unresolved & Competing Explanations
- **Frequency-Domain Instability vs. Time-Domain Stability:** While overtone eigenvalues $\omega_n$ migrate drastically along "Nollert-Price branches" under tiny environmental bumps (e.g., dark matter halos or accretion disks at $r \gg 2M$), prompt time-domain ringdown waveforms $h(t)$ often remain nearly unchanged until late times ($t \sim 2r_{\mathrm{bump}}$). Recent exact WKB and spectral studies ([`Miyachi:2025wkb`](https://arxiv.org/abs/2512.18631); [`GurtasDogan:2026qnm`](https://arxiv.org/abs/2610.37501)) leave open the exact quantitative bound linking the energy-norm pseudospectrum contour $\epsilon$ of realistic astrophysical dark-matter/matter profiles (Dehnen/NFW/King) to the mismatch $\mathcal{M}(h_{\mathrm{pert}}, h_{\mathrm{vac}})$ in LIGO-Virgo-KAGRA O4/O5 and LISA ringdown parameter estimation.

---

## Domain 3: Sound-Horizon-Independent $H_0$ & Inverse Distance Ladder Tensions

### 3.1 What Is Scientifically Established?
- **The Hubble Tension:** Early-universe $\Lambda$CDM inference from Planck + ACT DR6 yields $H_0 = 67.4 \pm 0.5\text{ km s}^{-1}\text{ Mpc}^{-1}$ ([`Planck:2018vi`](https://arxiv.org/abs/1807.06209)), in $>5\sigma$ tension with local Cepheid-calibrated SH0ES SNe Ia ($H_0 = 73.04 \pm 1.04\text{ km s}^{-1}\text{ Mpc}^{-1}$).
- **Sound Horizon vs. Matter-Radiation Equality Scale:** Standard BAO analyses calibrate distances using the sound horizon at baryon drag $r_d \approx 147.09\text{ Mpc}$. However, the full-shape matter power spectrum $P(k)$ and CMB lensing $\kappa\kappa$ also contain a second physical scale: the horizon size at matter-radiation equality $k_{\mathrm{eq}} \propto \Omega_m h^2$ ([`Mirpoorian:2026soundhorizon`](https://arxiv.org/abs/2610.31660); [`Najera:2026inverse`](https://arxiv.org/abs/2610.23657)).

### 3.2 What Remains Unresolved
- Extracting $H_0$ solely from $k_{\mathrm{eq}}$ (treating $r_d$ as an uncalibrated nuisance parameter or marginalizing over early recombination physics) suffers from severe degeneracies with the spectral tilt $n_s$, neutrino mass sum $\Sigma m_\nu$, and non-linear galaxy bias parameters ($b_1, b_2, b_{\mathcal{G}_2}$) in Effective Field Theory of Large-Scale Structure (EFTofLSS).

---

## Domain 4: Cosmic Birefringence & Axion-Like Particle (ALP) Dynamics

### 4.1 What Is Scientifically Established & Unresolved
- **Isotropic $EB$ Polarization Rotation:** Cross-correlations of Planck PR4 and ACT DR6 polarization maps report a tentative parity-violating rotation angle $\beta \approx 0.2^\circ\text{--}0.35^\circ$ of CMB linear polarization, modeled via a Chern-Simons coupling $\mathcal{L}_{\mathrm{int}} = -\frac{1}{4}g_{\phi\gamma}\phi F_{\mu\nu}\tilde{F}^{\mu\nu}$ to an ultralight axion-like field $\phi(t)$ ([`Huber:2026actbirefringence`](https://arxiv.org/abs/2610.19049); [`Sirletti:2026birefringence`](https://arxiv.org/abs/2610.29764)).
- **Unresolved Question:** If the ALP evolves during reionization ($z \lesssim 10$) rather than solely after recombination ($z \approx 1100$), the reionization optical depth bump in $C_\ell^{EB}$ and $C_\ell^{EE}$ changes shape AND the ALP's kinetic energy density modifies the background expansion $H(z)$ at intermediate redshifts—potentially coupling cosmic birefringence to the CMB-DESI BAO $\Omega_m$ discrepancy ([`Sirletti:2026birefringence`](https://arxiv.org/abs/2610.29764)).

---

## Domain 5: Nanohertz Gravitational Wave Background (PTA) vs. Early-Universe Phase Transitions

### 5.1 What Is Scientifically Established & Unresolved
- NANOGrav 15-yr, EPTA DR2, PPTA DR3, and CPTA detected a nanohertz stochastic gravitational wave background (SGWB) with Hellings-Downs spatial correlations.
- Cosmological explanations (supercooled first-order phase transitions, collapsing domain walls, scalar-induced GWs) oftenfit the PTA spectrum better than standard circular SMBBH inspirals, **but** recent 2025–2026 studies ([`Elahi:2026pta`](https://arxiv.org/abs/2609.25950)) demonstrate that the latent heat and dark radiation ($\Delta N_{\mathrm{eff}}$) injected near the MeV scale severely impact Big Bang Nucleosynthesis (BBN) and CMB damping tails.
