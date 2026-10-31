# RESEARCH STATUS — Project Frontier

**Last Updated:** 2026-10-31  
**Current Phase:** Milestone 7 Baseline Completed & Verified (`BM-01` PASSED; Ready for Milestone 8 Original Extensions)  
**Active Project:** `PR-01` — *Identifiability, Prior-Volume Effects, and Physical Stability of Phantom-Divide Crossing in Horndeski / EFT Dark Energy under DESI DR2, CMB, and SNe Ia*  
**Backup Project:** `PR-02` — *Environmental vs. Intrinsic Perturbations in Black Hole Ringdown: Quantitative Pseudospectrum Bounds and Time-Domain Observability with Chebyshev Spectral Methods*

---

## 1. Milestone Progress Tracker

| Milestone | Description | Status | Primary Artifact |
| :--- | :--- | :--- | :--- |
| **M1** | Environment Audit | **COMPLETED** | `infrastructure/environment/HARDWARE_AND_ENV_AUDIT.md` |
| **M2** | Skills Discovery & Installation | **COMPLETED** | `08-skills/installed-skills.md`, `08-skills/evaluated-skills.md` |
| **M3** | Scientific Repository Investigation | **COMPLETED** | `04-scientific-software/documentation/TOOLCHAIN_EVALUATION_AND_INTEGRATION.md` |
| **M4** | Systematic Literature Research | **COMPLETED** | `01-literature/literature-reviews/2026_COSMOLOGY_AND_GRAVITATION_FRONTIERS.md` |
| **M5** | Candidate Scoring & Project Selection | **COMPLETED** | `02-research-discovery/candidate-topics/FIVE_RANKED_CANDIDATES.md` |
| **M6** | Targeted Physics Learning Roadmap | **COMPLETED** | `00-foundation/learning-roadmap/PROJECT_01_LEARNING_ROADMAP.md` |
| **M7** | First Reproducible Benchmark (`BM-01`) | **COMPLETED (PASSED)** | `05-experiments/experiment-registry/BM_01_milestone7_report.json` |
| **M8** | Original Research Extensions (`EXP-01`–`EXP-03`) | **READY (NEXT)** | `03-projects/pr01-horndeski-crossing-identifiability/theory/THEORY_AND_DERIVATIONS.md` |

---

## 2. Current Research Question (PR-01)
> **Question:** To what extent is the $2.8\sigma\text{--}4.2\sigma$ preference for dynamical dark energy ($w_0 > -1, w_a < 0$) and phantom-divide crossing ($w(z) < -1 \to w(z) > -1$) reported in DESI DR1/DR2 + CMB + Type Ia Supernovae an artifact of the linear Chevallier-Polarski-Linder (CPL) Taylor expansion and uniform prior volume, versus a physically stable trajectory in ghost-free, gradient-stable Horndeski Effective Field Theory ($\alpha_K, \alpha_B, \alpha_M > 0, \alpha_T = 0$)?

### Sub-Questions
1. **Mapping Pathologies:** When a CPL trajectory $(w_0, w_a)$ crosses $w = -1$ at $z_c = -(1+w_0)/w_a$, single-scalar quintessence ($c_s^2 = 1$, $\alpha_B = 0$) develops a divergent kinetic coefficient or ghost instability ($Q_s \le 0$). What is the minimum braiding parameter $\alpha_B(a)$ required in EFT of Dark Energy to stabilize the DESI DR2 best-fit trajectory without violating gradient stability ($c_s^2 > 0$)?
2. **Prior-Volume & Basis Sensitivity:** If priors are placed uniformly on physical Horndeski EFT stability parameters $\{\alpha_K, \alpha_B, \alpha_M\}$ (or basis-independent Chebyshev/Padé expansions of $\rho_{\mathrm{DE}}(z)$) rather than uniform $(w_0, w_a)$, how much does the Bayesian evidence $\Delta \ln \mathcal{Z}$ and profile likelihood $\Delta \chi^2$ for phantom crossing shift?
3. **Calibration Discrepancy Sensitivity:** How do growth-rate ($f\sigma_8$) and ISW/CMB-lensing predictions of braided Horndeski crossing models distinguish genuine dynamical dark energy from a distance-ladder/supernova zero-point offset ($\Delta M_B$) between Pantheon+, Union3, and DES-SN5YR/DES-Dovekie?

---

## 3. Verified Core Bibliography (Sampled from `references.bib`)
- **DESI Collaboration (2025):** *DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints*, `arXiv:2503.14738`, DOI: `10.1103/tr6y-kpc6`.
- **DESI Collaboration (2024):** *DESI 2024 VI: Cosmological Constraints from the Measurements of Baryon Acoustic Oscillations*, `arXiv:2404.03002`, DOI: `10.1088/1475-7516/2025/02/021`.
- **Planck Collaboration (2020):** *Planck 2018 results. VI. Cosmological parameters*, `arXiv:1807.06209`, DOI: `10.1051/0004-6361/201833910`.
- **Brout et al. (2022):** *The Pantheon+ Analysis: Cosmological Constraints*, `arXiv:2202.04077`, DOI: `10.3847/1538-4357/ac8e04`.
- **Gubitosi, Piazza, & Vernizzi (2013):** *The Effective Field Theory of Dark Energy*, `arXiv:1210.0201`, DOI: `10.1088/1475-7516/2013/02/032`.
- **Bellini & Sawicki (2014):** *Maximal freedom at minimum cost: linear large-scale structure in general modifications of gravity*, `arXiv:1404.3713`, DOI: `10.1088/1475-7516/2014/07/050`.
- **Zumalacárregui et al. (2017):** *hi_class: Horndeski in the Cosmic Linear Anisotropy Solving System*, `arXiv:1605.06102`, DOI: `10.1088/1475-7516/2017/08/019`.
- **Tsujikawa (2025):** *Crossing the phantom divide in scalar-tensor and vector-tensor theories*, `arXiv:2508.17231`.
- **Cataneo & Koyama (2025):** *Non-parametric exploration of minimally coupled gravity with phantom crossing*, `arXiv:2512.13691`.
- **Ye, Lin, & Pan (2026):** *$\mathcal{H}$-EFTCAMB: A Cobaya-Integrated, Python-Wrapped Extension of EFTCAMB for Covariant Horndeski Gravity*, `arXiv:2603.01662`.
- **Sun, Li, & Du (2026):** *Assessing the consistency of the CPL dark-energy model with cosmological data: Simulation-calibrated neural posterior disagreement*, `arXiv:2610.31044`.
- **Jaramillo, Panosso Macedo, & Al Sheikh (2021):** *Pseudospectrum and black hole quasi-normal mode (in)stability*, `arXiv:2004.06434`, DOI: `10.1103/PhysRevX.11.031003`.

---

## 4. Current Confidence & Epistemic State
- **Environment & Toolchain Assessment:** Verified (`camb==2.0.5`, `cobaya==3.6.2`, `jax==0.7.1`, `sympy==1.14.0`, `astropy==7.2.3`, `getdist==1.7.7`, `pytest==9.2.0` installed and active in `.venv`).
- **Citation Database Integrity:** Verified (25/25 BibTeX entries in `01-literature/citation-database/references.bib` validated via live arXiv API query).
- **Symbolic Derivations (`L1 — Mathematically Valid`):** Verified via `SymPy` in `03-projects/pr01-horndeski-crossing-identifiability/derivations/flrw_and_horndeski_symbolic.py`. Caught and corrected a sign convention in the Padé $[1/1]$ antiderivative and proved that the naive proportional braiding ansatz $\alpha_B(a) = c_B \Omega_{\mathrm{DE}}(a)$ requires $c_B < 0$ at $w = -1$ to satisfy $D(a_c)c_s^2(a_c) > 0$.
- **Milestone 7 Baseline (`L2 — Numerically Validated` & `L5 — Independently Reproduced`):** All 9 automated `pytest` tests (`T7.1`–`T7.5` + symbolic + citation checks) passed:
  - `T7.1` ($E(z)$ vs. `CAMB` across $z \in [0, 1100]$): max relative error $2.58 \times 10^{-7}$ (gate: $< 10^{-5}$).
  - `T7.2` ($D_M(z), D_H(z)$ vs. `CAMB` at DESI DR2 redshifts): max relative error $7.93 \times 10^{-10}$ and $4.03 \times 10^{-12}$ (gate: $< 10^{-5}$).
  - `T7.3` ($r_d$ vs. Planck 2018 & `CAMB`): $r_d = 147.1030\text{ Mpc}$ ($8.83 \times 10^{-5}$ vs. Planck $147.09\text{ Mpc}$; $1.87 \times 10^{-6}$ vs. `CAMB`).
  - `T7.4` ($D_+(z)$ vs. `CAMB` $\sigma_8(z)/\sigma_8(0)$): max relative error $2.41 \times 10^{-4}$ (gate: $< 5 \times 10^{-4}$).
  - `T7.5` (DESI DR2 BAO + Planck 2018 compressed CMB fit): $\Lambda$CDM best-fit $H_0 = 68.48\text{ km s}^{-1}\text{ Mpc}^{-1}, \Omega_m = 0.3001, \chi^2 = 14.714$; CPL $w_0w_a$CDM best-fit $w_0 = -0.498, w_a = -1.484, \Omega_m = 0.3458, H_0 = 64.22\text{ km s}^{-1}\text{ Mpc}^{-1}, \chi^2 = 7.193$ ($\Delta\chi^2 = -7.521$, phantom crossing at $z_c = 0.511$).

---

## 5. Immediate Next Actions (Milestone 8 — Original Research Extensions)
1. **Execute Experiment `EXP-01` (Horndeski Stability Phase-Space & Prior-Volume Jacobian):** Map the exact region in $(w_0, w_a, c_K, c_B)$ that satisfies no-ghost ($D > 0$) and gradient/sub-luminal stability ($0 < c_s^2(z) \le 1$) across $z \in [0, 1100]$, and compute how the effective prior-volume weight shifts the $(w_0, w_a)$ posterior vs. the prior-independent frequentist profile likelihood $\Delta\chi^2_{\mathrm{prof}}$.
2. **Execute Experiment `EXP-02` (Basis Sensitivity Across CPL, Padé $[1/1]$, and Shifted Chebyshev):** Quantify whether the $\Delta\chi^2 = -7.52$ phantom-crossing preference at $z_c \approx 0.51$ persists or weakens under rational Padé and Chebyshev expansions of $w(a)$ when combined with Type Ia Supernova data.
3. **Execute Experiment `EXP-03` (Mandatory Braiding Growth Slip $\mu(a) > 1$ on $f\sigma_8(z)$):** Compute the minimum $f\sigma_8(z)$ enhancement induced by $\alpha_{B,\min}$ across the DESI DR2 $1\sigma/2\sigma$ phantom-crossing contour relative to `CAMB`'s standard `PPF` approximation ($\mu \equiv 1$).
