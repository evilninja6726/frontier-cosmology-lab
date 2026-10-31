# RESEARCH STATUS — Project Frontier

**Last Updated:** 2026-10-31  
**Current Phase:** Phase I & II Complete (Milestones 1–6 Executed; Milestone 7 Benchmark Specified & Ready for Execution)  
**Active Project:** `PR-01` — *Identifiability, Prior-Volume Effects, and Physical Stability of Phantom-Divide Crossing in Horndeski / EFT Dark Energy under DESI DR2, CMB, and SNe Ia*  
**Backup Project:** `PR-02` — *Environmental vs. Intrinsic Perturbations in Black Hole Ringdown: Quantitative Pseudospectrum Bounds and Time-Domain Observability with Chebyshev Spectral Methods*

---

## 1. Milestone Progress Tracker

| Milestone | Description | Status | Primary Artifact |
| :--- | :--- | :--- | :--- |
| **M1** | Environment Audit | **COMPLETED** | `infrastructure/environment/HARDWARE_AND_ENV_AUDIT.md` |
| **M2** | Skills Discovery & Security Audit | **COMPLETED** | `08-skills/installed-skills.md`, `08-skills/evaluated-skills.md` |
| **M3** | Scientific Repository Investigation | **COMPLETED** | `04-scientific-software/documentation/TOOLCHAIN_EVALUATION_AND_INTEGRATION.md` |
| **M4** | Systematic Literature Research | **COMPLETED** | `01-literature/literature-reviews/2026_COSMOLOGY_AND_GRAVITATION_FRONTIERS.md` |
| **M5** | Candidate Scoring & Project Selection | **COMPLETED** | `02-research-discovery/candidate-topics/FIVE_RANKED_CANDIDATES.md` |
| **M6** | Targeted Physics Learning Roadmap | **COMPLETED** | `00-foundation/learning-roadmap/PROJECT_01_LEARNING_ROADMAP.md` |
| **M7** | First Reproducible Benchmark | **SPECIFIED (NEXT)** | `03-projects/pr01-horndeski-crossing-identifiability/baseline/BENCHMARK_SPECIFICATION.md` |
| **M8** | Original Research Extensions | **GATED ON M7** | `03-projects/pr01-horndeski-crossing-identifiability/proposal/PROJECT_PROPOSAL.md` |

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
- **Environment & Toolchain Assessment:** Verified (100% tested on local Apple M5 hardware).
- **Citation Database Integrity:** Verified (100% of BibTeX entries in `01-literature/citation-database/references.bib` validated via live arXiv API query).
- **Primary Hypothesis Status:** **Preregistered (`H0`, `H1`, `H2`) — Untested** (awaiting Milestone 7 baseline reproduction).

---

## 5. Immediate Next Actions
1. **User Review & Approval:** Confirm selection of **PR-01** (Primary) and **PR-02** (Backup), and approve installation of the shortlisted scientific Python packages (`numpy`, `scipy`, `sympy`, `astropy`, `jax`, `matplotlib`, `camb`, `cobaya`) and vetted skills (`k-dense-ai/scientific-agent-skills` subset + `KalarisLabs/research-agent-skills@citation-verification`).
2. **Execute Milestone 7 Baseline:** Implement and run the automated $\Lambda$CDM and $w_0w_a$CDM benchmark suite comparing custom `SciPy`/`JAX` FLRW background + linear growth integrators against `CAMB` and published DESI DR2 (`arXiv:2503.14738`) / Planck 2018 (`arXiv:1807.06209`) tables to $< 10^{-4}$ relative error.
3. **Complete Module 1 of Learning Program:** Walk through the FLRW metric, comoving distances, and BAO sound horizon $r_d$ derivation alongside the Milestone 7 code.
