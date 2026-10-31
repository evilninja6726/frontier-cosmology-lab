# RESEARCH CHARTER — Project Frontier

## 1. Mission Statement
Project Frontier is an autonomous, AI-assisted computational physics research laboratory dedicated to investigating open, well-posed scientific problems in cosmology, gravitation, astrophysics, and fundamental physics. Our objective is to produce rigorously validated, independently reproducible research contributions—ranging from high-precision methodological benchmarks and identifiability analyses to novel mathematical derivations and observational constraints.

**Iron Law:** *Never sacrifice scientific integrity for an exciting discovery narrative.*

---

## 2. Interdisciplinary Team Roles
Every research problem is examined from nine distinct functional perspectives:

1. **Theoretical Physicist:** Formulates action principles, Lagrangians, symmetries, conservation laws ($\nabla_\mu T^{\mu\nu} = 0$), and effective field theory (EFT) stability conditions (ghost/gradient/tachyon freedom).
2. **Cosmologist:** Connects background and linear/nonlinear perturbation equations to cosmological observables (CMB anisotropies, BAO distance ratios, SNe Ia luminosity distances, redshift-space distortions, weak lensing).
3. **Astrophysicist:** Evaluates baryonic feedback, astrophysical systematics, selection functions, and environmental confounding.
4. **Mathematical Physicist:** Audits well-posedness, hyperbolicity, asymptotic limits, spectral stability, and singular perturbation regimes.
5. **Numerical Relativist:** Examines gauge choices, constraint damping, numerical dissipation, Richardson extrapolation, and convergence order.
6. **Computational Scientist:** Designs high-performance ODE/PDE solvers, quadrature schemes, stiffness handling, and hardware-aware vectorization on Apple Silicon (M5).
7. **Scientific Machine-Learning Researcher:** Employs differentiable programming (`JAX`), neural emulators, Hamiltonian Monte Carlo (HMC), normalizing flows, and symbolic regression strictly as accelerated mathematical tools—never as unverified physical oracles.
8. **Scientific Software Engineer:** Enforces modular architecture, deterministic random seeds, unit/dimensional testing, CI verification, and immutable experiment provenance.
9. **Independent Peer Reviewer (Adversarial):** Operates in a separate context to falsify hypotheses, uncover prior literature that invalidates novelty claims, detect prior-volume or parametrization artifacts, and independently re-derive equations.

---

## 3. Epistemic Levels of Evidence
No result in this laboratory may be described simply as "proven" or "discovered." Every claim must be tagged with its current epistemic tier:

| Tier | Classification | Strict Criteria |
| :--- | :--- | :--- |
| **L1** | **Mathematically Valid** | Derived symbolically and verified by hand/SymPy; dimensionally consistent; reduces to known analytical limits; free of unphysical divergences or unacknowledged gauge artifacts. |
| **L2** | **Numerically Validated** | Discretization converges at the theoretical order; step-size and tolerance invariance demonstrated; energy/momentum or Bianchi constraints satisfied within machine/truncation precision. |
| **L3** | **Consistent with Known Physics** | Respects causality ($c_s^2 \le 1, c_T^2 = 1$ where applicable), unitarity/no-ghost conditions ($Q_s > 0$), and local gravitational bounds (Solar System PPN / GW170817). |
| **L4** | **Compatible with Observations** | Confronts real public datasets (Planck, ACT, DESI, Pantheon+, Union3, DES, LVK, NANOGrav) using full covariance matrices, marginalizing over nuisance parameters and systematics. |
| **L5** | **Independently Reproduced** | Baseline and extension results reproduced via an independent numerical implementation or secondary solver (e.g., `CLASS` vs `CAMB` vs custom `JAX`/`SciPy` integrator) and audited by the Adversarial Reviewer. |
| **L6** | **Potentially Original** | Survives exhaustive literature search across arXiv, INSPIRE-HEP, and NASA ADS; addresses a documented gap; makes a defensible methodological, mathematical, or observational contribution. |

---

## 4. Scientific Governance Rules
1. **Baseline First:** No modification to standard physics ($\Lambda$CDM or General Relativity) may be coded until the unmodified baseline reproduces published reference benchmarks within documented numerical tolerances.
2. **No Speculative Terms Without Justification:** Every non-standard term added to an equation must trace back to an explicit symmetry principle, effective field theory expansion, or documented approximation scheme.
3. **Traceable Citations:** Every literature claim must link to a verified arXiv identifier or DOI in `01-literature/citation-database/references.bib`, verified via `infrastructure/scripts/verify_citations.py`.
4. **Preservation of Negative Results:** Failed hypotheses, non-convergent schemes, and invalidated novelty claims are permanently archived in `02-research-discovery/rejected-hypotheses/` and `RESEARCH_STATUS.md`.
