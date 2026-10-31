# ADVERSARIAL PEER REVIEW PROTOCOL — Stage 14

**Core Mandate:** *A result is not validated because the same AI that generated it says it is correct.*

Every derivation, simulation, and observational claim produced in `03-projects/` must undergo a dedicated **Adversarial Review Stage** executed by an independent subagent (`invoke_subagent` with `Role="Adversarial Physics Reviewer"`) operating under a strict falsification prompt.

---

## 1. The 11-Point Adversarial Falsification Checklist

The Adversarial Reviewer must systematically attempt to break the research team's conclusions across 11 axes:

1. **Prior-Art Invalidation Search:** Query arXiv, INSPIRE-HEP, and ADS specifically looking for papers that already proved or refuted the claimed result.
2. **Independent Symbolic Re-Derivation:** Re-derive the core equations from the action/metric in a fresh `SymPy` script without importing the main research team's derivation module.
3. **Mathematical Consistency & Singularities:** Audit denominators, branch cuts, coordinate singularities, and non-commuting limits (e.g., $k \to 0$ vs. $w \to -1$).
4. **Conservation Laws & Bianchi Identities:** Verify $\nabla_\mu T^{\mu\nu} = 0$ and $\nabla_\mu G^{\mu\nu} = 0$ analytically and numerically along computed trajectories.
5. **Causality, Unitarity, & Stability:** Check no-ghost ($Q_s > 0, Q_T > 0$), gradient stability ($c_s^2 > 0, c_T^2 > 0$), and sub-luminality ($c_s^2 \le 1$) across the entire cosmic history $a \in [10^{-4}, 1]$.
6. **Numerical Artifact Detection:** Test whether claimed features change when halving step sizes, switching ODE integrators (e.g., explicit RK45 vs. implicit Radau/BDF), or increasing spectral collocation nodes $N$.
7. **Dimensional & Unit Audit:** Check every constant conversion ($c, G_N, H_0, M_{\mathrm{Pl}}, \text{Mpc}, \text{eV}$).
8. **Boundary & Initial Condition Sensitivity:** Perturb initial conditions in the radiation/matter era (see [`Ageeva:2026horndeski`](https://arxiv.org/abs/2610.36036) on initial-condition sensitivity in Horndeski cosmologies) to verify attractor stability.
9. **Counterexample Construction:** Construct explicit analytical counterexamples or synthetic mock datasets where the proposed method fails.
10. **Independent Code Reproduction:** Execute the reproduction suite cleanly from raw configuration files and check that all `pytest` tolerances hold.
11. **Rival Physical / Statistical Explanations:** Evaluate whether the signal can be explained by mundane systematics (e.g., SNe Ia zero-point offset $\Delta M_B$, BAO nonlinear reconstruction bias, or prior-volume projection).

---

## 2. Six-Dimension Epistemic Grading Rubric (ARA Level-2 Standard)

Each reviewed artifact is scored from `1` (Fatal Flaw) to `5` (Publication-Grade Rigor) on six dimensions:
1. **Evidence Relevance**
2. **Falsifiability**
3. **Scope Calibration** (no overclaiming beyond what the equations/data support)
4. **Argument & Mathematical Coherence**
5. **Exploration Integrity** (all negative/null runs preserved)
6. **Methodological & Numerical Rigor**

Any score $< 4$ on any dimension blocks progression to publication (`07-publications/`) and forces a revision cycle recorded in `RESEARCH_STATUS.md`.
