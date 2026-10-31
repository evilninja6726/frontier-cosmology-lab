# RESEARCH WORKFLOWS & SKILL ORCHESTRATION — Project Frontier

This document defines how Antigravity skills, subagents, and local verification scripts orchestrate each stage of scientific discovery in `frontier-cosmology-lab`.

---

## Workflow 1: Systematic Literature & Gap Discovery
1. **Query Formulation:** Query arXiv (`astro-ph.CO`, `gr-qc`, `hep-th`), INSPIRE-HEP, and Crossref using targeted boolean operators.
2. **Citation Verification Gate:** Every candidate paper is added to `01-literature/citation-database/references.bib` and immediately audited via:
   ```bash
   /Users/rajat/personal/research/.venv/bin/python3 infrastructure/scripts/verify_citations.py
   ```
3. **Knowledge Graph Indexing (`graphify`):** When new papers (PDF/Markdown) or code modules are added to `frontier-cosmology-lab`, run `graphify update .` so cross-document relationships (equations, datasets, approximations) are queryable via `graphify query`.

---

## Workflow 2: Mathematical Derivation & Symbolic Verification
1. **Analytical Derivation:** Formulate the action/metric/perturbation equations in `03-projects/<project>/derivations/` with full dimensional accounting and explicit assumptions.
2. **Independent Symbolic Check (`sympy` skill):** Write an accompanying `SymPy` verification script that derives the equations from first principles (or verifies identities such as Bianchi identities $\nabla_\mu G^{\mu\nu} = 0$, conservation $\nabla_\mu T^{\mu\nu} = 0$, and asymptotic limits $w_a \to 0, \alpha_i \to 0$) and asserts exact symbolic equality (`simplify(lhs - rhs) == 0`).
3. **Adversarial Assumption Stress-Test (`grilling` skill):** Challenge every simplifying assumption (e.g., quasi-static approximation $k \gg aH$, constant $\alpha_i/\Omega_{\mathrm{DE}}$ scaling, spatial flatness $\Omega_k = 0$) before writing numerical solvers.

---

## Workflow 3: Baseline Reproduction Gate (Milestone 7)
1. **Reference Specification:** Document the published paper, exact equations, cosmological parameters, and numerical tolerances in `03-projects/<project>/baseline/BENCHMARK_SPECIFICATION.md`.
2. **Dual-Implementation Cross-Check:** Compute baseline observables using both an established community code (`CAMB` or `CLASS`) and an independent local implementation (`SciPy` / `JAX`).
3. **Automated `pytest` Gate:** No extension code in `simulations/` or `experiments/` may be executed until `pytest` confirms baseline agreement within specified tolerances ($< 10^{-4}$).

---

## Workflow 4: Independent Adversarial Peer Review (Stage 14)
1. **Isolated Subagent Invocation:** Launch a dedicated `self` or `research` subagent configured with `06-independent-review/peer-review/ADVERSARIAL_REVIEW_PROTOCOL.md` (and `peer-review` / `scientific-critical-thinking` skills).
2. **Falsification Mandate:** The reviewer is tasked solely with finding mathematical flaws, numerical artifacts, prior-volume biases, or overlooked prior literature that invalidates novelty claims.
3. **Epistemic Classification:** Assign each result an explicit evidence tier (`L1` through `L6` in `RESEARCH_CHARTER.md`) and update `RESEARCH_STATUS.md`.
