# Graph Report - frontier-cosmology-lab  (2026-10-31)

## Corpus Check
- 26 files · ~26,740 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 1, .bib 1)

## Summary
- 246 nodes · 364 edges · 24 communities (22 shown, 2 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `093cdf99`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- verify_citations.py
- Module 1: Relativistic FLRW Cosmology, Distance Measures, and the BAO Sound Horizon
- 2. Recommended Additional Skills (Vetted & Ready for Approval)
- 2. Detailed Profiles & Strongest Counterarguments
- RESEARCH STATUS — Project Frontier
- 2. Mathematical Architecture of the Investigation
- run_milestone7_benchmark.py
- RESEARCH WORKFLOWS & SKILL ORCHESTRATION — Project Frontier
- RESEARCH DECISION LOG (ADRs) — Project Frontier
- HARDWARE AND ENVIRONMENT AUDIT — Milestone 1
- RESEARCH CHARTER — Project Frontier
- Primary Project (`PR-01`): Horndeski Phantom-Crossing Identifiability & Prior-Volume Audit
- SCIENTIFIC SOFTWARE EVALUATION & INTEGRATION STRATEGY — Milestone 3
- PROJECT FRONTIER — Autonomous AI-Assisted Cosmology & Fundamental Physics Research Laboratory
- EPISTEMIC STANDARDS & VALIDATION GATES — Project Frontier
- ADVERSARIAL PEER REVIEW PROTOCOL — Stage 14
- EVALUATED SKILLS AUDIT — Milestone 2
- SCIENTIFIC GLOSSARY & CONVENTIONS — Project Frontier
- SPECIFIC RESEARCH GAPS & CONTRIBUTION CLASSIFICATION — Milestone 4 & 5
- frontier-cosmology-lab
- cosmology_baseline.py
- flrw_and_horndeski_symbolic.py
- MATHEMATICAL MODEL & RIGOROUS DERIVATIONS (`PR-01`) — Stage 9
- solve_subhorizon_growth

## God Nodes (most connected - your core abstractions)
1. `main()` - 17 edges
2. `RadiationCalibration` - 16 edges
3. `e_of_z_jax()` - 15 edges
4. `fit_desi_dr2_plus_cmb()` - 13 edges
5. `comoving_distance_dm_jax()` - 12 edges
6. `solve_subhorizon_growth()` - 12 edges
7. `TestMilestone7Baseline` - 12 edges
8. `sound_horizon_rd_jax()` - 11 edges
9. `DESIDR2BAOLikelihood` - 10 edges
10. `calibrate_radiation_from_camb()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `TestMilestone7Baseline` --uses--> `CosmologyParams`  [INFERRED]
  infrastructure/tests/test_milestone7_baseline.py → 03-projects/pr01-horndeski-crossing-identifiability/baseline/cosmology_baseline.py
- `TestMilestone7Baseline` --uses--> `DESIDR2BAOLikelihood`  [INFERRED]
  infrastructure/tests/test_milestone7_baseline.py → 03-projects/pr01-horndeski-crossing-identifiability/baseline/cosmology_baseline.py
- `TestMilestone7Baseline` --uses--> `Planck2018CompressedCMBLikelihood`  [INFERRED]
  infrastructure/tests/test_milestone7_baseline.py → 03-projects/pr01-horndeski-crossing-identifiability/baseline/cosmology_baseline.py
- `main()` --calls--> `e_of_z_jax()`  [EXTRACTED]
  03-projects/pr01-horndeski-crossing-identifiability/reproduction/run_milestone7_benchmark.py → 03-projects/pr01-horndeski-crossing-identifiability/baseline/cosmology_baseline.py
- `main()` --calls--> `comoving_distance_dm_jax()`  [EXTRACTED]
  03-projects/pr01-horndeski-crossing-identifiability/reproduction/run_milestone7_benchmark.py → 03-projects/pr01-horndeski-crossing-identifiability/baseline/cosmology_baseline.py

## Import Cycles
- None detected.

## Communities (24 total, 2 thin omitted)

### Community 0 - "verify_citations.py"
Cohesion: 0.13
Nodes (4): main(), parse_bib_entries(), verify_arxiv_ids(), TestCitationIntegrity

### Community 1 - "Module 1: Relativistic FLRW Cosmology, Distance Measures, and the BAO Sound Horizon"
Cohesion: 0.10
Nodes (19): 1.1 Intuition (Software / Systems Analogy), 1.2 Mathematical Formulation, 1.3 Physical Significance & Observable Distances, 1.4 Symbolic & Numerical Example (Python / SymPy), 1.5 How It Appears in `PR-01` & Exercises, 2.1 Intuition, 2.2 Mathematical Formulation, 2.3 Physical Significance & Role in `PR-01` (+11 more)

### Community 2 - "2. Recommended Additional Skills (Vetted & Ready for Approval)"
Cohesion: 0.20
Nodes (9): 1. Currently Installed Skills (Verified Active), 2. Recommended Additional Skills (Vetted & Ready for Approval), 3. Single Approved Batch Command (Awaiting User Confirmation), A. Workspace Skills (`/Users/rajat/personal/research/.agents/skills/`), B. Built-in & Global Antigravity Skills (`~/.gemini/antigravity/`), INSTALLED & RECOMMENDED AGENT SKILLS — Project Frontier, Tier 1: Official Antigravity Marketplace Plugin, Tier 2: High-Adoption Open-Source Scientific Skills (`k-dense-ai/scientific-agent-skills` — 55.5K GitHub Stars, MIT License) (+1 more)

### Community 3 - "2. Detailed Profiles & Strongest Counterarguments"
Cohesion: 0.22
Nodes (8): 1. Quantitative Comparison Matrix (Scored 1–10 per Criterion), 2. Detailed Profiles & Strongest Counterarguments, FIVE RANKED CANDIDATE RESEARCH PROJECTS — Milestone 5, Rank #1 (PRIMARY): `PR-01` — Identifiability, Prior-Volume Effects, and Physical Stability of Phantom-Divide Crossing in Horndeski / EFT Dark Energy under DESI DR2, CMB, and SNe Ia, Rank #2 (BACKUP): `PR-02` — Environmental vs. Intrinsic Perturbations in Black Hole Ringdown: Quantitative Pseudospectrum Bounds and Time-Domain Observability, Rank #3: `PR-04` — Truncated Hierarchy Reflections vs. Tight-Coupling Switching Artifacts in Differentiable (`JAX`) Einstein-Boltzmann Solvers, Rank #4: `PR-03` — Joint Tomographic Cosmic Birefringence ($EB$) and Background Expansion Constraints on Reionization-Era Axion-Like Particles (ACT DR6 + DESI DR2), Rank #5: `PR-05` — Non-Instantaneous Reheating and Neutrino Spectral Distortions in Supercooled First-Order Phase Transitions Confronting NANOGrav 15-yr and BBN

### Community 4 - "RESEARCH STATUS — Project Frontier"
Cohesion: 0.25
Nodes (7): 1. Milestone Progress Tracker, 2. Current Research Question (PR-01), 3. Verified Core Bibliography (Sampled from `references.bib`), 4. Current Confidence & Epistemic State, 5. Immediate Next Actions (Milestone 8 — Original Research Extensions), RESEARCH STATUS — Project Frontier, Sub-Questions

### Community 5 - "2. Mathematical Architecture of the Investigation"
Cohesion: 0.29
Nodes (6): 1. Executive Summary, 2. Mathematical Architecture of the Investigation, Phase A: Analytical Derivation & `SymPy` Verification, Phase B: Differentiable (`JAX`) Profile Likelihood vs. Bayesian MCMC Engine, Phase C: Observational Confrontation & SNe Calibration Split, PROJECT PROPOSAL (`PR-01`): Identifiability, Prior-Volume Effects, and Physical Stability of Phantom-Divide Crossing in Horndeski / EFT Dark Energy

### Community 6 - "run_milestone7_benchmark.py"
Cohesion: 0.08
Nodes (13): calibrate_radiation_from_camb(), CosmologyParams, DESIDR2BAOLikelihood, fit_desi_dr2_plus_cmb(), _joint_chi2(), _obj_cpl(), _obj_lcdm(), hubble_distance_dh_jax() (+5 more)

### Community 7 - "RESEARCH WORKFLOWS & SKILL ORCHESTRATION — Project Frontier"
Cohesion: 0.33
Nodes (5): RESEARCH WORKFLOWS & SKILL ORCHESTRATION — Project Frontier, Workflow 1: Systematic Literature & Gap Discovery, Workflow 2: Mathematical Derivation & Symbolic Verification, Workflow 3: Baseline Reproduction Gate (Milestone 7), Workflow 4: Independent Adversarial Peer Review (Stage 14)

### Community 8 - "RESEARCH DECISION LOG (ADRs) — Project Frontier"
Cohesion: 0.33
Nodes (5): `ADR-001`: Selection of Workspace Architecture & Non-Empty Directory Policy, `ADR-002`: Selective Vetting of Agent Skills vs. Full Collection Installation, `ADR-003`: Selection of Minimal Scientific Toolchain for Apple M5 (16 GB RAM), `ADR-004`: Selection of Primary (`PR-01`) and Backup (`PR-02`) Research Projects, RESEARCH DECISION LOG (ADRs) — Project Frontier

### Community 9 - "HARDWARE AND ENVIRONMENT AUDIT — Milestone 1"
Cohesion: 0.33
Nodes (5): 1. Host Hardware Capabilities, 2. Compilers, Package Managers, and Runtimes, 3. Missing Scientific Python Dependencies (To Be Installed for Milestone 7), HARDWARE AND ENVIRONMENT AUDIT — Milestone 1, Why This Stack Fits Our 16 GB Apple M5 Hardware

### Community 10 - "RESEARCH CHARTER — Project Frontier"
Cohesion: 0.33
Nodes (5): 1. Mission Statement, 2. Interdisciplinary Team Roles, 3. Epistemic Levels of Evidence, 4. Scientific Governance Rules, RESEARCH CHARTER — Project Frontier

### Community 11 - "Primary Project (`PR-01`): Horndeski Phantom-Crossing Identifiability & Prior-Volume Audit"
Cohesion: 0.40
Nodes (4): Hypothesis `H1.1` (Basis & Calibration Sensitivity of Phantom Crossing), Hypothesis `H1.2` (Horndeski Braiding Floor & Growth-Rate Slip), PREREGISTERED HYPOTHESIS REGISTER — Project Frontier, Primary Project (`PR-01`): Horndeski Phantom-Crossing Identifiability & Prior-Volume Audit

### Community 12 - "SCIENTIFIC SOFTWARE EVALUATION & INTEGRATION STRATEGY — Milestone 3"
Cohesion: 0.40
Nodes (4): 1. Comprehensive Evaluation of Candidate Repositories, 2. Recommended Minimal Toolchain for First Research Project (PR-01), Additional Modern Differentiable Packages Evaluated, SCIENTIFIC SOFTWARE EVALUATION & INTEGRATION STRATEGY — Milestone 3

### Community 13 - "PROJECT FRONTIER — Autonomous AI-Assisted Cosmology & Fundamental Physics Research Laboratory"
Cohesion: 0.40
Nodes (4): 1. Laboratory Architecture, 2. Quick Verification Commands, Deferred Directories (Created Upon Milestone Execution), PROJECT FRONTIER — Autonomous AI-Assisted Cosmology & Fundamental Physics Research Laboratory

### Community 14 - "EPISTEMIC STANDARDS & VALIDATION GATES — Project Frontier"
Cohesion: 0.50
Nodes (3): 1. Mandatory Pre-Implementation Mathematical Checklist, 2. Numerical Convergence & Testing Standards, EPISTEMIC STANDARDS & VALIDATION GATES — Project Frontier

### Community 15 - "ADVERSARIAL PEER REVIEW PROTOCOL — Stage 14"
Cohesion: 0.50
Nodes (3): 1. The 11-Point Adversarial Falsification Checklist, 2. Six-Dimension Epistemic Grading Rubric (ARA Level-2 Standard), ADVERSARIAL PEER REVIEW PROTOCOL — Stage 14

### Community 16 - "EVALUATED SKILLS AUDIT — Milestone 2"
Cohesion: 0.50
Nodes (3): 1. Summary of 11 Required `npx skills find` Searches, 2. Deep-Dive Audit: `KalarisLabs/research-agent-skills`, EVALUATED SKILLS AUDIT — Milestone 2

### Community 17 - "SCIENTIFIC GLOSSARY & CONVENTIONS — Project Frontier"
Cohesion: 0.50
Nodes (3): Core Cosmological & Gravitational Terms, Metric, Units, and Fundamental Conventions, SCIENTIFIC GLOSSARY & CONVENTIONS — Project Frontier

### Community 20 - "cosmology_baseline.py"
Cohesion: 0.11
Nodes (11): comoving_distance_dm_jax(), e_of_z_jax(), f_de_chebyshev_jax(), f_de_cpl_jax(), f_de_pade_jax(), _hu_sugiyama_z_drag_raw(), _hu_sugiyama_z_star_raw(), _sound_horizon_at_z_jax() (+3 more)

### Community 21 - "flrw_and_horndeski_symbolic.py"
Cohesion: 0.26
Nodes (3): verify_background_density_evolutions(), verify_eds_linear_growth_limit(), verify_horndeski_stability_and_crossing_theorem()

### Community 22 - "MATHEMATICAL MODEL & RIGOROUS DERIVATIONS (`PR-01`) — Stage 9"
Cohesion: 0.20
Nodes (9): 10-Point Mathematical Audit, 10-Point Mathematical Audit, 10-Point Mathematical Audit, 10-Point Mathematical Audit, Equation 1: Covariant Dark Energy Continuity & General Basis Density Evolution, Equation 2: Relativistic Multi-Component Friedmann Equation & BAO Distance Observables, Equation 3: Horndeski / EFT of Dark Energy Quadratic Action, No-Ghost Condition, & Sound Speed, Equation 4: Sub-Horizon Linear Matter Growth & Braiding-Induced Gravitational Slip (+1 more)

### Community 23 - "solve_subhorizon_growth"
Cohesion: 0.36
Nodes (4): solve_subhorizon_growth(), _dlnH_dlna(), _e_of_a(), _rhs()

## Knowledge Gaps
- **1 isolated node(s):** `frontier-cosmology-lab`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 142 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TestMilestone7Baseline` connect `run_milestone7_benchmark.py` to `verify_citations.py`, `flrw_and_horndeski_symbolic.py`, `solve_subhorizon_growth`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `fit_desi_dr2_plus_cmb()` (e.g. with `_joint_chi2()` and `_obj_cpl()`) actually correct?**
  _`fit_desi_dr2_plus_cmb()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `frontier-cosmology-lab` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `verify_citations.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13043478260869565 - nodes in this community are weakly interconnected._
- **Why does `solve_subhorizon_growth()` connect `solve_subhorizon_growth` to `verify_citations.py`, `cosmology_baseline.py`, `run_milestone7_benchmark.py`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Should `Module 1: Relativistic FLRW Cosmology, Distance Measures, and the BAO Sound Horizon` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._
- **Why does `fit_desi_dr2_plus_cmb()` connect `run_milestone7_benchmark.py` to `verify_citations.py`, `cosmology_baseline.py`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._