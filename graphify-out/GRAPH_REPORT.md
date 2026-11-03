# Graph Report - frontier-cosmology-lab  (2026-11-03)

## Corpus Check
- 38 files · ~56,685 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 8 file(s) not represented in the graph (top: .dat 2, .ipynb 2, (none) 1)

## Summary
- 394 nodes · 722 edges · 26 communities (24 shown, 2 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 65 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f6f30f04`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- verify_citations.py
- Module 1: Relativistic FLRW Cosmology, Distance Measures, and the BAO Sound Horizon
- 2. Recommended Additional Skills (Vetted & Ready for Approval)
- 2. Detailed Profiles & Strongest Counterarguments
- RESEARCH STATUS — Project Frontier
- 2. Mathematical Architecture of the Investigation
- run_milestone8_experiments.py
- numpy
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
- fit_desi_dr2_plus_cmb
- Cloud Compute Architecture for Project Frontier (Zero Local Thermal Load)
- INDEPENDENT ADVERSARIAL PEER REVIEW REPORT — STAGE 14 (`PR-01`)

## God Nodes (most connected - your core abstractions)
1. `RadiationCalibration` - 22 edges
2. `e_of_z_jax()` - 18 edges
3. `main()` - 17 edges
4. `CosmologyParams` - 15 edges
5. `comoving_distance_dm_jax()` - 15 edges
6. `solve_horndeski_growth_trajectory()` - 15 edges
7. `build_hyperboloidal_system()` - 15 edges
8. `solve_subhorizon_growth()` - 14 edges
9. `DESIDR2BAOLikelihood` - 14 edges
10. `Planck2018CompressedCMBLikelihood` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Workflow 4: Independent Adversarial Peer Review (Stage 14)` --references--> `L1()`  [INFERRED]
  08-skills/research-workflows.md → 03-projects/pr02-ringdown-pseudospectra/derivations/hyperboloidal_qnm_symbolic.py
- `4. Current Confidence & Epistemic State` --references--> `L2()`  [INFERRED]
  RESEARCH_STATUS.md → 03-projects/pr02-ringdown-pseudospectra/derivations/hyperboloidal_qnm_symbolic.py
- `TestMilestone7Baseline` --uses--> `CosmologyParams`  [INFERRED]
  infrastructure/tests/test_milestone7_baseline.py → 03-projects/pr01-horndeski-crossing-identifiability/baseline/cosmology_baseline.py
- `rad_cal()` --uses--> `CosmologyParams`  [INFERRED]
  infrastructure/tests/test_milestone8_extensions.py → 03-projects/pr01-horndeski-crossing-identifiability/baseline/cosmology_baseline.py
- `test_horndeski_growth_solver_and_milestone8_report()` --uses--> `CosmologyParams`  [INFERRED]
  infrastructure/tests/test_milestone8_extensions.py → 03-projects/pr01-horndeski-crossing-identifiability/baseline/cosmology_baseline.py

## Import Cycles
- None detected.

## Communities (26 total, 2 thin omitted)

### Community 0 - "verify_citations.py"
Cohesion: 0.09
Nodes (6): execute_research_pipeline_remote(), main(), main(), parse_bib_entries(), verify_arxiv_ids(), TestCitationIntegrity

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
Cohesion: 0.09
Nodes (21): 1. Executive Summary of Original Findings, 2. Quantitative Summary Tables, 3. Epistemic Classification of Claims (`EPISTEMIC_STANDARDS.md`), 4. Methodological Limitations & Caveats (Stage 14 Adversarial Peer Review), Milestone 8 Research Synthesis — Identifiability, Prior-Volume Effects, and Physical Stability of Phantom-Divide Crossing in Horndeski / EFT Dark Energy (`PR-01`), Table 1 — `EXP-02`: Joint Fits to DESI DR2 BAO + Planck 2018 CMB + Union3 SNe Ia ($N = 38$), Table 2 — `EXP-01` & `EXP-03`: Horndeski Stability Prior Volume & Growth Slip, L1() (+13 more)

### Community 5 - "2. Mathematical Architecture of the Investigation"
Cohesion: 0.29
Nodes (6): 1. Executive Summary, 2. Mathematical Architecture of the Investigation, Phase A: Analytical Derivation & `SymPy` Verification, Phase B: Differentiable (`JAX`) Profile Likelihood vs. Bayesian MCMC Engine, Phase C: Observational Confrontation & SNe Calibration Split, PROJECT PROPOSAL (`PR-01`): Identifiability, Prior-Volume Effects, and Physical Stability of Phantom-Divide Crossing in Horndeski / EFT Dark Energy

### Community 6 - "run_milestone8_experiments.py"
Cohesion: 0.06
Nodes (17): DESIDR2BAOLikelihood, compute_camb_sigma8_0(), find_crossing_redshift(), generate_milestone8_figure(), main(), run_exp01_horndeski_stability_and_prior_volume(), run_exp02_basis_and_calibration_audit(), fit_model() (+9 more)

### Community 7 - "numpy"
Cohesion: 0.10
Nodes (20): generate_pr02_figure(), main(), run_bm02_poschl_teller_validation(), run_exp04_pseudospectrum_and_condition_numbers(), run_exp05_wavelength_and_flea_perturbation_audit(), run_exp06_timedomain_observability_vs_echo_delay(), build_hyperboloidal_system(), chebyshev_lobatto_grid_and_matrices() (+12 more)

### Community 8 - "RESEARCH DECISION LOG (ADRs) — Project Frontier"
Cohesion: 0.22
Nodes (8): `ADR-001`: Selection of Workspace Architecture & Non-Empty Directory Policy, `ADR-002`: Selective Vetting of Agent Skills vs. Full Collection Installation, `ADR-003`: Selection of Minimal Scientific Toolchain for Apple M5 (16 GB RAM), `ADR-004`: Selection of Primary (`PR-01`) and Backup (`PR-02`) Research Projects, `ADR-005`: Scale-Invariant Horndeski Stability Ratios & Exact Analytical $c_K$ Integration, `ADR-006`: Dual-Tier Cloud Execution Architecture (`Modal` + `GitHub Actions`), `ADR-007`: Stage 14 Adversarial Peer Review Resolution & Split Verdict on Hypothesis `H1.2`, RESEARCH DECISION LOG (ADRs) — Project Frontier

### Community 9 - "HARDWARE AND ENVIRONMENT AUDIT — Milestone 1"
Cohesion: 0.33
Nodes (5): 1. Host Hardware Capabilities, 2. Compilers, Package Managers, and Runtimes, 3. Missing Scientific Python Dependencies (To Be Installed for Milestone 7), HARDWARE AND ENVIRONMENT AUDIT — Milestone 1, Why This Stack Fits Our 16 GB Apple M5 Hardware

### Community 10 - "RESEARCH CHARTER — Project Frontier"
Cohesion: 0.33
Nodes (5): 1. Mission Statement, 2. Interdisciplinary Team Roles, 3. Epistemic Levels of Evidence, 4. Scientific Governance Rules, RESEARCH CHARTER — Project Frontier

### Community 11 - "Primary Project (`PR-01`): Horndeski Phantom-Crossing Identifiability & Prior-Volume Audit"
Cohesion: 0.25
Nodes (7): Hypothesis `H1.1` (Basis & Calibration Sensitivity of Phantom Crossing), Hypothesis `H1.2` (Horndeski Braiding Floor & Growth-Rate Slip), Hypothesis `H2.1` (Exponential Non-Normality & Wavelength-Dependent Spectral Instability), Hypothesis `H2.2` (Prompt Time-Domain Ringdown Immunity vs. Causal Echo Delay), PREREGISTERED HYPOTHESIS REGISTER — Project Frontier, Primary Project (`PR-01`): Horndeski Phantom-Crossing Identifiability & Prior-Volume Audit, Project 2 (`PR-02`): Environmental vs. Intrinsic Perturbations in Black Hole Ringdown Pseudospectra

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
Cohesion: 0.07
Nodes (22): calibrate_radiation_from_camb(), comoving_distance_dm_jax(), CosmologyParams, e_of_z_jax(), f_de_chebyshev_jax(), f_de_cpl_jax(), f_de_pade_jax(), _hu_sugiyama_z_drag_raw() (+14 more)

### Community 21 - "flrw_and_horndeski_symbolic.py"
Cohesion: 0.15
Nodes (6): verify_background_density_evolutions(), verify_eds_linear_growth_limit(), verify_horndeski_stability_and_crossing_theorem(), main(), verify_hyperboloidal_wave_operator(), test_pr02_symbolic_derivations()

### Community 22 - "MATHEMATICAL MODEL & RIGOROUS DERIVATIONS (`PR-01`) — Stage 9"
Cohesion: 0.20
Nodes (9): 10-Point Mathematical Audit, 10-Point Mathematical Audit, 10-Point Mathematical Audit, 10-Point Mathematical Audit, Equation 1: Covariant Dark Energy Continuity & General Basis Density Evolution, Equation 2: Relativistic Multi-Component Friedmann Equation & BAO Distance Observables, Equation 3: Horndeski / EFT of Dark Energy Quadratic Action, No-Ghost Condition, & Sound Speed, Equation 4: Sub-Horizon Linear Matter Growth & Braiding-Induced Gravitational Slip (+1 more)

### Community 23 - "fit_desi_dr2_plus_cmb"
Cohesion: 0.13
Nodes (5): fit_desi_dr2_plus_cmb(), _joint_chi2(), _obj_cpl(), _obj_lcdm(), TestMilestone7Baseline

### Community 26 - "Cloud Compute Architecture for Project Frontier (Zero Local Thermal Load)"
Cohesion: 0.29
Nodes (6): Cloud Compute Architecture for Project Frontier (Zero Local Thermal Load), One-Time Setup, Running Experiments in the Cloud (0% Local CPU), Tier 1: Serverless Interactive Cloud via Modal (`modal.com`) — Recommended Primary Workflow, Tier 2: Automated Cloud CI/CD via GitHub Actions (100% Free on GitHub), Usage

### Community 27 - "INDEPENDENT ADVERSARIAL PEER REVIEW REPORT — STAGE 14 (`PR-01`)"
Cohesion: 0.33
Nodes (5): 1. Executive Summary & Editorial Verdict, 2. Quantitative Grading on the 6-Dimension ARA Rubric (1 to 5 Scale), 3. Summary of the 11-Point Adversarial Falsification Audit, INDEPENDENT ADVERSARIAL PEER REVIEW REPORT — STAGE 14 (`PR-01`), Key Findings Raised by the Reviewer & Resolution Status

## Knowledge Gaps
- **1 isolated node(s):** `frontier-cosmology-lab`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 216 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Union3SNLikelihood` connect `run_milestone8_experiments.py` to `RESEARCH STATUS — Project Frontier`, `cosmology_baseline.py`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `RadiationCalibration` (e.g. with `compute_eft_prior_volume_and_growth_slip()` and `solve_horndeski_growth_trajectory()`) actually correct?**
  _`RadiationCalibration` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `frontier-cosmology-lab` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `verify_citations.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0907258064516129 - nodes in this community are weakly interconnected._
- **Why does `1. Executive Summary of Original Findings` connect `RESEARCH STATUS — Project Frontier` to `run_milestone8_experiments.py`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `e_of_z_jax()` (e.g. with `.test_t7_1_hubble_expansion_vs_camb()` and `test_horndeski_kineticity_cancellation_theorem()`) actually correct?**
  _`e_of_z_jax()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `Module 1: Relativistic FLRW Cosmology, Distance Measures, and the BAO Sound Horizon` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._