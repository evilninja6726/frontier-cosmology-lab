# PROJECT FRONTIER — Autonomous AI-Assisted Cosmology & Fundamental Physics Research Laboratory

**Mission:** Investigate unanswered questions in cosmology, gravitation, astrophysics, and fundamental physics through systematic literature synthesis, rigorous mathematical derivation, reproducible numerical modelling, and adversarial independent falsification.

> **Core Principle:** *Think ambitiously. Derive rigorously. Simulate carefully. Challenge relentlessly. Publish honestly.*

---

## 1. Laboratory Architecture

This workspace (`frontier-cosmology-lab/`) is designed as a multi-year, modular, version-controlled scientific research environment. In accordance with our engineering discipline, **no empty directories are created merely for appearance**; directories are instantiated only when populated with substantive, version-controlled artifacts.

```text
frontier-cosmology-lab/
├── README.md                          # Workspace architecture & operational guide
├── RESEARCH_CHARTER.md                # Scientific governance, roles, & epistemic standards
├── RESEARCH_STATUS.md                 # Persistent cross-session state & milestone tracker
├── ROADMAP.md                         # Multi-stage execution plan & Month-1 schedule
├── pyproject.toml                     # Reproducible Python 3.12+ scientific environment spec
├── .gitignore                         # Version control exclusions for large data/chains
│
├── 00-foundation/
│   ├── learning-roadmap/
│   │   └── PROJECT_01_LEARNING_ROADMAP.md   # Targeted physics/math curriculum for PR-01
│   └── scientific-methodology/
│       └── EPISTEMIC_STANDARDS.md           # 6-tier evidence hierarchy & validation gates
│
├── 01-literature/
│   ├── literature-reviews/
│   │   └── 2026_COSMOLOGY_AND_GRAVITATION_FRONTIERS.md  # Systematic frontier survey
│   ├── research-gaps/
│   │   └── GAP_ANALYSIS_MATRIX.md           # Specific mathematical & numerical gaps
│   └── citation-database/
│       └── references.bib                   # Live-verified BibTeX database (arXiv/DOI)
│
├── 02-research-discovery/
│   ├── candidate-topics/
│   │   └── FIVE_RANKED_CANDIDATES.md        # 5 scored candidate projects & counterarguments
│   ├── novelty-assessments/
│   │   └── PRIMARY_AND_BACKUP_NOVELTY_AUDIT.md  # Prior-art audit for PR-01 & PR-02
│   └── hypothesis-register/
│       └── HYPOTHESIS_REGISTER.md           # Preregistered falsifiable hypotheses
│
├── 03-projects/
│   ├── pr01-horndeski-crossing-identifiability/
│   │   ├── proposal/PROJECT_PROPOSAL.md
│   │   ├── theory/THEORY_AND_DERIVATIONS.md
│   │   ├── derivations/flrw_and_horndeski_symbolic.py
│   │   ├── baseline/{BENCHMARK_SPECIFICATION.md, cosmology_baseline.py}
│   │   ├── simulations/horndeski_eft_solver.py
│   │   ├── experiments/run_milestone8_experiments.py
│   │   └── results/{MILESTONE8_RESEARCH_SYNTHESIS.md, *.png}
│   └── pr02-ringdown-pseudospectra/
│       ├── theory/THEORY_AND_DERIVATIONS.md
│       ├── derivations/hyperboloidal_qnm_symbolic.py
│       ├── simulations/chebyshev_pseudospectrum_solver.py
│       ├── experiments/run_pr02_experiments.py
│       └── results/{PR02_RESEARCH_SYNTHESIS.md, *.png}
│
├── 04-scientific-software/
│   └── documentation/
│       └── TOOLCHAIN_EVALUATION_AND_INTEGRATION.md  # Assessment of CLASS, CAMB, Cobaya, etc.
│
├── 05-experiments/
│   └── experiment-registry/
│       ├── BM_01_milestone7_report.json
│       ├── EXP_01_to_03_milestone8_report.json
│       └── PR02_BM02_and_EXP04_to_06_report.json
│
├── 06-independent-review/
│   └── peer-review/
│       ├── ADVERSARIAL_REVIEW_PROTOCOL.md   # Independent adversarial reviewer workflow
│       ├── PR01_ADVERSARIAL_REVIEW_REPORT.md # Stage 14 Review of PR-01 (ARA 4.80/5.00)
│       └── PR02_ADVERSARIAL_REVIEW_REPORT.md # Stage 14 Review of PR-02 (ARA 4.68/5.00)
│
├── 07-publications/
│   ├── figures/                             # Publication-grade multi-panel figures
│   └── manuscripts/
│       ├── PR01_Horndeski_Phantom_Crossing_Identifiability.tex
│       └── PR02_Black_Hole_Ringdown_Pseudospectra.tex
│
├── 08-skills/
│   ├── installed-skills.md                  # Active & recommended vetted skills + commands
│   ├── evaluated-skills.md                  # Audit of skills.sh & KalarisLabs vs k-dense-ai
│   └── research-workflows.md                # Skill orchestration across research phases
│
├── 09-knowledge-base/
│   ├── scientific-glossary.md               # Formal definitions, units, & conventions
│   └── research-decisions.md                # Architectural & Research Decision Records (ADRs)
│
├── notebooks/
│   ├── PR01_Cloud_Runner.ipynb              # 1-Click Google Colab Runner for PR-01
│   └── PR02_Ringdown_Cloud_Runner.ipynb     # 1-Click Google Colab Runner for PR-02
│
└── infrastructure/
    ├── cloud/
    │   ├── CLOUD_COMPUTE_GUIDE.md           # Zero-local-heat cloud execution guide
    │   ├── modal_runner.py                  # Serverless Modal runner
    │   └── cloud_research_pipeline.yml      # GitHub Actions CI/CD workflow
    ├── environment/
    │   └── HARDWARE_AND_ENV_AUDIT.md        # Apple M5 hardware & toolchain audit
    ├── scripts/
    │   └── verify_citations.py              # Zero-dependency arXiv/Crossref BibTeX verifier
    └── tests/
        ├── test_citation_integrity.py       # Automated integrity checks for BibTeX entries
        ├── test_milestone7_baseline.py      # Milestone 7 validation gate tests (PR-01)
        ├── test_milestone8_extensions.py    # Milestone 8 Horndeski EFT tests (PR-01)
        └── test_pr02_ringdown.py            # Hyperboloidal ringdown & pseudospectrum tests (PR-02)
```

---

## 2. 1-Click Cloud Execution (Zero Local CPU Heat)

All numerical simulations, symbolic proofs, and `pytest` suites are designed to run in the cloud on Google Colab with **0% local CPU load**:

- **Project `PR-01` (Horndeski Phantom-Crossing Identifiability):**  
  [Open `PR01_Cloud_Runner.ipynb` in Google Colab](https://colab.research.google.com/github/evilninja6726/frontier-cosmology-lab/blob/main/notebooks/PR01_Cloud_Runner.ipynb) (`13/13` tests passed in `39.30s`; suite runtime `84.07s`)
- **Project `PR-02` (Black Hole Ringdown Pseudospectra & Time-Domain Observability):**  
  [Open `PR02_Ringdown_Cloud_Runner.ipynb` in Google Colab](https://colab.research.google.com/github/evilninja6726/frontier-cosmology-lab/blob/main/notebooks/PR02_Ringdown_Cloud_Runner.ipynb) (`4/4` tests passed in `2.49s`; suite runtime `16.01s`)
