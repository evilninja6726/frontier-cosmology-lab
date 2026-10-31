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
│   └── pr01-horndeski-crossing-identifiability/
│       ├── proposal/
│       │   └── PROJECT_PROPOSAL.md          # Selected primary research project spec
│       └── baseline/
│           └── BENCHMARK_SPECIFICATION.md   # Milestone 7 reproducible baseline spec
│
├── 04-scientific-software/
│   └── documentation/
│       └── TOOLCHAIN_EVALUATION_AND_INTEGRATION.md  # Assessment of CLASS, CAMB, Cobaya, etc.
│
├── 06-independent-review/
│   └── peer-review/
│       └── ADVERSARIAL_REVIEW_PROTOCOL.md   # Independent adversarial reviewer workflow
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
└── infrastructure/
    ├── environment/
    │   └── HARDWARE_AND_ENV_AUDIT.md        # Apple M5 hardware & toolchain audit
    ├── scripts/
    │   └── verify_citations.py              # Zero-dependency arXiv/Crossref BibTeX verifier
    └── tests/
        └── test_citation_integrity.py       # Automated integrity checks for BibTeX entries
```

### Deferred Directories (Created Upon Milestone Execution)
- `03-projects/pr01-horndeski-crossing-identifiability/{theory,derivations,simulations,experiments,results,validation,paper,reproduction}/` — instantiated during Milestones 7–8.
- `05-experiments/{experiment-registry,configurations,execution-logs,artifacts}/` — instantiated when Milestone 7 benchmark runs execute.
- `07-publications/{manuscripts,figures,supplementary-material}/` — instantiated only after a result survives Stage 14 Adversarial Review.

---

## 2. Quick Verification Commands

Verify citation database integrity against live arXiv metadata:
```bash
/Users/rajat/personal/research/.venv/bin/python3 infrastructure/scripts/verify_citations.py
```

Run automated repository integrity checks:
```bash
/Users/rajat/personal/research/.venv/bin/python3 -m unittest discover -s infrastructure/tests
```
