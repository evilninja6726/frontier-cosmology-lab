# INSTALLED & RECOMMENDED AGENT SKILLS — Project Frontier

**Last Audited:** 2026-10-31  
**Policy:** Never reinstall existing skills. Never blindly install massive unvetted collections. Verify maintainer reputation, install count, license, and actual `SKILL.md` contents before installation.

---

## 1. Currently Installed Skills (Verified Active)

### A. Workspace Skills (`/Users/rajat/personal/research/.agents/skills/`)
| Skill Name | Source Repository | Status | Role in Project Frontier |
| :--- | :--- | :--- | :--- |
| **`find-skills`** | `vercel-labs/skills` | **Installed & Active** | Primary discovery mechanism via `npx skills find` across the open agent skills registry (`skills.sh`). |
| **`graphify`** | `Graphify-Labs/graphify` | **Installed & Active** | Builds persistent AST + semantic knowledge graphs (`graphify-out/graph.json`, `GRAPH_REPORT.md`) across code, papers, and derivations. |
| **`grilling` / `grill-me`** | `mattpocock/skills` | **Installed & Active** | Relentless adversarial stress-testing of research plans, mathematical assumptions, and experimental designs. |

### B. Built-in & Global Antigravity Skills (`~/.gemini/antigravity/`)
| Skill Name | Scope | Role in Project Frontier |
| :--- | :--- | :--- |
| **`generative_ui`** | Built-in | Interactive visual walkthroughs of physical concepts, phase portraits, and parameter space widgets. |
| **`automation`** | Built-in | Scheduling recurring background checks or long-running numerical sweeps. |
| **`security-review`** | Global | Auditing third-party scientific scripts and skill bundles before execution. |

---

## 2. Recommended Additional Skills (Vetted & Ready for Approval)

Following our systematic audit of 11 `npx skills find` queries and direct inspection of upstream `SKILL.md` files (detailed in [`evaluated-skills.md`](file:///Users/rajat/personal/research/frontier-cosmology-lab/08-skills/evaluated-skills.md)), we recommend installing **only 7 targeted, high-signal skills** rather than the full 281-skill `KalarisLabs` bundle (which contains ~240 unrelated biology, clinical, finance, and marketing skills).

### Tier 1: Official Antigravity Marketplace Plugin
- **Science Plugin (`antigravity-plugins-official/plugin/science`)**
  - **Source:** Official Google Antigravity Plugin Catalog
  - **Purpose:** Curated collection of official agent skills for scientific computing and research workflows.
  - **Installation:** Can be enabled directly via the Antigravity UI plugin card in chat.

### Tier 2: High-Adoption Open-Source Scientific Skills (`k-dense-ai/scientific-agent-skills` — 55.5K GitHub Stars, MIT License)
We inspected the raw `SKILL.md` files in `k-dense-ai/scientific-agent-skills` (the upstream source from which `KalarisLabs` mirrors most of its scientific skills). They are pure markdown instruction and Python pattern guides with **no privileged credential requirements**:

| Skill | Source & Installs | Exact Installation Command | Purpose in Frontier Lab | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **`literature-review`** | `k-dense-ai/scientific-agent-skills` (2.4K installs, MIT) | `npx skills add k-dense-ai/scientific-agent-skills@literature-review` | Structured PRISMA-style systematic literature synthesis, screening, and citation extraction across arXiv, Crossref, and ADS. | Default templates include biomedical (PubMed) examples alongside arXiv; we pair with direct arXiv API queries for `astro-ph.CO` / `gr-qc`. |
| **`hypothesis-generation`** | `k-dense-ai/scientific-agent-skills` (2.2K installs, MIT) | `npx skills add k-dense-ai/scientific-agent-skills@hypothesis-generation` | Formulating falsifiable hypotheses, competing physical mechanisms, and discriminating observational predictions. | Requires domain-specific physics constraints (e.g., EFT no-ghost bounds) supplied by our theoretical framework. |
| **`experimental-design`** | `k-dense-ai/scientific-agent-skills` (1.4K installs, MIT) | `npx skills add k-dense-ai/scientific-agent-skills@experimental-design` | Factorial design, parameter-space Latin hypercube sampling, control benchmarks, and sensitivity ablation design. | Geared toward general DOE; must be adapted for deterministic PDE/ODE convergence and Bayesian MCMC diagnostics. |
| **`scientific-critical-thinking`** | `k-dense-ai/scientific-agent-skills` (2.4K installs, MIT) | `npx skills add k-dense-ai/scientific-agent-skills@scientific-critical-thinking` | Identifying confounding variables, selection biases, prior-volume artifacts, and over-interpretation of $p$-values/$\Delta\chi^2$. | Qualitative rubric; must be backed by quantitative Fisher/Bayesian calculations. |
| **`peer-review`** | `k-dense-ai/scientific-agent-skills` (2.3K installs, MIT) | `npx skills add k-dense-ai/scientific-agent-skills@peer-review` | Structured stage-by-stage manuscript and derivation critique for our Stage 14 Adversarial Reviewer. | Does not automatically re-run code; must be paired with our automated `pytest` reproduction suite. |
| **`sympy`** | `k-dense-ai/scientific-agent-skills` (2.1K installs, MIT) | `npx skills add k-dense-ai/scientific-agent-skills@sympy` | Best practices for exact symbolic tensor/calculus derivations, asymptotic series, and `lambdify` code generation. | General SymPy patterns; cosmological perturbation equations still require explicit gauge definitions. |
| **`astropy`** | `k-dense-ai/scientific-agent-skills` (MIT) | `npx skills add k-dense-ai/scientific-agent-skills@astropy` | Cosmological distance calculations (`astropy.cosmology`), FITS table I/O, and dimensional `units`/`Quantity` enforcement. | `astropy.cosmology` handles background FLRW integrals, not linear Boltzmann perturbations (which require `CAMB`/`CLASS`). |

### Tier 3: Specialized Citation & Epistemic Verification (`KalarisLabs/research-agent-skills`)
While we **reject** installing the full 281-skill `KalarisLabs/research-agent-skills` repository blindly, two specific skills in `KalarisLabs` inspected via GitHub API provide genuine, zero-dependency value:

| Skill | Source & Installs | Exact Installation Command | Purpose in Frontier Lab | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **`citation-verification`** | `KalarisLabs/research-agent-skills` (178 installs, MIT skill license) | `npx skills add KalarisLabs/research-agent-skills@citation-verification` | Zero-dependency Python 3.9+ standard-library workflow querying Crossref, OpenAlex, and arXiv to catch hallucinated or mismatched DOIs/arXiv IDs. | Rate-limited by public APIs (~1–3 req/sec); we also implemented a local verifier in `infrastructure/scripts/verify_citations.py`. |
| **`ara-rigor-reviewer`** | `KalarisLabs/research-agent-skills` (MIT skill license) | `npx skills add KalarisLabs/research-agent-skills@ara-rigor-reviewer` | 6-dimension epistemic rubric (evidence relevance, falsifiability, scope calibration, argument coherence, exploration integrity, methodological rigor). | Expects specific markdown paths unless adapted to our `06-independent-review/` structure. |

---

## 3. Single Approved Batch Command (Awaiting User Confirmation)
Once approved by you, we can install the exact vetted subset into the workspace using:
```bash
npx skills add k-dense-ai/scientific-agent-skills@literature-review \
               k-dense-ai/scientific-agent-skills@hypothesis-generation \
               k-dense-ai/scientific-agent-skills@experimental-design \
               k-dense-ai/scientific-agent-skills@scientific-critical-thinking \
               k-dense-ai/scientific-agent-skills@peer-review \
               k-dense-ai/scientific-agent-skills@sympy \
               k-dense-ai/scientific-agent-skills@astropy \
               KalarisLabs/research-agent-skills@citation-verification
```
