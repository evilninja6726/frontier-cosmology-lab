# EVALUATED SKILLS AUDIT — Milestone 2

**Audit Date:** 2026-10-31  
**Discovery Tool Used:** `npx skills find <query>` + GitHub REST API raw inspection of `SKILL.md` and repository metadata.

---

## 1. Summary of 11 Required `npx skills find` Searches

We executed all 11 required searches via `npx skills find` and audited the returned candidates against our 8 evaluation criteria (source reputation, install count, GitHub activity, actual `SKILL.md` functionality, Antigravity compatibility, external dependencies/credentials, security, and duplication):

| Search Query | Top Candidates Surfaced (`skills.sh`) | Audit Verdict & Rationale |
| :--- | :--- | :--- |
| **`literature review`** | `mattpocock/skills@research` (745.5K), `firecrawl/firecrawl-workflows@firecrawl-research-papers` (33.9K), `affaan-m/ecc@literature-review` (6.4K), `k-dense-ai/scientific-agent-skills@literature-review` (2.4K) | **Select `k-dense-ai/scientific-agent-skills@literature-review`.** `firecrawl` requires a paid Firecrawl API key; `mattpocock/skills@research` is general software/codebase research; `k-dense-ai` is tailored to scientific literature synthesis and citation verification. |
| **`scientific computing`** | `k-dense-ai/scientific-agent-skills@{networkx,shap,dask,zarr-python,cirq,experimental-design}` (1.4K–2.1K), `mindrally/skills@scipy-best-practices` (1K) | **Select `k-dense-ai@experimental-design` & `astropy`.** Reject low-install (<200 installs) wrappers (`tondevrel`, `vamseeachanta`) that merely repeat standard NumPy docstrings. |
| **`physics simulation`** | `enzed/r3f-skills` (3.3K), `freshtechbro/claudedesignskills@babylonjs-engine` (2.6K), `nvidia/skills@omniverse-cad-to-simready` (2.4K), `omer-metin/skills-for-antigravity@physics-simulation` (178), `wentorai/research-plugins@physics-skills` (146) | **Reject all.** High-install results (`r3f`, `babylonjs`, `godot`, `blender`) are 3D game/UI physics engines, not computational physics/cosmology solvers. Low-install skills (<200 installs) lack mathematical rigor compared to direct `CAMB`/`CLASS`/`REBOUND` documentation. |
| **`numerical analysis`** | `leonchaox/qinyan-academic-skills@matlab` (74), `vamseeachanta/workspace-hub@numpy-numerical-analysis` (28), `e-eight/scicomp-skills@numerical-verification` (11) | **Reject all.** All candidates have <100 installs (failing the `find-skills` quality threshold of >1,000 installs / >100 repo stars) or target proprietary MATLAB. |
| **`symbolic mathematics`** | `k-dense-ai/scientific-agent-skills@sympy` (2.1K), `jaechang-hits/sciagent-skills@sympy-symbolic-math` (91) | **Select `k-dense-ai/scientific-agent-skills@sympy` (2.1K installs, 55.5K repo stars).** Clean, MIT-licensed SymPy verification and code-generation patterns. |
| **`data analysis`** | `microsoft/azure-skills@azure-kusto` (571.9K), `anthropics/skills@xlsx` (190.9K), `yuan1z0825/nature-skills@nature-data` (15.7K), `claude-office-skills/skills@data-analysis` (8K) | **Reject for cosmology.** Top results target enterprise Kusto, Excel spreadsheets, or business BI. Cosmological data analysis requires `astropy` (FITS/tables), `camb`/`cobaya`, and `getdist` MCMC chain analysis. |
| **`hypothesis generation`** | `k-dense-ai/scientific-agent-skills@hypothesis-generation` (2.2K), `k-dense-ai@hypogenic` (2K), `neolabhq/context-engineering-kit@propose-hypotheses` (1.2K) | **Select `k-dense-ai/scientific-agent-skills@hypothesis-generation` (2.2K installs).** Enforces evidence-bounded rival explanations, discriminating predictions, and preregistration plans. |
| **`experimental design`** | `k-dense-ai/scientific-agent-skills@scientific-critical-thinking` (2.4K), `k-dense-ai@experimental-design` (1.4K), `k-dense-ai@statistical-power` (1.4K) | **Select `k-dense-ai@experimental-design` and `k-dense-ai@scientific-critical-thinking`.** Reject `adaptyv` (wet-lab protein assay API). |
| **`peer review`** | `mattpocock/skills@code-review` (827.2K), `yuan1z0825/nature-skills@nature-reviewer` (14.3K), `imbad0202/academic-research-skills@academic-paper-reviewer` (10K), `k-dense-ai/scientific-agent-skills@peer-review` (2.3K) | **Select `k-dense-ai/scientific-agent-skills@peer-review` (2.3K installs).** Deterministic, local-only Python 3.11+ stdlib compatibility, explicit claim-evidence and reproducibility checks. |
| **`citation verification`** | `galaxy-dawn/claude-scholar@citation-verification` (1.5K), `wanshuiyin/auto-claude-code-research-in-sleep@citation-audit` (502), `kalarislabs/research-agent-skills@citation-verification` (178) | **Select `KalarisLabs/research-agent-skills@citation-verification`** (after source code inspection: 4,572 bytes, MIT-licensed, zero external dependencies, queries Crossref + OpenAlex + arXiv directly). |
| **`reproducibility`** | `k-dense-ai/scientific-agent-skills@peer-review` (2.3K), `lingzhi227/agent-research-skills@backward-traceability` (2.2K), `mikubaka88/ccfa-skills@ccf-submission-checker` (2K) | **Covered by `peer-review` + `ara-rigor-reviewer` + local `pytest` harness.** |

---

## 2. Deep-Dive Audit: `KalarisLabs/research-agent-skills`

Per explicit instructions in Section 3, we inspected `https://github.com/KalarisLabs/research-agent-skills` via the GitHub API:

1. **Repository Metadata:**
   - **Stars / Forks:** 83 stars, 6 forks (created recently, updated `2026-10-31`).
   - **License:** Repository-level license reports `NOASSERTION` (`Other`), whereas individual `SKILL.md` frontmatters declare `MIT` or `BSD-3-Clause`.
   - **Contents:** 281 skill directories in `skills/`.
2. **Relationship to `k-dense-ai/scientific-agent-skills`:**
   - By comparing raw `SKILL.md` files (`literature-review`, `experimental-design`, `peer-review`, `hypothesis-generation`, `scientific-critical-thinking`, `sympy`, `astropy`), we verified that `KalarisLabs/research-agent-skills` aggregates and lightly adapts the majority of its core scientific skills from `k-dense-ai/scientific-agent-skills` (which has **55,517 stars**, 4,888 forks, and an explicit repo-wide `MIT` license).
3. **Why We Do Not Blindly Install the Full `KalarisLabs` Collection:**
   - Over 240 of the 281 skills in `KalarisLabs/research-agent-skills` are domain-specific to wet-lab biology (`13c-metabolic-flux`, `bulk-rnaseq`, `cellxgene-census`, `alphagenome`), clinical medicine (`clinical-decision-support`), finance (`usfiscaldata`, `market-research-reports`), or third-party cloud APIs (`benchling-integration`, `firecrawl-research-index`). Installing all 281 skills would pollute the agent context window and violate least-privilege principles.
4. **Component-By-Component Verdict on `KalarisLabs` Capabilities:**
   - **`literature-review`** (15,182 bytes): Good PRISMA workflow, but references `parallel-cli` and `PubMed`. **Recommendation:** Prefer upstream `k-dense-ai/scientific-agent-skills@literature-review`.
   - **`research-analysis` / `scientific-critical-thinking`** (11,457 bytes): Pure analytical markdown guide with zero network dependencies. **Recommendation:** Prefer upstream `k-dense-ai/scientific-agent-skills@scientific-critical-thinking`.
   - **`experimental-design`** (14,845 bytes): Strong coverage of factorial designs, blocking, and response-surface optimization. **Recommendation:** Prefer upstream `k-dense-ai/scientific-agent-skills@experimental-design`.
   - **`data-analysis` / `exploratory-data-analysis`**: Generic tabular EDA; redundant with our custom `astropy` + `getdist` + `scipy` stack. **Recommendation:** Skip.
   - **`citation-verification`** (4,572 bytes): **Strong standalone utility.** Uses only Python 3.9+ standard library (`urllib.request`, `json`, `xml`) to query Crossref, OpenAlex, and arXiv for hallucinated/mismatched references. **Recommendation:** Approve selective installation (`npx skills add KalarisLabs/research-agent-skills@citation-verification`).
   - **`peer-review`** (14,030 bytes) & **`ara-rigor-reviewer`** (18,529 bytes): Strong epistemic grading rubrics (scoring evidence relevance, falsifiability, scope calibration, argument coherence, exploration integrity, and methodological rigor from 1 to 5). **Recommendation:** Approve `k-dense-ai@peer-review` and incorporate the 6-axis ARA epistemic rubric directly into [`06-independent-review/peer-review/ADVERSARIAL_REVIEW_PROTOCOL.md`](file:///Users/rajat/personal/research/frontier-cosmology-lab/06-independent-review/peer-review/ADVERSARIAL_REVIEW_PROTOCOL.md).
