# RESEARCH DECISION LOG (ADRs) — Project Frontier

This log permanently records major architectural, methodological, and scientific decisions so that prior choices are not repeatedly revisited without new evidence.

---

## `ADR-001`: Selection of Workspace Architecture & Non-Empty Directory Policy
- **Date:** 2026-10-31
- **Decision:** Establish `/Users/rajat/personal/research/frontier-cosmology-lab` as an independent Git repository following the 00–09 modular laboratory hierarchy, instantiating only directories that contain real, verified artifacts at each milestone.
- **Rationale:** Prevents placeholder clutter while ensuring long-term separation between literature, derivations, software adapters, experiment logs, and independent adversarial reviews.

---

## `ADR-002`: Selective Vetting of Agent Skills vs. Full Collection Installation
- **Date:** 2026-10-31
- **Decision:** Do **not** blindly install all 281 skills in `KalarisLabs/research-agent-skills`. Recommend installing only 7 vetted scientific skills from upstream `k-dense-ai/scientific-agent-skills` (55.5K ★, MIT) plus `KalarisLabs/research-agent-skills@citation-verification` (zero-dependency Python stdlib verifier) and the official Antigravity `science` plugin.
- **Rationale:** Source inspection via GitHub API revealed that >240 skills in `KalarisLabs` target wet-lab biology, clinical medicine, or commercial SaaS APIs, and the repo root license is `NOASSERTION`. Selective installation preserves security, context hygiene, and scientific focus.

---

## `ADR-003`: Selection of Minimal Scientific Toolchain for Apple M5 (16 GB RAM)
- **Date:** 2026-10-31
- **Decision:** Standardize on `Python 3.12` + `SymPy` + `CAMB` (prebuilt ARM64 wheel) + `JAX`/`SciPy` + `Cobaya`/`GetDist` + `Astropy` as our core toolchain for Project `PR-01`. Defer compiling `Einstein Toolkit` or downloading multi-terabyte `CAMELS` raw snapshots.
- **Rationale:** Apple M5 (10 cores, 16 GB unified memory) executes `CAMB`, `JAX` auto-diff, and `Cobaya` MCMC chains with sub-second per-evaluation speed, whereas 3D AMR numerical relativity (`Einstein Toolkit`) and raw hydrodynamic snapshots (`CAMELS`) exceed local memory/storage constraints and are unnecessary for `PR-01` and `PR-02`.

---

## `ADR-004`: Selection of Primary (`PR-01`) and Backup (`PR-02`) Research Projects
- **Date:** 2026-10-31
- **Decision:**
  - **Primary (`PR-01`, Score 92.0/100):** *Identifiability, Prior-Volume Effects, and Physical Stability of Phantom-Divide Crossing in Horndeski / EFT Dark Energy under DESI DR2, CMB, and SNe Ia*.
  - **Backup (`PR-02`, Score 90.5/100):** *Environmental vs. Intrinsic Perturbations in Black Hole Ringdown: Quantitative Pseudospectrum Bounds and Time-Domain Observability*.
- **Rationale:** Both projects address active 2025–2026 research debates (`DESI DR2` dynamical dark energy vs. `Jaramillo et al.` / 2026 WKB halo QNM claims), have 100% open observational data, admit exact symbolic derivations in `SymPy`, and run natively on Apple M5 hardware with clear falsification gates.
