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

---

## `ADR-005`: Scale-Invariant Horndeski Stability Ratios & Exact Analytical $c_K$ Integration
- **Date:** 2026-11-03
- **Decision:** Evaluate Horndeski no-ghost ($D(z)/\Omega_{\mathrm{DE}}(z) > 0$) and gradient stability ($N_s(z)/\Omega_{\mathrm{DE}}(z) > 0$) using scale-invariant ratios normalized by $\Omega_{\mathrm{DE}}(z) > 0$, and compute the sub-luminality survival fraction over kineticity $c_K \in [c_{K,\min}, c_{K,\max}]$ analytically via $c_K \ge \max_z [N_s(c_B, z)/\Omega_{\mathrm{DE}}(z) - \frac{3}{2}c_B^2 \Omega_{\mathrm{DE}}(z)]$.
- **Rationale:** Eliminates false numerical underflow at $z = 1100$ (where $\Omega_{\mathrm{DE}}(1100) \sim 10^{-16}$) and reduces 3D $(c_K, c_B, z)$ array memory and compute by $120\times$ while replacing discrete $c_K$ grid quadrature with the exact analytical survival integral.

---

## `ADR-006`: Dual-Tier Cloud Execution Architecture (`Modal` + `GitHub Actions`)
- **Date:** 2026-11-03
- **Decision:** Offload all multi-core `CAMB` / `JAX` / `SciPy` grid scans, non-linear optimizations, and `pytest` runs to a dual-tier cloud setup (`infrastructure/cloud/modal_runner.py` for serverless interactive execution on Modal containers and `.github/workflows/cloud_research_pipeline.yml` for automated GitHub Actions CI/CD).
- **Rationale:** Prevents multi-threaded OpenMP/JAX workloads from heating up the local MacBook while maintaining full deterministic reproducibility and automated artifact syncing back to the local repository.

---

## `ADR-007`: Stage 14 Adversarial Peer Review Resolution & Split Verdict on Hypothesis `H1.2`
- **Date:** 2026-11-03
- **Decision:** Record the nuanced/split empirical outcome for Hypothesis `H1.2` (refuted at the fine-tuned gradient-stability boundary $c_B = c_{B,\mathrm{crit}}$ where $\Delta f\sigma_8 / f\sigma_8 = +1.28\%\text{--}+1.61\%$, confirmed across the interior/median of the surviving EFT prior volume $c_B = -1.0$ where $\Delta f\sigma_8 / f\sigma_8 = +5.36\%\text{--}+6.72\%$), alongside all 5 structural methodological caveats identified by the independent Adversarial Physics Reviewer (`PR01_ADVERSARIAL_REVIEW_REPORT.md`).
- **Rationale:** Upholds the laboratory's core charter rule (*"Never sacrifice scientific integrity for an exciting discovery narrative"*) by preserving boundary counterexamples and methodological scope limits alongside confirmed findings (`H1.1` confirmed at $46.3\%$ reduction).

