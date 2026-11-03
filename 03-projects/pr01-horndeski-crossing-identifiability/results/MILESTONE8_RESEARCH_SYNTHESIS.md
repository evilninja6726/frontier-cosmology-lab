# Milestone 8 Research Synthesis — Identifiability, Prior-Volume Effects, and Physical Stability of Phantom-Divide Crossing in Horndeski / EFT Dark Energy (`PR-01`)

**Date:** 2026-11-03  
**Project ID:** `PR-01`  
**Experiment Registry:** [BM_01_milestone7_report.json](file:///Users/rajat/personal/research/frontier-cosmology-lab/05-experiments/experiment-registry/BM_01_milestone7_report.json), [EXP_01_to_03_milestone8_report.json](file:///Users/rajat/personal/research/frontier-cosmology-lab/05-experiments/experiment-registry/EXP_01_to_03_milestone8_report.json)  
**Source Code:** [cosmology_baseline.py](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr01-horndeski-crossing-identifiability/baseline/cosmology_baseline.py), [horndeski_eft_solver.py](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr01-horndeski-crossing-identifiability/simulations/horndeski_eft_solver.py), [run_milestone8_experiments.py](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr01-horndeski-crossing-identifiability/experiments/run_milestone8_experiments.py)

---

## 1. Executive Summary of Original Findings

Our investigation of the apparent phantom-divide crossing ($w(z)$ crossing $w = -1$ from $w(z > z_c) < -1$ to $w_0 > -1$) reported in recent joint analyses of DESI BAO, Planck CMB, and Type Ia Supernovae yielded three quantitative findings across theoretical EFT stability (`EXP-01`), observational basis/calibration sensitivity (`EXP-02`), and linear perturbation growth (`EXP-03`):

1. **Exact Kineticity Cancellation Theorem (`EXP-01`, Epistemic Level `L2`):**
   In Horndeski scalar-tensor gravity with luminal tensor propagation ($\alpha_T = 0$, GW170817) and minimal Planck mass running ($\alpha_M = 0$), symbolic derivation via `SymPy` ([flrw_and_horndeski_symbolic.py](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr01-horndeski-crossing-identifiability/derivations/flrw_and_horndeski_symbolic.py)) proves that both the gradient stability numerator $N_s(a) \equiv D(a)c_s^2(a)$ and the quasi-static sub-horizon gravitational coupling modification:
   $$\mu(a) - 1 \equiv \frac{G_{\mathrm{eff}}(a)}{G_N} - 1 = \frac{\alpha_B^2(a)}{2 D(a) c_s^2(a)} = \frac{\alpha_B^2(a)}{2 N_s(a)}$$
   are **strictly independent of the kineticity $\alpha_K(a)$**. While taking $\alpha_K \to \infty$ lowers $c_s^2 \to 0^+$ to satisfy sub-luminality ($c_s^2 \le 1$), $\alpha_K$ cancels identically out of the product $D(a)c_s^2(a)$, making it mathematically impossible to tune $\alpha_K$ to suppress the gravitational slip induced by kinetic braiding $\alpha_B \ne 0$.

2. **Mandatory Negative Braiding & Prior-Volume Compression at Phantom Crossing (`EXP-01`, Epistemic Level `L3`):**
   Under standard proportional EFT parameterization $\alpha_K(a) = c_K \Omega_{\mathrm{DE}}(a)$ and $\alpha_B(a) = c_B \Omega_{\mathrm{DE}}(a)$, crossing $w(z_c) = -1$ from phantom ($w < -1$) to quintessence ($w_0 > -1$) requires strictly negative kinetic braiding:
   - $c_B \in [-2.900, -0.2395]$ for the DESI DR2 + Planck CMB + Union3 best-fit ($w_0 = -0.688, w_a = -0.978, z_c = 0.469$).
   - $c_B \in [-2.984, -0.3174]$ for the DESI DR2 + Planck CMB best-fit ($w_0 = -0.498, w_a = -1.484, z_c = 0.511$).
   All $c_B \ge 0$ (including unbraided $k$-essence $c_B = 0$) violate gradient stability ($c_s^2 < 0$). Consequently, the surviving stable EFT prior volume $\mathcal{V}_{\mathrm{EFT}}(w_0, w_a)$ shrinks from **$9.36\%$** in the quintessence regime ($w_0 = -0.90, w_a = +0.05$) to **$5.86\%$** at the DESI+CMB+Union3 best-fit and **$4.32\%$** at the DESI+CMB best-fit, inducing an effective Bayesian prior-volume penalty of $\Delta(-2\ln\mathcal{V}_{\mathrm{EFT}}) = +0.94$ to $+1.55$ relative to frequentist profile likelihoods.

3. **Sensitivity to Low-$z$ Supernova Calibration Offsets & Basis Choice (`EXP-02`, Epistemic Level `L3`):**
   Across $N = 38$ combined data points (13 DESI DR2 BAO + 3 Planck 2018 CMB shift parameters + 22 Union3 SNe Ia bins), CPL $w_0w_a$CDM with a single uniform SNe magnitude offset $M_B$ improves the fit over $\Lambda$CDM by $\Delta\chi^2 = -13.459$ ($\Delta\mathrm{AIC} = -9.459, \Delta\mathrm{BIC} = -6.184$). However:
   - Introducing a single low-redshift ($z \le 0.10$) SNe Ia calibration zero-point split $\Delta M_{\mathrm{low-}z}$ (analytically marginalized via generalized least squares in [`Union3SNLikelihood`](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr01-horndeski-crossing-identifiability/simulations/horndeski_eft_solver.py#L42-L136)) absorbs $\Delta M_{\mathrm{low-}z} = -0.041\text{ mag}$ in $\Lambda$CDM ($\chi^2: 42.681 \to 35.802$) and reduces the CPL improvement over $\Lambda$CDM from $\Delta\chi^2 = -13.459$ to **$\Delta\chi^2 = -7.232$** — a **$46.3\%$ reduction** in phantom-crossing preference (**confirming preregistered Hypothesis `H1.1`**).
   - Replacing linear CPL with a bounded Rational Padé `[1/1]` basis ($b = 0.50$) yields nearly identical goodness-of-fit ($\chi^2 = 29.153, \Delta\chi^2 = -13.528, z_c = 0.436$) but penalizes $\Delta\mathrm{BIC}$ from $-6.184$ to $-2.615$, while an unconstrained 3-parameter Shifted Chebyshev basis suffers a runaway pathology ($c_2 = -0.955$) at $z > 1.5$ where SNe/BAO data are sparse.

4. **Nuanced Resolution of the Growth-Slip Hypothesis (`EXP-03`, Epistemic Level `L3`):**
   Confronting the mandatory Horndeski braiding against redshift-space distortion growth measurements $f\sigma_8(z)$ over $z \in [0.1, 1.5]$ reveals a sharp distinction between the **stability boundary** and the **bulk EFT prior volume**:
   - **At the fine-tuned gradient-stability boundary ($c_B = c_{B,\mathrm{crit}} = -0.2395$ to $-0.3174$):** The maximum gravitational coupling boost is $\max[\mu(z)-1] = 3.69\%$ to $4.68\%$, which produces a cumulative fractional shift in $f\sigma_8(z)$ of **$+1.28\%$** (for DESI+CMB+Union3) to **$+1.61\%$** (for DESI+CMB) relative to `CAMB`'s standard `PPF` ($\mu \equiv 1$) approximation. Evaluated against both the `HYPOTHESIS_REGISTER.md` threshold ($> 1.5\%$) and the `EXP-03` threshold ($> 2.5\%$), **Hypothesis `H1.2` is refuted at the minimal boundary $c_B = c_{B,\mathrm{crit}}$ for the joint DESI+CMB+Union3 fit ($1.28\% < 1.5\% < 2.5\%$)** and only marginally reaches $1.61\%$ for the BAO+CMB-only fit.
   - **Across the interior/median of the surviving EFT prior volume ($c_B = -1.0$):** The gravitational coupling boost reaches $\max[\mu(z)-1] = 14.5\%$ to $16.8\%$, inducing a **$+5.36\%$ to $+6.72\%$ fractional boost in $f\sigma_8(z)$** — exceeding both the $1.5\%$ and $2.5\%$ thresholds by more than $2\times$ (**confirming `H1.2` across the bulk prior volume**).

---

## 2. Quantitative Summary Tables

### Table 1 — `EXP-02`: Joint Fits to DESI DR2 BAO + Planck 2018 CMB + Union3 SNe Ia ($N = 38$)

| Model Variant | $k$ | $H_0$ [$\text{km s}^{-1}\text{Mpc}^{-1}$] | $\Omega_{m0}$ | $w_0$ | $w_a$ | 3rd Param ($b$ or $c_2$ or $\Delta M_{\mathrm{low-}z}$) | $\chi^2_{\mathrm{total}}$ | $\Delta\chi^2$ vs $\Lambda$CDM | $\Delta\mathrm{AIC}$ | $\Delta\mathrm{BIC}$ | Crossing $z_c$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\Lambda$CDM (Uniform $M_B$) | 4 | $68.18$ | $0.3047$ | $-1.000$ | $0.000$ | — | $42.681$ | $0.000$ | $0.000$ | $0.000$ | None |
| CPL $w_0w_a$CDM (Uniform $M_B$) | 6 | $66.74$ | $0.3198$ | $-0.688$ | $-0.978$ | — | $29.222$ | $-13.459$ | $-9.459$ | $-6.184$ | $0.469$ |
| Padé `[1/1]` (Uniform $M_B$) | 7 | $66.78$ | $0.3194$ | $-0.674$ | $-1.148$ | $b = 0.500$ | $29.153$ | $-13.528$ | $-7.528$ | $-2.615$ | $0.436$ |
| Chebyshev (Unconstrained $c_2$) | 7 | $66.85$ | $0.3186$ | $-1.491$ | $+0.258$ | $c_2 = -0.955$ | $25.008$ | $-17.673$ | $-11.673$ | $-6.760$ | $0.520$ (runaway) |
| $\Lambda$CDM + Low-$z$ SNe Split | 5 | $68.29$ | $0.3036$ | $-1.000$ | $0.000$ | $\Delta M = -0.041\text{ mag}$ | $35.802$ | $-6.879$ | $-4.879$ | $-3.241$ | None |
| CPL + Low-$z$ SNe Split | 7 | $65.41$ | $0.3335$ | $-0.556$ | $-1.326$ | $\Delta M = -0.016\text{ mag}$ | $28.570$ | **$-7.232$** (vs $\Lambda\text{CDM}_{\mathrm{split}}$) | $-3.232$ | $+0.043$ | $0.503$ |

### Table 2 — `EXP-01` & `EXP-03`: Horndeski Stability Prior Volume & Growth Slip

| Cosmological Benchmark | $(w_0, w_a)$ | Stable EFT Fraction $\mathcal{V}_{\mathrm{EFT}}$ | Allowed Braiding $c_B$ Range | Prior Penalty $-2\ln(\mathcal{V}/\mathcal{V}_{\mathrm{quint}})$ | $\max[\mu(z)-1]$ at $c_{B,\mathrm{crit}}$ | $\max\frac{\Delta f\sigma_8}{f\sigma_8}$ at $c_{B,\mathrm{crit}}$ | $\max\frac{\Delta f\sigma_8}{f\sigma_8}$ at $c_B = -1.0$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Quintessence Benchmark | $(-0.90, +0.05)$ | $9.36\%$ | $[-2.770, +0.290]$ | $0.000$ | $0.00\%$ ($c_B=0$) | $0.00\%$ | $+4.82\%$ |
| DESI DR2 + CMB + Union3 | $(-0.688, -0.978)$ | $5.86\%$ | $[-2.900, -0.2395]$ | $+0.937$ | $+3.69\%$ | **$+1.28\%$** | **$+5.36\%$** |
| DESI DR2 + CMB + Union3 (Split) | $(-0.556, -1.326)$ | $4.78\%$ | $[-2.960, -0.2950]$ | $+1.344$ | $+4.38\%$ | **$+1.49\%$** | **$+6.18\%$** |
| DESI DR2 + CMB Only | $(-0.498, -1.484)$ | $4.32\%$ | $[-2.984, -0.3174]$ | $+1.546$ | $+4.68\%$ | **$+1.61\%$** | **$+6.72\%$** |

---

## 3. Epistemic Classification of Claims (`EPISTEMIC_STANDARDS.md`)

- **`L1` (Literature Baseline):** DESI DR2 BAO + Planck CMB + SNe Ia exhibit a $2.5\sigma$–$3.9\sigma$ frequentist preference for $w_0 > -1, w_a < 0$ in the CPL parameterization.
- **`L2` (Analytical / Symbolic Proof):** Kineticity $\alpha_K(a)$ cancels identically out of $D(a)c_s^2(a)$ and $\mu(a) - 1 = \alpha_B^2(a)/[2 D(a)c_s^2(a)]$ in $\alpha_T = \alpha_M = 0$ Horndeski gravity; verified symbolically in [flrw_and_horndeski_symbolic.py](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr01-horndeski-crossing-identifiability/derivations/flrw_and_horndeski_symbolic.py).
- **`L3` (Validated Numerical Finding):**
  1. Low-$z$ ($z \le 0.10$) SNe Ia zero-point calibration freedom reduces $\Delta\chi^2_{\mathrm{CPL - \Lambda CDM}}$ by $46.3\%$ in DESI DR2 + Planck CMB + Union3 (`H1.1` confirmed against both $\ge 30\%$ and $\ge 35\%$ thresholds).
  2. Horndeski stability compresses the surviving EFT parameter volume by $37\%$–$54\%$ across the phantom-crossing best-fit region (`EXP-01`).
  3. Braiding growth slip $\Delta(f\sigma_8)/(f\sigma_8)_{\mathrm{PPF}}$ is $+1.28\%$–$+1.61\%$ at the stability boundary $c_{B,\mathrm{crit}}$ and $+5.36\%$–$+6.72\%$ in the bulk stable prior volume $c_B = -1.0$ (`H1.2` nuanced/split verdict).

---

## 4. Methodological Limitations & Caveats (Stage 14 Adversarial Peer Review)

Per the Stage 14 audit ([PR01_ADVERSARIAL_REVIEW_REPORT.md](file:///Users/rajat/personal/research/frontier-cosmology-lab/06-independent-review/peer-review/PR01_ADVERSARIAL_REVIEW_REPORT.md)), the following structural caveats govern the interpretation of our results:

1. **Compressed CMB Shift Prior vs. Full Planck Boltzmann Likelihood:** Our CMB constraint uses a 3-parameter Gaussian shift-parameter prior on $(R, l_A, \omega_b)$ anchored to the Planck 2018 flat $\Lambda$CDM fiducial cosmology, which captures geometric acoustic constraints to $< 0.2\sigma$ but omits late-time Integrated Sachs–Wolfe (ISW) and CMB lensing ($C_\ell^{\phi\phi}$) sensitivity to $\alpha_B \ne 0$.
2. **Proportional EFT Scaling Ansatz ($\alpha_i(a) = c_i \Omega_{\mathrm{DE}}(a)$) vs. General Time-Dependent $\alpha_B(a)$:** The bounds $c_B \in [-2.90, -0.2395]$, $\mathcal{V}_{\mathrm{EFT}} = 5.86\%$, and $\max_z[\mu(z)-1] \ge 3.69\%$ assume $\alpha_i(a) = c_i \Omega_{\mathrm{DE}}(a)$ on $(c_K, c_B) \in [0.01, 6.0] \times [-3.0, 3.0]$. In a general Horndeski model where $\alpha_B(a)$ is an arbitrary function of time, gradient stability at $w(a_c) = -1$ can be satisfied locally via $\left.\frac{d\alpha_B}{d\ln a}\right|_{a_c} < 0$ even if $\alpha_B(a_c) = 0$ at the crossing instant (Vikman 2005; Deffayet et al. 2010; Bellini & Sawicki 2014).
3. **Minimal Coupling ($\alpha_M = 0$) and Luminal Tensor ($\alpha_T = 0$) Restriction:** The Exact Kineticity Cancellation Theorem holds within the $\alpha_T = \alpha_M = 0$ Horndeski subspace. Allowing non-zero Planck mass running ($\alpha_M \ne 0$) introduces $-2\alpha_M(1 - \alpha_B/2)$ into $D c_s^2$ and modifies $\mu(a)$ and the gravitational slip $\eta \equiv \Phi/\Psi$.
4. **Quasi-Static Sub-Horizon Approximation Breakdown as $\alpha_K \to \infty$:** While $\alpha_K$ cancels algebraically out of $\mu(a) - 1 = \alpha_B^2 / [2 D c_s^2]$ in the quasi-static limit ($c_s k \gg aH$), taking $\alpha_K \to \infty$ sends $c_s^2 \to 0^+$. If $c_s^2 \lesssim (aH/k)^2 \sim 10^{-4}$ on galaxy survey scales ($k \sim 0.1\,h\,\text{Mpc}^{-1}$), the sound horizon shrinks inside the clustering scale and invalidates the quasi-static approximation.
5. **Low-$z$ Supernova Step-Function Degeneracy (`EXP-02`):** A free step offset $\Delta M_{\mathrm{low-}z}$ at $z = 0.10$ is partially degenerate with genuine late-time background deceleration/acceleration curvature between $z < 0.10$ and $z > 0.10$; distinguishing a photometric/host-mass calibration offset from physical late-time evolution requires joint multi-survey calibration priors across Union3, Pantheon+, and DES-SN5YR.

