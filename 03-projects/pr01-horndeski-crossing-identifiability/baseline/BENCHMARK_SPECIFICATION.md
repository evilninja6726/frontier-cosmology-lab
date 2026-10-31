# MILESTONE 7 BENCHMARK SPECIFICATION (`BM-01`)

**Rule:** *Do not proceed to original model modifications until the baseline passes its validation criteria.*

---

## 1. Benchmark Objective
Before introducing Horndeski EFT stability priors, Chebyshev/Padé bases, or growth-rate modifications, we must reproduce the standard **$\Lambda$CDM** and **CPL $w_0w_a$CDM** background distance ratios, sound horizon $r_d$, linear growth factor $D_+(z)$, and DESI DR2 BAO $\chi^2$ contours using both **community software (`CAMB`)** and our **independent local `JAX`/`SciPy` solver**.

---

## 2. Reference Sources & Fiducial Parameters
1. **Planck 2018 Baseline $\Lambda$CDM (`TT,TE,EE+lowE+lensing` Table 2 of [`Planck:2018vi`](https://arxiv.org/abs/1807.06209), DOI: `10.1051/0004-6361/201833910`):**
   - $\Omega_b h^2 = 0.02237$
   - $\Omega_c h^2 = 0.1200$
   - $100\theta_{\mathrm{MC}} = 1.04092$ ($H_0 = 67.36\text{ km s}^{-1}\text{ Mpc}^{-1}$, $\Omega_m = 0.3153$ including $\Sigma m_\nu = 0.06\text{ eV}$)
   - $\ln(10^{10} A_s) = 3.044$, $n_s = 0.9649$, $\tau_{\mathrm{reio}} = 0.0544$
   - **Expected Published Drag Sound Horizon:** $r_d = 147.09 \pm 0.26\text{ Mpc}$
2. **DESI DR2 BAO Measurements (Table III / IV of [`DESI:2025dr2`](https://arxiv.org/abs/2503.14738), DOI: `10.1103/tr6y-kpc6`):**
   - BGS ($z_{\mathrm{eff}} = 0.295$): $D_V(z)/r_d$
   - LRG1 ($z_{\mathrm{eff}} = 0.510$), LRG2 ($z_{\mathrm{eff}} = 0.706$), LRG3+ELG1 ($z_{\mathrm{eff}} = 0.934$), ELG2 ($z_{\mathrm{eff}} = 1.317$), QSO ($z_{\mathrm{eff}} = 1.491$), Ly$\alpha$ ($z_{\mathrm{eff}} = 2.330$): $(D_M(z)/r_d, D_H(z)/r_d)$

---

## 3. Quantitative Acceptance Criteria (Automated `pytest` Gates)

| Test ID | Observable / Quantity | Comparison Pair | Maximum Allowed Relative Error |
| :--- | :--- | :--- | :--- |
| **`T7.1`** | Normalized Hubble expansion $E(z) = H(z)/H_0$ for $\Lambda$CDM and CPL ($w_0 = -0.75, w_a = -0.85$) across $z \in [0, 1100]$ | Custom `JAX`/`SciPy` vs. `CAMB` (`PPF` background) | $< 1.0 \times 10^{-5}$ |
| **`T7.2`** | Transverse & radial BAO distances $D_M(z), D_H(z), D_V(z)$ at all 7 DESI DR2 redshifts | Custom `JAX`/`SciPy` vs. `CAMB` & `astropy.cosmology` | $< 1.0 \times 10^{-5}$ |
| **`T7.3`** | Comoving sound horizon at drag epoch $r_d(z_{\mathrm{drag}})$ at Planck 2018 fiducial parameters | `CAMB` integral vs. published Planck 2018 value ($147.09\text{ Mpc}$) | $< 1.0 \times 10^{-3}$ |
| **`T7.4`** | Sub-horizon linear growth factor $D_+(z)$ and growth rate $f(z) \equiv \frac{d\ln D_+}{d\ln a}$ in $\Lambda$CDM | Custom `JAX`/`SciPy` ODE solver vs. `CAMB` matter transfer function | $< 5.0 \times 10^{-4}$ |
| **`T7.5`** | Baseline $\chi^2_{\Lambda\mathrm{CDM}}$ and $\Delta\chi^2_{w_0w_a\mathrm{CDM}}$ on DESI DR2 BAO + compressed CMB | Local `JAX` profiler / `Cobaya` vs. published [`DESI:2025dr2`](https://arxiv.org/abs/2503.14738) best-fit $(w_0, w_a)$ | Best-fit $(w_0, w_a)$ within $\pm 0.02$ of published contour peak |
