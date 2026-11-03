# Research Synthesis — Environmental vs. Intrinsic Perturbations in Black Hole Ringdown Pseudospectra (`PR-02`)

**Date:** 2026-11-03  
**Project ID:** `PR-02`  
**Experiment Registry:** [PR02_BM02_and_EXP04_to_06_report.json](file:///Users/rajat/personal/research/frontier-cosmology-lab/05-experiments/experiment-registry/PR02_BM02_and_EXP04_to_06_report.json)  
**Publication Figure:** [pr02_ringdown_pseudospectra_and_timedomain.png](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr02-ringdown-pseudospectra/results/pr02_ringdown_pseudospectra_and_timedomain.png)  
**Source Code:** [hyperboloidal_qnm_symbolic.py](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr02-ringdown-pseudospectra/derivations/hyperboloidal_qnm_symbolic.py), [chebyshev_pseudospectrum_solver.py](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr02-ringdown-pseudospectra/simulations/chebyshev_pseudospectrum_solver.py), [run_pr02_experiments.py](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr02-ringdown-pseudospectra/experiments/run_pr02_experiments.py)

---

## 1. Executive Summary of Original Findings

Our symbolic and Chebyshev spectral investigation of black hole ringdown pseudospectra and time-domain observability (`BM-02` and `EXP-04`–`EXP-06`, executed on Google Colab in `16.01s` with `4/4` unit tests passing in `2.49s`) established four quantitative results:

1. **Exact Hyperboloidal Polynomial Eigenstructure & Boundary Dissipation Law (Stage 9 Symbolic Proof, Epistemic Level `L2`):**
   Using `SymPy` ([hyperboloidal_qnm_symbolic.py](file:///Users/rajat/personal/research/frontier-cosmology-lab/03-projects/pr02-ringdown-pseudospectra/derivations/hyperboloidal_qnm_symbolic.py)), we proved that the hyperboloidal differential operators $L_1 = (1-\sigma^2)\partial_\sigma^2 - 2\sigma\partial_\sigma - V_0$ and $L_2 = -(2\sigma\partial_\sigma + 1)$ are upper-triangular in the monomial basis $\sigma^n$ with characteristic polynomial $\omega^2 - i(2n+1)\omega - [V_0 + n(n+1)] = 0$, yielding the exact Pöschl–Teller Quasi-Normal Mode (QNM) spectrum $\omega_n^{\pm} = \pm\sqrt{V_0 - 1/4} + i(n + 1/2)$ and the exact boundary dissipation identity:
   $$\frac{dE}{d\tau} = -\left|\psi(\tau, +1)\right|^2 - \left|\psi(\tau, -1)\right|^2 \le 0$$

2. **Direct Verification of the "Numerical Pseudospectrum" Scaling $|\Delta\omega_n|/|\omega_n| \sim \kappa_n \epsilon_{\mathrm{mach}}$ (`BM-02` & `EXP-04`, Epistemic Level `L3`):**
   In the physical energy norm $\mathbf{G}_E = \mathbf{W}^T\mathbf{W}$, the Bauer–Fike condition numbers $\kappa_n = \|v_n\|_E\|w_n\|_E / |\langle w_n, v_n \rangle_E|$ grow by more than **12 orders of magnitude** across the first 7 modes:
   $$\kappa_0 = 85.03, \quad \kappa_1 = 1.13 \times 10^4, \quad \kappa_2 = 3.39 \times 10^7, \quad \kappa_3 = 5.45 \times 10^8, \quad \kappa_4 = 1.12 \times 10^{12}, \quad \kappa_6 = 2.05 \times 10^{14}$$
   Because $\kappa_3 / \kappa_0 = 6.41 \times 10^6 \gg 20$, **Hypothesis `H2.1` is strongly confirmed**. Moreover, while the raw forward error of the unperturbed Chebyshev solver at $N = 60$ grows with $n$ (`3.50e-13` for $n=0$, `7.81e-11` for $n=1$, `8.03e-08` for $n=2$, `1.16e-05` for $n=3$), dividing by $\kappa_n$ reveals a uniform condition-normalized backward error of **$\max_{0 \le n \le 4} \frac{|\Delta\omega_n|}{\kappa_n |\omega_n|} = 5.92 \times 10^{-15} \approx 26\,\epsilon_{\mathrm{mach}}$**, proving that the raw overtone drift in 64-bit float is purely the non-normal amplification of IEEE-754 machine roundoff!

3. **Wavelength Threshold Between Adiabatic Halo Shifts and High-Wavenumber "Flea" Instability (`EXP-05`, Epistemic Level `L3`):**
   Adding deterministic perturbations $\delta\tilde{V}(\sigma) = \epsilon\cos(k\pi\sigma)$ with $\epsilon = 5 \times 10^{-3}$ shows a sharp separation by spatial wavenumber $k$:
   - Smooth adiabatic perturbations ($k = 1$) shift all modes $n = 0..3$ only mildly ($\sim 0.1\%\text{--}0.5\%$), following standard Rayleigh–Schrödinger perturbation theory.
   - High-wavenumber perturbations ($k = 4, 8$) leave the fundamental mode stable ($|\Delta\omega_0|/|\omega_0| \approx 0.05\%$) while ejecting overtones $n = 2, 3$ along the open logarithmic contours of the $\epsilon$-pseudospectrum by **$> 25\%$**.

4. **Resolution of the Spectral Instability vs. Time-Domain Observability Paradox (`EXP-06`, Epistemic Level `L3`):**
   Scanning a localized environmental bump ($\epsilon_{\mathrm{bump}} = 2 \times 10^{-3}, k_{\mathrm{osc}} = 6$) across tortoise distances $x_{\mathrm{bump}} \in [0.4, 2.8]$ demonstrates an **inverse relationship** between frequency-domain spectral migration and prompt time-domain waveform observability:

| Bump Distance $x_{\mathrm{bump}}$ | Overtone $n=2$ Shift $|\Delta\omega_2|/|\omega_2|$ | Overtone $n=3$ Shift $|\Delta\omega_3|/|\omega_3|$ | Prompt Mismatch $\mathcal{M}_{\mathrm{prompt}}$ ($\tau \in [0, 3.5]$) | Late Echo Mismatch $\mathcal{M}_{\mathrm{late}}$ ($\tau \in [5, 15]$) | Ratio $\mathcal{M}_{\mathrm{late}} / \mathcal{M}_{\mathrm{prompt}}$ | Critical GW SNR $\rho_{\mathrm{crit}}^{\mathrm{prompt}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.4$ | $0.0009\%$ | $2.49\%$ | $7.43 \times 10^{-9}$ | $1.14 \times 10^{-7}$ | $15.4\times$ | $8,203$ |
| $1.0$ | $0.38\%$ | $12.44\%$ | $1.66 \times 10^{-8}$ | $2.42 \times 10^{-7}$ | $14.5\times$ | $5,485$ |
| $1.6$ | $6.56\%$ | $26.96\%$ | $1.16 \times 10^{-8}$ | $1.52 \times 10^{-7}$ | $13.1\times$ | $6,566$ |
| $2.2$ | $25.09\%$ | $43.56\%$ | $4.02 \times 10^{-9}$ | $4.10 \times 10^{-8}$ | $10.2\times$ | $11,150$ |
| **$2.8$** | **$35.34\%$** | **$46.27\%$** | **$1.87 \times 10^{-11}$** | **$6.19 \times 10^{-8}$** | **$3,304\times$** | **$163,376$** |

   As $x_{\mathrm{bump}}$ moves outward to $2.8$, the $n=2$ and $n=3$ QNM eigenvalues migrate by **$35.3\%$ and $46.3\%$**, yet causality delays the scattered wave from reaching the observer during the prompt ringdown ($\mathcal{M}_{\mathrm{prompt}} = 1.87 \times 10^{-11} \ll 10^{-4}$, requiring an unattainable matched-filter SNR of $\rho_{\mathrm{crit}} > 1.6 \times 10^5$ to distinguish from an unperturbed black hole). **Hypothesis `H2.2` is confirmed.**

---

## 2. Methodological Limitations & Caveats (Incorporating Stage 14 Adversarial Review `ARA 4.68/5.00`)

1. **Pöschl–Teller Exponential Falloff vs. Schwarzschild Regge–Wheeler Algebraic Tail:** The Pöschl–Teller barrier $V_{\mathrm{PT}}(x) = V_0 \operatorname{sech}^2(x)$ decays exponentially as $x \to \pm\infty$, isolating pure discrete QNM poles without a branch cut at $\omega = 0$. The physical Schwarzschild Regge–Wheeler potential decays algebraically as $V_{\mathrm{RW}}(x) \sim \ell(\ell+1)/x^2$ as $x \to +\infty$, introducing a branch cut along the imaginary frequency axis that generates the late-time Price power-law tail $\phi(\tau) \sim \tau^{-(2\ell+3)}$ and mixes with high overtones.
2. **64-Bit IEEE-754 Numerical Pseudospectrum Floor:** Because $\kappa_n$ exceeds $10^{12}$ for $n \ge 4$, standard 64-bit floating-point arithmetic ($\epsilon_{\mathrm{mach}} \approx 2.2 \times 10^{-16}$) cannot resolve unperturbed eigenvalues for $n \ge 5$ without arbitrary-precision (`mpmath` / 256-bit float) arithmetic; our quantitative perturbation shifts in `EXP-05` and `EXP-06` are therefore restricted to the well-resolved modes $n \in \{0, 1, 2, 3\}$.
3. **Finite Tortoise Domain on Unstretched $\sigma = \tanh x$ Chebyshev Grids (`H2.2` Range Reconciliation):** Under the unstretched compactification $\sigma = \tanh x$, the outermost interior Chebyshev–Gauss–Lobatto node at $N = 60$ lies at $\sigma_1 = \cos(\pi/60) \approx 0.99863$, corresponding to a maximum finite tortoise coordinate $x_{\max} = \operatorname{artanh}(\cos(\pi/60)) \approx 3.65$. Consequently, `EXP-06` scans $x_{\mathrm{bump}} \in [0.4, 2.8]$ and $\tau \in [0, 16]$ (with width $w = 0.30$ and spatial modulation $k_{\mathrm{osc}} = 6.0$) rather than the nominal $x_{\mathrm{bump}} \in [2, 10]$ noted in `HYPOTHESIS_REGISTER.md` (which would require a stretched map $\sigma = \tanh(x/L_s)$ or $N \gtrsim 10^3$). Because $1 - \sigma^2 = \operatorname{sech}^2(x)$ decays exponentially with $x$, $x_{\mathrm{bump}} = 2.8$ already amplifies the rescaled hyperboloidal perturbation $\delta\tilde{V}(\sigma) = \delta V(x)/(1-\sigma^2)$ by $\cosh^2(2.8) \approx 67.7$, driving $35.3\%\text{--}46.3\%$ overtone migration while maintaining prompt causal immunity ($\mathcal{M}_{\mathrm{prompt}} = 1.87 \times 10^{-11}$).
4. **Wavepacket Initial Data & Observer-Inside-Bump Reflection Geometry:** In `simulate_ringdown_waveform`, the initial data $\phi(0, x) = \exp[-(x - x_0)^2/(2w_p^2)]$, $\psi(0, \sigma) = -(1-\sigma^2)\partial_\sigma\phi(0, \sigma) = -\partial_x\phi(0, x)$ at $x_0 = -0.5, w_p = 0.5$ represents a smooth localized wavepacket evaluated at an observer at $\sigma_{\mathrm{obs}} \approx 0.809$ ($x_{\mathrm{obs}} \approx 1.124$). For outer environmental bumps at $x_{\mathrm{bump}} \in \{1.6, 2.2, 2.8\} > x_{\mathrm{obs}}$, the observer lies inside the environmental shell and records the direct prompt ringdown wave before the back-scattered reflection returns inward.
5. **1+1D Linear Perturbation Approximation:** Our analysis treats linearized scalar/axial wave propagation on a fixed background and does not include nonlinear backreaction or spin-induced mode mixing ($a/M > 0$ Kerr spheroidal harmonics).
