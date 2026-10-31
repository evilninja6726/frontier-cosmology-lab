# HARDWARE AND ENVIRONMENT AUDIT — Milestone 1

**Audit Date:** 2026-10-31  
**Workspace Root:** `/Users/rajat/personal/research/frontier-cosmology-lab`

---

## 1. Host Hardware Capabilities

| Parameter | Audited Value | Scientific Computing Implications |
| :--- | :--- | :--- |
| **Machine / OS** | macOS 25.6.0 (`Darwin Kernel 25.6.0`, `arm64`) | Native Apple Silicon UNIX environment; supports Accelerate/BLAS/Metal and ARM64 SIMD vectorization. |
| **Processor** | **Apple M5** (10 CPU cores, Unified Memory Architecture) | High single-thread IPC and fast unified memory bandwidth; well-suited for Boltzmann solvers (`CAMB`/`CLASS`), spectral PDE solvers, `JAX` JIT compilation, and medium-scale MCMC/HMC chains ($10^5\text{--}10^6$ evaluations). |
| **System RAM** | **16 GB** (`17,179,869,184` bytes Unified Memory) | Sufficient for linear Boltzmann codes, Fisher/MCMC inference, 1D/2D spectral GR perturbation solvers, and neural emulators (`CosmoPower`). Not suited for full 3D AMR numerical relativity (`Einstein Toolkit` BSSN BBH mergers require $>64\text{--}256\text{ GB}$ RAM) or raw multi-terabyte `CAMELS` hydrodynamical snapshots locally. |

---

## 2. Compilers, Package Managers, and Runtimes

| Tool | Status | Path / Version | Notes |
| :--- | :--- | :--- | :--- |
| **Python (Project `.venv`)** | **INSTALLED** | `/Users/rajat/personal/research/.venv/bin/python3` (`Python 3.12.15`) | Clean CPython 3.12.15 ARM64 build. Currently has minimal packages (`aiohttp`, `edge-tts`, `tabulate`). |
| **`uv` Package Manager** | **INSTALLED** | `/Users/rajat/personal/research/.local/bin/uv` | High-speed deterministic Python environment and dependency resolver. |
| **Homebrew** | **INSTALLED** | `/opt/homebrew/bin/brew` | Available for installing system C/C++/Fortran libraries if needed. |
| **C/C++ Compilers** | **INSTALLED** | `/usr/bin/clang`, `/usr/bin/gcc` (Apple Clang) | Ready for C/C++ extensions (`CLASS`, `REBOUND`). |
| **Fortran Compiler (`gfortran`)** | **MISSING** | Not installed (`brew install gcc` provides `gfortran`) | Not needed if installing `camb` via prebuilt ARM64 wheels (`uv pip install camb`), which bundles compiled `libcamb`. Needed only if modifying CAMB's Fortran90 source (`EFTCAMB`). |
| **CMake** | **MISSING** | Not installed (`brew install cmake`) | Needed only if compiling `CLASSpp_public` from source. |
| **Node.js / `npx`** | **INSTALLED** | `/Users/rajat/.nvm/versions/node/v24.22.0/bin/npx` | Used for `npx skills` CLI discovery and installation. |
| **Git** | **INSTALLED** | `/usr/bin/git` | Version control for reproducible research state. |

---

## 3. Missing Scientific Python Dependencies (To Be Installed for Milestone 7)

The `.venv` currently lacks the core scientific Python stack. We have declared the exact required dependencies in `pyproject.toml` and can install them in $<30$ seconds via `uv`:

```bash
/Users/rajat/personal/research/.local/bin/uv pip install --python /Users/rajat/personal/research/.venv/bin/python3 \
  numpy scipy sympy astropy matplotlib jax jaxlib camb cobaya getdist pytest
```

### Why This Stack Fits Our 16 GB Apple M5 Hardware
1. **Zero Fortran Compilation Overhead for Baseline:** `camb` provides pre-compiled macOS `arm64` wheels on PyPI (`libcamb` compiled for Apple Silicon), enabling immediate millisecond-scale CMB and BAO computation.
2. **Differentiable Acceleration:** `jax` (`cpu`/`metal` on Apple Silicon) enables exact automatic differentiation ($\nabla_\theta \ln \mathcal{L}$, Fisher matrices $F_{ij} = -\langle \partial_i \partial_j \ln \mathcal{L}\rangle$, and No-U-Turn Sampler HMC) without finite-difference noise.
3. **Symbolic Verification:** `sympy` runs locally with zero external dependencies to verify every analytical derivation (Christoffel symbols, Friedmann equations, Horndeski stability conditions $Q_s, c_s^2$) before numerical coding.
