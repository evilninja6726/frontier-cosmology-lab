# Cloud Compute Architecture for Project Frontier (Zero Local Thermal Load)

To keep local laptop CPU and thermal load at **0%** while executing intensive Boltzmann (`CAMB`), automatic-differentiation (`JAX`), and MCMC/optimization workloads, `frontier-cosmology-lab` provides a **Dual-Tier Cloud Execution Architecture**:

---

## Tier 1: Serverless Interactive Cloud via Modal (`modal.com`) — Recommended Primary Workflow

[modal_runner.py](file:///Users/rajat/personal/research/frontier-cosmology-lab/infrastructure/cloud/modal_runner.py) packages the local research repository into a remote Debian Bookworm Python 3.12 container (`cpu=4.0`, `memory=8192 MB`) with `CAMB`, `JAX`, `SciPy`, `SymPy`, `Cobaya`, `Astropy`, and `pytest` pre-installed, runs the experiments in Modal's cloud, and streams the `.json` reports and `.png` figures back to your local disk.

### One-Time Setup
```bash
# 1. Install the lightweight Modal client CLI into the virtual environment
/Users/rajat/personal/research/.venv/bin/pip install modal

# 2. Authenticate via browser (includes $30/month free cloud compute credits)
/Users/rajat/personal/research/.venv/bin/modal setup
```

### Running Experiments in the Cloud (0% Local CPU)
```bash
# Run Milestone 8 (EXP-01..03) + Full Pytest Suite remotely and sync artifacts locally:
/Users/rajat/personal/research/.venv/bin/modal run infrastructure/cloud/modal_runner.py

# Run both Milestone 7 (BM-01) and Milestone 8 (EXP-01..03) remotely:
/Users/rajat/personal/research/.venv/bin/modal run infrastructure/cloud/modal_runner.py --run-milestone7
```

---

## Tier 2: Automated Cloud CI/CD via GitHub Actions (100% Free on GitHub)

[cloud_research_pipeline.yml](file:///Users/rajat/personal/research/frontier-cosmology-lab/.github/workflows/cloud_research_pipeline.yml) executes the full symbolic derivation suite, Milestone 7 (`BM-01`), Milestone 8 (`EXP-01`–`EXP-03`), and `pytest` verification suite on GitHub's cloud `ubuntu-latest` runners (`4 vCPUs, 16 GB RAM`) on every push to `main` or via manual **Run workflow** (`workflow_dispatch`).

### Usage
1. Push `frontier-cosmology-lab` to your GitHub repository:
   ```bash
   git remote add origin https://github.com/<your-username>/frontier-cosmology-lab.git
   git push -u origin main
   ```
2. Open the **Actions** tab on GitHub -> **Project Frontier Cloud Research Pipeline** -> download the `frontier-cosmology-results` artifact bundle containing the JSON reports and publication PNG figures.
