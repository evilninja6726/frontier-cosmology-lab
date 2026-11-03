"""Serverless Cloud Execution Runner for Project Frontier via Modal (modal.com).

Executes all heavy cosmological simulations, JAX/SciPy optimizations, CAMB Boltzmann
calculations, and pytest verification suites on remote cloud containers so that local
laptop CPU/thermal load remains at 0%.

Usage:
    1. Authenticate once (opens browser):
       /Users/rajat/personal/research/.venv/bin/modal setup
    2. Run Milestone 8 + Pytest in the cloud and sync artifacts back locally:
       /Users/rajat/personal/research/.venv/bin/modal run infrastructure/cloud/modal_runner.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import modal

_THIS_FILE = pathlib.Path(__file__).resolve()
REPO_ROOT = (
    _THIS_FILE.parents[2]
    if len(_THIS_FILE.parents) > 2
    else pathlib.Path("/workspace/frontier-cosmology-lab")
)

# Define the reproducible scientific container image in Modal's cloud
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gfortran", "build-essential", "libopenblas-dev")
    .pip_install(
        "numpy>=2.0",
        "scipy>=1.14",
        "sympy>=1.13",
        "jax>=0.4.30",
        "jaxlib>=0.4.30",
        "camb>=1.5.8",
        "cobaya>=3.5",
        "astropy>=6.1",
        "getdist>=1.5",
        "matplotlib>=3.9",
        "pytest>=8.0",
    )
    .add_local_dir(
        local_path=str(REPO_ROOT),
        remote_path="/workspace/frontier-cosmology-lab",
        ignore=[".git", ".venv", "__pycache__", ".pytest_cache", "graphify-out"],
    )
)

app = modal.App("frontier-cosmology-lab", image=image)


@app.function(cpu=4.0, memory=8192, timeout=900)
def execute_research_pipeline_remote(run_milestone7: bool = False, run_milestone8: bool = True) -> dict:
    """Run pytest and Milestone benchmarks/experiments inside a remote cloud container."""
    remote_root = pathlib.Path("/workspace/frontier-cosmology-lab")
    pr01_dir = remote_root / "03-projects" / "pr01-horndeski-crossing-identifiability"
    env = {
        "PYTHONPATH": f"{remote_root}:{pr01_dir}",
        "OMP_NUM_THREADS": "4",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }

    logs = []
    if run_milestone7:
        proc7 = subprocess.run(
            [sys.executable, str(pr01_dir / "reproduction" / "run_milestone7_benchmark.py")],
            cwd=str(remote_root),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        logs.append("=== MILESTONE 7 (BM-01) ===\n" + proc7.stdout)

    if run_milestone8:
        proc8 = subprocess.run(
            [sys.executable, str(pr01_dir / "experiments" / "run_milestone8_experiments.py")],
            cwd=str(remote_root),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        logs.append("=== MILESTONE 8 (EXP-01..03) ===\n" + proc8.stdout)

    proc_test = subprocess.run(
        [sys.executable, "-m", "pytest", str(remote_root / "infrastructure" / "tests"), "-v"],
        cwd=str(remote_root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    logs.append("=== PYTEST SUITE ===\n" + proc_test.stdout)

    # Collect generated artifacts to stream back to the local workspace
    artifacts: dict[str, bytes] = {}
    artifact_rel_paths = [
        "05-experiments/experiment-registry/BM_01_milestone7_report.json",
        "05-experiments/experiment-registry/EXP_01_to_03_milestone8_report.json",
        "03-projects/pr01-horndeski-crossing-identifiability/results/milestone7_baseline_validation.png",
        "03-projects/pr01-horndeski-crossing-identifiability/results/milestone8_horndeski_identifiability.png",
    ]
    for rel in artifact_rel_paths:
        p = remote_root / rel
        if p.exists():
            artifacts[rel] = p.read_bytes()

    return {
        "logs": "\n\n".join(logs),
        "artifacts": artifacts,
    }


@app.local_entrypoint()
def main(run_milestone7: bool = False, run_milestone8: bool = True) -> None:
    """Trigger remote execution on Modal and write returned artifacts to local disk."""
    print("Dispatching Project Frontier research pipeline to Modal Cloud (0% local CPU)...")
    result = execute_research_pipeline_remote.remote(
        run_milestone7=run_milestone7, run_milestone8=run_milestone8
    )
    print(result["logs"])

    for rel_path, content in result["artifacts"].items():
        local_target = REPO_ROOT / rel_path
        local_target.parent.mkdir(parents=True, exist_ok=True)
        local_target.write_bytes(content)
        print(f"Synced remote artifact -> {local_target} ({len(content):,} bytes)")
