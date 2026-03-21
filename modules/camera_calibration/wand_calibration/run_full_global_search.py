import logging
import os
from pathlib import Path

from .full_global_search import (
    run_global_search,
    emit_diagnostics,
    ProbingConfig,
    GenerationDetailConfig,
)


camfile_dir = Path(r"J:\Fish\T0\Refraction\camFile")
obs_csv_path = Path(r"J:\Fish\T0\wand_points_selected.csv")
output_dir = Path(r"J:\Fish\T0\Full_Global_Search_multidim")
log_path = output_dir / "full_global_search.log"

output_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ],
    force=True,
)

logging.info("Logging to console and %s", log_path)

# Enable DEBUG logging for probing module to show detailed probing steps
logging.getLogger('modules.camera_calibration.wand_calibration.full_global_search').setLevel(logging.DEBUG)

# Use fixed worker count to ensure parallelization threshold is met
# Threshold: n_params >= 2 * max_workers
# With 36 params: max_workers <= 18 for parallelization
# Using 16 workers provides good parallelism while staying safely above threshold
auto_max_workers = 18
_cpu_count = os.cpu_count() or 1
logging.info("Using fixed max_workers=%d (CPU cores available: %d)", auto_max_workers, _cpu_count)

probing_config = ProbingConfig(probing_mode='multidim')

# Enable per-run generation-detail CSV output with search bounds
generation_detail_config = GenerationDetailConfig(
    enable=True,
    output_dir=output_dir,
    prefix="run",
)

result = run_global_search(
    camfile_dir=camfile_dir,
    obs_csv_path=obs_csv_path,
    wand_length=10.0,
    n_runs=6,
    max_evals_per_run=5000,
    max_generations=300,
    popsize=16,
    dist_coeff_num=0,
    seed_base=1234,
    probing_config=probing_config,
    # === Parallelization Configuration ===
    # Enable inter-run CMA-ES parallelization (multiple runs in parallel)
    enable_parallel=True,
    # Use fixed worker count (tuned to meet parallelization threshold: n_params >= 2 * max_workers)
    max_workers=auto_max_workers,
    # No worker timeout (workers can run as long as needed)
    worker_timeout=float('inf'),
    # === Generation Detail CSV Configuration ===
    # Enable per-run CSV files with search bounds (min/max per parameter per generation)
    generation_detail_config=generation_detail_config,
)

paths = emit_diagnostics(
    result,
    output_dir=output_dir,
    prefix="full_global",
    include_all_runs=True,
    generation_detail_config=generation_detail_config,
)

print("best objective:", result.best_objective)
print("best diagnostics:", result.best_diagnostics)
print("\nArtifacts:")
for k, v in paths.items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for rid, path in v.items():
            print(f"    run {rid}: {path}")
    else:
        print(f"  {k}: {v}")

print("\nSearch bounds saved in per-run CSV files:")
for run_id in sorted([r.run_id for r in result.runs]):
    print(f"  run{run_id:03d}_search_bounds.csv (or run{run_id:03d}_detail.csv)")

