import os
import scanpy as sc

# 1) Load SURD first (very important)
#    Here use the path to your local SURD-main/utils directory
load_surd_from_dir("/home/wgh/SOUD/SURD-main/utils")

# 2) Path settings
DATA_DIR = "/home/wgh/DEVCOM Suite test/tests/testthat/SURD/"

H5AD_OUT = os.path.join(
    DATA_DIR,
    "burkhardt21_merged_SURD.h5ad"
)

# Output directory of the main SURD pipeline
# (should already contain gem2out_unique_synergy_redundancy.csv)
SEGMENTS_DIR = os.path.join(
    DATA_DIR,
    "surd_segments_results"
)

# IMPORTANT: only join the file name here, do NOT prepend an absolute path again
parent_csv_path = os.path.join(
    SEGMENTS_DIR,
    "gem2out_unique_synergy_redundancy.csv"
)

# 3) Load h5ad
adata = sc.read(H5AD_OUT)

# 4) Run K sweep (post-hoc)
k_df, best_row = run_surd_k_sweep(
    adata,
    ctrl_name="Ctrl",
    treat_name="IFNg",
    parent_csv_path=parent_csv_path,
    out_dir=SEGMENTS_DIR,          # K sweep results will also be written here
    k_range=range(1, 21),
    top_parents_per_target=8,
    n_perm_small=50,
    nbins_target=16,
    min_bins_per_dim=6,
    max_total_bins=20_000_000,
    discretize="quantile",
    min_samples_block=100,
    make_plots=True,
)

print("K sweep finished. Results written to:",
      os.path.join(SEGMENTS_DIR, "k_sweep_summary.csv"))

if best_row is not None:
    print("Best K:", int(best_row["K"]), "score =", float(best_row["score"]))
else:
    print("No valid blocks found, k_df is empty. "
          "Please check the parent CSV or condition column.")
