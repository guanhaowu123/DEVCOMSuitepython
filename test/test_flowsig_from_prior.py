
import scanpy as sc
#from flowsig_from_prior import run_flowsig_pipeline

adata = sc.read("/home/wgh/DEVCOM Suite test/tests/testthat/flow/burkhardt21_merged.h5ad")

run_flowsig_pipeline(
    adata,
    prior_csv       = "/home/wgh/DEVCOM Suite test/tests/testthat/flow/human_L-R-TF_develop_filtered_LR_TF_top5000.csv",
    condition_key   = "Condition",  
    control_name    = "Ctrl",
    construct_gems  = True,
    n_gems          = 10,
    counts_layer    = "counts",
    inflow_construction = "v1",
    edge_threshold  = 0.7,
    save_h5ad       = "/home/wgh/DEVCOM Suite test/tests/testthat/flow/burkhardt21_merged_flow_learned.h5ad"
)
