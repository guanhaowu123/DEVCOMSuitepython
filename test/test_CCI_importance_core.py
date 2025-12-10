
import scanpy as sc
from DEVCOMSuitepython.CCI_importance_core import compute_gene_importance_all

cell_types = ["Endothelial", "Erythrocyte"]

res = compute_gene_importance_all(
    cell_types=cell_types,
    input_dir="/home/wgh/DEVCOM Suite test/tests/testthat/gene importance/",
    output_dir="/home/wgh/DEVCOM Suite test/tests/testthat/gene importance/",
    expr_suffix="_gene_expression.csv", 
    threshold_pcc=0.8,
    threshold_p_value=0.05,
    lambda_factor=0.1,
    n_jobs=None,   # None=auto
    verbose=True
)

