# gene_importance.py
from multiprocessing import Pool, cpu_count
import pandas as pd
import networkx as nx
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import os
import math


# ==============================
# Compute PCC and p-values for gene pairs
# ==============================
def calculate_pcc(expression_matrix, genes):
    """
    Sequentially compute PCC and p-values for all gene pairs.
    """
    correlation_matrix = pd.DataFrame(index=genes, columns=genes, dtype=float)
    p_value_matrix = pd.DataFrame(index=genes, columns=genes, dtype=float)

    for i, gene1 in enumerate(genes):
        for j, gene2 in enumerate(genes):
            if j >= i:  # avoid duplicate computations
                if gene1 != gene2:
                    r, p = pearsonr(expression_matrix.loc[gene1], expression_matrix.loc[gene2])
                else:
                    r, p = 1.0, 0.0  # self-correlation
                correlation_matrix.loc[gene1, gene2] = r
                correlation_matrix.loc[gene2, gene1] = r
                p_value_matrix.loc[gene1, gene2] = p
                p_value_matrix.loc[gene2, gene1] = p

    return correlation_matrix, p_value_matrix


# ==============================
# Single cell type: compute gene importance + log10 column
# ==============================
def calculate_gene_importance_for_cell_type(
    cell_type,
    input_dir,
    output_dir,
    expr_suffix="_gene_expression.csv",
    threshold_pcc=0.8,
    threshold_p_value=0.05,
    lambda_factor=0.1,
    add_log10=True,
    verbose=True,
):
    """
    Compute gene importance scores for a single cell type and save to CSV.

    Returns
    -------
    cell_type, result_df
    """
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(input_dir, f"{cell_type}{expr_suffix}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expression matrix for {cell_type} not found: {file_path}")

    if verbose:
        print(f"[{cell_type}] Reading expression matrix: {file_path}")

    expression_matrix = pd.read_csv(file_path, index_col=0)
    expression_matrix = expression_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Remove genes that are zero across all samples
    if verbose:
        print(f"[{cell_type}] Filtering genes with zero expression across all samples...")
    expression_matrix = expression_matrix.loc[~(expression_matrix == 0).all(axis=1)]

    # If no genes remain, skip this cell type
    if expression_matrix.empty:
        print(f"[{cell_type}] No genes passed the filter; skipping this cell type...")
        return cell_type, pd.DataFrame(columns=["Gene", "Importance_Score"])

    genes = expression_matrix.index

    # Sequentially compute PCC and p-values
    if verbose:
        print(f"[{cell_type}] Computing PCC and p-values...")
    correlation_matrix, p_value_matrix = calculate_pcc(expression_matrix, genes)

    # Multiple testing correction (Benjamini-Hochberg FDR)
    if verbose:
        print(f"[{cell_type}] Performing multiple testing correction (FDR-BH)...")
    p_values_flat = p_value_matrix.values.flatten()
    valid_indices = ~np.isnan(p_values_flat)
    adjusted_p_values = np.full_like(p_values_flat, np.nan, dtype=np.float64)
    adjusted_p_values[valid_indices] = multipletests(
        p_values_flat[valid_indices], method='fdr_bh'
    )[1]
    adjusted_p_value_matrix = pd.DataFrame(
        adjusted_p_values.reshape(p_value_matrix.shape),
        index=genes,
        columns=genes
    )

    # Build gene network
    if verbose:
        print(f"[{cell_type}] Building gene network...")
    G = nx.Graph()
    G.add_nodes_from(correlation_matrix.index)
    for i in correlation_matrix.index:
        for j in correlation_matrix.columns:
            if (
                i != j
                and abs(correlation_matrix.loc[i, j]) >= threshold_pcc
                and adjusted_p_value_matrix.loc[i, j] <= threshold_p_value
            ):
                G.add_edge(i, j, weight=abs(correlation_matrix.loc[i, j]))

    # Compute gene importance scores (keeping your original formula)
    def _calculate_gene_importance(G, expression_matrix, lambda_factor):
        importance_scores = {}
        max_expression = expression_matrix.mean(axis=1).max()
        if max_expression == 0:
            max_expression = 1.0  # avoid division by zero

        closeness_centrality = nx.closeness_centrality(G, distance="weight")
        for v in G.nodes():
            deg_v = G.degree(v)
            expression_v = expression_matrix.loc[v].mean()
            w_v = (deg_v * expression_v) / max_expression
            C_v = set(nx.node_connected_component(G, v))
            V_size_ratio = len(C_v) / len(G.nodes())
            closeness_sum = 0
            for u in G.nodes():
                if u != v and nx.has_path(G, v, u):
                    closeness_sum += 1 / nx.shortest_path_length(
                        G, source=v, target=u, weight="weight"
                    )
            mcc_sum = 0
            for clique in nx.find_cliques(G):
                if v in clique and closeness_centrality[v] != 0:
                    mcc_sum += math.factorial(len(clique) - 1) / closeness_centrality[v]
            score_v = w_v * V_size_ratio * closeness_sum + lambda_factor * mcc_sum
            importance_scores[v] = score_v
        return importance_scores

    if verbose:
        print(f"[{cell_type}] Computing gene importance scores...")
    importance_scores = _calculate_gene_importance(G, expression_matrix, lambda_factor)

    # For genes with score = 0, replace with their mean expression
    for gene, score in list(importance_scores.items()):
        if score == 0:
            importance_scores[gene] = float(expression_matrix.loc[gene].mean())

    # Build output DataFrame
    result_df = pd.DataFrame(importance_scores.items(), columns=["Gene", "Importance_Score"])

    # Same as your original: log10(Score + 1), which avoids negative values
    if add_log10:
        vals = result_df["Importance_Score"].astype(float).values
        result_df["Log10_Importance_Score"] = np.log10(vals + 1.0)

    # Save CSV (one file contains both raw scores + log10 scores)
    output_file = os.path.join(output_dir, f"{cell_type}_gene_importance_scores.csv")
    result_df.to_csv(output_file, index=False)
    if verbose:
        print(f"[{cell_type}] Gene importance scores saved to: {output_file}")

    return cell_type, result_df


# ==============================
# Batch wrapper: run multiple cell types (supports parallel)
# ==============================
def compute_gene_importance_all(
    cell_types,
    input_dir,
    output_dir,
    expr_suffix="_gene_expression.csv",
    threshold_pcc=0.8,
    threshold_p_value=0.05,
    lambda_factor=0.1,
    n_jobs=None,
    verbose=True,
):
    """
    Batch-compute gene importance scores for multiple cell types.
    Returns a dict {cell_type: DataFrame}.

    For each cell type, writes to output_dir:
        {cell_type}_gene_importance_scores.csv
    which contains:
        - Importance_Score
        - Log10_Importance_Score
    """
    os.makedirs(output_dir, exist_ok=True)

    if n_jobs is None:
        n_jobs = min(cpu_count(), max(1, len(cell_types)))

    args_list = [
        (
            cell_type,
            input_dir,
            output_dir,
            expr_suffix,
            threshold_pcc,
            threshold_p_value,
            lambda_factor,
            True,      # add_log10
            verbose,
        )
        for cell_type in cell_types
    ]

    results = {}

    if verbose:
        print(f"Starting computation for all cell types: {cell_types}")
        print(f"Number of processes: {n_jobs}")

    # Serial vs parallel modes
    if n_jobs == 1 or len(cell_types) == 1:
        for args in args_list:
            ct, df = calculate_gene_importance_for_cell_type(*args)
            results[ct] = df
    else:
        with Pool(processes=n_jobs) as pool:
            for ct, df in pool.starmap(calculate_gene_importance_for_cell_type, args_list):
                results[ct] = df

    return results
