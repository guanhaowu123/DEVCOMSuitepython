# flowsig_from_prior.py
from __future__ import annotations
from typing import Optional, Literal, Tuple, List, Dict
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
import scanpy as sc
import flowsig as fs

# ----------------------------- #
# Utility: extract dense expression (supports sparse)
# ----------------------------- #
def _dense_expr(adata: AnnData, genes: List[str] | str) -> np.ndarray:
    X = adata[:, genes].X
    if issparse(X):
        X = X.toarray()
    return np.asarray(X)

# ----------------------------- #
# Parse L-R-TF prior (tolerant to column names)
# ----------------------------- #
def _parse_lr_tf_prior(prior_df: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    cols = {c.lower(): c for c in prior_df.columns}
    L_cols = [cols[k] for k in ['l1', 'l2', 'l3', 'l4'] if k in cols]   # allow multiple L columns
    R_cols = [cols[k] for k in ['r1', 'r2', 'r3', 'r4', 'r5'] if k in cols]
    tf_col = cols.get('tf')
    if not L_cols or not R_cols or tf_col is None:
        raise ValueError("The prior CSV must contain at least: L1 (optional L2+), R1 (optional R2+), and tf columns.")

    lig_set, rec_set = set(), set()
    interactions_by_lig: Dict[str, List[str]] = {}
    tfs_by_rec: Dict[str, List[str]] = {}

    for _, row in prior_df.iterrows():
        lig_units = [str(row[c]).strip() for c in L_cols if pd.notnull(row[c]) and str(row[c]).strip() != ""]
        rec_units = [str(row[c]).strip() for c in R_cols if pd.notnull(row[c]) and str(row[c]).strip() != ""]
        tf = str(row[tf_col]).strip() if pd.notnull(row[tf_col]) else None
        if not lig_units or not rec_units:
            continue
        lig = "+".join(lig_units)
        rec = "+".join(rec_units)
        lig_set.add(lig); rec_set.add(rec)
        interactions_by_lig.setdefault(lig, []).append(f"{lig} - {rec}")
        if tf:
            if tf not in tfs_by_rec.setdefault(rec, []):
                tfs_by_rec[rec].append(tf)

    return sorted(lig_set), sorted(rec_set), interactions_by_lig, tfs_by_rec

# ----------------------------- #
# Construct outflow (geometric mean)
# ----------------------------- #
def construct_outflow_signals_from_prior(adata: AnnData, prior_df: pd.DataFrame) -> Tuple[AnnData, List[str]]:
    vars_set = set(adata.var_names)
    lig_complexes, _, interactions_by_lig, _ = _parse_lr_tf_prior(prior_df)
    lig_complexes = [lig for lig in lig_complexes if all((u in vars_set) for u in lig.split("+"))]
    if not lig_complexes:
        raise ValueError("No valid ligand complexes found in the expression matrix (all subunits must be in var_names).")

    split_ligs = [lig.split("+") for lig in lig_complexes]
    unique_units = sorted({u for units in split_ligs for u in units})
    lig_expr = _dense_expr(adata, unique_units)
    idx = {g: i for i, g in enumerate(unique_units)}

    logE = np.log(lig_expr + 1e-12)
    out_X = np.empty((adata.n_obs, len(lig_complexes)))
    for k, units in enumerate(split_ligs):
        cols = [idx[u] for u in units]
        out_X[:, k] = np.exp(logE[:, cols].mean(axis=1))

    out = AnnData(X=out_X)
    out.var.index = pd.Index(lig_complexes)
    out.var["Type"] = "outflow"
    out.var["Downstream_TF"] = ""
    out.var["Interaction"] = ['/'.join(sorted(set(interactions_by_lig.get(l, [])))) for l in lig_complexes]
    return out, lig_complexes

# ----------------------------- #
# Construct inflow (two construction modes)
# ----------------------------- #
def construct_inflow_signals_from_prior(
    adata: AnnData,
    prior_df: pd.DataFrame,
    *,
    tfs_to_use: Optional[List[str]] = None,
    construction: Literal["v1", "v2"] = "v1"
) -> Tuple[AnnData, List[str]]:
    vars_set = set(adata.var_names)
    _, rec_complexes, _, tfs_by_rec = _parse_lr_tf_prior(prior_df)
    rec_complexes = [rec for rec in rec_complexes if all((u in vars_set) for u in rec.split("+"))]
    if not rec_complexes:
        raise ValueError("No valid receptor complexes found in the expression matrix.")

    rec2tfs: Dict[str, List[str]] = {}
    for rec in rec_complexes:
        tfs = [t for t in dict.fromkeys(tfs_by_rec.get(rec, [])) if t in vars_set]
        if tfs_to_use is not None:
            tfs = [t for t in tfs if t in set(tfs_to_use)]
        rec2tfs[rec] = tfs

    split_recs = [rec.split("+") for rec in rec_complexes]
    unique_units = sorted({u for units in split_recs for u in units})
    rec_expr = _dense_expr(adata, unique_units)
    r_idx = {g: i for i, g in enumerate(unique_units)}

    unique_tfs = sorted({t for ts in rec2tfs.values() for t in ts})
    tf_expr = _dense_expr(adata, unique_tfs) if unique_tfs else None
    tf_idx = {g: i for i, g in enumerate(unique_tfs)} if unique_tfs else {}

    inflow_X = np.empty((adata.n_obs, len(rec_complexes)))

    if construction == "v1":
        logR = np.log(rec_expr + 1e-12)
        for k, units in enumerate(split_recs):
            cols = [r_idx[u] for u in units]
            base = np.exp(logR[:, cols].mean(axis=1))  # geometric mean
            tfs = rec2tfs[rec_complexes[k]]
            if tfs:
                tcols = [tf_idx[t] for t in tfs]
                base *= tf_expr[:, tcols].mean(axis=1)
            inflow_X[:, k] = base
    else:  # v2
        for k, units in enumerate(split_recs):
            cols = [r_idx[u] for u in units]
            rec_min = rec_expr[:, cols].min(axis=1)
            tfs = rec2tfs[rec_complexes[k]]
            if tfs:
                tcols = [tf_idx[t] for t in tfs]
                tf_max = tf_expr[:, tcols].max(axis=1)
                inflow_X[:, k] = np.sqrt(rec_min * tf_max)
            else:
                inflow_X[:, k] = rec_min

    # Record receptor→TF and all possible L-R interactions (for traceability)
    R_cols = [c for c in prior_df.columns if c.upper().startswith("R")]
    L_cols = [c for c in prior_df.columns if c.upper().startswith("L")]
    interactions = []
    downstream_tf_str = []
    for rec in rec_complexes:
        tmp = prior_df.copy()
        tmp["REC_COMPLEX"] = tmp[R_cols].apply(
            lambda r: '+'.join(
                [str(x).strip() for x in r if pd.notnull(x) and str(x).strip() != '']
            ),
            axis=1,
        )
        tmp = tmp[tmp["REC_COMPLEX"] == rec]
        ligs = tmp[L_cols].apply(
            lambda r: '+'.join(
                [str(x).strip() for x in r if pd.notnull(x) and str(x).strip() != '']
            ),
            axis=1,
        )
        interactions.append('/'.join(sorted({f"{l} - {rec}" for l in ligs if l})))
        downstream_tf_str.append('_'.join(rec2tfs[rec]))

    inflow = AnnData(X=inflow_X)
    inflow.var.index = pd.Index(rec_complexes)
    inflow.var["Type"] = "inflow"
    inflow.var["Downstream_TF"] = downstream_tf_str
    inflow.var["Interaction"] = interactions
    return inflow, rec_complexes

# ----------------------------- #
# Construct GEM (reusing FlowSig logic)
# ----------------------------- #
def construct_gem_expressions_from_Xgem(
    adata: AnnData,
    gem_expr_key: str = "X_gem",
    scale_gem_expr: bool = True,
    layer_key_for_scale: Optional[str] = None
) -> Tuple[AnnData, List[str]]:
    if gem_expr_key not in adata.obsm:
        raise KeyError(
            f"{gem_expr_key} is not found in adata.obsm. "
            "Please construct GEM using pyliger/NSF (or similar) first."
        )

    gem = np.asarray(adata.obsm[gem_expr_key])
    gem = gem / np.clip(gem.sum(axis=0, keepdims=True), 1e-12, None)

    names = [f"GEM-{i+1}" for i in range(gem.shape[1])]
    ad_gem = AnnData(X=gem)
    ad_gem.var.index = pd.Index(names)
    ad_gem.var["Type"] = "module"
    ad_gem.var["Downstream_TF"] = ""
    ad_gem.var["Interaction"] = ""

    if scale_gem_expr:
        if layer_key_for_scale is not None:
            X = adata.layers[layer_key_for_scale]
            totals = X.sum(axis=1)
            if issparse(totals):
                totals = np.asarray(totals).ravel()
            scale_factor = float(np.mean(totals))
        else:
            X = adata.X
            if issparse(X):
                X = X.toarray()
            scale_factor = float(np.expm1(X).sum(1).mean())
        ad_gem.X *= scale_factor
        sc.pp.log1p(ad_gem)

    return ad_gem, names

# ----------------------------- #
# Wrapper: build X_flow + variable annotations from prior CSV
# ----------------------------- #
def construct_flows_from_prior_csv(
    adata: AnnData,
    prior_csv_path: str,
    *,
    tfs_to_use: Optional[List[str]] = None,
    construction: Literal["v1", "v2"] = "v1",
    gem_expr_key: str = "X_gem",
    scale_gem_expr: bool = True,
    flowsig_expr_key: str = "X_flow",
    flowsig_network_key: str = "flowsig_network",
    layer_key_for_scale: Optional[str] = None
) -> None:
    prior_df = pd.read_csv(prior_csv_path)
    ad_out, _ = construct_outflow_signals_from_prior(adata, prior_df)
    ad_in,  _ = construct_inflow_signals_from_prior(
        adata,
        prior_df,
        tfs_to_use=tfs_to_use,
        construction=construction,
    )
    ad_gem, _ = construct_gem_expressions_from_Xgem(
        adata,
        gem_expr_key=gem_expr_key,
        scale_gem_expr=scale_gem_expr,
        layer_key_for_scale=layer_key_for_scale,
    )

    X_flow = np.hstack([ad_out.X, ad_in.X, ad_gem.X])
    var_names = ad_out.var_names.tolist() + ad_in.var_names.tolist() + ad_gem.var_names.tolist()
    flow_var_info = pd.DataFrame(
        {
            'Type': pd.concat([ad_out.var['Type'], ad_in.var['Type'], ad_gem.var['Type']]),
            'Downstream_TF': pd.concat(
                [ad_out.var['Downstream_TF'], ad_in.var['Downstream_TF'], ad_gem.var['Downstream_TF']]
            ),
            'Interaction': pd.concat(
                [ad_out.var['Interaction'], ad_in.var['Interaction'], ad_gem.var['Interaction']]
            ),
        },
        index=pd.Index(var_names),
    )

    adata.obsm[flowsig_expr_key] = X_flow
    adata.uns[flowsig_network_key] = {'flow_var_info': flow_var_info}

# ----------------------------- #
# One-shot FlowSig pipeline (optionally construct GEM first)
# ----------------------------- #
def run_flowsig_pipeline(
    adata: AnnData,
    prior_csv: str,
    *,
    condition_key: str,
    control_name: str,
    construct_gems: bool = True,
    n_gems: int = 20,
    counts_layer: Optional[str] = "counts",
    tfs_to_use: Optional[List[str]] = None,
    inflow_construction: Literal["v1","v2"] = "v1",
    edge_threshold: float = 0.7,
    flowsig_expr_key: str = "X_flow",
    flowsig_network_key: str = "flowsig_network",
    save_h5ad: Optional[str] = None
) -> AnnData:
    # 1) Optional: construct GEM
    if construct_gems:
        fs.pp.construct_gems_using_pyliger(
            adata,
            n_gems=n_gems,
            layer_key=counts_layer,
            condition_key=condition_key,
        )

    # 2) Assemble X_flow
    construct_flows_from_prior_csv(
        adata,
        prior_csv_path=prior_csv,
        tfs_to_use=tfs_to_use,
        construction=inflow_construction,
        gem_expr_key="X_gem",
        scale_gem_expr=True,
        flowsig_expr_key=flowsig_expr_key,
        flowsig_network_key=flowsig_network_key,
        layer_key_for_scale=counts_layer,
    )

    # 3) Variable selection
    fs.pp.determine_informative_variables(
        adata,
        spatial=False,
        condition_key=condition_key,
        control=control_name,
        qval_threshold=0.05,
        logfc_threshold=0.5,
    )

    # 4) Learn causal network
    fs.tl.learn_intercellular_flows(
        adata,
        condition_key=condition_key,
        control=control_name,
        use_spatial=False,
        n_jobs=1,
        n_bootstraps=50,
    )

    # 5) Biological direction check + low-confidence edge filtering
    fs.tl.apply_biological_flow(
        adata,
        flowsig_network_key=flowsig_network_key,
        adjacency_key="adjacency",
        validated_key="validated",
    )
    fs.tl.filter_low_confidence_edges(
        adata,
        edge_threshold=edge_threshold,
        flowsig_network_key=flowsig_network_key,
        adjacency_key="adjacency_validated",
        filtered_key="filtered",
    )

    if save_h5ad:
        adata.write(save_h5ad, compression="gzip")
    return adata
