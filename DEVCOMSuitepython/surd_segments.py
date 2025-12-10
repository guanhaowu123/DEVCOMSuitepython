from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors

try:
    from anndata import AnnData
except Exception:  # allow import even if anndata not installed at lint time
    AnnData = object  # type: ignore


# ===================== Utilities: find adjacency matrices in `uns` =====================

def _to_array_if_square(x):
    """
    Try to convert an object to a square 2D numpy array.
    """
    try:
        arr = x.A if hasattr(x, "A") else np.asarray(x)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] >= 2:
            return arr
    except Exception:
        return None
    return None


def _scan_uns_for_square_mats(container, prefix: str = ""):
    """
    Recursively search for 2D square matrices in a nested dict/list/tuple.
    """
    found = []
    if isinstance(container, dict):
        for k, v in container.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                found.extend(_scan_uns_for_square_mats(v, path))
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    arr = _to_array_if_square(item)
                    if arr is not None:
                        found.append((f"{path}[{i}]", arr))
            else:
                arr = _to_array_if_square(v)
                if arr is not None:
                    found.append((path, arr))
    else:
        arr = _to_array_if_square(container)
        if arr is not None:
            found.append((prefix or "<root>", arr))
    return found


def _get_flowsig_adjacency(
    adata,
    flowsig_network_key: str,
    adjacency_key_candidates,
    var_names,
    flow_key_hint: str | None = None,
):
    """
    Try to find the FlowSig adjacency matrix in adata.uns.
    """
    p = len(var_names)
    candidates = []

    if flowsig_network_key in adata.uns and isinstance(adata.uns[flowsig_network_key], dict):
        net = adata.uns[flowsig_network_key]
        candidates.extend(_scan_uns_for_square_mats(net, flowsig_network_key))

        for k in adjacency_key_candidates:
            try:
                v = net[k]
                arr = _to_array_if_square(v)
                if arr is not None:
                    candidates.append((f"{flowsig_network_key}.{k}", arr))
            except Exception:
                pass

    if not candidates:
        candidates.extend(_scan_uns_for_square_mats(adata.uns, ""))

    if not candidates:
        raise KeyError("No 2D square adjacency candidates found in adata.uns.")

    def _score(path: str, arr: np.ndarray) -> float:
        s = 0.0
        s -= abs(arr.shape[0] - p)
        pl = path.lower().replace(" ", "")
        if "flow" in pl:
            s += 2
        if flowsig_network_key.lower().replace(" ", "") in pl:
            s += 2
        if "causal" in pl or "learned" in pl:
            s += 1
        if "filtered" in pl or "validated" in pl:
            s += 1
        if flow_key_hint and flow_key_hint.lower().replace(" ", "") in pl:
            s += 1
        return s

    scored = sorted(
        [(_score(path, arr), path, arr) for path, arr in candidates],
        key=lambda z: (-z[0], z[1]),
    )
    best_score, best_path, best_arr = scored[0]

    if best_arr.shape[0] != p:
        m = min(best_arr.shape[0], p)
        best_arr = best_arr[:m, :m]

    print(f"[SURD] picked adjacency from uns path: {best_path}, shape={best_arr.shape}")
    return best_arr, best_path


# ===================== SURD import =====================

try:
    from surd import surd  # type: ignore
except Exception:
    surd = None  # type: ignore


# def load_surd_from_dir(utils_dir: str, module_name: str = "surd"):
#     """
#     Add the SURD utils directory to sys.path and import surd().
#     """
#     import sys
#     import importlib

#     utils_dir = os.path.abspath(utils_dir)
#     if utils_dir not in sys.path:
#         sys.path.insert(0, utils_dir)
#     mod = importlib.import_module(module_name)
#     globals()["surd"] = getattr(mod, "surd")
#     return mod

def load_surd_from_dir(utils_dir: str, module_name: str = "DEVCOMSuitepython.utils.surd"):
    import importlib
    mod = importlib.import_module(module_name)
    globals()["surd"] = getattr(mod, "surd")
    return mod

# ===================== Common key candidates =====================

CONDITION_KEY_CANDIDATES = ["Condition", "condition", "cond", "group", "stim", "Treatment"]
CELLTYPE_KEY_CANDIDATES = [
    "cell_type",
    "CellType",
    "celltype",
    "celltype_major",
    "MajorType",
    "celltype_final",
    "Type",
]
FLOW_X_OBSM_CANDIDATES = ["X_flow", "X_flow_cpdb", "X_flow_orig", "flowsig_X", "X_pca", "X_umap"]


def _pick_first_existing_key(obj, candidates: List[str]):
    for k in candidates:
        if k in obj:
            return k, obj[k]
    return None, None


def _get_condition_key(adata) -> str:
    k, _ = _pick_first_existing_key(adata.obs, CONDITION_KEY_CANDIDATES)
    if k is None:
        raise KeyError(f"Cannot find condition column: {CONDITION_KEY_CANDIDATES}")
    return k


def _get_celltype_key(adata) -> Optional[str]:
    k, _ = _pick_first_existing_key(adata.obs, CELLTYPE_KEY_CANDIDATES)
    return k


def _get_flow_X_and_names(
    adata, flowsig_network_key: str = "flowsig_network"
) -> Tuple[str, np.ndarray, List[str], List[str]]:
    k, X = _pick_first_existing_key(adata.obsm, FLOW_X_OBSM_CANDIDATES)
    if k is None:
        raise KeyError(f"No X_flow candidates found in obsm: {FLOW_X_OBSM_CANDIDATES}")
    X = np.asarray(X)

    var_names: Optional[List[str]] = None
    var_types: Optional[List[str]] = None

    if flowsig_network_key in adata.uns and isinstance(adata.uns[flowsig_network_key], dict):
        net = adata.uns[flowsig_network_key]
        if "flow_var_info" in net and hasattr(net["flow_var_info"], "index"):
            df = net["flow_var_info"]
            var_names = list(map(str, df.index.tolist()))
            if "Type" in df.columns:
                var_types = df["Type"].astype(str).tolist()

    if var_names is None:
        for k2 in ["flow_var_names", "flow_variables", "flowsig_var_names", "var_names_flow"]:
            if k2 in adata.uns and isinstance(adata.uns[k2], (list, np.ndarray)):
                var_names = list(map(str, adata.uns[k2]))
                break

    if var_names is None:
        var_names = [f"VAR_{i}" for i in range(X.shape[1])]
    if var_types is None:
        var_types = [""] * len(var_names)

    return k, X, var_names, var_types


# ===================== Standardization + histogram =====================

def _standardize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd


def _compute_hist_for_surd(
    y_future: np.ndarray,
    X_past: np.ndarray,
    nbins_target: int = 16,
    max_total_bins: int = 20_000_000,
    min_bins_per_dim: int = 6,
    mode: str = "quantile",
):
    y_future = np.asarray(y_future).reshape(-1, 1).astype(np.float32, copy=False)
    X_past = np.asarray(X_past).astype(np.float32, copy=False)
    assert y_future.shape[0] == X_past.shape[0]
    data = np.hstack([_standardize(y_future), _standardize(X_past)])
    n, d = data.shape

    per_dim = int(min(max(min_bins_per_dim, nbins_target), max_total_bins ** (1.0 / max(d, 1))))
    while (per_dim**d) > max_total_bins and per_dim > min_bins_per_dim:
        per_dim -= 1

    edges: List[np.ndarray] = []
    for i in range(d):
        x = data[:, i]
        x = x[np.isfinite(x)]
        if x.size == 0:
            e = np.linspace(-1, 1, per_dim + 1)
        elif mode == "quantile":
            qs = np.linspace(0, 1, per_dim + 1)
            e = np.quantile(x, qs)
            e = np.unique(e)
            if e.size < 3:
                lo, hi = np.min(x), np.max(x)
                if lo == hi:
                    lo, hi = lo - 1e-6, hi + 1e-6
                e = np.linspace(lo, hi, per_dim + 1)
        else:
            lo, hi = np.nanpercentile(x, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = np.min(x), np.max(x)
                if lo == hi:
                    lo, hi = lo - 1e-6, hi + 1e-6
            e = np.linspace(lo, hi, per_dim + 1)
        edges.append(e)

    hist, _ = np.histogramdd(data, bins=edges)
    hist = hist.astype(np.float64, copy=False)
    hist += 1e-14
    hist /= hist.sum()
    return hist


def _surd_decompose(
    y_future: np.ndarray,
    X_past: np.ndarray,
    nbins_target: int = 16,
    max_total_bins: int = 20_000_000,
    min_bins_per_dim: int = 6,
    mode: str = "quantile",
):
    if surd is None:
        raise RuntimeError("surd() not found. Call load_surd_from_dir() or `from surd import surd` first.")
    p = _compute_hist_for_surd(y_future, X_past, nbins_target, max_total_bins, min_bins_per_dim, mode)
    return surd(p)  # type: ignore


# ===================== Graph and parent set =====================

def _build_graph_from_adjacency(A: np.ndarray, var_names: List[str], thr: float = 1e-12) -> nx.DiGraph:
    A = np.asarray(A)
    n = min(A.shape[0], len(var_names))
    A = A[:n, :n]
    names = var_names[:n]
    G = nx.DiGraph()
    for i, nm in enumerate(names):
        G.add_node(nm, idx=i)
    rs, cs = np.where(A > thr)
    for i, j in zip(rs, cs):
        if i != j:
            G.add_edge(names[i], names[j], weight=float(A[i, j]))
    return G


def _parents_of(G: nx.DiGraph, tgt: str) -> List[str]:
    return list(G.predecessors(tgt))


# ===================== KNN matching & Δ construction =====================

def _knn_match_ctrl_for_each_treat(
    adata,
    condition_key: str,
    z_key: str,
    ctrl_name: str,
    treat_name: str,
    celltype_key: Optional[str] = None,
    k: int = 13,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    Z = adata.obsm[z_key].astype(np.float32, copy=False)
    cond = adata.obs[condition_key].astype(str).values

    if celltype_key is not None and celltype_key in adata.obs:
        ctype = adata.obs[celltype_key].astype(str).values
        groups = pd.Series(ctype).unique()
    else:
        ctype = np.array(["ALL"] * adata.n_obs)
        groups = np.array(["ALL"])

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for g in groups:
        mask_t = (cond == treat_name) & (ctype == g)
        mask_c = (cond == ctrl_name) & (ctype == g)
        if mask_t.sum() == 0 or mask_c.sum() == 0:
            continue
        Zt, Zc = Z[mask_t], Z[mask_c]
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        nn.fit(Zc)
        _, idx = nn.kneighbors(Zt, return_distance=True)

        t_idx = np.where(mask_t)[0]
        c_base = np.where(mask_c)[0]
        out[g] = (t_idx, c_base[idx])
    return out


def _build_delta_pairs(
    X_flow: np.ndarray,
    tgt_idx: int,
    parent_idx: List[int],
    match_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    Yd, Xd = [], []
    for _, (t_idx, ctrl_idx_matched) in match_dict.items():
        if t_idx.size == 0 or ctrl_idx_matched.size == 0:
            continue
        K = ctrl_idx_matched.shape[1]
        y_t = X_flow[t_idx, tgt_idx]
        y_c = X_flow[ctrl_idx_matched.reshape(-1), tgt_idx].reshape(t_idx.size, K).mean(axis=1)
        X_t = X_flow[t_idx][:, parent_idx]
        X_c = (
            X_flow[ctrl_idx_matched.reshape(-1)][:, parent_idx]
            .reshape(t_idx.size, K, len(parent_idx))
            .mean(axis=1)
        )
        Yd.append((y_t - y_c).reshape(-1, 1))
        Xd.append(X_t - X_c)

    if not Yd:
        return None, None
    return np.vstack(Yd).ravel(), np.vstack(Xd)


# ===================== SURD block & parent table =====================

def _run_surd_block(
    X_flow: np.ndarray,
    match_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    tgt_name: str,
    parent_names: List[str],
    name2col: Dict[str, int],
    nbins_target: int = 16,
    max_total_bins: int = 20_000_000,
    min_bins_per_dim: int = 6,
    mode: str = "quantile",
    min_samples_block: int = 20,
):
    """
    Run SURD for a single target and a given set of parents, and return the information decomposition result.
    """
    if not parent_names:
        return None
    tgt_idx = name2col[tgt_name]
    parent_idx = [name2col[p] for p in parent_names]

    y, X = _build_delta_pairs(X_flow, tgt_idx, parent_idx, match_dict)
    if y is None:
        return None

    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[mask], X[mask]
    if y.size < min_samples_block:
        return None

    I_R, I_S, MI, leak = _surd_decompose(y, X, nbins_target, max_total_bins, min_bins_per_dim, mode)
    sum_info = (sum(I_R.values()) + sum(I_S.values())) or 1.0

    uniq_rows = []
    red_rows = []
    syn_rows = []

    for k, v in I_R.items():
        if isinstance(k, tuple) and len(k) == 1:
            uniq_rows.append((parent_names[k[0] - 1], float(v)))
        elif isinstance(k, tuple) and len(k) >= 2:
            comb = tuple(parent_names[i - 1] for i in k)
            red_rows.append((comb, float(v)))
    for k, v in I_S.items():
        comb = tuple(parent_names[i - 1] for i in k)
        syn_rows.append((comb, float(v)))

    return {
        "I_R": I_R,
        "I_S": I_S,
        "sum_info": sum_info,
        "info_leak": float(leak),
        "uniq_rows": uniq_rows,
        "red_rows": red_rows,
        "syn_rows": syn_rows,
    }


def _parent_table(
    block_res,
    tgt_name: str,
    parent_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if block_res is None:
        return pd.DataFrame()

    sum_info = block_res["sum_info"]
    leak = block_res["info_leak"]
    uniq_rows = block_res["uniq_rows"]
    red_rows = block_res["red_rows"]
    syn_rows = block_res["syn_rows"]

    parents = sorted(
        {p for p, _ in uniq_rows}
        | {x for comb, _ in red_rows for x in comb}
        | {x for comb, _ in syn_rows for x in comb}
    )
    recs = []
    for p in parents:
        u = sum(v for q, v in uniq_rows if q == p)
        s = sum(v for comb, v in syn_rows if p in comb)
        r = sum(v for comb, v in red_rows if p in comb)
        recs.append(
            {
                "target": tgt_name,
                "parent": p,
                "unique_raw": u,
                "unique_norm": u / sum_info,
                "synergy_raw": s,
                "synergy_norm": s / sum_info,
                "redundancy_raw": r,
                "redundancy_norm": r / sum_info,
                "sum_info": sum_info,
                "info_leak": leak,
                "explained_fraction": 1.0 - leak,
                "parent_weight": (parent_weights or {}).get(p, np.nan),
            }
        )
    return pd.DataFrame(recs)


# ===================== Main function: run_surd_segments =====================

def run_surd_segments(
    adata: AnnData,
    *,
    ctrl_name: str,
    treat_name: str,
    out_dir: str,
    flowsig_network_key: str = "flowsig_network",
    adjacency_key_candidates: Tuple[str, ...] = ("adjacency_filtered", "adjacency_validated", "adjacency"),
    fast_mode: bool = True,
    nbins_target: int = 16,
    min_bins_per_dim: int = 6,
    max_total_bins: int = 20_000_000,
    discretize: str = "quantile",
    top_parents_per_step: int = 8,
    max_gems_per_outflow: int = 8,
    per_gem_inflow_cap: int = 5,
    max_parents_combined: int = 10,
    min_samples_block: int = 20,
    do_permutation: bool = True,
    n_perm: int = 60,
    subsample_for_perm: float = 0.7,
):
    """
    Full SURD segment analysis: inflow->GEM, GEM->outflow, inflow+GEM->outflow,
    plus row-shuffle negative control.

    No plotting is done here. The function only writes CSVs and stores results
    into adata.uns["surd"]["segments"].
    """
    os.makedirs(out_dir, exist_ok=True)

    # In fast_mode, downscale some parameters to speed up computation
    if fast_mode:
        nbins_target = min(nbins_target, 12)
        min_bins_per_dim = min(min_bins_per_dim, 5)
        max_total_bins = min(max_total_bins, 5_000_000)
        top_parents_per_step = min(top_parents_per_step, 6)
        max_gems_per_outflow = min(max_gems_per_outflow, 6)
        per_gem_inflow_cap = min(per_gem_inflow_cap, 4)
        max_parents_combined = min(max_parents_combined, 8)
        if n_perm > 60:
            n_perm = 60
        if min_samples_block > 20:
            min_samples_block = 20

    # 1) Extract information from obs/obsm/uns
    cond_key = _get_condition_key(adata)
    ctype_key = _get_celltype_key(adata)

    print(f"[SURD] condition_key = {cond_key}")
    print(f"[SURD] condition value_counts = {adata.obs[cond_key].value_counts().to_dict()}")
    if ctype_key is not None:
        print(f"[SURD] celltype_key = {ctype_key}, n_types = {adata.obs[ctype_key].nunique()}")
    else:
        print("[SURD] celltype_key = None (no stratification)")

    z_key, _ = _pick_first_existing_key(adata.obsm, FLOW_X_OBSM_CANDIDATES)
    if z_key is None:
        raise KeyError("No embedding (X_flow/X_umap etc.) found for matching.")
    flow_key, X_flow, var_names, var_types = _get_flow_X_and_names(adata, flowsig_network_key)
    X_flow = np.asarray(X_flow).astype(np.float32, copy=False)
    name2col = {n: i for i, n in enumerate(var_names)}

    # 2) Adjacency matrix
    if flowsig_network_key not in adata.uns:
        raise KeyError(f"{flowsig_network_key} not found in adata.uns.")
    A, adj_path = _get_flowsig_adjacency(
        adata,
        flowsig_network_key=flowsig_network_key,
        adjacency_key_candidates=adjacency_key_candidates,
        var_names=var_names,
        flow_key_hint=flow_key,
    )

    # 3) Build graph + KNN matching
    G = _build_graph_from_adjacency(A, var_names, thr=1e-12)
    match = _knn_match_ctrl_for_each_treat(
        adata, cond_key, z_key, ctrl_name=ctrl_name, treat_name=treat_name, celltype_key=ctype_key, k=13
    )
    if not match:
        print(
            "[SURD] WARNING: no KNN matches between Ctrl and Treat. "
            "Please check ctrl_name / treat_name / condition column."
        )
    else:
        total_treat_cells = sum(t_idx.size for (t_idx, _) in match.values())
        print(f"[SURD] matched_groups = {len(match)}, total_treat_cells = {total_treat_cells}")

    # ---- Split into inflow / GEM / outflow ----
    inflows, gems, outflows = [], [], []
    for n in var_names:
        t = str(var_types[name2col[n]]).lower()
        if "inflow" in t:
            inflows.append(n)
        elif "outflow" in t:
            outflows.append(n)
        elif "gem" in t or "module" in t:
            gems.append(n)

    print(f"[SURD] inflows={len(inflows)}, gems={len(gems)}, outflows={len(outflows)}")

    total_edges = G.number_of_edges()
    in2gem_edges = sum(1 for g in gems for p in _parents_of(G, g) if p in inflows)
    gem2out_edges = sum(1 for o in outflows for p in _parents_of(G, o) if p in gems)
    print(f"[SURD] graph_edges={total_edges}, inflow->GEM edges={in2gem_edges}, GEM->outflow edges={gem2out_edges}")

    # -------- Segment A: inflow -> GEM --------
    in2gem_tables: List[pd.DataFrame] = []
    for g in gems:
        parents = [p for p in _parents_of(G, g) if p in inflows]
        if not parents:
            continue
        ws = np.array([abs(G.edges[(p, g)]["weight"]) for p in parents])
        order = np.argsort(-ws)[:top_parents_per_step]
        parents = [parents[i] for i in order]
        p_w = {p: float(ws[i]) for i, p in zip(order, parents)}
        blk = _run_surd_block(
            X_flow,
            match,
            g,
            parents,
            name2col,
            nbins_target=nbins_target,
            max_total_bins=max_total_bins,
            min_bins_per_dim=min_bins_per_dim,
            mode=discretize,
            min_samples_block=min_samples_block,
        )
        df = _parent_table(blk, g, parent_weights=p_w)
        if not df.empty:
            df["segment"] = "inflow->GEM"
            in2gem_tables.append(df)

    in2gem_df = pd.concat(in2gem_tables, ignore_index=True) if in2gem_tables else pd.DataFrame()
    if not in2gem_df.empty:
        in2gem_df.to_csv(os.path.join(out_dir, "in2gem_unique_synergy_redundancy.csv"), index=False)
    print(f"[SURD] in2gem_df shape = {in2gem_df.shape}")

    # -------- Segment B: GEM -> outflow --------
    gem2out_tables: List[pd.DataFrame] = []
    out_to_gems: Dict[str, List[str]] = {}
    for o in outflows:
        parents = [p for p in _parents_of(G, o) if p in gems]
        out_to_gems[o] = parents[:]
        if not parents:
            continue
        ws = np.array([abs(G.edges[(p, o)]["weight"]) for p in parents])
        order = np.argsort(-ws)[:top_parents_per_step]
        parents = [parents[i] for i in order]
        p_w = {p: float(ws[i]) for i, p in zip(order, parents)}
        blk = _run_surd_block(
            X_flow,
            match,
            o,
            parents,
            name2col,
            nbins_target=nbins_target,
            max_total_bins=max_total_bins,
            min_bins_per_dim=min_bins_per_dim,
            mode=discretize,
            min_samples_block=min_samples_block,
        )
        df = _parent_table(blk, o, parent_weights=p_w)
        if not df.empty:
            df["segment"] = "GEM->outflow"
            gem2out_tables.append(df)

    gem2out_df = pd.concat(gem2out_tables, ignore_index=True) if gem2out_tables else pd.DataFrame()
    if not gem2out_df.empty:
        gem2out_df.to_csv(os.path.join(out_dir, "gem2out_unique_synergy_redundancy.csv"), index=False)
    print(f"[SURD] gem2out_df shape = {gem2out_df.shape}")

    # -------- Segment C: inflow + GEM -> outflow, plus 3-node paths --------
    inflow_of_gem: Dict[str, List[str]] = {
        g: [p for p in _parents_of(G, g) if p in inflows] for g in gems
    }
    combined_tables: List[pd.DataFrame] = []
    triple_tables: List[pd.DataFrame] = []

    for o in outflows:
        g_full = out_to_gems.get(o, [])
        if not g_full:
            continue

        # GEM trimming
        if len(g_full) > max_gems_per_outflow:
            w_g = np.array([abs(G.edges[(g, o)]["weight"]) for g in g_full])
            keep_idx = np.argsort(-w_g)[:max_gems_per_outflow]
            g_keep = [g_full[i] for i in keep_idx]
        else:
            g_keep = g_full[:]

        # inflow trimming
        in_union: List[str] = []
        for g in g_keep:
            ins = inflow_of_gem.get(g, [])
            if ins:
                w_ins = np.array([abs(G.edges[(i, g)]["weight"]) for i in ins])
                ord_ins = np.argsort(-w_ins)[:per_gem_inflow_cap]
                ins = [ins[i] for i in ord_ins]
            in_union.extend(ins)
        in_union = sorted(set(in_union))

        parents_combined = list(g_keep) + list(in_union)

        # global trimming if too many parents
        if len(parents_combined) > max_parents_combined:
            scores = []
            for p in parents_combined:
                if G.has_edge(p, o):
                    s = abs(G.edges[(p, o)]["weight"])
                elif p in in_union:
                    s = max(
                        (abs(G.edges[(p, g)]["weight"]) for g in g_keep if G.has_edge(p, g)),
                        default=0.0,
                    )
                else:
                    s = 0.0
                scores.append(s)
            order = np.argsort(-np.array(scores))[:max_parents_combined]
            parents_combined = [parents_combined[i] for i in order]

        if not parents_combined:
            continue

        blk = _run_surd_block(
            X_flow,
            match,
            o,
            parents_combined,
            name2col,
            nbins_target=nbins_target,
            max_total_bins=max_total_bins,
            min_bins_per_dim=min_bins_per_dim,
            mode=discretize,
            min_samples_block=min_samples_block,
        )
        if blk is None:
            continue

        parent_w = {p: float(G.edges[(p, o)]["weight"]) for p in parents_combined if G.has_edge(p, o)}
        df_p = _parent_table(blk, o, parent_weights=parent_w)
        if not df_p.empty:
            df_p["segment"] = "inflow+GEM->outflow"
            combined_tables.append(df_p)

        # 3-node path statistics
        sum_info = blk["sum_info"]
        parent_names = parents_combined
        idx2name = {i + 1: parent_names[i] for i in range(len(parent_names))}
        I_Rn = {tuple(idx2name[j] for j in k): v for k, v in blk["I_R"].items()}
        I_Sn = {tuple(idx2name[j] for j in k): v for k, v in blk["I_S"].items()}

        rows = []
        for g in [x for x in parents_combined if x in g_keep]:
            ins = [i for i in in_union if G.has_edge(i, g) and i in parents_combined]
            for i in ins:
                s_raw = sum(v for comb, v in I_Sn.items() if (i in comb and g in comb))
                r_raw = sum(v for comb, v in I_Rn.items() if (len(comb) >= 2 and i in comb and g in comb))
                u_in = I_Rn.get((i,), 0.0)
                u_g = I_Rn.get((g,), 0.0)
                rows.append(
                    {
                        "outflow": o,
                        "gem": g,
                        "inflow": i,
                        "synergy_raw": s_raw,
                        "synergy_norm": s_raw / sum_info,
                        "redundancy_raw": r_raw,
                        "redundancy_norm": r_raw / sum_info,
                        "inflow_unique_raw": u_in,
                        "inflow_unique_norm": u_in / sum_info,
                        "gem_unique_raw": u_g,
                        "gem_unique_norm": u_g / sum_info,
                        "explained_fraction": 1.0 - blk["info_leak"],
                    }
                )
        if rows:
            triple_tables.append(pd.DataFrame(rows))

    combined_df = pd.concat(combined_tables, ignore_index=True) if combined_tables else pd.DataFrame()
    if not combined_df.empty:
        combined_df.to_csv(os.path.join(out_dir, "combined_outflow_unique_synergy_redundancy.csv"), index=False)
    print(f"[SURD] combined_df shape = {combined_df.shape}")

    triple_df = pd.concat(triple_tables, ignore_index=True) if triple_tables else pd.DataFrame()
    if not triple_df.empty:
        triple_df.to_csv(os.path.join(out_dir, "path_triples_synergy_redundancy.csv"), index=False)
    print(f"[SURD] triple_df shape = {triple_df.shape}")

    # ---------- Row-shuffle negative control ----------
    perm_df = pd.DataFrame()
    if do_permutation and (not in2gem_df.empty or not gem2out_df.empty):

        def _perm_test(y: np.ndarray, X: np.ndarray, n_perm: int):
            I_R, I_S, MI, leak = _surd_decompose(y, X, nbins_target, max_total_bins, min_bins_per_dim, discretize)
            obs_u = sum(v for k, v in I_R.items() if isinstance(k, tuple) and len(k) == 1)
            obs_s = sum(I_S.values()) if len(I_S) else 0.0
            obs_e = 1.0 - float(leak)

            null_u = []
            null_s = []
            null_e = []
            for _ in range(n_perm):
                Xp = X[np.random.permutation(X.shape[0])]
                IR0, IS0, _, leak0 = _surd_decompose(y, Xp, nbins_target, max_total_bins, min_bins_per_dim, discretize)
                null_u.append(sum(v for k, v in IR0.items() if isinstance(k, tuple) and len(k) == 1))
                null_s.append(sum(IS0.values()) if len(IS0) else 0.0)
                null_e.append(1.0 - float(leak0))

            nu = np.asarray(null_u)
            ns = np.asarray(null_s)
            ne = np.asarray(null_e)
            pu = (1 + (nu >= obs_u).sum()) / (n_perm + 1)
            ps = (1 + (ns >= obs_s).sum()) / (n_perm + 1)
            return dict(
                obs_unique=obs_u,
                obs_synergy=obs_s,
                obs_explained=obs_e,
                null_unique_mean=float(nu.mean()),
                null_unique_std=float(nu.std(ddof=1) if nu.size > 1 else 0.0),
                null_synergy_mean=float(ns.mean()),
                null_synergy_std=float(ns.std(ddof=1) if ns.size > 1 else 0.0),
                null_explained_mean=float(ne.mean()),
                null_explained_std=float(ne.std(ddof=1) if ne.size > 1 else 0.0),
                p_unique=float(pu),
                p_synergy=float(ps),
            )

        rows_perm = []

        for seg_name, df_seg in [("inflow->GEM", in2gem_df), ("GEM->outflow", gem2out_df)]:
            if df_seg.empty:
                continue
            for tgt, df_t in df_seg.groupby("target"):
                score = (df_t["unique_norm"].fillna(0) + df_t["synergy_norm"].fillna(0))
                df_t = df_t.loc[score.sort_values(ascending=False).index]
                parents = [p for p in df_t["parent"].tolist() if p in name2col][:top_parents_per_step]
                if not parents:
                    continue

                tgt_idx = name2col[tgt]
                par_idx = [name2col[p] for p in parents]
                y, X = _build_delta_pairs(X_flow, tgt_idx, par_idx, match)
                if y is None or y.size < min_samples_block:
                    continue
                if subsample_for_perm < 1.0 and y.size > 500:
                    m = int(y.size * subsample_for_perm)
                    sel = np.random.choice(y.size, m, replace=False)
                    y, X = y[sel], X[sel]
                stats = _perm_test(y, X, n_perm)
                rows_perm.append(dict(segment=seg_name, target=tgt, n_parents=len(parents), **stats))

        perm_df = pd.DataFrame(rows_perm)
        if not perm_df.empty:
            perm_df.to_csv(os.path.join(out_dir, "permtest_rowshuffle_summary.csv"), index=False)
        print(f"[SURD] perm_df shape = {perm_df.shape}")
    else:
        print("[SURD] skip permutation (no segments or do_permutation=False)")

    # Write back to adata.uns
    adata.uns.setdefault("surd", {})
    adata.uns["surd"]["segments"] = {
        "params": dict(
            discretize=discretize,
            nbins_target=nbins_target,
            min_bins_per_dim=min_bins_per_dim,
            max_total_bins=max_total_bins,
            fast_mode=fast_mode,
            top_parents_per_step=top_parents_per_step,
            max_gems_per_outflow=max_gems_per_outflow,
            per_gem_inflow_cap=per_gem_inflow_cap,
            max_parents_combined=max_parents_combined,
            min_samples_block=min_samples_block,
            condition_key=cond_key,
            celltype_key=ctype_key,
            match_embedding_key=z_key,
            flow_matrix_key=flow_key,
            adjacency_path=adj_path,
        ),
        "in2gem": in2gem_df,
        "gem2out": gem2out_df,
        "combined_parent": combined_df,
        "path_triples": triple_df,
        "permtest_rowshuffle": perm_df,
    }

    return {
        "in2gem": in2gem_df,
        "gem2out": gem2out_df,
        "combined": combined_df,
        "triples": triple_df,
        "permtest_rowshuffle": perm_df,
    }
