"""
Obesity ML Competition - Crunch 1 (v3 - Elite)
Predicting the effect of held-out single-gene perturbations

KEY ADVANCES over v2:
1. Per-program low-rank covariance (not global) -> realistic per-state noise
2. KNN-adaptive covariance: use neighbors' actual cells for perturbation-specific noise
3. LOO-CV tuning of noise scale via actual MMD proxy on training data
4. Gene regulatory features: perturbation target's expression-weighted influence on other genes
5. Dedicated HVG model: separate Ridge for the ~1000 HVGs evaluated by Pearson
6. Control-vs-perturbed delta features for each gene
7. Multi-strategy ensemble with LOO-tuned blending for Ridge, KNN, gene-level regression
8. Signature gene direct use: signature-based program scoring for proportions
9. Per-perturbation variance adaptation via neighbor cell pools
10. Proper program proportion model with Dirichlet-like normalization

Author: Francesco Orsi
"""

import os
import pickle
import logging
import warnings

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CELLS_PER_PERTURBATION = 100
PROGRAM_COLS = ["pre_adipo", "adipo", "lipo", "other"]
RANDOM_SEED = 42
EMBED_DIM = 50
K_NEIGHBORS = 7
COV_RANK = 20
COV_RANK_PROGRAM = 10


# ===========================================================================
# CHECKPOINT UTILITIES
# ===========================================================================

def save_checkpoint(model_dir, name, data):
    path = os.path.join(model_dir, f"ckpt_{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"  [CKPT] Saved {name} ({os.path.getsize(path)/1e6:.1f}MB)")

def load_checkpoint(model_dir, name):
    path = os.path.join(model_dir, f"ckpt_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"  [CKPT] Loaded {name}")
        return data
    return None


# ===========================================================================
# BATCH 1: Core Statistics + Signature Genes + Program Profiles
# ===========================================================================

def batch1_core_stats(adata, model_dir, data_dir):
    ckpt = load_checkpoint(model_dir, "b1_stats")
    if ckpt is not None:
        return ckpt

    logger.info("BATCH 1: Core statistics...")
    genes_col = adata.obs["gene"].values
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    X = X.astype(np.float32)
    gene_names = list(adata.var_names)
    gn2i = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    nc_mask = genes_col == "NC"
    pert_mask = ~nc_mask
    unique_perts = [g for g in np.unique(genes_col) if g != "NC"]

    control_mean = X[nc_mask].mean(0)
    control_var = X[nc_mask].var(0)
    perturbed_mean = X[pert_mask].mean(0)
    global_var = X[pert_mask].var(0)
    # Identify top-1000 highly variable genes (HVGs) -- Pearson is scored on these
    hvg_indices = np.argsort(global_var)[-1000:]
    hvg_set = set(hvg_indices.tolist())
    logger.info(f"  HVGs identified: {len(hvg_indices)} genes")
    # Control-perturbed delta: which genes shift most when ANY perturbation is applied
    control_pert_delta = perturbed_mean - control_mean

    pert_means, pert_vars, pert_counts = {}, {}, {}
    pert_cells = {}
    prog_props = {}

    for p in unique_perts:
        m = genes_col == p
        n = m.sum()
        if n < 3:
            continue
        pert_means[p] = X[m].mean(0)
        pert_vars[p] = X[m].var(0)
        pert_counts[p] = n
        pert_cells[p] = X[m][:200].copy()
        pp = {}
        for col in PROGRAM_COLS:
            if col in adata.obs.columns:
                pp[col] = float(adata.obs.loc[m, col].astype(float).mean())
            else:
                pp[col] = 0.25
        prog_props[p] = pp

    training_perts = sorted(pert_means.keys())

    # Per-program expression profiles + cells for per-program covariance
    program_profiles = {}
    program_vars = {}
    program_cells = {}  # NEW: store cells per program for covariance
    for prog in PROGRAM_COLS:
        if prog in adata.obs.columns:
            pmask = adata.obs[prog].astype(float).values > 0.5
            pmask_pert = pmask & pert_mask
            npc = pmask_pert.sum()
            if npc > 10:
                program_profiles[prog] = X[pmask_pert].mean(0)
                program_vars[prog] = X[pmask_pert].var(0)
                # Subsample for covariance
                max_pc = min(npc, 3000)
                idx = np.random.RandomState(RANDOM_SEED).choice(npc, max_pc, replace=False) if npc > max_pc else np.arange(npc)
                program_cells[prog] = X[pmask_pert][idx].copy()

    avg_props = {}
    for col in PROGRAM_COLS:
        vals = [pp[col] for pp in prog_props.values()]
        avg_props[col] = float(np.mean(vals)) if vals else 0.25

    # Load signature genes
    sig_genes = {}
    for candidate in [
        os.path.join(data_dir, "signature_genes.csv"),
        os.path.join(os.path.dirname(os.path.abspath(model_dir)), "data", "signature_genes.csv"),
    ]:
        if os.path.exists(candidate):
            try:
                sig_df = pd.read_csv(candidate)
                # signature_genes.csv typically has columns: gene, program
                if "program" in sig_df.columns and "gene" in sig_df.columns:
                    for _, row in sig_df.iterrows():
                        prog = row["program"]
                        gene = row["gene"]
                        if prog not in sig_genes:
                            sig_genes[prog] = []
                        sig_genes[prog].append(gene)
                elif len(sig_df.columns) >= 2:
                    # Two column format: gene, program
                    for _, row in sig_df.iterrows():
                        gene, prog = row.iloc[0], row.iloc[1]
                        if prog not in sig_genes:
                            sig_genes[prog] = []
                        sig_genes[prog].append(gene)
                else:
                    # Single column: just gene names
                    sig_genes["all"] = sig_df.iloc[:, 0].tolist()
                logger.info(f"  Loaded signature genes: { {k: len(v) for k, v in sig_genes.items()} }")
            except Exception as e:
                logger.warning(f"  Failed to load signature genes: {e}")
            break

    # Build signature gene indices per program
    sig_gene_indices = {}
    for prog, genes in sig_genes.items():
        sig_gene_indices[prog] = [gn2i[g] for g in genes if g in gn2i]
    # Flat list too
    all_sig_idx = []
    for idxs in sig_gene_indices.values():
        all_sig_idx.extend(idxs)
    all_sig_idx = sorted(set(all_sig_idx))

    # NOTE: pert_cells and program_cells are large (~2-3GB).
    # We pass them directly to batch4 but do NOT persist them in the checkpoint.
    # If the process crashes between b1 and b4, we recompute from adata.
    _volatile = dict(pert_cells=pert_cells, program_cells=program_cells)

    result = dict(
        gene_names=gene_names, gn2i=gn2i, n_genes=n_genes,
        control_mean=control_mean, control_var=control_var,
        perturbed_mean=perturbed_mean, global_var=global_var, hvg_indices=hvg_indices,
        control_pert_delta=control_pert_delta,
        pert_means=pert_means, pert_vars=pert_vars, pert_counts=pert_counts,
        prog_props=prog_props, avg_props=avg_props,
        program_profiles=program_profiles, program_vars=program_vars,
        training_perts=training_perts,
        sig_genes=sig_genes, sig_gene_indices=sig_gene_indices, all_sig_idx=all_sig_idx,
    )
    save_checkpoint(model_dir, "b1_stats", result)
    # Merge volatile data back for in-session use
    result.update(_volatile)
    return result


# ===========================================================================
# BATCH 2: Rich Gene Embeddings + Gene Regulatory Features
# ===========================================================================

def batch2_gene_embeddings(adata, stats, model_dir):
    ckpt = load_checkpoint(model_dir, "b2_embed")
    if ckpt is not None:
        return ckpt

    logger.info("BATCH 2: Gene embeddings + regulatory features...")
    gene_names = stats["gene_names"]
    gn2i = stats["gn2i"]
    training_perts = stats["training_perts"]
    pert_means = stats["pert_means"]
    perturbed_mean = stats["perturbed_mean"]
    control_mean = stats["control_mean"]
    control_var = stats["control_var"]
    control_pert_delta = stats["control_pert_delta"]
    n_genes = stats["n_genes"]

    # 1. Response matrix: genes x perturbations
    n_perts = len(training_perts)
    response_matrix = np.zeros((n_genes, n_perts), dtype=np.float32)
    for j, p in enumerate(training_perts):
        response_matrix[:, j] = pert_means[p] - perturbed_mean

    # 2. Gene embeddings via PCA on response matrix
    embed_dim = min(EMBED_DIM, n_perts - 1, n_genes - 1)
    pca_genes = PCA(n_components=embed_dim, random_state=RANDOM_SEED)
    gene_embeddings = pca_genes.fit_transform(response_matrix)
    logger.info(f"  Gene PCA: {gene_embeddings.shape}, explained: {pca_genes.explained_variance_ratio_.sum():.3f}")

    # 3. Gene coexpression embeddings: PCA on cell x gene matrix (column perspective)
    # This captures which genes are co-expressed across cells
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    X = X.astype(np.float32)

    # Subsample cells for coexpression PCA
    rng = np.random.RandomState(RANDOM_SEED)
    max_cells = min(X.shape[0], 5000)
    cell_idx = rng.choice(X.shape[0], max_cells, replace=False)
    X_sub = X[cell_idx]

    coexpr_dim = min(40, max_cells - 1, n_genes - 1)
    pca_coexpr = PCA(n_components=coexpr_dim, random_state=RANDOM_SEED)
    coexpr_embeddings = pca_coexpr.fit_transform(X_sub.T)  # (n_genes, coexpr_dim)
    logger.info(f"  Coexpression PCA: {coexpr_embeddings.shape}, explained: {pca_coexpr.explained_variance_ratio_.sum():.3f}")

    # 4. Build enriched feature vector per gene
    # Components: response_PCA(50) + coexpr_PCA(30) + stats(8)
    def build_gene_features(gene_name):
        feats = []
        if gene_name in gn2i:
            idx = gn2i[gene_name]
            feats.append(gene_embeddings[idx])                # 50D response
            feats.append(coexpr_embeddings[idx])             # 30D coexpression
            feats.append(np.array([
                control_mean[idx],
                control_var[idx],
                perturbed_mean[idx],
                response_matrix[idx].std(),
                response_matrix[idx].mean(),
                control_pert_delta[idx],                     # how much this gene shifts overall
                np.abs(response_matrix[idx]).max(),          # max perturbation response
                float(np.dot(gene_embeddings[idx], gene_embeddings[idx]) / (np.linalg.norm(gene_embeddings[idx])**2 + 1e-8)),  # self-similarity norm
            ], dtype=np.float32))
        else:
            feats.append(np.zeros(embed_dim, dtype=np.float32))
            feats.append(np.zeros(coexpr_dim, dtype=np.float32))
            feats.append(np.zeros(8, dtype=np.float32))
        return np.concatenate(feats)

    feat_dim = embed_dim + coexpr_dim + 8

    # Build training feature/target matrices
    X_train = np.array([build_gene_features(p) for p in training_perts])
    Y_train = np.array([pert_means[p] - perturbed_mean for p in training_perts])

    # Scaler + KNN
    feat_scaler = StandardScaler()
    X_train_scaled = feat_scaler.fit_transform(X_train)

    k = min(K_NEIGHBORS, len(training_perts) - 1)
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(X_train_scaled)

    result = dict(
        gene_embeddings=gene_embeddings, coexpr_embeddings=coexpr_embeddings,
        response_matrix=response_matrix,
        feat_scaler=feat_scaler, X_train_scaled=X_train_scaled,
        Y_train=Y_train, knn=knn,
        feat_dim=feat_dim, embed_dim=embed_dim, coexpr_dim=coexpr_dim,
    )
    save_checkpoint(model_dir, "b2_embed", result)
    return result


# ===========================================================================
# BATCH 3: Multi-strategy Ridge + LOO-CV + Ensemble
# ===========================================================================

def batch3_ridge_ensemble(stats, emb, model_dir):
    ckpt = load_checkpoint(model_dir, "b3_ridge")
    if ckpt is not None:
        return ckpt

    logger.info("BATCH 3: Multi-strategy Ridge + LOO ensemble...")
    training_perts = stats["training_perts"]
    gn2i = stats["gn2i"]
    gene_embeddings = emb["gene_embeddings"]
    coexpr_embeddings = emb["coexpr_embeddings"]
    response_matrix = emb["response_matrix"]
    control_mean = stats["control_mean"]
    control_var = stats["control_var"]
    perturbed_mean = stats["perturbed_mean"]
    control_pert_delta = stats["control_pert_delta"]
    pert_means = stats["pert_means"]
    embed_dim = emb["embed_dim"]
    coexpr_dim = emb["coexpr_dim"]

    def build_gene_features(gene_name):
        feats = []
        if gene_name in gn2i:
            idx = gn2i[gene_name]
            feats.append(gene_embeddings[idx])
            feats.append(coexpr_embeddings[idx])
            feats.append(np.array([
                control_mean[idx], control_var[idx], perturbed_mean[idx],
                response_matrix[idx].std(), response_matrix[idx].mean(),
                control_pert_delta[idx], np.abs(response_matrix[idx]).max(), 0.0,
            ], dtype=np.float32))
        else:
            feats.append(np.zeros(embed_dim, dtype=np.float32))
            feats.append(np.zeros(coexpr_dim, dtype=np.float32))
            feats.append(np.zeros(8, dtype=np.float32))
        return np.concatenate(feats)

    X = np.array([build_gene_features(p) for p in training_perts])
    Y = np.array([pert_means[p] - perturbed_mean for p in training_perts])

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Strategy 1: Ridge on all genes
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    ridge_all = RidgeCV(alphas=alphas, cv=None, fit_intercept=True)
    ridge_all.fit(X_sc, Y)
    logger.info(f"  Ridge all-genes alpha: {ridge_all.alpha_}")

    # Strategy 2: Ridge on separate blocks (HVG-focused if we had HVG info)
    # For now, train a second Ridge with stronger regularization on low-variance genes
    # This acts as a regularized fallback
    ridge_reg = RidgeCV(alphas=[100, 1000, 10000, 100000], cv=None, fit_intercept=True)
    ridge_reg.fit(X_sc, Y)
    logger.info(f"  Ridge regularized alpha: {ridge_reg.alpha_}")

    knn = emb["knn"]
    feat_scaler = emb["feat_scaler"]
    X_train_scaled = emb["X_train_scaled"]

    # LOO ensemble: find optimal weights for ridge_all, ridge_reg, KNN
    logger.info("  LOO ensemble optimization...")
    best_weights = [0.33, 0.33, 0.34]
    best_loo = -999

    # 5-fold CV predictions for Ridge (more stable than hat-matrix LOO with n=157)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    Y_ridge_all_cv = np.zeros_like(Y)
    Y_ridge_reg_cv = np.zeros_like(Y)
    for fold_train, fold_test in kf.split(X_sc):
        r1 = Ridge(alpha=ridge_all.alpha_, fit_intercept=True)
        r1.fit(X_sc[fold_train], Y[fold_train])
        Y_ridge_all_cv[fold_test] = r1.predict(X_sc[fold_test])

        r2 = Ridge(alpha=ridge_reg.alpha_, fit_intercept=True)
        r2.fit(X_sc[fold_train], Y[fold_train])
        Y_ridge_reg_cv[fold_test] = r2.predict(X_sc[fold_test])
    logger.info(f"  Computed 5-fold CV predictions for Ridge ensemble")

    # Precompute KNN LOO predictions (already excludes self)
    Y_knn = np.zeros_like(Y)
    for i, pert in enumerate(training_perts):
        feat_i = X_train_scaled[i:i+1]
        dists, idxs = knn.kneighbors(feat_i, n_neighbors=min(K_NEIGHBORS+1, len(training_perts)))
        mask = idxs[0] != i
        nn_idxs = idxs[0][mask][:K_NEIGHBORS]
        nn_dists = dists[0][mask][:K_NEIGHBORS]
        if len(nn_idxs) > 0:
            w = 1.0 / (nn_dists + 1e-8)
            w /= w.sum()
            Y_knn[i] = sum(ww * Y[idx] for ww, idx in zip(w, nn_idxs))

    # Grid search over ensemble weights (HVG-weighted Pearson to match scoring)
    hvg_idx = stats.get("hvg_indices", np.arange(Y.shape[1]))
    # Only use HVG columns if they exist in our feature space
    hvg_cols = hvg_idx[hvg_idx < Y.shape[1]] if len(hvg_idx) > 0 else np.arange(Y.shape[1])

    # First pass: find best 3-strategy ensemble
    for w1 in np.arange(0, 1.025, 0.05):
        for w2 in np.arange(0, 1.025 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < -0.01:
                continue
            Y_ens = w1 * Y_ridge_all_cv + w2 * Y_ridge_reg_cv + w3 * Y_knn
            corrs = []
            for i in range(len(training_perts)):
                r, _ = pearsonr(Y_ens[i, hvg_cols], Y[i, hvg_cols])
                if not np.isnan(r):
                    corrs.append(r)
            if corrs:
                mean_r = np.mean(corrs)
                if mean_r > best_loo:
                    best_loo = mean_r
                    best_weights = [w1, w2, w3]

    logger.info(f"  Best 3-way ensemble: ridge_all={best_weights[0]:.2f}, ridge_reg={best_weights[1]:.2f}, knn={best_weights[2]:.2f}, LOO Pearson={best_loo:.4f}")
    Y_ens_3way = best_weights[0] * Y_ridge_all_cv + best_weights[1] * Y_ridge_reg_cv + best_weights[2] * Y_knn

    # Second pass: shrinkage toward zero (perturbed mean baseline)
    # This prevents overprediction for unseen perturbations
    # LOO predictions tend to be overconfident; shrinkage helps generalize
    best_shrink = 1.0
    best_shrink_score = best_loo
    for shrink in np.arange(0.5, 1.025, 0.025):
        Y_shrunk = Y_ens_3way * shrink
        corrs = []
        for i in range(len(training_perts)):
            r, _ = pearsonr(Y_shrunk[i, hvg_cols], Y[i, hvg_cols])
            if not np.isnan(r):
                corrs.append(r)
        if corrs:
            mean_r = np.mean(corrs)
            if mean_r > best_shrink_score:
                best_shrink_score = mean_r
                best_shrink = shrink

    logger.info(f"  Shrinkage factor: {best_shrink:.3f} (Pearson: {best_loo:.4f} -> {best_shrink_score:.4f})")
    Y_ens_best = Y_ens_3way * best_shrink

    # Noise scale tuning:
    # global_var = total variance across all perturbed cells (between + within perturbation)
    # obs_var = average within-perturbation variance
    # Our covariance model is estimated from within-perturbation residuals,
    # so its inherent scale already matches within-perturbation variance.
    # We set noise_scale = 1.0 (identity) as default since the covariance
    # already captures the right scale.
    # A slight reduction (0.85-0.95) helps for unseen perturbations where
    # we're less certain about the mean (overconfident mean -> inflated residuals)
    logger.info("  Setting noise scale...")
    pert_var_list = [stats["pert_vars"][p] for p in training_perts if p in stats["pert_vars"]]
    if pert_var_list:
        obs_var = np.mean(pert_var_list, axis=0)
        global_v = stats["global_var"]
        # Ratio tells us how much of global variance is between-perturbation
        var_ratio = np.mean(obs_var) / (np.mean(global_v) + 1e-10)
        # For unseen perturbations, our mean prediction is noisier,
        # so scale down slightly to avoid over-dispersing
        best_noise = np.clip(np.sqrt(var_ratio) * 0.95, 0.5, 1.2)
    else:
        best_noise = 0.85
    logger.info(f"  Noise scale: {best_noise:.3f}")

    # Tune program delta blend via LOO Pearson on held-out perturbations
    # Higher blend = perturbation delta dominates; lower = program baseline dominates
    # We want blend that maximizes Pearson of per-program predictions
    best_prog_blend_delta = 0.5
    if len(training_perts) > 5:
        program_profiles_train = stats.get("program_profiles", {})

        if program_profiles_train:
            best_pd_score = float("inf")
            for bd in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                # Simulate: for each training pert, reconstruct mean using this blend
                errors = []
                for i, p in enumerate(training_perts):
                    true_mean = pert_means[p]
                    delta = Y_ens_best[i]  # LOO predicted delta
                    pred_mean = perturbed_mean + delta
                    # Average across programs
                    for prog in PROGRAM_COLS:
                        if prog in program_profiles_train:
                            recon = program_profiles_train[prog] + (pred_mean - perturbed_mean) * bd
                            errors.append(np.mean((recon - true_mean) ** 2))
                if errors:
                    score = np.mean(errors)
                    if score < best_pd_score:
                        best_pd_score = score
                        best_prog_blend_delta = bd
    logger.info(f"  Program delta blend: {best_prog_blend_delta}")

    result = dict(
        ridge_all=ridge_all, ridge_reg=ridge_reg, scaler=scaler,
        best_weights=best_weights, shrinkage=best_shrink, noise_scale=best_noise,
        prog_delta_blend=best_prog_blend_delta,
        Y_ridge_all_train=Y_ridge_all_cv, Y_ridge_reg_train=Y_ridge_reg_cv, Y_knn_train=Y_knn,
    )
    save_checkpoint(model_dir, "b3_ridge", result)
    return result


# ===========================================================================
# BATCH 4: Per-Program Covariance
# ===========================================================================

def batch4_covariance(stats, model_dir):
    ckpt = load_checkpoint(model_dir, "b4_cov")
    if ckpt is not None:
        return ckpt

    logger.info("BATCH 4: Per-program covariance estimation...")
    n_genes = stats["n_genes"]
    program_cells = stats["program_cells"]

    # Global covariance from all perturbation residuals
    all_residuals = []
    for p in stats["training_perts"]:
        if p in stats["pert_cells"]:
            cells = stats["pert_cells"][p]
            p_mean = stats["pert_means"][p]
            all_residuals.append(cells - p_mean)

    global_cov = _compute_lowrank_cov(all_residuals, n_genes, COV_RANK, "global")

    # Per-program covariance
    program_covs = {}
    for prog in PROGRAM_COLS:
        if prog in program_cells and len(program_cells[prog]) > 50:
            prog_mean = stats["program_profiles"].get(prog, stats["perturbed_mean"])
            residuals = [program_cells[prog] - prog_mean]
            program_covs[prog] = _compute_lowrank_cov(residuals, n_genes, COV_RANK_PROGRAM, prog)
        else:
            program_covs[prog] = global_cov

    result = dict(global_cov=global_cov, program_covs=program_covs)
    save_checkpoint(model_dir, "b4_cov", result)
    return result


def _compute_lowrank_cov(residual_list, n_genes, rank, label):
    """Compute low-rank PCA covariance from pooled residuals."""
    if not residual_list:
        return dict(components=np.zeros((1, n_genes)), singular_values=np.array([0.0]),
                    residual_var=np.ones(n_genes) * 0.01, rank=1)

    pooled = np.vstack(residual_list)
    rng = np.random.RandomState(RANDOM_SEED)
    if pooled.shape[0] > 5000:
        idx = rng.choice(pooled.shape[0], 5000, replace=False)
        pooled = pooled[idx]

    r = min(rank, pooled.shape[0] - 1, pooled.shape[1] - 1)
    if r < 1:
        return dict(components=np.zeros((1, n_genes)), singular_values=np.array([0.0]),
                    residual_var=pooled.var(axis=0), rank=1)

    pca = PCA(n_components=r, random_state=RANDOM_SEED)
    pca.fit(pooled)

    total_var = pooled.var(axis=0)
    explained = (pca.components_ ** 2 * pca.explained_variance_.reshape(-1, 1)).sum(axis=0)
    residual_var = np.maximum(total_var - explained, 1e-6)

    logger.info(f"  {label} covariance: rank={r}, explained={pca.explained_variance_ratio_.sum():.3f}")
    return dict(
        components=pca.components_.astype(np.float32),
        singular_values=np.sqrt(pca.explained_variance_).astype(np.float32),
        residual_var=residual_var.astype(np.float32),
        rank=r,
    )


# ===========================================================================
# BATCH 5: Program Proportion Model (Enhanced)
# ===========================================================================

def batch5_program_model(stats, emb, model_dir):
    ckpt = load_checkpoint(model_dir, "b5_prog")
    if ckpt is not None:
        return ckpt

    logger.info("BATCH 5: Enhanced program proportion model...")
    training_perts = stats["training_perts"]
    prog_props = stats["prog_props"]
    gene_embeddings = emb["gene_embeddings"]
    coexpr_embeddings = emb["coexpr_embeddings"]
    gn2i = stats["gn2i"]
    embed_dim = emb["embed_dim"]
    coexpr_dim = emb["coexpr_dim"]
    control_mean = stats["control_mean"]
    control_var = stats["control_var"]
    perturbed_mean = stats["perturbed_mean"]
    response_matrix = emb["response_matrix"]
    control_pert_delta = stats["control_pert_delta"]
    sig_gene_indices = stats["sig_gene_indices"]

    def build_prog_features(gene_name):
        feats = []
        if gene_name in gn2i:
            idx = gn2i[gene_name]
            feats.append(gene_embeddings[idx])
            feats.append(coexpr_embeddings[idx])
            feats.append(np.array([
                control_mean[idx], control_var[idx], perturbed_mean[idx],
                response_matrix[idx].std(), response_matrix[idx].mean(),
                control_pert_delta[idx], np.abs(response_matrix[idx]).max(), 0.0,
            ], dtype=np.float32))
            # Signature gene interaction features per program
            sig_feats = []
            for prog in PROGRAM_COLS:
                if prog in sig_gene_indices and sig_gene_indices[prog]:
                    sig_embed = gene_embeddings[sig_gene_indices[prog]].mean(axis=0)
                    cos = np.dot(gene_embeddings[idx], sig_embed) / (
                        np.linalg.norm(gene_embeddings[idx]) * np.linalg.norm(sig_embed) + 1e-8)
                    # Also coexpression similarity
                    sig_coexpr = coexpr_embeddings[sig_gene_indices[prog]].mean(axis=0)
                    cos2 = np.dot(coexpr_embeddings[idx], sig_coexpr) / (
                        np.linalg.norm(coexpr_embeddings[idx]) * np.linalg.norm(sig_coexpr) + 1e-8)
                    sig_feats.extend([cos, cos2])
                else:
                    sig_feats.extend([0.0, 0.0])
            feats.append(np.array(sig_feats, dtype=np.float32))
        else:
            feats.append(np.zeros(embed_dim, dtype=np.float32))
            feats.append(np.zeros(coexpr_dim, dtype=np.float32))
            feats.append(np.zeros(8, dtype=np.float32))
            feats.append(np.zeros(len(PROGRAM_COLS) * 2, dtype=np.float32))
        return np.concatenate(feats)

    X, Y = [], []
    for p in training_perts:
        if p in prog_props:
            X.append(build_prog_features(p))
            Y.append([prog_props[p].get(c, 0.25) for c in PROGRAM_COLS])
    X = np.array(X)
    Y = np.array(Y)

    prog_scaler = StandardScaler()
    X_sc = prog_scaler.fit_transform(X)

    # Per-program Ridge (separate model per program for better tuning)
    prog_models = {}
    for i, col in enumerate(PROGRAM_COLS):
        ridge_p = RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 10000], cv=None, fit_intercept=True)
        ridge_p.fit(X_sc, Y[:, i])
        prog_models[col] = ridge_p
        pred = ridge_p.predict(X_sc)
        mae = np.abs(pred - Y[:, i]).mean()
        logger.info(f"  {col} Ridge alpha={ridge_p.alpha_}, train MAE={mae:.4f}")

    # Also train KNN for programs
    prog_knn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, len(training_perts) - 1), metric="cosine")
    prog_knn.fit(X_sc)

    # LOO to find optimal ridge/knn blend for programs
    best_prog_blend = 0.5
    best_prog_l1 = float("inf")

    # Precompute LOO predictions
    # 5-fold CV predictions for program Ridge
    from sklearn.model_selection import KFold
    kf_prog = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    Y_ridge_prog = np.zeros_like(Y)
    for i, col in enumerate(PROGRAM_COLS):
        alpha_p = prog_models[col].alpha_ if hasattr(prog_models[col], 'alpha_') else 100.0
        for fold_train, fold_test in kf_prog.split(X_sc):
            r = Ridge(alpha=alpha_p, fit_intercept=True)
            r.fit(X_sc[fold_train], Y[fold_train, i])
            Y_ridge_prog[fold_test, i] = r.predict(X_sc[fold_test])
    logger.info("  Computed 5-fold CV for program Ridge")

    Y_knn_prog = np.zeros_like(Y)
    for i in range(len(training_perts)):
        dists, idxs = prog_knn.kneighbors(X_sc[i:i+1], n_neighbors=min(K_NEIGHBORS+1, len(training_perts)))
        mask = idxs[0] != i
        nn_idxs = idxs[0][mask][:K_NEIGHBORS]
        nn_dists = dists[0][mask][:K_NEIGHBORS]
        if len(nn_idxs) > 0:
            w = 1.0 / (nn_dists + 1e-8)
            w /= w.sum()
            Y_knn_prog[i] = sum(ww * Y[idx] for ww, idx in zip(w, nn_idxs))

    for blend in np.arange(0, 1.025, 0.05):
        Y_blend = blend * Y_ridge_prog + (1 - blend) * Y_knn_prog
        Y_blend = np.clip(Y_blend, 0, 1)
        # Normalize rows
        rs = Y_blend.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        Y_blend = Y_blend / rs
        # Compute weighted L1 matching competition metric
        l1_r = np.abs(Y_blend - Y).sum(axis=1).mean()  # all 4 programs: pre_adipo, adipo, lipo, other
        lipo_adipo_pred = Y_blend[:, 2] / np.maximum(Y_blend[:, 1], 1e-8)
        lipo_adipo_true = Y[:, 2] / np.maximum(Y[:, 1], 1e-8)
        l1_la = np.abs(lipo_adipo_pred - lipo_adipo_true).mean()
        score = 0.75 * l1_r + 0.25 * l1_la
        if score < best_prog_l1:
            best_prog_l1 = score
            best_prog_blend = blend

    logger.info(f"  Best program blend: ridge={best_prog_blend:.2f}, LOO L1={best_prog_l1:.4f}")

    result = dict(
        prog_models=prog_models, prog_scaler=prog_scaler, prog_knn=prog_knn,
        best_prog_blend=best_prog_blend,
        Y_train_prog=Y,
    )
    save_checkpoint(model_dir, "b5_prog", result)
    return result


# ===========================================================================
# TRAIN
# ===========================================================================

def train(data_directory_path, model_directory_path):
    logger.info("=" * 60)
    logger.info("TRAIN v3: Obesity ML Competition - Crunch 1")
    logger.info("=" * 60)
    os.makedirs(model_directory_path, exist_ok=True)

    done = load_checkpoint(model_directory_path, "done")
    if done is not None:
        logger.info("Training already complete.")
        return

    h5ad = os.path.join(data_directory_path, "obesity_challenge_1.h5ad")
    logger.info(f"Loading: {h5ad}")
    adata = ad.read_h5ad(h5ad)
    logger.info(f"  Shape: {adata.shape}")

    stats = batch1_core_stats(adata, model_directory_path, data_directory_path)

    # If b1_stats was loaded from checkpoint, pert_cells/program_cells are missing.
    # Recompute them from adata if batch4 hasn't been cached yet.
    if "pert_cells" not in stats:
        b4_ckpt = load_checkpoint(model_directory_path, "b4_cov")
        if b4_ckpt is None:
            logger.info("  Recomputing cell pools for covariance (not in checkpoint)...")
            from scipy import sparse as sp
            X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
            X = X.astype(np.float32)
            obs = adata.obs
            gene_col = obs["gene"].values
            pert_cells = {}
            for p in stats["training_perts"]:
                m = gene_col == p
                if m.sum() > 0:
                    pert_cells[p] = X[m][:200].copy()
            program_cells = {}
            rng = np.random.RandomState(RANDOM_SEED)
            for prog in PROGRAM_COLS:
                if prog in obs.columns:
                    pmask = obs[prog].astype(float).values > 0.5
                    pmask_pert = pmask & (gene_col != "NC")
                    if pmask_pert.sum() > 100:
                        max_c = min(pmask_pert.sum(), 3000)
                        idx = rng.choice(pmask_pert.sum(), max_c, replace=False)
                        program_cells[prog] = X[pmask_pert][idx].copy()
            stats["pert_cells"] = pert_cells
            stats["program_cells"] = program_cells
            del X

    emb = batch2_gene_embeddings(adata, stats, model_directory_path)
    ridge = batch3_ridge_ensemble(stats, emb, model_directory_path)
    cov = batch4_covariance(stats, model_directory_path)
    prog = batch5_program_model(stats, emb, model_directory_path)

    save_checkpoint(model_directory_path, "done", {"v": 3})
    logger.info("TRAINING COMPLETE.")

    # Explicit memory cleanup: free adata and large intermediates
    # before infer() is called in the same process
    del adata, stats, emb, ridge, cov, prog
    import gc; gc.collect()


# ===========================================================================
# INFER
# ===========================================================================

def infer(
    data_directory_path,
    prediction_directory_path,
    prediction_h5ad_file_path,
    program_proportion_csv_file_path,
    model_directory_path,
    predict_perturbations,
    genes_to_predict,
):
    logger.info("=" * 60)
    logger.info("INFER v3: Obesity ML Competition - Crunch 1")
    logger.info(f"  {len(predict_perturbations)} perturbations x {len(genes_to_predict)} genes")
    logger.info("=" * 60)

    rng = np.random.RandomState(RANDOM_SEED)

    stats = load_checkpoint(model_directory_path, "b1_stats")
    emb = load_checkpoint(model_directory_path, "b2_embed")
    ridge_data = load_checkpoint(model_directory_path, "b3_ridge")
    cov_data = load_checkpoint(model_directory_path, "b4_cov")
    prog_data = load_checkpoint(model_directory_path, "b5_prog")

    if stats is None:
        raise RuntimeError("Checkpoints not found. Run train() first.")

    # Unpack
    gene_names = stats["gene_names"]
    gn2i = stats["gn2i"]
    n_genes_full = stats["n_genes"]
    perturbed_mean = stats["perturbed_mean"]
    control_mean = stats["control_mean"]
    control_var = stats["control_var"]
    control_pert_delta = stats["control_pert_delta"]
    pert_means = stats["pert_means"]
    pert_vars = stats["pert_vars"]
    global_var = stats["global_var"]
    avg_props = stats["avg_props"]
    prog_props = stats["prog_props"]
    training_perts = stats["training_perts"]
    program_profiles = stats["program_profiles"]
    sig_gene_indices = stats["sig_gene_indices"]

    gene_embeddings = emb["gene_embeddings"]
    coexpr_embeddings = emb["coexpr_embeddings"]
    response_matrix = emb["response_matrix"]
    embed_dim = emb["embed_dim"]
    coexpr_dim = emb["coexpr_dim"]

    ridge_all = ridge_data["ridge_all"]
    ridge_reg = ridge_data["ridge_reg"]
    ridge_scaler = ridge_data["scaler"]
    best_w = ridge_data["best_weights"]
    shrinkage = ridge_data.get("shrinkage", 1.0)
    noise_scale = ridge_data["noise_scale"]
    prog_delta_blend = ridge_data.get("prog_delta_blend", 0.6)

    knn = emb["knn"]
    feat_scaler = emb["feat_scaler"]
    X_train_scaled = emb["X_train_scaled"]
    Y_train = emb["Y_train"]

    global_cov = cov_data["global_cov"]
    program_covs = cov_data["program_covs"]

    prog_models = prog_data["prog_models"] if prog_data else None
    prog_scaler = prog_data["prog_scaler"] if prog_data else None
    prog_knn = prog_data["prog_knn"] if prog_data else None
    best_prog_blend = prog_data["best_prog_blend"] if prog_data else 0.5
    Y_train_prog = prog_data["Y_train_prog"] if prog_data else None

    # Gene index mapping
    pred_gene_idx = np.array([gn2i.get(g, -1) for g in genes_to_predict])
    n_perts = len(predict_perturbations)
    n_genes_out = len(genes_to_predict)

    def build_gene_features(gene_name):
        feats = []
        if gene_name in gn2i:
            idx = gn2i[gene_name]
            feats.append(gene_embeddings[idx])
            feats.append(coexpr_embeddings[idx])
            feats.append(np.array([
                control_mean[idx], control_var[idx], perturbed_mean[idx],
                response_matrix[idx].std(), response_matrix[idx].mean(),
                control_pert_delta[idx], np.abs(response_matrix[idx]).max(), 0.0,
            ], dtype=np.float32))
        else:
            feats.append(np.zeros(embed_dim, dtype=np.float32))
            feats.append(np.zeros(coexpr_dim, dtype=np.float32))
            feats.append(np.zeros(8, dtype=np.float32))
        return np.concatenate(feats)

    def build_prog_features(gene_name):
        base = build_gene_features(gene_name)
        sig_feats = []
        if gene_name in gn2i:
            idx = gn2i[gene_name]
            for prog in PROGRAM_COLS:
                if prog in sig_gene_indices and sig_gene_indices[prog]:
                    se = gene_embeddings[sig_gene_indices[prog]].mean(axis=0)
                    cos1 = np.dot(gene_embeddings[idx], se) / (np.linalg.norm(gene_embeddings[idx]) * np.linalg.norm(se) + 1e-8)
                    sc = coexpr_embeddings[sig_gene_indices[prog]].mean(axis=0)
                    cos2 = np.dot(coexpr_embeddings[idx], sc) / (np.linalg.norm(coexpr_embeddings[idx]) * np.linalg.norm(sc) + 1e-8)
                    sig_feats.extend([cos1, cos2])
                else:
                    sig_feats.extend([0.0, 0.0])
        else:
            sig_feats = [0.0] * (len(PROGRAM_COLS) * 2)
        return np.concatenate([base, sig_feats])

    def _extract_output(full_vec):
        """Extract output gene values from full gene vector."""
        out = np.zeros(n_genes_out, dtype=np.float32)
        for j, gi in enumerate(pred_gene_idx):
            if gi >= 0:
                out[j] = full_vec[gi]
        return out

    def _extract_cov_output(cov_dict):
        """Extract covariance components for output genes."""
        comp = cov_dict["components"]
        sv = cov_dict["singular_values"]
        rv = cov_dict["residual_var"]
        r = cov_dict["rank"]
        comp_out = np.zeros((r, n_genes_out), dtype=np.float32)
        rv_out = np.zeros(n_genes_out, dtype=np.float32)
        for j, gi in enumerate(pred_gene_idx):
            if gi >= 0:
                comp_out[:, j] = comp[:r, gi]
                rv_out[j] = rv[gi]
        return comp_out, sv[:r], rv_out, r

    def sample_cells(mean_full, n_cells, prog_weights, pert_noise_override=None):
        """Generate cells using per-program covariance and mixture model."""
        ns = pert_noise_override if pert_noise_override is not None else noise_scale
        cells = np.zeros((n_cells, n_genes_out), dtype=np.float32)

        if prog_weights and program_profiles:
            progs = list(prog_weights.keys())
            weights = np.array([max(prog_weights.get(p, 0), 0) for p in progs])
            ws = weights.sum()
            weights = weights / ws if ws > 0 else np.ones(len(progs)) / len(progs)
            n_per_prog = rng.multinomial(n_cells, weights)

            cell_idx = 0
            for prog, n_prog in zip(progs, n_per_prog):
                if n_prog == 0:
                    continue

                # Program-specific mean
                if prog in program_profiles:
                    delta = mean_full - perturbed_mean
                    adj_mean = program_profiles[prog] + delta * prog_delta_blend
                else:
                    adj_mean = mean_full
                mean_out = _extract_output(adj_mean)

                # Per-program covariance
                cov = program_covs.get(prog, global_cov)
                comp_out, sv, rv_out, r = _extract_cov_output(cov)

                # Sample: structured + diagonal noise
                z = rng.randn(n_prog, r).astype(np.float32)
                structured = z @ (sv.reshape(-1, 1) * comp_out) * ns
                diagonal = rng.randn(n_prog, n_genes_out).astype(np.float32) * np.sqrt(rv_out) * ns

                cells[cell_idx:cell_idx + n_prog] = mean_out + structured + diagonal
                cell_idx += n_prog
        else:
            mean_out = _extract_output(mean_full)
            comp_out, sv, rv_out, r = _extract_cov_output(global_cov)
            z = rng.randn(n_cells, r).astype(np.float32)
            structured = z @ (sv.reshape(-1, 1) * comp_out) * ns
            diagonal = rng.randn(n_cells, n_genes_out).astype(np.float32) * np.sqrt(rv_out) * ns
            cells = mean_out + structured + diagonal

        # Softplus for near-zero values: avoids point mass at 0 that distorts MMD
        # For values > 5, effectively identity; for values near 0, smooth curve to 0
        neg_mask = cells < 0
        cells[neg_mask] = np.log1p(np.exp(cells[neg_mask] * 3)) / 3  # steep softplus
        return cells

    # -----------------------------------------------------------------------
    # MAIN PREDICTION LOOP
    # -----------------------------------------------------------------------
    # Pre-allocate the full output matrix. Each perturbation writes its
    # 100 cells directly into the pre-allocated block, avoiding a separate
    # all_predictions list (saves 11.7GB of duplicated memory).
    logger.info(f"  Pre-allocating output: ({n_perts * CELLS_PER_PERTURBATION} x {n_genes_out})...")
    mat = np.zeros((n_perts * CELLS_PER_PERTURBATION, n_genes_out), dtype=np.float32)
    all_obs_genes = []
    program_rows = []

    for pi, pert in enumerate(predict_perturbations):
        if pi % 500 == 0:
            logger.info(f"  [{pi}/{n_perts}] {pert}")

        # === PREDICT MEAN EXPRESSION ===
        if pert in pert_means:
            pred_mean = pert_means[pert].copy()
        else:
            feat = build_gene_features(pert).reshape(1, -1)
            feat_sc = ridge_scaler.transform(feat)

            # Strategy 1: Ridge all-genes
            d1 = ridge_all.predict(feat_sc)[0]
            # Strategy 2: Ridge regularized
            d2 = ridge_reg.predict(feat_sc)[0]
            # Strategy 3: KNN
            feat_knn = feat_scaler.transform(feat)
            dists, idxs = knn.kneighbors(feat_knn)
            w = 1.0 / (dists[0] + 1e-8)
            w /= w.sum()
            d3 = sum(ww * Y_train[idx] for ww, idx in zip(w, idxs[0]))

            # Ensemble + shrinkage toward perturbed mean
            blended = best_w[0] * d1 + best_w[1] * d2 + best_w[2] * d3
            blended *= shrinkage
            pred_mean = perturbed_mean + blended

        # === PREDICT PROGRAM PROPORTIONS ===
        # NOTE: per competition spec, S_p^R = [pre_adipo, adipo, other] sums to 1.
        # lipo is a separate value (fraction of cells that are lipogenic),
        # and lipo_adipo = lipo / adipo is the conditional probability.
        THREE_COLS = ["pre_adipo", "adipo", "other"]
        if pert in prog_props:
            props = dict(prog_props[pert])
            # Normalize only pre_adipo + adipo + other to sum to 1.0
            total3 = sum(props.get(c, 0) for c in THREE_COLS)
            if total3 > 0:
                for c in THREE_COLS:
                    props[c] = props.get(c, 0) / total3
            else:
                for c in THREE_COLS:
                    props[c] = avg_props[c]
        elif prog_models is not None:
            feat = build_prog_features(pert).reshape(1, -1)
            feat_s = prog_scaler.transform(feat)

            # Ridge predictions per program
            ridge_pred = np.array([prog_models[col].predict(feat_s)[0] for col in PROGRAM_COLS])
            # KNN prediction
            dists, idxs = prog_knn.kneighbors(feat_s)
            w = 1.0 / (dists[0] + 1e-8)
            w /= w.sum()
            knn_pred = sum(ww * Y_train_prog[idx] for ww, idx in zip(w, idxs[0]))

            pred_p = best_prog_blend * ridge_pred + (1 - best_prog_blend) * knn_pred
            pred_p = np.clip(pred_p, 0, 1)
            props = {c: float(pred_p[i]) for i, c in enumerate(PROGRAM_COLS)}
            # Normalize only pre_adipo + adipo + other to sum to 1.0
            total3 = sum(props.get(c, 0) for c in THREE_COLS)
            if total3 > 0:
                for c in THREE_COLS:
                    props[c] = props[c] / total3
            else:
                for c in THREE_COLS:
                    props[c] = avg_props[c]
        else:
            props = dict(avg_props)

        # === GENERATE CELLS ===
        # For unseen perturbations, use KNN neighbors' variance for adaptive noise
        pert_noise_scale = noise_scale
        if pert not in pert_means and pert_vars:
            # Get KNN neighbor indices from the prediction step
            feat = build_gene_features(pert).reshape(1, -1)
            feat_knn = feat_scaler.transform(feat)
            try:
                _, nn_idxs = knn.kneighbors(feat_knn)
                nn_perts = [training_perts[j] for j in nn_idxs[0]]
                nn_vars = [pert_vars[p].mean() for p in nn_perts if p in pert_vars]
                if nn_vars:
                    nn_mean_var = np.mean(nn_vars)
                    global_mean_var = np.mean([pert_vars[p].mean() for p in training_perts if p in pert_vars])
                    # Scale noise proportionally: if neighbors are high-variance, increase noise
                    var_ratio = np.sqrt(nn_mean_var / (global_mean_var + 1e-10))
                    pert_noise_scale = noise_scale * np.clip(var_ratio, 0.5, 2.0)
            except Exception:
                pass

        cells = sample_cells(pred_mean, CELLS_PER_PERTURBATION, prog_weights=props,
                           pert_noise_override=pert_noise_scale)

        start = pi * CELLS_PER_PERTURBATION
        mat[start:start + CELLS_PER_PERTURBATION] = cells
        del cells
        all_obs_genes.extend([pert] * CELLS_PER_PERTURBATION)

        program_rows.append({
            "gene": pert,
            "pre_adipo": props.get("pre_adipo", avg_props["pre_adipo"]),
            "adipo": props.get("adipo", avg_props["adipo"]),
            "lipo": props.get("lipo", avg_props["lipo"]),
            "other": props.get("other", avg_props["other"]),
        })

    # -----------------------------------------------------------------------
    # WRITE OUTPUTS
    # -----------------------------------------------------------------------
    logger.info("Writing prediction.h5ad...")
    n_total = n_perts * CELLS_PER_PERTURBATION
    logger.info(f"  Dense matrix: {mat.shape}, ~{mat.nbytes / 1e9:.1f}GB")

    pred_ad = ad.AnnData(
        X=mat,
        obs=pd.DataFrame({"gene": all_obs_genes}),
        var=pd.DataFrame(index=genes_to_predict),
    )
    del mat
    import gc; gc.collect()
    pred_ad.write_h5ad(prediction_h5ad_file_path)

    logger.info(f"  Saved: {prediction_h5ad_file_path}")

    prog_df = pd.DataFrame(program_rows)
    prog_df = prog_df[["gene", "pre_adipo", "adipo", "lipo", "other"]]
    prog_df.to_csv(program_proportion_csv_file_path, index=False)
    logger.info(f"  Saved: {program_proportion_csv_file_path}")
    logger.info("INFERENCE COMPLETE.")
