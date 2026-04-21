import numpy as np


def _precision_recall(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def compute_f1(y_true, y_pred):
    precision, recall = _precision_recall(y_true, y_pred)
    return float((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)


def compute_fpr(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def compute_auc_roc(y_true, y_scores):
    y_true = np.asarray(y_true).astype(np.int64)
    y_scores = np.asarray(y_scores).astype(np.float64)
    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        return [], [], float("nan")

    thresholds = np.unique(y_scores)[::-1]
    thresholds = np.concatenate(([1.0 + 1e-12], thresholds, [-1e-12]))

    fpr_list = []
    tpr_list = []
    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(np.int64)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_list.append(float(tpr))
        fpr_list.append(float(fpr))

    order = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[order]
    tpr_sorted = np.array(tpr_list)[order]
    auc_score = float(np.trapz(tpr_sorted, fpr_sorted))
    return fpr_sorted.tolist(), tpr_sorted.tolist(), auc_score


def full_evaluation(y_true, y_pred, y_scores, label="Model"):
    precision, recall = _precision_recall(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    fpr = compute_fpr(y_true, y_pred)
    fpr_curve, tpr_curve, auc = compute_auc_roc(y_true, y_scores)

    print(f"=== {label} Evaluation ===")
    print(f"F1        : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"FPR       : {fpr:.4f}")
    if np.isnan(auc):
        print("AUC-ROC   : N/A (single-class subset)")
    else:
        print(f"AUC-ROC   : {auc:.4f}")

    return {
        "label": label,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "auc": float(auc),
        "roc_fpr": np.array(fpr_curve, dtype=np.float64),
        "roc_tpr": np.array(tpr_curve, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Label-aware clustering metrics (from scratch, NumPy only)
# ---------------------------------------------------------------------------
# The K-means model is unsupervised, but when we have ground-truth labels for
# the items being clustered we can judge whether clusters align with those
# labels. We implement three standard external metrics: purity, Adjusted Rand
# Index (ARI), and Normalized Mutual Information (NMI).


def _contingency_matrix(cluster_labels, y_true):
    cluster_labels = np.asarray(cluster_labels).astype(np.int64).ravel()
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    if cluster_labels.shape[0] != y_true.shape[0]:
        raise ValueError("cluster_labels and y_true must have the same length")

    clusters = np.unique(cluster_labels)
    classes = np.unique(y_true)
    table = np.zeros((clusters.size, classes.size), dtype=np.int64)
    for i, c in enumerate(clusters):
        mask = cluster_labels == c
        for j, k in enumerate(classes):
            table[i, j] = int(np.sum(mask & (y_true == k)))
    return table, clusters, classes


def compute_cluster_purity(cluster_labels, y_true):
    """Purity = (1/N) * sum_c max_k |cluster c ∩ class k|.

    Range: [0, 1]. Higher is better. Unlike ARI/NMI, purity is not corrected
    for chance and can be trivially high when K is large.
    """
    table, _, _ = _contingency_matrix(cluster_labels, y_true)
    n = table.sum()
    if n == 0:
        return 0.0
    return float(table.max(axis=1).sum()) / float(n)


def _comb2(x):
    # n choose 2, vectorised, for int arrays
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def compute_ari(cluster_labels, y_true):
    """Adjusted Rand Index.

    ARI in [-1, 1]. 0 means random-level agreement; 1 means perfect agreement
    (up to a permutation of cluster ids).
    """
    table, _, _ = _contingency_matrix(cluster_labels, y_true)
    n = table.sum()
    if n < 2:
        return 0.0

    sum_comb_cells = float(_comb2(table).sum())
    sum_comb_rows = float(_comb2(table.sum(axis=1)).sum())
    sum_comb_cols = float(_comb2(table.sum(axis=0)).sum())
    total_comb = float(_comb2(np.array([n]))[0])

    if total_comb == 0.0:
        return 0.0

    expected = sum_comb_rows * sum_comb_cols / total_comb
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        # Degenerate case (e.g. single cluster or single class): define ARI=0.
        return 0.0
    return float((sum_comb_cells - expected) / denom)


def compute_nmi(cluster_labels, y_true):
    """Normalized Mutual Information (arithmetic-mean normalization).

    NMI in [0, 1]. 1 means cluster assignment fully determines the label.
    """
    table, _, _ = _contingency_matrix(cluster_labels, y_true)
    n = float(table.sum())
    if n == 0:
        return 0.0

    row_sums = table.sum(axis=1).astype(np.float64)
    col_sums = table.sum(axis=0).astype(np.float64)

    # Entropies (natural log; the ratio is scale-invariant).
    def _entropy(counts):
        counts = counts[counts > 0]
        p = counts / n
        return float(-(p * np.log(p)).sum())

    h_u = _entropy(row_sums)
    h_v = _entropy(col_sums)
    if h_u <= 0.0 or h_v <= 0.0:
        return 0.0

    # Mutual information: sum over nonzero cells.
    mi = 0.0
    nz = np.argwhere(table > 0)
    for i, j in nz:
        nij = float(table[i, j])
        mi += (nij / n) * np.log((nij * n) / (row_sums[i] * col_sums[j]))

    return float(2.0 * mi / (h_u + h_v))


def cluster_quality_report(
    cluster_labels,
    y_true,
    inertia=None,
    silhouette=None,
    label="K-means",
):
    """Print and return a combined intrinsic + label-aware cluster report."""
    purity = compute_cluster_purity(cluster_labels, y_true)
    ari = compute_ari(cluster_labels, y_true)
    nmi = compute_nmi(cluster_labels, y_true)

    print(f"=== {label} Cluster Quality ===")
    if inertia is not None:
        print(f"Inertia     : {float(inertia):.4f}")
    if silhouette is not None:
        print(f"Silhouette  : {float(silhouette):.4f}")
    print(f"Purity      : {purity:.4f}")
    print(f"ARI         : {ari:.4f}")
    print(f"NMI         : {nmi:.4f}")

    return {
        "label": label,
        "inertia": None if inertia is None else float(inertia),
        "silhouette": None if silhouette is None else float(silhouette),
        "purity": float(purity),
        "ari": float(ari),
        "nmi": float(nmi),
    }

