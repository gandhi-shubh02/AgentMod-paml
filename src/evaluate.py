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

