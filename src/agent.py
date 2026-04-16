from typing import Dict, List

import numpy as np

try:
    from preprocessing import clean_text
except ImportError:
    from src.preprocessing import clean_text


RANK_TO_LABEL = {
    0: "Low-activity benign user",
    1: "High-activity benign user",
    2: "Occasional flagged user",
    3: "Repeat offender",
}


class ModerationAgent:
    def __init__(self, nn_model, kmeans_model, tfidf_vectorizer, user_features, scaler, alpha=0.7):
        self.nn_model = nn_model
        self.kmeans_model = kmeans_model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.user_features: Dict[int, np.ndarray] = user_features
        self.scaler = scaler
        self.alpha = float(alpha)

        self.cluster_to_risk = self._build_cluster_risk_mapping()
        self.cluster_to_label = {
            c: RANK_TO_LABEL[risk] for c, risk in self.cluster_to_risk.items()
        }

    def _build_cluster_risk_mapping(self):
        user_ids = sorted(self.user_features.keys())
        feats = np.vstack([self.user_features[int(uid)] for uid in user_ids]).astype(np.float64)
        scaled = self.scaler.transform(feats)
        clusters = self.kmeans_model.predict(scaled)

        cluster_mean_toxic_rate = {}
        for c in np.unique(clusters):
            idx = np.where(clusters == c)[0]
            # Toxic rate feature is at index 1 in original (unscaled) user features.
            cluster_mean_toxic_rate[int(c)] = float(feats[idx, 1].mean()) if idx.size > 0 else 0.0

        sorted_clusters = sorted(cluster_mean_toxic_rate, key=lambda c: cluster_mean_toxic_rate[c])
        cluster_to_risk = {cluster_id: rank for rank, cluster_id in enumerate(sorted_clusters)}
        return cluster_to_risk

    def _get_cluster_risk(self, user_id):
        if int(user_id) in self.user_features:
            feat = self.user_features[int(user_id)]
        else:
            all_feats = np.vstack(list(self.user_features.values()))
            feat = all_feats.mean(axis=0)

        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        cluster_id = int(self.kmeans_model.predict(feat_scaled)[0])
        risk_rank = int(self.cluster_to_risk.get(cluster_id, 0))
        return cluster_id, risk_rank

    def _compute_combined_score(self, p_nn, risk_rank):
        return self.alpha * float(p_nn) + (1.0 - self.alpha) * (float(risk_rank) / 3.0)

    def _make_decision(self, s):
        if s < 0.4:
            return "Allow"
        if s < 0.7:
            return "Flag for Review"
        return "Auto-Remove"

    def run(self, comment_text, user_id):
        trace: List[str] = []

        trace.append("Step 1 — NN classifier: preprocessing comment...")
        cleaned = clean_text(comment_text)
        x = self.tfidf_vectorizer.transform([cleaned])
        p_nn = float(self.nn_model.predict_proba(x)[0])
        trace.append(f"Step 1 — NN toxicity score: {p_nn:.4f}")

        trace.append(f"Step 2 — Looking up user {int(user_id)} behavioral cluster...")
        cluster_id, risk_rank = self._get_cluster_risk(int(user_id))
        cluster_label = self.cluster_to_label.get(cluster_id, "Unknown cluster")
        trace.append(f"Step 2 — Cluster: {cluster_label} (risk rank {risk_rank}/3)")

        s = self._compute_combined_score(p_nn, risk_rank)
        decision = self._make_decision(s)
        trace.append(
            f"Step 3 — Combined score: s = {self.alpha:.1f}×{p_nn:.3f} + {1-self.alpha:.1f}×{risk_rank}/3 = {s:.4f}"
        )
        trace.append("Step 3 — Decision threshold: Allow(<0.4) / Flag(0.4-0.7) / Remove(≥0.7)")
        trace.append(f"Step 3 — Final decision: {decision}")

        return {
            "comment": str(comment_text),
            "user_id": int(user_id),
            "step1_nn_score": p_nn,
            "step2_cluster_id": int(cluster_id),
            "step2_cluster_label": str(cluster_label),
            "step2_risk_rank": int(risk_rank),
            "step3_combined_score": float(s),
            "step3_alpha": float(self.alpha),
            "decision": str(decision),
            "reasoning_trace": trace,
        }

