"""Agentic moderation pipeline.

The course constraint forbids high-level ML libraries and LLM frameworks
(no CrewAI / AutoGen / LangGraph / LLM API calls). To keep the agentic
structure explicit in the code anyway, we split responsibilities across
three small single-purpose "role" classes and one coordinator:

    ┌───────────────────────────┐        ┌───────────────────────────┐
    │  TextToxicityAgent        │        │  UserRiskAgent            │
    │  - owns NN + TF-IDF       │        │  - owns K-means + scaler  │
    │  - produces p_nn ∈ [0,1]  │        │  - produces risk_rank 0-3 │
    └─────────────┬─────────────┘        └─────────────┬─────────────┘
                  │                                    │
                  └──────────────┬─────────────────────┘
                                 ▼
                       ┌───────────────────────┐
                       │  PolicyAgent          │
                       │  - gated fusion score │
                       │  - Allow/Flag/Remove  │
                       └──────────┬────────────┘
                                  ▼
                       ┌───────────────────────┐
                       │  ModerationAgent      │
                       │  coordinator + trace  │
                       └───────────────────────┘

Communication between roles is a plain Python dict passed step-by-step.
This is the same idea as message-passing between agents in frameworks
like CrewAI / LangGraph, just written from scratch.
"""

from typing import Any, Dict, List

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


# ---------------------------------------------------------------------------
# Role 1: text toxicity
# ---------------------------------------------------------------------------
class TextToxicityAgent:
    """Owns the text model: TF-IDF vectorizer + neural network classifier.

    Input : raw comment string.
    Output: {'p_nn': float in [0,1], 'cleaned_text': str}
    """

    name = "TextToxicityAgent"

    def __init__(self, nn_model, tfidf_vectorizer):
        self.nn_model = nn_model
        self.tfidf_vectorizer = tfidf_vectorizer

    def act(self, comment_text: str) -> Dict[str, Any]:
        cleaned = clean_text(str(comment_text))
        x = self.tfidf_vectorizer.transform([cleaned])
        p_nn = float(self.nn_model.predict_proba(x)[0])
        return {"p_nn": p_nn, "cleaned_text": cleaned}


# ---------------------------------------------------------------------------
# Role 2: user risk
# ---------------------------------------------------------------------------
class UserRiskAgent:
    """Owns the user model: MinMaxScaler + K-means cluster → risk rank.

    Input : user_id (int).
    Output: {'cluster_id': int, 'risk_rank': int in 0..K-1,
             'cluster_label': str}

    Risk ranks are derived once at construction time by sorting clusters
    on their mean `prior_toxic_rate` (lower rank = safer cluster).
    """

    name = "UserRiskAgent"

    def __init__(self, kmeans_model, scaler, user_features: Dict[int, np.ndarray]):
        self.kmeans_model = kmeans_model
        self.scaler = scaler
        self.user_features = user_features
        self.cluster_to_risk = self._build_cluster_risk_mapping()
        self.cluster_to_label = {
            c: RANK_TO_LABEL.get(risk, f"Cluster {c} (risk {risk})")
            for c, risk in self.cluster_to_risk.items()
        }

    def _build_cluster_risk_mapping(self) -> Dict[int, int]:
        user_ids = sorted(self.user_features.keys())
        feats = np.vstack(
            [self.user_features[int(uid)] for uid in user_ids]
        ).astype(np.float64)
        scaled = self.scaler.transform(feats)
        clusters = self.kmeans_model.predict(scaled)

        cluster_mean_toxic_rate: Dict[int, float] = {}
        for c in np.unique(clusters):
            idx = np.where(clusters == c)[0]
            # Feature index 1 = prior_toxic_rate (unscaled).
            cluster_mean_toxic_rate[int(c)] = (
                float(feats[idx, 1].mean()) if idx.size > 0 else 0.0
            )

        sorted_clusters = sorted(
            cluster_mean_toxic_rate, key=lambda c: cluster_mean_toxic_rate[c]
        )
        return {cid: rank for rank, cid in enumerate(sorted_clusters)}

    def act(self, user_id: int) -> Dict[str, Any]:
        if int(user_id) in self.user_features:
            feat = self.user_features[int(user_id)]
        else:
            all_feats = np.vstack(list(self.user_features.values()))
            feat = all_feats.mean(axis=0)

        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        cluster_id = int(self.kmeans_model.predict(feat_scaled)[0])
        risk_rank = int(self.cluster_to_risk.get(cluster_id, 0))
        cluster_label = str(self.cluster_to_label.get(cluster_id, "Unknown cluster"))
        return {
            "cluster_id": cluster_id,
            "risk_rank": risk_rank,
            "cluster_label": cluster_label,
        }


# ---------------------------------------------------------------------------
# Role 3: policy (fusion + decision)
# ---------------------------------------------------------------------------
class PolicyAgent:
    """Owns the decision policy: combined score + action thresholds.

    Fusion is a "gated" weighted sum: the user-risk signal is only mixed in
    when the text model is uncertain (|p_nn - 0.5| <= gate_width). Outside
    that band the decision is driven by the text score alone, which keeps
    the user signal from overriding confident text evidence.

        s = p_nn                                if |p_nn - 0.5| > gate_width
        s = α·p_nn + (1-α)·(risk_rank / K_max)  otherwise

    Action thresholds (low_thr, high_thr) map s → Allow / Flag / Remove.
    """

    name = "PolicyAgent"

    def __init__(
        self,
        alpha: float = 0.7,
        gate_width: float = 0.5,
        low_thr: float = 0.4,
        high_thr: float = 0.7,
        max_risk_rank: int = 3,
    ):
        self.alpha = float(alpha)
        self.gate_width = float(gate_width)
        self.low_thr = float(low_thr)
        self.high_thr = float(high_thr)
        self.max_risk_rank = int(max_risk_rank)

    def combined_score(self, p_nn: float, risk_rank: int) -> float:
        p_nn = float(p_nn)
        if abs(p_nn - 0.5) > self.gate_width:
            return p_nn
        risk_norm = float(risk_rank) / max(1, self.max_risk_rank)
        return self.alpha * p_nn + (1.0 - self.alpha) * risk_norm

    def decide(self, s: float) -> str:
        if s < self.low_thr:
            return "Allow"
        if s < self.high_thr:
            return "Flag for Review"
        return "Auto-Remove"

    def act(self, p_nn: float, risk_rank: int) -> Dict[str, Any]:
        s = self.combined_score(p_nn, risk_rank)
        return {"combined_score": float(s), "decision": self.decide(s)}


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------
class ModerationAgent:
    """Coordinator that invokes the three role-agents in sequence and
    produces a step-by-step reasoning trace.

    Backward-compatible with the original single-class API: constructor
    signature and `run(comment_text, user_id)` output keys are unchanged.
    """

    def __init__(
        self,
        nn_model,
        kmeans_model,
        tfidf_vectorizer,
        user_features,
        scaler,
        alpha: float = 0.7,
        gate_width: float = 0.5,
        low_thr: float = 0.4,
        high_thr: float = 0.7,
    ):
        self.text_agent = TextToxicityAgent(nn_model, tfidf_vectorizer)
        self.user_agent = UserRiskAgent(kmeans_model, scaler, user_features)
        self.policy_agent = PolicyAgent(
            alpha=alpha,
            gate_width=gate_width,
            low_thr=low_thr,
            high_thr=high_thr,
            max_risk_rank=max(1, len(self.user_agent.cluster_to_risk) - 1),
        )

        # Kept for backward compatibility with callers / Streamlit app.
        self.nn_model = nn_model
        self.kmeans_model = kmeans_model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.user_features = user_features
        self.scaler = scaler
        self.alpha = float(alpha)
        self.cluster_to_risk = self.user_agent.cluster_to_risk
        self.cluster_to_label = self.user_agent.cluster_to_label

    # Legacy helpers still used elsewhere (e.g. streamlit_app).
    def _get_cluster_risk(self, user_id):
        r = self.user_agent.act(int(user_id))
        return r["cluster_id"], r["risk_rank"]

    def _compute_combined_score(self, p_nn, risk_rank):
        return self.policy_agent.combined_score(float(p_nn), int(risk_rank))

    def _make_decision(self, s):
        return self.policy_agent.decide(float(s))

    def run(self, comment_text: str, user_id: int) -> Dict[str, Any]:
        trace: List[str] = []

        trace.append(f"[{self.text_agent.name}] preprocessing comment and scoring...")
        text_out = self.text_agent.act(comment_text)
        p_nn = float(text_out["p_nn"])
        trace.append(f"[{self.text_agent.name}] NN toxicity score p_nn = {p_nn:.4f}")

        trace.append(f"[{self.user_agent.name}] looking up behavioral cluster for user {int(user_id)}...")
        user_out = self.user_agent.act(int(user_id))
        cluster_id = int(user_out["cluster_id"])
        risk_rank = int(user_out["risk_rank"])
        cluster_label = str(user_out["cluster_label"])
        trace.append(
            f"[{self.user_agent.name}] cluster = {cluster_label} "
            f"(risk rank {risk_rank}/{self.policy_agent.max_risk_rank})"
        )

        policy_out = self.policy_agent.act(p_nn, risk_rank)
        s = float(policy_out["combined_score"])
        decision = str(policy_out["decision"])
        gated = abs(p_nn - 0.5) <= self.policy_agent.gate_width
        if gated:
            trace.append(
                f"[{self.policy_agent.name}] gated fusion: s = "
                f"{self.policy_agent.alpha:.2f}·{p_nn:.3f} + "
                f"{1 - self.policy_agent.alpha:.2f}·{risk_rank}/{self.policy_agent.max_risk_rank} "
                f"= {s:.4f}"
            )
        else:
            trace.append(
                f"[{self.policy_agent.name}] outside uncertainty band "
                f"(|p_nn-0.5|>{self.policy_agent.gate_width:.2f}); s = p_nn = {s:.4f}"
            )
        trace.append(
            f"[{self.policy_agent.name}] thresholds: Allow(<{self.policy_agent.low_thr}) "
            f"/ Flag({self.policy_agent.low_thr}-{self.policy_agent.high_thr}) "
            f"/ Remove(≥{self.policy_agent.high_thr})"
        )
        trace.append(f"[{self.policy_agent.name}] final decision: {decision}")

        return {
            "comment": str(comment_text),
            "user_id": int(user_id),
            "step1_nn_score": p_nn,
            "step2_cluster_id": cluster_id,
            "step2_cluster_label": cluster_label,
            "step2_risk_rank": risk_rank,
            "step3_combined_score": s,
            "step3_alpha": float(self.policy_agent.alpha),
            "step3_gate_width": float(self.policy_agent.gate_width),
            "decision": decision,
            "reasoning_trace": trace,
        }
