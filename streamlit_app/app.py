import os
import pickle
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from agent import ModerationAgent
from kmeans import KMeans
from neural_network import NeuralNetwork
from preprocessing import MinMaxScaler
from tfidf import TFIDFVectorizer


def _decision_color(decision: str) -> str:
    if decision == "Allow":
        return "green"
    if decision == "Flag for Review":
        return "orange"
    return "red"


def _risk_color(rank: int) -> str:
    return {0: "green", 1: "yellow", 2: "orange", 3: "red"}.get(rank, "gray")


@st.cache_resource
def load_artifacts():
    tfidf = TFIDFVectorizer(max_features=10000, ngram_range=(1, 2))
    tfidf.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.npz"))

    nn = NeuralNetwork(layer_sizes=[10000, 256, 128, 1], dropout_rate=0.3, seed=42)
    nn.load(os.path.join(MODELS_DIR, "nn_weights.npz"))

    km = KMeans(K=4, max_iter=100, tol=1e-4, seed=42)
    km.load(os.path.join(MODELS_DIR, "kmeans_centroids.npy"))

    scaler = MinMaxScaler()
    scaler.load(os.path.join(MODELS_DIR, "scaler.npz"))

    with open(os.path.join(MODELS_DIR, "user_features.pkl"), "rb") as f:
        user_features = pickle.load(f)

    return nn, km, tfidf, user_features, scaler


def map_slider_to_user(slider_value, user_features):
    target = slider_value / 100.0
    user_ids = list(user_features.keys())
    toxic_rates = np.array([user_features[uid][1] for uid in user_ids], dtype=np.float64)
    idx = int(np.argmin(np.abs(toxic_rates - target)))
    return int(user_ids[idx])


def page_live_demo(agent, user_features):
    st.title("AgentMod — Live Content Moderation")
    st.caption("Enter a comment to see the agent's moderation decision")

    comment = st.text_area("Enter a social media comment", height=140)
    slider = st.slider("Simulated user risk profile (prior toxic rate %)", min_value=0, max_value=100, value=20)

    if st.button("Analyze Comment"):
        user_id = map_slider_to_user(slider, user_features)
        result = agent.run(comment, user_id=user_id)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("NN Toxicity Score", value=f"{result['step1_nn_score']:.3f}")
            st.progress(float(np.clip(result["step1_nn_score"], 0.0, 1.0)))
        with c2:
            st.metric("User Cluster", value=result["step2_cluster_label"])
            badge_color = _risk_color(result["step2_risk_rank"])
            st.markdown(
                f"<span style='background-color:{badge_color};padding:4px 8px;border-radius:6px;'>"
                f"Risk Rank: {result['step2_risk_rank']}/3</span>",
                unsafe_allow_html=True,
            )
        with c3:
            st.metric("Combined Score", value=f"{result['step3_combined_score']:.3f}")
            color = _decision_color(result["decision"])
            st.markdown(
                f"<span style='color:{color};font-weight:700;'>Decision: {result['decision']}</span>",
                unsafe_allow_html=True,
            )

        with st.expander("Agent Reasoning Trace", expanded=True):
            for i, step in enumerate(result["reasoning_trace"], start=1):
                st.code(f"{i}. {step}")


def _metric_delta(new, base, higher_is_better=True):
    delta = new - base
    if not higher_is_better:
        delta = -delta
    return f"{delta:+.4f}"


def page_benchmark_dashboard():
    path = os.path.join(MODELS_DIR, "benchmark_results.npz")
    if not os.path.exists(path):
        st.error("Missing models/benchmark_results.npz. Run the notebook end-to-end first.")
        return

    bench = np.load(path, allow_pickle=True)
    full_nn = bench["full_nn"].item()
    full_agent = bench["full_agent"].item()
    border_nn = bench["border_nn"].item()
    border_agent = bench["border_agent"].item()
    alpha_values = bench["alpha_values"]
    alpha_fpr = bench["alpha_fpr"]

    st.title("Benchmark Dashboard")
    st.caption("AgentMod vs. NN-only Baseline")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Full Test Set")
        st.metric("F1 (AgentMod)", f"{full_agent['f1']:.4f}", _metric_delta(full_agent["f1"], full_nn["f1"], True))
        st.metric("FPR (AgentMod)", f"{full_agent['fpr']:.4f}", _metric_delta(full_agent["fpr"], full_nn["fpr"], False))
        st.metric("AUC-ROC (AgentMod)", f"{full_agent['auc']:.4f}")
    with col_b:
        st.subheader("Borderline Subset")
        st.metric("F1 (AgentMod)", f"{border_agent['f1']:.4f}", _metric_delta(border_agent["f1"], border_nn["f1"], True))
        st.metric("FPR (AgentMod)", f"{border_agent['fpr']:.4f}", _metric_delta(border_agent["fpr"], border_nn["fpr"], False))
        st.metric("AUC-ROC (AgentMod)", f"{border_agent['auc']:.4f}")

    roc_fig = go.Figure()
    roc_fig.add_trace(
        go.Scatter(
            x=full_nn["roc_fpr"],
            y=full_nn["roc_tpr"],
            mode="lines",
            line=dict(color="lightblue", dash="dash"),
            name="NN-only",
        )
    )
    roc_fig.add_trace(
        go.Scatter(
            x=full_agent["roc_fpr"],
            y=full_agent["roc_tpr"],
            mode="lines",
            line=dict(color="darkblue", dash="solid"),
            name="AgentMod",
        )
    )
    roc_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="gray", dash="dot"), name="Random")
    )
    roc_fig.update_layout(title="AUC-ROC Curves", xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(roc_fig, use_container_width=True)

    alpha_fig = go.Figure()
    alpha_fig.add_trace(
        go.Scatter(x=alpha_values, y=alpha_fpr, mode="lines+markers", line=dict(color="darkblue"), name="Borderline FPR")
    )
    alpha_fig.add_vline(x=0.7, line_dash="dash", line_color="gray")
    alpha_fig.update_layout(title="Alpha Ablation: Borderline FPR vs Alpha", xaxis_title="alpha", yaxis_title="FPR")
    st.plotly_chart(alpha_fig, use_container_width=True)

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(name="NN-only", x=["Full", "Borderline"], y=[full_nn["f1"], border_nn["f1"]], marker_color="lightblue"))
    bar_fig.add_trace(go.Bar(name="AgentMod", x=["Full", "Borderline"], y=[full_agent["f1"], border_agent["f1"]], marker_color="darkblue"))
    bar_fig.update_layout(barmode="group", title="F1 Comparison by Subset", yaxis_title="F1")
    st.plotly_chart(bar_fig, use_container_width=True)


def main():
    st.set_page_config(page_title="AgentMod", layout="wide")

    try:
        nn, km, tfidf, user_features, scaler = load_artifacts()
    except Exception as exc:
        st.error(f"Failed to load model artifacts from models/: {exc}")
        st.stop()

    page = st.sidebar.radio("Select Page", ["Live Moderation Demo", "Benchmark Dashboard"])
    agent = ModerationAgent(nn, km, tfidf, user_features, scaler, alpha=0.7)

    if page == "Live Moderation Demo":
        page_live_demo(agent, user_features)
    else:
        page_benchmark_dashboard()


if __name__ == "__main__":
    main()

