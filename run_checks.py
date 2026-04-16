import os
import pickle
import sys

import numpy as np


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> int:
    root = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(root, "data")
    models_dir = os.path.join(root, "models")
    src_dir = os.path.join(root, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    # Local imports after sys.path setup.
    from agent import ModerationAgent
    from kmeans import KMeans
    from neural_network import NeuralNetwork
    from preprocessing import MinMaxScaler
    from tfidf import TFIDFVectorizer

    print("=== AgentMod Checks ===")

    dataset_path = os.path.join(data_dir, "train.csv")
    if os.path.exists(dataset_path):
        _ok("Dataset found: data/train.csv")
    else:
        _warn("Dataset missing: data/train.csv (needed to train from notebook).")

    required_model_files = [
        "nn_weights.npz",
        "kmeans_centroids.npy",
        "tfidf_vectorizer.npz",
        "scaler.npz",
        "user_features.pkl",
        "benchmark_results.npz",
    ]

    missing = []
    for name in required_model_files:
        full = os.path.join(models_dir, name)
        if os.path.exists(full):
            _ok(f"Found models/{name}")
        else:
            missing.append(name)
            _warn(f"Missing models/{name}")

    if missing:
        _warn("Some artifacts are missing. Run notebooks/train_and_evaluate.ipynb end-to-end first.")
        # Still continue with what we can check.

    nn = None
    kmeans = None
    tfidf = None
    scaler = None
    user_features = None

    try:
        tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.npz")
        if os.path.exists(tfidf_path):
            tfidf = TFIDFVectorizer(max_features=10000, ngram_range=(1, 2))
            tfidf.load(tfidf_path)
            _ok("Loaded TF-IDF vectorizer")
    except Exception as exc:
        _fail(f"Could not load TF-IDF vectorizer: {exc}")
        return 1

    try:
        nn_path = os.path.join(models_dir, "nn_weights.npz")
        if os.path.exists(nn_path):
            nn = NeuralNetwork(layer_sizes=[10000, 256, 128, 1], dropout_rate=0.3, seed=42)
            nn.load(nn_path)
            _ok("Loaded neural network weights")
    except Exception as exc:
        _fail(f"Could not load neural network weights: {exc}")
        return 1

    try:
        km_path = os.path.join(models_dir, "kmeans_centroids.npy")
        if os.path.exists(km_path):
            kmeans = KMeans(K=4, max_iter=100, tol=1e-4, seed=42)
            kmeans.load(km_path)
            _ok("Loaded K-means centroids")
    except Exception as exc:
        _fail(f"Could not load K-means centroids: {exc}")
        return 1

    try:
        scaler_path = os.path.join(models_dir, "scaler.npz")
        if os.path.exists(scaler_path):
            scaler = MinMaxScaler()
            scaler.load(scaler_path)
            _ok("Loaded MinMax scaler")
    except Exception as exc:
        _fail(f"Could not load scaler: {exc}")
        return 1

    try:
        uf_path = os.path.join(models_dir, "user_features.pkl")
        if os.path.exists(uf_path):
            with open(uf_path, "rb") as f:
                user_features = pickle.load(f)
            if not isinstance(user_features, dict) or not user_features:
                _fail("user_features.pkl is empty or invalid.")
                return 1
            _ok(f"Loaded user features for {len(user_features)} users")
    except Exception as exc:
        _fail(f"Could not load user features: {exc}")
        return 1

    if all(x is not None for x in [nn, kmeans, tfidf, scaler, user_features]):
        try:
            agent = ModerationAgent(nn, kmeans, tfidf, user_features, scaler, alpha=0.7)
            sample_user_id = int(sorted(user_features.keys())[0])
            sample_text = "You are amazing and this is a nice helpful comment."
            result = agent.run(sample_text, sample_user_id)
            required_keys = {
                "comment",
                "user_id",
                "step1_nn_score",
                "step2_cluster_id",
                "step2_cluster_label",
                "step2_risk_rank",
                "step3_combined_score",
                "step3_alpha",
                "decision",
                "reasoning_trace",
            }
            missing_keys = required_keys.difference(result.keys())
            if missing_keys:
                _fail(f"Smoke test failed: missing result keys {sorted(missing_keys)}")
                return 1
            _ok("Agent smoke test passed")
            print(
                "Sample output:",
                {
                    "decision": result["decision"],
                    "nn_score": round(float(result["step1_nn_score"]), 4),
                    "combined_score": round(float(result["step3_combined_score"]), 4),
                },
            )
        except Exception as exc:
            _fail(f"Agent smoke test failed: {exc}")
            return 1
    else:
        _warn("Skipped smoke test because one or more model artifacts were missing.")

    bench_path = os.path.join(models_dir, "benchmark_results.npz")
    if os.path.exists(bench_path):
        try:
            bench = np.load(bench_path, allow_pickle=True)
            for key in ["full_nn", "full_agent", "border_nn", "border_agent", "alpha_values", "alpha_fpr"]:
                if key not in bench:
                    _fail(f"benchmark_results.npz missing key: {key}")
                    return 1
            _ok("Benchmark file contains expected keys")
        except Exception as exc:
            _fail(f"Could not validate benchmark_results.npz: {exc}")
            return 1

    print("=== Checks completed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

