from copy import deepcopy
from typing import Dict, List

import numpy as np

try:
    from evaluate import compute_f1
except ImportError:
    from src.evaluate import compute_f1


class NeuralNetwork:
    def __init__(self, layer_sizes, dropout_rate=0.3, seed=42):
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            w = self.rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            b = np.zeros((1, fan_out), dtype=np.float64)
            self.weights.append(w.astype(np.float64))
            self.biases.append(b)

        self.vel_w = [np.zeros_like(w) for w in self.weights]
        self.vel_b = [np.zeros_like(b) for b in self.biases]
        self.cache: Dict[str, List[np.ndarray]] = {}
        self.last_grads_w: List[np.ndarray] | None = None
        self.last_grads_b: List[np.ndarray] | None = None

    def relu(self, z):
        return np.maximum(0.0, z)

    def relu_grad(self, z):
        return (z > 0).astype(np.float64)

    def sigmoid(self, z):
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X, training=True):
        A = X.astype(np.float64)
        activations = [A]
        preacts = []
        dropout_masks = []

        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = self.relu(Z)
            preacts.append(Z)

            if training:
                keep_prob = 1.0 - self.dropout_rate
                mask = (self.rng.random(A.shape) < keep_prob).astype(np.float64) / keep_prob
                A = A * mask
            else:
                mask = np.ones_like(A)
            dropout_masks.append(mask)
            activations.append(A)

        Z_out = A @ self.weights[-1] + self.biases[-1]
        y_hat = self.sigmoid(Z_out)
        preacts.append(Z_out)
        activations.append(y_hat)

        self.cache = {"activations": activations, "preacts": preacts, "dropout_masks": dropout_masks}
        return y_hat

    def compute_loss(self, y_pred, y_true):
        y_true = y_true.reshape(-1, 1).astype(np.float64)
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    def _compute_gradients(self, X, y, training=True):
        y = y.reshape(-1, 1).astype(np.float64)
        y_pred = self.forward(X, training=training)
        acts = self.cache["activations"]
        zs = self.cache["preacts"]
        masks = self.cache["dropout_masks"]
        m = X.shape[0]

        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        delta = (y_pred - y) / m
        grads_w[-1] = acts[-2].T @ delta
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        for l in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[l + 1].T) * self.relu_grad(zs[l])
            delta = delta * masks[l]
            grads_w[l] = acts[l].T @ delta
            grads_b[l] = np.sum(delta, axis=0, keepdims=True)

        return grads_w, grads_b, y_pred

    def backward(self, X, y, learning_rate, momentum=0.9):
        grads_w, grads_b, _ = self._compute_gradients(X, y)
        self.last_grads_w = grads_w
        self.last_grads_b = grads_b

        for i in range(len(self.weights)):
            self.vel_w[i] = momentum * self.vel_w[i] - learning_rate * grads_w[i]
            self.vel_b[i] = momentum * self.vel_b[i] - learning_rate * grads_b[i]
            self.weights[i] += self.vel_w[i]
            self.biases[i] += self.vel_b[i]

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        batch_size=64,
        learning_rate=0.001,
        momentum=0.9,
        patience=5,
    ):
        history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
        n = X_train.shape[0]

        best_state = None
        best_val_f1 = -np.inf
        wait = 0

        for _ in range(epochs):
            order = self.rng.permutation(n)
            X_shuf = X_train[order]
            y_shuf = y_train[order]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                self.backward(X_shuf[start:end], y_shuf[start:end], learning_rate, momentum)

            train_probs = self.forward(X_train, training=False)
            val_probs = self.forward(X_val, training=False)

            train_loss = self.compute_loss(train_probs, y_train)
            val_loss = self.compute_loss(val_probs, y_val)
            train_pred = (train_probs.ravel() >= 0.5).astype(np.int64)
            val_pred = (val_probs.ravel() >= 0.5).astype(np.int64)
            train_f1 = compute_f1(y_train, train_pred)
            val_f1 = compute_f1(y_val, val_pred)

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["train_f1"].append(float(train_f1))
            history["val_f1"].append(float(val_f1))

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                wait = 0
                best_state = (
                    deepcopy(self.weights),
                    deepcopy(self.biases),
                    deepcopy(self.vel_w),
                    deepcopy(self.vel_b),
                )
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            self.weights, self.biases, self.vel_w, self.vel_b = best_state

        return history

    def predict_proba(self, X):
        return self.forward(X, training=False).ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def save(self, path):
        payload = {}
        payload["n_layers"] = np.array([len(self.weights)], dtype=np.int64)
        payload["dropout_rate"] = np.array([self.dropout_rate], dtype=np.float64)
        payload["seed"] = np.array([self.seed], dtype=np.int64)
        payload["layer_sizes"] = np.array(self.layer_sizes, dtype=np.int64)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            payload[f"W{i}"] = w
            payload[f"b{i}"] = b
        np.savez(path, **payload)

    def load(self, path):
        data = np.load(path)
        self.dropout_rate = float(data["dropout_rate"][0])
        self.seed = int(data["seed"][0])
        self.layer_sizes = data["layer_sizes"].astype(np.int64).tolist()
        n_layers = int(data["n_layers"][0])
        self.weights = [data[f"W{i}"].astype(np.float64) for i in range(n_layers)]
        self.biases = [data[f"b{i}"].astype(np.float64) for i in range(n_layers)]
        self.vel_w = [np.zeros_like(w) for w in self.weights]
        self.vel_b = [np.zeros_like(b) for b in self.biases]
        self.rng = np.random.default_rng(self.seed)


def gradient_check(nn, X_small, y_small, epsilon=1e-5):
    y_small = y_small.reshape(-1, 1).astype(np.float64)
    # Gradient checking must be deterministic; disable dropout behavior.
    grads_w, grads_b, _ = nn._compute_gradients(X_small, y_small.ravel(), training=False)

    max_relative_error = 0.0

    for layer_idx in range(len(nn.weights)):
        W = nn.weights[layer_idx]
        b = nn.biases[layer_idx]
        gW = grads_w[layer_idx]
        gb = grads_b[layer_idx]

        check_w = min(20, W.size)
        check_b = min(10, b.size)

        w_flat = W.ravel()
        gw_flat = gW.ravel()
        sample_w_idx = np.linspace(0, W.size - 1, num=check_w, dtype=np.int64)

        for idx in sample_w_idx:
            orig = w_flat[idx]
            w_flat[idx] = orig + epsilon
            l_plus = nn.compute_loss(nn.forward(X_small, training=False), y_small)
            w_flat[idx] = orig - epsilon
            l_minus = nn.compute_loss(nn.forward(X_small, training=False), y_small)
            w_flat[idx] = orig
            grad_num = (l_plus - l_minus) / (2.0 * epsilon)
            grad_ana = gw_flat[idx]
            denom = max(1e-12, abs(grad_num) + abs(grad_ana))
            rel_err = abs(grad_num - grad_ana) / denom
            if rel_err > max_relative_error:
                max_relative_error = rel_err

        b_flat = b.ravel()
        gb_flat = gb.ravel()
        sample_b_idx = np.linspace(0, b.size - 1, num=check_b, dtype=np.int64)
        for idx in sample_b_idx:
            orig = b_flat[idx]
            b_flat[idx] = orig + epsilon
            l_plus = nn.compute_loss(nn.forward(X_small, training=False), y_small)
            b_flat[idx] = orig - epsilon
            l_minus = nn.compute_loss(nn.forward(X_small, training=False), y_small)
            b_flat[idx] = orig
            grad_num = (l_plus - l_minus) / (2.0 * epsilon)
            grad_ana = gb_flat[idx]
            denom = max(1e-12, abs(grad_num) + abs(grad_ana))
            rel_err = abs(grad_num - grad_ana) / denom
            if rel_err > max_relative_error:
                max_relative_error = rel_err

    print(f"Gradient check max relative error: {max_relative_error:.8e}")
    return float(max_relative_error)

