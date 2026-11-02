import numpy as np


def ece_score(probs, y, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(task_type, y, out):
    """`out` = raw model outputs (logits or predictions), numpy."""
    if task_type == "classification":
        z = out - out.max(1, keepdims=True)
        probs = np.exp(z)
        probs /= probs.sum(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll = float(-logp[np.arange(len(y)), y].mean())
        onehot = np.eye(out.shape[1])[y]
        return {
            "loss": nll,
            "log_loss": nll,
            "accuracy": float((probs.argmax(1) == y).mean()),
            "ece": ece_score(probs, y),
            "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        }
    pred = out.reshape(-1)
    mse = float(((pred - y) ** 2).mean())
    return {
        "loss": mse,
        "rmse": float(np.sqrt(mse)),  # in standardized target units
        "r2": float(1 - mse / ((y - y.mean()) ** 2).mean()),
    }


def primary_metric(task_type):
    return "accuracy" if task_type == "classification" else "r2"
