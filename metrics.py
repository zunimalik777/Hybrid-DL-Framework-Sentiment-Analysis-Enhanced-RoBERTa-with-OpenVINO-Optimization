"""
Shared evaluation utilities: Accuracy, Precision, Recall, F1-score,
confusion matrix plotting, and latency/throughput benchmarking used to
compare every stage of the pipeline (VADER, GRU, RoBERTa, ONNX, OpenVINO
FP32/FP16, and NNCF INT8).
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred, average="binary") -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    }


def print_classification_report(y_true, y_pred, target_names=("negative", "positive")):
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", labels=("negative", "positive"),
                           save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def results_table(results: dict) -> pd.DataFrame:
    """results = {"Model Name": {"accuracy": .., "precision": .., "recall": .., "f1_score": ..}, ...}"""
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df.sort_values("f1_score", ascending=False)


def benchmark_latency(infer_fn, inputs, n_warmup=5, n_runs=50) -> dict:
    """Generic single-sample latency / throughput benchmark.

    `infer_fn` must accept a single element from `inputs` and run one
    forward pass. Works for PyTorch, ONNX Runtime, and OpenVINO callables.
    """
    for i in range(min(n_warmup, len(inputs))):
        infer_fn(inputs[i % len(inputs)])

    timings = []
    for i in range(n_runs):
        sample = inputs[i % len(inputs)]
        start = time.perf_counter()
        infer_fn(sample)
        timings.append(time.perf_counter() - start)

    timings = np.array(timings)
    return {
        "mean_latency_ms": round(float(timings.mean() * 1000), 3),
        "p50_latency_ms": round(float(np.percentile(timings, 50) * 1000), 3),
        "p95_latency_ms": round(float(np.percentile(timings, 95) * 1000), 3),
        "throughput_samples_per_sec": round(float(1.0 / timings.mean()), 2),
    }


def model_size_mb(path: str) -> float:
    import os

    if os.path.isdir(path):
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(path)
            for f in files
        )
    else:
        total = os.path.getsize(path)
    return round(total / (1024 * 1024), 3)
