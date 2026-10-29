"""Four disjoint chronological blocks: base, meta, calibration, untouched test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .history import load_matches, historical_features, FEATURES


def elo_probabilities(differences):
    expected = 1 / (1 + 10 ** (-np.clip(differences, -10000, 10000) / 400))
    draw = .10 + .14 * (1 - (2 * expected - 1) ** 2)
    result = np.column_stack((np.maximum(1 - expected - .5 * draw, 0), draw,
                              np.maximum(expected - .5 * draw, 0)))
    return result / result.sum(axis=1, keepdims=True)


def chronological_masks(dates):
    return {"base_train": dates < "2023-01-01",
            "stack_validation": (dates >= "2023-01-01") & (dates < "2024-01-01"),
            "calibration": (dates >= "2024-01-01") & (dates < "2025-01-01"),
            "test": dates >= "2025-01-01"}


def probability_features(p):
    return np.log(np.clip(p, 1e-8, 1 - 1e-8) / np.clip(1 - p, 1e-8, 1))


class HeldOutStack:
    def __init__(self, seed=42):
        self.models = [RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=1, random_state=seed),
                       ExtraTreesClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=1, random_state=seed)]
        self.meta = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        self.calibrators = []

    def _meta_input(self, x):
        return np.hstack([model.predict_proba(x) for model in self.models])

    def fit(self, x, y, masks):
        self.calibrators = []
        # No estimator is refit on a later block after calibration/test evaluation.
        for model in self.models:
            model.fit(x[masks["base_train"]], y[masks["base_train"]])
        self.meta.fit(self._meta_input(x[masks["stack_validation"]]), y[masks["stack_validation"]])
        raw = self.raw_proba(x[masks["calibration"]])
        logits = probability_features(raw)
        for label in range(3):
            estimator = LogisticRegression(max_iter=1000, random_state=42)
            estimator.fit(logits[:, [label]], (y[masks["calibration"]] == label).astype(int))
            self.calibrators.append(estimator)
        return self

    def raw_proba(self, x):
        forward = self.meta.predict_proba(self._meta_input(x))
        reverse = self.meta.predict_proba(self._meta_input(-x))[:, ::-1]
        return .5 * (forward + reverse)

    def predict_proba(self, x):
        def calibrated(values):
            logits = probability_features(self.raw_proba(values))
            result = np.column_stack([est.predict_proba(logits[:, [label]])[:, 1]
                                      for label, est in enumerate(self.calibrators)])
            return result / result.sum(axis=1, keepdims=True)
        return .5 * (calibrated(x) + calibrated(-x)[:, ::-1])


def metrics(y, p):
    if p.shape != (len(y), 3) or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("invalid three-class probability matrix")
    prediction = np.argmax(p, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=[0, 1, 2], zero_division=0)
    bins = []
    confidence = p.max(axis=1)
    assignments = np.minimum((confidence * 10).astype(int), 9)
    for index in range(10):
        low = index / 10
        select = assignments == index
        if select.any():
            bins.append({"lower": round(float(low), 1), "count": int(select.sum()),
                         "confidence": float(confidence[select].mean()),
                         "accuracy": float((prediction[select] == y[select]).mean())})
    return {"log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
            "multiclass_brier_sum": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
            "accuracy": float(accuracy_score(y, prediction)),
            "top_label_ece": float(sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in bins) / len(y)),
            "calibration_bins": bins, "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(support[i])}
                          for i, name in enumerate(["loss", "draw", "win"])}}


def run(data_path, output_dir, seed=42):
    matches, audit = load_matches(data_path)
    x, y, dates, identities, history = historical_features(matches)
    masks = chronological_masks(dates)
    for name, mask in masks.items():
        if mask.sum() < 20 or set(y[mask]) != {0, 1, 2}:
            raise ValueError(f"{name} needs >=20 rows and all three classes")
    stack = HeldOutStack(seed).fit(x, y, masks)
    simple = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    simple.fit(x[masks["base_train"]], y[masks["base_train"]])
    test = x[masks["test"]]
    probabilities = {"elo_only": elo_probabilities(test[:, 0]), "logistic": simple.predict_proba(test),
                     "random_forest": stack.models[0].predict_proba(test),
                     "extra_trees": stack.models[1].predict_proba(test),
                     "held_out_stack": stack.raw_proba(test), "sigmoid_stack": stack.predict_proba(test)}
    report = {"seed": seed, "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
              "features": FEATURES, "protocol": "rolling one-step-ahead test; earlier test outcomes update history only",
              "partitions": {name: {"rows": int(mask.sum()), "matches": int(mask.sum() // 2),
                                      "start": str(min(dates[mask])), "end": str(max(dates[mask])),
                                      "class_counts": np.bincount(y[mask], minlength=3).tolist()}
                             for name, mask in masks.items()},
              "test_metrics": {name: metrics(y[masks["test"]], p) for name, p in probabilities.items()}}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report, stack, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/international_football_2018_present.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report, _, _ = run(args.data, args.out, args.seed)
    for name, result in report["test_metrics"].items():
        print(f"{name}: log loss={result['log_loss']:.4f}, Brier={result['multiclass_brier_sum']:.4f}, accuracy={result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
