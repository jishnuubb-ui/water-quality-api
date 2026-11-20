"""
train.py - Step 2: compare models + imbalance strategies, tune, pick a threshold.

Run from the project root:
    python src/train.py            (full run, a few minutes)
    python src/train.py --quick    (tiny run to check everything works, ~1 minute)

What it does
  Stage 1: 3 models x 3 imbalance strategies, stratified 5-fold CV.
  Stage 2: tune Random Forest and HistGradientBoosting (random search).
  Stage 3: choose the decision threshold that maximises minority-class F1.
  Output : CSV tables, a PR-curve plot and artifacts/best_models.json.
           (Step 4 will use that JSON to train and save the final model.)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

import matplotlib
matplotlib.use("Agg")  # save plots to files, don't open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_prep import ROOT, TARGET, feature_columns, load_clean

ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
SEED = 42


# --------------------------------------------------------------------------------------
# SMOTE (written by hand so you can read it; no extra library needed)
# --------------------------------------------------------------------------------------
class SmoteClassifier(ClassifierMixin, BaseEstimator):
    """Wraps any classifier: before fitting, adds synthetic minority ("safe") rows.

    SMOTE idea: take a real safe sample, pick one of its nearest safe neighbours, and
    create a NEW sample somewhere on the straight line between the two.
    It is applied ONLY to the training part of each CV fold. The validation part stays
    untouched and real. (Applying SMOTE before splitting would leak copies of
    validation rows into training and give fake-good scores.)
    """

    def __init__(self, base=None, k=5, random_state=SEED):
        self.base = base
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        minority = 1 if (y == 1).sum() < (y == 0).sum() else 0
        Xm = X[y == minority]
        n_new = int((y != minority).sum() - len(Xm))          # enough to make classes 1:1
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(Xm)
        neigh = nn.kneighbors(Xm, return_distance=False)[:, 1:]  # drop the point itself
        base_i = rng.randint(0, len(Xm), n_new)                  # which real sample to start from
        neigh_i = neigh[base_i, rng.randint(0, self.k, n_new)]   # which neighbour to aim at
        gap = rng.rand(n_new, 1)                                 # how far along the line (0..1)
        X_new = Xm[base_i] + gap * (Xm[neigh_i] - Xm[base_i])
        X_res = np.vstack([X, X_new])
        y_res = np.concatenate([y, np.full(n_new, minority)])
        self.base_ = clone(self.base).fit(X_res, y_res)
        self.classes_ = self.base_.classes_
        return self

    def predict_proba(self, X):
        return self.base_.predict_proba(X)

    def predict(self, X):
        return self.base_.predict(X)


# --------------------------------------------------------------------------------------
# Model factory: (model name, imbalance strategy) -> ready-to-train pipeline
# --------------------------------------------------------------------------------------
def make_base(name, strategy):
    """strategy: 'none' | 'weights' | 'smote'. Class weights make rare-class mistakes cost more."""
    w = strategy == "weights"
    if name == "LogReg":
        return LogisticRegression(max_iter=3000, class_weight="balanced" if w else None)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1,
                                      class_weight="balanced_subsample" if w else None)
    if name == "HistGB":
        return HistGradientBoostingClassifier(random_state=SEED,
                                              class_weight="balanced" if w else None)
    raise ValueError(name)


def make_pipeline(name, strategy, **params):
    """Scaler -> (optional SMOTE) -> model. Scaling first matters for SMOTE and LogReg:
    otherwise 'perchlorate' (0-60) would dominate 'mercury' (0-0.01) in neighbour distances."""
    base = make_base(name, strategy)
    if params:
        base.set_params(**params)
    clf = SmoteClassifier(base) if strategy == "smote" else base
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def param_prefix(strategy):
    """Where the model's own parameters live inside the pipeline."""
    return "clf__base__" if strategy == "smote" else "clf__"


# Random-search spaces: each run tries random combinations from these lists.
SPACES = {
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.5],
    },
    "HistGB": {
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "max_iter": [100, 200, 400],
        "max_depth": [None, 4, 6, 8],
        "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0],
        "max_leaf_nodes": [15, 31, 63],
    },
}


def scores_from(y, proba, thr):
    pred = (proba >= thr).astype(int)
    return {"threshold": round(float(thr), 3),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "PR_AUC": average_precision_score(y, proba)}  # PR-AUC does not depend on threshold


def best_threshold(y, proba):
    """Probability cut-off that gives the best F1 for the 'safe' class."""
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[np.argmax(f1)])


def main(quick=False):
    t0 = time.time()
    df, _ = load_clean()
    X, y = df[feature_columns(df)].values, df[TARGET].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---------------- Stage 1: models x imbalance strategies ----------------
    print("\n=== Stage 1: models x imbalance strategies (stratified 5-fold CV, threshold 0.5) ===")
    rows = []
    models = ["LogReg"] if quick else ["LogReg", "RandomForest", "HistGB"]
    if quick:
        models = ["LogReg", "HistGB"]
    for name in models:
        for strategy in ["none", "weights", "smote"]:
            res = cross_validate(make_pipeline(name, strategy), X, y, cv=cv, n_jobs=-1,
                                 scoring={"precision": "precision", "recall": "recall",
                                          "f1": "f1", "pr_auc": "average_precision"})
            row = {"model": name, "strategy": strategy}
            for m in ["precision", "recall", "f1", "pr_auc"]:
                row[m] = res[f"test_{m}"].mean()
                row[m + "_std"] = res[f"test_{m}"].std()   # spread across the 5 folds
            rows.append(row)
            print(f"  {name:12s} {strategy:8s} F1={row['f1']:.3f}  PR-AUC={row['pr_auc']:.3f}  "
                  f"P={row['precision']:.3f}  R={row['recall']:.3f}   [{time.time()-t0:4.0f}s]")
    comp = pd.DataFrame(rows)
    comp.round(4).to_csv(ART / "model_comparison.csv", index=False)

    # ---------------- Stage 2: tune the two tree models ----------------
    print("\n=== Stage 2: random-search tuning (scored by PR-AUC) ===")
    n_iter = 3 if quick else 15
    best = {}
    for name in [m for m in ["RandomForest", "HistGB"] if m in models]:
        sub = comp[comp.model == name]
        strategy = sub.loc[sub.pr_auc.idxmax(), "strategy"]    # best ranking quality
        pre = param_prefix(strategy)
        grid = {pre + k: v for k, v in SPACES[name].items()}
        search = RandomizedSearchCV(make_pipeline(name, strategy), grid, n_iter=n_iter,
                                    scoring="average_precision", cv=cv, n_jobs=-1,
                                    random_state=SEED, refit=True)
        search.fit(X, y)
        params = {k.replace(pre, ""): (v.item() if hasattr(v, "item") else v)
                  for k, v in search.best_params_.items()}
        best[name] = {"strategy": strategy, "params": params, "cv_pr_auc": search.best_score_}
        print(f"  {name}: strategy={strategy}  PR-AUC={search.best_score_:.3f}  params={params}"
              f"   [{time.time()-t0:4.0f}s]")

    # ---------------- Stage 3: out-of-fold predictions + threshold ----------------
    # Fresh fold split (different seed) than the tuning one, to reduce optimism.
    # Every row is predicted by a model that never saw it ("out-of-fold").
    print("\n=== Stage 3: decision threshold (out-of-fold predictions) ===")
    cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    candidates = {"LogReg (baseline)": make_pipeline("LogReg", "weights")}
    for name, info in best.items():
        candidates[f"{name} (tuned)"] = make_pipeline(name, info["strategy"], **info["params"])

    final_rows, oof = [], {}
    for label, pipe in candidates.items():
        proba = cross_val_predict(pipe, X, y, cv=cv2, method="predict_proba", n_jobs=-1)[:, 1]
        oof[label] = proba
        thr = best_threshold(y, proba)
        for kind, t in [("default 0.5", 0.5), ("tuned", thr)]:
            final_rows.append({"model": label, "threshold_type": kind, **scores_from(y, proba, t)})
    final = pd.DataFrame(final_rows).round(4)
    final.to_csv(ART / "tuning_results.csv", index=False)
    print(final.to_string(index=False))

    # Winner = best F1 at its tuned threshold
    tuned = final[final.threshold_type == "tuned"]
    winner = tuned.loc[tuned.F1.idxmax()]
    wname = winner["model"].split(" ")[0]
    out = {"winner": winner["model"], "threshold": float(winner["threshold"]),
           "metrics": {k: float(winner[k]) for k in ["precision", "recall", "F1", "PR_AUC"]},
           "features": feature_columns(df), "models": best}
    if wname == "LogReg":   # baseline won (unlikely) -> record its setup too
        out["models"]["LogReg"] = {"strategy": "weights", "params": {}}
    (ART / "best_models.json").write_text(json.dumps(out, indent=2))
    print(f"\nWinner: {winner['model']}  threshold={winner['threshold']}  "
          f"F1={winner['F1']:.3f}  PR-AUC={winner['PR_AUC']:.3f}")

    # PR-curve plot: higher and further right = better
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, proba in oof.items():
        p, r, _ = precision_recall_curve(y, proba)
        ax.plot(r, p, label=f"{label} (AP={average_precision_score(y, proba):.2f})")
    ax.axhline(y.mean(), ls="--", color="grey", label="random guess")
    ax.set_xlabel("Recall (safe samples found)")
    ax.set_ylabel("Precision (predicted-safe that are really safe)")
    ax.set_title("Precision-Recall curves (out-of-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ART / "pr_curves.png", dpi=120)
    plt.close(fig)
    print(f"\nDone in {time.time()-t0:.0f}s. Files written to artifacts/")


if __name__ == "__main__":   # this guard is REQUIRED on Windows when using n_jobs=-1
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run just to test the code")
    main(quick=ap.parse_args().quick)
