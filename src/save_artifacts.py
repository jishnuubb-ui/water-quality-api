"""
save_artifacts.py - Step 4: train the FINAL models on all data and save them with joblib.

Run from the project root:   python src/save_artifacts.py

Why: the CV runs in Step 2 only measured how good a model is. For the web app we
need one fitted copy of everything, saved to disk, so the API can load it instantly.

Files written to artifacts/:
  classifier.joblib   the winning classifier (scaler + model), predicts P(safe)
  detectors.joblib    scaler, PCA(2), K-Means, Isolation Forest, LOF, DBSCAN core points
  metadata.json       feature order, decision threshold, CV metrics, data stats
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data_prep import ROOT, TARGET, feature_columns, load_clean
from src.screening import Screener
from src.train import make_pipeline

ART = ROOT / "artifacts"
SEED = 42
CONTAMINATION = 0.05      # same settings as analysis.py, so results match
KMEANS_K = 2              # best k from analysis.py (silhouette); change here if yours differed


def main():
    df, _ = load_clean()
    feats = feature_columns(df)
    X, y = df[feats], df[TARGET].values

    # ---------- 1. Final classifier (trained on ALL clean rows) ----------
    best = json.loads((ART / "best_models.json").read_text())
    mname = best["winner"].split(" ")[0]
    info = best["models"][mname]
    clf = make_pipeline(mname, info["strategy"], **info["params"]).fit(X, y)
    joblib.dump(clf, ART / "classifier.joblib")
    print(f"[1/3] classifier saved: {mname}, imbalance strategy={info['strategy']}, "
          f"threshold={best['threshold']}")

    # ---------- 2. Unsupervised models (same settings as analysis.py) ----------
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    pca = PCA(n_components=2).fit(Z)
    kmeans = KMeans(n_clusters=KMEANS_K, n_init=10, random_state=SEED).fit(Z)
    iforest = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=SEED).fit(Z)
    # novelty=True lets LOF score NEW samples later (the default mode can only score its training data)
    lof = LocalOutlierFactor(n_neighbors=35, contamination=CONTAMINATION, novelty=True).fit(Z)
    min_samples = 10
    kdist = NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)[0][:, -1]
    eps = float(np.percentile(kdist, 95))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z)
    core_nn = NearestNeighbors(n_neighbors=1).fit(Z[db.core_sample_indices_])
    joblib.dump({"features": feats, "scaler": scaler, "pca": pca, "kmeans": kmeans,
                 "iforest": iforest, "lof": lof, "db_core_nn": core_nn, "db_eps": eps},
                ART / "detectors.joblib")
    print(f"[2/3] detectors saved (K-Means k={KMEANS_K}, DBSCAN eps={eps:.2f}, "
          f"{len(db.core_sample_indices_)} core points)")

    # ---------- 3. Metadata the API and dashboard can read ----------
    stats = {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                 "mean": float(X[c].mean()), "median": float(X[c].median())} for c in feats}
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "features": feats,
        "classifier": best["winner"],
        "imbalance_strategy": info["strategy"],
        "threshold": best["threshold"],
        "cv_metrics": best["metrics"],
        "training_rows": int(len(df)),
        "safe_rate": round(float(y.mean()), 4),
        "kmeans_k": KMEANS_K,
        "feature_stats": stats,
    }
    (ART / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("[3/3] metadata.json saved")

    # ---------- Self-test: reload from disk and screen a few real rows ----------
    s = Screener()
    out = s.screen(df.head(5))
    out["is_safe_true"] = y[:5]
    print("\nSelf-test (first 5 rows, loaded back from disk):")
    print(out[["is_safe_true", "probability_safe", "predicted_safe", "anomaly",
               "n_violations", "category"]].to_string())
    full = s.screen(df)
    print(f"\nSelf-test on all rows: anomalies flagged={int(full.anomaly.sum())}, "
          f"predicted safe={int(full.predicted_safe.sum())} (true safe={int(y.sum())})")
    print("\nDone. Artifacts are in artifacts/")


if __name__ == "__main__":
    main()
