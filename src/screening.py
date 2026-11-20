"""
screening.py - loads the saved artifacts and screens new water samples.

The FastAPI backend (Step 5) imports this file, so the dashboard and the offline
analysis use exactly the same rules. Nothing here trains anything; it only loads
files created by save_artifacts.py and applies them.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from src.data_prep import ROOT

ART = ROOT / "artifacts"


def compute_violations(df, limits_cfg):
    """Rule-based check: which chemicals are above their EPA/WHO limit (limits.json)?"""
    settings, lim = limits_cfg["settings"], limits_cfg["limits"]
    counted = {k: d["limit"] for k, d in lim.items()
               if d["limit"] is not None and (d["basis"] == "health" or settings["count_aesthetic"])}
    viol = pd.DataFrame(index=df.index)
    ratios = pd.DataFrame(index=df.index)   # value / limit (2.0 = twice the limit)
    for chem, limit in counted.items():
        if limit == 0:                       # "must be absent": any detectable amount counts
            viol[chem] = df[chem] > 0
        else:
            ratios[chem] = df[chem] / limit
            viol[chem] = ratios[chem] > settings["min_exceedance_ratio"]
    out = pd.DataFrame(index=df.index)
    out["n_violations"] = viol.sum(axis=1)
    out["max_ratio"] = ratios.max(axis=1).round(2)
    out["violating_chemicals"] = viol.apply(lambda r: [c for c, v in r.items() if v], axis=1)
    out["confirmed_violation"] = (out.n_violations >= 1).astype(int)
    return out


def categorize(anomaly, violation):
    if anomaly and violation:
        return "anomaly + confirmed violation"
    if violation:
        return "confirmed violation only"
    if anomaly:
        return "anomaly only (unusual but within limits)"
    return "normal"


class Screener:
    def __init__(self):
        self.clf = joblib.load(ART / "classifier.joblib")          # predicts P(safe)
        self.det = joblib.load(ART / "detectors.joblib")           # scaler, PCA, K-Means, anomaly models
        self.meta = json.loads((ART / "metadata.json").read_text())
        self.limits_cfg = json.loads((ROOT / "limits.json").read_text())
        self.features = self.det["features"]
        self.threshold = self.meta["threshold"]

    def screen(self, df):
        """df: DataFrame with the 20 chemical columns (original units). Returns one row per sample."""
        X = df[self.features].astype(float)
        Z = self.det["scaler"].transform(X)
        d = self.det

        proba_safe = self.clf.predict_proba(X)[:, 1]

        if_score = -d["iforest"].score_samples(Z)
        anom_if = (d["iforest"].predict(Z) == -1).astype(int)
        lof_score = -d["lof"].score_samples(Z)
        anom_lof = (d["lof"].predict(Z) == -1).astype(int)
        # DBSCAN cannot score new points directly. A new point is "noise" if no DBSCAN
        # core sample lies within eps of it - exactly the rule DBSCAN itself uses.
        dist = d["db_core_nn"].kneighbors(Z, n_neighbors=1)[0][:, 0]
        anom_db = (dist > d["db_eps"]).astype(int)
        n_det = anom_if + anom_lof + anom_db
        anomaly = (n_det >= 2).astype(int)               # consensus: at least 2 of 3

        pcs = d["pca"].transform(Z)
        cluster = d["kmeans"].predict(Z)
        z_abs = pd.DataFrame(np.abs(Z), columns=self.features)
        unusual_because = z_abs.apply(lambda r: list(r.sort_values(ascending=False).head(3).index), axis=1)

        viol = compute_violations(X.reset_index(drop=True), self.limits_cfg)
        res = pd.DataFrame({
            "probability_safe": proba_safe.round(4),
            "predicted_safe": (proba_safe >= self.threshold).astype(int),
            "anomaly": anomaly,
            "anom_if": anom_if, "anom_lof": anom_lof, "anom_dbscan": anom_db,
            "if_score": np.round(if_score, 4), "lof_score": np.round(lof_score, 4),
            "cluster": cluster,
            "pc1": pcs[:, 0].round(3), "pc2": pcs[:, 1].round(3),
            "unusual_because": unusual_because.values,
        })
        res = pd.concat([res, viol.reset_index(drop=True)], axis=1)
        res["category"] = [categorize(a, v) for a, v in zip(res.anomaly, res.confirmed_violation)]
        return res
