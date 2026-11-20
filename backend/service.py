"""
service.py - turns raw model output into clean JSON-friendly dictionaries.

This file has NO web code. main.py (the FastAPI part) calls these functions.
Keeping them apart makes the logic easy to test and to read.
"""
import pandas as pd

from src.screening import Screener

# Dataset typo -> correct name. We accept both so callers using the original CSV header still work.
ALIASES = {"flouride": "fluoride"}


def normalise(sample: dict) -> dict:
    """Fix known column typos; leave everything else as it is."""
    return {ALIASES.get(k, k): v for k, v in sample.items()}


def to_frame(samples: list, features: list) -> pd.DataFrame:
    return pd.DataFrame([normalise(s) for s in samples])[features]


def classification_block(row, threshold):
    p = float(row["probability_safe"])
    return {
        "predicted_label": "safe" if int(row["predicted_safe"]) == 1 else "unsafe",
        "is_safe": int(row["predicted_safe"]),
        "probability_safe": round(p, 4),
        "decision_threshold": threshold,
    }


def violation_details(sample: dict, chemicals: list, limits_cfg: dict):
    out = []
    for chem in chemicals:
        d = limits_cfg["limits"][chem]
        limit, value = d["limit"], float(sample[chem])
        out.append({
            "chemical": chem,
            "value": value,
            "limit": limit,
            "times_limit": round(value / limit, 2) if limit else None,   # None when the limit is "absent"
            "basis": d["basis"],
        })
    return sorted(out, key=lambda x: (x["times_limit"] is None, -(x["times_limit"] or 0)))


def screening_block(row, sample, limits_cfg, threshold):
    anomaly = bool(row["anomaly"])
    violation = bool(row["confirmed_violation"])
    cls = classification_block(row, threshold)
    return {
        "classification": cls,
        "anomaly": {
            "is_anomaly": anomaly,
            "detectors_flagging": int(row["anom_if"] + row["anom_lof"] + row["anom_dbscan"]),
            "isolation_forest": bool(row["anom_if"]),
            "local_outlier_factor": bool(row["anom_lof"]),
            "dbscan_noise": bool(row["anom_dbscan"]),
            "isolation_forest_score": float(row["if_score"]),
            "lof_score": float(row["lof_score"]),
            "most_unusual_chemicals": list(row["unusual_because"]),
        },
        "violations": {
            "confirmed_violation": violation,
            "n_limits_exceeded": int(row["n_violations"]),
            "worst_times_limit": None if pd.isna(row["max_ratio"]) else float(row["max_ratio"]),
            "exceeded": violation_details(sample, list(row["violating_chemicals"]), limits_cfg),
        },
        "category": row["category"],
        "summary": (
            f"Classifier says {cls['predicted_label']} (P(safe)={cls['probability_safe']}); "
            f"{int(row['n_violations'])} regulatory limit(s) exceeded; "
            f"{'flagged as an anomaly' if anomaly else 'not an anomaly'}."
        ),
    }


def screen_many(screener: Screener, samples: list):
    df = to_frame(samples, screener.features)
    res = screener.screen(df)
    out = []
    for i, s in enumerate(samples):
        out.append(screening_block(res.iloc[i], normalise(s), screener.limits_cfg, screener.threshold))
    return out
