"""
analysis.py - Step 3: PCA, clusters, anomalies, limit violations, influential chemicals.

Run from the project root:   python src/analysis.py
Outputs go to artifacts/analysis/ (tables + plots) and artifacts/sample_screening.csv.

Big picture
  * PCA / K-Means / DBSCAN / Isolation Forest / LOF are UNSUPERVISED: they never see the
    is_safe label. We only use the label afterwards, to judge what they found.
  * "Confirmed violation" is a RULE: a chemical is above its EPA/WHO limit (limits.json).
  * "Anomaly" is a STATISTICAL idea: the sample looks unusual compared with the others.
    Unusual does not mean unsafe, and unsafe does not mean unusual. We keep them separate.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data_prep import ROOT, TARGET, feature_columns, load_clean
from src.train import make_pipeline

OUT = ROOT / "artifacts" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
CONTAMINATION = 0.05   # we ask IsolationForest/LOF to flag the 5% most unusual samples


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)


def main():
    df, _ = load_clean()
    feats = feature_columns(df)
    y = df[TARGET].values

    # Put every chemical on the same scale (mean 0, spread 1). Without this, perchlorate
    # (0-60) would drown out mercury (0-0.01) in every distance-based method below.
    scaler = StandardScaler()
    Z = scaler.fit_transform(df[feats])
    Zdf = pd.DataFrame(Z, columns=feats)

    # =============================== 1. PCA ===============================
    # PCA finds new axes (principal components) that capture as much variation as possible.
    pca = PCA().fit(Z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    print(f"\n[PCA] first 2 components explain {cum[1]:.1%}; {n90} components needed for 90%")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(evr) + 1), evr, label="each component")
    ax.plot(range(1, len(evr) + 1), cum, "r-o", ms=3, label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Share of variance")
    ax.set_title("PCA scree plot")
    ax.legend()
    savefig(fig, "pca_scree.png")

    loadings = pd.DataFrame(pca.components_[:3].T, index=feats, columns=["PC1", "PC2", "PC3"])
    loadings.round(3).to_csv(OUT / "pca_loadings.csv")
    top_load = {pc: loadings[pc].abs().sort_values(ascending=False).head(4).index.tolist()
                for pc in loadings}
    print("[PCA] chemicals that drive each component:", top_load)
    pc2 = pca.transform(Z)[:, :2]

    fig, ax = plt.subplots(figsize=(6, 5))
    for lab, col, name in [(0, "#c0392b", "unsafe"), (1, "#27ae60", "safe")]:
        ax.scatter(pc2[y == lab, 0], pc2[y == lab, 1], s=5, alpha=0.4, c=col, label=name)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("PCA projection coloured by label")
    ax.legend(markerscale=3)
    savefig(fig, "pca_scatter_label.png")
    json.dump({"explained_variance_ratio": evr.round(4).tolist(), "components_for_90pct": n90,
               "top_loadings": top_load}, open(OUT / "pca_report.json", "w"), indent=2)

    # =============================== 2. K-Means ===============================
    # Elbow: inertia (tightness) always falls as k grows - look for where it flattens.
    # Silhouette: -1..1, higher = clusters are well separated. We use both.
    ks = list(range(2, 11))
    inertia, sil = [], []
    rng = np.random.RandomState(SEED)
    sub = rng.choice(len(Z), size=min(4000, len(Z)), replace=False)  # subsample: silhouette is slow
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Z[sub], km.labels_[sub]))
    best_k = ks[int(np.argmax(sil))]
    print(f"[KMeans] silhouette by k: {dict(zip(ks, np.round(sil, 3)))}  -> best k = {best_k}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(ks, inertia, "o-")
    axs[0].set_title("Elbow (inertia)")
    axs[0].set_xlabel("k")
    axs[1].plot(ks, sil, "o-", color="darkgreen")
    axs[1].set_title("Silhouette score")
    axs[1].set_xlabel("k")
    savefig(fig, "kmeans_elbow_silhouette.png")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Z)
    df["cluster"] = km.labels_
    prof_rows = []
    for c in range(best_k):
        m = df.cluster == c
        zmean = Zdf[m.values].mean()
        prof_rows.append({
            "cluster": c, "size": int(m.sum()),
            "safe_rate": round(float(df.loc[m, TARGET].mean()), 3),
            "highest_chemicals": ", ".join(zmean.sort_values(ascending=False).head(3).index),
            "lowest_chemicals": ", ".join(zmean.sort_values().head(3).index),
        })
    cluster_report = pd.DataFrame(prof_rows)
    cluster_report.to_csv(OUT / "cluster_report.csv", index=False)
    print("[KMeans] cluster report:\n", cluster_report.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(pc2[:, 0], pc2[:, 1], c=df.cluster, s=5, cmap="tab10", alpha=0.5)
    ax.set_title(f"K-Means clusters (k={best_k}) in PCA space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    savefig(fig, "kmeans_clusters_pca.png")

    # =============================== 3. Anomaly detectors ===============================
    # (a) Isolation Forest: unusual points are easy to "isolate" with few random splits.
    iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=SEED).fit(Z)
    df["if_score"] = -iso.score_samples(Z)                  # higher = more unusual
    df["anom_if"] = (iso.predict(Z) == -1).astype(int)

    # (b) LOF: a point is unusual if its neighbourhood is much sparser than its neighbours'.
    lof = LocalOutlierFactor(n_neighbors=35, contamination=CONTAMINATION)
    df["anom_lof"] = (lof.fit_predict(Z) == -1).astype(int)
    df["lof_score"] = -lof.negative_outlier_factor_         # higher = more unusual

    # (c) DBSCAN: groups dense regions; points that belong to no dense region are "noise".
    # eps (neighbourhood radius) is picked from the k-distance curve: the distance to the
    # 10th nearest neighbour, taking the 95th percentile so ~5% of points end up as noise.
    min_samples = 10
    kdist = NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)[0][:, -1]
    eps = float(np.percentile(kdist, 95))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z)
    df["dbscan_cluster"] = db.labels_
    df["anom_dbscan"] = (db.labels_ == -1).astype(int)
    n_db_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"\n[DBSCAN] eps={eps:.2f}, clusters found={n_db_clusters}, noise={df.anom_dbscan.mean():.1%}")

    # Consensus: unusual according to at least 2 of the 3 detectors (fewer false alarms).
    df["n_detectors"] = df[["anom_if", "anom_lof", "anom_dbscan"]].sum(axis=1)
    df["anomaly"] = (df.n_detectors >= 2).astype(int)
    print(f"[Anomaly] flagged by IF={df.anom_if.sum()}, LOF={df.anom_lof.sum()}, "
          f"DBSCAN={df.anom_dbscan.sum()}, consensus(>=2)={df.anomaly.sum()}")
    agree = df[["anom_if", "anom_lof", "anom_dbscan"]].corr().round(2)
    agree.to_csv(OUT / "detector_agreement.csv")

    # Does "unusual" line up with the label? (Label is used ONLY here, for judging.)
    anom_vs_label = df.groupby(TARGET)[["anom_if", "anom_lof", "anom_dbscan", "anomaly"]].mean().round(3)
    anom_vs_label.index = ["unsafe (0)", "safe (1)"]
    anom_vs_label.to_csv(OUT / "anomaly_rate_by_label.csv")
    print("[Anomaly] share flagged, by true label:\n", anom_vs_label.to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pc2[:, 0], pc2[:, 1], s=4, c="lightgrey")
    a = df.anomaly.values == 1
    ax.scatter(pc2[a, 0], pc2[a, 1], s=14, c="purple", label="consensus anomaly")
    ax.set_title("Anomalies in PCA space")
    ax.legend()
    savefig(fig, "anomalies_pca.png")

    # =============================== 4. Confirmed violations (rules) ===============================
    cfg = json.load(open(ROOT / "limits.json"))
    settings, lim = cfg["settings"], cfg["limits"]
    counted = {k: d["limit"] for k, d in lim.items()
               if d["limit"] is not None and (d["basis"] == "health" or settings["count_aesthetic"])}
    ratio_min = settings["min_exceedance_ratio"]
    viol = pd.DataFrame(index=df.index)
    ratios = pd.DataFrame(index=df.index)   # value divided by its limit (2.0 = twice the limit)
    for chem, limit in counted.items():
        if limit == 0:      # "must be absent": any detectable amount counts as a violation
            viol[chem] = df[chem] > 0
        else:
            ratios[chem] = df[chem] / limit
            viol[chem] = ratios[chem] > ratio_min
    max_ratio = ratios.max(axis=1)
    df["n_violations"] = viol.sum(axis=1)
    df["max_ratio"] = max_ratio.round(2)
    df["violating_chemicals"] = viol.apply(lambda r: ";".join(r.index[r.values]), axis=1)
    df["confirmed_violation"] = (df.n_violations >= 1).astype(int)

    by_chem = pd.DataFrame({"share_exceeding": viol.mean().round(3),
                            "share_exceeding_if_unsafe": viol[y == 0].mean().round(3),
                            "share_exceeding_if_safe": viol[y == 1].mean().round(3)})
    by_chem.to_csv(OUT / "violations_by_chemical.csv")
    print(f"\n[Violations] chemicals counted: {len(counted)}")
    print(f"[Violations] samples with >=1 violation: {df.confirmed_violation.mean():.1%}")
    # How bad are the exceedances? Share of samples where ANY chemical is >= N x its limit.
    sev = pd.DataFrame({f">= {m}x limit": [(ratios.max(axis=1) >= m)[y == 0].mean(),
                                          (ratios.max(axis=1) >= m)[y == 1].mean()]
                        for m in [1, 2, 3, 5, 10, 20]}, index=["unsafe (0)", "safe (1)"]).round(3)
    sev.to_csv(OUT / "violation_severity_by_label.csv")
    print("[Violations] share of samples with a chemical at N times its limit:\n", sev.T.to_string())
    print("[Violations] average number of limits exceeded: "
          f"unsafe={df.loc[y == 0, 'n_violations'].mean():.2f}, safe={df.loc[y == 1, 'n_violations'].mean():.2f}")

    # =============================== 5. Anomaly vs violation vs label ===============================
    def category(r):
        if r.anomaly and r.confirmed_violation:
            return "anomaly + confirmed violation"
        if r.confirmed_violation:
            return "confirmed violation only"
        if r.anomaly:
            return "anomaly only (unusual but within limits)"
        return "normal"
    df["category"] = df.apply(category, axis=1)
    cat = pd.crosstab(df.category, df[TARGET].map({0: "label_unsafe", 1: "label_safe"}))
    cat.to_csv(OUT / "category_vs_label.csv")
    print("\n[Categories] vs true label:\n", cat.to_string())

    # False alarms = flagged as anomalous, yet the lab label says the water is SAFE.
    fa = df[(df.anomaly == 1) & (df[TARGET] == 1)].copy()
    drivers = Zdf.loc[fa.index].abs().apply(lambda r: ", ".join(r.sort_values(ascending=False).head(3).index), axis=1)
    fa["unusual_because"] = drivers
    fa[["cluster", "if_score", "lof_score", "n_detectors", "n_violations", "unusual_because"] + feats].round(3) \
        .to_csv(OUT / "false_alarms.csv", index_label="row_id")
    top_driver = drivers.str.split(", ").explode().value_counts().head(5)
    print(f"[False alarms] consensus anomalies labelled safe: {len(fa)} of {df.anomaly.sum()} anomalies")
    print("[False alarms] chemicals most often making them look unusual:", top_driver.to_dict())

    # =============================== 6. Influential chemicals ===============================
    # Permutation importance: shuffle one chemical on held-out data and see how much the
    # model's PR-AUC drops. Big drop = the model relies on that chemical.
    best = json.load(open(ROOT / "artifacts" / "best_models.json"))
    mname = best["winner"].split(" ")[0]
    info = best["models"][mname]
    Xtr, Xte, ytr, yte = train_test_split(df[feats], y, test_size=0.25, stratify=y, random_state=SEED)
    model = make_pipeline(mname, info["strategy"], **info["params"]).fit(Xtr, ytr)
    pi = permutation_importance(model, Xte, yte, scoring="average_precision",
                                n_repeats=10, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"pr_auc_drop": pi.importances_mean.round(4),
                        "std": pi.importances_std.round(4)}, index=feats).sort_values("pr_auc_drop", ascending=False)
    imp["pca_pc1_loading"] = loadings["PC1"].reindex(imp.index).round(3)
    imp["share_exceeding_limit"] = by_chem["share_exceeding"].reindex(imp.index)
    imp.to_csv(OUT / "influential_chemicals.csv", index_label="chemical")
    print(f"\n[Influence] {mname} permutation importance (drop in PR-AUC on held-out 25%):")
    print(imp.head(8).to_string())
    fig, ax = plt.subplots(figsize=(6, 6))
    imp["pr_auc_drop"][::-1].plot.barh(ax=ax, xerr=imp["std"][::-1])
    ax.set_title("Influential chemicals (permutation importance)")
    savefig(fig, "influential_chemicals.png")

    # =============================== 7. Master table for the dashboard ===============================
    keep = ["is_safe", "cluster", "if_score", "lof_score", "anom_if", "anom_lof", "anom_dbscan",
            "anomaly", "n_violations", "max_ratio", "violating_chemicals", "confirmed_violation", "category"]
    master = df[keep].copy()
    master["pc1"], master["pc2"] = pc2[:, 0].round(3), pc2[:, 1].round(3)
    master.round(4).to_csv(ROOT / "artifacts" / "sample_screening.csv", index_label="row_id")
    print("\nDone. Tables/plots in artifacts/analysis/, master table in artifacts/sample_screening.csv")


if __name__ == "__main__":
    main()
