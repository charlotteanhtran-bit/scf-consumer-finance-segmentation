"""Reproducible SCF subgroup preparation and clustering analysis."""
from __future__ import annotations

import json
import hashlib
from itertools import combinations
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_PATH = DATA_DIR / "SCFP2022.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/vaibhavselkar/"
    "Consumer-finance-in-USA-2022/main/SCFP2022.csv"
)
FINANCIAL_FEATURES = ["DEBT", "HOUSES", "ASSET", "INCOME", "NETWORTH"]
TWO_FEATURES = ["DEBT", "HOUSES"]


def verify_source(path: Path) -> None:
    manifest = json.loads((ROOT / "data_provenance.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Source checksum changed. Review the upstream data before updating data_provenance.json.")


def download_data(path: Path = DATA_PATH) -> Path:
    """Download the public SCF CSV once and cache it locally."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(DATA_URL, timeout=60) as response:
            path.write_bytes(response.read())
    verify_source(path)
    return path


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Read source data and verify fields needed downstream."""
    source = path or download_data()
    df = pd.read_csv(source, low_memory=False)
    required = {"YY1", "Y1", "WGT", "TURNFEAR", "NETWORTH", *FINANCIAL_FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"SCF file is missing required columns: {missing}")
    return df


def prepare_sample(df: pd.DataFrame, implicate: int = 1) -> pd.DataFrame:
    """Keep one imputation per family and define the Project 6 subgroup."""
    out = df.copy()
    out["IMPLIC"] = out["Y1"] - 10 * out["YY1"]
    out = out.loc[out["IMPLIC"].eq(implicate)].copy()
    if out["YY1"].duplicated().any():
        raise ValueError("Expected one row per family after selecting implicate 1.")
    out = out.loc[out["TURNFEAR"].eq(1) & out["NETWORTH"].lt(2_000_000)].copy()
    numeric = [*FINANCIAL_FEATURES, "WGT"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=numeric)
    out = out.loc[out["WGT"].gt(0)].copy()
    if len(out) < 20:
        raise ValueError(f"Too few complete subgroup rows for clustering: {len(out)}")
    return out.reset_index(drop=True)


def transform_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Signed log transform dollar values to reduce long-tail leverage."""
    x = frame[columns].astype(float).copy()
    return np.sign(x) * np.log1p(np.abs(x))


def rank_features(frame: pd.DataFrame) -> pd.Series:
    """Rank eligible dollar measures by variance after trimming 10% per tail."""
    transformed = transform_features(frame, FINANCIAL_FEATURES)
    score = transformed.apply(lambda col: float(trimmed_var(col, limits=(0.1, 0.1))))
    return score.sort_values(ascending=False)


def scaled_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return signed-log features; scaling is fitted inside each K-means pipeline."""
    return transform_features(frame, columns)


def fit_kmeans_pipeline(x: pd.DataFrame, k: int = 4, seed: int = 42):
    model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=k, random_state=seed, n_init=20),
    )
    model.fit(x)
    return model


def evaluate_k(x: pd.DataFrame, k_values=range(2, 13)) -> list[dict]:
    rows = []
    for k in k_values:
        model = fit_kmeans_pipeline(x, k=k)
        labels = model.named_steps["kmeans"].labels_
        scaled = model.named_steps["standardscaler"].transform(x)
        rows.append({
            "k": int(k),
            "inertia": float(model.named_steps["kmeans"].inertia_),
            "silhouette": float(silhouette_score(scaled, labels)),
        })
    return rows


def resample_stability(sample, features, reference):
    """Refit preprocessing and clusters on 20 household bootstrap samples."""
    rng = np.random.default_rng(42)
    x = scaled_matrix(sample, features)
    scores = []
    for _ in range(20):
        boot = x.iloc[rng.integers(0, len(x), len(x))]
        model = fit_kmeans_pipeline(boot)
        scores.append(float(adjusted_rand_score(reference, model.predict(x))))
    return {"replicates": 20, "mean_ari": float(np.mean(scores)), "min_ari": float(np.min(scores)),
            "note": "Agreement on original households; exploratory perturbation diagnostic, not a survey-design confidence interval."}


def implicate_sensitivity(raw, reference_sample, reference_labels, features):
    reference = pd.Series(reference_labels, index=reference_sample.YY1)
    results = []
    for implicate in range(2, 6):
        sample = prepare_sample(raw, implicate)
        model = fit_kmeans_pipeline(scaled_matrix(sample, features))
        labels = pd.Series(model.named_steps["kmeans"].labels_, index=sample.YY1)
        common = reference.index.intersection(labels.index)
        results.append({"implicate": implicate, "eligible_households": len(sample),
                        "common_households": len(common),
                        "ari_common_households": float(adjusted_rand_score(reference.loc[common], labels.loc[common]))})
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_data()
    sample = prepare_sample(raw)
    features = rank_features(sample)
    multi_features = features.head(5).index.tolist()

    x_two = scaled_matrix(sample, TWO_FEATURES)
    x_multi = scaled_matrix(sample, multi_features)
    k_scores = evaluate_k(x_two)
    model_two = fit_kmeans_pipeline(x_two)
    model_multi = fit_kmeans_pipeline(x_multi)
    labels_two = model_two.named_steps["kmeans"].labels_
    labels_multi = model_multi.named_steps["kmeans"].labels_
    x_multi_scaled = model_multi.named_steps["standardscaler"].transform(x_multi)
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(x_multi_scaled)

    seed_labels = []
    for seed in (17, 42, 73, 101, 2026):
        fitted = fit_kmeans_pipeline(x_multi, seed=seed)
        seed_labels.append(fitted.named_steps["kmeans"].labels_)
    stability = [
        float(adjusted_rand_score(seed_labels[i], seed_labels[j]))
        for i, j in combinations(range(len(seed_labels)), 2)
    ]

    assignments = sample[["YY1", "WGT", "TURNFEAR", *FINANCIAL_FEATURES]].copy()
    assignments["cluster_two_feature"] = labels_two
    assignments["cluster_multi_feature"] = labels_multi
    assignments["PC1"] = projected[:, 0]
    assignments["PC2"] = projected[:, 1]
    assignments.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)

    summary = {
        "source_rows": int(len(raw)),
        "source_columns": int(raw.shape[1]),
        "source_families": int(raw["YY1"].nunique()),
        "first_implicate_rows": int(raw["Y1"].sub(10 * raw["YY1"]).eq(1).sum()),
        "credit_fear_and_under_2m_rows": int(len(sample)),
        "filtered_sample_weight_sum": float(sample["WGT"].sum()),
        "selected_multifeatures": multi_features,
        "trimmed_variance": {key: float(value) for key, value in features.items()},
        "k_diagnostics_two_feature": k_scores,
        "k_diagnostics_multi_feature": evaluate_k(x_multi),
        "two_feature_presence_rule_ari": float(adjusted_rand_score(labels_two, sample["DEBT"].gt(0).astype(int) * 2 + sample["HOUSES"].gt(0).astype(int))),
        "resample_stability": resample_stability(sample, multi_features, labels_multi),
        "implicate_sensitivity": implicate_sensitivity(raw, sample, labels_multi, multi_features),
        "cluster_sample_sizes": assignments.groupby("cluster_multi_feature").size().to_dict(),
        "cluster_weighted_shares": (assignments.groupby("cluster_multi_feature")["WGT"].sum() / assignments["WGT"].sum()).to_dict(),
        "cluster_unweighted_medians": assignments.groupby("cluster_multi_feature")[FINANCIAL_FEATURES].median().to_dict(orient="index"),
        "two_feature_silhouette_k4": float(silhouette_score(
            model_two.named_steps["standardscaler"].transform(x_two), labels_two
        )),
        "multi_feature_silhouette_k4": float(silhouette_score(x_multi_scaled, labels_multi)),
        "agreement_between_feature_sets_ari": float(
            adjusted_rand_score(labels_two, labels_multi)
        ),
        "multi_feature_seed_stability_mean_ari": float(np.mean(stability)),
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cluster_weighted_profiles": assignments.groupby("cluster_multi_feature").apply(
            lambda g: pd.Series({
                f"{col}_weighted_mean": float(np.average(g[col], weights=g["WGT"]))
                for col in FINANCIAL_FEATURES
            }), include_groups=False
        ).to_dict(orient="index"),
        "interpretation": "Exploratory unweighted segments; WGT is used for descriptive profiles only, not creditworthiness or lending labels.",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
