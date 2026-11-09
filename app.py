"""Explore financial groups in the 2022 SCF sample."""
from __future__ import annotations

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from analyze import (
    FINANCIAL_FEATURES, TWO_FEATURES, evaluate_k, fit_kmeans_pipeline,
    load_raw_data, prepare_sample, rank_features, scaled_matrix,
)

st.set_page_config(page_title="SCF Consumer Segmentation", page_icon="📊", layout="wide")
st.title("U.S. Consumer Finance Segmentation")
st.caption("Explore debt, housing, income, assets, and net worth in the 2022 SCF.")

@st.cache_data(show_spinner="Downloading and preparing the public SCF data…")
def get_sample():
    raw = load_raw_data()
    return raw.shape, raw["YY1"].nunique(), prepare_sample(raw)

try:
    raw_shape, source_families, df = get_sample()
except Exception as exc:
    st.error(f"Could not prepare SCF data: {exc}")
    st.stop()

@st.cache_data
def cached_diagnostics(x):
    return evaluate_k(x)

@st.cache_resource
def cached_model(x, k):
    return fit_kmeans_pipeline(x, k=k)

ranked = rank_features(df)
default_features = ranked.head(5).index.tolist()
x_two = scaled_matrix(df, TWO_FEATURES)
diagnostics = cached_diagnostics(x_two)

with st.sidebar:
    st.header("Choose your model")
    k = st.slider("Number of clusters", min_value=2, max_value=8, value=4)
    features = st.multiselect("Financial variables", options=FINANCIAL_FEATURES,
                              default=default_features, max_selections=5)
    if len(features) < 2:
        st.warning("Choose at least two financial features for the PCA view.")
        st.stop()

x_multi = scaled_matrix(df, features)
two_model = cached_model(x_two, k)
multi_model = cached_model(x_multi, k)
two_labels = two_model.named_steps["kmeans"].labels_
multi_labels = multi_model.named_steps["kmeans"].labels_
pca_input = multi_model.named_steps["standardscaler"].transform(x_multi)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(pca_input)

top = st.columns(4)
top[0].metric("Source rows", f"{raw_shape[0]:,}")
top[1].metric("SCF families", f"{source_families:,}")
top[2].metric("Families in the filtered sample", f"{len(df):,}")
top[3].metric("PCA variance shown", f"{pca.explained_variance_ratio_.sum():.1%}")
tabs = st.tabs(["Two-feature clusters", "Multi-feature segments", "Feature diagnostics", "Data notes"])

with tabs[0]:
    st.subheader("Debt and home value")
    two_plot = df[["DEBT", "HOUSES", "WGT"]].copy()
    two_plot["cluster"] = two_labels.astype(str)
    two_plot["Debt (log) / $"] = np.log1p(two_plot["DEBT"].clip(lower=0))
    two_plot["Home value (log) / $"] = np.log1p(two_plot["HOUSES"].clip(lower=0))
    fig = px.scatter(two_plot, x="Debt (log) / $", y="Home value (log) / $", color="cluster",
                     size="WGT", size_max=12, opacity=0.65,
                     title="K-means clusters from DEBT and HOUSES (log1p dollar values)",
                     labels={"cluster": "Cluster"})
    fig.update_layout(legend_title_text="Cluster")
    st.plotly_chart(fig, width="stretch")
    st.info(f"Silhouette score: {silhouette_score(two_model.named_steps['standardscaler'].transform(x_two), two_labels):.3f}. It summarizes separation in the scaled feature space; it is not an accuracy score.")

with tabs[1]:
    st.subheader("Compare financial profiles")
    cloud = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": multi_labels.astype(str)})
    for col in FINANCIAL_FEATURES:
        cloud[col] = df[col].to_numpy()
    fig = px.scatter(cloud, x="PC1", y="PC2", color="cluster", hover_data=FINANCIAL_FEATURES,
                     opacity=0.68, title="Financial groups shown in two dimensions")
    st.plotly_chart(fig, width="stretch")
    st.caption("The chart compresses the selected variables into two axes. "
               f"Together they retain {pca.explained_variance_ratio_.sum():.1%} of the variation after transformation and scaling.")
    profile = df.assign(cluster=multi_labels).groupby("cluster").apply(
        lambda g: pd.Series({col: np.average(g[col], weights=g["WGT"]) for col in FINANCIAL_FEATURES}),
        include_groups=False)
    st.markdown("#### Survey-weighted mean profile (USD in 2022 survey units)")
    profile.insert(0, "sample_households", df.assign(cluster=multi_labels).groupby("cluster").size())
    profile.insert(1, "weighted_share", df.assign(cluster=multi_labels).groupby("cluster")["WGT"].sum() / df["WGT"].sum())
    st.dataframe(profile.style.format({c: "{:,.0f}" for c in profile.columns if c != "weighted_share"}).format({"weighted_share": "{:.1%}"}), width="stretch")
    st.caption("Shares refer to the filtered sample. Cluster numbers are just labels. Compare the means with the medians below to see how large values affect each profile.")
    st.markdown("#### Unweighted median profile (USD)")
    st.dataframe(df.assign(cluster=multi_labels).groupby("cluster")[FINANCIAL_FEATURES].median())
    st.metric("Agreement with two-feature clustering (ARI)", f"{adjusted_rand_score(two_labels, multi_labels):.3f}")

with tabs[2]:
    st.subheader("Compare cluster counts and variables")
    diag = pd.DataFrame(diagnostics)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=diag.k, y=diag.inertia, mode="lines+markers", name="Inertia", yaxis="y"))
    fig.add_trace(go.Scatter(x=diag.k, y=diag.silhouette, mode="lines+markers", name="Silhouette", yaxis="y2"))
    fig.update_layout(title="Cluster counts for debt and home value", xaxis_title="Number of clusters",
                      yaxis=dict(title="Inertia", side="left"),
                      yaxis2=dict(title="Silhouette", overlaying="y", side="right", range=[-1, 1]),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.markdown("#### Multi-feature cluster-count diagnostics")
    st.dataframe(pd.DataFrame(cached_diagnostics(x_multi)))
    rule_labels = df["DEBT"].gt(0).astype(int) * 2 + df["HOUSES"].gt(0).astype(int)
    st.metric("Two-feature agreement with debt/home presence rules (ARI)", f"{adjusted_rand_score(two_labels, rule_labels):.3f}")
    st.caption("When this agreement is high, the model mostly separates families by whether they have debt and home value. The table below ranks variation across the five available variables; it does not choose which ones to use.")
    st.dataframe(ranked.rename("trimmed variance after signed log1p").to_frame(), width="stretch")

with tabs[3]:
    st.subheader("About this sample")
    st.markdown("""
    - Data: the Federal Reserve\'s 2022 Survey of Consumer Finances, downloaded from the CSV mirror linked in the README.
    - Each family has five imputed records. This view keeps version 1, identified from Y1 and YY1, so each family appears once.
    - The sample includes families who held back from applying for credit because they expected denial (TURNFEAR == 1), with net worth below $2 million.
    - Cluster numbers are just labels. These groups have not been tested as credit-risk categories and are not suitable for lending decisions.
    - K-means treats each family equally. Survey weights are used for the profile summaries, but the analysis does not calculate uncertainty for the full SCF survey design.
    """)
    st.dataframe(df[["YY1", "TURNFEAR", *FINANCIAL_FEATURES, "WGT"]].head(25), width="stretch")
