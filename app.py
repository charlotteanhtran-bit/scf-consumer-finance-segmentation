"""Interactive Streamlit dashboard for the SCF consumer segmentation."""
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
st.caption("SCF 2022 · exploratory K-means clusters · public-use data")

@st.cache_data(show_spinner="Downloading and preparing the public SCF data…")
def get_sample():
    raw = load_raw_data()
    return raw.shape, prepare_sample(raw)

try:
    raw_shape, df = get_sample()
except Exception as exc:
    st.error(f"Could not prepare SCF data: {exc}")
    st.stop()

ranked = rank_features(df)
default_features = ranked.head(5).index.tolist()
x_two = scaled_matrix(df, TWO_FEATURES)
diagnostics = evaluate_k(x_two)

with st.sidebar:
    st.header("Analysis controls")
    k = st.slider("Number of clusters", min_value=2, max_value=8, value=4)
    features = st.multiselect("Multi-feature set", options=FINANCIAL_FEATURES,
                              default=default_features, max_selections=5)
    if not features:
        st.warning("Choose at least one financial feature.")
        st.stop()

x_multi = scaled_matrix(df, features)
two_model = fit_kmeans_pipeline(x_two, k=k)
multi_model = fit_kmeans_pipeline(x_multi, k=k)
two_labels = two_model.named_steps["kmeans"].labels_
multi_labels = multi_model.named_steps["kmeans"].labels_
pca_input = multi_model.named_steps["standardscaler"].transform(x_multi)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(pca_input)

top = st.columns(4)
top[0].metric("Source rows", f"{raw_shape[0]:,}")
top[1].metric("SCF families", f"{df['YY1'].nunique():,}")
top[2].metric("Credit-fearful, net worth below $2m", f"{len(df):,}")
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
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Silhouette score: {silhouette_score(two_model.named_steps['standardscaler'].transform(x_two), two_labels):.3f}. It summarizes separation in the scaled feature space; it is not an accuracy score.")

with tabs[1]:
    st.subheader("Multi-dimensional finance profiles, shown through PCA")
    cloud = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": multi_labels.astype(str)})
    for col in FINANCIAL_FEATURES:
        cloud[col] = df[col].to_numpy()
    fig = px.scatter(cloud, x="PC1", y="PC2", color="cluster", hover_data=FINANCIAL_FEATURES,
                     opacity=0.68, title="PCA projection of standardized multi-feature K-means clusters")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("PC1 and PC2 are linear combinations of selected standardized features. "
               f"Together they retain {pca.explained_variance_ratio_.sum():.1%} of transformed variance.")
    profile = df.assign(cluster=multi_labels).groupby("cluster").apply(
        lambda g: pd.Series({col: np.average(g[col], weights=g["WGT"]) for col in FINANCIAL_FEATURES}),
        include_groups=False)
    st.markdown("#### Survey-weighted mean profile (USD in 2022 survey units)")
    st.dataframe(profile.style.format("{:,.0f}"), use_container_width=True)
    st.metric("Agreement with two-feature clustering (ARI)", f"{adjusted_rand_score(two_labels, multi_labels):.3f}")

with tabs[2]:
    st.subheader("Select k and inspect candidate feature variance")
    diag = pd.DataFrame(diagnostics)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=diag.k, y=diag.inertia, mode="lines+markers", name="Inertia", yaxis="y"))
    fig.add_trace(go.Scatter(x=diag.k, y=diag.silhouette, mode="lines+markers", name="Silhouette", yaxis="y2"))
    fig.update_layout(title="k diagnostic on DEBT + HOUSES", xaxis_title="Number of clusters",
                      yaxis=dict(title="Inertia", side="left"),
                      yaxis2=dict(title="Silhouette", overlaying="y", side="right", range=[-1, 1]),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ranked.rename("trimmed variance after signed log1p").to_frame(), use_container_width=True)

with tabs[3]:
    st.subheader("Source, selection, and interpretation")
    st.markdown("""
    - Source: Federal Reserve 2022 SCF public data; CSV obtained from a public mirror documented in README.
    - The source stores five imputation records per family. We keep implicate 1 using the Federal Reserve's documented Y1 and YY1 construction so each family is counted once.
    - The dashboard analyzes TURNFEAR == 1 and net worth below $2 million, following Project 6's subgroup definition.
    - Cluster IDs are arbitrary labels. They are not validated risk classes and must not be used to approve, price, or deny credit.
    - The SCF is a complex survey. K-means is unweighted; family weights are used for descriptive cluster profiles only. This is not full design-based population inference.
    """)
    st.dataframe(df[["YY1", "TURNFEAR", "NETWORTH", *FINANCIAL_FEATURES, "WGT"]].head(25), use_container_width=True)
