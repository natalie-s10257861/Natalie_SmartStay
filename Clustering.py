# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
import joblib
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# %%
# load country profiles 

profiles = pd.read_csv("data/processed/part3_country_profiles.csv", index_col=0)
print(f"loaded: {profiles.shape[0]} countries, {profiles.shape[1]} features")
print(f"features: {profiles.columns.tolist()}")
print(profiles.round(1).to_string())

# also load raw visitor data for monthly charts later
visitors = None
if os.path.exists("data/processed/visitors_clean.csv"):
    visitors = pd.read_csv("data/processed/visitors_clean.csv", parse_dates=["date"])
    print(f"\nvisitor arrivals: {visitors.shape}")

# %%
# volume vs spending scatter

plot_df = profiles.reset_index().copy()

fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(
    plot_df["annual_volume"],
    plot_df["avg_spending"],
    s=100,
    c=plot_df["seasonality_index"],
    cmap="RdYlGn_r",
    alpha=0.7,
    edgecolors="black",
    linewidth=0.5
)
for _, row in plot_df.iterrows():
    ax.annotate(row["country"], (row["annual_volume"], row["avg_spending"]),
                fontsize=7, ha="center", va="bottom")

ax.set_xlabel("Annual Visitors")
ax.set_ylabel("Avg Spending per Visitor (SGD)")
ax.set_title("Tourist Markets: Volume vs Spending", fontweight="bold")
plt.colorbar(scatter, label="Seasonality Index")
plt.tight_layout()
plt.savefig("outputs/volume_vs_spending.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# feature scaling
# kmeans uses distance, so without scaling volume (millions) would dominate
# spending (thousands) — standardscaler puts everything on the same scale

feature_cols = ["annual_volume", "growth_rate", "seasonality_index",
                "avg_spending", "covid_recovery_pct", "volatility"]
feature_cols = [c for c in feature_cols if c in profiles.columns]
print(f"clustering features: {feature_cols}")

X = profiles[feature_cols].copy()

# log-transform volume so china doesn't dominate everything
if "annual_volume" in X.columns:
    X["annual_volume"] = np.log1p(X["annual_volume"])

X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"scaled: {X_scaled.shape}")


# %%
# find optimal K using elbow method and silhouette score

K_range = range(2, min(8, len(profiles)))
inertias = []
sil_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
axes[0].set_title("Elbow Method", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].grid(True)

axes[1].plot(K_range, sil_scores, "ro-", linewidth=2, markersize=8)
axes[1].set_title("Silhouette Score", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("outputs/optimal_k.png", dpi=150, bbox_inches="tight")
plt.show()

optimal_k = list(K_range)[np.argmax(sil_scores)]
print(f"optimal K = {optimal_k} (silhouette: {max(sil_scores):.3f})")


# %%
# run kmeans

K = 4  # adjust based on the elbow/silhouette results above

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
profiles["cluster"] = kmeans.fit_predict(X_scaled)

print(f"kmeans done (K={K}), silhouette: {silhouette_score(X_scaled, profiles['cluster']):.3f}")

for c in range(K):
    countries = profiles[profiles["cluster"] == c].index.tolist()
    print(f"  cluster {c}: {', '.join(countries)}")


# %%
# pca to 2 dimensions for visualization

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

profiles["pca_1"] = X_pca[:, 0]
profiles["pca_2"] = X_pca[:, 1]

print(f"pca variance explained: {pca.explained_variance_ratio_}")
print(f"total: {pca.explained_variance_ratio_.sum():.1%}")

# interactive scatter
fig = px.scatter(
    profiles.reset_index(), x="pca_1", y="pca_2",
    color="cluster", text="country",
    size="annual_volume",
    hover_data=["growth_rate", "avg_spending", "seasonality_index"],
    title=f"Tourist Segments (K={K}) — PCA",
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={
        "pca_1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        "pca_2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        "cluster": "Segment",
    }
)
fig.update_traces(textposition="top center")
fig.update_layout(height=600)
fig.write_html("outputs/cluster_scatter.html")
print("saved: outputs/cluster_scatter.html")

# static version
fig_s, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1], c=profiles["cluster"],
    s=np.clip(profiles["annual_volume"] / profiles["annual_volume"].max() * 500, 50, 500),
    cmap="Set2", alpha=0.7, edgecolors="black", linewidth=0.5
)
for _, row in profiles.reset_index().iterrows():
    ax.annotate(row["country"], (row["pca_1"], row["pca_2"]),
                fontsize=8, ha="center", va="bottom", fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title(f"Tourist Source Market Segments (K={K})", fontweight="bold")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("outputs/cluster_scatter_static.png", dpi=150, bbox_inches="tight")
plt.show()


# %%
# segment profiling — auto-label each cluster based on its characteristics

cluster_summary = profiles.groupby("cluster")[feature_cols].mean().round(1)

cluster_names = {}
for c in range(K):
    data = cluster_summary.loc[c]
    countries = profiles[profiles["cluster"] == c].index.tolist()

    spend_high = data.get("avg_spending", 0) > cluster_summary["avg_spending"].median() if "avg_spending" in cluster_summary.columns else False
    vol_high = data.get("annual_volume", 0) > cluster_summary["annual_volume"].median() if "annual_volume" in cluster_summary.columns else False
    growth_high = data.get("growth_rate", 0) > cluster_summary["growth_rate"].median() if "growth_rate" in cluster_summary.columns else False

    if spend_high and vol_high:
        name = "High-Value Power Markets"
    elif spend_high:
        name = "Premium Long-Haul Travellers"
    elif growth_high:
        name = "Emerging Growth Markets"
    else:
        name = "Regional Volume Drivers"

    cluster_names[c] = name

    print(f"\n{name}")
    print(f"  countries: {', '.join(countries[:8])}")
    for feat in feature_cols:
        print(f"  {feat:25s}: {data[feat]:>10.1f}")

profiles["segment_name"] = profiles["cluster"].map(cluster_names)


# %%
# radar chart comparing segments (matplotlib)

angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
angles += angles[:1]
colors = ["#2E86AB", "#E84855", "#28A745", "#FFC107", "#6C5CE7"]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for c in range(K):
    vals = cluster_summary.loc[c].values.tolist()
    vals_norm = [
        (v - cluster_summary[f].min()) / (cluster_summary[f].max() - cluster_summary[f].min() + 0.001)
        for v, f in zip(vals, feature_cols)
    ]
    vals_norm += vals_norm[:1]
    ax.plot(angles, vals_norm, "o-", label=cluster_names.get(c, f"Cluster {c}"),
            color=colors[c % len(colors)], linewidth=2)
    ax.fill(angles, vals_norm, alpha=0.1, color=colors[c % len(colors)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(feature_cols, fontsize=8)
ax.set_ylim(0, 1)
ax.set_title("Tourist Segments — Comparative Profile", fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig("outputs/segment_radar.png", dpi=150, bbox_inches="tight")
plt.show()


# %%
# monthly arrival patterns by cluster

if visitors is not None and "country" in visitors.columns:
    cluster_map = profiles["cluster"].to_dict()
    visitors_c = visitors.copy()
    visitors_c["cluster"] = visitors_c["country"].map(cluster_map)
    visitors_c = visitors_c.dropna(subset=["cluster"])
    visitors_c["cluster"] = visitors_c["cluster"].astype(int)

    arr_col = "arrivals" if "arrivals" in visitors_c.columns else \
              [c for c in visitors_c.select_dtypes(include=[np.number]).columns
               if c not in ["year", "month_num", "cluster"]][0]

    monthly_cluster = visitors_c.groupby(["month_num", "cluster"])[arr_col].mean().reset_index()
    monthly_cluster["segment"] = monthly_cluster["cluster"].map(cluster_names)

    segments = monthly_cluster["segment"].unique()
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8 / len(segments)
    for i, seg in enumerate(segments):
        seg_data = monthly_cluster[monthly_cluster["segment"] == seg]
        offset = (i - len(segments) / 2 + 0.5) * bar_width
        ax.bar(seg_data["month_num"] + offset, seg_data[arr_col],
               width=bar_width, label=seg, color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Arrivals")
    ax.set_title("Average Monthly Arrivals by Segment", fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/monthly_by_segment.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# save models and results

joblib.dump(kmeans, "models/cluster_model.pkl")
joblib.dump(scaler, "models/cluster_scaler.pkl")
joblib.dump(pca, "models/cluster_pca.pkl")
profiles.to_csv("data/processed/country_profiles_clustered.csv")
cluster_summary.to_csv("data/processed/cluster_summary.csv")

print("saved:")
print("  models/cluster_model.pkl")
print("  models/cluster_scaler.pkl")
print("  models/cluster_pca.pkl")
print("  data/processed/country_profiles_clustered.csv")
print("  data/processed/cluster_summary.csv")
# %%
