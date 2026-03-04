# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
warnings_off = True

import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

P = "data/processed"
MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
          7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def load_csv(name, **kw):
    path = f"{P}/{name}"
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def find_col(df, keywords, dtypes=("float64", "int64")):
    for c in df.columns:
        if any(k in c.lower() for k in keywords) and df[c].dtype in dtypes:
            return c
    return None


forecast = load_csv("part1_forecast_data.csv", parse_dates=["date"])
reviews = load_csv("reviews_clean.csv")
profiles = load_csv("country_profiles_clustered.csv", index_col=0)
feat_imp = load_csv("feature_importance.csv")
hotel_scores = load_csv("hotel_topic_scores.csv", index_col=0)
cluster_sum = load_csv("cluster_summary.csv", index_col=0)
sc_comp = load_csv("model_comparison_sentiment.csv", index_col=0)
word_imp = load_csv("sentiment_word_importance.csv")

print("loaded:")
for name, obj in [("forecast", forecast), ("reviews", reviews), ("profiles", profiles),
                   ("feat_imp", feat_imp), ("hotel_scores", hotel_scores),
                   ("cluster_sum", cluster_sum), ("sc_comp", sc_comp), ("word_imp", word_imp)]:
    if obj is not None:
        print(f"  {name}: {obj.shape}")
    else:
        print(f"  {name}: not found")


# %%
# Revenue Forecasting

if forecast is not None:
    occ = find_col(forecast, ["occ"])
    arr = find_col(forecast, ["room_rate", "arr", "avg_rate"])
    rev = find_col(forecast, ["revpar"])
    vis = "total_visitors" if "total_visitors" in forecast.columns else None

    if rev is None and occ and arr:
        forecast["revpar_calc"] = forecast[occ] / 100 * forecast[arr]
        rev = "revpar_calc"

    latest = forecast.iloc[-1]
    print("--- Revenue Forecasting ---\n")
    if occ:
        print(f"  occupancy:       {latest[occ]:.1f}%")
    if arr:
        print(f"  avg room rate:   ${latest[arr]:,.0f}")
    if rev:
        print(f"  revpar:          ${latest[rev]:,.0f}")
    if vis:
        print(f"  visitors:        {latest[vis]:,.0f}")


# %%
# occupancy, room rate, revpar, visitors over time

if forecast is not None:
    metrics = [(occ, "Occupancy Rate (%)"), (arr, "Avg Room Rate ($)"),
               (rev, "RevPAR ($)"), (vis, "Total Visitors")]
    metrics = [(c, label) for c, label in metrics if c and c in forecast.columns]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 3.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, metrics):
        ax.plot(forecast["date"], forecast[col], color="#2c3e50", linewidth=1.2, label="Monthly")
        ma = forecast[col].rolling(12).mean()
        ax.plot(forecast["date"], ma, color="#e74c3c", linewidth=2, linestyle="--", label="12-month MA")
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Hotel Performance Trends", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/dashboard_trends.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%
# occupancy heatmap by month and year

if forecast is not None and occ and "year" in forecast.columns:
    pv = forecast.pivot_table(values=occ, index="year", columns="month_num", aggfunc="mean")
    pv.columns = [MONTHS[m] for m in pv.columns]

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pv, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax, linewidths=0.5)
    ax.set_title("Occupancy Rate by Month and Year", fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig("outputs/dashboard_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%
# feature importance

if feat_imp is not None:
    top = feat_imp.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["importance"], color="#2c3e50")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (XGBoost)", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/dashboard_feat_imp.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%
# Guest Sentiment

if reviews is not None:
    print("--- Guest Sentiment ---\n")
    print(f"  reviews:    {len(reviews):,}")
    if "rating" in reviews.columns:
        print(f"  avg rating: {reviews['rating'].mean():.2f}/5")
    if "sentiment" in reviews.columns:
        pos_pct = (reviews["sentiment"] == "positive").mean() * 100
        print(f"  positive:   {pos_pct:.0f}%")


# %%
# rating distribution and sentiment split

if reviews is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "rating" in reviews.columns:
        reviews["rating"].value_counts().sort_index().plot(
            kind="bar", ax=axes[0], color="#2c3e50")
        axes[0].set_title("Rating Distribution", fontweight="bold")
        axes[0].set_xlabel("Stars")
        axes[0].set_ylabel("Count")

    if "sentiment" in reviews.columns:
        reviews["sentiment"].value_counts().plot(
            kind="pie", ax=axes[1], autopct="%1.1f%%",
            colors=["#27ae60", "#f39c12", "#c0392b"])
        axes[1].set_title("Sentiment Split", fontweight="bold")
        axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig("outputs/dashboard_sentiment.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%
# hotel ranking by average rating

if reviews is not None and "rating" in reviews.columns and "hotel_name" in reviews.columns:
    avg = reviews.groupby("hotel_name")["rating"].agg(["mean", "count"]).reset_index()
    avg.columns = ["hotel_name", "avg_rating", "n"]
    avg = avg[avg["n"] >= 5].sort_values("avg_rating", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(avg)))
    ax.barh(avg["hotel_name"], avg["avg_rating"], color=colors)
    ax.set_xlabel("Avg Rating")
    ax.set_title("Top 15 Hotels by Rating (min 5 reviews)", fontweight="bold")
    ax.set_xlim(3.5, 5.0)
    plt.tight_layout()
    plt.savefig("outputs/dashboard_hotel_ranking.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%
# service quality radar for top 4 hotels

if hotel_scores is not None:
    topic_cols = [c for c in hotel_scores.columns if c not in ("avg_rating", "num_reviews")]
    if topic_cols:
        top4 = hotel_scores.head(4).index.tolist()
        angles = np.linspace(0, 2 * np.pi, len(topic_cols), endpoint=False).tolist()
        angles += angles[:1]
        palette = ["#2c3e50", "#e74c3c", "#27ae60", "#f39c12"]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for i, hotel in enumerate(top4):
            vals = [float(hotel_scores.loc[hotel, c]) for c in topic_cols]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", label=hotel[:30], color=palette[i % 4], linewidth=2)
            ax.fill(angles, vals, alpha=0.1, color=palette[i % 4])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace("_mean", "") for c in topic_cols], fontsize=8)
        ax.set_title("Service Quality Radar — Top Hotels", fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
        plt.tight_layout()
        plt.savefig("outputs/dashboard_service_radar.png", dpi=150, bbox_inches="tight")
        plt.show()


# %%
# most predictive words

if word_imp is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pos = word_imp[word_imp["sentiment"] == "positive"].head(10)
    axes[0].barh(pos["word"], pos["coefficient"], color="#27ae60")
    axes[0].set_title("Positive Indicators", fontweight="bold")
    axes[0].invert_yaxis()

    neg = word_imp[word_imp["sentiment"] == "negative"].head(10)
    axes[1].barh(neg["word"], neg["coefficient"], color="#c0392b")
    axes[1].set_title("Negative Indicators", fontweight="bold")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig("outputs/dashboard_word_imp.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%
# model comparison (sentiment)

if sc_comp is not None:
    print("sentiment model comparison:")
    print(sc_comp.round(4).to_string())


# %%
# Tourist Source Market Segmentation
if profiles is not None:
    print("--- Tourist Segments ---\n")
    if "segment_name" in profiles.columns:
        for seg in profiles["segment_name"].unique():
            countries = profiles[profiles["segment_name"] == seg].index.tolist()
            print(f"  {seg}")
            print(f"    {', '.join(countries[:8])}\n")


# %%
# cluster scatter (PCA)

if profiles is not None and "pca_1" in profiles.columns:
    color_col = "segment_name" if "segment_name" in profiles.columns else "cluster"
    segments = profiles[color_col].unique()
    palette = ["#2c3e50", "#e74c3c", "#27ae60", "#f39c12", "#6C5CE7"]

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, seg in enumerate(segments):
        sub = profiles[profiles[color_col] == seg]
        ax.scatter(sub["pca_1"], sub["pca_2"], label=seg,
                   s=100, color=palette[i % len(palette)], alpha=0.7,
                   edgecolors="black", linewidth=0.5)

    for _, row in profiles.reset_index().iterrows():
        ax.annotate(row["country"], (row["pca_1"], row["pca_2"]),
                    fontsize=7, ha="center", va="bottom")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Tourist Source Markets — Cluster Map", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/dashboard_cluster_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%
# volume vs spending

if profiles is not None and "annual_volume" in profiles.columns:
    plot_df = profiles.reset_index().copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        plot_df["annual_volume"], plot_df["avg_spending"],
        s=100, c=plot_df["seasonality_index"], cmap="RdYlGn_r",
        alpha=0.7, edgecolors="black", linewidth=0.5
    )
    for _, row in plot_df.iterrows():
        ax.annotate(row["country"], (row["annual_volume"], row["avg_spending"]),
                    fontsize=7, ha="center", va="bottom")

    ax.set_xlabel("Annual Visitors")
    ax.set_ylabel("Avg Spending (SGD)")
    ax.set_title("Volume vs Spending", fontweight="bold")
    plt.colorbar(scatter, label="Seasonality Index")
    plt.tight_layout()
    plt.savefig("outputs/dashboard_vol_vs_spend.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%
# segment profiles table

if cluster_sum is not None:
    print("segment profiles:")
    print(cluster_sum.round(2).to_string())


# %%
# segment radar

if cluster_sum is not None and profiles is not None:
    feature_cols = cluster_sum.columns.tolist()
    cluster_names = profiles.groupby("cluster")["segment_name"].first().to_dict() if "segment_name" in profiles.columns else {}
    K = len(cluster_sum)

    angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
    angles += angles[:1]
    palette = ["#2c3e50", "#e74c3c", "#27ae60", "#f39c12", "#6C5CE7"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for c in range(K):
        vals = cluster_sum.iloc[c].values.tolist()
        vals_norm = [
            (v - cluster_sum[f].min()) / (cluster_sum[f].max() - cluster_sum[f].min() + 0.001)
            for v, f in zip(vals, feature_cols)
        ]
        vals_norm += vals_norm[:1]
        label = cluster_names.get(c, f"Cluster {c}")
        ax.plot(angles, vals_norm, "o-", label=label, color=palette[c % len(palette)], linewidth=2)
        ax.fill(angles, vals_norm, alpha=0.1, color=palette[c % len(palette)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Tourist Segments — Comparative Profile", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    plt.tight_layout()
    plt.savefig("outputs/dashboard_segment_radar.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%

print("--- Executive Summary ---\n")

if forecast is not None and occ:
    latest_occ = forecast[occ].iloc[-1]
    avg_occ = forecast[occ].mean()
    by_month = forecast.groupby("month_num")[occ].mean()
    peak = MONTHS[by_month.idxmax()]
    low = MONTHS[by_month.idxmin()]
    print(f"occupancy: {latest_occ:.1f}% (avg {avg_occ:.1f}%)")
    print(f"peak: {peak} ({by_month.max():.1f}%), low: {low} ({by_month.min():.1f}%)")

if reviews is not None and "rating" in reviews.columns:
    avg_r = reviews["rating"].mean()
    pos = (reviews["sentiment"] == "positive").mean() * 100 if "sentiment" in reviews.columns else 0
    print(f"\nguest rating: {avg_r:.2f}/5, {pos:.0f}% positive")
    if "hotel_name" in reviews.columns:
        top_hotel = reviews.groupby("hotel_name")["rating"].mean().idxmax()
        print(f"top hotel: {top_hotel}")

print("\nrecommendations:")
print("  1. dynamic pricing — raise rates during F1, CNY, year-end; promote Jan-Feb")
print("  2. service investment — staff quality is top driver of positive reviews")
print("  3. segment marketing — premium packages for long-haul, value deals for regional")
print("  4. event alignment — build campaigns around F1 GP, CNY, National Day")
print("  5. off-peak fill — target western long-haul markets year-round")

print("\nall charts saved to outputs/")