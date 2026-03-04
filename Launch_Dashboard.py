# %%
import os

required_processed = [
    "part1_forecast_data.csv",
    "reviews_clean.csv",
    "country_profiles_clustered.csv",
]

required_models = [
    "forecast_xgb_model.pkl",
    "sentiment_model.pkl",
    "cluster_model.pkl",
]

optional_processed = [
    "feature_importance.csv",
    "hotel_topic_scores.csv",
    "cluster_summary.csv",
    "model_comparison_forecast.csv",
    "model_comparison_sentiment.csv",
]

print("dashboard readiness check\n")

all_ok = True
print("required:")
for f in required_processed:
    path = f"data/processed/{f}"
    ok = os.path.exists(path)
    print(f"  {'ok' if ok else 'MISSING'}  {path}")
    if not ok:
        all_ok = False

for f in required_models:
    path = f"models/{f}"
    ok = os.path.exists(path)
    print(f"  {'ok' if ok else 'MISSING'}  {path}")
    if not ok:
        all_ok = False

print("\noptional:")
for f in optional_processed:
    path = f"data/processed/{f}"
    ok = os.path.exists(path)
    print(f"  {'ok' if ok else 'missing'}  {path}")

if all_ok:
    print("\nall good — open Step6_Dashboard.py and Run All Cells")
else:
    print("\nsome required files missing, go back and run the relevant steps")
# %%
