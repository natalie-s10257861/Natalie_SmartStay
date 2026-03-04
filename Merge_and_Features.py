# %%
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# %%
# load all cleaned datasets from processed

PROCESSED = "data/processed/"

def load_csv(filename, parse_dates_col=None):
    path = os.path.join(PROCESSED, filename)
    if not os.path.exists(path):
        return None
    if parse_dates_col:
        return pd.read_csv(path, parse_dates=[parse_dates_col])
    return pd.read_csv(path)

hotel_monthly = load_csv("hotel_monthly_clean.csv", parse_dates_col="date")
hotel_tier = load_csv("hotel_tier_clean.csv", parse_dates_col="date")
visitors = load_csv("visitors_clean.csv", parse_dates_col="date")
hotel_annual = load_csv("hotel_annual_clean.csv")
receipts = load_csv("tourism_receipts_clean.csv")
receipts_qtr = load_csv("tourism_receipts_qtr_clean.csv")
reviews = load_csv("reviews_clean.csv")

datasets = {
    "hotel_monthly": hotel_monthly,
    "hotel_tier": hotel_tier,
    "visitors": visitors,
    "hotel_annual": hotel_annual,
    "receipts": receipts,
    "receipts_qtr": receipts_qtr,
    "reviews": reviews,
}

for name, df in datasets.items():
    if df is not None:
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} cols")
    else:
        print(f"  {name}: not found")


# %%
# aggregate visitor arrivals by month
# the raw data has one row per country per month — we need totals plus
# the top 5 source markets as separate columns for the forecasting model

monthly_visitors = None

if visitors is not None:
    # find the arrivals column
    arr_col = None
    for col in visitors.columns:
        if "arrival" in col.lower() or col == "arrivals":
            if visitors[col].dtype in ["int64", "float64"]:
                arr_col = col
                break
    if arr_col is None:
        for col in visitors.select_dtypes(include=[np.number]).columns:
            if col not in ["year", "month_num"]:
                arr_col = col
                break

    print(f"arrivals column: '{arr_col}'")

    # total visitors per month
    monthly_visitors = (visitors.groupby("date")
                        .agg(total_visitors=(arr_col, "sum"))
                        .reset_index())

    print(f"monthly totals: {len(monthly_visitors)} months")
    print(f"range: {monthly_visitors['date'].min()} to {monthly_visitors['date'].max()}")
    print(f"avg monthly visitors: {monthly_visitors['total_visitors'].mean():,.0f}")

    # top 5 countries as individual columns
    if "country" in visitors.columns:
        top5 = visitors.groupby("country")[arr_col].sum().nlargest(5).index.tolist()
        print(f"top 5 source markets: {top5}")

        for country in top5:
            safe_name = country.lower().replace(" ", "_").replace("'", "")
            col_name = f"visitors_{safe_name}"
            country_data = (visitors[visitors["country"] == country]
                           .groupby("date")[arr_col].sum()
                           .reset_index()
                           .rename(columns={arr_col: col_name}))
            monthly_visitors = monthly_visitors.merge(country_data, on="date", how="left")

    monthly_visitors = monthly_visitors.fillna(0)
    print(f"\nvisitor table: {monthly_visitors.shape}")
    print(monthly_visitors.head().to_string())


# %%
# merge hotel monthly stats with visitor arrivals on date

if hotel_monthly is not None:
    master = hotel_monthly.copy()
    print(f"hotel monthly: {master.shape[1]} columns")

    if monthly_visitors is not None:
        master = master.merge(monthly_visitors, on="date", how="left")
        visitor_cols = [c for c in master.columns if "visitor" in c.lower()]
        master[visitor_cols] = master[visitor_cols].fillna(method="ffill").fillna(method="bfill")
        print(f"after merge: {master.shape[1]} columns (added {len(visitor_cols)} visitor cols)")

    print(f"master dataset: {master.shape}")
    print(master.head().to_string())
else:
    master = None
    print("hotel_monthly not available, can't build master dataset")


# %%
# feature engineering
# adds SG-specific event flags, covid indicators, lag features,
# rolling averages, and cyclical month encoding

def add_sg_features(df):
    df = df.copy()

    if "date" not in df.columns:
        print("no date column, skipping features")
        return df

    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # sg holidays and events
    cny_months = {
        2013: 2, 2014: 1, 2015: 2, 2016: 2, 2017: 1, 2018: 2, 2019: 2,
        2020: 1, 2021: 2, 2022: 2, 2023: 1, 2024: 2, 2025: 1, 2026: 2
    }
    df["is_cny"] = df.apply(
        lambda r: int(cny_months.get(r["year"]) == r["month_num"]), axis=1
    )

    f1_cancelled = [2020, 2021]
    df["is_f1_gp"] = df.apply(
        lambda r: int(r["month_num"] == 9 and r["year"] not in f1_cancelled), axis=1
    )

    df["is_national_day"] = (df["month_num"] == 8).astype(int)
    df["is_year_end"] = df["month_num"].isin([11, 12]).astype(int)
    df["is_school_hol"] = df["month_num"].isin([6, 11, 12]).astype(int)
    df["is_hari_raya"] = df["month_num"].isin([4, 5]).astype(int)
    df["is_deepavali"] = df["month_num"].isin([10, 11]).astype(int)

    # major one-off events (Trump-Kim summit, Taylor Swift Eras Tour, APEC)
    major_events = ["2018-06", "2023-03", "2024-03", "2023-11"]
    df["has_major_event"] = df["date"].dt.strftime("%Y-%m").isin(major_events).astype(int)

    print("added: event and holiday flags")

    # covid impact flags
    df["is_circuit_breaker"] = (
        (df["date"] >= "2020-04-01") & (df["date"] <= "2020-06-30")
    ).astype(int)
    df["is_covid"] = (
        (df["date"] >= "2020-02-01") & (df["date"] <= "2022-03-31")
    ).astype(int)
    df["is_post_covid"] = (df["date"] >= "2022-04-01").astype(int)

    print("added: covid flags")

    # lag and rolling features for occupancy
    occ_col = None
    for col in df.columns:
        if "occ" in col.lower() and df[col].dtype in ["float64", "int64"]:
            occ_col = col
            break

    if occ_col:
        for lag in [1, 2, 3, 6, 12]:
            df[f"occ_lag_{lag}"] = df[occ_col].shift(lag)

        df["occ_ma3"] = df[occ_col].rolling(3).mean()
        df["occ_ma6"] = df[occ_col].rolling(6).mean()
        df["occ_ma12"] = df[occ_col].rolling(12).mean()

        df["occ_mom"] = df[occ_col].diff()       # month-over-month
        df["occ_yoy"] = df[occ_col].diff(12)     # year-over-year

        print(f"added: occupancy lags and rolling averages (from '{occ_col}')")

    # lag features for average room rate
    arr_col = None
    for col in df.columns:
        cl = col.lower()
        if "room_rate" in cl or cl == "arr" or ("avg" in cl and "rate" in cl):
            if df[col].dtype in ["float64", "int64"]:
                arr_col = col
                break

    if arr_col:
        df["arr_lag_1"] = df[arr_col].shift(1)
        df["arr_lag_12"] = df[arr_col].shift(12)
        df["arr_ma3"] = df[arr_col].rolling(3).mean()
        print(f"added: room rate lags (from '{arr_col}')")

    # visitor demand features
    if "total_visitors" in df.columns:
        df["vis_lag_1"] = df["total_visitors"].shift(1)
        df["vis_lag_3"] = df["total_visitors"].shift(3)
        df["vis_growth"] = df["total_visitors"].pct_change() * 100
        df["vis_ma3"] = df["total_visitors"].rolling(3).mean()
        df["vis_ma12"] = df["total_visitors"].rolling(12).mean()
        print("added: visitor lags and growth")

    # cyclical month encoding (better than raw month number for ML)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    # calculate RevPAR if it doesn't already exist
    if occ_col and arr_col:
        if not any("revpar" in c.lower() for c in df.columns):
            df["revpar_calc"] = df[occ_col] / 100 * df[arr_col]
            print("added: revpar_calc")

    nan_rows = df.isnull().any(axis=1).sum()
    print(f"\ntotal: {df.shape[1]} columns ({nan_rows} rows have NaN from lag features)")
    return df


if master is not None:
    master_featured = add_sg_features(master)
    print(master_featured.head().to_string())
else:
    master_featured = None
    print("no master dataset, skipping feature engineering")


# %%
# save processed data

if master_featured is not None:
    master_featured.to_csv("data/processed/part1_forecast_data.csv", index=False)
    print(f"part 1 (forecasting): {master_featured.shape[0]} rows, {master_featured.shape[1]} cols")
    print("saved to data/processed/part1_forecast_data.csv")
    
if reviews is not None:
    print(f"part 2 (sentiment): {reviews.shape[0]} rows, {reviews.shape[1]} cols")
    print("already at data/processed/reviews_clean.csv")


# %%
# build country/region profiles for part 3 (clustering)

if visitors is not None and "country" in visitors.columns:
    arr_col = "arrivals" if "arrivals" in visitors.columns else \
              [c for c in visitors.select_dtypes(include=[np.number]).columns
               if c not in ["year", "month_num"]][0]

    print(f"arrivals column: '{arr_col}'")
    print(f"unique entries: {visitors['country'].nunique()}")
    print(f"year range: {visitors['year'].min()} to {visitors['year'].max()}")

    # use the most recent 2 years of whatever we have (not hardcoded to 2022)
    latest_year = visitors["year"].max()
    cutoff_year = latest_year - 1
    recent = visitors[visitors["year"] >= cutoff_year].copy()
    all_vis = visitors.copy()

    print(f"using data from {cutoff_year} onwards for profiling")

    profiles = {}
    for region in visitors["country"].unique():
        if region == "Not Stated":
            continue

        c_all = all_vis[all_vis["country"] == region]
        c_rec = recent[recent["country"] == region]
        if len(c_rec) == 0:
            continue

        # annual volume
        annual_vol = c_rec[c_rec["year"] == latest_year][arr_col].sum()
        if annual_vol == 0:
            annual_vol = c_rec[arr_col].sum() / max(c_rec["year"].nunique(), 1)

        # growth
        yearly = c_rec.groupby("year")[arr_col].sum()
        if len(yearly) >= 2:
            growth = (yearly.iloc[-1] - yearly.iloc[0]) / max(yearly.iloc[0], 1) * 100
        else:
            growth = 0

        # seasonality
        monthly = c_rec.groupby("month_num")[arr_col].mean()
        seasonality = monthly.std() / max(monthly.mean(), 1) if len(monthly) > 1 else 0
        peak_month = monthly.idxmax() if len(monthly) > 0 else 6

        # volatility
        month_totals = c_rec.groupby(c_rec["date"].dt.to_period("M"))[arr_col].sum()
        volatility = month_totals.std() / max(month_totals.mean(), 1) if len(month_totals) > 1 else 0

        profiles[region] = {
            "annual_volume": annual_vol,
            "growth_rate": round(growth, 1),
            "seasonality_index": round(seasonality, 3),
            "peak_month": peak_month,
            "volatility": round(volatility, 3),
        }

    country_profiles = pd.DataFrame(profiles).T
    country_profiles.index.name = "country"

    # estimated average spending per visitor (SGD)
    known_spending = {
        "Asia": 1200, "Europe": 1800, "Oceania": 1500,
        "Americas": 2000, "Africa": 1000,
    }
    country_profiles["avg_spending"] = country_profiles.index.map(
        lambda x: known_spending.get(x, 1000)
    )

    country_profiles.to_csv("data/processed/part3_country_profiles.csv")

    print(f"\nprofiles: {country_profiles.shape[0]} regions, {country_profiles.shape[1]} features")
    print(country_profiles.to_string())

else:
    country_profiles = None
    print("visitors data not available, skipping profiles")
# %%
# save hotel tier data for part 3 as well

if hotel_tier is not None:
    hotel_tier.to_csv("data/processed/part3_hotel_tier.csv", index=False)
    print(f"hotel tier data: {hotel_tier.shape[0]} rows, saved to data/processed/part3_hotel_tier.csv")


# %%
