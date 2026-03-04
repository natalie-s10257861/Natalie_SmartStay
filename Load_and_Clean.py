# %%
import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

pd.set_option("display.max_columns", 50)


# %%
# generic cleaner for data.gov.sg CSVs
# handles messy column names, string-formatted numbers, and auto-detects date columns

def auto_clean(df, name="dataset"):
    df = df.copy()
    print(f"\nCleaning: {name}")
    print(f"raw: {df.shape[0]} rows, {df.shape[1]} cols")

    # normalise column names
    df.columns = (df.columns.str.lower().str.strip()
                  .str.replace(" ", "_").str.replace("-", "_"))
    print(f"columns: {df.columns.tolist()}")

    # try to find and parse a date column
    date_col = None
    date_hints = ["month", "date", "year_month", "quarter", "period"]
    for col in df.columns:
        if not any(hint in col for hint in date_hints):
            continue
        sample = str(df[col].iloc[0])
        for fmt in [None, "%Y-%m", "%Y %b", "%b %Y", "%Y"]:
            try:
                pd.to_datetime(sample, format=fmt)
                date_col = col
                break
            except (ValueError, TypeError):
                continue
        if date_col:
            break

    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month_num"] = df["date"].dt.month
        valid = df["date"].notna().sum()
        print(f"date column: '{date_col}' — {valid} valid dates")
        print(f"range: {df['date'].min()} to {df['date'].max()}")
    else:
        print(f"no date column found")

    # convert string numbers to actual numbers
    converted = []
    for col in df.columns:
        if col in ("date", "year", "month_num", date_col):
            continue
        if df[col].dtype != "object":
            continue
        stripped = df[col].astype(str).str.replace(",", "").str.replace("%", "").str.strip()
        as_numeric = pd.to_numeric(stripped, errors="coerce")
        if as_numeric.notna().mean() > 0.5:
            df[col] = as_numeric
            converted.append(col)
    if converted:
        print(f"converted to numeric: {converted}")

    # fill gaps
    num_cols = df.select_dtypes(include=[np.number]).columns
    missing_before = df[num_cols].isnull().sum()
    has_missing = missing_before[missing_before > 0]
    if len(has_missing) > 0:
        print(f"missing values: {has_missing.to_dict()}")
        df[num_cols] = df[num_cols].fillna(method="ffill").fillna(method="bfill")
        print("filled with forward/backward fill")
    else:
        print("no missing values")

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"removed {dupes} duplicates")

    print(f"clean: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


# %%
# review-specific cleaning
# finds text/rating/hotel columns automatically,
# then runs NLP pipeline (lowercase, strip html, lemmatize, remove stopwords)

def clean_reviews(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    print(f"\nCleaning: TripAdvisor Reviews")
    print(f"raw: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"columns: {df.columns.tolist()}")

    # figure out which column is which
    # prioritise detail_comment/comment over count_review for the text column
    hotel_col = text_col = rating_col = None

    for col in df.columns:
        if any(w in col for w in ["hotel", "property", "business"]) and not hotel_col:
            if df[col].dtype == "object":
                hotel_col = col

    # check for the actual review text columns first
    text_priority = ["detail_comment", "comment", "review_text", "review", "text", "content", "body"]
    for candidate in text_priority:
        if candidate in df.columns and df[candidate].dtype == "object":
            text_col = candidate
            break

    for col in df.columns:
        if any(w in col for w in ["rating", "score", "star", "grade"]) and not rating_col:
            # prefer the per-review rating column over mean_rating
            if col == "rating":
                rating_col = col
                break
            elif not rating_col:
                rating_col = col

    # if per-review 'rating' column exists, prefer it over 'mean_rating'
    if "rating" in df.columns:
        rating_col = "rating"

    print(f"detected — hotel: '{hotel_col}', text: '{text_col}', rating: '{rating_col}'")

    if hotel_col:
        df["hotel_name"] = df[hotel_col].astype(str).str.strip()
    if text_col:
        df["review_text"] = df[text_col].astype(str)
    if rating_col:
        df["rating"] = pd.to_numeric(df[rating_col], errors="coerce")

    # drop anything that isn't a real review
    if "review_text" in df.columns:
        before = len(df)
        df = df.dropna(subset=["review_text"])
        df = df[df["review_text"].str.len() > 10]
        dropped = before - len(df)
        if dropped > 0:
            print(f"dropped {dropped} empty/short reviews")

    # label sentiment based on star rating
    if "rating" in df.columns:
        df["sentiment"] = df["rating"].apply(
            lambda x: "positive" if x >= 4 else ("negative" if x <= 2 else "neutral")
        )
        df["sentiment_binary"] = (df["rating"] >= 4).astype(int)
        print(f"rating distribution: {df['rating'].value_counts().sort_index().to_dict()}")
        print(f"sentiment split: {df['sentiment'].value_counts().to_dict()}")

    # text preprocessing
    if "review_text" in df.columns:
        print("preprocessing review text (this takes a minute)...")
        stop_words = set(stopwords.words("english"))
        stop_words.update([
            "hotel", "room", "stay", "stayed", "night", "singapore",
            "would", "also", "one", "get", "got", "us", "go", "went",
            "really", "could", "even", "much", "well", "back", "like"
        ])
        lemmatizer = WordNetLemmatizer()

        def preprocess(text):
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r"http\S+|www\S+", "", text)
            text = re.sub(r"<.*?>", "", text)
            text = re.sub(r"[^a-zA-Z\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(t) for t in tokens
                      if t not in stop_words and len(t) > 2]
            return " ".join(tokens)

        df["cleaned_text"] = df["review_text"].apply(preprocess)
        print(f"example original:  {df['review_text'].iloc[0][:120]}...")
        print(f"example cleaned:   {df['cleaned_text'].iloc[0][:120]}...")

    if "hotel_name" in df.columns:
        print(f"\ntop 10 hotels by review count:")
        print(df["hotel_name"].value_counts().head(10).to_string())

    print(f"clean: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


# %%
# Hotel Monthly Statistics

hotel_monthly = None
try:
    hotel_monthly = auto_clean(pd.read_csv("data/raw/hotel_monthly.csv"), "Hotel Monthly")
    print(hotel_monthly.head().to_string())
    print(f"\ndata types:\n{hotel_monthly.dtypes}")
except FileNotFoundError:
    print("hotel_monthly.csv not found")


# %%
# Hotel Statistics by Tier

hotel_tier = None
try:
    hotel_tier = auto_clean(pd.read_csv("data/raw/hotel_by_tier.csv"), "Hotel by Tier")

    tier_col = None
    for col in hotel_tier.columns:
        if any(w in col for w in ["tier", "category", "type", "class"]):
            tier_col = col
            break
    if tier_col:
        print(f"\ntier column: '{tier_col}'")
        print(f"unique tiers: {hotel_tier[tier_col].unique().tolist()}")

    print(hotel_tier.head().to_string())
    print(f"\ndata types:\n{hotel_tier.dtypes}")
except FileNotFoundError:
    print("hotel_by_tier.csv not found")


# %%
# International Visitor Arrivals
# the singstat file is wide format (countries as rows, months as columns like "2025may")
# we detect that and reshape to long format: date | country | arrivals

visitors = None
try:
    raw_visitors = pd.read_csv("data/raw/visitor_arrivals.csv")
    raw_visitors.columns = (raw_visitors.columns.str.lower().str.strip()
                            .str.replace(" ", "_").str.replace("-", "_"))

    if len(raw_visitors.columns) > 20:
        # wide format from singstat — reshape
        print(f"\nCleaning: Visitor Arrivals")
        print(f"wide format detected: {len(raw_visitors.columns)} columns, {len(raw_visitors)} rows")

        id_col = raw_visitors.columns[0]
        month_cols = [c for c in raw_visitors.columns if c != id_col]

        visitors = raw_visitors.melt(
            id_vars=[id_col],
            value_vars=month_cols,
            var_name="period",
            value_name="arrivals"
        )
        visitors = visitors.rename(columns={id_col: "country"})

        # parse period (format: "2025may", "2024dec")
        visitors["date"] = pd.to_datetime(visitors["period"], format="%Y%b", errors="coerce")
        visitors["arrivals"] = pd.to_numeric(visitors["arrivals"], errors="coerce")
        visitors = visitors.dropna(subset=["date", "arrivals"])
        visitors["year"] = visitors["date"].dt.year
        visitors["month_num"] = visitors["date"].dt.month
        visitors["country"] = visitors["country"].str.strip().str.title()

        # drop totals and aggregate rows
        drop_words = ["total", "grand", "overall", "international visitor"]
        visitors = visitors[~visitors["country"].str.lower().str.contains("|".join(drop_words), na=False)]

        visitors = visitors.drop(columns=["period"])
        visitors = visitors.sort_values(["country", "date"]).reset_index(drop=True)

        print(f"reshaped: {len(visitors)} rows")
        print(f"countries: {visitors['country'].nunique()}")
        print(f"range: {visitors['date'].min()} to {visitors['date'].max()}")
        print(visitors["country"].value_counts().head(15).to_string())
        print(visitors.head(10).to_string())

    else:
        # normal long format — use auto_clean
        visitors = auto_clean(raw_visitors, "Visitor Arrivals")

        country_col = None
        for col in visitors.columns:
            if any(w in col for w in ["country", "nationality", "region", "market", "place"]):
                if visitors[col].dtype == "object":
                    country_col = col
                    break
        if country_col:
            visitors["country"] = visitors[country_col].str.strip().str.title()
            print(f"\ncountry column: '{country_col}'")
            print(visitors["country"].value_counts().head(10).to_string())

        arrivals_col = None
        for col in visitors.columns:
            if any(w in col for w in ["arrival", "visitor", "number", "value", "count"]):
                if visitors[col].dtype in ["int64", "float64"]:
                    arrivals_col = col
                    break
        if arrivals_col and arrivals_col != "arrivals":
            visitors["arrivals"] = visitors[arrivals_col]

        print(visitors.head().to_string())

except FileNotFoundError:
    print("visitor_arrivals.csv not found")


# %%
# Annual Hotel Statistics

hotel_annual = None
try:
    hotel_annual = auto_clean(pd.read_csv("data/raw/hotel_annual.csv"), "Hotel Annual")
    print(hotel_annual.head().to_string())
    print(f"\ndata types:\n{hotel_annual.dtypes}")
except FileNotFoundError:
    print("hotel_annual.csv not found")


# %%
# Tourism Receipts (Annual)

receipts = None
try:
    receipts = auto_clean(pd.read_csv("data/raw/tourism_receipts.csv"), "Tourism Receipts")
    print(receipts.head().to_string())
    print(f"\ndata types:\n{receipts.dtypes}")
except FileNotFoundError:
    print("tourism_receipts.csv not found")


# %%
# Tourism Receipts (Quarterly)

receipts_qtr = None
try:
    receipts_qtr = auto_clean(pd.read_csv("data/raw/tourism_receipts_qtr.csv"), "Tourism Receipts (Quarterly)")
    print(receipts_qtr.head().to_string())
    print(f"\ndata types:\n{receipts_qtr.dtypes}")
except FileNotFoundError:
    print("tourism_receipts_qtr.csv not found (optional)")


# %%
# TripAdvisor SG Hotel Reviews

reviews = None
possible_names = [
    "tripadvisor_sg.csv",
    "tripadvisor_sg_reviews.csv",
    "Singapore_TripAdvisor_Reviews.csv",
    "tripadvisor-singapore-reviews.csv",
]

for fname in possible_names:
    path = f"data/raw/{fname}"
    if os.path.exists(path):
        reviews = clean_reviews(path)
        break

# fallback — looks for anything with 'trip' or 'review' in the filename
if reviews is None and os.path.exists("data/raw"):
    for f in os.listdir("data/raw"):
        if ("trip" in f.lower() or "review" in f.lower()) and f.endswith(".csv"):
            reviews = clean_reviews(f"data/raw/{f}")
            break

if reviews is not None:
    print(reviews.head().to_string())
    print(f"\ndata types:\n{reviews.dtypes}")
else:
    print("No TripAdvisor file found in data/raw/")


# %%
# save everything to data/processed/

output_map = {
    "hotel_monthly_clean.csv":        hotel_monthly,
    "hotel_tier_clean.csv":           hotel_tier,
    "visitors_clean.csv":             visitors,
    "hotel_annual_clean.csv":         hotel_annual,
    "tourism_receipts_clean.csv":     receipts,
    "tourism_receipts_qtr_clean.csv": receipts_qtr,
    "reviews_clean.csv":              reviews,
}

print("\nSaving cleaned datasets:\n")
count = 0
for filename, df in output_map.items():
    if df is not None:
        path = f"data/processed/{filename}"
        df.to_csv(path, index=False)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {filename} ({size_kb:.0f} KB)")
        count += 1
    else:
        print(f"  {filename} — skipped (not loaded)")

print(f"\n{count}/{len(output_map)} datasets saved to data/processed/")
# %%
