# %%
# install everything
import subprocess
import sys

packages = [
    "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
    "plotly", "streamlit", "prophet", "statsmodels", "xgboost",
    "nltk", "textblob", "gensim", "wordcloud", "joblib", "openpyxl"
]

for pkg in packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("vader_lexicon", quiet=True)

print("All packages ready.")

# %%
# set up folder structure
import os

for folder in ["data/raw", "data/processed", "models", "outputs"]:
    os.makedirs(folder, exist_ok=True)

# %%
# check which datasets are present in data/raw/
# if anything is missing, the download link is printed so you know where to get it
#
# ## download these files and place them in data/raw/ before running Step 1
# ## rename each file to match the names below

REQUIRED_DATASETS = {
    "hotel_monthly.csv":     "https://data.gov.sg/datasets/d_1db0bd2ffd95ac09c66db600e60d3400/view",
    "hotel_by_tier.csv":     "https://data.gov.sg/datasets/d_8da6783d5f7628ae6ada1c240015b7d7/view",
    "visitor_arrivals.csv":  "https://data.gov.sg — search 'International Visitor Arrivals by Country of Nationality'",
    "hotel_annual.csv":      "https://data.gov.sg/datasets/d_a728577abbe4ff3f3409b9129be28a53/view",
    "tourism_receipts.csv":  "https://data.gov.sg/datasets/d_e285a651ec353416054195528ca988a9/view",
    "tripadvisor_sg.csv":    "https://www.kaggle.com/datasets/chrisgharris/tripadvisor-singapore-reviews",
    "tourism_receipts_qtr.csv": "https://data.gov.sg/datasets/d_248d4c6574b5ac87cd31851ed3f697d6/view"
}

missing = []

print("Checking data/raw/...\n")

for filename, url in REQUIRED_DATASETS.items():
    path = os.path.join("data/raw", filename)
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  found: {filename} ({size_kb:.0f} KB)")
    else:
        print(f"  Missing: {filename}")
        print(f"           {url}")
        missing.append(filename)

if missing:
    print(f"\n{len(missing)} required file(s) still missing.")



# %%
