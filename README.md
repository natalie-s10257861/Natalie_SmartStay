# Project Overview
This is a self initiated data analytics and machine learning project I've undertaken that focuses on xx. It combines hotel revenue forecasting, guest sentiment analysis and tourist market segmentation to provide actionable intelligence for Singapore's Hospitality industry.
This project is built entirely built using Singapore-sourced data from data gov and real TripAdvisor hotel reviews from Kaggle, and addresses key industry problems through simple data analytics and machine learning

# Industry Problems
Key challenges faced by Singapore's Hospitality Industry that I've identified:
1. Demand volatility: fluctuation of occupancy can be seen drastically due to events (e.g F1, CNY), seasonality, and recently, the COVID-19 pandemic.
2. Service quality blind spots: hotels lack data-driven insight into guest pain points across specific service dimensions
3. Untargeted marketing: tourist source markets have vastly different travel behaviours, yet often receive the same marketing approach

# Composition of my project
1. Hotel Revenue & Occupancy Forecasting
2. Guest Sentiment Analysis & Service Intelligence
3. Tourist Segment Clustering & Targeting
4. Dashboards

# How to run on vscode
1. Clone and Setup
- git clone https://github.com/YOUR_USERNAME/smartstay-sg.git
- cd smartstay-sg
- pip install -r requirements.txt

2. Download all the datasets and rename them accordingly
- "hotel_monthly.csv": "https://data.gov.sg/datasets/d_1db0bd2ffd95ac09c66db600e60d3400/view",
- "hotel_by_tier.csv":     "https://data.gov.sg/datasets/d_8da6783d5f7628ae6ada1c240015b7d7/view",
- "visitor_arrivals.csv":  "https://data.gov.sg/datasets?query=International+Visitor+Arrivals+by+Country+&resultId=1622",
- "hotel_annual.csv":      "https://data.gov.sg/datasets/d_a728577abbe4ff3f3409b9129be28a53/view",
- "tourism_receipts.csv":  "https://data.gov.sg/datasets/d_e285a651ec353416054195528ca988a9/view",
- "tripadvisor_sg.csv":
  "https://www.kaggle.com/datasets/chrisgharris/tripadvisor-singapore-reviews",
- "tourism_receipts_qtr.csv": "https://data.gov.sg/datasets/d_248d4c6574b5ac87cd31851ed3f697d6/view"

3. Run Notebooks in order
- Setup_and_Download.py — installs dependencies and downloads the raw data
- Load_and_Clean.py — loads and cleans the raw data
- Merge_and_Features.py — merges datasets and engineers features (creates the country profiles, lag variables)
- Clustering.py — runs K-Means segmentation on the tourist market data
- Forecasting.py — trains and evaluates the Ridge/XGBoost/ arrival forecast models
- Sentiment.py — runs TF-IDF vectorisation, trains classifiers, LDA topic modelling, and pain point analysis
- Launch_Dashboard.py — launches the final interactive dashboard 
