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

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer

plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# %%
# load reviews

df = pd.read_csv("data/processed/reviews_clean.csv")
print(f"loaded: {df.shape[0]} reviews, {df.shape[1]} columns")

for col in ["hotel_name", "review_text", "rating", "sentiment", "cleaned_text"]:
    status = "ok" if col in df.columns else "MISSING"
    print(f"  {col}: {status}")

df = df.dropna(subset=["cleaned_text"])
df = df[df["cleaned_text"].str.len() > 5].reset_index(drop=True)
print(f"after filtering: {len(df)} usable reviews")


# %%
# eda: rating distribution, sentiment split, top hotels

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

df["rating"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#2E86AB")
axes[0].set_title("Rating Distribution", fontweight="bold")
axes[0].set_xlabel("Star Rating")
axes[0].set_ylabel("Count")

df["sentiment"].value_counts().plot(kind="pie", ax=axes[1],
    autopct="%1.1f%%", colors=["#28A745", "#DC3545", "#FFC107"])
axes[1].set_title("Sentiment Split", fontweight="bold")
axes[1].set_ylabel("")

df["hotel_name"].value_counts().head(10).plot(kind="barh", ax=axes[2], color="#6C5CE7")
axes[2].set_title("Top 10 Hotels by Reviews", fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/sentiment_eda.png", dpi=150, bbox_inches="tight")
plt.show()


# %%
# word clouds for positive vs negative reviews

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, sent in enumerate(["positive", "negative"]):
    text = " ".join(df[df["sentiment"] == sent]["cleaned_text"].dropna())
    if len(text) < 10:
        axes[i].set_title(f"{sent.upper()} — not enough data")
        continue
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap="Greens" if sent == "positive" else "Reds",
                   max_words=80, contour_width=2).generate(text)
    axes[i].imshow(wc, interpolation="bilinear")
    axes[i].set_title(f"{sent.upper()} Reviews", fontsize=14, fontweight="bold")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("outputs/wordclouds.png", dpi=150, bbox_inches="tight")
plt.show()


# %%
# tfidf vectorization

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

X = tfidf.fit_transform(df["cleaned_text"])
y = df["sentiment_binary"] if "sentiment_binary" in df.columns else (df["rating"] >= 4).astype(int)

print(f"tfidf matrix: {X.shape[0]} reviews x {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"train: {X_train.shape[0]} | test: {X_test.shape[0]}")
print(f"positive rate: {y.mean():.1%}")


# %%
# train classifiers

models = {
    "Naive Bayes": MultinomialNB(alpha=1.0),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "SVM (Linear)": LinearSVC(max_iter=2000, C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    results[name] = {"accuracy": acc, "f1": f1, "predictions": pred, "model": model}
    print(f"  {name}: accuracy={acc:.4f}, f1={f1:.4f}")


# %%
# model comparison

comp = pd.DataFrame({
    n: {"Accuracy": r["accuracy"], "F1 Score": r["f1"]}
    for n, r in results.items()
}).T.sort_values("F1 Score", ascending=False)

print(comp.round(4).to_string())

best_name = comp["F1 Score"].idxmax()
print(f"\nbest model: {best_name}")

fig, ax = plt.subplots(figsize=(10, 5))
comp.plot(kind="bar", ax=ax, color=["#2E86AB", "#E84855"])
ax.set_title("Sentiment Model Comparison", fontweight="bold")
ax.set_ylim(0.5, 1.0)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/model_comparison_sentiment.png", dpi=150, bbox_inches="tight")
plt.show()


# %%
# confusion matrix for best model

best_pred = results[best_name]["predictions"]
cm = confusion_matrix(y_test, best_pred)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nclassification report ({best_name}):")
print(classification_report(y_test, best_pred, target_names=["Negative", "Positive"]))


# %%
# most predictive words from logistic regression coefficients

if "Logistic Regression" in results:
    lr = results["Logistic Regression"]["model"]
    feature_names = tfidf.get_feature_names_out()
    coef = lr.coef_[0]

    top_pos_idx = coef.argsort()[-15:][::-1]
    top_neg_idx = coef.argsort()[:15]

    print("top 15 positive words:")
    for idx in top_pos_idx:
        print(f"  {feature_names[idx]:25s}  {coef[idx]:+.3f}")

    print("\ntop 15 negative words:")
    for idx in top_neg_idx:
        print(f"  {feature_names[idx]:25s}  {coef[idx]:+.3f}")

    word_importance = pd.DataFrame({
        "word": [feature_names[i] for i in list(top_pos_idx) + list(top_neg_idx)],
        "coefficient": [coef[i] for i in list(top_pos_idx) + list(top_neg_idx)],
        "sentiment": ["positive"] * 15 + ["negative"] * 15,
    })
    word_importance.to_csv("data/processed/sentiment_word_importance.csv", index=False)


# %%
# lda topic modeling 
# discover service themes in the reviews

n_topics = 6

texts = df["cleaned_text"].dropna().tolist()
tokenized = [text.split() for text in texts]

dictionary = corpora.Dictionary(tokenized)
dictionary.filter_extremes(no_below=5, no_above=0.8)
corpus = [dictionary.doc2bow(text) for text in tokenized]

lda = LdaModel(
    corpus=corpus, id2word=dictionary, num_topics=n_topics,
    random_state=42, passes=15, alpha="auto", per_word_topics=True
)

# auto-label each topic based on its top words
topic_labels = {}
for tid in range(n_topics):
    words = [w for w, _ in lda.show_topic(tid, topn=10)]

    label = f"Topic {tid + 1}"
    if any(w in words for w in ["clean", "dirty", "bathroom", "towel", "housekeeping", "shower"]):
        label = "Cleanliness"
    elif any(w in words for w in ["location", "mrt", "walk", "near", "convenient", "orchard", "station", "mall"]):
        label = "Location & Access"
    elif any(w in words for w in ["staff", "friendly", "helpful", "service", "check", "front", "desk", "reception"]):
        label = "Service Quality"
    elif any(w in words for w in ["breakfast", "food", "restaurant", "buffet", "bar", "dining", "coffee"]):
        label = "F&B / Dining"
    elif any(w in words for w in ["bed", "view", "spacious", "small", "comfortable", "quiet", "noise", "pillow"]):
        label = "Room Quality"
    elif any(w in words for w in ["price", "value", "expensive", "worth", "money", "cheap", "pay", "cost"]):
        label = "Value for Money"
    elif any(w in words for w in ["pool", "gym", "spa", "facility", "lounge", "club"]):
        label = "Facilities & Amenities"

    topic_labels[tid] = label
    print(f"\n  {label}")
    print(f"  {', '.join(words)}")


# %%
# score each hotel on each topic dimension

topic_scores_list = []
for doc_bow in corpus:
    dist = lda.get_document_topics(doc_bow, minimum_probability=0)
    scores = {topic_labels.get(t, f"Topic {t}"): prob for t, prob in dist}
    topic_scores_list.append(scores)

topic_df = pd.DataFrame(topic_scores_list)

scored = pd.concat([
    df[["hotel_name", "rating", "sentiment"]].iloc[:len(topic_df)].reset_index(drop=True),
    topic_df.reset_index(drop=True)
], axis=1)

# average per hotel
hotel_topic_scores = scored.groupby("hotel_name").agg({
    **{col: "mean" for col in topic_df.columns},
    "rating": ["mean", "count"]
}).round(3)
hotel_topic_scores.columns = ["_".join(c).strip("_") for c in hotel_topic_scores.columns]
hotel_topic_scores = hotel_topic_scores.rename(columns={"rating_mean": "avg_rating", "rating_count": "num_reviews"})
hotel_topic_scores = hotel_topic_scores.sort_values("avg_rating", ascending=False)

print(f"\nhotel scores ({hotel_topic_scores.shape[0]} hotels):")
print(hotel_topic_scores.head(10).to_string())
hotel_topic_scores.to_csv("data/processed/hotel_topic_scores.csv")

# %%
# radar chart comparing top 4 hotels across service dimensions
# plotly cant display inline charts in streamlit, so it is saved as html and open in browser to view

top4 = hotel_topic_scores.head(4).index.tolist()
categories = list(set(topic_labels.values()))
colors = ["#2E86AB", "#E84855", "#28A745", "#FFC107"]

fig = go.Figure()
for i, hotel in enumerate(top4):
    vals = []
    for cat in categories:
        matching = [c for c in hotel_topic_scores.columns
                    if cat.lower().replace(" ", "_").replace("/", "_").replace("&", "")
                    in c.lower().replace(" ", "_").replace("/", "_").replace("&", "")]
        vals.append(float(hotel_topic_scores.loc[hotel, matching[0]]) if matching else 0)
    vals.append(vals[0])
    cats = categories + [categories[0]]
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats, fill="toself", name=hotel,
        line=dict(color=colors[i % 4])
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    height=550,
    title="Top Hotels — Service Dimension Comparison"
)
fig.write_html("outputs/hotel_radar.html")
print("saved: outputs/hotel_radar.html (open in browser to view)")

# %%
# pain point analysis — top words in positive vs negative reviews per hotel

sia = SentimentIntensityAnalyzer()
top_hotels = df["hotel_name"].value_counts().head(8).index

for hotel in top_hotels:
    h = df[df["hotel_name"] == hotel]
    avg = h["rating"].mean()
    pos_pct = (h["sentiment"] == "positive").mean() * 100

    pos_words = " ".join(h[h["sentiment"] == "positive"]["cleaned_text"].dropna()).split()
    neg_words = " ".join(h[h["sentiment"] == "negative"]["cleaned_text"].dropna()).split()

    pos_top = pd.Series(pos_words).value_counts().head(5).index.tolist() if pos_words else ["N/A"]
    neg_top = pd.Series(neg_words).value_counts().head(5).index.tolist() if neg_words else ["N/A"]

    print(f"\n{hotel}")
    print(f"  {avg:.1f}/5 | {len(h)} reviews | {pos_pct:.0f}% positive")
    print(f"  strengths:   {', '.join(pos_top)}")
    print(f"  complaints:  {', '.join(neg_top)}")


# %%
# save models and results

best_model = results[best_name]["model"]
joblib.dump(best_model, "models/sentiment_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
comp.to_csv("data/processed/model_comparison_sentiment.csv")

print("saved:")
print("  models/sentiment_model.pkl")
print("  models/tfidf_vectorizer.pkl")
print("  data/processed/hotel_topic_scores.csv")
print("  data/processed/model_comparison_sentiment.csv")
print("  data/processed/sentiment_word_importance.csv")