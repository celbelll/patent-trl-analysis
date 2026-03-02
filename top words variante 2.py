import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_PATH = "batch_patents.csv"
TOP_K = 50

# 1. Charger les données
df = pd.read_csv(INPUT_PATH)

# 2. Construire le texte
df["text"] = (
    df["title"].fillna("") + " " +
    df["abstract"].fillna("") + " " +
    df["claims"].fillna("")
)

# 3. TF-IDF
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30000,
    stop_words="english",
    sublinear_tf=True
)

X = tfidf.fit_transform(df["text"])
features = np.array(tfidf.get_feature_names_out())

# 4. Score TF-IDF moyen sur tout le corpus
scores = np.asarray(X.mean(axis=0)).flatten()

top_idx = np.argsort(scores)[-TOP_K:]
top_words = features[top_idx]
top_scores = scores[top_idx]

df_top = pd.DataFrame({
    "ngram": top_words,
    "tfidf_score": top_scores
}).sort_values("tfidf_score", ascending=False)

df_top.to_csv("top_words_tfidf.csv", index=False)
print("✅ top_words_tfidf.csv généré")
