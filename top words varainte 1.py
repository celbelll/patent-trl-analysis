#=======================
# Ce code permet d'extraire les mots et bigrammes les plus fréquents dans le corpus de textes concaténés (titres, abstracts et claims) des brevets.
# Il utilise CountVectorizer pour vectoriser le texte et calcule la fréquence globale de chaque n-gramme.
# Le résultat est sauvegardé dans un fichier CSV "top_words_frequency.csv" avec les colonnes "ngram" et "frequency".
#=======================


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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

# 3. Vectorisation
vectorizer = CountVectorizer(
    ngram_range=(1, 2),   # (1,1) = mots seuls, (1,2) = mots + bigrammes
    max_features=20000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["text"])
features = np.array(vectorizer.get_feature_names_out())

# 4. Fréquence globale
freqs = np.asarray(X.sum(axis=0)).flatten()

top_idx = np.argsort(freqs)[-TOP_K:]
top_words = features[top_idx]
top_freqs = freqs[top_idx]

df_top = pd.DataFrame({
    "ngram": top_words,
    "frequency": top_freqs
}).sort_values("frequency", ascending=False)

df_top.to_csv("top_words_frequency.csv", index=False)
print("✅ top_words_frequency.csv généré")
