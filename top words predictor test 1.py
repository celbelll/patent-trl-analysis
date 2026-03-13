#===========================================
# Ce code permet d'entraîner un SVM sur un corpus de brevets avec des labels TRL, et d'extraire les n-grammes les plus importants pour chaque classe TRL.
# Il génère un graphique "figure_feature_importance_single_corpus.png" montrant les top n-grammes pour chaque TRL, et un fichier CSV "results_topwords_single_corpus.csv" listant les n-grammes et leurs poids pour chaque TRL.
#===========================================


# ==========================================
# topword_svm_single_corpus.py
# - SVM sur corpus unique
# - Top n-grammes par TRL
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

DATA_PATH = "mon_nouveau_corpus_avec_trl_avec_nettoyage.csv"

# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------

def plot_top_words(model, features, out_path, trl_classes=None, n_top=10):

    if trl_classes is None:
        trl_classes = list(model.classes_)

    fig, axes = plt.subplots(len(trl_classes), 1, figsize=(10, 5 * len(trl_classes)))

    if len(trl_classes) == 1:
        axes = [axes]

    for i, trl in enumerate(trl_classes):

        if trl not in model.classes_:
            continue

        class_idx = np.where(model.classes_ == trl)[0][0]
        coefs = model.coef_[class_idx]

        top_indices = np.argsort(coefs)[-n_top:]
        top_keywords = features[top_indices]
        top_coeffs = coefs[top_indices]

        sns.barplot(
            x=top_coeffs,
            y=top_keywords,
            ax=axes[i],
            orient='h'
        )

        axes[i].set_title(f"Top Predictors for TRL {trl}", fontsize=14)
        axes[i].set_xlabel("SVM Weight")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✅ Graphique sauvegardé : {out_path}")


def build_topwords_table(model, feature_names, top_k=50):
    rows = []
    for idx, trl in enumerate(model.classes_):
        coefs = model.coef_[idx]
        top_idx = np.argsort(coefs)[-top_k:]
        for j in top_idx:
            rows.append({
                "trl": int(trl),
                "ngram": feature_names[j],
                "weight": coefs[j]
            })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# 1️⃣ Chargement des données
# -------------------------------------------------------------------

print("\n=== Chargement du corpus ===")

df = pd.read_csv(DATA_PATH)

# Renommer colonne si nécessaire
if "trl_pred" in df.columns:
    df = df.rename(columns={"trl_pred": "trl"})

df = df.dropna(subset=["text_clean", "trl"])
df["trl"] = df["trl"].astype(int)

print(f"Corpus chargé : {len(df)} brevets")
print(f"Classes TRL présentes : {sorted(df['trl'].unique())}")

# -------------------------------------------------------------------
# 2️⃣ Vectorisation TF-IDF
# -------------------------------------------------------------------

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True
)

X = tfidf.fit_transform(df["text_clean"])
y = df["trl"]

feature_names = np.array(tfidf.get_feature_names_out())

# -------------------------------------------------------------------
# 3️⃣ Entraînement SVM
# -------------------------------------------------------------------

svm_model = LinearSVC(
    class_weight="balanced",
    C=1.0,
    dual="auto",
    random_state=42
)

svm_model.fit(X, y)

print("✅ SVM entraîné sur le corpus complet")

# -------------------------------------------------------------------
# 4️⃣ Top n-grammes par TRL
# -------------------------------------------------------------------

plot_top_words(
    svm_model,
    feature_names,
    out_path="figure_feature_importance_single_corpus.png",
    trl_classes=sorted(df["trl"].unique()),
    n_top=10
)

df_top = build_topwords_table(svm_model, feature_names, top_k=50)
df_top.to_csv("results_topwords_single_corpus.csv", index=False)

print("✅ Fichier sauvegardé : results_topwords_single_corpus.csv")

print("\n✅ Fin du script")
