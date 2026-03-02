import pandas as pd
import joblib
import re
import unicodedata
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

nltk.download('stopwords')

# =============================
# 1️⃣ Stopwords
# =============================

PATENT_STOPWORDS = {
    "said", "means", "first", "second", "third", "fourth", 
    "according", "thereof", "therein", "wherein", "whereof",
    "whereby", "thereby", "thereafter", "therefrom", 
    "claim", "claims", "embodiment", "embodiments",
    "figure", "figures", "fig", "further",
    "herein", "hereof", "hereto", "hereby",
    "may", "can", "one", "two", "plurality", "least", "end", 
    "portion", "part", "another", "example",
    "configured", "arranged", "adapted", "respective",
    "thereon", "thereupon", "therewith",
    "generally", "optionally", "typically", "preferably",
    "alternatively", "otherwise", "type", "base", "based",
    "including", "includes", "included",
    "provide", "provided", "providing",
    "comprise", "comprising", "comprises",
    "consist", "consisting",
    "also", "even", "yet", "still", "thus", "therefore",
    "unit", "module", "section"
}

STOPWORDS = set(stopwords.words("english")) | PATENT_STOPWORDS


# =============================
# 2️⃣ Fonctions de nettoyage
# =============================

def normalize_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t = text.lower()
    t = normalize_accents(t)

    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)

    t = re.sub(r"[^a-z0-9\s]", " ", t)

    t = re.sub(r"\s+", " ", t).strip()

    t = " ".join(word for word in t.split() if word not in STOPWORDS)

    return t


# =============================
# 3️⃣ Chargement modèle
# =============================

model = joblib.load("C:/Users/CéliaBelaziz/Downloads/frugal_trl_model.pkl")

PATENTS_PATH = "batch_patents.csv"
df = pd.read_csv(PATENTS_PATH, dtype=str)

# =============================
# 4️⃣ Construction colonne text
# =============================

for col in ("title", "abstract", "claims"):
    if col not in df.columns:
        df[col] = ""

df[["title", "abstract", "claims"]] = df[["title", "abstract", "claims"]].fillna("")

df["text"] = (
    df["title"].str.strip()
    + " "
    + df["abstract"].str.strip()
    + " "
    + df["claims"].str.strip()
)

df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

# =============================
# 5️⃣ Nettoyage
# =============================

df["text_clean"] = df["text"].apply(clean_text)

df_nouveau = df[df["text_clean"].str.strip() != ""]

# =============================
# 6️⃣ Prédiction
# =============================

df_nouveau["trl_pred"] = model.predict(df_nouveau["text_clean"])

print(df_nouveau[["text_clean", "trl_pred"]].head())

df_nouveau.to_csv("mon_nouveau_corpus_avec_trl_avec_nettoyage.csv", index=False)


# ==============================
# 📊 VISUALISATION DES TRL
# ==============================

total_patents = len(df_nouveau)
print(f"\nNombre total de brevets dans le corpus : {total_patents}")

trl_counts = df_nouveau["trl_pred"].value_counts().sort_index()

print("\nRépartition des TRL :")
print(trl_counts)

# Graphique 1 : Diagramme en barres
plt.figure()
plt.bar(trl_counts.index.astype(str), trl_counts.values)
plt.title("Répartition des niveaux TRL")
plt.xlabel("TRL")
plt.ylabel("Nombre de brevets")
plt.xticks(rotation=0)
plt.show()

# Graphique 2 : Diagramme circulaire
plt.figure()
plt.pie(trl_counts.values, labels=trl_counts.index.astype(str), autopct="%1.1f%%")
plt.title("Proportion des niveaux TRL")
plt.show()


# ==============================
# 🔍 ANALYSE DES FEATURES (LinearSVC)
# ==============================

feature_names = model.named_steps["tfidf"].get_feature_names_out()
coef = model.named_steps["clf"].coef_  # shape : (n_classes, n_features)
classes = model.classes_               # ex: [1, 2, 3, ..., 9]

TOP_N = 20

# ==============================
# 📁 CSV : top mots par TRL
# ==============================

rows = []
for i, trl_class in enumerate(classes):
    top_indices = np.argsort(coef[i])[::-1][:TOP_N]
    for rank, idx in enumerate(top_indices, start=1):
        rows.append({
            "trl": trl_class,
            "rank": rank,
            "word": feature_names[idx],
            "weight": coef[i][idx]
        })

df_features = pd.DataFrame(rows)
df_features.to_csv("top_features_par_trl.csv", index=False)
print("✓ top_features_par_trl.csv généré")

# ==============================
# 📊 Graphique : top features par TRL
# ==============================

n_classes = len(classes)
fig, axes = plt.subplots(n_classes, 1, figsize=(12, 5 * n_classes))

if n_classes == 1:
    axes = [axes]

for i, trl_class in enumerate(classes):
    df_trl = df_features[df_features["trl"] == trl_class].sort_values("weight", ascending=True)
    
    axes[i].barh(df_trl["word"], df_trl["weight"], color="steelblue", edgecolor="black")
    axes[i].set_title(f"Top {TOP_N} mots — TRL {trl_class}", fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Coefficient SVM", fontsize=10)
    axes[i].axvline(x=0, color="red", linestyle="--", linewidth=0.8)
    axes[i].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("top_features_par_trl.png", dpi=300, bbox_inches="tight")
plt.show()
print("✓ top_features_par_trl.png généré")[Errno 13] Permission denied: 'mon_nouveau_corpus_avec_trl_avec_nettoyage.csv'