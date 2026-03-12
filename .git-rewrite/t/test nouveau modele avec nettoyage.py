import pandas as pd
import joblib
import re
import unicodedata
import nltk
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

    # Séparer lettres/chiffres
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)

    # Supprimer caractères non alphanumériques
    t = re.sub(r"[^a-z0-9\s]", " ", t)

    # Normaliser espaces
    t = re.sub(r"\s+", " ", t).strip()

    # Supprimer stopwords
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
# 5️⃣ 🔥 Nettoyage appliqué ici
# =============================

df["text_clean"] = df["text"].apply(clean_text)

# Supprimer lignes vides après nettoyage
df_nouveau = df[df["text_clean"].str.strip() != ""]

# =============================
# 6️⃣ Prédiction sur texte nettoyé
# =============================

df_nouveau["trl_pred"] = model.predict(df_nouveau["text_clean"])

print(df_nouveau[["text_clean", "trl_pred"]].head())

df_nouveau.to_csv("mon_nouveau_corpus_avec_trl_avec_nettoyage.csv", index=False)


# ==============================
# 📊 VISUALISATION DES TRL
# ==============================

import matplotlib.pyplot as plt

# 🔢 Nombre total de brevets
total_patents = len(df_nouveau)
print(f"\nNombre total de brevets dans le corpus : {total_patents}")

# 📊 Répartition des TRL
trl_counts = df_nouveau["trl_pred"].value_counts().sort_index()

print("\nRépartition des TRL :")
print(trl_counts)

# ==============================
# 📊 Graphique 1 : Diagramme en barres
# ==============================

plt.figure()
plt.bar(trl_counts.index.astype(str), trl_counts.values)
plt.title("Répartition des niveaux TRL")
plt.xlabel("TRL")
plt.ylabel("Nombre de brevets")
plt.xticks(rotation=0)
plt.show()

# ==============================
# 📊 Graphique 2 : Diagramme circulaire
# ==============================

plt.figure()
plt.pie(trl_counts.values, labels=trl_counts.index.astype(str), autopct="%1.1f%%")
plt.title("Proportion des niveaux TRL")
plt.show()
