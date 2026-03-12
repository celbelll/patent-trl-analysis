# ==============================
# Ce code permet d'évaluer les niveaux TRL d'un corpus de brevets du modèle frugal_trl_model.pkl, après un nettoyage préalable du texte : suppression des accents, conversion en minuscules, séparation des lettres et chiffres, suppression des caractères non alphanumériques, normalisation des espaces et suppression des stopwords.
# Il génère 2 fichiers .csv : l'un qui reprend le fichier d'entrée avec trois colonnes supplémentaires : le texte (titre, abstract, claims) concaténé, le texte concaténé et nettoyé, et l'évaluation du trl; et l'autre qui répertorie les top mots par TRL (avec leur rang et leur poids dans le modèle).
# Et il affiche 3 diagrammes en sortie, un diagramme batons et un diagramme circulaire, pour visualiser la répartition des niveaux TRL, ainsi qu'un diagramme en barres horizontales pour visualiser les top mots par TRL.
# ==============================


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
    "otherwise", "type", "base", "based",
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

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

n_classes = len(classes)
ROW_HEIGHT = 0.45
PADDING = 2.5
height_per_subplot = TOP_N * ROW_HEIGHT + PADDING
total_height = height_per_subplot * n_classes

fig, axes = plt.subplots(n_classes, 1, figsize=(9, total_height))
if n_classes == 1:
    axes = [axes]

for i, trl_class in enumerate(classes):
    df_trl = df_features[df_features["trl"] == trl_class].sort_values("weight", ascending=True)
    axes[i].barh(df_trl["word"], df_trl["weight"], color="steelblue", edgecolor="black", height=0.6)
    axes[i].set_title(f"Top {TOP_N} mots — TRL {trl_class}", fontsize=12, fontweight="bold", pad=8)
    axes[i].set_xlabel("Coefficient SVM", fontsize=10)
    axes[i].tick_params(axis='y', labelsize=10)
    axes[i].axvline(x=0, color="red", linestyle="--", linewidth=0.8)
    axes[i].grid(axis="x", alpha=0.3)

fig.tight_layout(pad=3.0)
fig.savefig("top_features_par_trl.png", dpi=150, bbox_inches="tight")
print("✓ top_features_par_trl.png généré")

# --- Fenêtre Tkinter scrollable ---
root = tk.Tk()
root.title("Top features par TRL — utilisez la molette pour défiler")
root.geometry("1100x800")

frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

canvas_tk = tk.Canvas(frame)
scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas_tk.yview)
canvas_tk.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas_tk.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

inner_frame = tk.Frame(canvas_tk)
canvas_tk.create_window((0, 0), window=inner_frame, anchor="nw")

fig_canvas = FigureCanvasTkAgg(fig, master=inner_frame)
fig_canvas.draw()
fig_canvas.get_tk_widget().pack()

def on_configure(event):
    canvas_tk.configure(scrollregion=canvas_tk.bbox("all"))

inner_frame.bind("<Configure>", on_configure)

# Molette souris
def on_mousewheel(event):
    canvas_tk.yview_scroll(int(-1 * (event.delta / 120)), "units")

root.bind_all("<MouseWheel>", on_mousewheel)

root.protocol("WM_DELETE_WINDOW", root.destroy)

root.mainloop()
plt.close('all')