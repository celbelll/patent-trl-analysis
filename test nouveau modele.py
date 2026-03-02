import pandas as pd
import joblib
 
model = joblib.load("C:/Users/CéliaBelaziz/Downloads/frugal_trl_model.pkl")
print(model)

PATENTS_PATH = "batch_patents.csv"

# lecture CSV (si pas déjà fait)
df = pd.read_csv(PATENTS_PATH, dtype=str)

# s'assurer que les colonnes existent et remplacer NaN par chaîne vide
for col in ("title", "abstract", "claims"):
    if col not in df.columns:
        df[col] = ""
df[["title", "abstract", "claims"]] = df[["title", "abstract", "claims"]].fillna("")

# concaténer avec nettoyage des espaces
df["text"] = (
    df["title"].str.strip().fillna("")
    + " "
    + df["abstract"].str.strip().fillna("")
    + " "
    + df["claims"].str.strip().fillna("")
)
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

df_nouveau = df.dropna(subset=['text'])
 
df_nouveau['trl_pred'] = model.predict(df_nouveau['text'])
print(df_nouveau[['text', 'trl_pred']].head())
 
df_nouveau.to_csv("mon_nouveau_corpus_avec_trl.csv", index=False)



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
