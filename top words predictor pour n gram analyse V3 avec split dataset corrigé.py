# ==========================================
# topword_svm_single_corpus.py
# - SVM sur corpus unique splitté 80/20
# - Top n-char par TRL
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "mon_nouveau_corpus_avec_trl_avec_nettoyage.csv"

# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------

def clean_trl(val):
    try:
        return int(float(val))
    except Exception:
        return None

def plot_top_words(model, features, out_path, title_prefix, trl_classes=None, n_top=10):
    if trl_classes is None:
        trl_classes = list(model.classes_)

    available_classes = set(model.classes_)
    valid_classes = [c for c in trl_classes if c in available_classes]

    if not valid_classes:
        print("Aucune classe valide trouvée pour le plot.")
        return

    n_plots = len(valid_classes)
    cols = 2 if n_plots > 1 else 1
    rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, trl in enumerate(valid_classes):
        class_idx = np.where(model.classes_ == trl)[0][0]
        coefs = model.coef_[class_idx]

        top_indices = np.argsort(coefs)[-n_top:]
        top_keywords = features[top_indices]
        top_coeffs = coefs[top_indices]

        ax = axes[i]
        sns.barplot(
            x=top_coeffs,
            y=top_keywords,
            ax=ax,
            palette="viridis",
            orient='h'
        )
        ax.set_title(
            f"{title_prefix} TRL {trl}",
            fontsize=16,
            fontweight='bold'
        )
        ax.set_xlabel("SVM Coefficient Weight", fontsize=13)
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=16)

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


def plot_prediction_drivers(model, X_data, y_preds, features, out_path,
                            trl_classes=None, n_top=10):
    if trl_classes is None:
        trl_classes = list(model.classes_)

    present_classes = np.unique(y_preds)
    valid_classes = [c for c in trl_classes if c in present_classes]

    if not valid_classes:
        print("Aucune des classes demandées n'a été prédite.")
        return

    n_plots = len(valid_classes)
    cols = 2 if n_plots > 1 else 1
    rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, trl in enumerate(valid_classes):
        mask = (y_preds == trl)
        X_subset = X_data[mask]
        count = X_subset.shape[0]
        if count == 0:
            continue

        try:
            class_idx = np.where(model.classes_ == trl)[0][0]
            weights = model.coef_[class_idx]
        except IndexError:
            continue

        avg_tfidf = np.asarray(X_subset.mean(axis=0)).flatten()
        impact_scores = avg_tfidf * weights

        top_indices = np.argsort(impact_scores)[-n_top:]
        top_words = features[top_indices]
        top_scores_raw = impact_scores[top_indices]
        top_scores = np.sqrt(np.clip(top_scores_raw, a_min=0, a_max=None))

        ax = axes[i]
        sns.barplot(
            x=top_scores,
            y=top_words,
            ax=ax,
            palette="rocket",
            orient='h'
        )
        ax.set_title(
            f"Top words for TRL {trl} (n={count} docs)",
            fontsize=16,
            fontweight='bold'
        )
        ax.set_xlabel("Compressed impact (TF-IDF x SVM weight)", fontsize=13)
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Graphique sauvegardé : {out_path}")


# -------------------------------------------------------------------
# 1️⃣ Chargement et split 80/20
# -------------------------------------------------------------------

print("\n=== Chargement du corpus ===")

df = pd.read_csv(DATA_PATH)
df['label'] = df['trl_pred'].apply(clean_trl)
df['text'] = df['text_clean']
df = df.dropna(subset=['text', 'label'])
df = df[(df['label'] >= 1) & (df['label'] <= 9)].reset_index(drop=True)

print(f"Corpus chargé : {len(df)} brevets")
print(f"Classes TRL présentes : {sorted(df['label'].unique())}")

# Vérification que chaque classe a au moins 2 exemples pour le stratify
label_counts = df['label'].value_counts()
rare_labels = label_counts[label_counts < 2].index
if len(rare_labels) > 0:
    print(f"⚠️ Classes retirées (trop peu d'exemples) : {list(rare_labels)}")
    df = df[~df['label'].isin(rare_labels)]

df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"Train : {len(df_train)} brevets (80%)")
print(f"Test  : {len(df_test)} brevets (20%)")

# -------------------------------------------------------------------
# 2️⃣ Vectorisation TF-IDF
# -------------------------------------------------------------------

tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=50000,
    sublinear_tf=True
)

X_train = tfidf.fit_transform(df_train['text'])
y_train = df_train['label']

X_test = tfidf.transform(df_test['text'])
y_test = df_test['label'].astype(int).values

feature_names = np.array(tfidf.get_feature_names_out())

# -------------------------------------------------------------------
# 3️⃣ Entraînement SVM sur les 80%
# -------------------------------------------------------------------

svm_model = LinearSVC(
    class_weight='balanced',
    C=1.0,
    dual='auto',
    random_state=42
)

svm_model.fit(X_train, y_train)
print("✅ SVM entraîné sur 80% du corpus")

# Accuracy sur les 20% de test
y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy sur les 20% de test : {acc:.2%}")

# -------------------------------------------------------------------
# 4️⃣ Top n-grammes par TRL (basé sur les coefficients du modèle)
# -------------------------------------------------------------------

plot_top_words(
    svm_model,
    feature_names,
    out_path="figure_feature_importance.png",
    title_prefix="Top Predictors for",
    trl_classes=sorted(df['label'].unique()),
    n_top=10
)

df_top = build_topwords_table(svm_model, feature_names, top_k=50)
df_top.to_csv("results_topwords.csv", index=False)
print("✅ Fichier sauvegardé : results_topwords.csv")

# -------------------------------------------------------------------
# 5️⃣ Mots déclencheurs par classe prédite (sur les 20% de test)
# -------------------------------------------------------------------

print("\n=== Mots déclencheurs sur les 20% de test ===")

plot_prediction_drivers(
    model=svm_model,
    X_data=X_test,
    y_preds=y_pred,
    features=feature_names,
    out_path="figure_prediction_drivers.png",
    trl_classes=sorted(df['label'].unique()),
    n_top=10
)

print("\n✅ Fin du script")