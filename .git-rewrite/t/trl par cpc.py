import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. CLASSE DE DÉTECTION TRL
# ==========================================
class TRLDetectorInferrer:
    def __init__(self):
        self.TRL_LONG = {
            1: "basic principles observed fundamental research theoretical studies lowest maturity scientific discovery idea generation",
            2: "technology concept formulated hypothesis speculative application potential use cases theoretical framework no validation",
            3: "experimental proof of concept critical function validation feasibility study lab experiment analytical studies early prototype",
            4: "technology validation in lab component testing prototype in controlled environment laboratory verification low-fidelity",
            5: "technology validation in relevant environment high-fidelity prototype simulation industrial validation risk reduction",
            6: "system demonstration in relevant environment pilot functional prototype near-operational performance testing",
            7: "system demonstration in operational environment field testing real world conditions deployment pre-commercial",
            8: "system complete and qualified certification standards production readiness commercial ready final system",
            9: "system proven in operational environment market deployment full scale commercialized sales mission ready"
        } #bibliothèque
        self.trl_levels = np.arange(1, 10, dtype=int)
        self.ref_texts = [self.TRL_LONG[i] for i in self.trl_levels]

        self.VEC = TfidfVectorizer(
            analyzer="word", 
            ngram_range=(1, 2), 
            min_df=1, #Document Frequency -> Garde les mots qui apparaissent dans AU MOINS 1 document
            stop_words='english',
            sublinear_tf=True #Echelle log pour l'importance des termes
        )
        self.VEC.fit(self.ref_texts) # Entraînement du modèle sur les phrases de la bibliothèque
        self.ref_vectors = self.VEC.transform(self.ref_texts) # Convertis les mots en vecteurs

        self.REGEX_TRL = re.compile(r"\bTRL\s*[:=]?\s*([1-9])(?:\s*[-to]\s*([1-9]))?", re.IGNORECASE) #chercher les TRL en toutes lettres

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        text = re.sub(r"http\S+", "", text) #enlever les url
        return re.sub(r"\s+", " ", text).strip() # Remplace toute séquence d'espaces par 1 SEUL espace

    def get_sentence_context(self, text, start_idx, end_idx): #cherche la phrase de contexte avec reconaissance regex du trl
        s_bound = text.rfind('.', 0, start_idx) + 1
        e_bound = text.find('.', end_idx)
        if e_bound == -1: e_bound = len(text)
        
        snippet = text[s_bound:e_bound].strip()
        if len(snippet) > 300:
            snippet = "..." + text[max(0, start_idx-50):min(len(text), end_idx+50)] + "..."
        return snippet #si phrase trop longue on tronque à 50 caractères avant et après

    def detect_smart(self, text):
        text_clean = self.clean_text(text)
        if len(text_clean) < 10: return None, None, 0.0, ""
        
        # 1. Priorité REGEX
        match = self.REGEX_TRL.search(text_clean)
        if match:
            start_val = int(match.group(1))
            end_val = int(match.group(2)) if match.group(2) else start_val
            
            start_idx, end_idx = match.span()
            justif = self.get_sentence_context(text_clean, start_idx, end_idx)
            
            return max(start_val, end_val), "regex", 1.0, justif

        # 2. Fallback INFÉRENCE
        try:
            sentences = re.split(r'(?<=[.!?])\s+', text_clean)
            valid_sentences = [s for s in sentences if len(s) > 20]
            if not valid_sentences: valid_sentences = [text_clean]

            vec_sentences = self.VEC.transform(valid_sentences)
            sims = cosine_similarity(vec_sentences, self.ref_vectors)
            
            max_idx_flat = np.argmax(sims)
            sent_idx, trl_idx = np.unravel_index(max_idx_flat, sims.shape)
            best_score = sims[sent_idx, trl_idx]
            
            if best_score > 0.15: 
                found_trl = self.trl_levels[trl_idx]
                justif = valid_sentences[sent_idx].strip()
                return found_trl, "inference", float(best_score), justif
                
        except Exception:
            pass
            
        return None, None, 0.0, ""


# ==========================================
# 2. EXTRACTION DES CODES CPC
# ==========================================
def get_all_main_cpc_codes(cpc_string):
    """
    Extrait TOUS les codes CPC principaux (avant le /) d'une chaîne
    Enlève les doublons au sein d'un même brevet
    Exemple: "B61F5/24; H04L29/06; B61F5/30" -> ["B61F5", "H04L29"]
    """
    if not isinstance(cpc_string, str) or not cpc_string.strip():
        return []
    
    # Séparer tous les codes CPC
    codes = [c.strip() for c in cpc_string.split(';') if c.strip()]
    if not codes: #vérifie si la liste est vide après les itérations
        return []
    
    # Extraire la partie principale (avant le /) de chaque code
    main_codes = []
    for code in codes:
        if '/' in code:
            main_code = code.split('/')[0].strip()
        else:
            main_code = code.strip()
        
        if main_code:
            main_codes.append(main_code)
    
    # Enlever les doublons au sein d'un même brevet
    return list(set(main_codes))


# ==========================================
# 3. ANALYSE TRL PAR CPC
# ==========================================
def analyze_trl_by_cpc(df_silver, top_n_cpc=15):
    """Analyse la répartition des TRL par code CPC (tous les codes par brevet)"""
    
    print("\n" + "="*60)
    print("ANALYSE TRL PAR CODE CPC (TOUS LES CODES)")
    print("="*60)
    
    # Extraire TOUS les codes CPC principaux pour chaque brevet
    df_silver['cpc_main_list'] = df_silver['cpc'].apply(get_all_main_cpc_codes)
    
    # Créer une ligne par code CPC (expansion)
    # Un brevet avec 3 codes CPC sera compté 3 fois
    expanded_rows = []
    for idx, row in df_silver.iterrows():
        for cpc_code in row['cpc_main_list']:
            expanded_rows.append({
                'num': row.get('num', idx),
                'cpc_main': cpc_code,
                'label': row['label'],
                'source': row['source'],
                'conf': row['conf']
            })
    
    df_expanded = pd.DataFrame(expanded_rows)
    
    print(f"Brevets avec code CPC valide : {len(df_silver)}")
    print(f"Total d'associations brevet-CPC : {len(df_expanded)}")
    
    # Compter les associations par CPC
    cpc_counts = df_expanded['cpc_main'].value_counts()
    print(f"\nNombre total de codes CPC uniques : {len(cpc_counts)}")
    
    # Sélectionner les TOP N CPC
    top_cpcs = cpc_counts.head(top_n_cpc).index.tolist()
    
    print(f"\nTop {top_n_cpc} codes CPC les plus fréquents :")
    for cpc, count in cpc_counts.head(top_n_cpc).items():
        print(f"  {cpc:15s} : {count:4d} associations brevet-CPC")
    
    # Créer une matrice CPC × TRL
    crosstab = pd.crosstab(
        df_expanded['cpc_main'], 
        df_expanded['label'],
        margins=True,
        margins_name='Total'
    )
    
    # Filtrer pour ne garder que les TOP CPC
    crosstab_top = crosstab.loc[top_cpcs, :]
    
    print(f"\n{'='*60}")
    print(f"MATRICE TRL PAR CPC (Top {top_n_cpc})")
    print(f"{'='*60}")
    print(crosstab_top)
    
    # Sauvegarder la matrice complète
    crosstab.to_csv('trl_par_cpc_matrice_complete.csv')
    print(f"\n✓ Matrice complète sauvegardée : trl_par_cpc_matrice_complete.csv")
    
    return df_expanded, crosstab_top, top_cpcs


# ==========================================
# 4. VISUALISATIONS
# ==========================================
def create_visualizations(df_expanded, crosstab_top, top_cpcs):
    """Génère les visualisations de la répartition TRL par CPC"""
    
    print(f"\n{'='*60}")
    print("GÉNÉRATION DES VISUALISATIONS")
    print(f"{'='*60}")
    
    # 1. Heatmap : Répartition TRL par CPC
    plt.figure(figsize=(14, 10))
    
    # Exclure la colonne 'Total' pour la heatmap
    heatmap_data = crosstab_top.drop('Total', axis=1) if 'Total' in crosstab_top.columns else crosstab_top
    
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Nombre de brevets'},
                linewidths=0.5)
    plt.title(f'Répartition des TRL par code CPC (Top {len(top_cpcs)})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Niveau TRL', fontsize=12)
    plt.ylabel('Code CPC', fontsize=12)
    plt.tight_layout()
    plt.savefig('heatmap_trl_par_cpc.png', dpi=300, bbox_inches='tight')
    print("✓ heatmap_trl_par_cpc.png")
    plt.close()
    
    # 2. Distribution des TRL pour chaque CPC (graphiques séparés)
    n_cpcs = len(top_cpcs)
    fig, axes = plt.subplots(n_cpcs, 1, figsize=(12, 4*n_cpcs))
    
    if n_cpcs == 1:
        axes = [axes]
    
    for idx, cpc in enumerate(top_cpcs):
        df_cpc = df_expanded[df_expanded['cpc_main'] == cpc]
        trl_dist = df_cpc['label'].value_counts().sort_index()
        
        axes[idx].bar(trl_dist.index, trl_dist.values, color='steelblue', edgecolor='black')
        axes[idx].set_xlabel('Niveau TRL', fontsize=10)
        axes[idx].set_ylabel('Nombre de brevets', fontsize=10)
        axes[idx].set_title(f'{cpc} ({len(df_cpc)} associations brevet-CPC)', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xticks(range(1, 10))
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribution_trl_par_cpc.png', dpi=300, bbox_inches='tight')
    print("✓ distribution_trl_par_cpc.png")
    plt.close()
    
    # 3. TRL moyen par CPC
    trl_means = df_expanded.groupby('cpc_main')['label'].mean().sort_values(ascending=False)
    trl_means_top = trl_means.loc[top_cpcs]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(trl_means_top)))
    plt.barh(range(len(trl_means_top)), trl_means_top.values, color=colors)
    plt.yticks(range(len(trl_means_top)), trl_means_top.index)
    plt.xlabel('TRL moyen', fontsize=12)
    plt.title(f'TRL moyen par code CPC (Top {len(top_cpcs)})', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('trl_moyen_par_cpc.png', dpi=300, bbox_inches='tight')
    print("✓ trl_moyen_par_cpc.png")
    plt.close()
    
    # 4. Résumé statistique par CPC
    summary_stats = df_expanded.groupby('cpc_main')['label'].agg(['count', 'mean', 'std', 'min', 'max'])
    summary_stats = summary_stats.loc[top_cpcs]
    summary_stats.columns = ['Nb_associations', 'TRL_moyen', 'TRL_std', 'TRL_min', 'TRL_max']
    summary_stats = summary_stats.round(2)
    
    print(f"\n{'='*60}")
    print(f"STATISTIQUES TRL PAR CPC (Top {len(top_cpcs)})")
    print(f"{'='*60}")
    print(summary_stats)
    
    summary_stats.to_csv('statistiques_trl_par_cpc.csv')
    print(f"\n✓ statistiques_trl_par_cpc.csv")


# ==========================================
# 5. PIPELINE PRINCIPAL
# ==========================================
def main():
    PROJECTS_FILE = "batch_patents.csv"
    
    print(f"--- Chargement : {PROJECTS_FILE} ---")
    try:
        df = pd.read_csv(PROJECTS_FILE, sep=';', on_bad_lines='skip', low_memory=False)
        if len(df.columns) < 3:
             df = pd.read_csv(PROJECTS_FILE, sep=',', on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"Erreur lecture : {e}")
        return

    print(f"Total brevets : {len(df)}")
    
    # Standardisation colonnes
    df.columns = [c.lower() for c in df.columns]
    
    # Vérifier que la colonne CPC existe
    if 'cpc' not in df.columns:
        print("⚠️ ERREUR : La colonne 'cpc' est absente du fichier CSV")
        return
    
    # Préparation Texte
    df['text'] = df['title'].fillna('') + ". " + df['abstract'].fillna('') + ". " + df['claims'].fillna('')
    
    # Nettoyage
    df = df[df['text'].str.len() > 50].copy()
    print(f"Projets avec texte valide : {len(df)}")
    
    # --- DÉTECTION TRL ---
    print("\nLancement de l'analyse TRL...")
    detector = TRLDetectorInferrer()
    
    results = df['text'].apply(lambda x: detector.detect_smart(x))
    
    df['label'] = [r[0] for r in results]
    df['source'] = [r[1] for r in results]
    df['conf'] = [r[2] for r in results]
    df['justification'] = [r[3] for r in results]
    
    # Filtrage
    df_silver = df.dropna(subset=['label']).copy()
    df_silver['label'] = df_silver['label'].astype(int)
    
    print(f"\n✅ TRL détectés pour {len(df_silver)} brevets")
    
    # --- ANALYSE TRL PAR CPC ---
    df_expanded, crosstab_top, top_cpcs = analyze_trl_by_cpc(df_silver, top_n_cpc=15)
    
    # --- VISUALISATIONS ---
    create_visualizations(df_expanded, crosstab_top, top_cpcs)
    
    print("\n" + "="*60)
    print("✅ ANALYSE TERMINÉE")
    print("="*60)
    print("\nFichiers générés :")
    print("  - trl_par_cpc_matrice_complete.csv")
    print("  - statistiques_trl_par_cpc.csv")
    print("  - heatmap_trl_par_cpc.png")
    print("  - distribution_trl_par_cpc.png")
    print("  - trl_moyen_par_cpc.png")
    print("="*60)

if __name__ == "__main__":
    main()