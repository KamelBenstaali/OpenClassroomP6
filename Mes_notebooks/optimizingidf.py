# -*- coding: utf-8 -*-
import os
import re
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from itertools import product
from joblib import Parallel, delayed

# =====================================================================
#                      Téléchargements NLTK
# =====================================================================
def _ensure_nltk_resources():
    pkgs = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for pkg, path in pkgs.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass
_ensure_nltk_resources()

# =====================================================================
#                            Fonctions utilitaires
# =====================================================================
def extract_main_category(category_tree):
    if pd.isna(category_tree):
        return "Unknown"
    try:
        clean_tree = str(category_tree).strip('[]"')
        categories = clean_tree.split(">>")
        return categories[0].strip()
    except Exception:
        return "Error"

def clean_text(text):
    if pd.isna(text) or text == "":
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize_and_process(text):
    if not text:
        return []
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) >= 3]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

def evaluate_clustering_no_reduction(features, true_labels, n_clusters=7, random_state=42, verbose=True):
    results = []
    try:
        if verbose:
            print("="*60)
            print("🔍 ÉVALUATION SANS RÉDUCTION DIMENSIONNELLE")
            print("="*60)
            print(f"   Dimensions: {features.shape}")

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        ari_score = adjusted_rand_score(true_labels, cluster_labels)

        results.append({
            'Dimensions': features.shape[1],
            'ARI_Score': ari_score,
            'Inertia': kmeans.inertia_
        })

        if verbose:
            print(f"   ✅ ARI: {ari_score:.4f}")
    except Exception as e:
        if verbose:
            print(f"   ❌ Erreur: {e}")
        results.append({'Dimensions': None, 'ARI_Score': np.nan, 'Inertia': np.nan})
    return pd.DataFrame(results)

# =====================================================================
#               Grid Search TF-IDF combiné PARALLÉLISÉ
# =====================================================================
from scipy.sparse import hstack

def _fit_vectorizer(texts_fit, params):
    """Helper pour créer et fitter un TfidfVectorizer"""
    vec = TfidfVectorizer(
        max_features=params['max_features'],
        ngram_range=params['ngram_range'],
        lowercase=True,
        stop_words='english',
        min_df=params['min_df'],
        max_df=params['max_df'],
    )
    vec.fit(texts_fit)
    return vec

def _test_combination(params_name, params_desc, texts_name, texts_desc, true_labels, n_clusters, random_state, enrich_vocab):
    """Exécute UN essai de combinaison name+desc et retourne les résultats."""
    try:
        # Fit vectorizers
        fit_corpus_name = (list(texts_name) + list(texts_desc)) if enrich_vocab else texts_name
        fit_corpus_desc = (list(texts_desc) + list(texts_name)) if enrich_vocab else texts_desc

        name_vec = _fit_vectorizer(fit_corpus_name, params_name)
        desc_vec = _fit_vectorizer(fit_corpus_desc, params_desc)

        X_name = name_vec.transform(texts_name)
        X_desc = desc_vec.transform(texts_desc)

        if X_name.shape[1] == 0 or X_desc.shape[1] == 0:
            return None  # skip empty

        X_combined = hstack([X_name, X_desc]).toarray()

        res = evaluate_clustering_no_reduction(
            X_combined, true_labels, n_clusters=n_clusters, random_state=random_state, verbose=False
        )
        ari = float(res.loc[0, "ARI_Score"])
        inertia = float(res.loc[0, "Inertia"])

        return {
            "params_name": params_name,
            "params_desc": params_desc,
            "name_dim": X_name.shape[1],
            "desc_dim": X_desc.shape[1],
            "combined_dim": X_combined.shape[1],
            "ARI_Score": ari,
            "Inertia": inertia
        }

    except Exception:
        return None

def grid_search_tfidf_combined_parallel(
    texts_name,
    texts_desc,
    true_labels,
    n_clusters=7,
    random_state=42,
    name_grid=None,
    desc_grid=None,
    enrich_vocab_with_each_other=True,
    n_jobs=-1,
    verbose=True
):
    """
    Grid search parallèle TF-IDF : teste toutes les combinaisons (name × desc) sur plusieurs CPU.
    """
    # grilles par défaut
    if name_grid is None:
        name_grid = {
            "max_features": [50, 100, 200],
            "ngram_range": [(1,1), (1,2)],
            "min_df": [1, 0.01],
            "max_df": [0.70, 0.95],
        }
    if desc_grid is None:
        desc_grid = {
            "max_features": [100, 200, 500],
            "ngram_range": [(1,1), (1,2), (1,3)],
            "min_df": [1, 2, 0.01],
            "max_df": [0.70, 0.85, 0.95],
        }

    # combinaisons possibles
    name_combos = [
        dict(max_features=mf, ngram_range=ng, min_df=mindf, max_df=maxdf)
        for mf, ng, mindf, maxdf in product(
            name_grid["max_features"], name_grid["ngram_range"],
            name_grid["min_df"], name_grid["max_df"]
        )
        if not (isinstance(mindf, float) and isinstance(maxdf, float) and mindf > maxdf)
    ]

    desc_combos = [
        dict(max_features=mf, ngram_range=ng, min_df=mindf, max_df=maxdf)
        for mf, ng, mindf, maxdf in product(
            desc_grid["max_features"], desc_grid["ngram_range"],
            desc_grid["min_df"], desc_grid["max_df"]
        )
        if not (isinstance(mindf, float) and isinstance(maxdf, float) and mindf > maxdf)
    ]

    total = len(name_combos) * len(desc_combos)
    if verbose:
        print(f"🔧 Nombre total de combinaisons testées : {total}")
        print(f"💻 Exécution parallèle sur {n_jobs if n_jobs != -1 else os.cpu_count()} cœurs...")

    start = time.time()

    # parallélisation
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_test_combination)(
            params_name, params_desc, texts_name, texts_desc, true_labels,
            n_clusters, random_state, enrich_vocab_with_each_other
        )
        for params_name, params_desc in product(name_combos, desc_combos)
    )

    # filtrer les None
    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("Aucun résultat valide (trop de filtrage min_df/max_df ?)")

    results_df = pd.DataFrame(results).sort_values("ARI_Score", ascending=False).reset_index(drop=True)

    best = results_df.iloc[0].to_dict()
    best["params_name"] = results_df.iloc[0]["params_name"]
    best["params_desc"] = results_df.iloc[0]["params_desc"]

    elapsed = time.time() - start
    if verbose:
        print(f"✅ Terminé en {elapsed/60:.1f} minutes. Meilleur ARI={best['ARI_Score']:.4f}")

    return results_df, best

# =====================================================================
#                           Chargement & Prétraitement
# =====================================================================
print("Chargement du CSV...")
df = pd.read_csv('./Dataset+projet+pretraitement+textes+images/Flipkart/flipkart_com-ecommerce_sample_1050.csv')

df['category'] = df['product_category_tree'].apply(extract_main_category)
categories = df['category'].values

print("Nettoyage et tokenisation...")
for col in ['product_name', 'description']:
    df[f'{col}_cleaned'] = df[col].apply(clean_text)
    df[f'{col}_tokens'] = df[col].apply(lambda x: tokenize_and_process(clean_text(x)))
    df[f'{col}_processed'] = df[f'{col}_tokens'].apply(lambda x: ' '.join(x))

# =====================================================================
#                   Lancement du grid search parallèle
# =====================================================================
print("\nLancement du grid search parallèle TF-IDF (features combinées)...")

name_grid = {
    "max_features": [50, 100],
    "ngram_range": [(1,1), (1,2), (1,3)],
    "min_df": [1, 2, 3, 0.01, 0.02],
    "max_df": [0.70, 0.85, 0.95],
}

desc_grid = {
    "max_features": [100, 200],
    "ngram_range": [(1,1), (1,2), (1,3), (1,4)],
    "min_df": [1, 2, 3, 0.01, 0.02, 0.05],
    "max_df": [0.70, 0.80, 0.90, 0.95],
}


results_parallel, best_combined = grid_search_tfidf_combined_parallel(
    texts_name=df['product_name_processed'].fillna(''),
    texts_desc=df['description_processed'].fillna(''),
    true_labels=categories,
    n_clusters=7,
    random_state=42,
    name_grid=name_grid,
    desc_grid=desc_grid,
    enrich_vocab_with_each_other=True,
    n_jobs=-1,   # utilise tous les cœurs
    verbose=True
)

print("\n===== TOP 10 combinaisons =====")
print(results_parallel.head(10))
print("\n===== MEILLEURE COMBINAISON =====")
print("Name:", best_combined["params_name"])
print("Desc:", best_combined["params_desc"])
print(f"Best ARI: {best_combined['ARI_Score']:.4f}")
