#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Générateur de dataset de commentaires pour l'analyse de sentiment.

Ce script télécharge et prépare un dataset de commentaires en français
à partir de Hugging Face, adapté pour le fine-tuning d'un modèle d'analyse
de sentiment.
"""

import os
import pandas as pd
import csv
from datasets import load_dataset
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "../data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "commentaires_articles.csv")
MAX_SAMPLES = 1000  # Nombre maximum de commentaires à inclure
SEED = 42


def prepare_dataset():
    """
    Prépare un dataset à partir des critiques Allocine.
    """
    print("Téléchargement du dataset Allocine depuis Hugging Face...")

    # Charger le dataset
    dataset = load_dataset("allocine")

    # Extraire les données pertinentes
    reviews = dataset["train"]["review"]
    labels = dataset["train"]["label"]  # Déjà au format binaire

    # Créer la liste de données
    data = []
    for review, label in tqdm(zip(reviews, labels), total=len(reviews), desc="Traitement des commentaires"):
        # Nettoyage minimal des textes
        review = review.replace('"', "'")  # Remplacer guillemets doubles par simples

        # Filtrer les commentaires trop courts ou vides
        if len(review.strip()) < 10:
            continue

        data.append({
            "text": review,
            "label": label,
            "source": "Allocine"
        })

    print(f"Commentaires extraits: {len(data)}")

    # Créer le DataFrame
    df = pd.DataFrame(data)

    return df


def create_balanced_dataset(df, max_samples):
    """
    Crée un dataset équilibré avec un nombre égal d'exemples positifs et négatifs.
    """
    print("Équilibrage du dataset...")

    # Séparer les commentaires positifs et négatifs
    positive_df = df[df["label"] == 1]
    negative_df = df[df["label"] == 0]

    # Déterminer combien d'exemples prendre de chaque catégorie
    sample_size = min(len(positive_df), len(negative_df), max_samples // 2)

    # Échantillonner de manière équilibrée
    balanced_positive = positive_df.sample(n=sample_size, random_state=SEED)
    balanced_negative = negative_df.sample(n=sample_size, random_state=SEED)

    # Combiner et mélanger
    balanced_df = pd.concat([balanced_positive, balanced_negative])
    final_df = balanced_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return final_df


def save_dataframe_to_csv(df, output_file, columns):
    """
    Sauvegarde un DataFrame dans un fichier CSV avec une gestion appropriée des guillemets.
    """
    # Sélectionner les colonnes spécifiées
    df_to_save = df[columns].copy()

    # Utiliser le mode d'échappement correct pour les guillemets
    df_to_save.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')


def main():
    """Fonction principale du script."""
    print("=== Génération du dataset de commentaires pour l'analyse de sentiment ===")

    try:
        # Créer le dataset
        df = prepare_dataset()

        # Équilibrer le dataset
        balanced_df = create_balanced_dataset(df, MAX_SAMPLES)

        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Sauvegarder les fichiers
        output_columns = ["text", "label"]
        save_dataframe_to_csv(balanced_df, OUTPUT_FILE, output_columns)

        full_output_file = OUTPUT_FILE.replace(".csv", "_full.csv")
        save_dataframe_to_csv(balanced_df, full_output_file, ["text", "label", "source"])

        # Afficher les statistiques
        print("\n=== Dataset généré avec succès ! ===")
        print(f"Nombre total de commentaires : {len(balanced_df)}")
        print(f"Commentaires positifs : {balanced_df['label'].sum()} ({balanced_df['label'].mean() * 100:.1f}%)")
        print(
            f"Commentaires négatifs : {len(balanced_df) - balanced_df['label'].sum()} ({(1 - balanced_df['label'].mean()) * 100:.1f}%)")
        print(f"\nFichier CSV sauvegardé : {OUTPUT_FILE}")
        print(f"Fichier complet avec métadonnées : {full_output_file}")

        # Afficher quelques exemples
        print("\n=== Exemples de commentaires ===")
        sample_df = balanced_df.sample(n=min(5, len(balanced_df)), random_state=SEED)
        for i, (_, row) in enumerate(sample_df.iterrows()):
            sentiment = "POSITIF" if row["label"] == 1 else "NÉGATIF"
            text_preview = row["text"][:100] + "..." if len(row["text"]) > 100 else row["text"]
            print(f"{i + 1}. [{sentiment}] {text_preview}")

    except Exception as e:
        print(f"Erreur lors de la génération du dataset : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()