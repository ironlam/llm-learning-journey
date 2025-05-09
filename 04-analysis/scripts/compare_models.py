#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de comparaison entre un modèle générique et un modèle fine-tuné
pour l'analyse de sentiment.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Configuration
FINE_TUNED_MODEL_PATH = "../../02-fine-tuning/sentiment-model/final-model"
OUTPUT_DIR = "../results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "model_comparison.png")

# Liste d'exemples intéressants à comparer
examples = [
    "Quelle analyse profonde, je n'aurais jamais pu trouver ces informations sur Wikipedia !",
    "Cet article n'est pas si mal pour un lundi matin.",
    "Bravo pour cet article qui enfonce des portes ouvertes !",
    "J'ai beaucoup appris en lisant cet article, merci à l'auteur.",
    "On nous prend vraiment pour des idiots avec ce genre d'analyse simpliste."
]


def analyze_with_both_models(examples):
    """
    Analyse les exemples avec les deux modèles et retourne les résultats.
    """
    print("Chargement des modèles...")

    # Charger les deux modèles
    generic_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    fine_tuned_model = pipeline("sentiment-analysis", model=FINE_TUNED_MODEL_PATH)

    # Fonction pour convertir les labels du modèle générique en positif/négatif
    def convert_label(label):
        # Le modèle générique utilise des étoiles (1 à 5 stars)
        stars = int(label.split()[0])
        return "POSITIF" if stars > 3 else "NÉGATIF"

    # Analyser les exemples avec les deux modèles
    results = []
    for i, example in enumerate(examples):
        # Analyse avec le modèle générique
        generic_result = generic_model(example)[0]
        generic_sentiment = convert_label(generic_result["label"])
        generic_score = float(generic_result["score"])

        # Analyse avec le modèle fine-tuné
        fine_tuned_result = fine_tuned_model(example)[0]
        fine_tuned_label = fine_tuned_result["label"]
        fine_tuned_score = float(fine_tuned_result["score"])

        # Déterminer si le modèle fine-tuné prédit positif ou négatif
        # Adapter selon les labels de sortie de votre modèle
        if fine_tuned_label == "LABEL_0":
            fine_tuned_sentiment = "NÉGATIF"  # Ajuster selon votre modèle
        else:
            fine_tuned_sentiment = "POSITIF"  # Ajuster selon votre modèle

        results.append({
            "Texte": example,
            "Modèle générique": f"{generic_sentiment} ({generic_score:.2f})",
            "Modèle fine-tuné": f"{fine_tuned_sentiment} ({fine_tuned_score:.2f})",
            "generic_sentiment": generic_sentiment,
            "generic_score": generic_score,
            "fine_tuned_sentiment": fine_tuned_sentiment,
            "fine_tuned_score": fine_tuned_score
        })

    return results


def create_comparison_visualization(results, examples):
    """
    Crée une visualisation comparative des résultats.
    """
    plt.figure(figsize=(12, 3 * len(examples)))

    # Créer un subplot pour chaque exemple
    for i, example in enumerate(examples):
        plt.subplot(len(examples), 1, i + 1)

        # Extraire les scores spécifiques à cet exemple
        generic_score = results[i]["generic_score"]
        fine_tuned_score = results[i]["fine_tuned_score"]

        # Déterminer les couleurs en fonction du sentiment
        generic_color = "#4CAF50" if results[i]["generic_sentiment"] == "POSITIF" else "#F44336"
        fine_tuned_color = "#4CAF50" if results[i]["fine_tuned_sentiment"] == "POSITIF" else "#F44336"

        # Créer les barres
        data = [generic_score, fine_tuned_score]
        labels = ["Générique", "Fine-tuné"]
        colors = [generic_color, fine_tuned_color]

        bars = plt.barh(labels, data, color=colors)

        # Formater le titre pour qu'il soit plus court si nécessaire
        short_example = example[:50] + "..." if len(example) > 50 else example
        plt.title(f"Exemple {i + 1}: {short_example}")

        # Formater le graphique
        plt.xlim(0, 1)
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.2f}",
                     ha='left', va='center')

    plt.tight_layout()

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sauvegarder la visualisation
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"Visualisation sauvegardée dans '{OUTPUT_PLOT}'")


def main():
    """Fonction principale."""
    # Analyser les exemples avec les deux modèles
    results = analyze_with_both_models(examples)

    # Créer un DataFrame pour l'affichage et la sauvegarde
    display_columns = ["Texte", "Modèle générique", "Modèle fine-tuné"]
    df = pd.DataFrame(results)[display_columns]

    # Afficher les résultats
    print("\nComparaison des modèles:")
    print(df)

    # Sauvegarder les résultats dans un CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRésultats sauvegardés dans '{OUTPUT_CSV}'")

    # Créer et sauvegarder la visualisation
    create_comparison_visualization(results, examples)


if __name__ == "__main__":
    main()