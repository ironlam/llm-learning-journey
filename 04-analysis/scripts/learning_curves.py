#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualisation des courbes d'apprentissage à partir des logs d'entraînement.
"""

import matplotlib.pyplot as plt
import pandas as pd
import json
import glob
import os
import sys

# Configuration
LOGS_DIR = "../../02-fine-tuning/sentiment-model"
OUTPUT_FILE = "../results/learning_curves.png"


def extract_metrics_from_logs(logs_dir):
    """
    Extrait les métriques à partir des logs d'entraînement.
    Gère différents formats de logs possibles.
    """
    # Chercher les fichiers de log
    checkpoint_dirs = glob.glob(os.path.join(logs_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        print(f"Aucun checkpoint trouvé dans {logs_dir}")
        # Chercher directement trainer_state.json
        log_files = glob.glob(os.path.join(logs_dir, "**/trainer_state.json"), recursive=True)
        if not log_files:
            print(f"Aucun fichier de log trouvé dans {logs_dir}")
            return None
        latest_log = max(log_files, key=os.path.getctime)
    else:
        # Utiliser le fichier du dernier checkpoint
        latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
        latest_log = os.path.join(latest_checkpoint, "trainer_state.json")
        if not os.path.exists(latest_log):
            print(f"Fichier trainer_state.json non trouvé dans {latest_checkpoint}")
            return None

    print(f"Utilisation du fichier de log: {latest_log}")

    # Charger les données
    with open(latest_log, 'r') as f:
        log_data = json.load(f)

    # Extraire les métriques
    if 'log_history' in log_data:
        metrics = log_data['log_history']

        # Convertir en DataFrame
        df = pd.DataFrame(metrics)

        # Afficher les colonnes disponibles pour le débogage
        print(f"Colonnes disponibles dans les logs: {df.columns.tolist()}")

        return df
    else:
        print("Format de log non reconnu. Aucune entrée 'log_history' trouvée.")
        return None


def create_visualizations(df):
    """
    Crée différentes visualisations à partir des métriques.
    S'adapte aux colonnes disponibles dans les données.
    """
    if df is None or len(df) == 0:
        print("Pas de données à visualiser.")
        return False

    plt.figure(figsize=(15, 10))

    # 1. Visualisation de la perte d'entraînement
    plt.subplot(2, 2, 1)
    if 'loss' in df.columns:
        train_df = df[df['loss'].notna()]
        plt.plot(train_df.index, train_df['loss'], 'b-', marker='o')
        plt.title('Perte d\'entraînement')
        plt.xlabel('Étapes')
        plt.ylabel('Perte')
        plt.grid(True)
    elif 'train_loss' in df.columns:
        train_df = df[df['train_loss'].notna()]
        plt.plot(train_df.index, train_df['train_loss'], 'b-', marker='o')
        plt.title('Perte d\'entraînement')
        plt.xlabel('Étapes')
        plt.ylabel('Perte')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Données de perte d\'entraînement non disponibles',
                 ha='center', va='center', transform=plt.gca().transAxes)

    # 2. Visualisation de la perte d'évaluation
    plt.subplot(2, 2, 2)
    if 'eval_loss' in df.columns:
        eval_df = df[df['eval_loss'].notna()]
        plt.plot(eval_df.index, eval_df['eval_loss'], 'r-', marker='o')
        plt.title('Perte d\'évaluation')
        plt.xlabel('Étapes')
        plt.ylabel('Perte')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Données de perte d\'évaluation non disponibles',
                 ha='center', va='center', transform=plt.gca().transAxes)

    # 3. Visualisation de l'exactitude
    plt.subplot(2, 2, 3)
    if 'eval_accuracy' in df.columns:
        eval_df = df[df['eval_accuracy'].notna()]
        plt.plot(eval_df.index, eval_df['eval_accuracy'], 'g-', marker='o')
        plt.title('Exactitude d\'évaluation')
        plt.xlabel('Étapes')
        plt.ylabel('Exactitude')
        plt.grid(True)
        plt.ylim(0, 1)  # L'exactitude est entre 0 et 1
    else:
        plt.text(0.5, 0.5, 'Données d\'exactitude non disponibles',
                 ha='center', va='center', transform=plt.gca().transAxes)

    # 4. Visualisation du F1-score
    plt.subplot(2, 2, 4)
    if 'eval_f1' in df.columns:
        eval_df = df[df['eval_f1'].notna()]
        plt.plot(eval_df.index, eval_df['eval_f1'], 'orange', marker='o')
        plt.title('F1-Score d\'évaluation')
        plt.xlabel('Étapes')
        plt.ylabel('F1-Score')
        plt.grid(True)
        plt.ylim(0, 1)  # Le F1-score est entre 0 et 1
    else:
        plt.text(0.5, 0.5, 'Données de F1-score non disponibles',
                 ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()

    # Créer le dossier s'il n'existe pas
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Courbes d'apprentissage sauvegardées dans {OUTPUT_FILE}")

    return True


def main():
    """Fonction principale."""
    print("=== Visualisation des courbes d'apprentissage ===")

    # Extraire les métriques des logs
    df = extract_metrics_from_logs(LOGS_DIR)

    # Créer les visualisations
    success = create_visualizations(df)

    if not success:
        print("Échec de la création des visualisations.")
        sys.exit(1)

    print("Visualisations créées avec succès !")


if __name__ == "__main__":
    main()