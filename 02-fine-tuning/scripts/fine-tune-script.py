#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning d'un modèle de langage pour l'analyse de sentiment des commentaires.

Ce script permet de fine-tuner un modèle pré-entraîné de la bibliothèque Hugging Face
pour l'analyse de sentiment, en utilisant un dataset de commentaires d'articles.
"""

import os
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset, load_dataset

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Configure et parse les arguments en ligne de commande.
    
    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Fine-tune un modèle pour l'analyse de sentiment")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Nom ou chemin du modèle pré-entraîné",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imdb",
        help="Nom du dataset à utiliser pour le fine-tuning",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sentiment-model",
        help="Répertoire de sortie pour sauvegarder le modèle",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Nombre d'époques d'entraînement",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Taille du batch pour l'entraînement",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Taux d'apprentissage initial",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Longueur maximale des séquences",
    )
    parser.add_argument(
        "--use_custom_data",
        action="store_true",
        help="Utiliser un dataset personnalisé au lieu d'un dataset HF",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        default="",
        help="Chemin vers le fichier CSV du dataset personnalisé",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="Proportion des données à utiliser pour la validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine pour la reproductibilité",
    )
    return parser.parse_args()

def load_custom_dataset(data_path: str, validation_split: float = 0.2, seed: int = 42) -> Dict[str, Dataset]:
    """
    Charge un dataset personnalisé depuis un fichier CSV.
    
    Args:
        data_path (str): Chemin vers le fichier CSV
        validation_split (float, optional): Proportion des données pour la validation. Par défaut 0.2.
        seed (int, optional): Graine pour la reproductibilité. Par défaut 42.
        
    Returns:
        Dict[str, Dataset]: Dictionnaire contenant les datasets d'entraînement et de test
        
    Raises:
        ValueError: Si les colonnes requises sont manquantes
    """
    logger.info(f"Chargement du dataset personnalisé depuis {data_path}")
    df = pd.read_csv(data_path)
    
    # Vérifier les colonnes requises
    required_columns = ["text", "label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est requise dans le dataset")
    
    # Préparation des données
    train_df, test_df = train_test_split(df, test_size=validation_split, random_state=seed)
    
    # Conversion en datasets Hugging Face
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return {"train": train_dataset, "test": test_dataset}

def preprocess_data(datasets: Dict[str, Dataset], tokenizer, max_length: int = 128) -> Dict[str, Dataset]:
    """
    Prétraite les données en les tokenisant.
    
    Args:
        datasets (Dict[str, Dataset]): Dictionnaire des datasets
        tokenizer: Tokenizer à utiliser
        max_length (int, optional): Longueur maximale des séquences. Par défaut 128.
        
    Returns:
        Dict[str, Dataset]: Datasets tokenisés
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_datasets = {}
    for split in datasets:
        # Préserver la structure originale du dataset mais supprimer la colonne "text"
        # après tokenisation pour éviter la duplication des données
        tokenized_datasets[split] = datasets[split].map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"] if "text" in datasets[split].column_names else []
        )
        
        # Assurez-vous que les labels sont correctement formatés
        if "label" in tokenized_datasets[split].column_names:
            tokenized_datasets[split] = tokenized_datasets[split].cast_column("label", torch.LongTensor)
    
    return tokenized_datasets

def compute_metrics(pred) -> Dict[str, float]:
    """
    Calcule les métriques d'évaluation.
    
    Args:
        pred: Prédictions du modèle
        
    Returns:
        Dict[str, float]: Dictionnaire des métriques
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def fine_tune_model(args: argparse.Namespace) -> str:
    """
    Fonction principale pour le fine-tuning du modèle.
    
    Args:
        args (argparse.Namespace): Arguments de la ligne de commande
        
    Returns:
        str: Chemin vers le modèle final
        
    Raises:
        ValueError: Si le chemin du dataset personnalisé n'est pas spécifié
    """
    # Définir la graine pour la reproductibilité
    torch.manual_seed(args.seed)
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Chargement du modèle et du tokenizer
    logger.info(f"Chargement du modèle: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2  # 2 classes: positif/négatif
    )
    
    # Chargement du dataset
    if args.use_custom_data:
        if not args.custom_data_path:
            raise ValueError("Le chemin vers le dataset personnalisé n'est pas spécifié")
        datasets = load_custom_dataset(
            args.custom_data_path, 
            validation_split=args.validation_split,
            seed=args.seed
        )
    else:
        logger.info(f"Chargement du dataset: {args.dataset_name}")
        try:
            datasets = load_dataset(args.dataset_name)
        except Exception as e:
            logger.error(f"Erreur lors du chargement du dataset {args.dataset_name}: {e}")
            raise
            
        # Assurez-vous que les clés sont 'train' et 'test'
        if "validation" in datasets and "test" not in datasets:
            datasets["test"] = datasets["validation"]
        elif "test" not in datasets:
            # Si aucun ensemble de test n'est disponible, créez-en un à partir de l'ensemble d'entraînement
            train_test = datasets["train"].train_test_split(test_size=args.validation_split, seed=args.seed)
            datasets["train"] = train_test["train"]
            datasets["test"] = train_test["test"]
    
    # Prétraitement des données
    logger.info("Prétraitement des données...")
    tokenized_datasets = preprocess_data(datasets, tokenizer, max_length=args.max_length)
    
    # Afficher quelques statistiques sur les données
    logger.info(f"Nombre d'exemples d'entraînement: {len(tokenized_datasets['train'])}")
    logger.info(f"Nombre d'exemples de test: {len(tokenized_datasets['test'])}")
    
    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        report_to="none",  # Désactiver les rapports Tensorboard/Wandb
        seed=args.seed,
    )
    
    # Initialisation du Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Lancement de l'entraînement
    logger.info("Début du fine-tuning...")
    trainer.train()
    
    # Évaluation finale
    logger.info("Évaluation du modèle...")
    eval_results = trainer.evaluate()
    for metric_name, value in eval_results.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Sauvegarde du modèle final
    final_model_path = os.path.join(args.output_dir, "final-model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Modèle final sauvegardé à: {final_model_path}")
    
    return final_model_path

def main():
    """Point d'entrée principal du script."""
    # Parse les arguments
    args = parse_args()
    
    # Fine-tune le modèle
    try:
        model_path = fine_tune_model(args)
        logger.info(f"Fine-tuning terminé avec succès. Modèle sauvegardé à: {model_path}")
        
        # Afficher comment utiliser le modèle
        logger.info("\nPour utiliser le modèle fine-tuné:")
        logger.info(f"  from transformers import pipeline")
        logger.info(f"  sentiment_analyzer = pipeline('sentiment-analysis', model='{model_path}')")
        logger.info(f"  result = sentiment_analyzer('Votre texte à analyser')")
        logger.info(f"  print(result)")
        
    except Exception as e:
        logger.error(f"Une erreur s'est produite pendant le fine-tuning: {e}")
        raise

if __name__ == "__main__":
    main()
