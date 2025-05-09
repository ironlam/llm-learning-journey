#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application Gradio pour déployer un modèle d'analyse de sentiment fine-tuné.

Cette application permet aux utilisateurs d'entrer un texte et d'obtenir
une analyse de sentiment en temps réel, avec visualisation des résultats.
"""

import os
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

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
    parser = argparse.ArgumentParser(description="Déployer un modèle d'analyse de sentiment avec Gradio")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./sentiment-model/final-model",
        help="Chemin vers le modèle fine-tuné",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Partager l'application publiquement avec un lien temporaire",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="default",
        choices=["default", "huggingface", "grass", "peach"],
        help="Thème de l'interface Gradio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port sur lequel l'application sera exécutée",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activer le mode débogage pour plus de détails",
    )
    return parser.parse_args()

def load_model(model_path: str) -> Optional[Any]:
    """
    Charge le modèle et le tokenizer, puis prépare le pipeline d'analyse de sentiment.
    
    Args:
        model_path (str): Chemin vers le modèle fine-tuné
        
    Returns:
        Optional[Any]: Le pipeline d'analyse de sentiment ou None en cas d'erreur
    """
    logger.info(f"Chargement du modèle depuis {model_path}")
    # Vérifier si le chemin du modèle existe
    if not os.path.exists(model_path):
        logger.warning(f"Le chemin du modèle {model_path} n'existe pas")
        logger.info("Utilisation d'un modèle par défaut de Hugging Face...")
        try:
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                return_all_scores=True
            )
            return sentiment_analyzer
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle par défaut: {e}")
            return None
    
    try:
        # Charger le tokenizer et le modèle
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Créer le pipeline
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True
        )
        
        logger.info("Modèle chargé avec succès")
        return sentiment_analyzer
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        logger.info("Tentative de chargement d'un modèle par défaut...")
        try:
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                return_all_scores=True
            )
            return sentiment_analyzer
        except Exception as e2:
            logger.error(f"Erreur lors du chargement du modèle par défaut: {e2}")
            return None

def analyze_sentiment(text: str, sentiment_analyzer: Any) -> Tuple[Dict[str, float], plt.Figure]:
    """
    Analyse le sentiment du texte et retourne les scores et une visualisation.

    Args:
        text (str): Texte à analyser
        sentiment_analyzer (Any): Pipeline d'analyse de sentiment

    Returns:
        Tuple[Dict[str, float], plt.Figure]: Les scores de sentiment et la visualisation
    """
    if not text.strip():
        return {"Message": "Veuillez entrer un texte pour l'analyse."}, None

    if sentiment_analyzer is None:
        return {"Erreur": "Le modèle n'a pas pu être chargé. Veuillez réessayer."}, None

    # Analyser le sentiment
    try:
        result = sentiment_analyzer(text)

        # Correspondance entre les labels techniques et labels conviviaux
        label_mapping = {
            "LABEL_0": "NÉGATIF",
            "LABEL_1": "POSITIF",
        }

        # Formater les résultats pour l'affichage avec des labels conviviaux
        if isinstance(result[0], list):
            # Si le modèle retourne tous les scores
            scores = {}
            for item in result[0]:
                # Convertir le label si possible
                friendly_label = label_mapping.get(item["label"], item["label"])
                scores[friendly_label] = float(item["score"])
        else:
            # Si le modèle retourne seulement le meilleur score
            technical_label = result[0]["label"]
            friendly_label = label_mapping.get(technical_label, technical_label)
            scores = {friendly_label: float(result[0]["score"])}

        # Créer une visualisation avec matplotlib
        labels = list(scores.keys())
        values = list(scores.values())

        # Déterminer la couleur en fonction du sentiment
        colors = []
        for label in labels:
            if label == "POSITIF" or "POSITIVE" in label:
                colors.append("#4CAF50")  # Vert pour positif
            elif label == "NÉGATIF" or "NEGATIVE" in label:
                colors.append("#F44336")  # Rouge pour négatif
            else:
                colors.append("#2196F3")  # Bleu pour neutre

        # Créer la visualisation avec matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color=colors)
        ax.set_title("Analyse de sentiment")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)

        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()

        return scores, fig
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du sentiment: {e}")
        return {"Erreur": f"Une erreur s'est produite: {str(e)}"}, None

def analyze_batch(file: gr.File, sentiment_analyzer: Any) -> pd.DataFrame:
    """
    Analyse un lot de textes à partir d'un fichier CSV.

    Args:
        file (gr.File): Fichier CSV contenant les textes à analyser
        sentiment_analyzer (Any): Pipeline d'analyse de sentiment

    Returns:
        pd.DataFrame: DataFrame contenant les résultats de l'analyse
    """
    try:
        # Lire le fichier CSV
        df = pd.read_csv(file.name)
        if "text" not in df.columns:
            return pd.DataFrame({"Erreur": ["Le fichier CSV doit contenir une colonne 'text'"]})

        # Correspondance entre les labels techniques et labels conviviaux
        label_mapping = {
            "LABEL_0": "NÉGATIF",
            "LABEL_1": "POSITIF",
            # Ajouter d'autres correspondances si nécessaire
        }

        # Analyser les sentiments
        results = []
        for text in df["text"]:
            if not isinstance(text, str):
                results.append({"texte": str(text), "erreur": "Le texte n'est pas une chaîne de caractères"})
                continue

            try:
                result = sentiment_analyzer(text)[0]
                if isinstance(result, list):
                    result = result[0]

                # Convertir le label technique en label convivial
                technical_label = result["label"]
                friendly_label = label_mapping.get(technical_label, technical_label)

                results.append({
                    "texte": text[:50] + "..." if len(text) > 50 else text,
                    "sentiment": friendly_label,
                    "score": result["score"]
                })
            except Exception as e:
                results.append({"texte": text[:50] + "...", "erreur": str(e)})

        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse par lot: {e}")
        return pd.DataFrame({"Erreur": [str(e)]})

def create_examples() -> List[List[str]]:
    """
    Crée des exemples pour l'interface Gradio.
    
    Returns:
        List[List[str]]: Liste d'exemples
    """
    return [
        ["Cet article est vraiment informatif et bien écrit. J'ai beaucoup appris!"],
        ["Je suis déçu par la qualité de ce contenu, il manque de profondeur."],
        ["L'article aborde des points intéressants mais aurait pu être plus développé."],
        ["Cette analyse est totalement biaisée et ne présente qu'un seul point de vue."],
        ["Incroyable article, précis et agréable à lire. Je le recommande vivement."]
    ]

def create_app(sentiment_analyzer: Any, theme: str = "default") -> gr.Blocks:
    """
    Crée l'interface utilisateur Gradio.
    
    Args:
        sentiment_analyzer (Any): Pipeline d'analyse de sentiment
        theme (str, optional): Thème de l'interface. Par défaut "default".
        
    Returns:
        gr.Blocks: L'application Gradio
    """
    with gr.Blocks(title="Analyseur de Sentiment pour Commentaires d'Articles", theme=theme) as app:
        gr.Markdown(
            """
            # 📊 Analyseur de Sentiment pour Commentaires d'Articles
            
            Cet outil analyse le sentiment des commentaires ou textes que vous saisissez.
            Entrez simplement votre texte ci-dessous et découvrez son sentiment prédominant!
            
            *Modèle fine-tuné dans le cadre d'un atelier d'apprentissage LLM.*
            """
        )
        
        with gr.Tabs():
            with gr.TabItem("Analyse individuelle"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Texte à analyser",
                            placeholder="Entrez un commentaire ou un texte à analyser...",
                            lines=5
                        )
                        analyze_btn = gr.Button("Analyser le sentiment", variant="primary")
                        
                    with gr.Column():
                        sentiment_scores = gr.JSON(label="Scores de sentiment")
                        sentiment_plot = gr.Plot(label="Visualisation")
                
                # Ajouter des exemples
                gr.Examples(
                    examples=create_examples(),
                    inputs=text_input,
                )
                
                # Définir le comportement lors du clic sur le bouton
                analyze_btn.click(
                    fn=lambda text: analyze_sentiment(text, sentiment_analyzer),
                    inputs=text_input,
                    outputs=[sentiment_scores, sentiment_plot],
                )
                
                # Permettre l'analyse également lorsque l'utilisateur appuie sur Entrée
                text_input.submit(
                    fn=lambda text: analyze_sentiment(text, sentiment_analyzer),
                    inputs=text_input,
                    outputs=[sentiment_scores, sentiment_plot],
                )
            
            with gr.TabItem("Analyse par lot"):
                gr.Markdown(
                    """
                    ## Analyse de plusieurs textes
                    
                    Téléchargez un fichier CSV contenant une colonne 'text' avec les textes à analyser.
                    """
                )
                file_input = gr.File(label="Fichier CSV", file_types=[".csv"])
                analyze_batch_btn = gr.Button("Analyser le lot", variant="primary")
                batch_results = gr.DataFrame(label="Résultats de l'analyse par lot")
                
                analyze_batch_btn.click(
                    fn=lambda file: analyze_batch(file, sentiment_analyzer),
                    inputs=file_input,
                    outputs=batch_results,
                )
        
        gr.Markdown(
            """
            ## 📝 À propos
            
            Cette application utilise un modèle de langage fine-tuné spécifiquement pour l'analyse 
            de sentiment des commentaires d'articles. Elle fait partie de mon parcours d'apprentissage 
            dans le domaine des LLM.
            
            ### Comment ça marche?
            
            Le modèle analyse le texte et attribue des scores de probabilité pour différents sentiments 
            (positif/négatif). Plus le score est élevé pour une catégorie, plus le modèle est confiant 
            dans sa prédiction.
            
            [GitHub](https://github.com/ironlam/llm-learning-journey) | [Medium](https://medium.com/@diaby.lamine)
            """
        )
    
    return app

def main():
    """Point d'entrée principal du script."""
    # Parser les arguments
    args = parse_args()
    
    # Configurer le niveau de logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Mode débogage activé")
    
    # Charger le modèle
    sentiment_analyzer = load_model(args.model_path)
    if sentiment_analyzer is None:
        logger.error("Impossible de charger un modèle. L'application s'arrête.")
        return
    
    # Créer et lancer l'application
    logger.info("Création de l'interface Gradio...")
    app = create_app(sentiment_analyzer, theme=args.theme)
    
    logger.info(f"Lancement de l'application sur le port {args.port}...")
    app.launch(
        server_name="0.0.0.0",  # Rend l'app accessible depuis n'importe quelle IP
        server_port=args.port,
        share=args.share,
        debug=args.debug,
    )
    logger.info("Application arrêtée.")

if __name__ == "__main__":
    main()
