---
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
models:
  - ironlam/sentiment-analysis-french-model
title: sentiment-analysis-french-model
emoji: 📊
colorFrom: blue
colorTo: green
pinned: true
license: mit
install_requirements: true
---

# Analyseur de Sentiment pour Commentaires d'Articles

Cette application déploie un modèle fine-tuné pour l'analyse de sentiment des commentaires d'articles. Elle fait partie de mon parcours d'apprentissage dans le domaine des LLM.

## Utilisation

Entrez simplement votre texte dans la zone prévue et cliquez sur "Analyser le sentiment". L'application vous fournira une analyse en temps réel avec une visualisation des résultats.

## À propos du modèle

Ce modèle a été fine-tuné à partir de DistilBERT sur un dataset de commentaires d'articles pour détecter le sentiment exprimé (positif/négatif). Il fait partie de mon parcours d'apprentissage des modèles de langage (LLM).

## Ressources

- [Article Medium](https://medium.com/@diaby.lamine)
- [Code source sur GitHub](https://github.com/ironlam/llm-learning-journey)
- [Modèle sur Hugging Face](https://huggingface.co/ironlam/sentiment-analysis-model)

## Installation locale

```bash
pip install -r requirements.txt
python app.py
```

## Licence

MIT
