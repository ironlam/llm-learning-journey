from transformers import pipeline
# Pipeline pour la reconnaissance d'entités nommées
ner_pipeline = pipeline(
    "ner",
    model="Jean-Baptiste/camembert-ner",
    aggregation_strategy="simple"  # regroupe les tokens appartenant à la même entité
)
# Exemple de phrase
text = "Le président Emmanuel Macron s'est rendu à Marseille pour discuter avec Orange et SFR."
results = ner_pipeline(text)
cleaned_results = [
    {**r, "score": float(r["score"])}  # déstructure le dict et convertit score pour éviter le résultat en np.float32
    for r in results
]

print(cleaned_results)