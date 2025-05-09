# dataset_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger votre dataset
dataset_path = "../../02-fine-tuning/data/commentaires_articles.csv"
output_file = "../results/dataset_visualization.png"

# Charger les données
df = pd.read_csv(dataset_path)

# Vérifier que la colonne 'label' existe
if 'label' not in df.columns:
    print("Erreur: La colonne 'label' n'existe pas dans le dataset")
    exit(1)

# Créer une figure pour les visualisations
plt.figure(figsize=(15, 10))

# 1. Distribution des sentiments
plt.subplot(2, 2, 1)
sentiment_counts = df['label'].value_counts()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Distribution des sentiments dans le dataset')
plt.xlabel('Sentiment (0=Négatif, 1=Positif)')
plt.ylabel('Nombre d\'exemples')

# Ajouter les valeurs sur les barres
for i, count in enumerate(sentiment_counts.values):
    plt.text(i, count + 5, f"{count}", ha='center')

# 2. Distribution de la longueur des textes
plt.subplot(2, 2, 2)
df['text_length'] = df['text'].apply(len)
sns.histplot(data=df, x='text_length', hue='label', bins=30, kde=True)
plt.title('Distribution de la longueur des textes par sentiment')
plt.xlabel('Longueur du texte (caractères)')
plt.ylabel('Fréquence')

# 3. Longueur moyenne par sentiment
plt.subplot(2, 2, 3)
avg_length = df.groupby('label')['text_length'].mean()
sns.barplot(x=avg_length.index, y=avg_length.values)
plt.title('Longueur moyenne des textes par sentiment')
plt.xlabel('Sentiment (0=Négatif, 1=Positif)')
plt.ylabel('Longueur moyenne (caractères)')

# Ajouter les valeurs sur les barres
for i, avg in enumerate(avg_length.values):
    plt.text(i, avg + 5, f"{avg:.1f}", ha='center')

# 4. Répartition entre training et test (simulation de la division)
plt.subplot(2, 2, 4)
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
sizes = [train_size, test_size]
labels = ['Training (80%)', 'Test (20%)']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
plt.title('Répartition du dataset pour l\'entraînement')

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Visualisations du dataset sauvegardées dans '{output_file}'")