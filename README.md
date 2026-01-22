# FineTuneLLM-HF-

FineTuneLLM-HF est un guide complet et un pipeline pratique pour affiner des grands modèles de langage pré-entraînés (comme BERT) sur vos propres données spécifiques. En exploitant la bibliothèque Hugging Face Transformers et les jeux de données TensorFlow, ce projet montre comment préparer, tokeniser et formater vos données, créer des datasets PyTorch personnalisés, et utiliser l’API Trainer pour entraîner efficacement un modèle de classification de texte adapté à votre tâche. 
Ce workflow facilite la personnalisation des modèles de langage pour des applications concrètes telles que la classification d’avis clients, la catégorisation d’articles ou l’analyse de tickets de support.  
Résumé du projet (en quelques points) 
- Chargement des données : utilisation de TensorFlow Datasets (TFDS) pour récupérer un jeu de données textuel structuré (exemple : ag_news_subset). 
- Prétraitement : tokenisation des textes avec le tokenizer associé au modèle pré-entraîné (ex : bert-base-uncased), avec gestion du padding et de la troncature. 
- Conversion des données : transformation des datasets TensorFlow en tenseurs PyTorch compatibles avec Hugging Face. 
- Création d’un Dataset PyTorch personnalisé : encapsulation des données tokenisées pour faciliter l’entraînement. Chargement du modèle pré-entraîné : import d’un modèle BERT adapté à la classification multi-classes. 
- Configuration de l’entraînement : définition des paramètres via TrainingArguments (batch size, nombre d’époques, fréquence d’évaluation, etc.). 
- Entraînement avec Hugging Face Trainer : boucle d’entraînement simplifiée, gestion automatique de l’optimisation et de l’évaluation. 
- Sauvegarde du modèle et du tokenizer : pour réutilisation et déploiement ultérieurs.
