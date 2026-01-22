# Import des packages nécessaires
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# -------------------------------
# Étape 1 : Chargement du jeu de données
# -------------------------------
ds, info = tfds.load('ag_news_subset', with_info=True, as_supervised=True)
train_ds, test_ds = ds['train'], ds['test']

# -------------------------------
# Étape 2 : Prétraitement et tokenisation
# -------------------------------
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_batch(texts, labels):
    tokens = tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    tokens["labels"] = labels
    return tokens

def tfds_to_torch(dataset):
    texts, labels = [], []
    for text, label in tfds.as_numpy(dataset):
        texts.append(text.decode())
        labels.append(label)
    return tokenize_batch(texts, torch.tensor(labels))

train_encodings = tfds_to_torch(train_ds)
test_encodings = tfds_to_torch(test_ds)

# -------------------------------
# Étape 3 : Création d'un Dataset PyTorch personnalisé
# -------------------------------
class AGNewsDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

train_dataset = AGNewsDataset(train_encodings)
test_dataset = AGNewsDataset(test_encodings)

# -------------------------------
# Étape 4 : Chargement du modèle pré-entraîné pour classification
# -------------------------------
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=4)

# -------------------------------
# Étape 5 : Configuration de l'entraînement avec Trainer
# -------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    do_train=True,
    do_eval=True,
    eval_steps=100,
    save_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# -------------------------------
# Étape 6 : Entraînement du modèle
# -------------------------------
trainer.train()

# -------------------------------
# Étape 7 : Sauvegarde du modèle et du tokenizer
# -------------------------------
model.save_pretrained("./fine-tuned-bert-agnews")
tokenizer.save_pretrained("./fine-tuned-bert-agnews")
