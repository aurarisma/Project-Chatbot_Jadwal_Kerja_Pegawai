import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import nltk
from torch.utils.data import Dataset, DataLoader
import json
import os
import re
from nltk.stem.porter import PorterStemmer

# Unduh punkt tokenizer jika belum ada
nltk.download('punkt')

# Inisialisasi stemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Load dataset intent
df = pd.read_csv("dataset_intent_jadwal_kerja.csv")

all_words = []
tags = []
xy = []

# Proses setiap baris
for index, row in df.iterrows():
    tag = str(row['intent'])
    text = str(row['text'])
    tokens = tokenize(text)
    tokens = [stem(w) for w in tokens if w not in ['?', '.', ',', '!']]
    all_words.extend(tokens)
    xy.append((tokens, tag))
    if tag not in tags:
        tags.append(tag)

all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Dataset PyTorch
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(xy)
        self.x_data = [bag_of_words(tokens, all_words) for (tokens, _) in xy]
        self.y_data = [tags.index(tag) for (_, tag) in xy]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Hyperparameters
batch_size = 8
hidden_size = 8
input_size = len(all_words)
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = torch.tensor(words).to(device)
        labels = torch.tensor(labels).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Simpan model
model_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(model_data, FILE)

print(f'Training selesai. Model disimpan ke {FILE}')
