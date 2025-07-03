# === FILE: chatbot_app.py ===
# Chatbot Jadwal Kerja dengan Streamlit

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import json
import re
import os
import random

# Pastikan tokenizer punkt tersedia
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Fungsi tokenize dan stem
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

# Cek dan Load model
if not os.path.exists("data.pth"):
    st.error("File 'data.pth' tidak ditemukan.")
    st.stop()

data = torch.load("data.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

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

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Load dataset jadwal kerja
if not os.path.exists("dataset_jadwal_kerja.csv"):
    st.error("File 'dataset_jadwal_kerja.csv' tidak ditemukan.")
    st.stop()

jadwal_df = pd.read_csv("dataset_jadwal_kerja.csv")

# Fungsi respons chatbot
def get_response(intent, entity):
    if intent == "jadwal_hari_ini":
        hari_ini = pd.Timestamp.today().day_name()
        result = jadwal_df[jadwal_df['hari'].str.lower() == hari_ini.lower()]
    elif intent == "jadwal_hari_tertentu" and entity.get("hari"):
        result = jadwal_df[jadwal_df['hari'].str.lower() == entity["hari"].lower()]
    elif intent == "kerja_berdasarkan_lokasi" and entity.get("lokasi"):
        result = jadwal_df[jadwal_df['lokasi'].str.lower() == entity["lokasi"].lower()]
    elif intent == "kerja_per_shift" and entity.get("shift"):
        result = jadwal_df[jadwal_df['shift'].str.lower() == entity["shift"].lower()]
    elif intent == "jadwal_pegawai" and entity.get("pegawai"):
        result = jadwal_df[jadwal_df['pegawai'].str.lower().str.contains(entity["pegawai"].lower())]
    else:
        return "Maaf, saya belum dapat menemukan informasi yang sesuai."

    if not result.empty:
        list_kerja = result.apply(
            lambda row: f"{row['hari']}: {row['pegawai']} (Jam {row['jam_mulai']}–{row['jam_selesai']}, Lokasi: {row['lokasi']})",
            axis=1
        ).tolist()
        return "Berikut jadwal kerja yang ditemukan:\n- " + "\n- ".join(list_kerja)
    else:
        return "Tidak ditemukan jadwal kerja yang sesuai."

# Ekstraksi entitas dari pertanyaan
def extract_entity(text):
    hari_list = ["senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"]
    shift_list = ["pagi", "siang", "malam"]
    entity = {}

    for h in hari_list:
        if h in text.lower():
            entity["hari"] = h
            break
    for s in shift_list:
        if s in text.lower():
            entity["shift"] = s
            break

    lokasi = jadwal_df["lokasi"].dropna().unique()
    for l in lokasi:
        if l.lower() in text.lower():
            entity["lokasi"] = l
            break

    pegawai_list = jadwal_df["pegawai"].dropna().unique()
    for p in pegawai_list:
        if p.lower().split()[0] in text.lower():
            entity["pegawai"] = p
            break

    return entity

# Prediksi intent
def predict_class(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.40:
        return tag
    else:
        return "unknown"

# Streamlit antarmuka
st.title("🧑‍💼 Chatbot Jadwal Kerja Pegawai")
st.markdown("Tanyakan tentang jadwal kerja berdasarkan hari, shift, lokasi, atau pegawai.")

user_input = st.text_input("Ketik pertanyaan kamu:", "Siapa yang kerja hari Senin?")

if st.button("Tanya"):
    intent = predict_class(user_input)
    entity = extract_entity(user_input)
    response = get_response(intent, entity)
    st.write(response)
