import torch

data = torch.load("data.pth")
print("Isi keys:", data.keys())

print("Jumlah kata unik:", len(data["all_words"]))
print("Semua tag:", data["tags"])
print("Ukuran input:", data["input_size"])
print("Ukuran hidden:", data["hidden_size"])
print("Ukuran output:", data["output_size"])