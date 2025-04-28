import numpy as np
# inputs
temperature = 20
humidity = 60

x = np.array([temperature, humidity])

#nöron #weights
weights = np.array([0.4, 0.6])

# eşik değer
bias = -20

#noron çıktısı
output = np.dot(x, weights) + bias # ağırlıklar ve girişlerin çarpımı + bias
print("Nöronun ham çıktısı:", output)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

activated_output = sigmoid(output) # sigmoid aktivasyon fonksiyonu
print("Aktive edilmiş çıktı:", activated_output)