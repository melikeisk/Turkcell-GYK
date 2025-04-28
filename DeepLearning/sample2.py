import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 

# ages (inputs)
x = np.array([5, 6, 7, 8, 9, 10], dtype=float).reshape(-1, 1)

# heights (outputs)
y = np.array([110, 116, 123, 130, 136, 142], dtype=float).reshape(-1, 1)

# Ölçeklendiriciler (Scalers)
x_scaler = StandardScaler()
y_scaler = StandardScaler()

# Veriyi ölçeklendir
x_scaled = x_scaler.fit_transform(x)  
y_scaled = y_scaler.fit_transform(y)

# Modeli oluştur
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, input_shape=[1]),
    tf.keras.layers.Dense(units=1)
])

# Modeli derle
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğit
model.fit(x_scaled, y_scaled, epochs=500, verbose=0)

# Modeli test et
test_ages = np.array([[7.5]])
test_ages_scaled = x_scaler.transform(test_ages)  # Test verisini ölçeklendir
predicted_height_scaled = model.predict(test_ages_scaled)  # Tahmin yapılır
predicted_height = y_scaler.inverse_transform(predicted_height_scaled)  # Sonuçları orijinal ölçeğe geri döndür

print(f"Yaşı {test_ages[0][0]} olan çocuğun tahmin edilen boyu: {predicted_height[0][0]} cm")
