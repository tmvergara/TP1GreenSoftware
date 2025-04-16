from datetime import time

from codecarbon import EmissionsTracker
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import numpy as np
import time

# Iniciar el medidor de emisiones
tracker = EmissionsTracker()
tracker.start()

# Cargar el dataset MNIST
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Flatten (de 28x28 → 784)
X_train_full = X_train_full.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Usamos solo una parte para hacerlo más rápido (opcional)
X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=0.2, random_state=42)

# Entrenar un MLP (red neuronal simple)
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20, verbose=True, random_state=42)
start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()
print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos.")

# Evaluar el modelo
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Parar el tracker y mostrar emisiones
emissions = tracker.stop()
print(f"\nEmisiones estimadas: {emissions:.10f} kgCO2eq")
