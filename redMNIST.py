import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import time

tracker = EmissionsTracker() # inicia tracker
tracker.start()

digits = load_digits()
X = digits.data  # c/u un array de 64 píxeles
y = digits.target

# 3. Visualizar 10 imágenes del dataset
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Etiqueta: {digits.target[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

print("Entrenando el modelo...")
start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()
print(f"Precisión en entrenamiento: {clf.score(X_train, y_train):.4f}")
print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos.")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy del modelo: {accuracy:.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

emissions = tracker.stop()
print(f"\n🌱 Emisiones estimadas: {emissions:.10f} kgCO2eq")

# 0.0000000016 kgCO2eq
# 0.0000014489 kgCO2eq
