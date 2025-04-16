from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Cargar los datos
digits = load_digits()
X = digits.data      # variables (64 píxeles por imagen)
y = digits.target    # clases (dígitos 0-9)

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Crear y entrenar el árbol de decisión
clf = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limito profundidad para visualizar
clf.fit(X_train, y_train)

# 4. Visualizar el árbol
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, class_names=[str(i) for i in range(10)], feature_names=[f'pixel_{i}' for i in range(64)])
plt.title("Árbol de Decisión para clasificación de dígitos")
plt.show()
