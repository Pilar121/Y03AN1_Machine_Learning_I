from sklearn.datasets import make_classification
import pandas as pd

# Crear datos sintéticos
X, y = make_classification(
    n_samples=5000,
    n_features=17,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    weights=[0.595, 0.405],  # 60% no moroso, 40% moroso
    random_state=42
)

# Nombres de columnas
columnas = [f"feature_{i+1}" for i in range(X.shape[1])]

# Crear DataFrame
df = pd.DataFrame(X, columns=columnas)
df["moroso"] = y

# Guardar CSV
df.to_csv("datos_sinteticos.csv", index=False)

# Mensaje de confirmación
print("Dataset sintético generado con éxito en: datos_sinteticos.csv")
