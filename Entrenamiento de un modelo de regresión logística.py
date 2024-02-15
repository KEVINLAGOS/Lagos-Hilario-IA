# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# valores de características
suscriptores = [1200, 800, 1500, 2000, 600]
duracion_promedio = [12, 8, 10, 15, 6]
frecuencia_publicacion = [2, 1, 3, 4, 1]

# Variable objetivo (0 para "no ver" y 1 para "ver")
ver_canal_youtube = [1, 0, 1, 1, 0]

# Crear un DataFrame con los valores y nombres de características
data = pd.DataFrame({
    'Suscriptores': suscriptores,
    'Duración Promedio': duracion_promedio,
    'Frecuencia de Publicación': frecuencia_publicacion,
    'Ver Canal YouTube': ver_canal_youtube
})

# Dividir  los datos en conjuntos de entrenamiento y prueba
X = data[['Suscriptores', 'Duración Promedio', 'Frecuencia de Publicación']]
y = data['Ver Canal YouTube']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear una instancia del modelo de regresión logística
model = LogisticRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')
# Ejemplo [1000 suscriptores, 10 minutos de duración promedio, 3 videos por semana]
new_channel = np.array([[1000, 10, 3]])
probabilities = model.predict_proba(new_channel)
print(f'Probabilidad de ver el nuevo canal: {probabilities[0][1]:.2f}')
