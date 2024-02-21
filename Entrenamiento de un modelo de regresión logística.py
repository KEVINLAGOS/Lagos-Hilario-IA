import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Cargar los datos desde el archivo CSV
data = pd.read_csv(r'C:\Users\KEVII\Desktop\IA\Lagos-Hilario-IA\spotify-2023.csv', encoding='latin1')

# Eliminar filas con valores faltantes si es necesario
data.dropna(inplace=True)

# Definir qué significa "novedoso"
data['novedoso'] = (data['energy_%'] > 70) & (data['acousticness_%'] < 30) & (data['instrumentalness_%'] < 20)

# Seleccionar las características relevantes y la variable objetivo
X = data[['energy_%', 'acousticness_%', 'instrumentalness_%']]
y = data['novedoso']

# Normalizar las características
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_scaled, y)

# Calcular la probabilidad de que una canción de un nuevo artista sea novedosa
nueva_cancion = [[100, 25, 15]]  # Características de la nueva canción
nueva_cancion_scaled = scaler.transform(nueva_cancion)
probabilidad_novedad = model.predict_proba(nueva_cancion_scaled)
print("Probabilidad de que la nueva cancion sea considerada novedosa:", probabilidad_novedad[0][1])
