import numpy as np

# Datos de ejemplo: matriz 3x2
data = np.array([[1, 2],
                 [3, 4],
                 [5, 6]])

# Aplicar la descomposición SVD
U, S, Vt = np.linalg.svd(data)

# Reducir la dimensionalidad manteniendo solo la primera columna de Vt
n_components = 1
reduced_data = np.dot(data, Vt[:n_components].T)

print("Datos originales:")
print(data)
print("\nDatos reducidos:")
print(reduced_data)

# #Numpy para operaciones matematicas y matplotlib para visualizacion de datos y PIL para trabajar con imagenes 
# import numpy as np 
# import matplotlib.pyplot as plt
# from PIL import Image

# #Cargar la imagen de  mi escritorio
# imagen = Image.open(r'C:\Users\KEVII\Desktop\Captura de pantalla 2022-07-22 124605.png')  # Reemplaza 'imagen.jpg' con la ruta de tu imagen

# # Convertir la imagen a una matriz NumPy
# matrizdeimagen = np.array(imagen)

# # Obtener las dimensiones de la imagen (si es una imagen a color)
# alto, ancho, canales = matrizdeimagen.shape

# # Aplanar la imagen en una matriz 2D para poder aplicar SVD
# ImagenAplanada = matrizdeimagen.reshape(alto, ancho * canales)

# # Calcular la SVD de la imagen aplanada
# U, S, Vt = np.linalg.svd(ImagenAplanada, full_matrices=False)

# # Reducir la dimensionalidad manteniendo solo los componentes principales más significativos
# n_componentes = 100  # Número de componentes principales a mantener
# imagen_reducida = np.dot(U[:, :n_componentes], np.diag(S[:n_componentes])).dot(Vt[:n_componentes, :])

# # Convertir la imagen reducida a un formato adecuado para visualización
# imagen_reducida = imagen_reducida.reshape(alto, ancho, canales)
# imagen_reducida = np.clip(imagen_reducida, 0, 255).astype(np.uint8)

# # Visualizar la imagen original y la imagen reducida
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(matrizdeimagen)
# plt.title('Imagen Original')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(imagen_reducida)
# plt.title('Imagen Reducida')
# plt.axis('off')

# plt.show()
