import cv2
import numpy as np

# Lee la imagen
image = cv2.imread(r'C:\Users\KEVII\Desktop\Captura de pantalla 2022-07-22 124605.png')

# Definir la matriz de transformación lineal (escalar y sesgar)
matriz_transformacion = np.array([[1.2, 0, 50], 
                                  [0, 1.2, 50]])

# Aplicar la transformación lineal utilizando la función warpAffine de OpenCV
imagen_transformada = cv2.warpAffine(image, matriz_transformacion, (image.shape[1], image.shape[0]))

# Mostrar la imagen original y la imagen transformada
cv2.namedWindow('Imagen Original', cv2.WINDOW_NORMAL)
cv2.imshow('Imagen Original', image)

# Redimensiona la imagen transformada al tamaño de la ventana
imagen_transformada_resized = cv2.resize(imagen_transformada, (image.shape[1], image.shape[0]))

cv2.namedWindow('Imagen Transformada', cv2.WINDOW_NORMAL)
cv2.imshow('Imagen Transformada', imagen_transformada_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
