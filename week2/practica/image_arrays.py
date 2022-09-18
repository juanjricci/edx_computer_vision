from array import array
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

my_image = 'lenna.png'
image = Image.open(my_image)
im_array = np.asarray(image)
# print(type(im_array))

# creo copia de la imagen en forma de array para no modificarlo directamente
array_copy = np.array(image)

# devuelvo tupla de dimensiones del array (filas, columnas. colores)
print(array_copy.shape)

# imprimo el array para ver los valores de intensidad
# print(array_copy)
# print(array_copy[0, 0])

# podemos encontrar la intensidad minima y maxima del array
print(array_copy.min())
print(array_copy.max())

# INDEXANDO
# ploteamos el array como imagen

#plt.figure(figsize=(10,10))
#plt.imshow(array_copy)
#plt.show()

# devolvemos las primeras 256 filas de la primera mitad de la imagen
# rows = 256
# plt.figure(figsize=(10,10))
# plt.imshow(array_copy[0:rows,:,:])
# plt.show()

# devolovemos las primeras 256 columnas
# columns = 256
# plt.figure(figsize=(10,10))
# plt.imshow(array_copy[:,0:columns,:])
# plt.show()

# si queremos reasignar el array a otra variable usamos copy()
A = array_copy.copy()
#plt.imshow(A)
#plt.show()

# TRABAJANDO CON COLORES
baboon = Image.open('baboon.png')
# baboon_array = np.array(baboon)
# plt.figure(figsize=(10,10))
# plt.imshow(baboon_array)
# plt.show()

# ploteamos el canal del rojo en gris
baboon_array = np.array(baboon)
# plt.figure(figsize=(10,10))
# plt.imshow(baboon_array[:,:,0], cmap='gray')
# plt.show()

# o hacemos una array con todos los colores en 0 menos el rojo
baboon_red = baboon_array.copy()
baboon_red[:,:,1] = 0
baboon_red[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_red)
plt.show()

# se puede hacer lo mismo para los otros colores
baboon_blue = baboon_array.copy()
baboon_blue[:,:,0] = 0
baboon_blue[:,:,1] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_blue)
plt.show()

# tarea: quitar el canal azul de lenna.png
blue_lenna = Image.open('lenna.png')
blue_array = np.array(blue_lenna)
blue_array[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(blue_array)
plt.show()
