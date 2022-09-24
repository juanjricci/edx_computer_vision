import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np


# funcion para plotear 2 imgenes juntas
def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()

# GEOMETRIC TRANSFORMATIONS
# example: resize, shift, reshape, rotate
image = Image.open("images/lenna.jpg")
# plt.imshow(image)
# plt.show()

# escalamos: duplicamos el eje horizontal
# usamos la funcion resize()
width, height = image.size
new_width = 2 * width
new_image = image.resize((new_width, height))
# plt.imshow(new_image)
# plt.show()

# ploteamos ambas imagenes juntas
plot_image(image, new_image, 'Original', 'Scaled horizontally')

# ahora escalamos el eje vertical
new_height = 2 * height
new_image = image.resize((width, new_height))
plot_image(image, new_image, 'Original', 'Scaled vertically')

# escalamos tanto vertical como horizontal
new_image = image.resize((new_width, new_height))
plot_image(image, new_image, 'Original', 'Scaled')

# si queremos achicar la imagen tenemos q dividir
new_width = width // 2
new_height = height // 2
new_image = image.resize((new_width, new_height))
plot_image(image, new_image, 'Original', 'Shrinked')

# ROTATION
# usamos la funcion rotate(theta)
theta = 45
new_image = image.rotate(theta)
plot_image(image, new_image, 'Original', 'Rotated')

# OPERACIONES MATEMATICAS
# ARRAY OPERATIONS
# convertimos la imagen a un array
image = np.array(image)
# le sumamos una constante
# esto le suma esa consante a cada pixel de la imagen
# cambiando su intensidad
new_image = image + 20
plot_image(image, new_image, 'Original', 'Added constant')
# tambien podemos multiplicar los pixeles
new_image = image * 10
plot_image(image, new_image, 'Original', 'Multiplied constant')
# podemos tmb sumar un array del mismo tamano
# definimos una "imagen" con pixel de intensidad random
# esta imagen va a llamarse Noise pq es ruido para la imagen original
Noise = np.random.normal(0, 20, (height, width, 3)).astype(np.uint8)
new_image = image + Noise
plot_image(image, new_image, 'Original', 'Image + Noise')
# de la misa forma lo podemos multiplicar
new_image = image * Noise
plot_image(image, new_image, 'Original', 'Image * Noise')

# OPERACIONES DE MATRIZ
# Las imagenes en escala de grises son matrices.
im_gray = Image.open("images/barbara.png")
im_gray = ImageOps.grayscale(im_gray)
# convertimos a array
im_gray = np.array(im_gray)
plt.imshow(im_gray,cmap='gray')
plt.show()
# Singular Value Descomposition
# descompone la matriz en un producto de 3 matrices
U, s, V = np.linalg.svd(im_gray , full_matrices=True)
# print(s.shape)
# vemos q s no es rectangular, entonces lo convertimos en una matriz diagonal
S = np.zeros((im_gray.shape[0], im_gray.shape[1]))
S[:im_gray.shape[0], :im_gray.shape[0]] = np.diag(s)
# ploteamos las matrices U y V
plot_image(U, V, title_1="Matrix U", title_2="Matrix V")
plt.imshow(S, cmap='gray')
plt.show()
# buscamos el resultado del produco de las 3 matrices
# primero multiplicamos S y U y lo asignamo a B
B = S.dot(V)
plt.imshow(B,cmap='gray')
plt.show()
# despues si multiplicamos U, S y B veremos toda la imagen
A = U.dot(B)
plt.imshow(A,cmap='gray')
plt.show()
# resulta q muchos elementos son redundantes
# podemos eliminar algunas filas y columnas de Y y de V
# y aproximarnos a la imagen encontrando el producto
for n_component in [1,10,100,200,500]:
    S_new = S[:, :n_component]
    V_new = V[:n_component, :]
    A = U.dot(S_new.dot(V_new))
    plt.imshow(A,cmap='gray')
    plt.title("Number of Components:"+str(n_component))
    plt.show()
# podemos ver que con solo 200 componentes ya podemos representar la imagen
