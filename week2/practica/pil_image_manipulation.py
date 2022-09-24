from turtle import width
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np


baboon = np.array(Image.open('baboon.png'))
# plt.figure(figsize=(5,5))
# plt.imshow(baboon)
# plt.show()

# demostramos q al hacer modificaciones en una imagen, no se altera la copia
# B = baboon.copy()
# baboon[:,:,] = 0
# plt.figure(figsize=(10,10))
# plt.subplot(121)
# plt.imshow(baboon)
# plt.title("baboon")
# plt.subplot(122)
# plt.imshow(B)
# plt.title("array B")
# plt.show()

# flipping images
image = Image.open('lenna.png')
array = np.array(image)
width, height, C = array.shape
print('width, height, C', width, height, C)

# empecemos a dar vuelta la imagen
# primero creamos un array del tipo np.uint8 con el mismo tamaño
array_flip = np.zeros((width, height, C), dtype=np.uint8)
# asignamos la primera fila de pixeles a la ultima fila del nuevo array
# despues seguimos con las otras filas
for i,row in enumerate(array):
    array_flip[width - 1 - i, :, :] = row
# plt.figure(figsize=(5,5))
# plt.imshow(array_flip)
# plt.show()

# PIL tiene otras formas mas simples para dar vuelta la imagen
# en este caso con ImageOps lo hacemos verticalmente
im_flip = ImageOps.flip(image)
plt.figure(figsize=(5,5))
plt.imshow(im_flip)
plt.show()

# podemos darla vuelta horizontalmente
im_mirror = ImageOps.mirror(image)
plt.figure(figsize=(5,5))
plt.imshow(im_mirror)
plt.show()

# podemos hacer el flip mediante valores int
# 0 = "FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
# 1 ="FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
# 2 = "ROTATE_90": Image.ROTATE_90,
# 3 = "ROTATE_180": Image.ROTATE_180,
# 4 = "ROTATE_270": Image.ROTATE_270,
# 5 = "TRANSPOSE": Image.TRANSPOSE, 
# 6 = "TRANSVERSE": Image.TRANSVERSE
im_transpose = image.transpose(2) # == image.transpose(Image.ROTATE_90)
plt.imshow(im_transpose)
plt.show()

# ploteamos todos comparandolo con la original
# flip = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
#         "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
#         "ROTATE_90": Image.ROTATE_90,
#         "ROTATE_180": Image.ROTATE_180,
#         "ROTATE_270": Image.ROTATE_270,
#         "TRANSPOSE": Image.TRANSPOSE, 
#         "TRANSVERSE": Image.TRANSVERSE}

# for key, values in flip.items():
#     plt.figure(figsize=(10,10))
#     plt.subplot(1,2,1)
#     plt.imshow(image)
#     plt.title("orignal")
#     plt.subplot(1,2,2)
#     plt.imshow(image.transpose(values))
#     plt.title(key)
#     plt.show()

# CROPPING
# para cortar la imagen nececitamos definir 2 variables
# para cortar verticalmente:
# upper es el indice de la primer fila que queremos incluir en la imagen
# lower es el indice de la ultima fila que queremos incluir
upper = 150 # fila 150 de la imagen
lower = 400 # fila 400 de la imagen
crop_top = array[upper: lower,:,:]
plt.figure(figsize=(5,5))
plt.imshow(crop_top)
plt.show()

# para cortar horizontalmente:
# right es el indice de la primer columna que queremos incluir
# left es el indice de la ultima columna que queremos incluir
left = 150
right = 400
crop_horizontal = array[: ,left:right,:]
plt.figure(figsize=(5,5))
plt.imshow(crop_horizontal)
plt.show()

# tmb se puede cortar usando el metodo crop() de PIL
image = Image.open("baboon.png")
crop_image = image.crop((left, upper, right, lower))
plt.figure(figsize=(5,5))
plt.imshow(crop_image)
plt.show()

# CAMBIANDO PIXELES ESPECIFICOS
array_sq = np.copy(array)
# en un rango definido por upper:lower y left:right seteamos los canales verde y azul a 0
array_sq[upper:lower, left:right, 1:2] = 0
plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
plt.imshow(array)
plt.title("orignal")
plt.subplot(1,2,2)
plt.imshow(array_sq)
plt.title("Altered Image")
plt.show()

# dibujamos sobre la imagen con ImageDraw
image_draw = image.copy()
# definimos un constructor que crea un objeto para dibujar en la imagen definida in im=
image_fn = ImageDraw.Draw(im=image_draw)
# cualquier metodo que apliquemos sobre image_fn cambiara image_draw
# ahora dibujamos el rectangulo definido por upper, lower, right, left
# primero definimos el rectangulo
shape = [left, upper, right, lower]
# lo dibujamos sobre image_fn y le definimos un color
image_fn.rectangle(xy=shape,fill="red")
plt.figure(figsize=(5,5))
plt.imshow(image_draw)
plt.show()

# podemos usar otrs formas ademas de rectangulos
# como, por ejemplo, alguna fuente con ImageFont
image_fn.text(xy=(0,0),text="box",fill=(255,255,255))
plt.figure(figsize=(5,5))
plt.imshow(image_draw)
plt.show()

# podemos superponer una imagen sobre otra reasignando los pixeles de una a la otra
array_baboon = np.array(image)
array_baboon[upper:lower,left:right,:]=array[upper:lower,left:right,:]
plt.imshow(array_baboon)
plt.show()

# se puede hacer los mismo con el metodo paste()
image_lenna = Image.open("lenna.png")
image_lenna.paste(crop_image, box=(left,upper))
plt.imshow(image_lenna)
plt.show()

# tarea: hacer flip y mirror a una imagen
im = Image.open('baboon.png')
im_flip = ImageOps.flip(im)
im_mirror = ImageOps.mirror(im)
plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
plt.imshow(im_flip)
plt.title("Fliped")
plt.subplot(1,2,2)
plt.imshow(im_mirror)
plt.title("Mirrored")
plt.show()
