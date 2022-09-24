import cv2
import matplotlib.pyplot as plt


my_image = 'lenna.png'

image = cv2.imread(my_image)
# me lo guarda como array
# print(image)
# print(type(image))
# print(image.shape)
# print(image.max())
# print(image.min())

# se muestra diferente pq openCV no es RGB sino BGR
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

# cambiamos a RGB
new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(new_image)
plt.show()

# save as
# cv2.imwrite('lenna.jpg', image)

# convertir a GRAYSCALE
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(image_gray.shape)
plt.figure(figsize=(10, 10))
# debemos especifica que el color map es gris
plt.imshow(image_gray, cmap='gray')
plt.show()
cv2.imwrite('lenna_gray.jpg', image_gray)

# se puede cargar directamente en escala de grises
im_gray = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10,10))
plt.imshow(im_gray,cmap='gray')
plt.show()

# jugamos con los canales de colores
baboon = cv2.imread('baboon.png')
baboon_rgb = cv2.cvtColor(baboon, cv2. COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(baboon_rgb)
plt.show()

# obtenemos los diferentes colores BGR
blue, green, red = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]
im_bgr = cv2.vconcat([blue, green, red])
plt.figure(figsize=(10,10))
plt.imshow(im_bgr, cmap='gray')
plt.title('Blue (top), Green (middle), Red (bottom)')
plt.show()

# indexamos
# mostramos las primeras 256 filas
rows = 256
plt.figure(figsize=(10,10))
plt.imshow(new_image[0:rows,:,:])
plt.show()

# mostramos las primeras 256 columnas
columns = 256
plt.figure(figsize=(10,10))
plt.imshow(new_image[:,0:columns,:])
plt.show()

# si queremos alterar el array hacemos copy para mantener intacto el original
# A = new_image.copy()
# plt.imshow(A)
# plt.show()

# manipulamos los elementos mediante la indexacion
baboon_red = baboon_rgb.copy()
baboon_red[:,:,1] = 0 # verde = 0
baboon_red[:, :, 2] = 0 # azul = 0
plt.figure(figsize=(10, 10))
plt.imshow(baboon_red)
plt.show()
# se puede hacer lo mismo con el azul y con el verde

# tarea: convertir baboon.png a RGB y quitar el canal azul
baboon_sin_azul = cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB)
baboon_sin_azul[:, :, 2] = 0
plt.figure(figsize=(10, 10))
plt.imshow(baboon_sin_azul)
plt.show()