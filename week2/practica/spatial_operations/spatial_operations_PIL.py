import matplotlib.pyplot as plt
from PIL import Image
import numpy as np # vamos a usarlo para crear los kernels para el filtrado


def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2)
    plt.title(title_2)
    plt.show()

# LINEAR FILTERING
# esto puede ayudar a, por ejemplo, eliminar el ruido de una imagen
# Convultion es una forma estandar de hacerlo
image = Image.open("images/barbara.png")
# hacemos q la imagen tenga ruido para luego trabajarlo
# primero obtenemos el numero de filas y columnas
rows, cols = image.size
# creamos una imagen con ruido random y convertimos a uint8 para q los valores vayan de 0 a 255
noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
# sumamos el ruido a la imagen
noisy_image = image + noise
# creamos la imagen desde el array 
noisy_image = Image.fromarray(noisy_image)
plot_image(image, noisy_image, title_1="Orignal", title_2="Image Plus Noise")






