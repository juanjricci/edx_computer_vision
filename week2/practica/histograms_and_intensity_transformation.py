import matplotlib.pyplot as plt
import cv2
import numpy as np

# esto es solo para plotear dos imagenes una al lado de la otra
def plot_image(image_1, image_2,title_1="Orignal", title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()

# para plotear 2 histogramas uno al lado del otro
def plot_hist(old_image, new_image,title_old="Orignal", title_new="New Image"):
    intensity_values=np.array([x for x in range(256)])
    plt.subplot(1, 2, 1)
    plt.bar(intensity_values, cv2.calcHist([old_image],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(title_old)
    plt.xlabel('intensity')
    plt.subplot(1, 2, 2)
    plt.bar(intensity_values, cv2.calcHist([new_image],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(title_new)
    plt.xlabel('intensity')
    plt.show()

toy_image = np.array([[0,2,2],[1,1,1],[1,1,2]],dtype=np.uint8)
plt.imshow(toy_image, cmap="gray")
plt.show()
print("toy_image:",toy_image)

my_image = 'imagenes/goldhill.bmp'

# obtenemos la imagen en escala de grises
goldhill = cv2.imread(my_image,cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10,10))
plt.imshow(goldhill,cmap="gray")
plt.show()

# calculamos el histograma
hist = cv2.calcHist([goldhill],[0], None, [256], [0,256])

# imprimimos el histograma como grafico de barras
intensity_values = np.array([x for x in range(hist.shape[0])])
plt.bar(intensity_values, hist[:,0], width = 5)
plt.title("Bar histogram")
plt.show()

# lo podemos convertir en una funcion de probabilidad
PMF = hist / (goldhill.shape[0] * goldhill.shape[1])
plt.plot(intensity_values,hist)
plt.title("histogram")
plt.show()

# podemos aplicar un histograma para cada canal de color
baboon = cv2.imread("imagenes/baboon.png")
plt.imshow(cv2.cvtColor(baboon,cv2.COLOR_BGR2RGB))
plt.show()

color = ('blue','green','red')
for i,col in enumerate(color):
    histr = cv2.calcHist([baboon],[i],None,[256],[0,256])
    plt.plot(intensity_values,histr,color = col,label=col+" channel")
    plt.xlim([0,256])
plt.legend()
plt.title("Histogram Channels")
plt.show()

# podemos ver la imagen como una funcion f(x,y)
# donde x son las filas e y las columnas
# si hacemos g(x,y) = 2f(x,y) + 1 estariamos multiplicando
# cada pixel de la imagen por 2 y sumandole 1

# La transformacion de la intensidad depende solo de una variable
# esto es un mapeo en escala de grises,
# entonces podemos decir que la variable r = f(x,y) es la intensidad de gris
# entonces la transformacion seria s = T(r)

# IMAGEN EN NEGATIVOS
# consideremos una imagen con L valores de intensidad
# es decir, q van a estar en el intervalo [0, L-1]
# Podemos invertir las intensidades de la sgte forma:
# g(x,y) = (L - 1) - f(x,y) o s = (L - 1) - r
# ejemplo L = 256 => g(x,y) = 255 - f(x,y)

# invertimos la toy_image
neg_toy_image = 255 - toy_image
print("toy image\n", neg_toy_image)
print("image negatives\n", neg_toy_image)
plt.figure(figsize=(5,5))
plt.subplot(1, 2, 1) 
plt.imshow(toy_image,cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(neg_toy_image,cmap="gray")
plt.show()
print("toy_image:",toy_image)

# invertimos una de las imagenes
image = cv2.imread("imagenes/mammogram.png", cv2.IMREAD_GRAYSCALE)
cv2.rectangle(image, pt1=(160, 212), pt2=(250, 289), color = (255), thickness=2)
img_neg = 255 - image
plt.figure(figsize=(5,5))
plt.subplot(1, 2, 1) 
plt.imshow(image,cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(img_neg,cmap="gray")
plt.show()

# AJUSTES DE BRILLO Y CONTRASTE
# ajuste de contrase --> multiplicar por alfa
# aljuste de brillo --> sumar beta
# g(x,y) = alpha * f(x,y) + beta
# ahora en vez de usar las operaciones sobre el array,
# directamente usamos convertScaleAbs
# esta funcion escala, clcula los valores absolutos y
# convierte el resultado a 8-bit para que caiga dentro
# del intervalo [0, 255]
# para control de brillo hacemos alpha = 1 y beta = 100
alpha = 1 # Simple contrast control
beta = 100   # Simple brightness control   
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
plot_image(goldhill, new_image, title_1 = "Orignal", title_2 = "brightness control")
# ploteamos el hitograma
plt.figure(figsize=(10,5))
plot_hist(goldhill, new_image, "Orignal", "brightness control")
# incrementamos el contraste
plt.figure(figsize=(10,5))
alpha = 2# Simple contrast control
beta = 0 # Simple brightness control   # Simple brightness control
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
plot_image(goldhill,new_image,"Orignal","contrast control")
plt.figure(figsize=(10,5))
plot_hist(goldhill, new_image,"Orignal","contrast control")
# adaptamos el brillo oscureciendo la imagen y subiendo el contraste
plt.figure(figsize=(10,5))
alpha = 3 # Simple contrast control
beta = -200  # Simple brightness control   
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
plot_image(goldhill, new_image, "Orignal", "brightness & contrast control")
plt.figure(figsize=(10,5))
plot_hist(goldhill, new_image, "Orignal", "brightness & contrast control")

# EQUALIZACION
zelda = cv2.imread("imagenes/zelda.png",cv2.IMREAD_GRAYSCALE)
new_image = cv2.equalizeHist(zelda)
plot_image(zelda,new_image,"Orignal","Histogram Equalization")
plt.figure(figsize=(10,5))
plot_hist(zelda, new_image,"Orignal","Histogram Equalization")

# THRESHOLDING (UMBRALIZACION) Y SEGMENTACION
# thresholding --> para extraer objetos de una imagen
# segmantation --> para extraer texto, para imagenes medicas e industriales.

# definimos una funcion para el thresholding
def thresholding(input_img,threshold,max_value=255, min_value=0):
    N,M=input_img.shape
    image_out=np.zeros((N,M),dtype=np.uint8)
        
    for i  in range(N):
        for j in range(M):
            if input_img[i,j]> threshold:
                image_out[i,j]=max_value
            else:
                image_out[i,j]=min_value
                
    return image_out

threshold = 1
max_value = 2
min_value = 0
thresholding_toy = thresholding(toy_image, threshold=threshold, max_value=max_value, min_value=min_value)
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(toy_image, cmap="gray")
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(thresholding_toy, cmap="gray")
plt.title("Image After Thresholding")
plt.show()

# ahora con una imagen
image = cv2.imread("imagenes/cameraman.jpeg", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap="gray")
plt.show()
hist = cv2.calcHist([goldhill], [0], None, [256], [0, 256])
plt.bar(intensity_values, hist[:, 0], width=5)
plt.title("Bar histogram")
plt.show()
threshold = 87
max_value = 255
min_value = 0
new_image = thresholding(image, threshold=threshold, max_value=max_value, min_value=min_value)
plot_image(image, new_image, "Orignal", "Image After Thresholding")
plt.figure(figsize=(10,5))
plot_hist(image, new_image, "Orignal", "Image After Thresholding")
ret, new_image = cv2.threshold(image,threshold,max_value,cv2.THRESH_BINARY)
plot_image(image,new_image,"Orignal","Image After Thresholding")
plot_hist(image, new_image,"Orignal","Image After Thresholding")
ret, new_image = cv2.threshold(image,86,255,cv2.THRESH_TRUNC)
plot_image(image,new_image,"Orignal","Image After Thresholding")
plot_hist(image, new_image,"Orignal","Image After Thresholding")
ret, otsu = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
plot_image(image,otsu,"Orignal","Otsu")
plot_hist(image, otsu,"Orignal"," Otsu's method")
