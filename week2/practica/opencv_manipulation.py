import matplotlib.pyplot as plt
import cv2
import numpy as np

# FLIPPING WITH OPENCV
image = cv2.imread("lenna.png")

# openCV tiene varias formas de dar vuelta una imagen
# su metodo flip() tiene el siguiente parametro flipcode
# flipcode = 0: flip vertically around the x-axis
# flipcode > 0: flip horizontally around y-axis positive value
# flipcode < 0: flip vertically and horizontally, flipping around both

# probemoslo en un bucle
for flipcode in [0,1,-1]:
    im_flip =  cv2.flip(image,flipcode)
    plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
    plt.title("flipcode: "+ str(flipcode))
    plt.show()

# podemos usar tmb la funcion rotate()
# su parametro indica que tipo de rotacion queremos aplicar
flip = {
    # 0
    "ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE,
    # 1
    "ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE,
    # 2
    "ROTATE_180":cv2.ROTATE_180
    }
im_flip = cv2.rotate(image, 0)
plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
plt.title('flipcode: 0')
plt.show()

for key, value in flip.items():
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("orignal")
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(cv2.rotate(image,value), cv2.COLOR_BGR2RGB))
    plt.title(key)
    plt.show()

# CROPPING
# crop top
upper = 150
lower = 400
crop_top = image[upper: lower,:,:]
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(crop_top, cv2.COLOR_BGR2RGB))
plt.show()

# crop horizontal
left = 150
right = 400
crop_horizontal = image[: ,left:right,:]
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(crop_horizontal, cv2.COLOR_BGR2RGB))
plt.show()

# CAMBIANDO PIXELES ESPECIFICOS
array_sq = np.copy(image)
array_sq[upper:lower,left:right,:] = 0
plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title("orignal")
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(array_sq,cv2.COLOR_BGR2RGB))
plt.title("Altered Image")
plt.show()

# creamos formas con openCV
# primero con rectangle()
start_point, end_point = (left, upper),(right, lower)
image_draw = np.copy(image)
cv2.rectangle(image_draw, pt1=start_point, pt2=end_point, color=(0, 255, 0), thickness=3) 
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
plt.show()

# luego texto con putText()
image_draw=cv2.putText(img=image,text='Stuff',org=(10,500),color=(255,255,255),fontFace=4,fontScale=5,thickness=2)
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(image_draw,cv2.COLOR_BGR2RGB))
plt.show()