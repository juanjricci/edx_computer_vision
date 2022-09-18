from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# funcion para concatenar dos imagenes
def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


my_image = "lenna.png"

image = Image.open(my_image)
type(image)

image.show()
print(image.size)
print(image.mode)

im = image.load()
x = 0
y = 1
im[y,x]

image.save("lenna.jpg")

image_gray = ImageOps.grayscale(image)
image_gray.show()
print(image_gray.mode)

# image_gray.quantize(256 // 2)
# image_gray.show()

for n in range(3,8):
    plt.figure(figsize=(10,10))
    plt.imshow(get_concat_h(image_gray,  image_gray.quantize(256//2**n))) 
    plt.title("256 Quantization Levels  left vs {}  Quantization Levels right".format(256//2**n))
    plt.show()