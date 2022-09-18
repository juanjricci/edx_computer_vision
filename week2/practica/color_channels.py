from PIL import Image, ImageOps
import matplotlib.pyplot as plt


# funcion para concatenar dos imagenes
def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


baboon = Image.open('baboon.png')
red, green, blue = baboon.split()

r = get_concat_h(baboon, red)
b = get_concat_h(baboon, blue)
g = get_concat_h(baboon, green)

r.show()
b.show()
g.show()