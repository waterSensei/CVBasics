import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

mimg = img.imread('image/image.jpg')
mimg = 255 - mimg
plt.imshow(mimg)
plt.show()
