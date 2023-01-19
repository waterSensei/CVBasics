import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

mimg = img.imread('image/image.jpg')
[row, column] = [len(mimg), len(mimg[0])]

# Negative
mimgNeg = 255 - mimg
plt.subplot(3, 2, 1)
plt.imshow(mimgNeg)

# Flip (or use PIL)
mingFlip = np.zeros((row, column, 3))
for i in range(column):
    mingFlip[:, (column-i)-1] = mimg[:, i]
plt.subplot(3, 2, 2)
plt.imshow(mingFlip/255)

# Color Channl Swap
mingCSwap = np.zeros((row, column, 3))
mingCSwap[:, :, 0] = mimg[:, :, 2]
mingCSwap[:, :, 1] = mimg[:, :, 1]
mingCSwap[:, :, 2] = mimg[:, :, 0]
plt.subplot(3, 2, 3)
plt.imshow(mingCSwap/255)

plt.show()
