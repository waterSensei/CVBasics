import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

mimg = img.imread('image/image.jpg')
[row, column] = [len(mimg), len(mimg[0])]

# Negative
mimgNeg = 255 - mimg
plt.subplot(3, 2, 1)
plt.title('Negative Image')
plt.axis('off')
plt.imshow(mimgNeg)

# Flip (or use PIL)
mimgFlip = np.zeros((row, column, 3))
for i in range(column):
    mimgFlip[:, (column-i)-1] = mimg[:, i]
plt.subplot(3, 2, 2)
plt.title('Flipped Image')
plt.axis('off')
plt.imshow(mimgFlip/255)

# Color Channl Swap
mimgCSwap = np.zeros((row, column, 3))
mimgCSwap[:, :, 0] = mimg[:, :, 2]
mimgCSwap[:, :, 1] = mimg[:, :, 1]
mimgCSwap[:, :, 2] = mimg[:, :, 0]
plt.subplot(3, 2, 3)
plt.title('Color Channal Swapped Image')
plt.axis('off')
plt.imshow(mimgCSwap/255)

# Average
mimgAvg = np.zeros((row, column, 3))
mimgAvg = (mimgFlip + mimg)/2
plt.subplot(3, 2, 4)
plt.title('Average Image')
plt.axis('off')
plt.imshow(mimgAvg/255)

# Greyscale
mimgGrey = np.zeros((row, column, 3))
grey = mimg[:, :, 0]*0.2989 + mimg[:, :, 1]*0.5870 + mimg[:, :, 2]*0.1140
mimgGrey[:, :, 0] = grey
mimgGrey[:, :, 1] = grey
mimgGrey[:, :, 2] = grey
plt.subplot(3, 2, 5)
plt.title('Grayscale Image')
plt.axis('off')
plt.imshow(mimgGrey/255)

# Noise
mimgNoise = np.zeros((row, column, 3))
noise = np.random.randint(0, 127, [row, column, 3])
mimgNoise = mimgGrey + noise
mimgNoise[mimgNoise > 255] = 255
plt.subplot(3, 2, 6)
plt.title('Noise Added Image')
plt.axis('off')
plt.imshow(mimgNoise/255)

plt.show()
