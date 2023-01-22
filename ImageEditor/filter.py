import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from PIL import Image as pil

mimg = img.imread('image/image.jpg')
[row, column] = [len(mimg), len(mimg[0])]
print([row, column])

mimgNoise = np.zeros((row, column, 3))
noise = np.random.randint(0, 127, [row, column])
grey = mimg[:, :, 0]*0.2989 + mimg[:, :, 1]*0.5870 + mimg[:, :, 2]*0.1140
mimgNoise[:, :, 0] = grey + noise
mimgNoise[:, :, 1] = grey + noise
mimgNoise[:, :, 2] = grey + noise
mimgNoise[mimgNoise > 255] = 255
plt.subplot(2, 2, 1)
plt.title('Noise Added Image')
plt.axis('off')
plt.imshow(mimgNoise/255)


def myGaussFilter(mimgNoise, kernelsize, sigma):
    mimgNoise = mimgNoise/255

    # Gaussian Kernel
    d = int((kernelsize - 1)/2)
    G = np.zeros((kernelsize, kernelsize))
    for y in range(kernelsize):
        for x in range(kernelsize):
            a = x - (kernelsize+1)/2
            b = y - (kernelsize+1)/2
            G[x, y] = (1/(2*np.pi*sigma ** 2)) * \
                np.exp(-(a ** 2+b ** 2)/(2*sigma ** 2))

    # Iterate every pixel to apply Gaussian filter
    [row, column] = [len(mimgNoise), len(mimgNoise[0])]
    output_image = np.zeros((row, column, 3))
    temp = np.zeros((row+2*d, column+2*d, 3))
    # print([len(temp), len(temp[0])])
    # print([len(mimgNoise), len(mimgNoise[0])])
    temp[d:row+d, d:column+d, :] = mimgNoise
    G = G/np.sum(G)
    f = np.zeros((2*d+1, 2*d+1, 3))
    for y in range(row):
        for x in range(column):
            I = temp[y:y+2*d+1, x:x+2*d+1, :]
            # Calculate Gaussian filtered image
            f[:, :, 0] = I[:, :, 0]*G
            f[:, :, 1] = I[:, :, 1]*G
            f[:, :, 2] = I[:, :, 2]*G
            output_image[y, x, 0] = np.sum(f[:, :, 0])
            output_image[y, x, 1] = np.sum(f[:, :, 1])
            output_image[y, x, 2] = np.sum(f[:, :, 2])
    return output_image


output_image_g1 = myGaussFilter(mimgNoise, 5, 2)
output_image_g2 = myGaussFilter(mimgNoise, 5, 0.5)
output_image_g3 = myGaussFilter(mimgNoise, 15, 2)
output_image_g4 = myGaussFilter(mimgNoise, 15, 5)
plt.subplot(2, 2, 2)
plt.imshow(output_image_g1)
plt.axis('off')
plt.title("Selfbuilt function \sigma = 2, K:5x5")

plt.subplot(2, 2, 3)
plt.imshow(output_image_g2)
plt.axis('off')
plt.title("Selfbuilt function \sigma = 0.5, K:5x5")

plt.subplot(2, 2, 4)
plt.imshow(output_image_g3)
plt.axis('off')
plt.title("Selfbuilt function \sigma = 2, K:15x15")
plt.show()
