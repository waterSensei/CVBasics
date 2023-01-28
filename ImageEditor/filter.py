import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

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

plt.figure
plt.subplot(7, 2, 1)
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


def myBilateralFilter(mimgNoise, kernelsize, sigD, sigR):
    mimgNoise = mimgNoise/255
    d = int((kernelsize - 1)/2)
    G = np.zeros((kernelsize, kernelsize))
    for y in range(kernelsize):
        for x in range(kernelsize):
            a = x - (kernelsize+1)/2+1
            b = y - (kernelsize+1)/2+1
            G[x, y] = (1/(2*np.pi*sigD ** 2)) * \
                np.exp(-(a ** 2+b ** 2)/(2*sigD ** 2))

    # Iterate every pixel to apply bilateral filter
    [row, column] = [len(mimgNoise), len(mimgNoise[0])]
    output_image = np.zeros((row, column, 3))
    temp = np.zeros((row+2*d, column+2*d, 3))
    temp[d:row+d, d:column+d, :] = mimgNoise
    for y in range(row):
        for x in range(column):
            I = temp[y:y+2*d+1, x:x+2*d+1, :]
            Ld = I[:, :, 0]-mimgNoise[y, x, 0]
            ad = I[:, :, 1]-mimgNoise[y, x, 1]
            bd = I[:, :, 2]-mimgNoise[y, x, 2]
            r = np.exp(-(np.square(Ld)+np.square(ad) +
                       np.square(bd))/(2*sigR**2))
            W = r*G
            W = W/np.sum(W)
            output_image[y, x, 0] = np.sum(W*I[:, :, 0])
            output_image[y, x, 1] = np.sum(W*I[:, :, 1])
            output_image[y, x, 2] = np.sum(W*I[:, :, 2])
    print('--------------------------------------')
    return output_image


output_image_g1 = myGaussFilter(mimgNoise, 5, 2)
output_image_g2 = myGaussFilter(mimgNoise, 5, 0.5)
output_image_g3 = myGaussFilter(mimgNoise, 15, 2)
output_image_g4 = myGaussFilter(mimgNoise, 15, 5)

output_image_bg1 = myBilateralFilter(mimg, 5, 2, 0.2)
output_image_bg2 = myBilateralFilter(mimg, 5, 2, 1)
output_image_bg3 = myBilateralFilter(mimg, 5, 18, 0.2)
output_image_bg4 = myBilateralFilter(mimg, 5, 18, 1)
output_image_bg5 = myBilateralFilter(mimg, 15, 2, 0.2)
output_image_bg6 = myBilateralFilter(mimg, 15, 2, 1)
output_image_bg7 = myBilateralFilter(mimg, 15, 18, 0.2)
output_image_bg8 = myBilateralFilter(mimg, 15, 18, 1)


plt.subplot(7, 2, 3)
plt.imshow(output_image_g1)
plt.axis('off')
plt.title("Selfbuilt function \sigma = 2, K:5x5")

plt.subplot(7, 2, 4)
plt.imshow(output_image_g2)
plt.axis('off')
plt.title("Selfbuilt function \sigma = 0.5, K:5x5")

plt.subplot(7, 2, 5)
plt.imshow(output_image_g3)
plt.axis('off')
plt.title("Selfbuilt function \sigma = 2, K:15x15")

plt.subplot(7, 2, 6)
plt.imshow(output_image_g4)
plt.axis('off')
plt.title("Selfbuilt function \sigma = 5, K:15x15")


plt.subplot(7, 2, 7)
plt.imshow(output_image_bg1)
plt.axis('off')
plt.title("K = 5, \sigma_d = 2, \sigma_r = 0.2")

plt.subplot(7, 2, 8)
plt.imshow(output_image_bg2)
plt.axis('off')
plt.title("K = 5, \sigma_d = 2, \sigma_r = 1")

plt.subplot(7, 2, 9)
plt.imshow(output_image_bg3)
plt.axis('off')
plt.title("K = 5, \sigma_d = 18, \sigma_r = 0.2")

plt.subplot(7, 2, 10)
plt.imshow(output_image_bg4)
plt.axis('off')
plt.title("K = 5, \sigma_d = 18, \sigma_r = 1")

plt.subplot(7, 2, 11)
plt.imshow(output_image_bg5)
plt.axis('off')
plt.title("K = 15, \sigma_d = 2, \sigma_r = 0.2")

plt.subplot(7, 2, 12)
plt.imshow(output_image_bg6)
plt.axis('off')
plt.title("K = 15, \sigma_d = 2, \sigma_r = 1")

plt.subplot(7, 2, 13)
plt.imshow(output_image_bg7)
plt.axis('off')
plt.title("K = 15, \sigma_d = 18, \sigma_r = 0.2")

plt.subplot(7, 2, 14)
plt.imshow(output_image_bg8)
plt.axis('off')
plt.title("K = 15, \sigma_d = 18, \sigma_r = 1")
plt.show()
