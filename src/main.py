# Imports
import math

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from scipy import fftpack, signal

# Constants
IMAGES_PATH = "../resources/"
CLRMAP = clr.LinearSegmentedColormap.from_list("clrmap", [(0, 0, 0), (1, 1, 1)], N=256)

MATRIX_Y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                     [12, 12, 14, 19, 26, 58, 60, 55],
                     [14, 13, 16, 24, 40, 57, 69, 56],
                     [14, 17, 22, 29, 51, 87, 80, 72],
                     [18, 22, 37, 56, 68, 109, 103, 77],
                     [24, 35, 55, 64, 81, 104, 113, 92],
                     [49, 64, 78, 87, 103, 121, 120, 101],
                     [72, 92, 95, 98, 112, 100, 103, 99]])

MATRIX_CBCR = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                        [18, 21, 26, 66, 99, 99, 99, 99],
                        [24, 26, 56, 99, 99, 99, 99, 99],
                        [47, 66, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99]])


# Ex 3.1
def openImage(img):
    if isinstance(img, str):
        image = plt.imread(IMAGES_PATH + img)
    else:
        image = img

    # Show image
    plt.figure()
    plt.imshow(image)
    plt.show()


# Ex 3.2 and 3.3
def openImageClrmap(image, colormap):
    # Check channel for colormap
    colormap = {"red":   clr.LinearSegmentedColormap.from_list("clrmap", [(0, 0, 0), (1, 0, 0)], N=256),
                "green": clr.LinearSegmentedColormap.from_list("clrmap", [(0, 0, 0), (0, 1, 0)], N=256),
                "blue":  clr.LinearSegmentedColormap.from_list("clrmap", [(0, 0, 0), (0, 0, 1)], N=256),
                "grey":  clr.LinearSegmentedColormap.from_list("clrmap", [(0, 0, 0), (1, 1, 1)], N=256)
                }[colormap.lower()]

    # Show image
    plt.figure()
    plt.imshow(image, colormap)
    plt.show()


# Ex 3.4
def separateImage(image):
    imageComponents = []

    # Appending arrays to list
    for i in range(3):
        imageComponents.append(image[:, :, i])

    return imageComponents


# Ex 3.4
def joinImage(red, green, blue):
    # Joining arrays
    image = np.dstack((red, green, blue))

    return image


# Ex 4.1
def imagePadding(image):
    # Get image dimensions
    dim = image.shape

    lines_added = (16 - dim[0] % 16) % 16
    columns_added = (16 - dim[1] % 16) % 16

    lines_repeat = np.repeat([image[-1, :]], lines_added, axis=0)
    image = np.vstack((image, lines_repeat))
    columns_repeat = np.repeat(image[:, -1:], columns_added, axis=1)
    image = np.hstack((image, columns_repeat))

    if printPadding:
        print("Old image dimensions: {}x{}".format(dim[0], dim[1]))
        print("New image dimensions: {}x{}\n".format(image.shape[0], image.shape[1]))

    if showPadding:
        plt.imshow(image)
        plt.title("Padded image")
        plt.show()

    return image, dim


# Ex 4.1
def invPadding(image):
    dim = image.shape

    image = image[:dimensions[0], :dimensions[1], :]

    if printPadding:
        print("Old image dimensions: {}x{}".format(dim[0], dim[1]))
        print("New image dimensions: {}x{}\n".format(image.shape[0], image.shape[1]))

    if showPadding:
        plt.imshow(image)
        plt.title("Unpadded image")
        plt.show()

    return image


# Ex 5.1
def imageYCbCr(R, G, B):
    shape = R.shape

    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]])

    imageYcbcr = np.zeros((shape[0], shape[1], 3))

    for i in range(3):
        imageYcbcr[:, :, i] = matrix[i][0] * R + matrix[i][1] * G + matrix[i][2] * B

    imageYcbcr[:, :, (1, 2)] += 128

    if printYCbCr:
        print("YCbCr image (Y channel):")
        print(imageYcbcr[:, :, 0])
        print()

        print("YCbCr image (Cb channel):")
        print(imageYcbcr[:, :, 1])
        print()

        print("YCbCr image (Cr channel):")
        print(imageYcbcr[:, :, 2])
        print()

    if showYCbCr:
        plt.imshow(imageYcbcr[:, :, 0], CLRMAP)
        plt.title("YCbCr image (Y channel)")
        plt.show()

        plt.imshow(imageYcbcr[:, :, 1], CLRMAP)
        plt.title("YCbCr image (Cb channel)")
        plt.show()

        plt.imshow(imageYcbcr[:, :, 2], CLRMAP)
        plt.title("YCbCr image (Cr channel)")
        plt.show()

    return imageYcbcr


# Ex 5.1
def invYCbCr(image):
    shape = np.shape(image)
    matrix = np.linalg.inv(np.array([[0.299, 0.587, 0.114],
                                     [-0.168736, -0.331264, 0.5],
                                     [0.5, -0.418688, -0.081312]]))

    rgb = np.zeros(shape)

    for i in range(3):
        rgb[:, :, i] = matrix[i][0] * image[:, :, 0] + matrix[i][1] * (image[:, :, 1] - 128) + matrix[i][2] * (
                image[:, :, 2] - 128)
        rgb[:, :, i] = np.round(rgb[:, :, i])
        rgb[:, :, i][rgb[:, :, i] > 255] = 255
        rgb[:, :, i][rgb[:, :, i] < 0] = 0

    if printYCbCr:
        print("RGB image (R channel):")
        print(rgb.astype(np.uint8)[:, :, 0])
        print()

        print("RGB image (G channel):")
        print(rgb.astype(np.uint8)[:, :, 1])
        print()

        print("RGB image (B channel):")
        print(rgb.astype(np.uint8)[:, :, 2])
        print()

    if showYCbCr:
        plt.imshow(rgb.astype(np.uint8))
        plt.title("RGB image")
        plt.show()

    return rgb.astype(np.uint8)


# Ex 6.1
def imageDownsampling(ycbcr):
    imageComponents = []

    # Appending arrays to list
    for i in range(3):
        imageComponents.append(ycbcr[:, :, i])

    y_d = imageComponents[0]

    if sampling == "4:2:2":
        cb_d = np.delete(imageComponents[1], np.s_[1::2], 1)
        cr_d = np.delete(imageComponents[2], np.s_[1::2], 1)
    elif sampling == "4:2:0":
        cb_d = np.delete(imageComponents[1], np.s_[1::2], 1)
        cb_d = np.delete(cb_d, np.s_[1::2], 0)
        cr_d = np.delete(imageComponents[2], np.s_[1::2], 1)
        cr_d = np.delete(cr_d, np.s_[1::2], 0)
    else:
        print("Invalid downsampling variant!")
        return -1

    if printDownsampling:
        print("Y (Downsampling): {}x{}\n".format(y_d.shape[0], y_d.shape[1]))
        print("Cb (Downsampling): {}x{}\n".format(cb_d.shape[0], cb_d.shape[1]))
        print("Cr (Downsampling): {}x{}\n".format(cr_d.shape[0], cr_d.shape[1]))

    if showDownsampling:
        plt.imshow(y_d, CLRMAP)
        plt.title("DCT (Y channel)")
        plt.show()

        plt.imshow(cb_d, CLRMAP)
        plt.title("DCT (Cb channel)")
        plt.show()

        plt.imshow(cr_d, CLRMAP)
        plt.title("DCT (Cr channel)")
        plt.show()

    return [y_d, cb_d, cr_d]


# Ex 6.1
def imageUpsampling(y_d, cb_d, cr_d):
    if sampling == "4:2:2":
        if interpolation:
            cb_ch = signal.resample(cb_d, len(cb_d[0]) * 2, axis=1)
            cr_ch = signal.resample(cr_d, len(cr_d[0]) * 2, axis=1)
        else:
            cb_ch = np.repeat(cb_d, 2, axis=1)
            cr_ch = np.repeat(cr_d, 2, axis=1)
    elif sampling == "4:2:0":
        if interpolation:
            cb_ch = signal.resample(cb_d, len(cb_d) * 2, axis=0)
            cb_ch = signal.resample(cb_ch, len(cb[0]) * 2, axis=1)
            cr_ch = signal.resample(cr_d, len(cr_d) * 2, axis=0)
            cr_ch = signal.resample(cr_ch, len(cr[0]) * 2, axis=1)
        else:
            cb_ch = np.repeat(cb_d, 2, axis=0)
            cb_ch = np.repeat(cb_ch, 2, axis=1)
            cr_ch = np.repeat(cr_d, 2, axis=0)
            cr_ch = np.repeat(cr_ch, 2, axis=1)
    else:
        print("Invalid downsampling variant!")
        return -1

    y_ch = y_d

    if printDownsampling:
        print("Y (Upsampling): {}x{}\n".format(y.shape[0], y.shape[1]))
        print("Cb (Upsampling): {}x{}\n".format(cb.shape[0], cb.shape[1]))
        print("Cr (Upsampling): {}x{}\n".format(cr.shape[0], cr.shape[1]))

    if showDownsampling:
        plt.imshow(y, CLRMAP)
        plt.title("DCT (Y channel)")
        plt.show()

        plt.imshow(cb, CLRMAP)
        plt.title("DCT (Cb channel)")
        plt.show()

        plt.imshow(cr, CLRMAP)
        plt.title("DCT (Cr channel)")
        plt.show()

    return [y_ch, cb_ch, cr_ch]


# Ex 7.1
def imageDct(channel):
    channel_dct = fftpack.dct(fftpack.dct(channel, norm="ortho").T, norm="ortho").T

    # Show image
    plt.figure()
    plt.imshow(np.log(np.abs(channel_dct) + 0.0001))
    plt.show()

    return channel_dct


# Ex 7.1
def invDct(channel_dct):
    channel = fftpack.idct(fftpack.idct(channel_dct, norm="ortho").T, norm="ortho").T

    return channel


# Ex 7.2.1
def dctcustom(channel):
    dim = np.shape(channel)
    dct = np.zeros((dim[0], dim[1]))

    for i in range(0, int(dim[0] / blocks)):
        for j in range(0, int(dim[1] / blocks)):
            img = fftpack.dct(
                fftpack.dct(channel[i * blocks:(i + 1) * blocks, j * blocks:(j + 1) * blocks], norm="ortho").T,
                norm="ortho").T
            dct[i * blocks:(i + 1) * blocks, j * blocks:(j + 1) * blocks] = img

    if printDCT:
        print("DCT ({}x{})".format(blocks, blocks))
        print(dct)
        print()

    if showDCT:
        plt.imshow(np.log(np.abs(dct) + 0.0001), CLRMAP)
        plt.title("DCT ({}x{})".format(blocks, blocks))
        plt.show()

    return dct


# Ex 7.2.1
def invdctcustom(channel_dct):
    dim = np.shape(channel_dct)
    channel = np.zeros((dim[0], dim[1]))

    for i in range(0, int(dim[0] / blocks)):
        for j in range(0, int(dim[1] / blocks)):
            img = fftpack.idct(
                fftpack.idct(channel_dct[i * blocks:(i + 1) * blocks, j * blocks:(j + 1) * blocks], norm="ortho").T,
                norm="ortho").T
            channel[i * blocks:(i + 1) * blocks, j * blocks:(j + 1) * blocks] = img

    if printDCT:
        print("Inverse DCT ({}x{})".format(blocks, blocks))
        print(channel)
        print()

    if showDCT:
        plt.imshow(np.log(np.abs(channel) + 0.0001), CLRMAP)
        plt.title("Inverse DCT ({}x{})".format(blocks, blocks))
        plt.show()

    return channel


# Ex 8.1
def quantization(channel, type):
    dim = np.shape(channel)
    quant = np.zeros((dim[0], dim[1]))

    if quality < 50:
        sf = 50 / quality
    else:
        sf = (100 - quality) / 50

    if type == "Y":
        activematrix = MATRIX_Y
    elif type == "Cb" or type == "Cr":
        activematrix = MATRIX_CBCR
    else:
        print("Invalid type for channel")
        return -1

    for i in range(0, int(dim[0] / 8)):
        for j in range(0, int(dim[1] / 8)):
            if sf == 0:
                img = np.around(channel[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8])
            else:
                qsf = np.round(activematrix * sf)
                qsf[qsf > 255] = 255
                qsf[qsf < 1] = 1
                img = np.divide(channel[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8], qsf)
                img = np.around(img)
            quant[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = img

    if printQuantization:
        # Teste para o primeiro bloco 8x8 do canal quantizado
        print("Quantization ({} channel)".format(type))
        print(quant)
        print()

    if showQuantization:
        plt.imshow(np.log(np.abs(quant) + 0.0001), CLRMAP)
        plt.title("Quantization ({} channel)".format(type))
        plt.show()

    return quant


# Ex 8.1
def inv_quantization(channel_q, type):
    dim = np.shape(channel_q)
    channel = np.zeros((dim[0], dim[1]))

    if quality < 50:
        sf = 50 / quality
    else:
        sf = (100 - quality) / 50

    if type == "Y":
        activematrix = MATRIX_Y
    elif type == "Cb" or type == "Cr":
        activematrix = MATRIX_CBCR
    else:
        print("Invalid type for channel")
        return -1

    for i in range(0, int(dim[0] / 8)):
        for j in range(0, int(dim[1] / 8)):
            if sf == 0:
                img = np.around(channel_q[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8])
            else:
                qsf = np.round(activematrix * sf)
                qsf[qsf > 255] = 255
                qsf[qsf < 1] = 1
                img = np.multiply(channel_q[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8], qsf)
                img = np.around(img)
            channel[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = img

    if printQuantization:
        print("Inverse quantization ({} channel)".format(type))
        print(channel)
        print()

    if showQuantization:
        plt.imshow(np.log(np.abs(channel) + 0.0001), CLRMAP)
        plt.title("Inverse quantization ({} channel)".format(type))
        plt.show()

    return channel


# Ex 9.1
def differential(channel, type):
    dim = np.shape(channel)
    dpcm = np.copy(channel)

    for i in range(0, int(dim[0] / blocks)):
        for j in range(0, int(dim[1] / blocks)):
            if i == 0 and j == 0:
                img = channel[i * blocks][j * blocks]
            else:
                if j != 0:
                    img = channel[i * blocks][j * blocks] - channel[i * blocks][(j - 1) * blocks]
                else:
                    img = channel[i * blocks][j * blocks] - channel[(i - 1) * blocks][
                        (int(dim[1] / blocks) - 1) * blocks]

            dpcm[i * blocks][j * blocks] = img

    if printDPCM:
        # Teste para o primeiro bloco 8x8 do canal quantizado
        print("DPCM ({} channel)".format(type))
        print(dpcm)
        print()

    if showDPCM:
        plt.imshow(np.log(np.abs(dpcm) + 0.0001), CLRMAP)
        plt.title("DPCM ({} channel)".format(type))
        plt.show()

    return dpcm


# Ex 9.1
def inv_differential(channel_d, type):
    dim = np.shape(channel_d)
    channel = np.copy(channel_d)

    for i in range(0, int(dim[0] / blocks)):
        for j in range(0, int(dim[1] / blocks)):
            if i == 0 and j == 0:
                img = channel_d[i * blocks][j * blocks]
            else:
                if j != 0:
                    img = channel_d[i * blocks][j * blocks] + channel[i * blocks][(j - 1) * blocks]
                else:
                    img = channel_d[i * blocks][j * blocks] + channel[(i - 1) * blocks][
                        (int(dim[1] / blocks) - 1) * blocks]

            channel[i * blocks][j * blocks] = img

    if printDPCM:
        # Teste para o primeiro bloco 8x8 do canal quantizado
        print("Inverse DPCM ({} channel)".format(type))
        print(channel_d)
        print()

    if showDPCM:
        plt.imshow(np.log(np.abs(channel) + 0.0001), CLRMAP)
        plt.title("Inverse DPCM ({} channel)".format(type))
        plt.show()

    return channel


# Ex 10
def errorImage(originalImage, newImage):
    # Did function receive a .bmp or an array
    if isinstance(originalImage, str):
        image = plt.imread(IMAGES_PATH + originalImage)
    else:
        image = originalImage

    R, G, B = separateImage(image)
    image = imageYCbCr(R, G, B)
    R, G, B = separateImage(newImage)
    newImage = imageYCbCr(R, G, B)

    if showError:
        plt.imshow((np.abs(image - newImage))[:, :, 0], CLRMAP)
        plt.title("Error image")
        plt.show()

    m = image.shape[0]
    n = image.shape[1]

    image = invYCbCr(image).astype(float)
    newImage = invYCbCr(newImage).astype(float)

    MSE = np.sum(np.square(image-newImage)) / (m * n)
    RMSE = np.sqrt(MSE)
    P = np.sum(np.square(image)) / (m * n)
    SNR = 10 * math.log10(P / MSE)
    PSNR = 10 * math.log10(np.square(np.max(image)) / MSE)

    if printError:
        print("MSE: {}\nRMSE: {}\nSNR: {}\nPSNR: {}".format(round(MSE, 2), round(RMSE, 2), round(SNR, 2), round(PSNR, 2)))


# Ex 2
def encoder(img):
    # Did function receive a .bmp or an array
    if isinstance(img, str):
        image = plt.imread(IMAGES_PATH + img)
    else:
        image = img

    # Padding
    global dimensions
    image, dimensions = imagePadding(image)

    # Convert to YCbCr
    imageR, imageG, imageB = separateImage(image)
    convertedImageYCbCr = imageYCbCr(imageR, imageG, imageB)

    # Downsampling
    y_d, cb_d, cr_d = imageDownsampling(convertedImageYCbCr)

    # DCT
    y_dct = dctcustom(y_d)
    cb_dct = dctcustom(cb_d)
    cr_dct = dctcustom(cr_d)

    # Quantization
    y_quant = quantization(y_dct, "Y")
    cb_quant = quantization(cb_dct, "Cb")
    cr_quant = quantization(cr_dct, "Cr")

    # Differential Coding
    y_diff = differential(y_quant, "Y")
    cb_diff = differential(cb_quant, "Cb")
    cr_diff = differential(cr_quant, "Cr")

    return y_diff, cb_diff, cr_diff, image


# Ex 2
def decoder(y_ch, cb_ch, cr_ch):
    # Inverse Differential Coding
    y_inv_diff = inv_differential(y_ch, "Y")
    cb_inv_diff = inv_differential(cb_ch, "Cb")
    cr_inv_diff = inv_differential(cr_ch, "Cr")

    # Inverse Quantization
    y_inv_quant = inv_quantization(y_inv_diff, "Y")
    cb_inv_quant = inv_quantization(cb_inv_diff, "Cb")
    cr_inv_quant = inv_quantization(cr_inv_diff, "Cr")

    # Inverse DCT
    y_inv_dct = invdctcustom(y_inv_quant)
    cb_inv_dct = invdctcustom(cb_inv_quant)
    cr_inv_dct = invdctcustom(cr_inv_quant)

    # Upsampling
    [y_upsampled, cb_upsampled, cr_upsampled] = imageUpsampling(y_inv_dct, cb_inv_dct, cr_inv_dct)

    # Join YCbCr channels
    joinedImage = joinImage(y_upsampled, cb_upsampled, cr_upsampled)

    # Convert to RGB
    RGBimage = invYCbCr(joinedImage)

    # Rever padding
    decodedImg = invPadding(RGBimage)

    return decodedImg


# Main
if __name__ == "__main__":
    # Show steps
    showPadding = False
    showYCbCr = False
    showDownsampling = False
    showDCT = False
    showQuantization = False
    showDPCM = False
    showError = False

    # Print steps
    printPadding = False
    printYCbCr = False
    printDownsampling = False
    printDCT = False
    printQuantization = False
    printDPCM = False
    printError = False

    # Encoding/decoding parameters
    file = "barn_mountains.bmp"  # File must be in specified path (Check program constants)
    sampling = "4:2:0"           # 4:2:0 V 4:2:2
    blocks = 8                   # 8 V 64
    quality = 75                 # n ∈ [1, 100]
    interpolation = True         # True V False
    dimensions = [0, 0]

    # Original image
    openImage(file)

    # Encoded image
    y, cb, cr, encodedImage = encoder(file)

    # Decoded Image
    decodedImage = decoder(y, cb, cr)
    openImage(decodedImage)

    # Error Image
    errorImage(file, decodedImage)
