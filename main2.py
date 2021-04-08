import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, exp

ori = cv2.imread("brrr.jpg", 0)
img_ori = cv2.imread("brrr.jpg", 0)
plain = np.zeros(img_ori.shape)
noise_only = plain
rows, cols = img_ori.shape
img = img_ori
for i in range(rows):
    for j in range(cols):
        noise_only[i][j] = plain[i][j] + 10 * np.sin(i) + 10 * np.sin(j)
        img[i][j] = img[i][j] + 3 * np.sin(i) + 3 * np.sin(j)


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distance1(point1, point2):
    return sqrt((point1[0] - point2[0] - 20) ** 2 + (point1[1] - point2[1] - 40) ** 2)


def distance2(point1, point2):
    return sqrt((point1[0] - point2[0] + 20) ** 2 + (point1[1] - point2[1] + 40) ** 2)


def idealBRF(D0, W, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < (D0 - W / 2):
                base[y, x] = 1
            if distance((y, x), center) > (D0 + W / 2):
                base[y, x] = 1
    return base


def butterBRF(D0, W, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            upperside = distance((y, x), center) * W
            bottomside = 1 + (distance((y, x), center) * distance((y, x), center)) - (D0 * D0)
            base[y, x] = 1 / (1 + ((upperside / bottomside) ** (2 * n)))
    return base


def gaussBRF(D0, W, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            upperside = (distance((y, x), center) ** 2) - (D0 ** 2)
            bottomside = distance((y, x), center) * W + 1
            exp_power = -0.5 * ((upperside / bottomside) ** 2)
            base[y, x] = 1 - exp(exp_power)
    return base


def notchIdealBRF(D0, imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance1((y, x), center) <= D0 or distance2((y, x), center) <= D0:
                base[y, x] = 0
    return base


def notchButterBRF(D0, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    D0 = D0 * 20
    for x in range(cols):
        for y in range(rows):
            upperone = D0 ** 2
            bottomone = 1 + (distance1((y, x), center) ** 2) * (distance2((y, x), center) ** 2)
            bottomside = 1 + ((upperone / bottomone) ** n)
            base[y, x] = 1 / bottomside
    return base


def notchGaussBRF(D0, imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    D0 = D0 * 20
    for x in range(cols):
        for y in range(rows):
            upperside = (distance2((y, x), center) ** 2) * (distance1((y, x), center) ** 2)
            bottomside = D0 ** 2
            exp_power = -0.5 * (upperside / bottomside)
            base[y, x] = 1 - exp(exp_power)
    return base


def idealBPF(D0, W, imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < (D0 - W / 2):
                base[y, x] = 0
            if distance((y, x), center) > (D0 + W / 2):
                base[y, x] = 0
    return base


def butterBPF(D0, W, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            upperside = distance((y, x), center) * W
            bottomside = 1 + (distance((y, x), center) * distance((y, x), center)) - (D0 * D0)
            base[y, x] = ((upperside / bottomside) ** (2 * n)) / (1 + ((upperside / bottomside) ** (2 * n)))
    return base


def gaussBPF(D0, W, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            upperside = (distance((y, x), center) ** 2) - (D0 ** 2)
            bottomside = distance((y, x), center) * W + 1
            exp_power = -0.5 * ((upperside / bottomside) ** 2)
            base[y, x] = exp(exp_power)
    return base


def notchIdealBPF(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance1((y, x), center) <= D0 or distance2((y, x), center) <= D0:
                base[y, x] = 1
    return base


def notchButterBPF(D0, imgShape, n):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    D0 = D0 * 20
    for x in range(cols):
        for y in range(rows):
            upperone = D0 ** 2
            bottomone = 1 + (distance1((y, x), center) ** 2) * (distance2((y, x), center) ** 2)
            bottomside = 1 + ((upperone / bottomone) ** n)
            base[y, x] = 1 - (1 / bottomside)
    return base


def notchGaussBPF(D0, imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    D0 = D0 * 20
    for x in range(cols):
        for y in range(rows):
            upperside = (distance2((y, x), center) ** 2) * (distance1((y, x), center) ** 2)
            bottomside = D0 ** 2
            exp_power = -0.5 * (upperside / bottomside)
            base[y, x] = exp(exp_power)
    return base


def BRF():
    idealBrf = idealBRF(70, 60, img.shape)
    butterBrf = butterBRF(70, 60, img.shape, 4)
    gaussBrf = gaussBRF(70, 60, img.shape)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # Centered multiply with filter
    centerIdealBrf = fshift * idealBrf
    centerButterBrf = fshift * butterBrf
    centerGaussBrf = fshift * gaussBrf

    # Decentralize
    decIdealBrf = np.fft.ifftshift(centerIdealBrf)
    decButterBrf = np.fft.ifftshift(centerButterBrf)
    decGaussBrf = np.fft.ifftshift(centerGaussBrf)

    # Inverse
    inverseIdealBrf = np.fft.ifft2(decIdealBrf)
    inverseButterBrf = np.fft.ifft2(decButterBrf)
    inverseGaussBrf = np.fft.ifft2(decGaussBrf)

    plt.figure(figsize=(12, 6), constrained_layout=False)
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(341), plt.imshow(ori, "gray"), plt.title("Original", fontsize=8)
    plt.subplot(342), plt.imshow(plain, cmap='gray'), plt.title("Noise", fontsize=8)
    plt.subplot(343), plt.imshow(img, cmap='gray'), plt.title("Noise Image", fontsize=8)
    plt.subplot(344), plt.imshow(np.log(1+np.abs(fshift)), cmap='gray'), plt.title("Spectrum Noise", fontsize=8)
    plt.subplot(345), plt.imshow(idealBrf, "gray"), plt.title("Ideal BRF", fontsize=8)
    plt.subplot(346), plt.imshow(np.abs(inverseIdealBrf), "gray"), plt.title("Ideal BRF", fontsize=8)
    plt.subplot(347), plt.imshow(butterBrf, "gray"), plt.title("Butterworth BRF", fontsize=8)
    plt.subplot(3, 4, 8), plt.imshow(np.abs(inverseButterBrf), "gray"), plt.title("Butterworth BRF", fontsize=8)
    plt.subplot(349), plt.imshow(gaussBrf, "gray"), plt.title("Gaussian BRF", fontsize=8)
    plt.subplot(3, 4, 10), plt.imshow(np.abs(inverseGaussBrf), "gray"), plt.title("Gaussian BRF", fontsize=8)
    plt.show()


def BPF():
    idealBpf = idealBPF(30, 30, img.shape)
    butterBpf = butterBPF(20, 25, img.shape, 4)
    gaussBpf = gaussBPF(70, 50, img.shape)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # Centered multiply with filter
    centerIdealBpf = fshift * idealBpf
    centerButterBpf = fshift * butterBpf
    centerGaussBpf = fshift * gaussBpf

    # Decentralize
    decIdealBpf = np.fft.ifftshift(centerIdealBpf)
    decButterBpf = np.fft.ifftshift(centerButterBpf)
    decGaussBpf = np.fft.ifftshift(centerGaussBpf)

    # Inverse
    inverseIdealBpf = np.fft.ifft2(decIdealBpf)
    inverseButterBpf = np.fft.ifft2(decButterBpf)
    inverseGaussBpf = np.fft.ifft2(decGaussBpf)

    plt.figure(figsize=(12, 6), constrained_layout=False)
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(341), plt.imshow(ori, "gray"), plt.title("Original", fontsize=8)
    plt.subplot(342), plt.imshow(plain, cmap='gray'), plt.title("Noise", fontsize=8)
    plt.subplot(343), plt.imshow(img, cmap='gray'), plt.title("Noise Image", fontsize=8)
    plt.subplot(344), plt.imshow(np.log(1+np.abs(fshift)), cmap='gray'), plt.title("Spectrum Noise", fontsize=8)
    plt.subplot(345), plt.imshow(idealBpf, "gray"), plt.title("Ideal BPF", fontsize=8)
    plt.subplot(346), plt.imshow(np.abs(inverseIdealBpf), "gray"), plt.title("Ideal BPF", fontsize=8)
    plt.subplot(347), plt.imshow(butterBpf, "gray"), plt.title("Butterworth BPF", fontsize=8)
    plt.subplot(348), plt.imshow(np.abs(inverseButterBpf), "gray"), plt.title("Butterworth BPF", fontsize=8)
    plt.subplot(349), plt.imshow(gaussBpf, "gray"), plt.title("Gaussian BPF", fontsize=8)
    plt.subplot(3, 4, 10), plt.imshow(np.abs(inverseGaussBpf), "gray"), plt.title("Gaussian BPF", fontsize=8)
    plt.show()


def BRFN():
    idealBrfN = notchIdealBRF(50, img.shape)
    butterBrfN = notchButterBRF(60, img.shape, 3)
    gaussBrfN = notchGaussBRF(70, img.shape)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # Centered multiply with filter
    centerIdealBrfN = fshift * idealBrfN
    centerButterBrfN = fshift * butterBrfN
    centerGaussBrfN = fshift * gaussBrfN

    # Decentralize
    decIdealBrfN = np.fft.ifftshift(centerIdealBrfN)
    decButterBrfN = np.fft.ifftshift(centerButterBrfN)
    decGaussBrfN = np.fft.ifftshift(centerGaussBrfN)

    # Inverse
    inverseIdealBrfN = np.fft.ifft2(decIdealBrfN)
    inverseButterBrfN = np.fft.ifft2(decButterBrfN)
    inverseGaussBrfN = np.fft.ifft2(decGaussBrfN)

    plt.figure(figsize=(12, 6), constrained_layout=False)
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(341), plt.imshow(ori, "gray"), plt.title("Original", fontsize=8)
    plt.subplot(342), plt.imshow(plain, cmap='gray'), plt.title("Noise", fontsize=8)
    plt.subplot(343), plt.imshow(img, cmap='gray'), plt.title("Noise Image", fontsize=8)
    plt.subplot(344), plt.imshow(np.log(1+np.abs(fshift)), cmap='gray'), plt.title("Spectrum Noise", fontsize=8)
    plt.subplot(345), plt.imshow(idealBrfN, "gray"), plt.title("Ideal BRFN", fontsize=8)
    plt.subplot(346), plt.imshow(np.abs(inverseIdealBrfN), "gray"), plt.title("Ideal BRFN", fontsize=8)
    plt.subplot(347), plt.imshow(butterBrfN, "gray"), plt.title("Butterworth BRFN", fontsize=8)
    plt.subplot(348), plt.imshow(np.abs(inverseButterBrfN), "gray"), plt.title("Butterworth BRFN", fontsize=8)
    plt.subplot(349), plt.imshow(gaussBrfN, "gray"), plt.title("Gaussian BRFN", fontsize=8)
    plt.subplot(3, 4, 10), plt.imshow(np.abs(inverseGaussBrfN), "gray"), plt.title("Gaussian BRFN", fontsize=8)
    plt.show()


def BPFN():
    idealBpfN = notchIdealBPF(50, img.shape)
    butterBpfN = notchButterBRF(60, img.shape, 3)
    gaussBpfN = notchGaussBPF(70, img.shape)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # Centered multiply with filter
    centerIdealBpfN = fshift * idealBpfN
    centerButterBpfN = fshift * butterBpfN
    centerGaussBpfN = fshift * gaussBpfN

    # Decentralize
    decIdealBpfN = np.fft.ifftshift(centerIdealBpfN)
    decButterBpfN = np.fft.ifftshift(centerButterBpfN)
    decGaussBpfN = np.fft.ifftshift(centerGaussBpfN)

    # Inverse
    inverseIdealBpfN = np.fft.ifft2(decIdealBpfN)
    inverseButterBpfN = np.fft.ifft2(decButterBpfN)
    inverseGaussBpfN = np.fft.ifft2(decGaussBpfN)

    plt.figure(figsize=(12, 6), constrained_layout=False)
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(341), plt.imshow(ori, "gray"), plt.title("Original", fontsize=8)
    plt.subplot(342), plt.imshow(plain, cmap='gray'), plt.title("Noise", fontsize=8)
    plt.subplot(343), plt.imshow(img, cmap='gray'), plt.title("Noise Image", fontsize=8)
    plt.subplot(344), plt.imshow(np.log(1+np.abs(fshift)), cmap='gray'), plt.title("Spectrum Noise", fontsize=8)
    plt.subplot(345), plt.imshow(idealBpfN, "gray"), plt.title("Ideal BRFN", fontsize=8)
    plt.subplot(346), plt.imshow(np.abs(inverseIdealBpfN), "gray"), plt.title("Ideal BRFN", fontsize=8)
    plt.subplot(347), plt.imshow(butterBpfN, "gray"), plt.title("Butterworth BRFN", fontsize=8)
    plt.subplot(348), plt.imshow(np.abs(inverseButterBpfN), "gray"), plt.title("Butterworth BRFN", fontsize=8)
    plt.subplot(349), plt.imshow(gaussBpfN, "gray"), plt.title("Gaussian BRFN", fontsize=8)
    plt.subplot(3, 4, 10), plt.imshow(np.abs(inverseGaussBpfN), "gray"), plt.title("Gaussian BRFN", fontsize=8)
    plt.show()


while 1:
    print("Choose your option : ")
    print("1. Band Reject Filter")
    print("2. Band Pass Filter")
    print("3. Band Reject Notch Filter")
    print("4. Band Pass Notch filter")
    ch = int(input())
    if ch == 1:
        BRF()
    elif ch == 2:
        BPF()
    elif ch == 3:
        BRFN()
    elif ch == 4:
        BPFN()
    elif ch == 5:
        exit()
    else:
        print("Invalid Option !!!")
