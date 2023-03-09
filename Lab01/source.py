import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def display_image(img, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.show()


def convert_to_utf8(arr):
    return cv.normalize(arr, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


def get_magnitude_spectrum(img):
    abs = np.abs(img)
    return np.log(abs, out=np.zeros_like(abs), where=(abs != 0))


def fft(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    return img_fft


def inverse_fft(img_fft):
    img_fft = np.fft.ifftshift(img_fft)
    img_filtered = np.abs(np.fft.ifft2(img_fft))
    return img_filtered


if __name__ == '__main__':
    img = cv.imread('input.png', cv.COLOR_BGR2GRAY)
    height, width = img.shape
    center = (width // 2, height // 2)

    img_fft = fft(img)
    magnitude_spectrum = get_magnitude_spectrum(img_fft)
    display_image(magnitude_spectrum)

    mask = cv.threshold(magnitude_spectrum, 14, 255, cv.THRESH_BINARY)[1]
    cv.circle(mask, center, 20, 0, cv.FILLED)
    display_image(mask, 'gray')

    img_fft[mask != 0] = 0

    magnitude_spectrum_filtered = get_magnitude_spectrum(img_fft)
    display_image(magnitude_spectrum_filtered)

    img_filtered = inverse_fft(img_fft)
    display_image(img_filtered, 'gray')

    cv.imwrite('fft_mag.png', convert_to_utf8(magnitude_spectrum))
    cv.imwrite('fft_mag_filtered.png', convert_to_utf8(magnitude_spectrum_filtered))
    cv.imwrite('output.png', img_filtered)
