import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv.dilate(src=marker, kernel=kernel)
        expanded = cv.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


if __name__ == '__main__':
    img = cv.imread('coins.png')
    img_gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

    _, coins_mask = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY_INV)
    coins_mask_closed = cv.morphologyEx(coins_mask, cv.MORPH_CLOSE,
                                        cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_saturation = img_hsv[:, :, 1]

    _, copper_coin = cv.threshold(img_saturation, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    copper_coin_open = cv.morphologyEx(copper_coin, cv.MORPH_OPEN,
                                       cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    reconstructed = morphological_reconstruction(copper_coin_open, coins_mask_closed)

    cv.imshow('original', img)
    cv.imshow('coin mask', reconstructed)

    cv.imwrite('coin_mask.png', reconstructed)

    cv.waitKey()
