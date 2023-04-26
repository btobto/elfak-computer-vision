import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 30


def stitch(dst_img, src_img, min_match_count=MIN_MATCH_COUNT):
    gray_dst_img = cv.cvtColor(dst_img, cv.COLOR_BGR2GRAY)
    gray_src_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

    detector = cv.SIFT_create()

    kp1, des1 = detector.detectAndCompute(gray_dst_img, None)
    kp2, des2 = detector.detectAndCompute(gray_src_img, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.2 * n.distance:
            good.append(m)

    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
    else:
        mask = None

    dst = cv.warpPerspective(src_img, M, (src_img.shape[1] + dst_img.shape[1], src_img.shape[0] + dst_img.shape[0]))
    dst[0:dst_img.shape[0], 0:dst_img.shape[1]] = dst_img

    return dst


if __name__ == '__main__':
    img_1 = cv.imread('./1.jpg')
    img_2 = cv.imread('./2.jpg')
    img_3 = cv.imread('./3.jpg')

    full_image = stitch(img_1, stitch(img_2, img_3))

    cv.imshow('Output', full_image)
    cv.imwrite('output.png', full_image)

    cv.waitKey()
