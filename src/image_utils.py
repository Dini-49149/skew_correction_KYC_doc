import cv2
import numpy as np
import matplotlib.pyplot as plt

def imShow(image):
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def image_processing_passport_front(img):
    kernel = np.ones((7,7), np.uint8)
    dilated_img = cv2.dilate(img, kernel)
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = diff_img.copy()
    norm_img = cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    _, thr_img = cv2.threshold(norm_img, 200, 0, cv2.THRESH_TRUNC)
    norm_img = cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    adjusted = adjust_gamma(norm_img, gamma=0.5)
    gray=cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    return gray
