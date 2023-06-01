import cv2
import numpy as np
import math
from pythonRLSA import rlsa
from image_utils import image_processing_passport_front

def skew_correction_passport(img):
    image = cv2.imread(img)
    image = cv2.resize(image, (600, 400))
    gray = image_processing_passport_front(image)
    (thresh, image_binary) = cv2.threshold(gray,0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image_rlsa_horizontal = rlsa.rlsa(image_binary, True, False,30)
    (thresh, image_rlsa_horizontal_inv) = cv2.threshold(image_rlsa_horizontal,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    output = cv2.connectedComponentsWithStats(image_rlsa_horizontal_inv, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    count=0
    sum_angle=0
    positive_skew=[]
    negative_skew=[]
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        componentMask = (labels == i).astype("uint8") * 255
        white_pixels = cv2.countNonZero(componentMask)
        skewness=(h/w)
        rotation_angle=math.degrees(math.atan(skewness))
        if(rotation_angle>45):
            rotation_angle=90-rotation_angle
        sum_angle+=rotation_angle
        count+=1
        if skewness>0:
            positive_skew.append(rotation_angle)
        else:
            negative_skew.append(rotation_angle)
    if count>0:
        avg_angle=sum_angle/count
    else:
        avg_angle=0
    if avg_angle<0:
        avg_angle=90+avg_angle
    print("average_angle", avg_angle)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


