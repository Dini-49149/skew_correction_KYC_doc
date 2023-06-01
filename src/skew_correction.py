import cv2
import numpy as np
import math
from pythonRLSA import rlsa
from src.image_utils import image_processing_passport_front

def skew_correction_passport(img):
    image = cv2.imread(img)
    image = cv2.resize(image, (600, 400))
    gray = image_processing_passport_front(image)
    (thresh, image_binary) = cv2.threshold(gray,0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image_rlsa_horizontal = rlsa.rlsa(image_binary, True, False,30)
    (thresh, image_rlsa_horizontal_inv) = cv2.threshold(image_rlsa_horizontal,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imwrite('../../data/rlsa_horizontal.jpg', image_rlsa_horizontal_inv)
    output = cv2.connectedComponentsWithStats(image_rlsa_horizontal_inv, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    count=0
    sum_angle=0
    positive_skew=[]
    negative_skew=[]
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format( i + 1, numLabels)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            componentMask = (labels == i).astype("uint8") * 255
            rectangle_area=w*h
            y_connected=h-((rectangle_area-area)/w)
            x_connected=area/y_connected
            alpha=x_connected/y_connected
            
            xm,ym=(x,y+(y_connected/2))
            a=(cY-ym)/(cX-xm)

            radian=math.atan(a)

            angle=radian*(180/math.pi)

            if w>=400 and alpha>=30:
                sum_angle+=angle
                count+=1
                output = image_rlsa_horizontal.copy()
                componentMask = (labels == i).astype("uint8") * 255

                componentMask_copy=componentMask.copy()
                cv2.rectangle(componentMask_copy, (x, y), (x + w, y + h), (255, 255, 255), 1)
                cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 1)
                white_list=[]
                black_list=[]
                for i in range(int(xm),int(cX)):
                    y_line=int(a*(float(i)-xm)+ym)
                    x_line=int(i)

                    if componentMask[y_line,x_line]==255:         
                        white_list.append(255)

                    elif componentMask[y_line,x_line]==0:
                        black_list.append(0)
                    #cv2.circle(componentMask_copy, (x_line,y_line), radius=0, color=(255, 255, 255), thickness=0)
                if len(black_list)>len(white_list):
                    positive_skew.append('True')
                else:
                    negative_skew.append('True')

    avg_skew_angle=sum_angle/count    
    if len(positive_skew)>len(negative_skew):           
            rotation_angle=360-avg_skew_angle
    else:
        rotation_angle=avg_skew_angle
    print(rotation_angle)
    skew_allowable=0.5
    if avg_skew_angle>skew_allowable:
        image=cv2.imread(img)
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        
        return gray, image_rlsa_horizontal_inv, rotated
    else:
        
        return gray, image_rlsa_horizontal_inv, rotated



