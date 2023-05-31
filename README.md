# skew-correction-KYC-Documents
## Go through the skew correction for passport ipython notebook file and other two are similar to it.
At present, our world is progressing rapidly towards process automation with the help of AI technologies, one of which is used to automate ID card information extraction. In this process, documents are being digitalized and information is extracted and stored from a text extracted using Optical Character Recognition, cutting edge technology for text extraction. The text extraction accuracy is observed to be higher in zero skewed images with text in black and white background, consequently, we need to eliminate the skewness in the document uploaded.
 
There are different skew correction methods available for text images having black text over the white background as shown in fig1. These methods sometimes do not work for KYC Documents like voter, passport, Driving licence etc, because they contain other information such as pictures, signature, QR codes, watermarks etc. So, In this post, we will discuss my approach of solving the skew correction by taking the passport sample as shown in fig2.

<p align="center"> <img src="https://user-images.githubusercontent.com/71541898/133197675-9ab27ae9-f100-4aa6-8615-661c62e2d586.jpg" width="450" height="350"/> </p> 
<p align="center">fig1: skewed text images at different angles</p>
<p align="center"> <img src="https://user-images.githubusercontent.com/71541898/133245481-2c94f835-d512-485f-b5d6-ad63756ea97a.jpg" width="450" height="350"/> </p>
<p align="center">fig 2: Passport sample for skew correction</p>

## Dependencies

Python3, matplotlib, numpy, opencv 3.

### Getting started

Steps to skew correction are as follows.

1. Load the image and resize it to a particular size that will help to choose best connected component further:
    ```
    img='passport_front_copy.jpg'
    image = cv2.imread(img)
    image = cv2.resize(image, (600, 400))
    ```

2. Do the Image preprocessing to get more visible clear text on plane background:
    ```
    gray = image_processing_passport_front(image). # look at the code for image preprocessing
    ```

3. Convert the image to binary 0 or 1:
    ```
    (thresh, image_binary) = cv2.threshold(gray,0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU).
    ```
4. Do block segmentation and text descrimination using [pythonRLSA](https://pypi.org/project/pythonRLSA/):
    ```
    from pythonRLSA import rlsa
    image_rlsa_horizontal = rlsa.rlsa(image_binary, True, False,30)
    ```

5. Inverse the binary image using opencv thresh_binary_inv method which we can extract the stats of that component:
    ```
    (thresh, image_rlsa_horizontal_inv) = cv2.threshold(image_rlsa_horizontal,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ```
6. Applying opencv [connectedComponentsWithStats](https://www.pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/) method to obtain the stats of each connected component:
    ```
    output = cv2.connectedComponentsWithStats(image_rlsa_horizontal_inv, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    
    This method returns the following information:

    1. The bounding box of the connected component
    2. The area (in pixels) of the component
    3. The centroid/center (x, y)-coordinates of the component
    ```

7. Finding out the angle of rotation with the help of stats obtained from the connectedComponentWithStats method and also positive or negative:
   
     The known and unknown parameters are understood from the diagram. we can say that the area of the triangle is ```1/2(h-y_connected*w) and 2*area of triangle+area of              component=area of rectangle```. so the parameters x_connected, y_connected and skew anlge wether it is positive or negative, are calculated as shown in first and second figures

     <p float="left">
       <img src="https://user-images.githubusercontent.com/71541898/133263122-d64dca79-eee4-4f35-ab28-ead56a51ef59.jpg" width="450" height="350"/>
       <img src="https://user-images.githubusercontent.com/71541898/133262898-e01e6fff-78d1-40c2-9389-2022e1604e1e.jpg" width="450" height="350"/>
     </p>
     
                           
                            

8. Filter out the best connected components having rectangle shape by iterating through each component:
   You can try with your own parameters and their values and see which works for your case. For example, i have taken width of the rectangle and alpha= x_connected/y_connected
    ```
    if w>=400 and alpha>=30: is true, then add to the total skew angle and at final calculate 
    the avg skew angle for all the filtered connected components
    ```
9. Compensate the skew angle :
   Here i have taken permissible skew angle 0.5 and if it is greater than the allowable, then skew correction takes place
    ```
    skew_allowable=0.5
    if avg_skew_angle>skew_allowable:
        image=cv2.imread(img)
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
    ```

##### Note :
##### 1. Go throught the connectComponentWithStat method parameters and look at the pythonRLSA library. Try with your own KYC Documents, required different image preprocessing for each may be, need to do parameter tuning wherever required (RLSA, in filtering the best connected componets that resembles the shape of rectangle )
##### 2. You can also calculate the average skew angle for all the connected components and see if it works for your KYC document.
