#Import Libraries

import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from keras.preprocessing.image import img_to_array

#Loading the image
# im = cv2.imread("input_1.jpg")
def return_shape(image):
    return image.shape

def increase_size(image,kernel,ratio):
    img_height,img_width = return_shape(image)
    img_height = int(ratio * img_height)
    img_width = int(ratio * img_width)
    resized = cv2.resize(image,(img_width,img_height),interpolation=cv2.INTER_AREA)
    resized = cv2.erode(resized, kernel, iterations = 1)
    resized = cv2.dilate(resized, kernel, iterations = 1)
    resized = cv2.erode(resized, kernel, iterations = 1)
    return resized


def get_licence_number(image,roi_box,loaded_model,label_map):
    #Converting image to Grayscale
    im = image[roi_box.ymin:roi_box.ymax,roi_box.xmin:roi_box.xmax]
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(im_gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    #Finding Contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None
     
    # loop over the contours
    for c in cnts:
    	# approximate the contour
    	peri = cv2.arcLength(c, True)
    	approx = cv2.approxPolyDP(c, 0.03 * peri, True)
     
    	#The contour has four vertices
    	if len(approx) == 4:
    		displayCnt = approx
    		break

    img_erode = four_point_transform(im_gray, displayCnt.reshape(4, 2))

    kernel = np.ones((3,3),np.uint8)

    while return_shape(img_erode)[1] < 720 or return_shape(img_erode)[0] < 240:
        img_erode = increase_size(img_erode, kernel, 2)

    thresh = cv2.threshold(img_erode, 180, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(thresh, 0, 255,cv2.THRESH_BINARY_INV)[1]

    cv2.imshow("New",thresh)

    ret, resized = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV)

    ctrs,_ = cv2.findContours(resized.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []

    c_h,c_w = return_shape(resized) 
    
    for ctr in ctrs: 
        (x, y, w, h) = cv2.boundingRect(ctr)
        if w >=int(c_w/30) and h >= int(c_h/4):
            rects.append(cv2.boundingRect(ctr))

    cv2.imshow("Input_image_number_plate",resized)

    list_alpha_numerics = []
    #Predicting the alphanumerics
    for rect in rects:
        roi = resized[rect[1] - 10:rect[1] + rect[3] + 10, rect[0] - 10:rect[0] + rect[2] + 10]
        if roi.any():
        # Resize the image   
            roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
            # roi = cv2.dilate(roi, (3, 3))
            X = img_to_array(roi)
            X = X/255
            X = X.reshape(-1,64,64,1)
            nbr = loaded_model.predict_classes(X)
            for key in label_map.keys():
                if label_map[key] == nbr:
                    nbr =  key
            list_alpha_numerics.append(nbr)
            cv2.imshow("rects",roi)
            cv2.waitKey()
    return list_alpha_numerics