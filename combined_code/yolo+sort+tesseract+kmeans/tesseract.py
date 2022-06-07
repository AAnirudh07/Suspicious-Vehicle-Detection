import os
import cv2

'''
cap = cv2.VideoCapture(PATH)
ret, frame = cap.read()
frame = frame[:100,:500]
cv2.imshow("Cropped Image", frame)
cv2.waitKey()
'''

PATH = "D:\out.mp4"

import cv2
import pytesseract
import argparse
pytesseract.pytesseract.tesseract_cmd = "C:\Program Files (x86)\Tesseract-OCR\\tesseract"

def preprocess(image, target=30):
    '''
    Preprocesses the image for Tesseract
    '''
    
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # correct the text height
    ratio = max(0.4, min(1.5, target / image.shape[0]))
    image = cv2.resize(image,
                       dsize=None,
                       fx=ratio,
                       fy=ratio,
                       interpolation=cv2.INTER_LANCZOS4)

    return image
    # apply Otsu thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

# cap = cv2.VideoCapture(PATH)
# ret = True
# while True:
#     ret,frame = cap.read()
    # frame = frame[:75,:460]
def gettime(frame):
    frame = frame[70:120,70:800]
    frame = preprocess(frame)
    text = pytesseract.image_to_string(frame)   
    return text
    # cv2.imshow("Time Frame", frame)
    # key = cv2.waitKey(1) 
    # if key == 27: #esc key stops the process
    #     break
