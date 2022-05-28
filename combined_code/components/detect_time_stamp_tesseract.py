import os
import cv2

'''
cap = cv2.VideoCapture(PATH)
ret, frame = cap.read()
frame = frame[:100,:500]
cv2.imshow("Cropped Image", frame)
cv2.waitKey()
'''

PATH = "D:/Projects/vehicle classification/data/videos/Ch8_20220112161012.mp4"

import cv2
import pytesseract
import argparse

def preprocess(image, target=30):
    '''
    Preprocesses the image for Tesseract
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[:100,:500]

    # correct the text height
    ratio = max(0.5, min(1.5, target / image.shape[0]))
    image = cv2.resize(image,
                       dsize=None,
                       fx=ratio,
                       fy=ratio,
                       interpolation=cv2.INTER_LANCZOS4)

    # apply Otsu thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

cap = cv2.VideoCapture(PATH)
ret = True
while True:
    ret,frame = cap.read()
    frame = preprocess(frame)
    # use Tesseract to OCR the image
    text = pytesseract.image_to_string(frame)   
    print(text)
    cv2.imshow("",frame)
    key = cv2.waitKey(5) 
    if key == 27: #esc key stops the process
        break
