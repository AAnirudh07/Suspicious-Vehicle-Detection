import cv2

#change the video path & result path basedn on your system
video_path  = "D:/Projects/vehicle classification/data/videos/Ch8_20220112161012.mp4"
result_path = "D:/Projects/vehicle classification/data/frames"


cap = cv2.VideoCapture(video_path)
i = 0
 
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
     
    # Save Frame by Frame into disk using imwrite method
    cv2.imwrite(result_path+ "/" + "Frame" + str(i) + ".jpg", frame)
    i += 1
 
cap.release()
cv2.destroyAllWindows()