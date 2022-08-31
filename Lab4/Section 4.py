import cv2
import os

cam = cv2.VideoCapture(0)

ret, img = cam.read()

path = "C:/Users/Amar/Desktop"

if not ret:
    print("Failed")

if ret:
    
    os.chdir(path)
    
    cv2.namedWindow("cam-test", cv2.WINDOW_NORMAL)
    
    cv2.imshow("cam-test", img)
    
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    cv2.imwrite("slika.jpg", img)
