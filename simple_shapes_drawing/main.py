import numpy as np
import cv2

#pip install opencv-python
#https://realpython.com/python-opencv-color-spaces/
#https://www.tutorialspoint.com/draw-geometric-shapes-on-images-using-python-opencv-module
def draw_circle():
    my_img = np.zeros((400, 400, 3), dtype="uint8")
    # creating circle
    cv2.circle(my_img, (200, 200), 80, (0, 20, 200), 10)
    cv2.imshow('Window', my_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
