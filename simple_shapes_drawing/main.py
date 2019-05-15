import numpy as np
import cv2


# https://realpython.com/python-opencv-color-spaces/
# https://www.tutorialspoint.com/draw-geometric-shapes-on-images-using-python-opencv-module


def draw_circle(sizes, colors, radius):
    my_img = np.zeros(sizes, dtype="uint8")
    # creating circle
    cv2.circle(my_img, (200, 200), radius, colors, 50)
    cv2.imshow('Window', my_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


sizes = (400, 400, 3)
colors = (0, 20, 200)
radius = 50
draw_circle(sizes, colors, radius)
