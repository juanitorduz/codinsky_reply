import numpy as np
import cv2


# https://realpython.com/python-opencv-color-spaces/
# https://www.tutorialspoint.com/draw-geometric-shapes-on-images-using-python-opencv-module


def draw_circle(sizes, center, radius, colors, thickness):
    my_img = np.zeros(sizes, dtype="uint8")
    # creating circle
    cv2.circle(my_img, center, radius, colors, thickness)
    cv2.imshow('Window', my_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


sizes = (400, 400, 3)
colors = (0, 20, 200)
radius = 50
center = (200, 200)
thickness = -1
draw_circle(sizes, center, radius, colors)
