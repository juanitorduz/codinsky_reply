import numpy as np
import cv2

# https://realpython.com/python-opencv-color-spaces/
# https://www.tutorialspoint.com/draw-geometric-shapes-on-images-using-python-opencv-module

# circle params
sizes_circle = (400, 400, 3)
colors_circle = (0, 20, 200)
radius_circle = 50
center_circle = (200, 200)
thickness_circle = -1


def draw_circle(sizes, center, radius, colors, thickness, save=False):
    my_img = np.zeros(sizes, dtype="uint8")
    # creating circle
    cv2.circle(my_img, center, radius, colors, thickness)

    if save:
        try:
            # save the image
            cv2.imwrite('python_grey.png', my_img)
        except Exception as ex:
            print("Exception saving the image :\n'{}'".format(ex))
    else:
        # show the image in an external window
        cv2.imshow('Window', my_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    draw_circle(sizes_circle, center_circle, radius_circle, colors_circle, thickness_circle)
