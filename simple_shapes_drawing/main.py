import numpy as np
import cv2

# https://realpython.com/python-opencv-color-spaces/
# https://www.tutorialspoint.com/draw-geometric-shapes-on-images-using-python-opencv-module

# images are normalized between 0 - 255 
MAX_VALUE = 255
# circle params
sizes_circle = (400, 400, 3)
colors_circle = (0, 20, 200)
radius_circle = 50
center_circle = (200, 200)
thickness_circle = -1
## rect
thickness_rect = 1
vert_1 = (210, 200)
vert_2 = (310, 100)



def removal_background():
    file_name = "python_grey.png"

    src = cv2.imread(file_name, 1)
    print(src)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    cv2.imwrite("test.png", dst)



def draw_circle(sizes, center, radius, colors, thickness, save=False):
    my_img = np.ones(sizes, dtype="uint8")
    my_img *= MAX_VALUE
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




def draw_rectangle(sizes, vert_1, vert_2, colors, thickness, save=False):
    my_img = np.zeros(sizes, dtype="uint8")
    # creating circle
    cv2.rectangle(my_img, vert_1, vert_2, colors, thickness)
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
    #draw_rectangle(sizes_circle, vert_1, vert_2, colors_circle, thickness_rect)
    draw_circle(sizes_circle, center_circle, radius_circle, colors_circle, thickness_circle, True)
    #removal_background()
