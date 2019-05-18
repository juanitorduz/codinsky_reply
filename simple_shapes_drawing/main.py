import numpy as np
import cv2
import random

# https://realpython.com/python-opencv-color-spaces/
# https://www.tutorialspoint.com/draw-geometric-shapes-on-images-using-python-opencv-module

# images are normalized between 0 (black) - 255 (white)
MAX_VALUE = 255
MAX_THICKNESS = 100

# circle params
n_rects = 2
n_circles = 2
dict_shapes = {"rects": n_rects, "circles": n_circles}
WIDTH = 896
HEIGHT = 504
img_size = (HEIGHT, WIDTH, 3)


def convert_rgb_bgr(color):
    return (color[2], color[1], color[0])


# COLORS
saturn_red = convert_rgb_bgr((195, 130, 87))
cinnabar = convert_rgb_bgr((227, 66, 52))
blue = convert_rgb_bgr((86, 130, 162))
green = convert_rgb_bgr((168, 177, 128))
yellow = convert_rgb_bgr((234, 193, 55))
background_color = convert_rgb_bgr((236, 229, 190))

colors = [saturn_red, cinnabar, blue, green, yellow]


def get_rand(max_value):
    r = random.randint(0, max_value - 1)
    # print(r,max_value)
    return r


def main(dict_shapes, colors, sizes):
    # create the image matrix
    my_img = np.full(sizes, background_color, dtype="uint8")

    # iteration through all the dict to create shapes
    for k, v in dict_shapes.items():
        for i in range(v):

            # take a random color from the one given and random thickness
            current_color = colors[get_rand(len(colors))]
            # thickness = get_rand(min(sizes[0], sizes[1]) / 4)
            thickness = get_rand(MAX_THICKNESS)

            if k == "rects":
                vert_1 = (get_rand(sizes[0]), get_rand(sizes[1]))
                vert_2 = (get_rand(sizes[0]), get_rand(sizes[1]))
                cv2.rectangle(my_img, vert_1, vert_2, current_color, thickness)
            elif k == "circles":
                center = (get_rand(sizes[0]), get_rand(sizes[1]))
                radius = (get_rand(min(sizes[0], sizes[1]) / 2))

                cv2.circle(my_img, center, radius, current_color, thickness)

    cv2.imshow('Window', my_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # convert into bgr
    main(dict_shapes, colors, img_size)
