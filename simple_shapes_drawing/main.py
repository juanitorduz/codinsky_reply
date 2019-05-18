import numpy as np
import cv2
import random

# https://realpython.com/python-opencv-color-spaces/
# https://www.tutorialspoint.com/draw-geometric-shapes-on-images-using-python-opencv-module

# images are normalized between 0 (black) - 255 (white)
MAX_VALUE = 255
MAX_THICKNESS = 30

# circle params
n_rects = 4
n_circles = 9
n_lines = 7
dict_shapes = {
    "rects": n_rects,
    "circles": n_circles,
    "lines": n_lines
}

WIDTH = 896
HEIGHT = 504
# HEIGHT and WIDTH are inverted
img_size = (HEIGHT, WIDTH, 4)


def convert_rgb_bgr(color):
    if len(color) == 4:
        return (color[2], color[1], color[0], color[3])
    elif len(color) == 3:
        return (color[2], color[1], color[0])
    else:
        return -1


# COLORS
alpha_saturn_red = 1
alpha_cinnabar = 1
alpha_blue = 1
alpha_green = 0.1
alpha_yellow = 1
alpha_background = 1
alpha = 0.5
saturn_red = convert_rgb_bgr((195, 130, 87, alpha_saturn_red))
cinnabar = convert_rgb_bgr((227, 66, 52, alpha_cinnabar))
blue = convert_rgb_bgr((86, 130, 162, alpha_blue))
green = convert_rgb_bgr((168, 177, 128, alpha_green))
yellow = convert_rgb_bgr((234, 193, 55, alpha_yellow))
background_color = convert_rgb_bgr((236, 229, 190, alpha_background))

colors = [saturn_red, cinnabar, blue, green, yellow]
#colors = [green, yellow]


def get_rand(max_value):
    r = random.randint(0, max_value - 1)
    # print(r,max_value)
    return r


def main(dict_shapes, colors, size):
    # create the image matrix
    print(size, background_color)
    my_img = np.full(size, background_color, dtype="uint8")

    # iteration through all the dict to create shapes
    for k, v in dict_shapes.items():
        for i in range(v):

            # take a random color from the one given and random thickness
            current_color = colors[get_rand(len(colors))]
            # thickness = get_rand(min(sizes[0], sizes[1]) / 4)
            thickness = get_rand(MAX_THICKNESS)

            if k == "rects":
                #
                thickness = random.randint(0,1)*-1
                vert_1 = (get_rand(size[1]), get_rand(size[0]))
                vert_2 = (get_rand(size[1]), get_rand(size[0]))
                cv2.rectangle(my_img, vert_1, vert_2, current_color, thickness)

            elif k == "circles":
                center = (get_rand(size[1]), get_rand(size[0]))
                radius = (get_rand(min(size[1], size[0]) / 2))
                cv2.circle(my_img, center, radius, current_color, thickness)

            elif k == "lines":
                lineThickness = 2
                # lines are always black
                current_color = (0, 0, 0)
                cv2.line(my_img, (get_rand(size[0]), get_rand(size[1])), (get_rand(size[0]), get_rand(size[1])),
                         current_color, lineThickness)



    cv2.imshow('Window', my_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # convert into bgr
    main(dict_shapes, colors, img_size)
