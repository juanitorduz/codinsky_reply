import numpy as np
import cv2
import random

# https://realpython.com/python-opencv-color-spaces/
# https://www.tutorialspoint.com/draw-geometric-shapes-on-images-using-python-opencv-module

# images are normalized between 0 (black) - 255 (white)
MAX_VALUE = 255
MAX_THICKNESS = 30

# circle params
n_rects = 3
n_circles = 3
n_lines_parallel = 3
n_lines_bezier = 0
n_lines_vertex = 1

dict_shapes = {
    "rects": n_rects,
    "circles": n_circles,
    "lines_parallel": n_lines_parallel,
    "lines_bezier": n_lines_bezier,
    "lines_vertex": n_lines_vertex
}

WIDTH = 896
HEIGHT = 504

WIDTH_BORDER = 800
HEIGHT_BORDER = 450
OFFSET = 100
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

colors_ = [saturn_red, cinnabar, blue, green, yellow]
# colors = [green, yellow]


Happy = convert_rgb_bgr((227, 66, 52, 89))
Sad = convert_rgb_bgr((72, 35, 80, 31))
Neutral = convert_rgb_bgr((138, 115, 159, 62))
Chic = convert_rgb_bgr((52, 0, 21, 20))
Lonely = convert_rgb_bgr((128, 182, 211, 83))
Honesty = convert_rgb_bgr((227, 66, 52, 89))
Anxiety = convert_rgb_bgr((86, 130, 162, 64))
Exhausted = convert_rgb_bgr((27, 54, 138, 54))
Odd = convert_rgb_bgr((236, 154, 55, 93))
Valuable = convert_rgb_bgr((234, 193, 55, 92))
Exciting = convert_rgb_bgr((168, 177, 128, 69))
Pleasant = convert_rgb_bgr((183, 189, 214, 84))

colors_ = [Happy, Sad, Neutral, Chic, Lonely, Honesty, Anxiety, Exhausted, Odd, Valuable, Exciting, Pleasant]
happy_colors = [Happy, Honesty, Valuable, Exciting, Pleasant]
sad_colors = [Sad, Neutral, Anxiety, Exhausted, Odd]


def get_rand(max_value, min_value=0):
    r = random.randint(min_value, max_value - 1)
    # print(r,max_value)
    return r


def main(dict_shapes, colors, size):
    # create the image matrix
    my_img = np.full(size, background_color, dtype="uint8")
    my_img = np.full(size, Exhausted, dtype="uint8")

    # iteration through all the dict to create shapes
    for k, v in dict_shapes.items():
        for i in range(v):

            # take a random color from the one given and random thickness
            current_color = colors[get_rand(len(colors))]
            # thickness = get_rand(min(sizes[0], sizes[1]) / 4)
            # thickness = get_rand(MAX_THICKNESS)
            thickness = 2

            if k == "rects":

                # thickness = random.randint(0, 1) * -1
                thickness = -1

                vert_1 = (get_rand(WIDTH_BORDER - OFFSET, OFFSET), get_rand(HEIGHT_BORDER - OFFSET))
                vert_2 = (get_rand(WIDTH_BORDER - OFFSET, OFFSET), get_rand(HEIGHT_BORDER - OFFSET))
                cv2.rectangle(my_img, vert_1, vert_2, current_color, thickness)

            elif k == "circles":

                center = (get_rand(WIDTH_BORDER - OFFSET, OFFSET), get_rand(HEIGHT_BORDER - OFFSET, OFFSET))
                # radius = (get_rand(min(size[1], size[0]) / 2))
                radius = (get_rand(min(WIDTH_BORDER, size[0]) / 8))
                thickness = radius
                cv2.circle(my_img, center, radius, current_color, thickness)
                cv2.circle(my_img, center, int(radius * 0.01), background_color, thickness)

            elif k == "lines_parallel":
                lineThickness = 2
                # lines are always black
                current_color = (0, 0, 0)
                x_pos_1 = get_rand(WIDTH_BORDER - OFFSET, OFFSET)
                # to constraint the length of the lines
                x_pos_2 = x_pos_1 + get_rand(100, 50)

                y_pos_1 = get_rand(HEIGHT_BORDER - OFFSET, OFFSET)
                y_pos_2 = y_pos_1 + get_rand(100, 50)

                parallel_offset = 20

                cv2.line(my_img, (x_pos_1, y_pos_1), (x_pos_2, y_pos_2), current_color, lineThickness)
                cv2.line(my_img, (x_pos_1, y_pos_1 + parallel_offset), (x_pos_2, y_pos_2 + parallel_offset),
                         current_color, lineThickness)
                cv2.line(my_img, (x_pos_1, y_pos_1 + parallel_offset * 2), (x_pos_2, y_pos_2 + parallel_offset * 2),
                         current_color, lineThickness)
                cv2.line(my_img, (x_pos_1, y_pos_1 + parallel_offset * 3), (x_pos_2, y_pos_2 + parallel_offset * 3),
                         current_color, lineThickness)

            elif k == "lines_vertex":
                lineThickness = 2
                # lines are always black
                current_color = (0, 0, 0)
                x_pos_1 = get_rand(WIDTH_BORDER - OFFSET, OFFSET)
                # to constraint the length of the lines
                x_pos_2 = x_pos_1 + get_rand(100, 50)

                y_pos_1 = get_rand(HEIGHT_BORDER - OFFSET, OFFSET)
                y_pos_2 = y_pos_1 + get_rand(100, 50)

                parallel_offset = 20
                # 50 %
                if get_rand(2, 0):
                    cv2.line(my_img, (x_pos_1, y_pos_1), (x_pos_2, y_pos_2), current_color, lineThickness)
                    cv2.line(my_img, (x_pos_1, y_pos_1), (x_pos_2, y_pos_2 + parallel_offset * 3),
                             current_color, lineThickness)
                else:
                    cv2.line(my_img, (x_pos_1, y_pos_1), (x_pos_2 + parallel_offset * 3, y_pos_2), current_color,
                             lineThickness)
                    cv2.line(my_img, (x_pos_1, y_pos_1), (x_pos_2, y_pos_2),
                             current_color, lineThickness)


            elif k == "lines_bezier":
                pass

    cv2.imshow('Window', my_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # convert into bgr
    main(dict_shapes, colors_, img_size)
