import numpy as np
import cv2
import random


def get_rand(max_value, min_value=0):
    r = random.randint(min_value, max_value - 1)
    # print(r,max_value)
    return r


# https://realpython.com/python-opencv-color-spaces/
# https://www.tutorialspoint.com/draw-geometric-shapes-on-images-using-python-opencv-module

def main():
    # images are normalized between 0 (black) - 255 (white)
    MAX_VALUE = 255
    MAX_THICKNESS = 30

    WIDTH = 896
    HEIGHT = 504

    WIDTH_BORDER = 800
    HEIGHT_BORDER = 450
    OFFSET = 100
    # HEIGHT and WIDTH are inverted
    size = (HEIGHT, WIDTH, 4)

    def convert_rgb_bgr(color):
        if len(color) == 4:
            return (color[2], color[1], color[0], color[3])
        elif len(color) == 3:
            return (color[2], color[1], color[0])
        else:
            return -1

    input_dict = {'Sentiment': {'neg': 0.508, 'neu': 0.492, 'pos': 0.0, 'compound': -0.4767},
                  'Category': 'happy',
                  'Style': 'circle, vertex',
                  'Color': [227, 66, 52, 89]}

    # determine background
    compund = input_dict["Sentiment"]["compound"]
    if compund >= 0:
        background_color = convert_rgb_bgr((27, 54, 138, 54))
    else:
        background_color = convert_rgb_bgr((52, 0, 21, 20))

    val_rect = 10
    val_circle = 10
    val_lines = 10
    # circle params
    n_rects = int(val_rect * abs(compund))
    n_circles = int(val_circle * abs(compund))
    n_lines_parallel = int(val_lines * abs(compund))
    n_lines_bezier = 0
    n_lines_vertex = 0
    n_triangles = int(val_lines * abs(compund))

    dict_shapes = {
        "rects": n_rects,
        "circles": n_circles,
        "lines_parallel": n_lines_parallel,
        "lines_bezier": n_lines_bezier,
        "lines_vertex": n_lines_vertex,
        "triangles": n_triangles
    }

    colors = input_dict["Color"]

    # create the image matrix
    my_img = np.full(size, background_color, dtype="uint8")

    # iteration through all the dict to create shapes
    for k, v in dict_shapes.items():
        for i in range(v):

            # take a random color from the one given and random thickness
            if len(colors) > 1:
                random_color_index = get_rand(len(colors))
                current_color = colors[random_color_index]
            else:
                current_color = colors
            # thickness = get_rand(min(sizes[0], sizes[1]) / 4)
            # thickness = get_rand(MAX_THICKNESS)
            thickness = 2

            if k == "rects":

                # thickness = random.randint(0, 1) * -1
                thickness = -1

                vert_1 = (get_rand(WIDTH_BORDER - OFFSET, OFFSET), get_rand(HEIGHT_BORDER - OFFSET))
                vert_2 = (get_rand(WIDTH_BORDER - OFFSET, OFFSET), get_rand(HEIGHT_BORDER - OFFSET))
                cv2.rectangle(my_img, vert_1, vert_2, current_color, thickness)
                # remove
                # colors.remove(current_color)
            elif k == "circles":

                center = (get_rand(WIDTH_BORDER - OFFSET, OFFSET), get_rand(HEIGHT_BORDER - OFFSET, OFFSET))
                # radius = (get_rand(min(size[1], size[0]) / 2))
                radius = (get_rand(min(WIDTH_BORDER, size[0]) / 8))
                thickness = radius
                cv2.circle(my_img, center, radius, current_color, thickness)
                cv2.circle(my_img, center, int(radius * 0.01), background_color, thickness)
                # remove
                # colors.remove(current_color)

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

                print("first points {},{}".format((x_pos_1, y_pos_1), (x_pos_2, y_pos_2)))
                print("second points {},{}".format((x_pos_1, y_pos_1 + parallel_offset),
                                                   (x_pos_2, y_pos_2 + parallel_offset)))
                print("third points {},{}".format((x_pos_1, y_pos_1 + parallel_offset * 2),
                                                  (x_pos_2, y_pos_2 + parallel_offset * 2)))
                print("fourth points {},{}".format((x_pos_1, y_pos_1 + parallel_offset * 3),
                                                   (x_pos_2, y_pos_2 + parallel_offset * 3)))

                print("NEW")
                m = (y_pos_1 - y_pos_2) / (x_pos_1 - x_pos_2)
                new_m = - ((x_pos_1 - x_pos_2) / (y_pos_1 - y_pos_2))
                b = int(y_pos_1 - m * x_pos_1)
                print(b)
                new_y_1 = int((new_m * x_pos_1) + (b + m * x_pos_1))
                new_y_2 = int((new_m * x_pos_2) + (b + m * x_pos_2))

                cv2.line(my_img, (x_pos_1 + 4, new_y_1), (x_pos_2 + 4, new_y_2), current_color, lineThickness)
                cv2.line(my_img, (x_pos_1 + 8, new_y_1), (x_pos_2 + 8, new_y_2), current_color, lineThickness)
                cv2.line(my_img, (x_pos_1 + 16, new_y_1), (x_pos_2 + 16, new_y_2), current_color, lineThickness)

                print("first points {},{}".format((x_pos_1 + 4, new_y_1), (x_pos_2 + 4, new_y_2)))
                print("second points {},{}".format((x_pos_1 + 8, new_y_1), (x_pos_2 + 8, new_y_2)))
                print("third points {},{}".format((x_pos_1 + 16, new_y_1), (x_pos_2 + 16, new_y_2)))


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

            elif k == "triangles":
                triangle = np.array([[100, 300], [400, 800], [100, 900]], np.int32)
                cv2.fillConvexPoly(my_img, triangle, current_color)
                # remove
                # colors.remove(current_color)

            elif k == "lines_bezier":
                pass

    cv2.imshow('Window', my_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
