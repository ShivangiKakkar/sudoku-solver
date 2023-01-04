import cv2
import os
import numpy as np


# Helper functions for getting square image

def euclidian_distance(point1, point2):
    # Calcuates the euclidian distance between the point1 and point2
    # used to calculate the length of the four sides of the square
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance


def order_corner_points(corners):
    # The points obtained from contours may not be in order because of the skewness  of the image, or
    # because of the camera angle. This function returns a list of corners in the right order
    sort_corners = [(corner[0][0], corner[0][1]) for corner in corners]
    sort_corners = [list(ele) for ele in sort_corners]
    x, y = [], []

    for i in range(len(sort_corners[:])):
        x.append(sort_corners[i][0])
        y.append(sort_corners[i][1])

    centroid = [sum(x) / len(x), sum(y) / len(y)]

    for _, item in enumerate(sort_corners):
        if item[0] < centroid[0]:
            if item[1] < centroid[1]:
                top_left = item
            else:
                bottom_left = item
        elif item[0] > centroid[0]:
            if item[1] < centroid[1]:
                top_right = item
            else:
                bottom_right = item

    ordered_corners = [top_left, top_right, bottom_right, bottom_left]

    return np.array(ordered_corners, dtype="float32")


def image_preprocessing(image, corners):
    # This function undertakes all the preprocessing of the image and return
    ordered_corners = order_corner_points(corners)
    print("ordered corners: ", ordered_corners)
    top_left, top_right, bottom_right, bottom_left = ordered_corners

    # Determine the widths and heights  ( Top and bottom ) of the image and find the max of them for transform

    width1 = euclidian_distance(bottom_right, bottom_left)
    width2 = euclidian_distance(top_right, top_left)

    height1 = euclidian_distance(top_right, bottom_right)
    height2 = euclidian_distance(top_left, bottom_right)

    width = max(int(width1), int(width2))
    height = max(int(height1), int(height2))

    # To find the matrix for warp perspective function we need dimensions and matrix parameters
    dimensions = np.array([[0, 0], [width, 0], [width, width],
                           [0, width]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    transformed_image = cv2.warpPerspective(image, matrix, (width, width))

    transformed_image = cv2.resize(transformed_image, (252, 252), interpolation=cv2.INTER_AREA)

    return transformed_image

    # main function


def get_square_box_from_image(image):
    # This function returns the top-down view of the puzzle in grayscale.
    #

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscaled", gray)
    cv2.waitKey(0)

    blur = cv2.medianBlur(gray, 3)
    adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    cv2.imshow("Adaptive Thresh", adaptive_threshold)
    cv2.waitKey(0)

    corners = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corners = corners[0] if len(corners) == 2 else corners[1]
    corners = sorted(corners, key=cv2.contourArea, reverse=True)


    for corner in corners:
        length = cv2.arcLength(corner, True)
        approx = cv2.approxPolyDP(corner, 0.015 * length, True)
        print("Approx: ", approx)
        puzzle_image = image_preprocessing(image, approx)
        cv2.imshow("Warped - Result", puzzle_image)
        cv2.waitKey(0)

        break

    return puzzle_image

    # Call the get_square_box_from_image method on any sudoku image to get the top view of the puzzle





if __name__ == '__main__':
    original = cv2.imread("images\p3.jpg")
    sudoku = get_square_box_from_image(original)
    grid = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY) # VERY IMPORTANT
    grid = cv2.bitwise_not(cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1))
    cv2.imshow('gt',grid)
    cv2.waitKey(0)

    edge_h = np.shape(grid)[0]
    edge_w = np.shape(grid)[1]
    celledge_h = edge_h // 9
    celledge_w = np.shape(grid)[1] // 9

    tempgrid = []

    for i in range(celledge_h, edge_h + 1, celledge_h):
        for j in range(celledge_w, edge_w + 1, celledge_w):
            rows = grid[i - celledge_h:i]
            tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

    # Creating the 9X9 grid of images
    finalgrid = []
    for i in range(0, len(tempgrid) - 8, 9):
        finalgrid.append(tempgrid[i:i + 9])
    # Converting all the cell images to np.array
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])
    try:
        for i in range(9):
            for j in range(9):
                os.remove("BoardCells/cell" + str(i) + str(j) + ".jpg")
    except:
        pass

    for i in range(9):
        for j in range(9):
            cv2.imwrite(str("BoardCells/cell" + str(i) + str(j) + ".jpg"), finalgrid[i][j])

    print(len(finalgrid[0]), len(finalgrid))