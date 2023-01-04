# DataFlair Sudoku solver

import cv2
import numpy as np

classes = np.arange(0, 10)
from .preproc import get_square_box_from_image

# print(model.summary())
input_size = 48

# split the board into 81 individual images
def split_boxes(board):
    """Takes a sudoku board and split it into 81 cells.
        each cell contains an element of that board either given or an empty cell."""
    rows = np.vsplit(board,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0
            cv2.imshow("Splitted block", box)
            cv2.waitKey(1)
            boxes.append(box)


    cv2.destroyAllWindows()
    return boxes

def displayNumbers(img, numbers, color=(0, 255, 0)):
    """Displays 81 numbers in an image or mask at the same position of each cell of the board"""
    W = int(img.shape[1]/9)
    H = int(img.shape[0]/9)
    for i in range (9):
        for j in range (9):
            if numbers[(j*9)+i] !=0:
                cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int((j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img


def image_to_array(impath, model):

    img = cv2.imread(impath)
    cv2.imshow("Input Image", img)
    cv2.waitKey(0)

    img = get_square_box_from_image(img)

    bufferimg = cv2.resize(img, (450, 450))

    gray = cv2.cvtColor(bufferimg, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    rois = split_boxes(gray)
    rois = np.array(rois).reshape(-1, input_size, input_size, 1)

    # get prediction
    prediction = model.predict(rois)
    # print(prediction)

    predicted_numbers = []
    # get classes from prediction
    for i in prediction:
        index = (np.argmax(i))  # returns the index of the maximum number of the array
        predicted_number = classes[index]
        predicted_numbers.append(predicted_number)

    return predicted_numbers

if __name__ == '__main__':
    from tensorflow.keras.models import load_model

    model = load_model('imdetect\model-OCR.h5')

    grid_list = image_to_array('images\\test_pictures\Foto2.jpg', model=model)
    grid = np.reshape(grid_list, (9, 9))

    print(grid_list)
    print(grid)