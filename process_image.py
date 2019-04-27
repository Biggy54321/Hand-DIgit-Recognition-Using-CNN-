# get the required libraries (modules)
import cv2
import numpy as np
import cnn

# create a network model and restore its trained status from the checkpoint files
model = cnn.CNN()
model.restore_model()

# function to recognize the digits in an image and print the prediction above them
def recognize_digits(image):
    # convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smoothen the image using Gaussian Blur
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # threshold the image in a inverted manner to get the digit in white and background in black
    _, image_threshold = cv2.threshold(image_gray, 90, 255, cv2.THRESH_BINARY_INV)
    image_threshold = cv2.resize(image_threshold, (600, 600))

    # find the contours of the digits in the image
    _, contours, _ = cv2.findContours(image_threshold.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # finds the bounding rectangle for each of the recognized contour
    rectangle = []
    for itr in contours:
        rectangle = rectangle + [cv2.boundingRect(itr)]

    # get the copies of thresholded image and the original image
    image_threshold_copy = image_threshold.copy()
    image_copy = cv2.resize(image, (600, 600))

    # create the rectangles and write the predicted text on the resized original image
    for rec in rectangle:
        if (((rec[2]) > (image_threshold_copy.shape[1] / 20)) or ((rec[3]) > (image_threshold_copy.shape[0] / 20))):

            # determine the padding
            padding = rec[3] // 6

            #check if coordinates of rectangle are out of bounds
            if((rec[1] - padding > 600) or (rec[1] - padding < 0)):
                continue
            if((rec[1] + rec[3] + padding > 600) or (rec[1] + rec[3] + padding < 0)):
                continue
            if((rec[0] - padding > 600) or (rec[0] - padding < 0)):
                continue
            if((rec[0] + rec[2] +  padding > 600) or (rec[0] + rec[2] + padding < 0)):
                continue            
            
            # extract the digit from the thresholded image using its rectangular bounds
            feed_image = image_threshold_copy[rec[1] - padding: (rec[1] + rec[3] + padding), rec[0] - padding: (rec[0] + rec[2] + padding)]
            
            # resize the image to be fed to the network to 28, 28
            feed_image = cv2.resize(feed_image, (28, 28))

            # draw the rectangle around the fed image in the original resized image
            cv2.rectangle(image_copy, (rec[0], rec[1]), (rec[0] + rec[2], rec[1] + rec[3]), (255, 0, 170), 2)

            # predict the digit by feeding the current cropped image
            prediction = model.predict(feed_image.reshape(1, 28, 28, 1).astype(np.float32))

            # write the prediction above the rectangle in the original resized image 0, 0, 255
            cv2.putText(image_copy, str(int(prediction)), (rec[0] + (rec[2] // 2), rec[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)

    return image_copy

def recognize_gesture(image):
    # smoothen the image using Gaussian Blur
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # threshold the image in a inverted manner to get the digit in white and background in black
    _, image_threshold = cv2.threshold(image_blur, 90, 255, cv2.THRESH_BINARY)
    image_threshold = cv2.resize(image_threshold, (600, 600))

    # find the contours of the digits in the image
    _, contours, _ = cv2.findContours(image_threshold.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # finds the bounding rectangle for each of the recognized contour
    rectangle = []
    for itr in contours:
        if cv2.contourArea(itr) > 2000:
            rectangle = rectangle + [cv2.boundingRect(itr)]

    # get the copies of thresholded image and the original image
    image_threshold_copy = image_threshold.copy()
    image_copy = cv2.resize(image, (600, 600))

    # create the rectangles and write the predicted text on the resized original image
    for rec in rectangle:
        if (((rec[2]) > (image_threshold_copy.shape[1] / 20)) or ((rec[3]) > (image_threshold_copy.shape[0] / 20))):

            # determine the padding
            padding = rec[3] // 6

            #check if coordinates of rectangle are out of bounds
            if((rec[1] - padding > 600) or (rec[1] - padding < 0)):
                continue
            if((rec[1] + rec[3] + padding > 600) or (rec[1] + rec[3] + padding < 0)):
                continue
            if((rec[0] - padding > 600) or (rec[0] - padding < 0)):
                continue
            if((rec[0] + rec[2] +  padding > 600) or (rec[0] + rec[2] + padding < 0)):
                continue            
            
            # extract the digit from the thresholded image using its rectangular bounds
            feed_image = image_threshold_copy[rec[1] - padding: (rec[1] + rec[3] + padding), rec[0] - padding: (rec[0] + rec[2] + padding)]
            
            # resize the image to be fed to the network to 28, 28
            feed_image = cv2.resize(feed_image, (28, 28))

            # draw the rectangle around the fed image in the original resized image
            cv2.rectangle(image_copy, (rec[0], rec[1]), (rec[0] + rec[2], rec[1] + rec[3]), (255, 255, 255), 2)

            # predict the digit by feeding the current cropped image
            prediction = model.predict(feed_image.reshape(1, 28, 28, 1).astype(np.float32))

            # write the prediction above the rectangle in the original resized image 0, 0, 255
            cv2.putText(image_copy, str(int(prediction)), (rec[0] + (rec[2] // 2), rec[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)

    return image_copy
