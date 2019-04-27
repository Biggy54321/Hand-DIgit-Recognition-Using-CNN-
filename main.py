# get the required libraries
import os
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk as gtk
import cv2
import numpy as np
import process_image as pi

# define event for UPLOAD button
def upload_file_from_system(button):
    # create a browser window aka file choose dialog box
    dialog = gtk.FileChooserDialog("Upload Image", None, 0,
                        (gtk.STOCK_CANCEL, gtk.ResponseType.CANCEL,
                         gtk.STOCK_OK, gtk.ResponseType.OK))
    dialog.set_default_response(gtk.ResponseType.OK)

    # get the user action on the dialob box
    response = dialog.run()

    # if user clicks ok then get the file name that is uploaded
    if response == gtk.ResponseType.OK:
        image_path = dialog.get_filename()
    # if user clicks cancel then do nothing
    elif response == gtk.ResponseType.CANCEL:
        pass

    # close the file choose dialog box
    dialog.destroy()

    # open the image at the obtained path from the dialog box
    image = cv2.imread(image_path)

    # pass the image for prediction
    result_image = pi.recognize_digits(image)

    # write the image to the current directory
    cv2.imwrite('result_image.jpg', result_image)
    result_image_path = os.getcwd() + '/result_image.jpg'
    
    # display the image on the window
    image = gtk.Image()
    image.set_from_file(result_image_path)
    fixed.put(image, 600, 250)
    window.add(fixed)
    window.show_all()

    # delete the saved image
    os.remove(result_image_path)

# define the event for CAPTURE button
def capture_image_from_webcam(button):
    # start the webcam
    capture = cv2.VideoCapture(0)
    while(True):
        # read a frame from the video
        ret, frame = capture.read()
        
        # display each frame
        cv2.imshow("Capture Image (Press 'c' to capture)", frame)
        
        # if the capture key is pressed then stop recording and predict the answer for that frame
        if(cv2.waitKey(1) & 0xFF == ord('c')):
            image = frame
            break
        
    # close the webcam
    capture.release()
    # close the image window
    cv2.destroyAllWindows()

    # predict for the captured image
    result_image = pi.recognize_digits(image)

    # write the image to the current directory
    cv2.imwrite('result_image.jpg', result_image)
    result_image_path = os.getcwd() + '/result_image.jpg'
    
    # display the image on the window
    image = gtk.Image()
    image.set_from_file(result_image_path)
    fixed.put(image, 600, 250)
    window.add(fixed)
    window.show_all()

    # delete the saved image
    os.remove(result_image_path)

# define the function for LIVE CAMERA button
def predict_live_from_webcam(button):
    # start the webcam
    capture = cv2.VideoCapture(0)
    frame_count = 0
    while(True):
        # read a frame for every iteration
        ret, frame = capture.read()

        # predict the result for every frame
        result_frame = pi.recognize_digits(frame)
        cv2.imshow("Live Prediction (Press 'q' to exit)", result_frame)
        
        # set q to quit the functioning
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # close the webcam
    capture.release()
    #close the iamge window
    cv2.destroyAllWindows()

# define the event for GESTURE button
def draw_the_digit_by_hand(button):

    # define the require variables
    activate_flag = 0
    masked_frames = []
    result = np.zeros((480, 640, 3), np.uint8)
    
    # define the upper and lower limits for the blue color which is accepted while drawing
    lower_blue = np.array([85, 160, 40])
    upper_blue = np.array([160, 255, 110])
    
    # start the webcam
    cap = cv2.VideoCapture(0)    
    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # check if the user presses the required key if yes then toggle the mode
        if (cv2.waitKey(1) & 0xFF == ord('s')):
            if activate_flag == 0:
                activate_flag = 1
            else:
                break;
        # check if drawing mode is activated if yes then start recording the drawn frames
        if activate_flag == 1:
            # get the saturated frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # mask all the colors except blue specified by the given ranges
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            masked_frames.append(mask)

            # display the masked image
            result = np.bitwise_or(result, cv2.bitwise_and(frame, frame, mask = mask))
            cv2.imshow("Gesture Draw (press 's' to stop recording)", result)
        else:
            cv2.imshow("Gesture Draw (press 's' to start recording)", frame)
            
    # stop the webcam
    cap.release()
    cv2.destroyAllWindows()

    # add up all the frames to get the drawn image
    image = masked_frames[0]
    for i in range(1, len(masked_frames)):
        image = np.bitwise_or(image, masked_frames[i])

    # erode the image
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations = 1)

    # predict
    result_image = pi.recognize_gesture(image)
    
    # write the image to the current directory
    cv2.imwrite('result_image.jpg', result_image)
    result_image_path = os.getcwd() + '/result_image.jpg'
    
    # display the image on the window
    image = gtk.Image()
    image.set_from_file(result_image_path)
    fixed.put(image, 600, 250)
    window.add(fixed)
    window.show_all()

    # delete the saved image
    os.remove(result_image_path)

# exit button event to close the window
def close_window(exit):
    gtk.main_quit()
    
# create a window
window = gtk.Window()
window.set_title("Digit Recognition")

# adding an image in the background which requires an overlay
overlay = gtk.Overlay()
window.add(overlay)
background = gtk.Image.new_from_file('images/slider-bg-1.jpg')
overlay.add(background)
window.add(overlay)

# create a grid which will hold all the buttons and the labels
grid = gtk.Grid()

# create buttons and label objects
label = gtk.Label("DIGIT RECOGNITION")	
button1 = gtk.Button("UPLOAD...")
button1.set_size_request(130, 60)
button2 = gtk.Button("CAPTURE")
button2.set_size_request(130, 60)
button3 = gtk.Button("LIVE CAMERA")
button3.set_size_request(130, 60)
button4 = gtk.Button("GESTURE")
button4.set_size_request(130, 60)
exit = gtk.Button("EXIT")
exit.set_size_request(130, 60)

# fix the buttons and labels on the window
fixed = gtk.Fixed()
fixed.put(button1, 1700, 110)
fixed.put(button2, 1700, 310)
fixed.put(button3, 1700, 510)
fixed.put(button4, 1700, 710)
fixed.put(exit, 1700, 910)
fixed.put(label, 900, 30)

# add the fixed object to the grid
grid.add(fixed)

# link the buttons and their event functions
button1.connect("clicked", upload_file_from_system)
button2.connect("clicked", capture_image_from_webcam)
button3.connect("clicked", predict_live_from_webcam)
button4.connect("clicked", draw_the_digit_by_hand)
exit.connect("clicked", close_window)


# run the window
overlay.add_overlay(grid)
window.add(fixed)
window.connect("destroy", gtk.main_quit)
window.fullscreen()
window.show_all()
gtk.main()
