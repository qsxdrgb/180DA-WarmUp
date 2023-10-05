'''
REFERENCES:
https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097

'''



import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Display the resulting frame
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500: #GET BIG BOUNDARIES
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            height, width, _ = frame.shape

            roi_width = width // 4
            roi_height = height // 4

            x = (width - roi_width) // 2
            y = (height - roi_height) // 2

            roi = frame[y:y+roi_height, x:x+roi_width]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = roi.reshape((roi.shape[0] * roi.shape[1], 3))
            # KMeans clustering on the ROI
            clt = KMeans(n_clusters=3)
            clt.fit(roi)

            hist = find_histogram(clt)
            bar = plot_colors2(hist, clt.cluster_centers_)

            bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)
            cv2.imshow("Dominant Colors", bar)
            break


    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)


    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break




# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()