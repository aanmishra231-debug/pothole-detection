#importing all libraries for function access
import cv2
import os
import numpy as np
import pandas as pd

#get absolute pathing for project
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path=os.path.join(BASE_DIR,"data","pothole_road.jpg")

#image being stored as matrix in bgr format in src
image=cv2.imread(image_path)

if image is None:
    print("Error:Image not found")
    exit()

#pre-processing image

#convert bgr to grayscale, binary output yields better results

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#blurring, helps make contouring easier

blur=cv2.medianBlur(gray,5)

#edge detection using canny, one of the main reasons to use OpenCV

edge=cv2.Canny(blur,50,150)

#making sure the image border doesnt become a contour
h, w = edge.shape

border = 10  # pixels to remove from edges

edge[:border, :] = 0
edge[-border:, :] = 0
edge[:, :border] = 0
edge[:, -border:] = 0


#dilate the edges to make sure lines connect
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edge, kernel, iterations=1)

#contour making

contours, hierarchy = cv2.findContours(
    dilated,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

all_contours = image.copy()
cv2.drawContours(all_contours, contours, -1, (0, 255, 0), 2)

cv2.imshow("All Contours", all_contours)
print("Total contours:", len(contours))

#filter contours to find potholes

contour_view = image.copy()
cv2.drawContours(contour_view, contours, -1, (0, 255, 0), 2)
cv2.imshow("All contours", contour_view)
min_area = 500  # tune later
filtered = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_area:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = w / float(h)

    # Reject vertical & very thin objects
    if aspect_ratio < 0.7 or aspect_ratio > 4.5:
        continue

    filtered.append(cnt)
height = image.shape[0]

final = []

for cnt in filtered:
    x, y, w, h = cv2.boundingRect(cnt)
    center_y = y + h / 2

    if center_y < height * 0.5:
        continue

    final.append(cnt)
result = image.copy()
cv2.drawContours(result, final, -1, (0, 0, 255), 2)
pothole_count = len(final)
cv2.putText(
    result,
    f"Potholes detected: {pothole_count}",
    (20, 40),  # top-left corner
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

cv2.imshow("Final pothole candidates", result)
print("Potholes detected:", len(final))




#Display Statements
cv2.imshow("original", image)
cv2.imshow("grayscale", gray)
cv2.imshow("Blurred",blur)
cv2.imshow("Edges",edge)
cv2.imshow("dilated",dilated)

cv2.waitKey(0)
cv2.destroyAllWindows()