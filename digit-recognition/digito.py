import cv2
import imutils
import numpy as np
from keras.models import model_from_json
import keras
from keras import backend as K


def get_img_contour_thresh(img):
    x = 0
    y = 0
    h, w, _ = img.shape
    # x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh1 = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

"""img = cv2.imread("9.jpg")
h, w, _ = img.shape
img, contours, thresh = get_img_contour_thresh(img)

ans = ''

if len(contours) > 0:
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) > 500:
        x, y, w, h = cv2.boundingRect(contour)
        newImage = thresh[y:y + h, x:x + w]
        newImage = cv2.resize(newImage, (28, 28))
        newImage = np.array(newImage)
        newImage = newImage.astype('float32')
        newImage /= 255

        if K.image_data_format() == 'channels_first':
            newImage = newImage.reshape(1, 28, 28)
        else:
            newImage = newImage.reshape(28, 28, 1)
        newImage = np.expand_dims(newImage, axis=0)
        ans = model.predict(newImage).argmax()

x = 0
y = 0
h, w, _ = img.shape
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(img, "CNN : " + str(ans), (10, 320),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imshow("Frame", img)
cv2.imshow("Contours", thresh)
print(ans)

cv2.waitKey(0)
cv2.destroyAllWindows()"""

# Read the input image
im = cv2.imread("digitos.jpg")
# im = cv2.resize(im, (200, 200))

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
# print(ret)
# print(im_th)
# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
# ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.

for rect in rects:
    #contorno = max(rect, key=cv2.contourArea)
    #print(cv2.contourArea(contorno))
    area = abs(((rect[2] - rect[0]) * (rect[3]) - rect[1]))
    print(area)
    if area > 3000:
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # print(rect)

        try:
            newImage = cv2.resize(roi, (28,28))
            newImage = np.array(newImage)
            newImage = newImage.astype('float32')
            newImage /= 255

            if K.image_data_format() == 'channels_first':
                newImage = newImage.reshape(1, 28, 28)
            else:
                newImage = newImage.reshape(28, 28, 1)
            newImage = np.expand_dims(newImage, axis=0)
            nbr = model.predict(newImage).argmax()
            print(nbr)
            cv2.putText(im, str(int(nbr)), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        except cv2.error:
            print("uy")
            nbr = 0
            cv2.putText(im, str(int(0)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        # Resize the image
        # roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        #roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        # roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        # nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        # nbr = "2"
        # cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.imshow("cour", im_th)
cv2.waitKey(0)