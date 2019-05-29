import cv2
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
