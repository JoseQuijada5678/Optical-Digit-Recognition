import matplotlib.pyplot as plt

def display_sample(image,label,predicted=999):
    #Reshape the 768 values to a 28x28 image
    plt.title('Label: %d, Predicted: %d' % (label, predicted))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
    

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