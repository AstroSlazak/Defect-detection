import cv2
import numpy as np

# Connect smalls pixels in larger blobs
def morph(image, kernel = (5, 5), show=False):
    element_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element_closing)
    if show:
        cv2.imshow(f'Kernel: {kernel}', img)
    return img

# Remove elements smaller than minimum size
# https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv
def component(image, connectivity = 8, min_size = 100, show=False):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity= connectivity)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img[output == i + 1] = 255
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    if show:
        cv2.imshow(f'Min size: {min_size}', img)
    return img

# Approximation of the contour
def approx_cont(contours, look_back):
    contour = []
    for n in range(len(contours)):
        cont = []
        cont_back = np.concatenate([contours[n][-look_back:], contours[n]])
        for i in range(len(cont_back[:-look_back])):
            inner_list = []
            slice = cont_back[i: i + (2*look_back)].mean(axis=0)
            inner_list.append(int(slice[0][0]))
            inner_list.append(int(slice[0][1]))
            cont.append(inner_list)
        contour.append(np.array(cont).reshape((-1, 1,2)).astype(np.int32))
    return contour

# Read and resize image
image = cv2.imread('scratch_3.jpg')
image = cv2.resize(image, (640, 480))
cv2.imshow("Base image", image)

# Applaying Gaussian Blur on input image
image_blurred = cv2.GaussianBlur(image, (25, 25),0)

# Applaying Sobel Edge detections
sobelX = cv2.Sobel(image_blurred ,cv2.CV_64F,1,0, ksize=-1)
sobelY = cv2.Sobel(image_blurred,cv2.CV_64F,0,1, ksize=-1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
image_sobel = cv2.bitwise_or(sobelX, sobelY)

# Change image from BGR to Gray
imageGRAY = cv2.cvtColor(image_sobel, cv2.COLOR_BGR2GRAY)

# Again applaying Gaussian Blur
imageGRAY = cv2.GaussianBlur(imageGRAY, (7,7),0)
# cv2.imshow("Base image", imageGRAY)

# Change pixels greater than 0,3*max image pixel to 255
imageGRAY[imageGRAY > 0.3*np.max(imageGRAY)] = 255

# Change pixels values to 0 if is < 126 otherwise to 255
_, imageGRAY = cv2.threshold(imageGRAY, 126, 255, cv2.THRESH_BINARY)

# Merge pixels using opencv closing morphology functions
image_morph = morph(imageGRAY, kernel=(3,3), show=False)
# Remove smalls bloobs
image_conn = component(image_morph, min_size= 250, show=False)
# Merge pixels using opencv closing morphology functions
image_morph = morph(image_conn, kernel=(11,11), show=False)
# Remove smalls bloobs
image_conn = component(image_morph, min_size= 750, show=False)

# Normalize image
image_norm = cv2.normalize(image_conn, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Finding contours on binary image
_, cont, hierarchy = cv2.findContours(image_norm.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Approximation contours by mean of 10 points before and 10 points after
new_cont = approx_cont(cont, look_back=10)

# Draw raw contours on input image
image_cont = cv2.drawContours(image.copy(), cont ,-1, (0,0,255), 2)
cv2.imshow('Image with raw contours', image_cont)

# Draw approximated contours on input image
image_cont_approx = cv2.drawContours(image.copy(), new_cont ,-1, (0,0,255), 2)
cv2.imshow('Image with approx contours', image_cont_approx)

# Draw mask on image
# image_cont_approx_mask = cv2.drawContours(image.copy(), new_cont ,-1, (0,0,255), -1)
# image_final = cv2.addWeighted(image_cont_approx_mask, 0.5, image, 1 - 0.5, 0, image)
# cv2.imshow("Image with mask", image_final)

cv2.waitKey(0)
