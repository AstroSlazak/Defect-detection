import cv2
import numpy as np
from scipy import ndimage

# Code based on https://link.springer.com/article/10.1186/s13640-017-0187-0#Equ1

# Logarithmic transformations of input image
def log_transformation(img):
    A = 0.7
    B = 1.2
    mask = np.ones((3, 3))
    mask[1, 1] = 0

    average_image = ndimage.generic_filter(img.astype(np.float32), np.nanmean, footprint=mask, mode='constant', cval=np.NaN)
    negative_image = 255 - img
    log_image = np.log(negative_image)
    final_log = A*average_image + B*(log_image - average_image)
    return cv2.convertScaleAbs(final_log)

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

# Connect smalls pixels in larger blobs
def morph(image, kernel = (5, 5), show=False):
    element_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element_closing)
    if show:
        cv2.imshow(f'Kernel: {kernel}', img)
    return img

# Read, resize, and change image from BGR to Gray
image = cv2.imread("current.jpg")
image = cv2.resize(image, (640, 480))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image after transformation to gray", image_gray)
# Applaying Log transformation on gray image
image_log = log_transformation(image_gray)
# cv2.imshow("Image after Logarithm trans", image_log)

#Applaying bilateral filter
image_bilateral = cv2.bilateralFilter(image_log, 5, 35, 16)
# cv2.imshow("Image after Bilateral Filter", image_bilateral)

# Normalize the image after filtering
image_bilateral = cv2.normalize(image_bilateral, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Applaying Canny Edge detections
image_canny = cv2.Canny(image_bilateral, 90, 100)
# cv2.imshow("Image after Canny transformation", image_canny)

# Merge pixels using opencv closing morphology functions
image_morph = morph(image_canny , kernel=(5,5), show=False)

# Remove smalls bloobs
image_conn = component(image_morph, min_size= 125, show=False)
# cv2.imshow("Image after closing edges", image_conn)

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
