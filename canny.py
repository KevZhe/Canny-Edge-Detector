import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 7x7 Gaussian mask
GAUSSIAN = np.array([
    [1, 1, 2, 2, 2, 1, 1],
    [1, 2, 2, 4, 2, 2, 1],
    [2, 2, 4, 8, 4, 2, 2],
    [2, 4, 8, 16, 8, 4, 2],
    [2, 2, 4, 8, 4, 2, 2],
    [1, 2, 2, 4, 2, 2, 1],
    [1, 1, 2, 2, 2, 1, 1]
])

# 3x3 masks
ZERO_DEGREES = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

FORTY_FIVE_DEGREES = np.array([
    [0, 1, 2],
    [-1, 0, 1],
    [-2, -1, 0]
])

NINETY_DEGREES = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

ONE_HUNDRED_THIRTY_FIVE_DEGREES = np.array([
    [2, 1, 0],
    [1, 0, -1],
    [0, -1, -2]
])

# Functions
def convolve(mat, filtermask):
    #get number of rows and cols in matrix
    rows, cols = mat.shape[0], mat.shape[1]

    #calculate the range of the mask
    radius, length = int(filtermask.shape[0] / 2), filtermask.shape[0]

    #initialize result
    res = np.zeros((mat.shape[0] - radius * 2, mat.shape[1] - radius * 2))

    #perform convolutions
    for i in range(rows - length + 1):
        for j in range(cols - length + 1):
            res[i,j] = np.sum(filtermask * mat[i:i+length, j:j+length])

    return res

def gaussian_filter(image):

    #smooth image by convolving it with mask
    smoothed = convolve(image, GAUSSIAN)
    
    #smooth image by dividing by 140
    smoothed /= 140

    #return smoothed image
    return smoothed
                

def filters(image):

    #compute gradients
    mat1, mat2, mat3, mat4 = convolve(image, ZERO_DEGREES), convolve(image, FORTY_FIVE_DEGREES),convolve(image, NINETY_DEGREES), convolve(image, ONE_HUNDRED_THIRTY_FIVE_DEGREES)

    rows, cols = mat1.shape[0], mat1.shape[1]

    #initialize result
    res = np.zeros((rows, cols))

    #compute edge angles
    for i in range(rows):
        for j in range(cols):
            res[i,j]= max([np.abs(mat1[i,j]), np.abs(mat2[i,j]), np.abs(mat3[i,j]), np.abs(mat4[i,j])]) / 4

    gradientdirections = np.array([mat1, mat2, mat3, mat4])

    return res, gradientdirections

def quantize_angle(gradient_directions, i, j):
    """
    Quantize angle to index of the filter that produces the maximum response.
    Args:
        gradient_directions (numpy.ndarray): Array of gradient directions from 4 different filters.
        i (int): Row index.
        j (int): Column index.
    Returns:
        int: Quantized angle.
    """
    # Get index of mask that produced the maximum absolute response
    quantized = np.argmax([np.abs(gradient_directions[0][i, j]), np.abs(gradient_directions[1][i, j]), 
                           np.abs(gradient_directions[2][i, j]), np.abs(gradient_directions[3][i, j])])

    return quantized

def non_maxima_suppression(gradient_magnitude, gradient_directions):
    """
    Non-maximum suppression.
    Args:
        gradient_magnitude (numpy.ndarray): Gradient magnitude.
        gradient_directions (numpy.ndarray): Array of gradient directions from 4 different filters.
    Returns:
        numpy.ndarray: Image after non-maximum suppression.
    """
    suppressed = np.zeros(gradient_magnitude.shape)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            # If gradient magnitude is undefined, set to 0

            if np.isnan(gradient_magnitude[i, j]):
                suppressed[i, j] = 0
                continue

            neighbors = np.array([0,0])
            
            angle = quantize_angle(gradient_directions, i, j)
            if angle == 0:
                neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
            elif angle == 1:
                neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]
            elif angle == 2:
                neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
            elif angle == 3:
                neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
            
            # If neighbors are undefined, set to 0
            if np.isnan(neighbors[0]) or np.isnan(neighbors[1]):
                suppressed[i, j] = 0
                continue
            
            # If gradient magnitude is greater than neighbors, keep it
            if gradient_magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = gradient_magnitude[i, j]
            else:
                suppressed[i, j] = 0
    
    hold = suppressed.copy()
    hold[hold == 0] = np.nan

    percentiles = [25, 50, 75]
    supp_percentiles = np.array([])
    for percentile in percentiles:
        supp_percentiles = np.append(supp_percentiles, np.nanpercentile(hold, percentile))
        print(f"{percentile}%: {np.nanpercentile(hold, percentile)}")

    print(f"Mean: {np.mean(suppressed)}")

    return suppressed, supp_percentiles

def simple_threshold(image, thresholds):
    """
    Simple thresholding.
    Args:
        image (numpy.ndarray): Image to be processed.
        thresholds (list): Thresholds.
    Returns:
        numpy.ndarray: Image after simple thresholding.
    """
    
    twenty_edges = np.zeros(image.shape)
    fifty_edges = np.zeros(image.shape)
    seventy_edges = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= thresholds[0]:
                twenty_edges[i, j] = 1
            if image[i, j] >= thresholds[1]:
                fifty_edges[i, j] = 1
            if image[i, j] >= thresholds[2]:
                seventy_edges[i, j] = 1
    
    return [twenty_edges, fifty_edges, seventy_edges]
    

