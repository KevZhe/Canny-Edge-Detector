import argparse
import sys
from canny import *
import os

parser = argparse.ArgumentParser(description='Canny edge detection.')
parser.add_argument('--image', type=str, default='example.jpg', help='Path to image.')
args = parser.parse_args()

def visualize(image, image_edges, nms_image, directory):
    """
    Visualize image and its edges.
    Args:
        image (numpy.ndarray): Image to be processed.
        image_edges (List): List of image edges.
    """
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(image_edges[0], cmap='gray')
    plt.title('Edges (25%)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(image_edges[1], cmap='gray')
    plt.title('Edges (50%)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(image_edges[2], cmap='gray')
    plt.title('Edges (75%)')
    plt.axis('off')

    plt.savefig(directory + "/comparison.png")

    plt.clf()

    suppressed = np.ravel(nms_image)

    plt.hist(suppressed, bins = 30)
    plt.title("Distribution of Magnitude after NMS")
    plt.xlabel("Magnitude")
    plt.ylabel("Count")

    plt.savefig(directory + "/histogram.png")

    plt.clf()

def main():

    filename = args.image

    image = np.array(Image.open("testimages/" + filename).convert('L'))

    newpath = 'output/' + filename[:-4] 

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Gaussian filter
    image_smooth = gaussian_filter(image)
    guassian_smooth = Image.fromarray((image_smooth).astype(np.uint8))
    guassian_smooth.save(newpath + "/guassian_smooth.bmp")

    # Gradient magnitude and direction
    gradient_magnitude, gradient_directions = filters(image_smooth)
    magnitude = Image.fromarray((gradient_magnitude*255).astype(np.uint8))
    magnitude.save(newpath + "/magnitude.bmp")

    # Non-maxima suppression
    image_suppressed, thresholds = non_maxima_suppression(gradient_magnitude, gradient_directions)
    suppressed = Image.fromarray((image_suppressed*255).astype(np.uint8))
    suppressed.save(newpath + "/nmssuppressed.bmp")


    # Simple threshold
    image_edges = simple_threshold(image_suppressed, thresholds)
    i1, i2, i3 = image_edges[0], image_edges[1], image_edges[2]

    image1 = Image.fromarray((i1*255).astype(np.uint8))
    image1.save(newpath + "/threshold25.bmp")
    image2 = Image.fromarray((i2*255).astype(np.uint8))
    image2.save(newpath + "/threshold50.bmp")
    image3 = Image.fromarray((i3*255).astype(np.uint8))
    image3.save(newpath + "/threshold75.bmp")

    visualize(image, image_edges, image_suppressed, newpath)

if __name__ == "__main__":
    main()