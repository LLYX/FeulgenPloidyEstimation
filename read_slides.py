import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import openslide.deepzoom

from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from skimage.color import rgb2gray, rgb2hed
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu


def create_tile_generator(filename, tile_size, overlap, limit_bounds):
    return openslide.deepzoom.DeepZoomGenerator(openslide.OpenSlide(filename), tile_size, overlap, limit_bounds)

def colour_deconv(rgb_image):
    cmap_counter = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
    cmap_feulgen_1 = LinearSegmentedColormap.from_list('mycmap', ['white',
                                                'darkred'])
    cmap_feulgen_2 = LinearSegmentedColormap.from_list('mycmap', ['white',
                                                'darkviolet'])

    ihc_hed = rgb2hed(rgb_image)

    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(rgb_image)
    ax[0].set_title("Original image")

    ax[1].imshow(ihc_hed[:, :, 0], cmap=cmap_counter)
    ax[1].set_title("Counterstain")

    ax[2].imshow(ihc_hed[:, :, 1], cmap=cmap_feulgen_2)
    ax[2].set_title("Feulgen 2")

    ax[3].imshow(ihc_hed[:, :, 2], cmap=cmap_feulgen_1)
    ax[3].set_title("Feulgen 1")

    for a in ax.ravel():
        a.axis('off')

    fig.tight_layout()

    # Rescale stain signals and give them a fluorescence look
    h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
    d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
    zdh = np.dstack((np.zeros_like(h), d, h))

    fig = plt.figure()
    axis = plt.subplot(1, 1, 1, sharex=ax[0], sharey=ax[0])
    axis.imshow(zdh)
    axis.set_title("Stain separated image (rescaled)")
    axis.axis('off')
    plt.show()

    return (zdh * 255).astype(np.uint8)

def threshold(image_gray):
    image_gray = cv2.GaussianBlur(image_gray, (7, 7), 1)
    ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU)
    
    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    max_cnt_area = cv2.contourArea(cnts[0])
    
    if max_cnt_area > 50000:
        ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    return thresh

def apply_morphology(thresh):
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

    return mask

def contrast_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(image)

    return cl1

def gradient_image(image):
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    laplacian = cv2.Laplacian(image,cv2.CV_64F)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # plt.imshow(gradient)
    # plt.show()

    # plt.imshow(laplacian)
    # plt.show()

    # plt.imshow(sobelx)
    # plt.show()

    # plt.imshow(sobely)
    # plt.show()

    # gradient = laplacian - sobelx - sobely
    
    return gradient

def remove_too_small(binary_mask):
    labels, nlabels = ndimage.label(binary_mask)

    for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
        cell = binary_mask[label_coords]
    
        # Check if the label size is too small
        if np.product(cell.shape) < 20: 
            print('Label {} is too small! Setting to 0.'.format(label_ind))
            binary_mask = np.where(labels==label_ind+1, 0, binary_mask)

    # Regenerate the labels
    labels, nlabels = ndimage.label(binary_mask)

    return binary_mask, labels, nlabels

def instance_segmentation(mask):
    labels, nlabels = ndimage.label(mask)

    label_arrays = []
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        label_arrays.append(label_mask)

    return remove_too_small(mask)

def watershed(image, mask, gradient):
    # noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations = 3)
    plt.imshow(sure_bg)
    plt.show()
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 0)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)

    sure_fg, labels, nlabels = remove_too_small(sure_fg)

    plt.imshow(sure_fg)
    plt.show()
            
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    plt.imshow(unknown)
    plt.show()

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1 # + gradient
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    plt.imshow(markers)
    plt.show()

    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    # image[gradient > 0] = [0, 255, 0]

    plt.imshow(image)
    plt.show()

    plt.imshow(markers)
    plt.show()

if __name__ == "__main__":
    image_generator = openslide.deepzoom.DeepZoomGenerator(openslide.OpenSlide("/home/lxu/Documents/Feulgen Project/Aug8_Backup/images/tma_40x.svs"), 480, 0, True)

    image = np.array(image_generator.get_tile(17, (39, 51)))
    im_gray = contrast_enhancement(
        cv2.cvtColor(
            colour_deconv(image), cv2.COLOR_BGR2GRAY))

    plt.imshow(im_gray)

    plt.show()

    grad_image = gradient_image(im_gray)

    plt.imshow(im_gray - grad_image)
    plt.show()

    im_gray = (im_gray - grad_image).astype('uint8')

    thresh = threshold(im_gray)
    mask = apply_morphology(thresh)

    plt.imshow(mask)
    plt.show()

    mask, labels, nlabels = instance_segmentation(mask)

    print(nlabels)

    plt.imshow(mask)
    plt.show()

    watershed(image, mask, gradient_image(mask).astype(np.uint8))