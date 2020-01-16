import cv2
import numpy as np

from constants import *

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
    A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label is not None and label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if label is None:
        return colormap

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def generate_labels(model_path):
    """ Generates labels for known models.
    """
    label_names = None
    full_label_map = None
    if "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz" in model_path:
        label_names = np.asarray([
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
        ])
        full_label_map = np.arange(len(label_names)).reshape(len(label_names), 1)

    if "deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz" in model_path:
        label_names = np.asarray([
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
            'train', 'motorcycle', 'bicycle'
        ])
        full_label_map = np.arange(len(label_names)).reshape(len(label_names), 1)

    full_color_map = label_to_color_image(full_label_map)
    return label_names, full_color_map

def generate_legend(model_path):
    """Generates the legend for a model.
    """
    labels, color_map = generate_labels(model_path)

    legend = np.ones((CAP_HEIGHT, LEG_WIDTH, 3), np.uint8) * 255
    for i in range(len(labels)):
        label = labels[i]
        r, g, b = tuple(color_map[i,0,:])
        r, g, b = int(r), int(g), int(b)

        x = PALETTE_SPACING
        y = PALETTE_SPACING + (PALETTE_SIZE + PALETTE_SPACING) * i
        w = PALETTE_SIZE
        h = PALETTE_SIZE

        cv2.rectangle(legend, (x,y), (x+w,y+h), (r, g, b), -1)

        x += w + PALETTE_SPACING
        y += h - PALETTE_SPACING

        cv2.putText(legend, label, (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))

    return legend

# RGB/BGR conversion.
def convert(image):
    """RGB/BGR conversion.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display an image.
def display_image(window, image):
    """Displays an image.
    """
    # Convert the image to BGR.
    cv2.imshow(window, convert(image))
