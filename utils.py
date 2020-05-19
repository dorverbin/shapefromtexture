import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def create_dir_if_needed(directory):
    """
    Create directory if it does not exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created directory {}".format(directory))


def plot_image(image, title=None, vmin=None, vmax=None):
    """
    Helper function for image plotting. vmin and vmax are used for clipping (unless they're None).
    """
    plt.ion()

    # Make sure image is either 2d (grayscale) or 3d with three channels (RGB).
    if len(image.shape) == 2:
        params = {'cmap': 'gray'}
    elif len(image.shape) == 3 and image.shape[2] == 1:
        params = {'cmap': 'gray'}
        image = image[:, :, 0]
    elif len(image.shape) == 3 and image.shape[2] == 3:
        params = {}
    else:
        raise ValueError("image must be 2D or 3D with three channels (but has shape {})".format(image.shape))

    # If vmin and vmax are set, apply clipping
    if vmin is not None and vmax is not None:
        params['vmin'] = vmin
        params['vmax'] = vmax
        image = np.clip(image, vmin, vmax)
    try:
        plt.imshow(image, **params)
        if title is not None:
            plt.title(title)
        plt.show()
        plt.pause(0.001)
    except:
        return


def save_image(filename, img):
    """
    Save image `img` as `filename`.
    """
    # If grayscale, save as is. If RGB, flip channels (because opencv uses BGR rather than RGB).
    if len(img.shape) == 2 or len(img.shape) == 3 and img.shape[2] == 1:
        cv2.imwrite(filename, img)
    else:
        cv2.imwrite(filename, img[:, :, ::-1])