import numpy as np
import tensorflow as tf

"""
Implement pyramid operations as explained in Section 5.2 of our paper, and introduced in:
    Barron and Malik, "Shape, illumination, and reflectance from shading", TPAMI, 2015.
"""


def kernels(mult=1.4):
    """
    Create binomial kernels.

    Inputs
    ------
    mult       Sum of each binomial kernel. See paper for explanation.

    Outputs
    -------
    h_kernel   Horizontal binomial kernel of shape [5, 1, 1, 1] and sum mult.
    w_kernel   Vertical binomial kernel of shape [1, 5, 1, 1] and sum mult.
    """
    h_kernel = mult * np.array([0.0625, 0.25, 0.3750, 0.25, 0.0625], dtype=np.float32)
    w_kernel = mult * np.array([0.0625, 0.25, 0.3750, 0.25, 0.0625], dtype=np.float32)

    return h_kernel[:, np.newaxis], w_kernel[np.newaxis, :]


def downsample(img):
    """
    Downsample an image by a factor of 2 using a binomial 5-tap filter (multiplied by a factor mult) as AA filter.

    Inputs
    ------
    img        Input image. Tensor of shape [N, H, W, C].

    Outputs
    -------
               Downsampled image. Tensor of shape [N, H/2, W/2, C].
    """

    # Blur by applying a binomial 5x5 kernel with sum mult^2 and subsample by a factor of 2.
    C = img.shape[3]

    # Get kernels, and turn them into a single kernel performing per-channel blurring
    h_kernel, w_kernel = kernels()
    kernel = np.zeros((5, 5, C, C), dtype=np.float32)
    for i in range(C):
        kernel[:, :, i, i] = h_kernel * w_kernel

    return tf.nn.conv2d(img, kernel, strides=(1, 2, 2, 1), padding='SAME')


def upsample(img, output_shape):
    """
    Implement the transpose of to downsample().

    Inputs
    ------
    img           Input image. Tensor of shape [N, H, W, C].
    output_shape  Shape of output image.

    Outputs
    -------
                  Downsampled image. Tensor of shape [N, 2H, 2W, C].
    """
    # Number of channels
    C = img.shape[3]

    # Get kernel for deconvolution (transpose of convolution)
    h_kernel, w_kernel = kernels()
    kernel = np.zeros((5, 5, C, C), dtype=np.float32)
    for i in range(C):
        kernel[:, :, i, i] = h_kernel * w_kernel

    return tf.nn.conv2d_transpose(img, kernel, output_shape, strides=(1, 2, 2, 1), padding='SAME')


def im2pyr(img, num_levels):
    """
    Convert an image into a pyramid.

    Inputs
    ------
    img           Input image. Tensor of shape [N, H, W, C].
    num_levels    Number of pyramid levels.

    Outputs
    -------
    pyr           Image pyramid, represented as a list of tensors of decreasing spatial size.
    """
    # Initialize pyramid using the original image.
    pyr = [img]

    # The next level of the pyramid is a downsampled version of the previous image.
    for n in range(num_levels - 1):
        pyr.append(downsample(pyr[-1]))

    return pyr


def pyr2im(pyr):
    """
    Go from pyramid to image by applying the transpose operator.

    Inputs
    ------
    pyr           Image pyramid, represented as a list of tensors of decreasing spatial size.

    Outputs
    -------
    img           Output image, computed by applying the transpose of the pyramid-generating operator to the pyramid.
    """

    # Initialize by taking the smallest scale.
    img = pyr[-1]

    # For each larger scale, upsample and add the image in the new scale.
    for next_img in pyr[:-1][::-1]:
        img = upsample(img, next_img.shape) + next_img

    return img
