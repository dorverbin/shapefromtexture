import tensorflow as tf

def apply_warps(img, loc, warps, shape):
    """
    Apply affine warps to images.

    Inputs
    ------
    img          Input images, tensor of shape [N, H, W, C]
    loc          A tensor holding the center pixel location for each image. Must have shape [N, 2].
    warps        A translation-free affine matrix for each pixel, determining the transformation
                 of each patch. Shape [H, W, 4]. The order of elements is [m11, m12, m21, m22].
    shape        A tuple (H_out, W_out) specifying the output shape (must be the same for entire batch).


    Outputs
    -------
    output       A tensor of shape [N, H_out, W_out, C].

    Description
    -----------
    For each i = 0, ..., N-1, loc[i, :] = (x0, y0) defines a patch location. For all i, we
    use the matrix [warps[y0, x0, 0], warps[y0, x0, 1] ; warps[y0, x0, 2], warps[y0, x0, 3]] to warp a
    regular grid centered at (x0, y0). The input image, img[i, :, :, :], sampled by this grid is
    output[i, :, :, :].
    """

    # Get sizes of input image
    N = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]

    # Define half width and height for output patches
    hw = (tf.cast(shape[1], dtype=tf.float32) - 1) / 2.0
    hh = (tf.cast(shape[0], dtype=tf.float32) - 1) / 2.0

    # Create grid for output patches
    x_, y_ = tf.meshgrid(tf.range(-hw, hw+1), tf.range(-hh, hh+1))

    # Get warp matrix for each location in loc
    warps_r = tf.reshape(warps, [1, H, W, 4])  # Reshaped to [1, H, W, 4]
    warps_rt = tf.tile(warps_r, [N, 1, 1, 1])  # Tiled to [N, H, W, 4]
    warp = tf.contrib.resampler.resampler(warps_rt, loc)  # Shape [N, 4]

    # Get elements of warp matrix (each has shape [N, 1, 1, 1])
    w11 = tf.reshape(warp[:, 0], [-1, 1, 1, 1])
    w12 = tf.reshape(warp[:, 1], [-1, 1, 1, 1])
    w21 = tf.reshape(warp[:, 2], [-1, 1, 1, 1])
    w22 = tf.reshape(warp[:, 3], [-1, 1, 1, 1])

    # Reshape x, y to [1, H_out, W_out, 1]
    sh = tf.stack([1, shape[0], shape[1], 1], axis=0)
    x = tf.reshape(x_, sh)
    y = tf.reshape(y_, sh)

    # Compute warped grid (shape [N, shape[0], shape[1], 1] each)
    x_sample = tf.clip_by_value(tf.reshape(loc[:, 0], [-1, 1, 1, 1]) + w11 * x + w12 * y,
                                0.0, tf.cast(W, dtype=tf.float32)-1.0)
    y_sample = tf.clip_by_value(tf.reshape(loc[:, 1], [-1, 1, 1, 1]) + w21 * x + w22 * y,
                                0.0, tf.cast(H, dtype=tf.float32)-1.0)

    # Sample using the warped grid
    sample_points = tf.concat([x_sample, y_sample], axis=-1)
    output = tf.contrib.resampler.resampler(img, sample_points)

    return output
