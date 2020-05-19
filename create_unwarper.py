import tensorflow as tf
from pyramid import pyr2im
import numpy as np


def create_unwarper(H, W):
    """
    Create unwarper.

    Inputs
    ------
    H                      Height of image
    W                      Width of image

    Outputs
    -------
    warps                  Warp matrices
    n                      Normal map
    t                      Tangent vector map
    n_smoothness_loss      Smoothness loss over normals
    t_smoothness_loss      Smoothness loss over tangent vectors
    integrability_loss     Integrability loss

    Description
    -----------
    Represent the x and y components of the normals as an image pyramid (see Section 5.2 in paper). Using the normal
    and tangent vector at each pixel, we define the warp matrix at each pixel and store it under `warps`. The losses
    are defined in Section 5.1 in the paper.
    """

    with tf.variable_scope('warp'):

        # Create pyramids for nx and ny
        n_levels = int(np.ceil(min([np.log2(H), np.log2(W)]))) + 1  # Number of levels: 1 pixel in highest level
        n_pyramids = [[], []]
        for i in range(n_levels):
            div = 2 ** i
            for j in range(2):
                init = np.float32(np.random.rand(1, int(np.ceil(H / div)), int(np.ceil(W / div)), 1) - 0.5) * 0.0001
                if j == 0:  # Only print once
                    print("Initializing normal pyramid level {} of shape {}".format(i, init.shape))
                n_pyramids[j].append(tf.get_variable(name='n{}{}'.format(j, i), initializer=init))

        # Get nx and ny from pyramid representation
        nxu = pyr2im(n_pyramids[0])[0, :, :, 0]
        nyu = pyr2im(n_pyramids[1])[0, :, :, 0]
        nzu = np.ones((H, W), dtype=np.float32)

        # Normalize ||n(x, y)|| = 1
        n_norm = tf.sqrt(nxu ** 2 + nyu ** 2 + nzu ** 2)
        nx = nxu / n_norm
        ny = nyu / n_norm
        nz = nzu / n_norm

        # Stack to shape [H, W, 3]
        n = tf.stack([nx, ny, nz], axis=-1)

        # Create coefficients for tangent vectors
        coeff1_init = np.float32(np.random.rand(1, H, W, 1) - 0.5) * 0.2
        coeff2_init = 1.0 + np.float32(np.random.rand(1, H, W, 1) - 0.5) * 0.2
        coeff1 = tf.get_variable(name='c1', dtype=tf.float32, initializer=coeff1_init)[0, :, :, 0]
        coeff2 = tf.get_variable(name='c2', dtype=tf.float32, initializer=coeff2_init)[0, :, :, 0]

        # Define tangent vector to be orthogonal to normal vector
        txu = nz * coeff2
        tyu = nz * coeff1
        tzu = -ny * coeff1 - nx * coeff2

        # Normalize ||t(x, y)|| = 1
        t_norm = tf.sqrt(txu ** 2 + tyu ** 2 + tzu ** 2 + 1e-4)
        tx = txu / t_norm
        ty = tyu / t_norm
        tz = tzu / t_norm

        # Stack to shape [H, W, 3]
        t = tf.stack([tx, ty, tz], axis=-1)

        # Define warp matrices
        w11 = tx
        w12 = ny * tz - nz * ty
        w21 = ty
        w22 = nz * tx - nx * tz
        warps = tf.stack([w11, w12, w21, w22], axis=-1)

        # Compute smoothness losses for n, t
        n_smoothness_loss = tf.reduce_mean(tf.square(n[1:, :, :] - n[:-1, :, :])) + \
                            tf.reduce_mean(tf.square(n[:, 1:, :] - n[:, :-1, :]))
        t_smoothness_loss = tf.reduce_mean(tf.square(t[1:, :, :] - t[:-1, :, :])) + \
                            tf.reduce_mean(tf.square(t[:, 1:, :] - t[:, :-1, :]))

        # Compute integrability loss
        p = nx / nz
        q = ny / nz

        # In the Horn paper, i corresponds to x and j to y so we're flipped
        # relative to it. Additionally they assume x goes right and y goes up.
        #
        # We can instead integrate clockwise, and then the signs are:
        #
        #      p: +               p: +
        #      q: -  (i, j)       q: +  (i, j+1)
        #
        #
        #
        #      p: -               p: -
        #      q: -  (i+1, j)     q: +  (i+1, j+1)

        pi0j0 = p[:-1, :-1]   # p_{i,j}
        pi1j0 = p[1:,  :-1]   # p_{i+1,j}
        pi0j1 = p[:-1, 1: ]   # p_{i,j+1}
        pi1j1 = p[1:,  1: ]   # p_{i+1,j+1}

        qi0j0 = q[:-1, :-1]   # q_{i,j}
        qi1j0 = q[1:,  :-1]   # q_{i+1,j}
        qi0j1 = q[:-1, 1: ]   # q_{i,j+1}
        qi1j1 = q[1:,  1: ]   # q_{i+1,j+1}

        integrability_loss = tf.reduce_mean(tf.square(  pi0j0 + pi0j1 - pi1j0 - pi1j1
                                                      - qi0j0 + qi0j1 - qi1j0 + qi1j1))

    return warps, n, t, n_smoothness_loss, t_smoothness_loss, integrability_loss


def create_w_optimizers(shape_loss, n_learning_rate, t_learning_rate):
    """
    Create unwarper optimizers.

    Inputs
    ------
    shape_loss             Overall unwarper loss (see Equation 2 in paper).
    n_learning_rate        Learning rate for normal vectors.
    t_learning_rate        Learning rate for tangent vectors.

    Outputs
    -------
    train_shape            Operations for optimizing normal and tangent vector maps.
    """

    # Create optimizers for n, t, generator and discriminator
    n_optimizer_shape_vars = tf.train.AdamOptimizer(learning_rate=n_learning_rate)
    t_optimizer_shape_vars = tf.train.AdamOptimizer(learning_rate=t_learning_rate)

    # Collect shape parameters and define
    n_params = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='warp') if 'n' in x.name]
    t_params = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='warp') if 'c' in x.name]

    # Shape training operations
    train_shape = list()
    train_shape.append(n_optimizer_shape_vars.minimize(shape_loss, var_list=n_params))
    train_shape.append(t_optimizer_shape_vars.minimize(shape_loss, var_list=t_params))

    return train_shape
