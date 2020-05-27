import tensorflow as tf
import numpy as np
from make_net import make_net
from sft_utils import get_z_shape_list


def create_generator_discriminator(gen_arch, disc_arch, z_shape, dim_z_global, dim_z_local_list, dim_z_periodic_list,
                                   global_mlp_hidden_units, do_tie_phases,
                                   do_print=True, disc_input=None):
    """
    Create generator and discriminator networks.

    Inputs
    ------
    gen_arch                    Architecture string of generator network.
    disc_arch                   Architecture string of discriminator network.
    z_shape                     Tuple (of length 2) describing input noise shape.
    dim_z_global                Dimension of global noise
    dim_z_local_list            List of local noise dimensions
    dim_z_periodic_list         List of periodic noise dimensions
    global_mlp_hidden_units     Number of hidden units for MLP predicting wave vectors from global dimensions.
    do_tie_phases
    do_print                    If True, pring network architectures.
    disc_input                  If specified, use as input to discriminator instead of taking the output from the
                                generator.

    Outputs
    -------
    d_ph                        Dictionary holding all placeholders. Keys are names, values are placeholders.
    d_tensors                   Dictionary holding all tensors. Keys are names, values are tensors.

    """

    # Compute total local dimensions and list of spatial sizes
    dim_z_local = sum(dim_z_local_list)
    shape_list = get_z_shape_list(z_shape, len(dim_z_local_list), gen_arch)

    # Create phase placeholder
    with tf.variable_scope('phase'):
        is_training = tf.placeholder(tf.bool, shape=())

    # Create discriminator input placeholder (unless disc_input is specified)
    with tf.variable_scope('discriminator_input/'):
        if disc_input is None:
            disc_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='disc_input')

    # Create input placeholders to generator (i.e. global and local dimensions)
    with tf.variable_scope('generator_input/'):
        # Create global z input
        if dim_z_global > 0:
            global_gen_input = tf.placeholder(tf.float32, shape=[None, 1, 1, dim_z_global], name='input_z_global')
            # Tile global z spatially and add to list
            gen_input = tf.tile(global_gen_input, [1, shape_list[0][0], shape_list[0][1], 1])
        else:
            gen_input = None

        # Create local z input
        if dim_z_local > 0:
            local_gen_input_list = []

            for i, (dzl, shape) in enumerate(zip(dim_z_local_list, shape_list)):
                if dzl > 0:
                    print("Creating local tensor of shape {}x{}x{}".format(shape[0], shape[1], dzl))
                    local_gen_input_list.append(tf.placeholder(tf.float32, shape=[None]+list(shape)+[dzl],
                                                               name='input_z_local_scale{}'.format(i)))
                else:
                    local_gen_input_list.append(None)
        else:
            local_gen_input_list = [None] * len(dim_z_local_list)

    # Get total number of periodic dimensions
    dim_z_periodic = sum(dim_z_periodic_list)

    # If there is at least one periodic dimension, create a placeholder for phases (otherwise create list of Nones).
    if dim_z_periodic > 0:
        # Create a placeholder for each periodic map
        with tf.variable_scope('generator_input/'):
            random_phases = tf.placeholder(tf.float32, shape=[None, 1, 1, dim_z_periodic if not do_tie_phases else 2],
                                           name='random_phases')

        # Set variables for wave vectors if learned directly. Otherwise use MLP to predict them from global dimensions.
        if dim_z_global == 0:
            with tf.variable_scope('wave_vectors'):
                w = tf.get_variable('w', shape=[1, 1, 1, dim_z_periodic],
                                    initializer=tf.initializers.random_normal(mean=0.0, stddev=0.1))
                theta = tf.get_variable('theta', shape=[1, 1, 1, dim_z_periodic],
                                        initializer=tf.initializers.random_uniform(minval=0.0, maxval=2*np.pi))
        else:
            with tf.variable_scope('wave_vectors'):
                hidden = tf.layers.dense(global_gen_input, units=global_mlp_hidden_units, activation=tf.nn.relu)
                w = tf.reshape(tf.layers.dense(hidden, units=dim_z_periodic, activation=None, name='w'),
                               [-1, 1, 1, dim_z_periodic])
                theta = tf.reshape(tf.layers.dense(hidden, units=dim_z_periodic, activation=None, name='theta'),
                                   [-1, 1, 1, dim_z_periodic])

        # Compute cumulative sum of periodic dimension list for selecting indices of theta and w.
        p_end_ind = list(np.cumsum(dim_z_periodic_list))
        p_start_ind = [0] + p_end_ind[:-1]

        # Create list of periodic maps using learned w and theta.
        periodic_gen_input_list = []
        with tf.variable_scope('generator_input/'):
            with tf.variable_scope('z_periodic'):
                for i, (ps, pe, shape) in enumerate(zip(p_start_ind, p_end_ind, shape_list)):
                    # Only create map if the dimension needed to be added is > 0
                    if pe > ps:
                        print("Creating periodic tensor of shape {}x{}x{}".format(shape[0], shape[1], pe - ps))
                        kx = 0.5 * tf.nn.sigmoid(w[:, :, :, ps:pe]) * tf.cos(theta[:, :, :, ps:pe])
                        ky = 0.5 * tf.nn.sigmoid(w[:, :, :, ps:pe]) * tf.sin(theta[:, :, :, ps:pe])

                        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                        sinusoid_args = kx * x[np.newaxis, :, :, np.newaxis] + ky * y[np.newaxis, :, :, np.newaxis]

                        if do_tie_phases:
                            # TODO: For correctly matching the phases over different scales we must multiply
                            # random_phases by by 2 ** i. The results in the paper were obtained without this factor
                            # by mistake. Both options are implemented below, but the original one used for the paper
                            # is active for reproducibility purposes, although both versions seem to work well.
                            p_concat_tensor = tf.sin(2 * np.pi * sinusoid_args + 2 * np.pi *
                                                     (kx*random_phases[:, :, :, :1] + ky*random_phases[:, :, :, 1:]),
                                                     name='periodic_scale{}'.format(i))
                            # p_concat_tensor = tf.sin(2 * np.pi * sinusoid_args + 2 * np.pi *
                            #                         (kx*random_phases[:, :, :, :1] +
                            #                          ky*random_phases[:, :, :, 1:]) * 2 ** i,
                            #                          name='periodic_scale{}'.format(i))

                        else:
                            p_concat_tensor = tf.sin(2 * np.pi * sinusoid_args + random_phases[:, :, :, ps:pe],
                                                     name='periodic_scale{}'.format(i))

                        periodic_gen_input_list.append(p_concat_tensor)

                    else:
                        periodic_gen_input_list.append(None)  # no periodic tensor

    else:
        periodic_gen_input_list = [None] * len(dim_z_periodic_list)

    assert len(periodic_gen_input_list) == len(local_gen_input_list), "Must be same length (even if some are None)"

    # Create list of inputs to each deconvolution layer
    concat_tensor_list = []
    for i in range(len(periodic_gen_input_list)):
        # Get the ith local and periodic tensor
        l_tensor = local_gen_input_list[i]
        p_tensor = periodic_gen_input_list[i]

        # Get list of tensors input to the ith deconvolution, ignoring any Nones.
        curr_concat_list = []
        for t in [p_tensor, l_tensor]:
            if t is not None:
                curr_concat_list.append(t)

        # Concatenate the tensors and append to concat_tensor_list.
        if len(curr_concat_list) > 0:
            concat_tensor_list.append(tf.concat(curr_concat_list, axis=3, name='concat_list_{}'.format(i)))
        else:
            concat_tensor_list.append(None)

    # Create generator network
    with tf.variable_scope('generator'):
        gen_output = make_net(gen_input, gen_arch, is_generator=True, concat_list=concat_tensor_list.copy(),
                              norm_layer='BN', is_training=is_training, do_print=do_print)

        # Create scaled version of gen_output in [0, 1]
        gen_sample = tf.identity(tf.nn.tanh(gen_output) * 0.5 + 0.5, name='gen_sample')

    # Create discriminator network.
    with tf.variable_scope('discriminator'):
        # Get discriminator output when applied to disc_input (either placeholder or input to function).
        disc_real = tf.identity(make_net(disc_input, disc_arch, is_generator=False, norm_layer='None',
                                         is_training=is_training, do_print=do_print), name='disc_real')
        # Get discriminator output when applied to the output of the generator
        disc_fake = tf.identity(make_net(gen_sample, disc_arch, is_generator=False, norm_layer='None',
                                         reuse=True, is_training=is_training, do_print=do_print), name='disc_fake')

    # Return placeholders and tensors
    d_ph = {'is_training':          is_training,
            'disc_input':           disc_input,
            'global_gen_input':     global_gen_input,
            'local_gen_input_list': local_gen_input_list}

    if dim_z_periodic > 0:
        d_ph['random_phases'] = random_phases

    d_tensors = {'disc_fake':  disc_fake,
                 'disc_real':  disc_real,
                 'gen_sample': gen_sample}

    return d_ph, d_tensors


def create_loss(disc_real, disc_fake, wd_mult):
    """
    Create losses for discriminator and generator.

    Inputs
    ------
    disc_real    Logits output by discriminator applied to unwarper output
    disc_fake    Logits output by discriminator applied to generator output
    wd_mult      Weight decay amount.

    Outputs
    -------
    gen_loss     Generator loss
    disc_loss    Discriminator loss
    """

    # Get all generator and discriminator kernels
    generator_kernels = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                      scope='generator') if 'kernel' in x.name]
    discriminator_kernels = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='discriminator') if 'kernel' in x.name]

    # Create weight decay elements
    with tf.variable_scope('weight_decay'):
        g_wd = wd_mult * tf.add_n([tf.nn.l2_loss(kernel) for kernel in generator_kernels])
        d_wd = wd_mult * tf.add_n([tf.nn.l2_loss(kernel) for kernel in discriminator_kernels])

    # Get losses
    with tf.variable_scope('loss'):
        # Generator loss is -log(sigmoid(disc_fake))
        gen_loss = g_wd + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_fake),
                                                                                 logits=disc_fake))

        # Discriminator loss is -log(sigmoid(disc_real) - log(1 - sigmoid(disc_real))
        disc_loss = d_wd + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_real),
                                                                                  logits=disc_real)) + \
                           tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_fake),
                                                                                  logits=disc_fake))

    return gen_loss, disc_loss


def create_dg_optimizers(gen_loss, disc_loss, learning_rate, beta1):
    """
    Create optimizers for discriminator and generator.

    Inputs
    ------
    gen_loss        Generator loss
    disc_loss       Discriminator loss
    learning_rate   Learning rate for Adam opetimizer
    beta1           beta1 parameter for Adam optimizer

    Outputs
    -------
    train_gen       Generator optimization operation
    train_disc      Discriminator optimization operation
    """
    with tf.variable_scope('optimizers'):
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

        # Make sure batchnorm layer parameters are updated.
        bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        gen_bn_update_ops = [x for x in bn_update_ops if 'generator' in x.name]
        disc_bn_update_ops = [x for x in bn_update_ops if 'discriminator' in x.name]

        # Create optimization operations
        with tf.control_dependencies(gen_bn_update_ops):
            gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator') + \
                       tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='wave_vectors')
            train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)

        with tf.control_dependencies(disc_bn_update_ops):
            disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator') + \
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='wave_vectors')
            train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

        return train_gen, train_disc


def create_no_nan_assertions():
    """
    Create operations for making sure there are no NaN values in tensors.

    Outputs
    -------
              List of operations asserting there exist no NaN values.
    """

    with tf.variable_scope('no_nan_assertions'):
        no_nan_assertions = [tf.verify_tensor_all_finite(t, 'Tensor {} contains bad values!'.format(t.name))
                             for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

    return no_nan_assertions
