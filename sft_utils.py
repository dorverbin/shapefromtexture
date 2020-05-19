import numpy as np

def get_z_shape_list(z_shape, num_layers, gen_arch):
    """
    Computes the spatial size of the input to all deconv layers of the generator.

    Inputs
    ------
    z_shape              The spatial size of the input to the first deconvolution
    num_layers           Number of layers in generator
    gen_arch             Generator architecture

    Outputs
    -------
    z_shape_list         A list whose ith element is the spatial input size for the

    """

    # First, make sure the kernel size of all layers is the same.
    ind1 = -1
    kernel_sizes = []
    while True:
        ind1 = gen_arch.find('D', ind1+1)  # Index of "D"
        ind2 = gen_arch.find('-', ind1+1)  # Index of next "_"
        if ind1 >= 0 and ind2 < 0:
            k = int(gen_arch[ind1+1:])
        elif ind1 >= 0:
            k = int(gen_arch[ind1+1:ind2])
        if ind1 < 0 or ind2 < 0:
            break
        kernel_sizes.append(k)
    if len(set(kernel_sizes)) != 1:
        raise ValueError('All deconvolution kernel sizes must be the same ({})'.format(kernel_sizes))

    # The (constant) kernel size
    kernel_size = kernel_sizes[0]

    # For each deconv, multiply by 2 and add kernel_size-2, see Section 4.1 of paper.
    return [[d * 2 ** i + kernel_size - 2 for d in z_shape] for i in range(num_layers)]


def get_feed_dict(local_gen_input_list, global_gen_input, random_phases, dim_z_local_list, dim_z_global,
                  dim_z_periodic, do_tie_phases, batch_size, z_shape_list):
    """
    Returns a dictionary to be used as a feed_dict. Fills placeholders needed for generator.

    Inputs
    ------
    local_gen_input_list   List of placeholders used as local inputs to generator
    global_gen_input       Placeholder used as global input to generator
    random_phases          Placeholder used as phases for periodic input to generator
    dim_z_local_list       List of local input dimension input to each deconvolution
    dim_z_global           Global input dimension (only input to first deconvolution)
    dim_z_periodic         Number of periodic dimensions used in total (not a list)
    do_tie_phases          If True, only generate two numbers used as shifts. if False,
                           generator one phase for each periodic dimension.
    batch_size             Number of images in a batch
    z_shape_list           List of spatial sizes input to each deconvolution layer

    Outputs
    -------
    feed_dict              dictionary with all placeholders as keys and the fed values as values
    """

    # Generate local and global noise, as well as shifts (either phases or two shifts)
    z_local_list, z_global, phases = sample_zl_zg_phases(dim_z_local_list, dim_z_global, dim_z_periodic if not do_tie_phases else 2,
                                                         batch_size, z_shape_list)

    feed_dict = {}

    # Add local maps to dictionary
    for local_gen_input, z_local, dz in zip(local_gen_input_list, z_local_list, dim_z_local_list):
        if dz > 0:
            feed_dict[local_gen_input] = z_local

    # Add global maps to dictionary
    if dim_z_global > 0:
        feed_dict[global_gen_input] = z_global

    # Add phases (or shifts) to dictionary. If do_tie_phases is True, multiply by an arbitrary factor
    # to make sure shift is large enough (rather than just leaving it at [0, 2pi])
    if dim_z_periodic > 0:
        feed_dict[random_phases] = phases if not do_tie_phases else phases * z_shape_list[0][0] * 10.0 / (2*np.pi)

    return feed_dict


def sample_zl_zg_phases(dim_z_local_list, dim_z_global, dim_z_periodic, batch_size, z_shape_list):
    """
    Sample local, global noise, and phases.

    Inputs
    ------
    dim_z_local_list       List of local input dimension input to each deconvolution
    dim_z_global           Global input dimension (only input to first deconvolution)
    dim_z_periodic         Number of periodic dimensions used in total (not a list)
    batch_size             Number of images in a batch
    z_shape_list           List of spatial sizes input to each deconvolution layer

    Outputs
    -------
    z_local_list           List of uniformly sampled maps (or None values when the local dimension is 0)
    z_global               Tensor of shape [batch_size, 1, 1, dim_z_global] sampled uniformly in [-1, 1]
    phases                 Tensor of shape [batch_size, 1, 1, dim_z_periodic] sampled uniformly in [0, 2pi]
    """

    # Local noise maps are uniform in [0, 1] and are spatially i.i.d.
    z_local_list = []
    for i, (z_shape, dzl) in enumerate(zip(z_shape_list, dim_z_local_list)):
        # If number of local dimensions is positive, sample and append to list. Otherwise append None.
        if dzl > 0:
            z_local = np.random.uniform(-1.0, 1.0, size=[batch_size, z_shape[0], z_shape[1], dzl])
            z_local_list.append(z_local)
        else:
            z_local_list.append(None)
    else:
        z_local = None  # If there are no local dimensions at all, return None

    # Global noise is uniform in [0, 1], and constant spatially.
    if dim_z_global > 0:
        z_global = np.random.uniform(-1., 1., size=[batch_size, 1, 1, dim_z_global])
    else:
        z_global = None

    # Phases are uniform in [0, 2pi]
    if dim_z_periodic > 0:
        phases = np.random.uniform(0, 2*np.pi, size=[batch_size, 1, 1, dim_z_periodic])
    else:
        phases = None

    return z_local_list, z_global, phases
