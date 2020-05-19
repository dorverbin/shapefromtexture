import tensorflow as tf

# Currently the generator can be represented by the string "F-1024_F-6272_R-7-7-128_D5-64_D5-1". Each block is followed by ReLU, BN (except for output and reshape layers); D has stride of 2.
# and the discriminator can be represented by the string   "C5-64_C5-128_R-6272_F-1024_F-1" Each block is followed by Leaky ReLU, BN (except for input and reshape layers); C has stride of 2.


def op2layer(op):
    """
    Create tensorflow operation from string.

    Inputs
    ------
    op      String describing operation.

    Outputs
    -------
            Operation correspond to op.
    """
    if op == 'reshape':
        return tf.reshape
    elif op == 'dense':
        return tf.layers.dense
    elif op == 'conv':
        return tf.layers.conv2d
    elif op == 'deconv':
        return deconv_and_crop
    elif op == 'batchnorm':
        return tf.layers.batch_normalization
    elif op == 'instancenorm':
        return tf.contrib.layers.instance_norm
    elif op == 'relu':
        return tf.nn.relu
    elif op == 'lrelu':
        return tf.nn.leaky_relu
    elif op == 'sigmoid':
        return tf.nn.sigmoid
    else:
        raise ValueError('Layer op {} not implemented yet'.format(op))


def arch_string2list(arch_str, is_generator, norm_layer='BN'):
    """
    Create architecture list from string.

    Inputs
    ------
    arch_str       String describing architecture
    is_generator   True for generator, False for discriminator.
    norm_layer     Normalization layer to use. BN for batchnorm, IN for instance norm, None for nothing.

    Outputs
    arch_list      List of layers. Each entry is a dictionary with all parameters needed to create layer.

    Description
    -----------
    arch_str is a string describing the architecture. The different layers are separated by an underscore. The different
    operations supported are:
    C: convolution, must have format C<k>-<nout> where k is the kernel size and nout is the number of output channels.
    D: deconvolution, must have format D<k>-<nout> where k is the kernel size and nout is the number of output channels.
    R: reshape, must have format R-<nout> for vectorization or R-<h>-<w>-<c> for unvectorizing.
    F: fully connected, must have format F-<nout> where nout is the number of output features.

    is_generator is used for determining the activation of the final layer as well as whether to use ReLU or Leaky ReLU,
    as described in [3] Bergmann, Jetchev, and Vollgraf, "Learning texture manifolds with the periodic spatial GAN".
    """
    arch_list = []
    last_layer = 'bottom'  # Input to network
    layers = arch_str.split('_')
    for i, layer in enumerate(layers):
        fields = layer.split('-')
        ty = fields[0][0]  # Layer type
        if ty == 'F':
            assert len(fields) == 2, "A fully connected layer must have format 'F-<nout>'"
            layer_name = 'fc{}'.format(i + 1)
            params = {'units': int(fields[1])}
            op = 'dense'
            layer = {'name': layer_name,
                     'op':   'dense'}
        elif ty == 'R':
            assert len(fields) in [2, 4], \
                "A reshape layer must have 1 number for vectorizing ('R-<nout>') or 3 for creating image ('R-<h>-<w>-<c>')"
            layer_name = 'reshape{}'.format(i + 1)
            params = {'shape': tuple([-1] + [int(x) for x in fields[1:]])}
            op = 'reshape'
            layer = {'name':  layer_name,
                     'op':    'reshape'}
        elif ty in ['C', 'D']:
            op = 'deconv' if ty == 'D' else 'conv'
            assert len(fields) == 2,\
                "A {} layer must have format '{}<k>-<nout>', where k is the kernel height and width".format(op, ty)
            layer_name = '{}{}'.format(op, i + 1)
            params = {'kernel_size': int(fields[0][1:]),
                      'strides':      (2, 2),
                      'filters':      int(fields[1]),  # Number of filters
                      'padding':      'same' if op == 'deconv' else 'valid'}

            layer = {'name':         layer_name,
                     'op':           op}
        else:
            raise ValueError('Layer type {} not implemented yet'.format(ty))

        # Add name, op, params and bottom to layer dictionary
        layer['name']   = layer_name
        layer['op']     = op
        layer['params'] = params
        layer['bottom'] = last_layer

        # Add layer to list
        arch_list.append(layer)

        # Now add activation layer, if needed.
        last_layer = layer_name

        # If generator and any layer except for reshape layers and last layer (last before any reshape layers),
        # add ReLU and then norm. If discriminator and any layer except for reshape, first and last, add leaky ReLU
        # and then norm. If first layer only add leaky ReLU. If last layer add nothing.
        if is_generator and ty != 'R' and not all([l[0] == 'R' for l in layers[i+1:]]):
            layer = create_relu_layer(last_layer, layer_name + '_relu')
            arch_list.append(layer)
            last_layer = layer_name + '_relu'

            if norm_layer == 'BN':
                layer = create_batchnorm_layer(last_layer, layer_name + '_bn')
                arch_list.append(layer)
                last_layer = layer_name + '_bn'
            elif norm_layer == 'IN':
                layer = create_instancenorm_layer(last_layer, layer_name + '_in')
                arch_list.append(layer)
                last_layer = layer_name + '_in'

        elif not is_generator and ty != 'R' and i != len(layers) - 1:
            layer = create_lrelu_layer(last_layer, layer_name + '_lrelu')
            arch_list.append(layer)
            last_layer = layer_name + '_lrelu'

            if i != 0:
                if norm_layer == 'BN':
                    layer = create_batchnorm_layer(last_layer, layer_name + '_bn')
                    arch_list.append(layer)
                    last_layer = layer_name + '_bn'
                elif norm_layer == 'IN':
                    layer = create_instancenorm_layer(last_layer, layer_name + '_in')
                    arch_list.append(layer)
                    last_layer = layer_name + '_in'

        else:
            last_layer = layer_name

    return arch_list


def create_batchnorm_layer(bottom, name):
    return create_layer(bottom, name, 'batchnorm')


def create_instancenorm_layer(bottom, name):
    return create_layer(bottom, name, 'instancenorm')


def create_relu_layer(bottom, name):
    return create_layer(bottom, name, 'relu')


def create_lrelu_layer(bottom, name):
    return create_layer(bottom, name, 'lrelu')


def create_layer(bottom, name, op):
    layer = {'name':   name,
             'op':     op,
             'bottom': bottom}
    return layer





def gen2disc(gen_str):
    """
    Implement DCGAN rules to convert generator string to discriminator string.
    """
    disc_str = ''
    layers_inv = gen_str.split('_')[::-1]
    for i in range(len(layers_inv)):
        ty = layers_inv[i][0]
        if i == len(layers_inv) - 1:
            nout = 1
        else:
            nout_loc = layers_inv[i+1].rfind('-') + 1
            nout     = layers_inv[i+1][nout_loc:]

        if ty == 'D':
            kernel_size = layers_inv[i][1:layers_inv[i].find('-')]
            disc_str += 'C{}-{}_'.format(kernel_size, nout)
        elif ty in ['R', 'F']:
            disc_str += '{}-{}_'.format(ty, nout)
        else:
            raise ValueError('Layer type {} not implemented yet'.format(ty))

    return disc_str[:-1]  # Remove last underscore


def deconv_and_crop(bottom, kernel_size, strides, filters, padding, name):
    """
    Operation implementing deconvolution + crop, as described in Section 4.1 of our paper.
    """
    res = tf.layers.conv2d_transpose(bottom, filters=filters, kernel_size=kernel_size, strides=strides, padding='valid', name=name)
    d = kernel_size - 2
    return res[:, d:-d, d:-d, :]


def make_net(bottom, arch_str, is_generator, concat_list=[], norm_layer='BN', reuse=False, is_training=None,
             do_print=True):
    """
    Create generator/discriminator from architecture string.

    Inputs
    ------
    bottom          Input tensor to network.
    arch_str        Architecture string. See arch_string2list for details on formatting.
    is_generator    True if network being created is generator, False if discriminator.
    concat_list     List of tensors to concatenate to each deconvolution input (in generator).
    norm_layer      Normalization layer to use throughout network (BN for batchnorm, IN for instane norm, None for none)
    reuse           If set to True, reuse weights.
    is_training     If set to True, run batchnorm layers in training mode. Otherwise run in evaluation mode.
    do_print        If set to True, print network architecture.

    Outputs
    -------
    layer_tf        Output tensor of network.
    """

    assert is_training is not None, "is_training must be set"

    # Create arch list from arch string
    arch_list = arch_string2list(arch_str, is_generator=is_generator, norm_layer=norm_layer)

    # Make sure that the generator is being created if any input is given (other than bottom)
    assert len(concat_list) == 0 or is_generator, "Discriminator should not have concat_list"

    # Print network architecture
    if do_print:
        print('Creating {} network:'.format('generator' if is_generator else 'discriminator'))
        print_arch_list(arch_list)

    # Create dictionary of layer names
    name2layer = {'bottom': bottom}

    # Create layers one by one
    for layer in arch_list:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # Set batchnorm mode
        if layer['op'] == 'batchnorm':
            layer['params'] = {'training': is_training}

        # If layer has parameters, apply them. For deconvolution perform concatenation.
        if 'params' in layer.keys():
            if layer['op'] == 'deconv' and len(concat_list) > 0:
                tensor_to_concat = concat_list.pop(0)  # Pop the first tensor on list
                # If it's None, don't concat anything. Otherwise, concat it to layer's input.
                if tensor_to_concat is None:
                    bottom_ = name2layer[layer['bottom']]
                else:
                    deconv_bottom = name2layer[layer['bottom']]
                    # If there is no bottom, just take tensor_to_concat (e.g. if dim_z_global = dim_z_local = 0)
                    if deconv_bottom is not None:
                        bottom_ = tf.concat([deconv_bottom, tensor_to_concat], axis=3)
                    else:
                        bottom_ = tensor_to_concat
            else:
                bottom_ = name2layer[layer['bottom']]

            layer_tf = op2layer(layer['op'])(bottom_, **layer['params'], name=layer['name'])
            if do_print:
                print("Creating {} layer {} with parameters {}".format(layer['op'], layer['name'], layer['params']))
        elif layer['op'] == 'instancenorm':  # Instace norm layer does not take a name parameter for some reason...
            layer_tf = op2layer(layer['op'])(name2layer[layer['bottom']], reuse=reuse, scope=layer['name'])
            if do_print:
                print("Creating {} layer {}".format(layer['op'], layer['name']))
        else:
            layer_tf = op2layer(layer['op'])(name2layer[layer['bottom']], name=layer['name'])
            if do_print:
                print("Creating {} layer {}".format(layer['op'], layer['name']))

        name2layer[layer['name']] = layer_tf

    assert len(concat_list) == 0, "Not all concat tensors were used (length at the end is {})".format(len(concat_list))

    # For generator, find kernel size (which we assert is constant for all deconv layers)
    # and crop k - 2 pixels from each of the two dimensions
    if is_generator:
        for layer in arch_list:
            if layer['op'] == 'deconv':
                kernel_size = layer['params']['kernel_size']
                break
        return layer_tf[:, :-(kernel_size-2), :-(kernel_size-2), :]

    return layer_tf


def print_arch_list(arch_list):
    """
    Print architecture from an architecture list (obtained from arch_string2list()).
    """
    # Print output layer
    print("{:^30}".format('output'))
    print(" " + " " * 13 + "|" + " " * 14 + " ")

    # For each layer (from top to bottom) print a square with its name and its op
    for i, layer in enumerate(arch_list[::-1]):
        print(" " + "-" * 28 + " ")
        print("| {:16} {:10}|".format(layer['name'], layer['op']))
        print(" " + "-" * 28 + " ")
        print(" " + " " * 13 + "|" + " " * 14 + " ")

    # Finally print the input layer
    print("{:^30}".format('input'))


if __name__ == '__main__':
    """
    Example usage.
    """
    gen_arch = "D5-512_D5-256_D5-128_D5-64_D5-3"
    disc_arch = gen2disc(gen_arch)

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='input')

    dim_z = 191
    with tf.variable_scope('generator'):
        generated_img = make_net(x, gen_arch, is_generator=True, is_training=True)
    with tf.variable_scope('discriminator'):
        p_real = make_net(generated_img, gen_arch, is_generator=True, is_training=True)
