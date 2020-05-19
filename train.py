import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time
import os
import sys
from utils import create_dir_if_needed, plot_image, save_image
from PIL import Image
import argparse
from make_net import gen2disc
from create_networks import create_generator_discriminator, create_loss, create_dg_optimizers, create_no_nan_assertions
from apply_warps import apply_warps
from create_unwarper import create_unwarper, create_w_optimizers
from sft_utils import get_z_shape_list, get_feed_dict


def param_names():
    """
    Create list of all parameter names.
    """
    return ['num_steps', 'gen_arch', 'disc_arch', 'dim_z_local_list', 'dim_z_periodic_list', 'dim_z_global',
            'num_gan_updates', 'num_shape_updates', 'n_learning_rate', 't_learning_rate',
            'n_smoothness_weight', 't_smoothness_weight', 'int_weight', 'output_shape_gen', 'input_shape_w',
            'image_path']


def str2bool(s):
    """
    Get boolean from string.
    """
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'




def train(args):
    """
    Train three-player game.
    """

    # Create folders if they don't exist
    image_dir = os.path.join(args.output_folder, 'output_images')
    model_dir = os.path.join(args.output_folder, 'models')
    create_dir_if_needed(image_dir)
    create_dir_if_needed(model_dir)

    # For disc_arch use the discriminator for gen_arch (using DCGAN rules), unless specified otherwise
    if args.disc_arch == '':
        args.disc_arch = gen2disc(args.gen_arch)

    # Save parameters to file
    with open(os.path.join(model_dir, 'params.txt'), 'w') as f:
        for p in param_names():
            f.write('{:<30} {:>30}\n'.format(p, str(getattr(args, p))))

    tf.reset_default_graph()

    # Define constants for training
    batch_size = 25           # number of patches to use for each batch
    learning_rate = 0.0002    # network learning rate
    beta1 = 0.5               # parameter for Adam optimizer
    wd_mult = 1e-8            # weight decay parameter

    # Define constants for network architecture
    do_tie_phases = True
    global_mlp_hidden_units = 60

    # Compute total number of periodic dimensions
    dim_z_periodic = sum(args.dim_z_periodic_list)

    # Load input image
    img = np.array(Image.open(args.image_path), dtype=np.float32) / 255.0
    img_height = img.shape[0]
    img_width  = img.shape[1]
    num_channels = img.shape[2]

    # Make sure number of concatenations is the same as number of deconvolutions
    num_deconv = args.gen_arch.count('D')
    assert len(args.dim_z_periodic_list) == num_deconv, \
        "Number of concatenations must be the same as the number of deconvolutions"
    assert len(args.dim_z_local_list) == num_deconv, \
        "Number of concatenations must be the same as the number of deconvolutions"

    # Define shape of input noise
    z_shape = [x // (2 ** num_deconv) for x in args.output_shape_gen]
    assert tuple([x * 2 ** num_deconv for x in z_shape]) == args.output_shape_gen, \
        "Output size must be divisible by scale factor ({} vs. {})".format(args.output_shape_gen, 2 ** num_deconv)

    assert int(args.gen_arch[args.gen_arch.rfind('-')+1:]) == num_channels, \
        "Output of generator must match number of channels in image"

    # Define placeholders for sample points and patch sizes
    loc = tf.placeholder(tf.float32, shape=[None, 2], name='loc')
    shape = tf.placeholder(tf.int64, shape=[2], name='shape')

    # Create unwarper
    warps, n, t, n_smoothness_loss, t_smoothness_loss, integrability_loss = create_unwarper(img_height, img_width)

    # Get unwarped patches
    input_img_batch = np.repeat(img[np.newaxis, :, :, :], repeats=batch_size, axis=0)
    disc_input = apply_warps(input_img_batch, loc, warps, shape)

    # Create generator and discriminator (feeding the unwarped patches to the discriminator via disc_input)
    dict_ph, dict_tensors = create_generator_discriminator(args.gen_arch, args.disc_arch, z_shape, args.dim_z_global,
                                                           args.dim_z_local_list, args.dim_z_periodic_list,
                                                           global_mlp_hidden_units, do_tie_phases, do_print=False,
                                                           disc_input=disc_input)

    # Get placeholders from generator and discriminator
    is_training           = dict_ph.get('is_training', None)
    disc_input            = dict_ph.get('disc_input', None)
    global_gen_input      = dict_ph.get('global_gen_input', None)
    local_gen_input_list  = dict_ph.get('local_gen_input_list', None)
    random_phases         = dict_ph.get('random_phases', None)

    # Get tensors from generator and discriminator
    disc_fake             = dict_tensors.get('disc_fake', None)
    disc_real             = dict_tensors.get('disc_real', None)
    gen_sample            = dict_tensors.get('gen_sample', None)

    # Define losses for three players
    shape_loss = args.int_weight * integrability_loss \
        + args.n_smoothness_weight * n_smoothness_loss \
        + args.t_smoothness_weight * t_smoothness_loss \
        + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_real), logits=disc_real))

    gen_loss, disc_loss = create_loss(disc_real, disc_fake, wd_mult)

    # Create optimizers and update step operations for unwarper, generator and discriminator
    train_gen, train_disc = create_dg_optimizers(gen_loss, disc_loss, learning_rate, beta1)
    train_shape = create_w_optimizers(shape_loss, args.n_learning_rate, args.t_learning_rate)

    # Create assertions to check that no values in the network are NaN
    no_nan_assertions = create_no_nan_assertions()

    # Create list of input sizes for each deconvolution layer
    # Use padding as explained in Section 4.1 of the paper.
    z_shape_list = get_z_shape_list(z_shape, len(args.dim_z_local_list), gen_arch=args.gen_arch)

    # Save all network parameters (except those from the unwarper -- the normal and tangent maps are saved as images)
    vars_to_save = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'warp' not in x.name]
    saver = tf.train.Saver(vars_to_save, max_to_keep=1)

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Define variables used for timing training
        i_prev = 0
        t_prev = time.time()

        # For each iteration, train generator and then train discriminator. Every num_gan_updates iterations,
        # update shape parameters for num_shape_updates iterations.
        for step in range(1, args.num_steps+1):

            ##########################################
            #####                                #####
            #####        Train generator         #####
            #####                                #####
            ##########################################

            # Sample Z for generator input
            feed_dict_gen_step = get_feed_dict(local_gen_input_list, global_gen_input, random_phases,
                                               args.dim_z_local_list, args.dim_z_global, dim_z_periodic, do_tie_phases,
                                               batch_size, z_shape_list)

            feed_dict_gen_step[is_training] = True  # Set batchnorm mode to training

            # Perform step
            _, gl = sess.run([train_gen, gen_loss], feed_dict=feed_dict_gen_step)


            ##########################################
            #####                                #####
            #####      Train discriminator       #####
            #####                                #####
            ##########################################

            # Sample Z for generator input
            feed_dict_disc_step = get_feed_dict(local_gen_input_list, global_gen_input, random_phases,
                                                args.dim_z_local_list, args.dim_z_global, dim_z_periodic, do_tie_phases,
                                                batch_size, z_shape_list)

            # Select size of patches randomly from input_shape_w
            shape_ = np.array([np.random.permutation(args.input_shape_w)[0]] * 2)

            # Generate random crop (start_row and start_col are the topmost row and leftmost column of the crop)
            start_row = np.random.randint(0, img_height-shape_[0], batch_size)
            start_col = np.random.randint(0, img_width-shape_[1], batch_size)

            # Add values to feed_dict
            feed_dict_disc_step[shape] = shape_
            feed_dict_disc_step[loc] = np.concatenate([start_col[:, np.newaxis] + (shape_[1] - 1) / 2.0,
                                                       start_row[:, np.newaxis] + (shape_[0] - 1) / 2.0], axis=1)

            feed_dict_disc_step[is_training] = True  # Set batchnorm mode to training

            # Perform step
            _, dl = sess.run([train_disc, disc_loss], feed_dict=feed_dict_disc_step)


            ##########################################
            #####                                #####
            #####         Train unwarper         #####
            #####                                #####
            ##########################################

            if step % args.num_gan_updates == 0:
                for _ in range(args.num_shape_updates):

                    # Select size of patches randomly from input_shape_w
                    shape_ = np.array([np.random.permutation(args.input_shape_w)[0]] * 2)

                    # Generate random crop (start_row and start_col are the topmost row and leftmost column of the crop)
                    start_row = np.random.randint(0, img_height-shape_[0], batch_size)
                    start_col = np.random.randint(0, img_width-shape_[1], batch_size)

                    feed_dict_shape_step = {is_training: False}  # Run batchnorm in evaluation mode
                    feed_dict_shape_step[shape] = shape_
                    feed_dict_shape_step[loc] = np.concatenate([start_col[:, np.newaxis] + (shape_[1] - 1) / 2.0,
                                                                start_row[:, np.newaxis] + (shape_[0] - 1) / 2.0],
                                                               axis=1)

                    # Perform step
                    _ = sess.run(train_shape, feed_dict=feed_dict_shape_step)

            # Make sure no tensors have any NaN every once in a while
            if step % 300 == 0:
                sess.run(no_nan_assertions)

            # Log progress
            if step % 100 == 0 or step == 1:
                logline = "Step {}: generator loss: {}; discriminator loss: {}; time per iteration: {} seconds."\
                    .format(step, gl, dl, (time.time() - t_prev) / (step - i_prev))
                t_prev = time.time()
                i_prev = step
                print(logline)

                sys.stdout.flush()

            # Plot stuff every once in a while
            if step % 100 == 0:

                # Sample Z for generator input
                feed_dict = get_feed_dict(local_gen_input_list, global_gen_input, random_phases, args.dim_z_local_list,
                                          args.dim_z_global, dim_z_periodic, do_tie_phases, 9, z_shape_list)

                feed_dict[is_training] = False  # Run batchnorm in evaluation mode

                # Get patch shapes
                shape_ = np.array([np.random.permutation(args.input_shape_w)[0]] * 2)

                # Generate random crop (start_row and start_col are the topmost row and leftmost column of the crop)
                start_row = np.random.randint(0, img_height-shape_[0], batch_size)
                start_col = np.random.randint(0, img_width-shape_[1], batch_size)


                feed_dict[shape] = shape_

                feed_dict[loc] = np.concatenate([start_col[:, np.newaxis] + (shape_[1] - 1) / 2.0,
                                                 start_row[:, np.newaxis] + (shape_[0] - 1) / 2.0], axis=1)

                # Get output of generator and unwarper
                generated_samples, disc_input_ = sess.run([gen_sample, disc_input], feed_dict=feed_dict)
                H, W = generated_samples.shape[1:3]
                b = 2  # Border, in pixels
                sqrt_samples = 3  # Number of rows and columns in grid of samples
                imgs = np.ones((sqrt_samples*H+(sqrt_samples-1)*b, sqrt_samples*W+(sqrt_samples-1)*b, 3),
                               dtype=np.float32)
                for row in range(sqrt_samples):
                    for col in range(sqrt_samples):
                        imgs[row*(H+b):row*(H+b)+H, col*(W+b):col*(W+b)+W, :] = \
                            generated_samples[row*sqrt_samples+col, :, :, :]  # Reshape to (3H, 3W, n_channels)

                H, W = disc_input_.shape[1:3]
                real_imgs = np.ones((sqrt_samples*H+(sqrt_samples-1)*b, sqrt_samples*W+(sqrt_samples-1)*b, 3),
                                    dtype=np.float32)
                for row in range(sqrt_samples):
                    for col in range(sqrt_samples):
                        real_imgs[row*(H+b):row*(H+b)+H, col*(W+b):col*(W+b)+W, :] = \
                            disc_input_[row*sqrt_samples+col, :, :, :]  # Reshape to (3H, 3W, n_channels)

                # Get normal maps
                n_ = sess.run(n)
                n_[:, :, 1] *= -1.0       # Flip y axis in normals
                n_img = (n_ + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]

                # Plot output of generator, normal vectors, output of unwarper
                if args.do_plot:
                    plt.figure(1)
                    plt.subplot(1, 3, 1)
                    plot_image(imgs, vmin=0.0, vmax=1.0)
                    plt.subplot(1, 3, 2)
                    plot_image(n_img, title='niter={}'.format(step), vmin=0.0, vmax=1.0)
                    plt.subplot(1, 3, 3)
                    plot_image(real_imgs, vmin=0.0, vmax=1.0)
                    plt.show()

            # Every one in a while, save images
            if step % 500 == 0:
                # Save normal map
                image_filename = os.path.join(image_dir, 'n_img_iter_{}.jpg'.format(step))
                save_image(image_filename, np.uint8(n_img * 255.0))

                # Save tangent map
                t_ = sess.run(t)
                t_[:, :, 1] *= -1.0       # Flip y axis in tangent vectors
                t_img = (t_ + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
                image_filename = os.path.join(image_dir, 't_img_iter_{}.jpg'.format(step))
                save_image(image_filename, np.uint8(t_img * 255.0))

                # Save output of generator
                image_filename = os.path.join(image_dir, 'generator_output_iter_{}.jpg'.format(step))
                save_image(image_filename, np.uint8(imgs * 255.0))

                # Save output of unwarper
                image_filename = os.path.join(image_dir, 'unwarper_output_iter_{}.jpg'.format(step))
                save_image(image_filename, np.uint8(real_imgs * 255.0))

            # Save generator and discriminator to file
            if args.do_save_model:
                if step % 1000 == 0:
                    saver.save(sess, os.path.join(model_dir, "model{}.ckpt".format(step)), write_meta_graph=False)


if __name__ == '__main__':
    # Define dictionary holding all parameters
    d = dict()
    d['num_steps'] = dict(type=int, default=25000, help="Number of training iterations.")
    d['gen_arch'] = dict(type=str, default="D5-256_D5-128_D5-64_D5-3",
                         help="Generator architecture (see make_net.arch_string2list() for more information).")
    d['disc_arch'] = dict(type=str, default="", help="Discriminator architecture. If unspecified use DCGAN rules.")
    d['dim_z_local_list'] = dict(type=int, nargs='*', default=(0, 0, 2, 2), help="List of local dimensions.")
    d['dim_z_periodic_list'] = dict(type=int, nargs='*', default=(2, 2, 0, 0), help="List of periodic dimensions.")
    d['dim_z_global'] = dict(type=int, default=2, help="Number of global dimensions.")

    d['num_gan_updates'] = dict(type=int, default=20, help="Number of G/D updates at a time.")
    d['num_shape_updates'] = dict(type=int, default=200, help="Number of W updates at a time.")

    d['n_learning_rate'] = dict(type=float, default=0.001, help="Learning rate for normal vectors.")
    d['t_learning_rate'] = dict(type=float, default=0.05, help="Learning rate for tangent vectors.")

    d['n_smoothness_weight'] = dict(type=float, default=100.0, help="Smoothness loss weight for normal vectors.")
    d['t_smoothness_weight'] = dict(type=float, default=100.0, help="Smoothness loss weight for tangent vectors.")
    d['int_weight'] = dict(type=float, default=10000000.0, help="Integration loss weight.")

    d['output_shape_gen'] = dict(type=int, nargs=2, default=(192, 192), help="Spatial size of generator output.")
    d['input_shape_w'] = dict(type=int, nargs='*', default=(192, 160, 128, 96), help="Spatial size of unwarper output.")

    d['image_path'] = dict(type=str, default=None, help="Path of input image to use.")

    d['do_plot'] = dict(type=str2bool, default='False', help="Whether or not to plot results to screen.")
    d['do_save_model'] = dict(type=str2bool, default='False', help="Whether or not to save models to disk.")
    d['output_folder'] = dict(type=str, default='', help="Folder for saving results and model.")

    # Create argument parser
    parser = argparse.ArgumentParser(description='Define parameters for three-player game.')

    # Fill arguments from dictionary
    for k in d.keys():
        parser.add_argument('-' + k, **d[k])

    args = parser.parse_args()

    train(args)
