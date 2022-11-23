import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from visualize import plot_reconstruction

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr, nbit=16):
    return resolve16(model, tf.expand_dims(lr, axis=0), nbit=nbit)[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float16)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def resolve16(model, lr_batch, nbit=16):
    if nbit == 8:
        casttype = tf.uint8
    elif nbit == 16:
        casttype = tf.uint16
    else:
        print("Wrong number of bits")
        exit()
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 2**nbit - 1)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, casttype)
    return sr_batch


def evaluate(model, dataset, nbit=8, show_image=False):
    psnr_values = []
    has_uq = 'uq' in model.__name__
    lr_output, hr_output, sr_output, uq_output = None, None, None, None
    for idx, (lr, hr) in enumerate(dataset):
        sr = resolve16(model, lr, nbit=nbit)  # hack
        uq = None
        if has_uq:
            sr = sr[:, :, :, 0, None]
            uq = sr[:, :, :, 1, None]
        else:
            if lr.shape[-1] == 1:
                sr = sr[..., 0, None]
        psnr_value = psnr(hr, sr, nbit=nbit)[0]
        psnr_values.append(psnr_value)
        # we only need to show one, just pick the first one
        if idx == 0:
            lr_output, hr_output, sr_output, uq_output = lr, hr, sr, uq
    if show_image:
        # plot images here
        print('v1')
        plot_reconstruction(datalr=lr_output, datahr=hr_output, datasr=sr_output, datauq=uq_output)

        plt.hist(sr_output.numpy().flatten(), bins=20)
        plt.yscale("log")
        plt.title("SR histogram")
        plt.show()
        plt.hist(hr_output.numpy().flatten(), bins=20)
        plt.yscale("log")
        plt.title("HR histogram")
        plt.show()
        if has_uq:
            plt.hist(uq_output.numpy().flatten(), bins=20)
            plt.yscale("log")
            plt.title("Uncertainty histogram")
            plt.show()
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------
# def normalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
#    if True:
#        return (x - rgb_mean) / 127.5
#    elif nbit==16:
#        return (x - 2.**15)/2.**15


# def denormalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
#    if True:
#        return x * 127.5 + rgb_mean


def normalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
    if nbit == 8:
        return (x - rgb_mean) / 127.5
    elif nbit == 16:
        return (x - 2.0**15) / 2.0**15


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
    if nbit == 8:
        return x * 127.5 + rgb_mean
    elif nbit == 16:
        return x * 2**15 + 2**15


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2, nbit=8):
    return tf.image.psnr(x1, x2, max_val=2**nbit - 1)


def psnr16(x1, x2):
    return tf.image.psnr(x1, x2, max_val=2**16 - 1)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
