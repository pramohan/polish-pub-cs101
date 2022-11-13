import tensorflow_addons as tfa

from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model

from model.common import normalize, denormalize, pixel_shuffle

import tensorflow as tf


def wdsr_a(
    scale,
    num_filters=32,
    num_res_blocks=8,
    res_block_expansion=4,
    res_block_scaling=None,
):
    return wdsr(
        scale,
        num_filters,
        num_res_blocks,
        res_block_expansion,
        res_block_scaling,
        res_block_a,
    )


def wdsr_b(
    scale,
    num_filters=32,
    num_res_blocks=8,
    res_block_expansion=6,
    res_block_scaling=None,
    nchan=1,
):
    return wdsr(
        scale,
        num_filters,
        num_res_blocks,
        res_block_expansion,
        res_block_scaling,
        res_block_b,
        nchan,
    )

def wdsr_b_uq(
    scale,
    num_filters=32,
    num_res_blocks=8,
    res_block_expansion=6,
    res_block_scaling=None,
    nchan=1,
    output_chan=2,
):
    x_in = Input(shape=(None, None, output_chan))
    x = Lambda(normalize)(x_in)

    # main branch
    #    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    m = conv2d_weightnorm(num_filters, output_chan, padding="same")(x)
    for i in range(num_res_blocks):
        m = res_block_b(
            m,
            num_filters,
            res_block_expansion,
            kernel_size=3,
            scaling=res_block_scaling,
        )
    m = conv2d_weightnorm(
        output_chan * scale**2, 3, padding="same", name=f"conv2d_main_scale_{scale}"
    )(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(
        output_chan * scale**2, 5, padding="same", name=f"conv2d_skip_scale_{scale}"
    )(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = tf.keras.activations.sigmoid(x)
    x = Lambda(denormalize)(x)

    return Model(x_in, x, name="wdsr_b_uq")


def wdsr_mc(
    scale,
    num_filters=32,
    num_res_blocks=8,  # 32 for wdsr-b
    res_block_expansion=6,
    res_block_scaling=None,
    nchan=1,
):
    x_in = Input(shape=(None, None, nchan))
    x = Lambda(normalize)(x_in)

    # main branch
    #    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    m = dropout_mc_wrapper(x)
    m = conv2d_weightnorm(num_filters, nchan, padding="same")(m)
    for i in range(num_res_blocks):
        m = dropout_mc_wrapper(m)
        m = res_block_b(
            m,
            num_filters,
            res_block_expansion,
            kernel_size=3,
            scaling=res_block_scaling,
        )
    m = dropout_mc_wrapper(m)
    m = conv2d_weightnorm(
        nchan * scale**2, 3, padding="same", name=f"conv2d_main_scale_{scale}"
    )(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    m = dropout_mc_wrapper(m)
    s = conv2d_weightnorm(
        nchan * scale**2, 5, padding="same", name=f"conv2d_skip_scale_{scale}"
    )(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = Lambda(denormalize)(x)

    return Model(x_in, x, name="wdsr")


def wdsr2(
    scale,
    num_filters,
    num_res_blocks,
    res_block_expansion,
    res_block_scaling,
    res_block,
):
    x_in = Input(shape=(None, None, 1))
    x = Lambda(normalize)(x_in)

    # main branch
    m = conv2d_weightnorm(num_filters, 3, padding="same")(x)
    #    m = conv2d_weightnorm(num_filters, 1, padding='same')(x)
    for i in range(num_res_blocks):
        m = res_block(
            m,
            num_filters,
            res_block_expansion,
            kernel_size=3,
            scaling=res_block_scaling,
        )
    m = conv2d_weightnorm(
        3 * scale**2, 3, padding="same", name=f"conv2d_main_scale_{scale}"
    )(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(
        3 * scale**2, 5, padding="same", name=f"conv2d_skip_scale_{scale}"
    )(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = Lambda(denormalize)(x)

    return Model(x_in, x, name="wdsr")


def dropout_mc_wrapper(x, rate=0.15):
    # print('Dropout being used!')
    return tf.nn.dropout(x, rate)


def wdsr(
    scale,
    num_filters,
    num_res_blocks,
    res_block_expansion,
    res_block_scaling,
    res_block,
    nchan=1,
):
    x_in = Input(shape=(None, None, nchan))
    x = Lambda(normalize)(x_in)

    # main branch
    #    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    m = conv2d_weightnorm(num_filters, nchan, padding="same")(x)
    for i in range(num_res_blocks):
        m = res_block(
            m,
            num_filters,
            res_block_expansion,
            kernel_size=3,
            scaling=res_block_scaling,
        )
    m = conv2d_weightnorm(
        nchan * scale**2, 3, padding="same", name=f"conv2d_main_scale_{scale}"
    )(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(
        nchan * scale**2, 5, padding="same", name=f"conv2d_skip_scale_{scale}"
    )(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = Lambda(denormalize)(x)

    return Model(x_in, x, name="wdsr")


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x = conv2d_weightnorm(
        num_filters * expansion, kernel_size, padding="same", activation="relu"
    )(x_in)
    x = conv2d_weightnorm(num_filters, kernel_size, padding="same")(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = conv2d_weightnorm(
        num_filters * expansion, 1, padding="same", activation="relu"
    )(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding="same")(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding="same")(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def conv2d_weightnorm(filters, kernel_size, padding="same", activation=None, **kwargs):
    return tfa.layers.WeightNormalization(
        Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs),
        data_init=False,
    )
