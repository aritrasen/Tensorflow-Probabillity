# import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.layers import util as tfp_layers_util


def gen_priordist(std=0.1):
    def default_multivariate_normal_fn(dtype, shape, name, trainable,
                                       add_variable_fn, std=std):

        del name, trainable, add_variable_fn   # unused
        tfd = tf.contrib.distributions
        dist = tfd.Normal(
            loc=tf.zeros(shape, dtype),
            scale=dtype.as_numpy_dtype(std)
        )
        # keep mean as 0, change scales
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
    return default_multivariate_normal_fn


def gen_postdist(std, is_singular=False,
                 trainmean=True):

    tfd = tf.contrib.distributions

    def _fn(dtype, shape, name, trainable, add_variable_fn,
            is_singular=is_singular):

        loc = add_variable_fn(
            name=name + '_loc',
            trainable=trainmean,
            shape=shape,
            dtype=dtype,
            initializer=tf.random_normal_initializer(stddev=0.1)
        )

        dist = tfd.Normal(loc=loc, scale=std)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
    return _fn


def convnet(inshape, numclass, activation=tf.nn.relu,
            priorstd=1, poststd=None, isBay=False):
    priorfn = gen_priordist(std=priorstd)
    if poststd is None:
        postfn = tfp_layers_util.default_mean_field_normal_fn()
    else:
        postfn = gen_postdist(std=poststd)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape(inshape))
    # if isBay:
    #     layer = tfp.layers.Convolution2DFlipout(
    #         32, kernel_size=3, padding="SAME",
    #         activation=self.activation)
    # else:
    layer = tf.keras.layers.Conv2D(
        32, kernel_size=3, padding="SAME",
        activation=activation)
    model.add(layer)
    model.add(
        tf.keras.layers.MaxPool2D(
            pool_size=[2, 2], strides=[2, 2],
            padding='SAME')
    )
    model.add(tf.keras.layers.Flatten())
    if isBay:
        model.add(
            tfp.layers.DenseFlipout(
                numclass,
                kernel_prior_fn=priorfn,
                kernel_posterior_fn=postfn
            )
        )
    else:
        model.add(tf.keras.layers.Dense(numclass))

    return model

# MNIST dataset


def fullnet(numclass, activation=tf.nn.relu,
            priorstd=1, poststd=None, layer_sizes=[100, 50, 10], isBay=False):

    priorfn = gen_priordist(std=priorstd)
    if poststd is None:
        postfn = tfp_layers_util.default_mean_field_normal_fn()
    else:
        postfn = gen_postdist(std=poststd)

    model = tf.keras.Sequential()
    for i in range(len(layer_sizes[:-1])):
        if isBay:
            layer = tfp.layers.DenseFlipout(
                layer_sizes[i],
                activation=activation,
                kernel_prior_fn=priorfn,
                kernel_posterior_fn=postfn
            )
        else:
            layer = tf.keras.layers.Dense(
                layer_sizes[i],
                activation=activation
            )
        model.add(layer)

    if isBay:
        model.add(
            tfp.layers.DenseFlipout(
                numclass,
                kernel_prior_fn=priorfn,
                kernel_posterior_fn=postfn
            )
        )
    else:
        model.add(tf.keras.layers.Dense(numclass))

    return model
