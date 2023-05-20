import tensorflow as tf
from functools import reduce
import operator

# https://arxiv.org/pdf/1902.08153.pdf


def _grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return tf.stop_gradient(y - y_grad) + y_grad


@tf.custom_gradient
def _round_pass(x):
    # Can be implemented more efficiently with custom gradient
    y = tf.round(x)

    def grad(dy):
        return dy
    return y, grad


class Quantize(tf.keras.layers.Layer):
    def __init__(self, is_activation=True, p=8, regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.is_activation = is_activation
        self.regularizer = regularizer
        if False:
            # quantizing activation
            # qn = 0
            # qp = 2 ** p - 1
            self.qn = 0
            self.qp = 2 ** p - 1
        else:
            # We'll say that we never have positivity guaranteed
            # quantizing weight
            self.qn = -2 ** (p - 1)
            self.qp = 2 ** (p - 1) - 1

    def build(self, input_shape):
        self.s = self.add_weight(name='quantize_scale',
                                 shape=[],
                                 initializer=tf.keras.initializers.Constant(
                                     999),
                                 trainable=True,
                                 regularizer=self.regularizer,
                                 constraint=tf.keras.constraints.NonNeg())

        if self.is_activation:
            n_features = input_shape[1]
            assert n_features > 8, "n_features <= 8, is your data format correct?"
        else:
            n_features = reduce(operator.mul, input_shape, 1)
        self.n_features = n_features

    def call(self, inputs):

        if inputs.shape[0] is not None:
            # !!! hack to get quantization initialized
            mean, std = tf.reduce_mean(inputs), tf.math.reduce_std(inputs)
            s_init = tf.math.maximum(
                tf.abs(mean + 3 * std), tf.abs(mean - 3 * std)) / 2**(self.p - 1) + 1e-8
            self.s.assign(tf.cast(tf.greater_equal(self.s, 100), self.dtype)
                          * s_init + tf.cast(tf.less(self.s, 100), self.dtype) * self.s)

        dtype = inputs.dtype
        qn, qp, n_features = tf.cast(self.qn, dtype), tf.cast(
            self.qp, dtype), tf.cast(self.n_features, dtype)

        s_grad_scale = tf.math.rsqrt(n_features * qp)
        s_scale = _grad_scale(self.s, s_grad_scale)

        x = inputs / s_scale
        x = tf.clip_by_value(x, qn, qp)
        x = _round_pass(x)
        x = x * s_scale
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'p': self.p, 'is_activation': self.is_activation,
                      's': self.s, 's_initialized': self.s_initialized})
        return config


class QuantizedConv(tf.keras.layers.Layer):
    def __init__(self, filters, p=8, use_bias=False, kernel_regularizer=None, regularizer=None, kernel_initializer="glorot_normal", data_format="channels_first", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.p = p
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.regularizer = regularizer
        if kernel_regularizer is None:
            self.kernel_regularizer = regularizer
        self.kernel_initializer = kernel_initializer
        self.data_format = data_format

    def build(self, input_shape, **kwargs):
        in_channels = input_shape[1] if self.data_format == "channels_first" else input_shape[-1]
        assert in_channels > 8, "in_channels <= 8, is your data format correct?"
        self.weight = self.add_weight(name='kernel',
                                      shape=[3, 3, in_channels, self.filters],
                                      initializer=self.kernel_initializer,
                                      trainable=True, regularizer=self.kernel_regularizer)
        self.bias = self.add_weight(name='bias', shape=[self.filters],
                                    initializer=tf.keras.initializers.Zeros(), trainable=True) if self.use_bias else None
        self.weight_quantize = Quantize(
            is_activation=False, p=self.p, regularizer=self.regularizer, name=self.name+"/weight_quantize")

    def call(self, inputs):
        weight = self.weight_quantize(self.weight)
        return tf.nn.conv2d(inputs, weight, strides=1, padding='SAME',
                            data_format="NCHW" if self.data_format == "channels_first" else "NHWC") + (self.bias if self.use_bias else 0)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'p': self.p, 'use_bias': self.use_bias, 'kernel_regularizer': self.kernel_regularizer,
                      'kernel_initializer': self.kernel_initializer, 'data_format': self.data_format, 'weight': self.weight, 'bias': self.bias, 'weight_quantize': self.weight_quantize})
        return config


class QuantizedDense(tf.keras.layers.Layer):
    def __init__(self, out_units, p=8, use_bias=False, kernel_regularizer=None, regularizer=None, kernel_initializer="glorot_normal", **kwargs):
        super().__init__(**kwargs)
        self.out_units = out_units
        self.p = p
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.regularizer = regularizer
        if kernel_regularizer is None:
            self.kernel_regularizer = regularizer
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape, **kwargs):
        in_units = input_shape[-1]
        self.weight = self.add_weight(name='kernel',
                                      shape=[in_units, self.out_units],
                                      initializer=self.kernel_initializer,
                                      trainable=True, regularizer=self.kernel_regularizer)
        self.bias = self.add_weight(name='bias', shape=[self.filters],
                                    initializer=tf.keras.initializers.Zeros(), trainable=True) if self.use_bias else None
        self.weight_quantize = Quantize(
            is_activation=False, p=self.p, regularizer=self.regularizer, name=self.name+"/kernel_quantize")

    def call(self, inputs):
        weight = self.weight_quantize(self.weight)
        return tf.einsum('bsi,io->bso', inputs, weight)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'p': self.p, 'use_bias': self.use_bias, 'kernel_regularizer': self.kernel_regularizer,
                      'kernel_initializer': self.kernel_initializer, 'data_format': self.data_format, 'weight': self.weight, 'bias': self.bias, 'weight_quantize': self.weight_quantize})
        return config


def quantized_dense(in_units, name, p=8, regularizer=None, **kwargs):
    return tf.keras.Sequential([
        Quantize(is_activation=True, p=p, regularizer=regularizer,
                 name=name),
        QuantizedDense(in_units, p=p, name=name,
                       regularizer=regularizer, **kwargs)

    ])
