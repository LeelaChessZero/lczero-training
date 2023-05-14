#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import os
import random
import tensorflow as tf
import time
import bisect
import tensorflow_addons as tfa

import lc0_az_policy_map
import attention_policy_map as apm
import proto.net_pb2 as pb
from functools import reduce
import operator
from math import sqrt
from net import Net


def max_min_avg_pooling(x):
    axes = (2,3) if x.shape[1] > 8 else (1,2)
    max_pooled = tf.math.reduce_max(x, axis=axes, keepdims=True)
    min_pooled = tf.math.reduce_min(x, axis=axes, keepdims=True)
    avg_pooled = tf.math.reduce_mean(x, axis=axes, keepdims=True)
    return tf.concat([max_pooled, min_pooled, avg_pooled], axis=1 if 3 in axes else -1)

def katago_pooling_layer(x, name):
    data_format = 'channels_first' if x.shape[1] > 8 else 'channels_last'
    out_filters = x.shape[1] if data_format == 'channels_first' else x.shape[-1]
    pooled = max_min_avg_pooling(x)
    return tf.keras.layers.Conv2D(filters, 1, 1, padding='same', name=name, data_format=data_format )(pooled)
    

# https://arxiv.org/pdf/1902.08153.pdf

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return tf.stop_gradient(y - y_grad) + y_grad


def round_pass(x):
    # Can be implemented more efficiently with custom gradient
    y = tf.round(x)
    y_grad = x
    return tf.stop_gradient(y - y_grad) + y_grad

class Quantize(tf.keras.layers.Layer):
    def __init__(self, is_activation=True, p=8, regularizer=None, min_value = 0, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.is_activation = is_activation
        self.regularizer = regularizer
        self.min_value = min_value
        assert not (min_value != 0 and is_activation), "min_value != 0 and is_activation"
        if self.is_activation:
            # quantizing activation
            # qn = 0
            # qp = 2 ** p - 1
            self.qn = 0
            self.qp = 2 ** p - 1
        else:
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
            s_init = tf.math.maximum(tf.abs(mean + 3 * std), tf.abs(mean - 3 * std)) / 2**(self.p - 1) + 1e-8
            self.s.assign(tf.cast(tf.greater_equal(self.s, 100), self.dtype) * s_init + tf.cast(tf.less(self.s, 100), self.dtype) * self.s)
        
        if self.min_value != 0:
            inputs = inputs - self.min_value
            

        dtype = inputs.dtype
        qn, qp, n_features = tf.cast(self.qn, dtype), tf.cast(self.qp, dtype), tf.cast(self.n_features, dtype)

        s_grad_scale = tf.math.rsqrt(self.n_features * qp)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = inputs / s_scale
        x = tf.clip_by_value(x, qn, qp)
        x = round_pass(x)
        x = x * s_scale

        if self.min_value != 0:
            x = x + self.min_value

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'p': self.p, 'is_activation': self.is_activation, 's': self.s, 's_initialized': self.s_initialized})
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
        self.kernel = self.add_weight(name='kernel',
                                        shape=[3, 3, in_channels, self.filters],
                                        initializer=self.kernel_initializer,
                                        trainable=True, regularizer=self.kernel_regularizer)
        self.bias = self.add_weight(name='bias', shape=[self.filters],
            initializer=tf.keras.initializers.Zeros(), trainable=True) if self.use_bias else None
        self.kernel_quantize = Quantize(is_activation=False, p=self.p, regularizer=self.regularizer, name=self.name+"/kernel_quantize")
    
    def call(self, inputs):
        kernel = self.kernel_quantize(self.kernel)
        return tf.nn.conv2d(inputs, kernel, strides=1, padding='SAME',
            data_format="NCHW" if self.data_format == "channels_first" else "NHWC" ) + (self.bias if self.use_bias else 0)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'p': self.p, 'use_bias': self.use_bias, 'kernel_regularizer': self.kernel_regularizer, 'kernel_initializer': self.kernel_initializer, 'data_format': self.data_format, 'weight': self.weight, 'bias': self.bias, 'weight_quantize': self.weight_quantize})
        return config

def quantized_conv(filters, name, p=4, regularizer=None, min_value=0, **kwargs):
    assert regularizer is not None, "quantization is unstable with regularization"
    return tf.keras.Sequential([
        Quantize(is_activation=True, p=p, regularizer=regularizer, min_value=min_value, name=name+"/activation_quantize"),
        QuantizedConv(filters, p=p, name=name+"/conv2d", regularizer=regularizer, **kwargs)
        # tf.keras.layers.Conv2D(filters,
        #                                 3,
        #                                 use_bias=False,
        #                                 padding='same',
        #                                 kernel_initializer='glorot_normal',
        #                                 kernel_regularizer=regularizer,
        #                                 data_format='channels_first',
        #                                 name=name + '/1/conv2d')
    ])


def make_sparsity_mask(x):
    # pick indices 2 out of every 4 values with greatest magnitude
    # (i.e. 1/2 of the values)
    # 2 out of 4 sparsity pattern

    # for conv2d
    shape = x.shape
    in_channels = x.shape[2]
    out_channels = x.shape[3]

    x_abs = tf.abs(x)
    x_abs = tf.reshape(x_abs, [-1, 4])

    top_2 = tf.math.top_k(x_abs, k=2, sorted=True)
    second_largest = top_2.values[:, 1:2]
    comparison = tf.math.greater_equal(x_abs, second_largest)
    comparison = tf.cast(comparison, tf.float32)
    mask = tf.reshape(
        comparison, shape)

    return mask



class ApplySqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)

    def build(self, input_dimens):
        self.reshape_size = input_dimens[1][1]

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        gammas, betas = tf.split(tf.reshape(excited,
                                            [-1, self.reshape_size, 1, 1]),
                                 2,
                                 axis=1)
        return tf.nn.sigmoid(gammas) * x + betas


class ApplyPolicyMap(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplyPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(lc0_az_policy_map.make_map())

    def call(self, inputs):
        h_conv_pol_flat = tf.reshape(inputs, [-1, 80 * 8 * 8])
        return tf.matmul(h_conv_pol_flat,
                         tf.cast(self.fc1, h_conv_pol_flat.dtype))


class ApplyAttentionPolicyMap(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplyAttentionPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(apm.make_map())

    def call(self, logits, pp_logits):
        logits = tf.concat([tf.reshape(logits, [-1, 64 * 64]),
                            tf.reshape(pp_logits, [-1, 8 * 24])],
                           axis=1)
        return tf.matmul(logits, tf.cast(self.fc1, logits.dtype))

class Metric:
    def __init__(self, short_name, long_name, suffix='', **kwargs):
        self.short_name = short_name
        self.long_name = long_name
        self.suffix = suffix
        self.value = 0.0
        self.count = 0

    def assign(self, value):
        self.value = value
        self.count = 1

    def accumulate(self, value):
        if self.count > 0:
            self.value = self.value + value
            self.count = self.count + 1
        else:
            self.assign(value)

    def merge(self, other):
        assert self.short_name == other.short_name
        self.value = self.value + other.value
        self.count = self.count + other.count

    def get(self):
        if self.count == 0:
            return self.value
        return self.value / self.count

    def reset(self):
        self.value = 0.0
        self.count = 0


class TFProcess:
    def __init__(self, cfg):
        self.cfg = cfg
        self.net = Net()
        self.root_dir = os.path.join(self.cfg['training']['path'],
                                     self.cfg['name'])

        # Network structure
        self.RESIDUAL_FILTERS = self.cfg['model']['filters']
        self.RESIDUAL_BLOCKS = self.cfg['model']['residual_blocks']
        self.SE_ratio = self.cfg['model']['se_ratio']
        self.policy_channels = self.cfg['model'].get('policy_channels', 32)
        self.pol_embedding_size = self.cfg['model'].get('pol_embedding_size', self.RESIDUAL_FILTERS)
        self.pol_encoder_layers = self.cfg['model'].get('pol_encoder_layers', 0)
        self.pol_encoder_heads = self.cfg['model'].get('pol_encoder_heads', 2)
        self.pol_encoder_d_model = self.cfg['model'].get('pol_encoder_d_model', self.RESIDUAL_FILTERS)
        self.pol_encoder_dff = self.cfg['model'].get('pol_encoder_dff', (self.RESIDUAL_FILTERS*1.5)//1)
        self.policy_d_model = self.cfg['model'].get('policy_d_model', self.RESIDUAL_FILTERS)
        self.dropout_rate = self.cfg['model'].get('dropout_rate', 0.0)
        precision = self.cfg['training'].get('precision', 'single')
        loss_scale = self.cfg['training'].get('loss_scale', 128)
        self.virtual_batch_size = self.cfg['model'].get(
            'virtual_batch_size', None)

        if precision == 'single':
            self.model_dtype = tf.float32
        elif precision == 'half':
            self.model_dtype = tf.float16
        else:
            raise ValueError("Unknown precision: {}".format(precision))

        # Scale the loss to prevent gradient underflow
        self.loss_scale = 1 if self.model_dtype == tf.float32 else loss_scale

        policy_head = self.cfg['model'].get('policy', 'convolution')
        value_head = self.cfg['model'].get('value', 'wdl')
        moves_left_head = self.cfg['model'].get('moves_left', 'v1')
        input_mode = self.cfg['model'].get('input_type', 'classic')
        default_activation = self.cfg['model'].get('default_activation', 'relu')

        self.POLICY_HEAD = None
        self.VALUE_HEAD = None
        self.MOVES_LEFT_HEAD = None
        self.INPUT_MODE = None
        self.DEFAULT_ACTIVATION = None

        if policy_head == "classical":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CLASSICAL
        elif policy_head == "convolution":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CONVOLUTION
        elif policy_head == "attention":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_ATTENTION
            if self.pol_encoder_layers > 0:
                self.net.set_pol_headcount(self.pol_encoder_heads)
        else:
            raise ValueError(
                "Unknown policy head format: {}".format(policy_head))

        self.net.set_policyformat(self.POLICY_HEAD)

        if value_head == "classical":
            self.VALUE_HEAD = pb.NetworkFormat.VALUE_CLASSICAL
            self.wdl = False
        elif value_head == "wdl":
            self.VALUE_HEAD = pb.NetworkFormat.VALUE_WDL
            self.wdl = True
        else:
            raise ValueError(
                "Unknown value head format: {}".format(value_head))

        self.net.set_valueformat(self.VALUE_HEAD)

        if moves_left_head == "none":
            self.MOVES_LEFT_HEAD = pb.NetworkFormat.MOVES_LEFT_NONE
            self.moves_left = False
        elif moves_left_head == "v1":
            self.MOVES_LEFT_HEAD = pb.NetworkFormat.MOVES_LEFT_V1
            self.moves_left = True
        else:
            raise ValueError(
                "Unknown moves left head format: {}".format(moves_left_head))

        self.net.set_movesleftformat(self.MOVES_LEFT_HEAD)

        if input_mode == "classic":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE
        elif input_mode == "frc_castling":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CASTLING_PLANE
        elif input_mode == "canonical":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION
        elif input_mode == "canonical_100":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES
        elif input_mode == "canonical_armageddon":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON
        elif input_mode == "canonical_v2":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2
        elif input_mode == "canonical_v2_armageddon":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON
        else:
            raise ValueError(
                "Unknown input mode format: {}".format(input_mode))

        self.net.set_input(self.INPUT_MODE)

        if default_activation == "relu":
            self.net.set_defaultactivation(pb.NetworkFormat.DEFAULT_ACTIVATION_RELU)
            self.DEFAULT_ACTIVATION = 'relu'
        elif default_activation == "mish":
            self.net.set_defaultactivation(pb.NetworkFormat.DEFAULT_ACTIVATION_MISH)
            self.DEFAULT_ACTIVATION = tfa.activations.mish
        else:
            raise ValueError(
                "Unknown default activation type: {}".format(default_activation))

        self.swa_enabled = self.cfg['training'].get('swa', False)

        # Limit momentum of SWA exponential average to 1 - 1/(swa_max_n + 1)
        self.swa_max_n = self.cfg['training'].get('swa_max_n', 0)

        self.renorm_enabled = self.cfg['training'].get('renorm', False)
        self.renorm_max_r = self.cfg['training'].get('renorm_max_r', 1)
        self.renorm_max_d = self.cfg['training'].get('renorm_max_d', 0)
        self.renorm_momentum = self.cfg['training'].get(
            'renorm_momentum', 0.99)

        if self.cfg['gpu'] == 'all':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.strategy = tf.distribute.MirroredStrategy()
            tf.distribute.experimental_set_strategy(self.strategy)
        elif "," in str(self.cfg['gpu']):
            active_gpus = []
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            for i in self.cfg['gpu'].split(","):
                active_gpus.append("GPU:" + i)
            self.strategy = tf.distribute.MirroredStrategy(active_gpus)
            tf.distribute.experimental_set_strategy(self.strategy)
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(gpus)
            tf.config.experimental.set_visible_devices(gpus[self.cfg['gpu']],
                                                       'GPU')
            tf.config.experimental.set_memory_growth(gpus[self.cfg['gpu']],
                                                     True)
            self.strategy = None
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(gpus)
            tf.config.experimental.set_visible_devices(gpus[self.cfg['gpu']],
                                                       'GPU')
            tf.config.experimental.set_memory_growth(gpus[self.cfg['gpu']],
                                                     True)
            self.strategy = None
        if self.model_dtype == tf.float16:
             tf.keras.mixed_precision.set_global_policy('mixed_float16')


        self.global_step = tf.Variable(0,
                                       name='global_step',
                                       trainable=False,
                                       dtype=tf.int64)

    def init(self, train_dataset, test_dataset, validation_dataset=None):
        if self.strategy is not None:
            self.train_dataset = self.strategy.experimental_distribute_dataset(
                train_dataset)
        else:
            self.train_dataset = train_dataset
        self.train_iter = iter(self.train_dataset)
        if self.strategy is not None:
            self.test_dataset = self.strategy.experimental_distribute_dataset(
                test_dataset)
        else:
            self.test_dataset = test_dataset
        self.test_iter = iter(self.test_dataset)
        if self.strategy is not None and validation_dataset is not None:
            self.validation_dataset = self.strategy.experimental_distribute_dataset(
                validation_dataset)
        else:
            self.validation_dataset = validation_dataset
        if self.strategy is not None:
            this = self
            with self.strategy.scope():
                this.init_net()
        else:
            self.init_net()

    def init_net(self):
        self.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
        input_var = tf.keras.Input(shape=(112, 8, 8))
        outputs = self.construct_net(input_var)
        self.model = tf.keras.Model(inputs=input_var, outputs=outputs)

        # swa_count initialized regardless to make checkpoint code simpler.
        self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
        self.swa_weights = None
        if self.swa_enabled:
            # Count of networks accumulated into SWA
            self.swa_weights = [
                tf.Variable(w, trainable=False) for w in self.model.weights
            ]

        self.active_lr = tf.Variable(0.01, trainable=False)
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=lambda: self.active_lr, momentum=0.9, nesterov=True)
        self.orig_optimizer = self.optimizer
        if self.loss_scale != 1:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                self.optimizer, initial_scale=self.loss_scale)
        if self.cfg['training'].get('lookahead_optimizer'):
            import tensorflow_addons as tfa
            self.optimizer = tfa.optimizers.Lookahead(self.optimizer)

        def correct_policy(target, output):
            output = tf.cast(output, tf.float32)
            # Calculate loss on policy head
            if self.cfg['training'].get('mask_legal_moves'):
                # extract mask for legal moves from target policy
                move_is_legal = tf.greater_equal(target, 0)
                # replace logits of illegal moves with large negative value (so that it doesn't affect policy of legal moves) without gradient
                illegal_filler = tf.zeros_like(output) - 1.0e10
                output = tf.where(move_is_legal, output, illegal_filler)
            # y_ still has -1 on illegal moves, flush them to 0
            target = tf.nn.relu(target)
            return target, output

        def policy_loss(target, output):
            target, output = correct_policy(target, output)
            policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(target), logits=output)
            return tf.reduce_mean(input_tensor=policy_cross_entropy)

        self.policy_loss_fn = policy_loss

        def policy_accuracy(target, output):
            target, output = correct_policy(target, output)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.policy_accuracy_fn = policy_accuracy

        def moves_left_mean_error_fn(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(tf.abs(target - output))

        self.moves_left_mean_error = moves_left_mean_error_fn

        def policy_entropy(target, output):
            target, output = correct_policy(target, output)
            softmaxed = tf.nn.softmax(output)
            return tf.math.negative(
                tf.reduce_mean(
                    tf.reduce_sum(tf.math.xlogy(softmaxed, softmaxed),
                                  axis=1)))

        self.policy_entropy_fn = policy_entropy

        def policy_uniform_loss(target, output):
            uniform = tf.where(tf.greater_equal(target, 0),
                               tf.ones_like(target), tf.zeros_like(target))
            balanced_uniform = uniform / tf.reduce_sum(
                uniform, axis=1, keepdims=True)
            target, output = correct_policy(target, output)
            policy_cross_entropy = \
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(balanced_uniform),
                                                        logits=output)
            return tf.reduce_mean(input_tensor=policy_cross_entropy)

        self.policy_uniform_loss_fn = policy_uniform_loss

        q_ratio = self.cfg['training'].get('q_ratio', 0)
        assert 0 <= q_ratio <= 1

        # Linear conversion to scalar to compute MSE with, for comparison to old values
        wdl = tf.expand_dims(tf.constant([1.0, 0.0, -1.0]), 1)

        self.qMix = lambda z, q: q * q_ratio + z * (1 - q_ratio)
        # Loss on value head
        if self.wdl:

            def value_loss(target, output):
                output = tf.cast(output, tf.float32)
                value_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.stop_gradient(target), logits=output)
                return tf.reduce_mean(input_tensor=value_cross_entropy)

            self.value_loss_fn = value_loss

            def mse_loss(target, output):
                output = tf.cast(output, tf.float32)
                scalar_z_conv = tf.matmul(tf.nn.softmax(output), wdl)
                scalar_target = tf.matmul(target, wdl)
                return tf.reduce_mean(input_tensor=tf.math.squared_difference(
                    scalar_target, scalar_z_conv))

            self.mse_loss_fn = mse_loss
        else:

            def value_loss(target, output):
                return tf.constant(0)

            self.value_loss_fn = value_loss

            def mse_loss(target, output):
                output = tf.cast(output, tf.float32)
                scalar_target = tf.matmul(target, wdl)
                return tf.reduce_mean(input_tensor=tf.math.squared_difference(
                    scalar_target, output))

            self.mse_loss_fn = mse_loss

        if self.moves_left:

            def moves_left_loss(target, output):
                # Scale the loss to similar range as other losses.
                scale = 20.0
                target = target / scale
                output = tf.cast(output, tf.float32) / scale
                if self.strategy is not None:
                    huber = tf.keras.losses.Huber(
                        10.0 / scale, reduction=tf.keras.losses.Reduction.NONE)
                else:
                    huber = tf.keras.losses.Huber(10.0 / scale)
                return tf.reduce_mean(huber(target, output))
        else:
            moves_left_loss = None

        self.moves_left_loss_fn = moves_left_loss

        pol_loss_w = self.cfg['training']['policy_loss_weight']
        val_loss_w = self.cfg['training']['value_loss_weight']

        if self.moves_left:
            moves_loss_w = self.cfg['training']['moves_left_loss_weight']
        else:
            moves_loss_w = tf.constant(0.0, dtype=tf.float32)
        reg_term_w = self.cfg['training'].get('reg_term_weight', 1.0)

        def _lossMix(policy, value, moves_left, reg_term):
            return pol_loss_w * policy + val_loss_w * value + moves_loss_w * moves_left + reg_term_w * reg_term

        self.lossMix = _lossMix

        def accuracy(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.accuracy_fn = accuracy

        # Order must match the order in process_inner_loop
        self.train_metrics = [
            Metric('P', 'Policy Loss'),
            Metric('V', 'Value Loss'),
            Metric('ML', 'Moves Left Loss'),
            Metric('Reg', 'Reg term'),
            Metric('Total', 'Total Loss'),
            Metric(
                'V MSE', 'MSE Loss'
            ),  # Long name here doesn't mention value for backwards compatibility reasons.
        ]
        self.time_start = None
        self.last_steps = None

        # Order must match the order in calculate_test_summaries_inner_loop
        self.test_metrics = [
            Metric('P', 'Policy Loss'),
            Metric('V', 'Value Loss'),
            Metric('ML', 'Moves Left Loss'),
            Metric(
                'V MSE', 'MSE Loss'
            ),  # Long name here doesn't mention value for backwards compatibility reasons.
            Metric('P Acc', 'Policy Accuracy', suffix='%'),
            Metric('V Acc', 'Value Accuracy', suffix='%'),
            Metric('ML Mean', 'Moves Left Mean Error'),
            Metric('P Entropy', 'Policy Entropy'),
            Metric('P UL', 'Policy UL'),
        ]

        # Set adaptive learning rate during training
        self.cfg['training']['lr_boundaries'].sort()
        self.warmup_steps = self.cfg['training'].get('warmup_steps', 0)
        self.lr = self.cfg['training']['lr_values'][0]
        self.test_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-test".format(self.cfg['name'])))
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-train".format(self.cfg['name'])))
        if vars(self).get('validation_dataset', None) is not None:
            self.validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-validation".format(self.cfg['name'])))
        if self.swa_enabled:
            self.swa_writer = tf.summary.create_file_writer(
                os.path.join(os.getcwd(),
                             "leelalogs/{}-swa-test".format(self.cfg['name'])))
            self.swa_validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-swa-validation".format(self.cfg['name'])))
        self.checkpoint = tf.train.Checkpoint(optimizer=self.orig_optimizer,
                                              model=self.model,
                                              global_step=self.global_step,
                                              swa_count=self.swa_count)
        self.checkpoint.listed = self.swa_weights
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.root_dir,
            max_to_keep=50,
            keep_checkpoint_every_n_hours=24,
            checkpoint_name=self.cfg['name'])

    def replace_weights(self, proto_filename, ignore_errors=False):
        self.net.parse_proto(proto_filename)

        filters, blocks = self.net.filters(), self.net.blocks()
        if not ignore_errors:
            if self.RESIDUAL_FILTERS != filters:
                raise ValueError("Number of filters doesn't match the network")
            if self.RESIDUAL_BLOCKS != blocks:
                raise ValueError("Number of blocks doesn't match the network")
            if self.POLICY_HEAD != self.net.pb.format.network_format.policy:
                raise ValueError("Policy head type doesn't match the network")
            if self.VALUE_HEAD != self.net.pb.format.network_format.value:
                raise ValueError("Value head type doesn't match the network")

        # List all tensor names we need weights for.
        names = []
        for weight in self.model.weights:
            names.append(weight.name)

        new_weights = self.net.get_weights_v2(names)
        for weight in self.model.weights:
            if 'renorm' in weight.name:
                # Renorm variables are not populated.
                continue

            try:
                new_weight = new_weights[weight.name]
            except KeyError:
                error_string = 'No values for tensor {} in protobuf'.format(
                    weight.name)
                if ignore_errors:
                    print(error_string)
                    continue
                else:
                    raise KeyError(error_string)

            if reduce(operator.mul, weight.shape.as_list(),
                      1) != len(new_weight):
                error_string = 'Tensor {} has wrong length. Tensorflow shape {}, size in protobuf {}'.format(
                    weight.name, weight.shape.as_list(), len(new_weight))
                if ignore_errors:
                    print(error_string)
                    continue
                else:
                    raise KeyError(error_string)

            if weight.shape.ndims == 4:
                # Rescale rule50 related weights as clients do not normalize the input.
                if weight.name == 'input/conv2d/kernel:0' and self.net.pb.format.network_format.input < pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
                    num_inputs = 112
                    # 50 move rule is the 110th input, or 109 starting from 0.
                    rule50_input = 109
                    for i in range(len(new_weight)):
                        if (i % (num_inputs * 9)) // 9 == rule50_input:
                            new_weight[i] = new_weight[i] * 99

                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weight.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weight, shape=shape)
                weight.assign(tf.transpose(a=new_weight, perm=[2, 3, 1, 0]))
            elif weight.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weight.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weight, shape=shape)
                weight.assign(tf.transpose(a=new_weight, perm=[1, 0]))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weight, shape=weight.shape)
                weight.assign(new_weight)
        # Replace the SWA weights as well, ensuring swa accumulation is reset.
        if self.swa_enabled:
            self.swa_count.assign(tf.constant(0.))
            self.update_swa()
        # This should result in identical file to the starting one
        # self.save_leelaz_weights('restored.pb.gz')

    def restore(self):
        if self.manager.latest_checkpoint is not None:
            print("Restoring from {0}".format(self.manager.latest_checkpoint))
            self.checkpoint.restore(self.manager.latest_checkpoint)

    def process_loop(self, batch_size, test_batches, batch_splits=1):
        if self.swa_enabled:
            # split half of test_batches between testing regular weights and SWA weights
            test_batches //= 2
        # Make sure that ghost batch norm can be applied
        if self.virtual_batch_size and batch_size % self.virtual_batch_size != 0:
            # Adjust required batch size for batch splitting.
            required_factor = self.virtual_batch_size * self.cfg[
                'training'].get('num_batch_splits', 1)
            raise ValueError(
                'batch_size must be a multiple of {}'.format(required_factor))

        # Get the initial steps value in case this is a resume from a step count
        # which is not a multiple of total_steps.
        steps = self.global_step.read_value()
        self.last_steps = steps
        self.time_start = time.time()
        self.profiling_start_step = None

        total_steps = self.cfg['training']['total_steps']
        self.calculate_test_summaries(test_batches, steps, write=False)

        for _ in range(steps % total_steps, total_steps):
            self.process(batch_size, test_batches, batch_splits=batch_splits)

    @tf.function()
    def read_weights(self):
        return [w.read_value() for w in self.model.weights]

    @tf.function()
    def process_inner_loop(self, x, y, z, q, m):
        with tf.GradientTape() as tape:
            outputs = self.model(x, training=True)
            policy = outputs[0]
            value = outputs[1]
            policy_loss = self.policy_loss_fn(y, policy)
            reg_term = sum(self.model.losses)
            if self.wdl:
                value_ce_loss = self.value_loss_fn(self.qMix(z, q), value)
                value_loss = value_ce_loss
            else:
                value_mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
                value_loss = value_mse_loss
            if self.moves_left:
                moves_left = outputs[2]
                moves_left_loss = self.moves_left_loss_fn(m, moves_left)
            else:
                moves_left_loss = tf.constant(0.)
            total_loss = self.lossMix(policy_loss, value_loss, moves_left_loss,
                                      reg_term)
            if self.loss_scale != 1:
                total_loss = self.optimizer.get_scaled_loss(total_loss)
        if self.wdl:
            mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
        else:
            value_loss = self.value_loss_fn(self.qMix(z, q), value)
        metrics = [
            policy_loss,
            value_loss,
            moves_left_loss,
            reg_term,
            total_loss,
            # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
            # get comparable values.
            mse_loss / 4.0,
        ]
        return metrics, tape.gradient(total_loss, self.model.trainable_weights)

    @tf.function()
    def strategy_process_inner_loop(self, x, y, z, q, m):
        metrics, new_grads = self.strategy.run(self.process_inner_loop,
                                               args=(x, y, z, q, m))
        metrics = [
            self.strategy.reduce(tf.distribute.ReduceOp.MEAN, m, axis=None)
            for m in metrics
        ]
        return metrics, new_grads

    def apply_grads(self, grads, effective_batch_splits):
        grads = [
            g[0] for g in self.orig_optimizer.gradient_aggregator(
                zip(grads, self.model.trainable_weights))
        ]
        if self.loss_scale != 1:
            grads = self.optimizer.get_unscaled_gradients(grads)
        max_grad_norm = self.cfg['training'].get(
            'max_grad_norm', 10000.0) * effective_batch_splits
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights),
                                       experimental_aggregate_gradients=False)
        return grad_norm

    @tf.function()
    def strategy_apply_grads(self, grads, effective_batch_splits):
        grad_norm = self.strategy.run(self.apply_grads,
                                      args=(grads, effective_batch_splits))
        grad_norm = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         grad_norm,
                                         axis=None)
        return grad_norm

    @tf.function()
    def merge_grads(self, grads, new_grads):
        return [tf.math.add(a, b) for (a, b) in zip(grads, new_grads)]

    @tf.function()
    def strategy_merge_grads(self, grads, new_grads):
        return self.strategy.run(self.merge_grads, args=(grads, new_grads))

    def train_step(self, steps, batch_size, batch_splits):
        # need to add 1 to steps because steps will be incremented after gradient update
        if (steps +
                1) % self.cfg['training']['train_avg_report_steps'] == 0 or (
                    steps + 1) % self.cfg['training']['total_steps'] == 0:
            before_weights = self.read_weights()

        # Run training for this batch

        old_weights = {}
        for layer in self.model.layers:
            if "conv" in layer.name and isinstance(layer, tf.keras.layers.Conv2D):
                old_weights[layer.name] = tf.Variable(layer.kernel)
                sparsity_mask = make_sparsity_mask(layer.kernel)
                layer.kernel.assign(layer.kernel * sparsity_mask)
            
            # calculate sparsity

        grads = None
        for _ in range(batch_splits):
            x, y, z, q, m = next(self.train_iter)
            if self.strategy is not None:
                metrics, new_grads = self.strategy_process_inner_loop(
                    x, y, z, q, m)
            else:
                metrics, new_grads = self.process_inner_loop(x, y, z, q, m)
            if not grads:
                grads = new_grads
            else:
                if self.strategy is not None:
                    grads = self.strategy_merge_grads(grads, new_grads)
                else:
                    grads = self.merge_grads(grads, new_grads)
            # Keep running averages
            for acc, val in zip(self.train_metrics, metrics):
                acc.accumulate(val)

        for layer in self.model.layers:
            if "conv" in layer.name and isinstance(layer, tf.keras.layers.Conv2D):
                layer.kernel.assign(old_weights[layer.name])

        # Gradients of batch splits are summed, not averaged like usual, so need to scale lr accordingly to correct for this.
        effective_batch_splits = batch_splits
        if self.strategy is not None:
            effective_batch_splits = batch_splits * self.strategy.num_replicas_in_sync
        self.active_lr.assign(self.lr / effective_batch_splits)
        if self.strategy is not None:
            grad_norm = self.strategy_apply_grads(grads,
                                                  effective_batch_splits)
        else:
            grad_norm = self.apply_grads(grads, effective_batch_splits)

        # Note: grads variable at this point has not been unscaled or
        # had clipping applied. Since no code after this point depends
        # upon that it seems fine for now.



        # Update steps.
        self.global_step.assign_add(1)
        steps = self.global_step.read_value()

        if steps % self.cfg['training'][
                'train_avg_report_steps'] == 0 or steps % self.cfg['training'][
                    'total_steps'] == 0:
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                steps_elapsed = steps - self.last_steps
                speed = batch_size * (tf.cast(steps_elapsed, tf.float32) /
                                      elapsed)
            print("step {}, lr={:g}".format(steps, self.lr), end='')
            for metric in self.train_metrics:
                print(" {}={:g}{}".format(metric.short_name, metric.get(),
                                          metric.suffix),
                      end='')
            print(" ({:g} pos/s)".format(speed))

            after_weights = self.read_weights()
            with self.train_writer.as_default():
                for metric in self.train_metrics:
                    tf.summary.scalar(metric.long_name,
                                      metric.get(),
                                      step=steps)
                tf.summary.scalar("LR", self.lr, step=steps)
                tf.summary.scalar("Gradient norm",
                                  grad_norm / effective_batch_splits,
                                  step=steps)
                self.compute_update_ratio(before_weights, after_weights, steps)
            self.train_writer.flush()

            self.time_start = time_end
            self.last_steps = steps
            for metric in self.train_metrics:
                metric.reset()
        return steps

    def process(self, batch_size, test_batches, batch_splits):
        # Get the initial steps value before we do a training step.
        steps = self.global_step.read_value()

        # By default disabled since 0 != 10.
        if steps % self.cfg['training'].get('profile_step_freq',
                                            1) == self.cfg['training'].get(
                                                'profile_step_offset', 10):
            self.profiling_start_step = steps
            tf.profiler.experimental.start(
                os.path.join(os.getcwd(),
                             "leelalogs/{}-profile".format(self.cfg['name'])))

        # Run test before first step to see delta since end of last run.
        if steps % self.cfg['training']['total_steps'] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps + 1):
                # Steps is given as one higher than current in order to avoid it
                # being equal to the value the end of a run is stored against.
                self.calculate_test_summaries(test_batches, steps + 1)
                if self.swa_enabled:
                    self.calculate_swa_summaries(test_batches, steps + 1)

        # Determine learning rate
        lr_values = self.cfg['training']['lr_values']
        lr_boundaries = self.cfg['training']['lr_boundaries']
        steps_total = steps % self.cfg['training']['total_steps']
        self.lr = lr_values[bisect.bisect_right(lr_boundaries, steps_total)]
        if self.warmup_steps > 0 and steps < self.warmup_steps:
            self.lr = self.lr * tf.cast(steps + 1,
                                        tf.float32) / self.warmup_steps

        with tf.profiler.experimental.Trace("Train", step_num=steps):
            steps = self.train_step(steps, batch_size, batch_splits)

        if self.swa_enabled and steps % self.cfg['training']['swa_steps'] == 0:
            self.update_swa()

        # Calculate test values every 'test_steps', but also ensure there is
        # one at the final step so the delta to the first step can be calculated.
        if steps % self.cfg['training']['test_steps'] == 0 or steps % self.cfg[
                'training']['total_steps'] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps):
                self.calculate_test_summaries(test_batches, steps)
                if self.swa_enabled:
                    self.calculate_swa_summaries(test_batches, steps)

        if self.validation_dataset is not None and (
                steps % self.cfg['training']['validation_steps'] == 0
                or steps % self.cfg['training']['total_steps'] == 0):
            with tf.profiler.experimental.Trace("Validate", step_num=steps):
                if self.swa_enabled:
                    self.calculate_swa_validations(steps)
                else:
                    self.calculate_test_validations(steps)

        # Save session and weights at end, and also optionally every 'checkpoint_steps'.
        if steps % self.cfg['training']['total_steps'] == 0 or (
                'checkpoint_steps' in self.cfg['training']
                and steps % self.cfg['training']['checkpoint_steps'] == 0):
            evaled_steps = steps.numpy()
            self.manager.save(checkpoint_number=evaled_steps)
            print("Model saved in file: {}".format(
                self.manager.latest_checkpoint))
            path = os.path.join(self.root_dir, self.cfg['name'])
            leela_path = path + "-" + str(evaled_steps)
            swa_path = path + "-swa-" + str(evaled_steps)
            self.net.pb.training_params.training_steps = evaled_steps
            self.save_leelaz_weights(leela_path)
            if self.swa_enabled:
                self.save_swa_weights(swa_path)

        if self.profiling_start_step is not None and (
                steps >= self.profiling_start_step +
                self.cfg['training'].get('profile_step_count', 0)
                or steps % self.cfg['training']['total_steps'] == 0):
            tf.profiler.experimental.stop()
            self.profiling_start_step = None

    def calculate_swa_summaries(self, test_batches, steps):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_test_writer, self.test_writer = self.test_writer, self.swa_writer
        print('swa', end=' ')
        self.calculate_test_summaries(test_batches, steps)
        self.test_writer = true_test_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    @tf.function()
    def calculate_test_summaries_inner_loop(self, x, y, z, q, m):
        outputs = self.model(x, training=False)
        policy = outputs[0]
        value = outputs[1]
        policy_loss = self.policy_loss_fn(y, policy)
        policy_accuracy = self.policy_accuracy_fn(y, policy)
        policy_entropy = self.policy_entropy_fn(y, policy)
        policy_ul = self.policy_uniform_loss_fn(y, policy)
        if self.wdl:
            value_loss = self.value_loss_fn(self.qMix(z, q), value)
            mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
            value_accuracy = self.accuracy_fn(self.qMix(z, q), value)
        else:
            value_loss = self.value_loss_fn(self.qMix(z, q), value)
            mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
            value_accuracy = tf.constant(0.)
        if self.moves_left:
            moves_left = outputs[2]
            moves_left_loss = self.moves_left_loss_fn(m, moves_left)
            moves_left_mean_error = self.moves_left_mean_error(m, moves_left)
        else:
            moves_left_loss = tf.constant(0.)
            moves_left_mean_error = tf.constant(0.)
        metrics = [
            policy_loss,
            value_loss,
            moves_left_loss,
            mse_loss / 4,
            policy_accuracy * 100,
            value_accuracy * 100,
            moves_left_mean_error,
            policy_entropy,
            policy_ul,
        ]
        return metrics

    @tf.function()
    def strategy_calculate_test_summaries_inner_loop(self, x, y, z, q, m):
        metrics = self.strategy.run(self.calculate_test_summaries_inner_loop,
                                    args=(x, y, z, q, m))
        metrics = [
            self.strategy.reduce(tf.distribute.ReduceOp.MEAN, m, axis=None)
            for m in metrics
        ]
        return metrics

    def calculate_test_summaries(self, test_batches, steps, write=True):
        for metric in self.test_metrics:
            metric.reset()
        for _ in range(0, test_batches):
            x, y, z, q, m = next(self.test_iter)
            if self.strategy is not None:
                metrics = self.strategy_calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            else:
                metrics = self.calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            for acc, val in zip(self.test_metrics, metrics):
                acc.accumulate(val)
        self.net.pb.training_params.learning_rate = self.lr
        self.net.pb.training_params.mse_loss = self.test_metrics[3].get()
        self.net.pb.training_params.policy_loss = self.test_metrics[0].get()
        # TODO store value and value accuracy in pb
        self.net.pb.training_params.accuracy = self.test_metrics[4].get()
        if write:
            with self.test_writer.as_default():
                for metric in self.test_metrics:
                    tf.summary.scalar(metric.long_name, metric.get(), step=steps)
                for w in self.model.weights:
                    tf.summary.histogram(w.name, w, step=steps)
            self.test_writer.flush()

        print("step {},".format(steps), end='')
        for metric in self.test_metrics:
            print(" {}={:g}{}".format(metric.short_name, metric.get(),
                                      metric.suffix),
                  end='')
        print()

    def calculate_swa_validations(self, steps):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_validation_writer, self.validation_writer = self.validation_writer, self.swa_validation_writer
        print('swa', end=' ')
        self.calculate_test_validations(steps)
        self.validation_writer = true_validation_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def calculate_test_validations(self, steps):
        for metric in self.test_metrics:
            metric.reset()
        for (x, y, z, q, m) in self.validation_dataset:
            if self.strategy is not None:
                metrics = self.strategy_calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            else:
                metrics = self.calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            for acc, val in zip(self.test_metrics, metrics):
                acc.accumulate(val)
        with self.validation_writer.as_default():
            for metric in self.test_metrics:
                tf.summary.scalar(metric.long_name, metric.get(), step=steps)
        self.validation_writer.flush()

        print("step {}, validation:".format(steps), end='')
        for metric in self.test_metrics:
            print(" {}={:g}{}".format(metric.short_name, metric.get(),
                                      metric.suffix),
                  end='')
        print()

    @tf.function()
    def compute_update_ratio(self, before_weights, after_weights, steps):
        """Compute the ratio of gradient norm to weight norm.

        Adapted from https://github.com/tensorflow/minigo/blob/c923cd5b11f7d417c9541ad61414bf175a84dc31/dual_net.py#L567
        """
        deltas = [
            after - before
            for after, before in zip(after_weights, before_weights)
        ]
        delta_norms = [tf.math.reduce_euclidean_norm(d) for d in deltas]
        weight_norms = [
            tf.math.reduce_euclidean_norm(w) for w in before_weights
        ]
        ratios = [(tensor.name, tf.cond(w != 0., lambda: d / w, lambda: -1.))
                  for d, w, tensor in zip(delta_norms, weight_norms,
                                          self.model.weights)
                  if not 'moving' in tensor.name]
        for name, ratio in ratios:
            tf.summary.scalar('update_ratios/' + name, ratio, step=steps)
        # Filtering is hard, so just push infinities/NaNs to an unreasonably large value.
        ratios = [
            tf.cond(r > 0, lambda: tf.math.log(r) / 2.30258509299,
                    lambda: 200.) for (_, r) in ratios
        ]
        tf.summary.histogram('update_ratios_log10',
                             tf.stack(ratios),
                             buckets=1000,
                             step=steps)

    def update_swa(self):
        num = self.swa_count.read_value()
        for (w, swa) in zip(self.model.weights, self.swa_weights):
            swa.assign(swa.read_value() * (num / (num + 1.)) + w.read_value() *
                       (1. / (num + 1.)))
        self.swa_count.assign(min(num + 1., self.swa_max_n))

    def save_swa_weights(self, filename):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        self.save_leelaz_weights(filename)
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def save_leelaz_weights(self, filename):
        numpy_weights = []
        for weight in self.model.weights:
            numpy_weights.append([weight.name, weight.numpy()])
        self.net.fill_net_v2(numpy_weights)
        self.net.save_proto(filename)

    def batch_norm(self, input, name, scale=False):
        if self.renorm_enabled:
            clipping = {
                "rmin": 1.0 / self.renorm_max_r,
                "rmax": self.renorm_max_r,
                "dmax": self.renorm_max_d
            }
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                axis=1,
                fused=False,
                center=True,
                scale=scale,
                renorm=True,
                renorm_clipping=clipping,
                renorm_momentum=self.renorm_momentum,
                name=name)(input)
        else:
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                axis=1,
                center=True,
                scale=scale,
                virtual_batch_size=self.virtual_batch_size,
                name=name)(input)

    def squeeze_excitation(self, inputs, channels, name):
        assert channels % self.SE_ratio == 0

        pooled = tf.keras.layers.GlobalAveragePooling2D(
            data_format='channels_first')(inputs)
        squeezed = tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(tf.keras.layers.Dense(
            channels // self.SE_ratio,
            kernel_initializer='glorot_normal',
            kernel_regularizer=self.l2reg,
            name=name + '/se/dense1')(pooled))
        excited = tf.keras.layers.Dense(2 * channels,
                                        kernel_initializer='glorot_normal',
                                        kernel_regularizer=self.l2reg,
                                        name=name + '/se/dense2')(squeezed)
        return ApplySqueezeExcitation()([inputs, excited])

    def conv_block(self,
                   inputs,
                   filter_size,
                   output_channels,
                   name,
                   bn_scale=False):
        conv = tf.keras.layers.Conv2D(output_channels,
                                      filter_size,
                                      use_bias=False,
                                      padding='same',
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=self.l2reg,
                                      data_format='channels_first',
                                      name=name + '/conv2d')(inputs)
        return tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(self.batch_norm(
            conv, name=name + '/bn', scale=bn_scale))

    def residual_block(self, inputs, channels, name):
        min_value = -0.31 if self.DEFAULT_ACTIVATION == tfa.activations.mish else 0.0
        
        quantize = False
        if quantize:
            conv1 = quantized_conv(channels // 2, use_bias=False,
                                       kernel_initializer='glorot_normal',
                                       regularizer=self.l2reg,
                                       data_format='channels_first',
                                       name=name + '/1', min_value=min_value)(inputs)
        else:
            conv1 = tf.keras.layers.Conv2D(channels // 2,
                                        3,
                                        use_bias=False,
                                        padding='same',
                                        kernel_initializer='glorot_normal',
                                        kernel_regularizer=self.l2reg,
                                        data_format='channels_first',
                                        name=name + '/1/conv2d')(inputs)


        out1 = tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(self.batch_norm(conv1,
                                                                  name +
                                                                  '/1/bn',
                                                                  scale=False))
        if quantize:
            conv2 = quantized_conv(channels, use_bias=False,
                                        kernel_initializer='glorot_normal',
                                        regularizer=self.l2reg,
                                        data_format='channels_first',
                                        name=name + '/2', min_value=min_value)(out1)
        else:
            conv2 = tf.keras.layers.Conv2D(channels,
                                        3,
                                        use_bias=False,
                                        padding='same',
                                        kernel_initializer='glorot_normal',
                                        kernel_regularizer=self.l2reg,
                                        data_format='channels_first',
                                        name=name + '/2/conv2d')(out1)
        out2 = self.batch_norm(conv2,
                                                       name + '/2/bn',
                                                       scale=True)
        if True:
            out2 = self.squeeze_excitation(out2,
                                        channels,
                                        name=name + '/se')
        return tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(tf.keras.layers.add(
            [inputs, out2]))

    @staticmethod
    def split_heads(inputs, batch_size, num_heads, depth):
        if num_heads < 2:
            return inputs
        reshaped = tf.reshape(inputs, (batch_size, 64, num_heads, depth))
        return tf.transpose(reshaped, perm=[0, 2, 1, 3])  # (batch_size, num_heads, 64, depth)

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], self.model_dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, scaled_attention_logits

    # multi-head attention in encoder blocks
    def mha(self, inputs, emb_size, d_model, num_heads, name):
        assert d_model % num_heads == 0
        depth = d_model // num_heads
        # query, key, and value vectors for self-attention
        q = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_normal', name=name + '/wq')(inputs)
        k = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_normal', name=name + '/wk')(inputs)
        v = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_normal', name=name + '/wv')(inputs)
        # split q, k and v into smaller vectors of size 'depth' -- one for each head in multi-head attention
        batch_size = tf.shape(q)[0]
        q = self.split_heads(q, batch_size, num_heads, depth)
        k = self.split_heads(k, batch_size, num_heads, depth)
        v = self.split_heads(v, batch_size, num_heads, depth)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        if num_heads > 1:
            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            scaled_attention = tf.reshape(scaled_attention, (batch_size, -1, d_model))  # concatenate heads
        # final dense layer
        output = tf.keras.layers.Dense(emb_size, kernel_initializer='glorot_normal',
                                       name=name + "/dense")(scaled_attention)
        return output, attention_weights

    # 2-layer dense feed-forward network in encoder blocks
    def ffn(self, inputs, emb_size, dff, name):
        dense1 = tf.keras.layers.Dense(dff, kernel_initializer='glorot_normal', activation='selu',
                                       name=name + "/dense1")(inputs)
        return tf.keras.layers.Dense(emb_size, kernel_initializer='glorot_normal', name=name + "/dense2")(dense1)

    def encoder_layer(self, inputs, emb_size, d_model, num_heads, dff, name, training):
        attn_output, attn_wts = self.mha(inputs, emb_size, d_model, num_heads, name=name + "/mha")
        # dropout for weight regularization
        attn_output = tf.keras.layers.Dropout(self.dropout_rate, name=name + "/dropout1")(attn_output, training=training)
        # skip connection + layernorm
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=name + "/ln1")(inputs + attn_output)
        # feed-forward network
        ffn_output = self.ffn(out1, emb_size, dff, name=name + "/ffn")
        ffn_output = tf.keras.layers.Dropout(self.dropout_rate, name=name + "/dropout2")(ffn_output, training=training)
        out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=name + "/ln2")(out1 + ffn_output)
        return out2, attn_wts

    def construct_net(self, inputs):
        print("Constructing network...")
        flow = self.conv_block(inputs,
                               filter_size=3,
                               output_channels=self.RESIDUAL_FILTERS,
                               name='input',
                               bn_scale=True)
        for i in range(self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow,
                                       self.RESIDUAL_FILTERS,
                                       name='residual_{}'.format(i + 1))

        # Policy head
        if self.POLICY_HEAD == pb.NetworkFormat.POLICY_CONVOLUTION:
            conv_pol = self.conv_block(flow,
                                       filter_size=3,
                                       output_channels=self.RESIDUAL_FILTERS,
                                       name='policy1')
            conv_pol2 = tf.keras.layers.Conv2D(
                80,
                3,
                use_bias=True,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                bias_regularizer=self.l2reg,
                data_format='channels_first',
                name='policy')(conv_pol)
            h_fc1 = ApplyPolicyMap()(conv_pol2)
        elif self.POLICY_HEAD == pb.NetworkFormat.POLICY_CLASSICAL:
            conv_pol = self.conv_block(flow,
                                       filter_size=1,
                                       output_channels=self.policy_channels,
                                       name='policy')
            h_conv_pol_flat = tf.keras.layers.Flatten()(conv_pol)
            h_fc1 = tf.keras.layers.Dense(1858,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          bias_regularizer=self.l2reg,
                                          name='policy/dense')(h_conv_pol_flat)
        elif self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION:
            # transpose and reshape
            tokens = tf.transpose(flow, perm=[0, 2, 3, 1])
            tokens = tf.reshape(tokens, [-1, 64, self.RESIDUAL_FILTERS])

            # SQUARE EMBEDDING: found to increase attention head performance
            tokens = tf.keras.layers.Dense(self.pol_embedding_size, kernel_initializer='glorot_normal',
                                           kernel_regularizer=self.l2reg, activation='selu',
                                           name='policy/embedding')(tokens)

            # ENCODER LAYERS: intermediate layers of self-attention with residual connections
            attn_wts = []
            for i in range(self.pol_encoder_layers):
                tokens, attn_wts_l = self.encoder_layer(tokens, self.pol_embedding_size, self.pol_encoder_d_model,
                                                        self.pol_encoder_heads, self.pol_encoder_dff,
                                                        name='policy/enc_layer_{}'.format(i + 1), training=True
                                                        )
                attn_wts.append(attn_wts_l)

            # create queries and keys for policy self-attention
            queries = tf.keras.layers.Dense(self.policy_d_model, kernel_initializer='glorot_normal',
                                            name='policy/attention/wq')(tokens)
            keys = tf.keras.layers.Dense(self.policy_d_model, kernel_initializer='glorot_normal',
                                         name='policy/attention/wk')(tokens)

            # PAWN PROMOTION: create promotion logits using scalar offsets generated from the promotion-rank keys
            dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1], self.model_dtype))  # constant for scaling
            promotion_keys = keys[:, -8:, :]
            # queen, rook, bishop, knight order
            promotion_offsets = tf.keras.layers.Dense(4, kernel_initializer='glorot_normal',
                                                      name='policy/attention/ppo', use_bias=False)(promotion_keys)
            promotion_offsets = tf.transpose(promotion_offsets, perm=[0, 2, 1]) * dk  # Bx4x8
            # knight offset is added to the other three
            promotion_offsets = promotion_offsets[:, :3, :] + promotion_offsets[:, 3:4, :]

            # POLICY SELF-ATTENTION: self-attention weights are interpreted as from->to policy
            matmul_qk = tf.matmul(queries, keys, transpose_b=True)  # Bx64x64 (from 64 queries, 64 keys)

            # q, r, and b promotions are offset from the default promotion logit (knight)
            n_promo_logits = matmul_qk[:, -16:-8, -8:]  # default traversals from penultimate rank to promotion rank
            q_promo_logits = tf.expand_dims(n_promo_logits + promotion_offsets[:, 0:1, :], axis=3)  # Bx8x8x1
            r_promo_logits = tf.expand_dims(n_promo_logits + promotion_offsets[:, 1:2, :], axis=3)
            b_promo_logits = tf.expand_dims(n_promo_logits + promotion_offsets[:, 2:3, :], axis=3)
            promotion_logits = tf.concat([q_promo_logits, r_promo_logits, b_promo_logits], axis=3)  # Bx8x8x3
            promotion_logits = tf.reshape(promotion_logits, [-1, 8, 24])  # logits now alternate a7a8q,a7a8r,a7a8b,...,

            # scale the logits by dividing them by sqrt(d_model) to stabilize gradients
            promotion_logits = promotion_logits / dk  # Bx8x24 (8 from-squares, 3x8 promotions)
            policy_attn_logits = matmul_qk / dk       # Bx64x64 (64 from-squares, 64 to-squares)

            attn_wts.append(promotion_logits)
            attn_wts.append(policy_attn_logits)

            # APPLY POLICY MAP: output becomes Bx1856
            h_fc1 = ApplyAttentionPolicyMap()(policy_attn_logits, promotion_logits)
        else:
            raise ValueError("Unknown policy head type {}".format(
                self.POLICY_HEAD))

        # Value head
        conv_val = self.conv_block(flow,
                                   filter_size=1,
                                   output_channels=32,
                                   name='value')
        h_conv_val_flat = tf.keras.layers.Flatten()(conv_val)
        h_fc2 = tf.keras.layers.Dense(128,
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=self.l2reg,
                                      activation=self.DEFAULT_ACTIVATION,
                                      name='value/dense1')(h_conv_val_flat)
        if self.wdl:
            h_fc3 = tf.keras.layers.Dense(3,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          bias_regularizer=self.l2reg,
                                          name='value/dense2')(h_fc2)
        else:
            h_fc3 = tf.keras.layers.Dense(1,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          activation='tanh',
                                          name='value/dense2')(h_fc2)

        # Moves left head
        if self.moves_left:
            conv_mov = self.conv_block(flow,
                                       filter_size=1,
                                       output_channels=8,
                                       name='moves_left')
            h_conv_mov_flat = tf.keras.layers.Flatten()(conv_mov)
            h_fc4 = tf.keras.layers.Dense(
                128,
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                activation=self.DEFAULT_ACTIVATION,
                name='moves_left/dense1')(h_conv_mov_flat)

            h_fc5 = tf.keras.layers.Dense(1,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          activation='relu',
                                          name='moves_left/dense2')(h_fc4)
        else:
            h_fc5 = None

        # attention weights added as optional output for analysis -- ignored by backend
        if self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION:
            if self.moves_left:
                outputs = [h_fc1, h_fc3, h_fc5, attn_wts]
            else:
                outputs = [h_fc1, h_fc3, attn_wts]
        elif self.moves_left:
            outputs = [h_fc1, h_fc3, h_fc5]
        else:
            outputs = [h_fc1, h_fc3]

        return outputs
