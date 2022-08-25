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
import tensorflow as tf
import time
import bisect
import lc0_az_policy_map
import attention_policy_map as apm
import proto.net_pb2 as pb
from functools import reduce
import operator
from net import Net
from math import sqrt


class VariationalGen(tf.keras.layers.Layer):
    def __init__(self, out_size: int, name: str = None):
        super().__init__(name=name)
        self.out_size = out_size
        self.values_gen = tf.keras.layers.Dense(out_size)
        self.eps_gen = tf.keras.layers.Dense(out_size)

    def call(self, x, training=None):
        values = self.values_gen(x)
        if training:
            eps = self.eps_gen(x)
            offsets = eps * tf.random.normal(eps.shape, dtype=eps.dtype)
            out = values + offsets
        else:
            out = values
        return out


def dcd(x, squeezed, out_channels: int, hidden_sz: int = 16, name: str = None, use_bias: bool = False, **kwargs):
    assert name is not None
    kernel = tf.keras.layers.Dense(
        hidden_sz ** 2, name=name+'/kernel_gen')(squeezed)
    kernel = tf.keras.layers.Reshape([hidden_sz, hidden_sz])(kernel)
    compressed = tf.keras.layers.Dense(
        hidden_sz, name=name+'/compress_dense')(x)
    processed = compressed @ kernel
    processed = tf.keras.layers.LayerNormalization(name=name+'/processed_ln')(processed)
    decompressed = tf.keras.layers.Dense(
        out_channels, name=name+'/decompress_dense')(processed)
    return decompressed
    #return tf.keras.layers.LayerNormalization(name=name+'/layer_norm')(decompressed)


class Gating(tf.keras.layers.Layer):
    def __init__(self, name=None, additive=True, init_value=None, **kwargs):
        self.additive = additive
        if init_value is None:
            init_value = 0 if self.additive else 1
        self.init_value = init_value
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.gate = self.add_weight(name='gate',
                                    shape=input_shape[1:],
                                    initializer=tf.constant_initializer(
                                        self.init_value),
                                    trainable=True)

    def call(self, inputs):
        return tf.add(inputs, self.gate) if self.additive else tf.multiply(inputs, self.gate)


def ma_gating(inputs, name):
    out = Gating(name=name+'/mult_gate', additive=False)(inputs)
    out = Gating(name=name+'/add_gate', additive=True)(out)
    return out


class DyDense(tf.keras.layers.Layer):

    def __init__(self, out_channels: int, n_kernels: int, name=None, per_channel=False, use_bias=False, full_bias=False, kernel_initializer='glorot_normal', activation=None, **kwargs):
        assert name is not None
        super().__init__(name=name, **kwargs)
        self.per_channel = per_channel
        self.use_bias = use_bias
        self.full_bias = full_bias
        self.out_channels = out_channels
        self.n_kernels = n_kernels
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.in_channels = input_shape[-1]

        if self.per_channel:
            self.per_channel_reshape = tf.keras.layers.Reshape(
                [self.n_kernels, self.out_channels])

        kernel_initializer = self.kernel_initializer
        if isinstance(kernel_initializer, str):
            kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        kernels = [kernel_initializer(shape=[self.in_channels, self.out_channels])
                   for _ in range(self.n_kernels)]
        kernels = tf.concat(kernels, axis=0)
        kernels = kernels.numpy()

        self.kernels = self.add_weight(name=self.name+'/kernels',
                                       shape=[self.n_kernels,
                                              self.in_channels, self.out_channels],
                                       initializer=tf.constant_initializer(
                                           kernels),
                                       trainable=True)
        if self.use_bias:
            if self.full_bias:
                self.bias = self.add_weight(name=self.name+'/bias',
                                            shape=[self.n_kernels,
                                                   64, self.out_channels],
                                            initializer='zeros',
                                            trainable=True)
            else:
                self.bias = self.add_weight(name=self.name+'/bias',
                                            shape=[self.n_kernels,
                                                   self.channels],
                                            initializer='zeros',
                                            trainable=True)
        else:
            self.bias = None

        self.temperature = self.add_weight(
            trainable=False, initializer=tf.constant_initializer(30.0), name=self.name+'/temperature')

    def call(self, inputs, attention):
        if self.per_channel:
            attention = self.per_channel_reshape(attention)
        attention = tf.nn.softmax(
            attention / tf.cast(self.temperature, inputs.dtype), axis=1)

        if self.per_channel:
            kernel = tf.einsum('bkj, kij->bij', attention, self.kernels)
        else:
            kernels = tf.reshape(
                self.kernels, [self.n_kernels, self.in_channels * self.out_channels])
            kernel = attention @ kernels
            # now has batch dim at start
            kernel = tf.reshape(
                kernel, [-1, self.in_channels, self.out_channels])

        out = inputs @ kernel
        if self.use_bias:
            if self.full_bias:
                bias_einsum = 'bkj, krj->brj' if self.per_channel else 'bk, krj->brj'

                bias = tf.einsum(bias_einsum, attention, self.bias)
            else:
                bias_einsum = 'bkj, kj->bj' if self.per_channel else 'bk, kj->bj'
                if self.use_bias:
                    bias = tf.einsum(bias_einsum, attention, self.bias)
                    bias = tf.expand_dims(bias, 1)

            out = out + bias
        return out


def dydense(inputs, squeezed, out_channels, n_kernels, name=None, activation=None, **kwargs):
    attention_sz = n_kernels * out_channels if kwargs.get('per_channel') else n_kernels
    attention = tf.keras.layers.Dense(attention_sz,
                                      name=name+'/attention_dense')(squeezed)
    activation = tf.keras.activations.get(
        activation) if isinstance(activation, str) else activation
    out = DyDense(out_channels, n_kernels,
                  name=name, **kwargs)(inputs, attention)
    tf.keras.layers.LayerNormalization(name=name+'/ln')(out)
    if activation is not None:
        out = activation(out)
    return out


def apply_dytalking_heads(X, weights, out_heads: int = None):
    in_heads = weights.shape[-1]
    out_heads = in_heads if out_heads is None else out_heads
    channels = X.shape[-1]
    kernel_init = tf.keras.initializers.RandomNormal(stddev = 0.1 / sqrt(in_heads * channels))
    bias_init = tf.keras.initializers.RandomNormal(stddev = 2 / sqrt(in_heads + out_heads))
    kernel = tf.keras.layers.Dense(in_heads * out_heads, kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    kernel = tf.reshape(kernel, [-1, 64, in_heads, out_heads])
    return weights @ kernel


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
        self.embedding_size = self.cfg['model']['embedding_size']
        self.input_gate = self.cfg['model'].get('input_gate')
        self.pol_embedding_size = self.cfg['model'].get(
            'policy_embedding_size', self.embedding_size)
        self.val_embedding_size = self.cfg['model'].get(
            'value_embedding_size', 32)
        self.mov_embedding_size = self.cfg['model'].get(
            'moves_left_embedding_size', 8)
        self.encoder_layers = self.cfg['model']['encoder_layers']
        self.encoder_heads = self.cfg['model']['encoder_heads']
        self.encoder_d_model = self.cfg['model']['encoder_d_model']
        self.encoder_dff = self.cfg['model'].get(
            'encoder_dff', (self.embedding_size*1.5)//1)
        self.policy_d_model = self.cfg['model'].get(
            'policy_d_model', self.embedding_size)
        self.dropout_rate = self.cfg['model'].get('dropout_rate', 0.0)
        self.arc_encoding = self.cfg['model'].get('arc_encoding', False)

        self.logit_gate = self.cfg['model'].get('logit_gate', 0)
        self.use_talking_heads = self.cfg['model'].get('talking_heads', False)
        dydense_usage = self.cfg['model'].get('dydense_usage', '')
        assert isinstance(
            dydense_usage, str), f'dydense_usage must be a string but is {dydense_usage}'
        self.dydense_usage = dydense_usage.lower()
        self.dydense_kernels = self.cfg['model'].get('dydense_kernels', None)
        self.dydense_pc = self.cfg['model'].get('dydense_pc', False)
        self.dydense_temp_start = self.cfg['model'].get(
            'dydense_temp_start', 30)
        self.dydense_temp_anneal_steps = self.cfg['model'].get(
            'dydense_temp_anneal_steps', 100_000)
        
        self.use_dyrelu = self.cfg['model'].get('use_dyrelu', False)
        self.use_dcd = self.cfg['model'].get('use_dcd', False)
        self.dcd_spec = self.cfg['model'].get('dcd_spec', '')
        self.dcd_size = self.cfg['model'].get('dcd_size', 32)

        self.dytalking_heads = self.cfg['model'].get('dytalking_heads', False)

        precision = self.cfg['training'].get('precision', 'single')
        loss_scale = self.cfg['training'].get('loss_scale', 128)
        self.virtual_batch_size = self.cfg['model'].get(
            'virtual_batch_size', None)

        self.use_fullgen = self.cfg['model'].get('use_fullgen', False)
        self.use_pergen = self.cfg['model'].get('use_pergen', False)
        self.fullgen_hidden_channels = self.cfg['model'].get(
            'fullgen_hidden_channels')
        self.fullgen_big_hidden_sz = self.cfg['model'].get(
            'fullgen_big_hidden_sz')
        self.fullgen_activation = self.cfg['model'].get('fullgen_activation')
        self.fullgen_hidden_sz = self.cfg['model'].get('fullgen_hidden_sz')
        self.fullgen_out_maps = self.cfg['model'].get('fullgen_out_maps')
        self.fullgen_history = self.cfg['model'].get('fullgen_history', 1)
        self.logit_history = self.cfg['model'].get('logit_history', 1)


        if precision == 'single':
            self.model_dtype = tf.float32
        elif precision == 'half':
            self.model_dtype = tf.float16
        else:
            raise ValueError("Unknown precision: {}".format(precision))

        # Scale the loss to prevent gradient underflow
        self.loss_scale = 1 if self.model_dtype == tf.float32 else loss_scale

        policy_head = self.cfg['model'].get('policy', 'attention')
        value_head = self.cfg['model'].get('value', 'wdl')
        moves_left_head = self.cfg['model'].get('moves_left', 'v1')
        input_mode = self.cfg['model'].get('input_type', 'classic')
        default_activation = self.cfg['model'].get(
            'default_activation', 'relu')

        self.POLICY_HEAD = None
        self.VALUE_HEAD = None
        self.MOVES_LEFT_HEAD = None
        self.INPUT_MODE = None
        self.DEFAULT_ACTIVATION = None

        if policy_head == "attention":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_ATTENTION
            if self.encoder_layers > 0:
                self.net.set_headcount(self.encoder_heads)
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
            self.net.set_defaultactivation(
                pb.NetworkFormat.DEFAULT_ACTIVATION_RELU)
            self.DEFAULT_ACTIVATION = 'relu'
        elif default_activation == "mish":
            self.net.set_defaultactivation(
                pb.NetworkFormat.DEFAULT_ACTIVATION_MISH)
            import tensorflow_addons as tfa
            self.DEFAULT_ACTIVATION = tfa.activations.mish
        elif default_activation == "gelu":
            self.DEFAULT_ACTIVATION = 'gelu'
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
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(gpus)
            tf.config.experimental.set_visible_devices(gpus[self.cfg['gpu']],
                                                       'GPU')
            tf.config.experimental.set_memory_growth(gpus[self.cfg['gpu']],
                                                     True)
            self.strategy = None

        if self.model_dtype == tf.float16:
            tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

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
        # !!!
        self.l2reg = tf.keras.regularizers.l2(
            l=0.5 * (0.0001))  # originally .5
        if self.use_fullgen:
            self.fullgen_scale = tf.Variable(0.0, dtype=self.model_dtype)
        input_var = tf.keras.Input(shape=(112, 8, 8))

        outputs = self.construct_net(input_var)
        self.model = tf.keras.Model(inputs=input_var, outputs=outputs)
        outputs_wlh_fixed = [outputs[0], tf.nn.softmax(
            outputs[1], name='wlh_softmax'), *(outputs[2:])]
        self.model_to_save = tf.keras.Model(
            inputs=input_var, outputs=outputs_wlh_fixed)
        print("model parameters:", self.model.count_params())

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
                self.optimizer)
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
            Metric('P Acc', 'Policy Accuracy', suffix='%'),
            Metric('V Acc', 'Value Accuracy', suffix='%'),
            Metric('P Entropy', 'Policy Entropy'),
            Metric('P UL', 'Policy UL'),
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

    def replace_weights(self, proto_filename: str, ignore_errors: bool = False):
        self.net.parse_proto(proto_filename)

        filters, blocks = self.net.filters(), self.net.blocks()
        if not ignore_errors:
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

    def process_loop(self, batch_size: int, test_batches: int, batch_splits: int = 1):
        if self.swa_enabled:
            # split half of test_batches between testing regular weights and SWA weights
            test_batches //= 2
        # Make sure that ghost batch norm can be applied
        if self.virtual_batch_size and batch_size % self.virtual_batch_size != 0:
            # Adjust required batch size for batch splitting.
            required_factor = self.virtual_batch_sizes * self.cfg[
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

        def loop():
            for _ in range(steps % total_steps, total_steps):
                while os.path.exists('stop'):
                    time.sleep(1)
                self.process(batch_size, test_batches,
                             batch_splits=batch_splits)

        from importlib.util import find_spec
        if find_spec('rich') is not None:
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
            from rich.table import Column

            self.progressbar = Progress(
                BarColumn(),
                "[progress.percentage]{task.percentage:>4.2f}%",
                TimeRemainingColumn(),
                TextColumn("{task.completed:.2f} of {task.total} steps completed.",
                           table_column=Column(ratio=1)),
                # TextColumn("Policy accuracy {task.train_metrics[6].get():.2f}", table_column=Column(ratio=1)),
                SpinnerColumn(),
            )
            with self.progressbar:
                self.progresstask = self.progressbar.add_task(
                    f"[green]Doing {total_steps} training steps", total=total_steps)
                loop()
        else:
            print('Warning, rich module not found, disabling progress bar')
            loop()

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

        policy_accuracy = self.policy_accuracy_fn(y, policy)
        policy_entropy = self.policy_entropy_fn(y, policy)
        policy_ul = self.policy_uniform_loss_fn(y, policy)

        if self.wdl:
            mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
            value_accuracy = self.accuracy_fn(self.qMix(z, q), value)
        else:
            value_loss = self.value_loss_fn(self.qMix(z, q), value)
            value_accuracy = tf.constant(0.)
        metrics = [
            policy_loss,
            value_loss,
            moves_left_loss,
            reg_term,
            total_loss,
            # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
            # get comparable values.
            mse_loss / 4.0,
            policy_accuracy * 100,
            value_accuracy * 100,
            policy_entropy,
            policy_ul,
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

    def apply_grads(self, grads, effective_batch_splits: int):
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
    def strategy_apply_grads(self, grads, effective_batch_splits: int):
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

    def train_step(self, steps: int, batch_size: int, batch_splits: int):
        # need to add 1 to steps because steps will be incremented after gradient update
        if (steps +
                1) % self.cfg['training']['train_avg_report_steps'] == 0 or (
                    steps + 1) % self.cfg['training']['total_steps'] == 0:
            before_weights = self.read_weights()

        # Run training for this batch
        grads = None
        for batch_id in range(batch_splits):
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

            if hasattr(self, 'progressbar'):
                self.progressbar.update(self.progresstask, completed=steps.numpy(
                ).item() - 1 + (batch_id+1) / batch_splits)
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

    def process(self, batch_size: int, test_batches: int, batch_splits: int):
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

        # Set DyDense temperature
        if self.dydense_usage != '':
            temperature = 1 + self.dydense_temp_start * max(
                1 - steps / self.dydense_temp_anneal_steps, 0)
            temperature = tf.cast(temperature, tf.float32)

            
            for layer in self.model.layers:
                if isinstance(layer, DyDense):
                    layer.temperature.assign(temperature)


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
            if False:  # !!!
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
            else:

                backup = self.read_weights()
                for (swa, w) in zip(self.swa_weights, self.model.weights):
                    w.assign(swa.read_value())
                evaled_steps = steps.numpy()
                tf.saved_model.save(self.model_to_save, os.path.join(
                    self.root_dir, self.cfg['name']) + "-" + str(evaled_steps))
                for (old, w) in zip(backup, self.model.weights):
                    w.assign(old)

        if self.profiling_start_step is not None and (
                steps >= self.profiling_start_step +
                self.cfg['training'].get('profile_step_count', 0)
                or steps % self.cfg['training']['total_steps'] == 0):
            tf.profiler.experimental.stop()
            self.profiling_start_step = None

    def calculate_swa_summaries(self, test_batches: int, steps: int):
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

    def calculate_test_summaries(self, test_batches: int, steps: int):
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

    def calculate_swa_validations(self, steps: int):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_validation_writer, self.validation_writer = self.validation_writer, self.swa_validation_writer
        print('swa', end=' ')
        self.calculate_test_validations(steps)
        self.validation_writer = true_validation_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def calculate_test_validations(self, steps: int):
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
    def compute_update_ratio(self, before_weights, after_weights, steps: int):
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

    def save_swa_weights(self, filename: str):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        self.save_leelaz_weights(filename)
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def save_leelaz_weights(self, filename: str):
        numpy_weights = []
        for weight in self.model.weights:
            numpy_weights.append([weight.name, weight.numpy()])
        self.net.fill_net_v2(numpy_weights)
        self.net.save_proto(filename)

    @staticmethod
    def split_heads(inputs, batch_size: int, num_heads: int, depth: int):
        if num_heads < 2:
            return inputs
        reshaped = tf.reshape(inputs, (batch_size, 64, num_heads, depth))
        # (batch_size, num_heads, 64, depth)
        return tf.transpose(reshaped, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, name: str = None, logit_gate: int = 0, dytalking_heads: bool = False, talking_heads: bool = False, inputs=None, use_simple_gating=False, squeezed=None, X=None):
        if use_simple_gating:
            assert inputs is not None

        # 0 h 64 d, 0 h d 64
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        batch_size = tf.shape(q)[0]
        dk = tf.cast(tf.shape(k)[-1], self.model_dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        heads = scaled_attention_logits.shape[1]
        # 0 h 64 64
        if talking_heads:
            scaled_attention_logits = tf.transpose(
                scaled_attention_logits, perm=[0, 2, 3, 1])

            to_concat = []

            if self.use_fullgen:
                # Update past logits
                self.past_logits.append(scaled_attention_logits)
                num_take = min(len(self.past_logits),
                               self.logit_history)
                to_concat.extend(self.past_logits[-num_take:])

                # Update past fullgen
                fullgen_weights = self.fullgen_weights(
                    inputs, self.fullgen_hidden_channels, self.fullgen_big_hidden_sz, self.fullgen_hidden_sz, self.fullgen_out_maps, name=name+'/fullgen', activation=self.fullgen_activation)
                self.fullgen_past_weights.append(fullgen_weights)
                num_take = min(len(self.fullgen_past_weights),
                               self.fullgen_history)
                to_concat.extend(self.fullgen_past_weights[-num_take:])
                if self.use_pergen:
                    to_concat.append(self.make_per_weights(inputs, name))
                if len(to_concat) == 1:
                    scaled_attention_logits = to_concat[0]
                else:
                    scaled_attention_logits = tf.concat(to_concat, axis=-1)

            if dytalking_heads:
                assert X is not None
                scaled_attention_logits = apply_dytalking_heads(X, scaled_attention_logits, heads)
            else:
                scaled_attention_logits = tf.keras.layers.Dense(heads, name=name+'/talking_heads1',
                                                                use_bias=False, kernel_initializer='glorot_normal')(scaled_attention_logits)

            if logit_gate >= 1:
                scaled_attention_logits = Gating(
                    name=name+'/logit_gate')(scaled_attention_logits)
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-2)



            if logit_gate >= 2:
                attention_weights = Gating(
                    name=name+'/early_mult_gate', additive=False)(attention_weights)

            if dytalking_heads:
                attention_weights = apply_dytalking_heads(X, attention_weights)
            else:
                attention_weights = tf.keras.layers.Dense(heads, name=name+'/talking_heads2',
                                                          use_bias=False, kernel_initializer='glorot_normal')(attention_weights)
            attention_weights = tf.transpose(attention_weights, [0, 3, 1, 2])

        else:
            if logit_gate >= 1:
                assert name is not None
                scaled_attention_logits = Gating(
                    name=name+'/logit_gate')(scaled_attention_logits)

            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        if logit_gate >= 3:
            attention_weights = Gating(
                name=name+'/late_mult_gate', additive=False)(attention_weights)

        output = tf.matmul(attention_weights, v)

        if heads > 1:
            output = tf.transpose(
                output, perm=[0, 2, 1, 3])
            output = tf.keras.layers.Reshape(
                (64, -1))(output)
        return output, scaled_attention_logits

    def make_per_weights(self, inputs, name: str):
        # aka pergen, improvements not worth the latency cost
        name=name+'/per_weights'
        compressed = tf.keras.layers.Dense(64, name=name+'/compress')(inputs)
        compressed = tf.keras.layers.Reshape([1, 64 * 64])(compressed)
        per_weights = self.per_weight_gen(compressed)
        per_weights = tf.keras.layers.Reshape([64, 64, 4])(per_weights)
        return per_weights

    def dense_layer(self, inputs, sz: int, *, name: str, squeezed=None, use_dydense: bool = False, dydense_kernels: int = None,
                    dydense_per_channel: bool = None, gating: bool = None, **kwargs):

        dyrelu = kwargs.get('activation') == 'dyrelu'
        linear_scale = kwargs.get('activation') == 'linear_scale'
        assert not linear_scale, 'linear_scale not implemented yet!'
        if dyrelu or linear_scale:
            kwargs['activation'] = None

        if use_dydense or linear_scale or dyrelu:
            assert squeezed is not None
            '''
            if squeezed is None:
                print(
                    f'Warning, squeezed needed but not provided in {name}, squeezing input')
                squeezed = tf.reduce_mean(inputs, axis=1)
                squeezed = tf.keras.layers.Dense(
                    32, activation='relu')(squeezed)
            '''

        # !!! better work this out
        if gating:
            kwargs['use_bias'] = False

        if use_dydense:
            if dydense_kernels is None:
                print(
                    f'INFO: dydense_kernels not provided in dense_layer {name}, using default {self.dydense_kernels=}')
            dydense_kernels = self.dydense_kernels
            if dydense_per_channel is None:
                print(
                    f'INFO: dydense_per_channel not provided in dense_layer {name}, using default {self.dydense_pc=}')
                dydense_per_channel = self.dydense_pc
            out = dydense(inputs, squeezed, sz, name=name, n_kernels=dydense_kernels,
                          per_channel=dydense_per_channel, **kwargs)
        else:
            out = tf.keras.layers.Dense(sz, name=name, **kwargs)(inputs)

        if gating:
            out = ma_gating(out, name=name)

        return out

    # multi-head attention in encoder blocks
    def mha(self, inputs, emb_size: int, d_model: int, num_heads: int, initializer, name: str, squeezed=None):
        assert d_model % num_heads == 0
        depth = d_model // num_heads
        # query, key, and value vectors for self-attention
        # inputs b, 64, sz

        if 's' in self.dcd_spec and self.use_dcd:
            inputs = inputs + dcd(inputs, squeezed, emb_size, self.dcd_size, name=name)


        q = self.dense_layer(inputs,
                             d_model,  kernel_initializer='glorot_normal', name=name + '/wq', use_dydense='q' in self.dydense_usage, squeezed=squeezed)
        k = self.dense_layer(inputs,
                             d_model, kernel_initializer='glorot_normal', name=name + '/wk', use_dydense='k' in self.dydense_usage, squeezed=squeezed)
        v = self.dense_layer(inputs,
                             d_model, kernel_initializer=initializer, name=name + '/wv', use_dydense='v' in self.dydense_usage, squeezed=squeezed)

        # split q, k and v into smaller vectors of size 'depth' -- one for each head in multi-head attention
        batch_size = tf.shape(q)[0]
        q = self.split_heads(q, batch_size, num_heads, depth)
        k = self.split_heads(k, batch_size, num_heads, depth)
        v = self.split_heads(v, batch_size, num_heads, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, name=name, logit_gate = self.logit_gate, talking_heads=self.use_talking_heads, inputs=inputs, use_simple_gating=False, dytalking_heads=self.dytalking_heads, squeezed=squeezed, X=inputs)

        # final dense layer
        output = self.dense_layer(scaled_attention,
                                  emb_size, kernel_initializer=initializer, name=name + "/dense_fin", use_dydense='o' in self.dydense_usage, squeezed=squeezed)
        return output, attention_weights

    # 2-layer dense feed-forward network in encoder blocks
    def ffn(self, inputs, emb_size: int, dff: int, initializer, name: str, squeezed=None):
        activation = 'dyrelu' if self.use_dyrelu else self.DEFAULT_ACTIVATION
        if self.use_dcd and 'f' in self.dcd_spec:
            inputs = inputs + dcd(inputs, squeezed, emb_size, self.dcd_size, name=name)
        dense1 = self.dense_layer(inputs, dff, kernel_initializer=initializer, activation=activation,
                                  name=name + "/dense1", use_dydense='1' in self.dydense_usage, squeezed=squeezed)
        return self.dense_layer(dense1, emb_size, kernel_initializer=initializer, name=name + "/dense2",
                                use_dydense='2' in self.dydense_usage, squeezed=squeezed)

    def encoder_layer(self, inputs, emb_size: int, d_model: int, num_heads: int, dff: int, name: str, training: bool):
        # DeepNorm
        alpha = tf.cast(tf.math.pow(
            2. * self.encoder_layers, 0.25), self.model_dtype)
        beta = tf.cast(tf.math.pow(
            8. * self.encoder_layers, -0.25), self.model_dtype)
        xavier_norm = tf.keras.initializers.VarianceScaling(
            scale=beta, mode='fan_avg', distribution='truncated_normal')
        # multihead attention
        squeezed = tf.keras.layers.GlobalAveragePooling1D(
            name=name+'/avgpool')(inputs)
        squeezed = tf.keras.layers.Dense(
            32, activation='relu', name=name+'/squeezed_dense')(squeezed)
        squeezed = tf.keras.layers.LayerNormalization(
            name=name+'/squeezed_dense/ln')(squeezed)

        attn_output, attn_wts = self.mha(
            inputs, emb_size, d_model, num_heads, xavier_norm, name=name + "/mha", squeezed=squeezed)

        # dropout for weight regularization
        attn_output = tf.keras.layers.Dropout(
            self.dropout_rate, name=name + "/dropout1")(attn_output, training=training)
        # skip connection + layernorm
        out1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=name + "/ln1")(inputs * alpha + attn_output)
        # feed-forward network
        ffn_output = self.ffn(out1, emb_size, dff,
                              xavier_norm, name=name + "/ffn", squeezed=squeezed)
        ffn_output = tf.keras.layers.Dropout(
            self.dropout_rate, name=name + "/dropout2")(ffn_output, training=training)
        out2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=name + "/ln2")(out1 * alpha + ffn_output)
        return out2, attn_wts

    def fullgen_weights(self, inputs, hidden_channels: int, big_hidden_sz: int, hidden_sz: int, out_channels: int, name: str, activation='relu'):
        compressed = tf.keras.layers.Dense(
            hidden_channels, name=name+'/compress', use_bias=False)(inputs)
        compressed = tf.reshape(compressed, [-1, 64 * hidden_channels])
        hidden = tf.keras.layers.Dense(
            big_hidden_sz, name=name+'/hidden1_dense', activation=activation)(compressed)
        hidden = tf.keras.layers.LayerNormalization(
            name=name+'/hidden1_ln')(hidden)
        hidden = tf.keras.layers.Dense(
            hidden_sz, name=name+'/hidden2_dense', activation=activation)(hidden)

        hidden = tf.keras.layers.LayerNormalization(
            name=name+'/hidden2_ln')(hidden)
        weights = self.full_weight_gen_dense(hidden)
        return tf.reshape(weights, [-1, 64, 64, out_channels])

    def construct_net(self, inputs, name: str = ''):
        # Policy head
        assert self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION
        attn_wts = []
        # TODO: re-add support for policy encoder blocks
        # do some input processing
        if self.use_fullgen:
            self.fullgen_past_weights = []
            self.past_logits = []
            self.full_weight_gen_dense = tf.keras.layers.Dense(
                self.fullgen_out_maps * 64 * 64, name=name+'weight_gen', use_bias=False)
        
        flow = tf.transpose(inputs, perm=[0, 2, 3, 1])
        flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])
        # add positional encoding for each square to the input
        if self.arc_encoding:
            self.POS_ENC = apm.make_pos_enc()
            positional_encoding = tf.broadcast_to(tf.convert_to_tensor(self.POS_ENC, dtype=flow.dtype),
                                                            [tf.shape(flow)[0], 64, tf.shape(self.POS_ENC)[2]])
            flow = tf.concat([flow, positional_encoding], axis=2)

        # square embedding
        flow = tf.keras.layers.Dense(self.embedding_size, kernel_initializer='glorot_normal',
                                        kernel_regularizer=self.l2reg, activation=self.DEFAULT_ACTIVATION,
                                        name=name+'embedding')(flow)

        # !!!
        # if self.input_gate:
        flow = ma_gating(flow, name=name+'embedding')

        for i in range(self.encoder_layers):
            flow, attn_wts_l = self.encoder_layer(flow, self.embedding_size, self.encoder_d_model,
                                                    self.encoder_heads, self.encoder_dff,
                                                    name=name+'encoder_{}'.format(i + 1), training=True)
            attn_wts.append(attn_wts_l)

        flow_ = flow
        
        # policy embedding
        tokens = tf.keras.layers.Dense(self.pol_embedding_size, kernel_initializer='glorot_normal',
                                        kernel_regularizer=self.l2reg, activation=self.DEFAULT_ACTIVATION,
                                        name=name+'policy/embedding')(flow_)

        # create queries and keys for policy self-attention
        queries = tf.keras.layers.Dense(self.policy_d_model, kernel_initializer='glorot_normal',
                                        name=name+'policy/attention/wq')(tokens)
        keys = tf.keras.layers.Dense(self.policy_d_model, kernel_initializer='glorot_normal',
                                        name=name+'policy/attention/wk')(tokens)

        # POLICY SELF-ATTENTION: self-attention weights are interpreted as from->to policy
        # Bx64x64 (from 64 queries, 64 keys)
        matmul_qk = tf.matmul(queries, keys, transpose_b=True)
        # queries = tf.keras.layers.Dense(self.policy_d_model, kernel_initializer='glorot_normal',
        #                                 name='policy/attention/wq')(flow)
        # keys = tf.keras.layers.Dense(self.policy_d_model, kernel_initializer='glorot_normal',
        #                              name='policy/attention/wk')(flow)

        # PAWN PROMOTION: create promotion logits using scalar offsets generated from the promotion-rank keys
        # constant for scaling
        dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1], self.model_dtype))
        promotion_keys = keys[:, -8:, :]
        # queen, rook, bishop, knight order
        promotion_offsets = tf.keras.layers.Dense(4, kernel_initializer='glorot_normal',
                                                    name=name+'policy/attention/ppo', use_bias=False)(promotion_keys)
        promotion_offsets = tf.transpose(
            promotion_offsets, perm=[0, 2, 1]) * dk  # Bx4x8
        # knight offset is added to the other three
        promotion_offsets = promotion_offsets[:,
                                                :3, :] + promotion_offsets[:, 3:4, :]

        # q, r, and b promotions are offset from the default promotion logit (knight)
        # default traversals from penultimate rank to promotion rank
        n_promo_logits = matmul_qk[:, -16:-8, -8:]
        q_promo_logits = tf.expand_dims(
            n_promo_logits + promotion_offsets[:, 0:1, :], axis=3)  # Bx8x8x1
        r_promo_logits = tf.expand_dims(
            n_promo_logits + promotion_offsets[:, 1:2, :], axis=3)
        b_promo_logits = tf.expand_dims(
            n_promo_logits + promotion_offsets[:, 2:3, :], axis=3)
        promotion_logits = tf.concat(
            [q_promo_logits, r_promo_logits, b_promo_logits], axis=3)  # Bx8x8x3
        # logits now alternate a7a8q,a7a8r,a7a8b,...,
        promotion_logits = tf.reshape(promotion_logits, [-1, 8, 24])

        # scale the logits by dividing them by sqrt(d_model) to stabilize gradients
        # Bx8x24 (8 from-squares, 3x8 promotions)
        promotion_logits = promotion_logits / dk
        # Bx64x64 (64 from-squares, 64 to-squares)
        policy_attn_logits = matmul_qk / dk

        attn_wts.append(promotion_logits)
        attn_wts.append(policy_attn_logits)

        # APPLY POLICY MAP: output becomes Bx1856
        h_fc1 = ApplyAttentionPolicyMap()(policy_attn_logits, promotion_logits)
    

        # Value head
        assert self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION and self.encoder_layers > 0
        embedded_val = tf.keras.layers.Dense(self.val_embedding_size, kernel_initializer='glorot_normal',
                                                kernel_regularizer=self.l2reg, activation=self.DEFAULT_ACTIVATION,
                                                name=name+'value/embedding')(flow)
        h_val_flat = tf.keras.layers.Flatten()(embedded_val)
        h_fc2 = tf.keras.layers.Dense(128,
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=self.l2reg,
                                      activation=self.DEFAULT_ACTIVATION,
                                      name=name+'value/dense1')(h_val_flat)
        # WDL head
        if self.wdl:
            h_fc3 = tf.keras.layers.Dense(3,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          bias_regularizer=self.l2reg,
                                          name=name+'value/dense2')(h_fc2)
        else:
            h_fc3 = tf.keras.layers.Dense(1,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          activation='tanh',
                                          name=name+'value/dense2')(h_fc2)

        # Moves left head
        if self.moves_left:
            embedded_mov = tf.keras.layers.Dense(self.mov_embedding_size, kernel_initializer='glorot_normal',
                                                    kernel_regularizer=self.l2reg, activation=self.DEFAULT_ACTIVATION,
                                                    name=name+'moves_left/embedding')(flow)
            h_mov_flat = tf.keras.layers.Flatten()(embedded_mov)

            h_fc4 = tf.keras.layers.Dense(
                128,
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                activation=self.DEFAULT_ACTIVATION,
                name=name+'moves_left/dense1')(h_mov_flat)

            h_fc5 = tf.keras.layers.Dense(1,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          activation='relu',
                                          name=name+'moves_left/dense2')(h_fc4)
        else:
            h_fc5 = None

        # attention weights added as optional output for analysis -- ignored by backend
        if self.moves_left:
            outputs = [h_fc1, h_fc3, h_fc5]
        else:
            outputs = [h_fc1, h_fc3]

        return outputs
