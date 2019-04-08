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
import lc0_az_policy_map
import proto.net_pb2 as pb

from net import Net


def weight_variable(shape, name=None):
    """Xavier initialization"""
    if len(shape) == 4:
        receptive_field = shape[0] * shape[1]
        fan_in = shape[2] * receptive_field
        fan_out = shape[3] * receptive_field
    else:
        fan_in = shape[0]
        fan_out = shape[1]
    # truncated normal has lower stddev than a regular normal distribution, so need to correct for that
    trunc_correction = np.sqrt(1.3)
    stddev = trunc_correction * np.sqrt(2.0 / (fan_in + fan_out))
    initial = tf.truncated_normal(shape, stddev=stddev)
    weights = tf.Variable(initial, name=name)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
    return weights

# Bias weights for layers not followed by BatchNorm
# We do not regularlize biases, so they are not
# added to the regularlizer collection


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')

class TFProcess:
    def __init__(self, cfg):
        self.cfg = cfg
        self.net = Net()
        self.root_dir = os.path.join(
            self.cfg['training']['path'], self.cfg['name'])

        # Network structure
        self.RESIDUAL_FILTERS = self.cfg['model']['filters']
        self.RESIDUAL_BLOCKS = self.cfg['model']['residual_blocks']
        self.SE_ratio = self.cfg['model']['se_ratio']
        self.policy_channels = self.cfg['model'].get('policy_channels', 32)

        policy_head = self.cfg['model'].get('policy', 'convolution')
        value_head  = self.cfg['model'].get('value', 'wdl')

        self.POLICY_HEAD = None
        self.VALUE_HEAD = None

        if policy_head == "classical":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CLASSICAL
        elif policy_head == "convolution":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CONVOLUTION
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

        # For exporting
        self.weights = []

        self.swa_enabled = self.cfg['training'].get('swa', False)

        # Limit momentum of SWA exponential average to 1 - 1/(swa_max_n + 1)
        self.swa_max_n = self.cfg['training'].get('swa_max_n', 0)

        self.renorm_enabled = self.cfg['training'].get('renorm', False)
        self.renorm_max_r = self.cfg['training'].get('renorm_max_r', 1)
        self.renorm_max_d = self.cfg['training'].get('renorm_max_d', 0)
        self.renorm_momentum = self.cfg['training'].get('renorm_momentum', 0.99)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90,
                                    allow_growth=True, visible_device_list="{}".format(self.cfg['gpu']))
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(config=config)

        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.placeholder(tf.float32)

    def init(self, dataset, train_iterator, test_iterator):
        # TF variables
        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            self.handle, dataset.output_types, dataset.output_shapes)
        self.next_batch = iterator.get_next()
        self.train_handle = self.session.run(train_iterator.string_handle())
        self.test_handle = self.session.run(test_iterator.string_handle())
        self.init_net(self.next_batch)

    def init_net(self, next_batch):
        self.x = next_batch[0]  # tf.placeholder(tf.float32, [None, 112, 8*8])
        self.y_ = next_batch[1] # tf.placeholder(tf.float32, [None, 1858])
        self.z_ = next_batch[2] # tf.placeholder(tf.float32, [None, 3])
        self.q_ = next_batch[3] # tf.placeholder(tf.float32, [None, 3])
        self.batch_norm_count = 0
        self.y_conv, self.z_conv = self.construct_net(self.x)

        # Calculate loss on policy head
        if self.cfg['training'].get('mask_legal_moves'):
            # extract mask for legal moves from target policy
            move_is_legal = tf.greater_equal(self.y_, 0)
            # replace logits of illegal moves with large negative value (so that it doesn't affect policy of legal moves) without gradient
            illegal_filler = tf.zeros_like(self.y_conv) - 1.0e10
            self.y_conv = tf.where(move_is_legal, self.y_conv, illegal_filler)
        # y_ still has -1 on illegal moves, flush them to 0
        self.y_ = tf.nn.relu(self.y_)
        policy_cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                    logits=self.y_conv)
        self.policy_loss = tf.reduce_mean(policy_cross_entropy)

        q_ratio = self.cfg['training'].get('q_ratio', 0)
        assert 0 <= q_ratio <= 1
        target = self.q_ * q_ratio + self.z_ * (1 - q_ratio)

        # Linear conversion to scalar to compute MSE with, for comparison to old values
        wdl = tf.expand_dims(tf.constant([1.0, 0.0, -1.0]), 1)
        scalar_target = tf.matmul(target, wdl)

        # Loss on value head
        if self.wdl:
            value_cross_entropy = \
                tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                    logits=self.z_conv)
            self.value_loss = tf.reduce_mean(value_cross_entropy)
            scalar_z_conv = tf.matmul(tf.nn.softmax(self.z_conv), wdl)
            self.mse_loss = \
                tf.reduce_mean(tf.squared_difference(scalar_target, scalar_z_conv))
        else:
            self.value_loss = tf.constant(0)
            self.mse_loss = \
                tf.reduce_mean(tf.squared_difference(scalar_target, self.z_conv))

        # Regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_term = \
            tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        # For training from a (smaller) dataset of strong players, you will
        # want to reduce the factor in front of self.mse_loss here.
        pol_loss_w = self.cfg['training']['policy_loss_weight']
        val_loss_w = self.cfg['training']['value_loss_weight']
        if self.wdl:
            value_loss = self.value_loss
        else:
            value_loss = self.mse_loss
        loss = pol_loss_w * self.policy_loss + \
            val_loss_w * value_loss + self.reg_term

        # Set adaptive learning rate during training
        self.cfg['training']['lr_boundaries'].sort()
        self.warmup_steps = self.cfg['training'].get('warmup_steps', 0)
        self.lr = self.cfg['training']['lr_values'][0]

        # You need to change the learning rate here if you are training
        # from a self-play training set, for example start with 0.005 instead.
        opt_op = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True)

        # Do swa after we contruct the net
        if self.swa_enabled:
            # Count of networks accumulated into SWA
            self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
            # Build the SWA variables and accumulators
            accum = []
            load = []
            n = self.swa_count
            for w in self.weights:
                name = w.name.split(':')[0]
                var = tf.Variable(
                    tf.zeros(shape=w.shape), name='swa/'+name, trainable=False)
                accum.append(
                    tf.assign(var, var * (n / (n + 1.)) + tf.stop_gradient(w) * (1. / (n + 1.))))
                load.append(tf.assign(w, var))
            with tf.control_dependencies(accum):
                self.swa_accum_op = tf.assign_add(n, 1.)
            self.swa_load_op = tf.group(*load)

        # Accumulate (possibly multiple) gradient updates to simulate larger batch sizes than can be held in GPU memory.
        gradient_accum = [tf.Variable(tf.zeros_like(
            var.initialized_value()), trainable=False) for var in tf.trainable_variables()]
        self.zero_op = [var.assign(tf.zeros_like(var))
                        for var in gradient_accum]

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            gradients = opt_op.compute_gradients(loss)
        self.accum_op = [accum.assign_add(
            gradient[0]) for accum, gradient in zip(gradient_accum, gradients)]
        # gradients are num_batch_splits times higher due to accumulation by summing, so the norm will be too
        max_grad_norm = self.cfg['training'].get(
            'max_grad_norm', 10000.0) * self.cfg['training'].get('num_batch_splits', 1)
        gradient_accum, self.grad_norm = tf.clip_by_global_norm(
            gradient_accum, max_grad_norm)
        self.train_op = opt_op.apply_gradients(
            [(accum, gradient[1]) for accum, gradient in zip(gradient_accum, gradients)], global_step=self.global_step)

        correct_policy_prediction = \
            tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        correct_policy_prediction = tf.cast(correct_policy_prediction, tf.float32)
        self.policy_accuracy = tf.reduce_mean(correct_policy_prediction)
        correct_value_prediction = \
            tf.equal(tf.argmax(self.z_conv, 1), tf.argmax(self.z_, 1))
        correct_value_prediction = tf.cast(correct_value_prediction, tf.float32)
        self.value_accuracy = tf.reduce_mean(correct_value_prediction)

        self.avg_policy_loss = []
        self.avg_value_loss = []
        self.avg_mse_loss = []
        self.avg_reg_term = []
        self.time_start = None
        self.last_steps = None

        # Summary part
        self.test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelalogs/{}-test".format(self.cfg['name'])), self.session.graph)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelalogs/{}-train".format(self.cfg['name'])), self.session.graph)
        if self.swa_enabled:
            self.swa_writer = tf.summary.FileWriter(
                os.path.join(os.getcwd(), "leelalogs/{}-swa-test".format(self.cfg['name'])), self.session.graph)
        self.histograms = [tf.summary.histogram(
            weight.name, weight) for weight in self.weights]

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.session.run(self.init)

    def replace_weights(self, new_weights):
        for e, weights in enumerate(self.weights):
            if weights.shape.ndims == 4:
                # Rescale rule50 related weights as clients do not normalize the input.
                if e == 0:
                    num_inputs = 112
                    # 50 move rule is the 110th input, or 109 starting from 0.
                    rule50_input = 109
                    for i in range(len(new_weights[e])):
                        if (i % (num_inputs*9))//9 == rule50_input:
                            new_weights[e][i] = new_weights[e][i]*99

                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(
                    tf.transpose(new_weight, [2, 3, 1, 0])))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(
                    tf.transpose(new_weight, [1, 0])))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[e], shape=weights.shape)
                self.session.run(tf.assign(weights, new_weight))
        # This should result in identical file to the starting one
        # self.save_leelaz_weights('restored.txt')

    def restore(self, file):
        print("Restoring from {0}".format(file))
        self.saver.restore(self.session, file)

    def process_loop(self, batch_size, test_batches, batch_splits=1):
        # Get the initial steps value in case this is a resume from a step count
        # which is not a multiple of total_steps.
        steps = tf.train.global_step(self.session, self.global_step)
        total_steps = self.cfg['training']['total_steps']
        for _ in range(steps % total_steps, total_steps):
            self.process(batch_size, test_batches, batch_splits=batch_splits)

    def process(self, batch_size, test_batches, batch_splits=1):
        if not self.time_start:
            self.time_start = time.time()

        # Get the initial steps value before we do a training step.
        steps = tf.train.global_step(self.session, self.global_step)
        if not self.last_steps:
            self.last_steps = steps

        if self.swa_enabled:
            # split half of test_batches between testing regular weights and SWA weights
            test_batches //= 2

        # Run test before first step to see delta since end of last run.
        if steps % self.cfg['training']['total_steps'] == 0:
            # Steps is given as one higher than current in order to avoid it
            # being equal to the value the end of a run is stored against.
            self.calculate_test_summaries(test_batches, steps + 1)
            if self.swa_enabled:
                self.calculate_swa_summaries(test_batches, steps + 1)

        # Make sure that ghost batch norm can be applied
        if batch_size % 64 != 0:
            # Adjust required batch size for batch splitting.
            required_factor = 64 * \
                self.cfg['training'].get('num_batch_splits', 1)
            raise ValueError(
                'batch_size must be a multiple of {}'.format(required_factor))

        # Determine learning rate
        lr_values = self.cfg['training']['lr_values']
        lr_boundaries = self.cfg['training']['lr_boundaries']
        steps_total = steps % self.cfg['training']['total_steps']
        self.lr = lr_values[bisect.bisect_right(lr_boundaries, steps_total)]
        if self.warmup_steps > 0 and steps < self.warmup_steps:
            self.lr = self.lr * (steps + 1) / self.warmup_steps

        # need to add 1 to steps because steps will be incremented after gradient update
        if (steps + 1) % self.cfg['training']['train_avg_report_steps'] == 0 or (steps + 1) % self.cfg['training']['total_steps'] == 0:
            before_weights = self.session.run(self.weights)

        # Run training for this batch
        self.session.run(self.zero_op)
        for _ in range(batch_splits):
            policy_loss, value_loss, mse_loss, reg_term, _, _ = self.session.run(
                [self.policy_loss, self.value_loss, self.mse_loss, self.reg_term, self.accum_op,
                    self.next_batch],
                feed_dict={self.training: True, self.handle: self.train_handle})
            # Keep running averages
            # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
            # get comparable values.
            mse_loss /= 4.0
            self.avg_policy_loss.append(policy_loss)
            if self.wdl:
                self.avg_value_loss.append(value_loss)
            self.avg_mse_loss.append(mse_loss)
            self.avg_reg_term.append(reg_term)
        # Gradients of batch splits are summed, not averaged like usual, so need to scale lr accordingly to correct for this.
        corrected_lr = self.lr / batch_splits
        _, grad_norm = self.session.run([self.train_op, self.grad_norm],
                                        feed_dict={self.learning_rate: corrected_lr, self.training: True, self.handle: self.train_handle})

        # Update steps since training should have incremented it.
        steps = tf.train.global_step(self.session, self.global_step)

        if steps % self.cfg['training']['train_avg_report_steps'] == 0 or steps % self.cfg['training']['total_steps'] == 0:
            pol_loss_w = self.cfg['training']['policy_loss_weight']
            val_loss_w = self.cfg['training']['value_loss_weight']
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                steps_elapsed = steps - self.last_steps
                speed = batch_size * (steps_elapsed / elapsed)
            avg_policy_loss = np.mean(self.avg_policy_loss or [0])
            avg_value_loss = np.mean(self.avg_value_loss or [0])
            avg_mse_loss = np.mean(self.avg_mse_loss or [0])
            avg_reg_term = np.mean(self.avg_reg_term or [0])
            print("step {}, lr={:g} policy={:g} value={:g} mse={:g} reg={:g} total={:g} ({:g} pos/s)".format(
                steps, self.lr, avg_policy_loss, avg_value_loss, avg_mse_loss, avg_reg_term,
                pol_loss_w * avg_policy_loss + val_loss_w * avg_value_loss + avg_reg_term,
                speed))

            after_weights = self.session.run(self.weights)
            update_ratio_summaries = self.compute_update_ratio(
                before_weights, after_weights)

            train_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Policy Loss", simple_value=avg_policy_loss),
                tf.Summary.Value(tag="Value Loss", simple_value=avg_value_loss),
                tf.Summary.Value(tag="Reg term", simple_value=avg_reg_term),
                tf.Summary.Value(tag="LR", simple_value=self.lr),
                tf.Summary.Value(tag="Gradient norm",
                                 simple_value=grad_norm / batch_splits),
                tf.Summary.Value(tag="MSE Loss", simple_value=avg_mse_loss)])
            self.train_writer.add_summary(train_summaries, steps)
            self.train_writer.add_summary(update_ratio_summaries, steps)
            self.time_start = time_end
            self.last_steps = steps
            self.avg_policy_loss, self.avg_value_loss, self.avg_mse_loss, self.avg_reg_term = [], [], [], []

        if self.swa_enabled and steps % self.cfg['training']['swa_steps'] == 0:
            self.update_swa()

        # Calculate test values every 'test_steps', but also ensure there is
        # one at the final step so the delta to the first step can be calculted.
        if steps % self.cfg['training']['test_steps'] == 0 or steps % self.cfg['training']['total_steps'] == 0:
            self.calculate_test_summaries(test_batches, steps)
            if self.swa_enabled:
                self.calculate_swa_summaries(test_batches, steps)

        # Save session and weights at end, and also optionally every 'checkpoint_steps'.
        if steps % self.cfg['training']['total_steps'] == 0 or (
                'checkpoint_steps' in self.cfg['training'] and steps % self.cfg['training']['checkpoint_steps'] == 0):
            path = os.path.join(self.root_dir, self.cfg['name'])
            save_path = self.saver.save(self.session, path, global_step=steps)
            print("Model saved in file: {}".format(save_path))
            leela_path = path + "-" + str(steps)
            swa_path = path + "-swa-" + str(steps)
            self.net.pb.training_params.training_steps = steps
            self.save_leelaz_weights(leela_path)
            print("Weights saved in file: {}".format(leela_path))
            if self.swa_enabled:
                self.save_swa_weights(swa_path)
                print("SWA Weights saved in file: {}".format(swa_path))

    def calculate_swa_summaries(self, test_batches, steps):
        self.snap_save()
        self.session.run(self.swa_load_op)
        true_test_writer, self.test_writer = self.test_writer, self.swa_writer
        print('swa', end=' ')
        self.calculate_test_summaries(test_batches, steps)
        self.test_writer = true_test_writer
        self.snap_restore()

    def calculate_test_summaries(self, test_batches, steps):
        sum_policy_accuracy = 0
        sum_value_accuracy = 0
        sum_mse = 0
        sum_policy = 0
        sum_value = 0
        for _ in range(0, test_batches):
            test_policy, test_value, test_policy_accuracy, test_value_accuracy, test_mse, _ = self.session.run(
                [self.policy_loss, self.value_loss, self.policy_accuracy, self.value_accuracy, self.mse_loss,
                 self.next_batch],
                feed_dict={self.training: False,
                           self.handle: self.test_handle})
            sum_policy_accuracy += test_policy_accuracy
            sum_mse += test_mse
            sum_policy += test_policy
            if self.wdl:
                sum_value_accuracy += test_value_accuracy
                sum_value += test_value
        sum_policy_accuracy /= test_batches
        sum_policy_accuracy *= 100
        sum_policy /= test_batches
        sum_value /= test_batches
        if self.wdl:
            sum_value_accuracy /= test_batches
            sum_value_accuracy *= 100
        # Additionally rescale to [0, 1] so divide by 4
        sum_mse /= (4.0 * test_batches)
        self.net.pb.training_params.learning_rate = self.lr
        self.net.pb.training_params.mse_loss = sum_mse
        self.net.pb.training_params.policy_loss = sum_policy
        # TODO store value and value accuracy in pb
        self.net.pb.training_params.accuracy = sum_policy_accuracy
        if self.wdl:
            test_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Policy Accuracy", simple_value=sum_policy_accuracy),
                tf.Summary.Value(tag="Value Accuracy", simple_value=sum_value_accuracy),
                tf.Summary.Value(tag="Policy Loss", simple_value=sum_policy),
                tf.Summary.Value(tag="Value Loss", simple_value=sum_value),
                tf.Summary.Value(tag="MSE Loss", simple_value=sum_mse)]).SerializeToString()
        else:
            test_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Policy Accuracy", simple_value=sum_policy_accuracy),
                tf.Summary.Value(tag="Policy Loss", simple_value=sum_policy),
                tf.Summary.Value(tag="MSE Loss", simple_value=sum_mse)]).SerializeToString()
        test_summaries = tf.summary.merge(
            [test_summaries] + self.histograms).eval(session=self.session)
        self.test_writer.add_summary(test_summaries, steps)
        print("step {}, policy={:g} value={:g} policy accuracy={:g}% value accuracy={:g}% mse={:g}".\
            format(steps, sum_policy, sum_value, sum_policy_accuracy, sum_value_accuracy, sum_mse))

    def compute_update_ratio(self, before_weights, after_weights):
        """Compute the ratio of gradient norm to weight norm.

        Adapted from https://github.com/tensorflow/minigo/blob/c923cd5b11f7d417c9541ad61414bf175a84dc31/dual_net.py#L567
        """
        deltas = [after - before for after,
                  before in zip(after_weights, before_weights)]
        delta_norms = [np.linalg.norm(d.ravel()) for d in deltas]
        weight_norms = [np.linalg.norm(w.ravel()) for w in before_weights]
        ratios = [(tensor.name, d / w) for d, w, tensor in zip(delta_norms, weight_norms, self.weights) if not 'moving' in tensor.name]
        all_summaries = [
            tf.Summary.Value(tag='update_ratios/' +
                             name, simple_value=ratio)
            for name, ratio in ratios]
        ratios = np.log10([r for (_, r) in ratios if 0 < r < np.inf])
        all_summaries.append(self.log_histogram('update_ratios_log10', ratios))
        return tf.Summary(value=all_summaries)

    def log_histogram(self, tag, values, bins=1000):
        """Logs the histogram of a list/vector of values.

        From https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
        """
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        return tf.Summary.Value(tag=tag, histo=hist)

    def update_swa(self):
        # Add the current weight vars to the running average.
        num = self.session.run(self.swa_accum_op)
        num = min(num, self.swa_max_n)
        self.swa_count.load(float(num), self.session)

    def snap_save(self):
        # Save a snapshot of all the variables in the current graph.
        if not hasattr(self, 'save_op'):
            save_ops = []
            rest_ops = []
            for var in self.weights:
                if isinstance(var, str):
                    var = tf.get_default_graph().get_tensor_by_name(var)
                name = var.name.split(':')[0]
                v = tf.Variable(var, name='save/'+name, trainable=False)
                save_ops.append(tf.assign(v, var))
                rest_ops.append(tf.assign(var, v))
            self.snap_save_op = tf.group(*save_ops)
            self.snap_restore_op = tf.group(*rest_ops)
        self.session.run(self.snap_save_op)

    def snap_restore(self):
        # Restore variables in the current graph from the snapshot.
        self.session.run(self.snap_restore_op)

    def save_swa_weights(self, filename):
        self.snap_save()
        self.session.run(self.swa_load_op)
        self.save_leelaz_weights(filename)
        self.snap_restore()

    def save_leelaz_weights(self, filename):
        all_weights = []
        all_evals = []
        for e, weights in enumerate(self.weights):
            work_weights = None
            if weights.shape.ndims == 4:
                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                work_weights = tf.transpose(weights, [3, 2, 0, 1])
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                work_weights = tf.transpose(weights, [1, 0])
            else:
                # Biases, batchnorm etc
                work_weights = weights
            all_evals.append(work_weights)
        nparrays = self.session.run(all_evals)
        for e, nparray in enumerate(nparrays):
            # Rescale rule50 related weights as clients do not normalize the input.
            if e == 0:
                num_inputs = 112
                # 50 move rule is the 110th input, or 109 starting from 0.
                rule50_input = 109
                wt_flt = []
                for i, weight in enumerate(np.ravel(nparray)):
                    if (i % (num_inputs*9))//9 == rule50_input:
                        wt_flt.append(weight/99)
                    else:
                        wt_flt.append(weight)
            else:
                wt_flt = [wt for wt in np.ravel(nparray)]
            all_weights.append(wt_flt)

        self.net.fill_net(all_weights)
        self.net.save_proto(filename)

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def squeeze_excitation(self, x, channels, ratio):

        assert channels % ratio == 0

        # NCHW format reduced to NC
        net = tf.reduce_mean(x, axis=[2, 3])

        W_fc1 = weight_variable([channels, channels // ratio], name='se_fc1_w')
        b_fc1 = bias_variable([channels // ratio], name='se_fc1_b')
        self.weights.append(W_fc1)
        self.weights.append(b_fc1)

        net = tf.nn.relu(tf.add(tf.matmul(net, W_fc1), b_fc1))

        W_fc2 = weight_variable(
            [channels // ratio, 2 * channels], name='se_fc2_w')
        b_fc2 = bias_variable([2 * channels], name='se_fc2_b')
        self.weights.append(W_fc2)
        self.weights.append(b_fc2)

        net = tf.add(tf.matmul(net, W_fc2), b_fc2)
        net = tf.reshape(net, [-1, 2 * channels, 1, 1])

        # Split to scale and bias
        gammas, betas = tf.split(net, 2, axis=1)

        out = tf.nn.sigmoid(gammas) * x + betas

        return out

    def batch_norm(self, inputs, scale=False):
        if self.renorm_enabled:
            clipping = {
                "rmin": 1.0/self.renorm_max_r,
                "rmax": self.renorm_max_r,
                "dmax": self.renorm_max_d
                }
            return tf.layers.batch_normalization(
                inputs, epsilon=1e-5, axis=1, fused=True,
                center=True, scale=scale,
                renorm=True, renorm_clipping=clipping,
                renorm_momentum=self.renorm_momentum,
                training=self.training)
        else:
            return tf.layers.batch_normalization(
                inputs, epsilon=1e-5, axis=1, fused=True,
                center=True, scale=scale,
                virtual_batch_size=64,
                training=self.training)

    def conv_block(self, inputs, filter_size, input_channels, output_channels, bn_scale=False):
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        weight_key = self.get_batchnorm_key()
        conv_key = weight_key + "/conv_weight"
        W_conv = weight_variable([filter_size, filter_size,
                                  input_channels, output_channels], name=conv_key)

        with tf.variable_scope(weight_key):
            h_bn = self.batch_norm(conv2d(inputs, W_conv), scale=bn_scale)
        h_conv = tf.nn.relu(h_bn)

        gamma_key = weight_key + "/batch_normalization/gamma"
        if bn_scale:
            gamma_key = gamma_key + ":0"
        beta_key = weight_key + "/batch_normalization/beta:0"
        mean_key = weight_key + "/batch_normalization/moving_mean:0"
        var_key = weight_key + "/batch_normalization/moving_variance:0"

        if bn_scale:
            gamma = tf.get_default_graph().get_tensor_by_name(gamma_key)
        else:
            gamma = tf.Variable(tf.ones(shape=[output_channels]),
                                name=gamma_key, trainable=False)
        beta = tf.get_default_graph().get_tensor_by_name(beta_key)
        mean = tf.get_default_graph().get_tensor_by_name(mean_key)
        var = tf.get_default_graph().get_tensor_by_name(var_key)

        self.weights.append(W_conv)
        self.weights.append(gamma)
        self.weights.append(beta)
        self.weights.append(mean)
        self.weights.append(var)

        return h_conv

    def residual_block(self, inputs, channels):
        # First convnet
        orig = tf.identity(inputs)
        weight_key_1 = self.get_batchnorm_key() + "/"
        conv_key_1 = weight_key_1 + "conv_weight"
        W_conv_1 = weight_variable([3, 3, channels, channels], name=conv_key_1)

        # Second convnet
        weight_key_2 = self.get_batchnorm_key() + "/"
        conv_key_2 = weight_key_2 + "conv_weight"
        W_conv_2 = weight_variable([3, 3, channels, channels], name=conv_key_2)

        with tf.variable_scope(weight_key_1):
            h_bn1 = self.batch_norm(conv2d(inputs, W_conv_1))
        h_out_1 = tf.nn.relu(h_bn1)
        with tf.variable_scope(weight_key_2):
            h_bn2 = self.batch_norm(conv2d(h_out_1, W_conv_2), scale=True)

        gamma_key_1 = weight_key_1 + "/batch_normalization/gamma"
        beta_key_1 = weight_key_1 + "/batch_normalization/beta:0"
        mean_key_1 = weight_key_1 + "/batch_normalization/moving_mean:0"
        var_key_1 = weight_key_1 + "/batch_normalization/moving_variance:0"

        gamma_1 = tf.Variable(tf.ones(shape=[channels]),
                              name=gamma_key_1, trainable=False)
        beta_1 = tf.get_default_graph().get_tensor_by_name(beta_key_1)
        mean_1 = tf.get_default_graph().get_tensor_by_name(mean_key_1)
        var_1 = tf.get_default_graph().get_tensor_by_name(var_key_1)

        gamma_key_2 = weight_key_2 + "/batch_normalization/gamma:0"
        beta_key_2 = weight_key_2 + "/batch_normalization/beta:0"
        mean_key_2 = weight_key_2 + "/batch_normalization/moving_mean:0"
        var_key_2 = weight_key_2 + "/batch_normalization/moving_variance:0"

        gamma_2 = tf.get_default_graph().get_tensor_by_name(gamma_key_2)
        beta_2 = tf.get_default_graph().get_tensor_by_name(beta_key_2)
        mean_2 = tf.get_default_graph().get_tensor_by_name(mean_key_2)
        var_2 = tf.get_default_graph().get_tensor_by_name(var_key_2)

        self.weights.append(W_conv_1)
        self.weights.append(gamma_1)
        self.weights.append(beta_1)
        self.weights.append(mean_1)
        self.weights.append(var_1)

        self.weights.append(W_conv_2)
        self.weights.append(gamma_2)
        self.weights.append(beta_2)
        self.weights.append(mean_2)
        self.weights.append(var_2)

        # Must be after adding weights to self.weights
        with tf.variable_scope(weight_key_2):
            h_se = self.squeeze_excitation(h_bn2, channels, self.SE_ratio)
        h_out_2 = tf.nn.relu(tf.add(h_se, orig))

        return h_out_2

    def construct_net(self, planes):
        # NCHW format
        # batch, 112 input channels, 8 x 8
        x_planes = tf.reshape(planes, [-1, 112, 8, 8])

        # Input convolution
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=112,
                               output_channels=self.RESIDUAL_FILTERS,
                               bn_scale=True)
        # Residual tower
        for _ in range(0, self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow, self.RESIDUAL_FILTERS)

        # Policy head
        if self.POLICY_HEAD == pb.NetworkFormat.POLICY_CONVOLUTION:
            conv_pol = self.conv_block(flow, filter_size=3,
                                       input_channels=self.RESIDUAL_FILTERS,
                                       output_channels=self.RESIDUAL_FILTERS)
            W_pol_conv = weight_variable([3, 3,
                                          self.RESIDUAL_FILTERS, 80], name='W_pol_conv2')
            b_pol_conv = bias_variable([80], name='b_pol_conv2')
            self.weights.append(W_pol_conv)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, b_pol_conv)
            self.weights.append(b_pol_conv)
            conv_pol2 = tf.nn.bias_add(
                conv2d(conv_pol, W_pol_conv), b_pol_conv, data_format='NCHW')

            h_conv_pol_flat = tf.reshape(conv_pol2, [-1, 80*8*8])
            fc1_init = tf.constant(lc0_az_policy_map.make_map())
            W_fc1 = tf.get_variable("policy_map",
                                    initializer=fc1_init,
                                    trainable=False)
            h_fc1 = tf.matmul(h_conv_pol_flat, W_fc1, name='policy_head')
        elif self.POLICY_HEAD == pb.NetworkFormat.POLICY_CLASSICAL:
            conv_pol = self.conv_block(flow, filter_size=1,
                                       input_channels=self.RESIDUAL_FILTERS,
                                       output_channels=self.policy_channels)
            h_conv_pol_flat = tf.reshape(
                conv_pol, [-1, self.policy_channels*8*8])
            W_fc1 = weight_variable(
                [self.policy_channels*8*8, 1858], name='fc1/weight')
            b_fc1 = bias_variable([1858], name='fc1/bias')
            self.weights.append(W_fc1)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, b_fc1)
            self.weights.append(b_fc1)
            h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1),
                           b_fc1, name='policy_head')
        else:
            raise ValueError(
                "Unknown policy head type {}".format(self.POLICY_HEAD))

        # Value head
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=32)
        h_conv_val_flat = tf.reshape(conv_val, [-1, 32*8*8])
        W_fc2 = weight_variable([32 * 8 * 8, 128], name='fc2/weight')
        b_fc2 = bias_variable([128], name='fc2/bias')
        self.weights.append(W_fc2)
        self.weights.append(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        value_outputs = 3 if self.wdl else 1
        W_fc3 = weight_variable([128, value_outputs], name='fc3/weight')
        b_fc3 = bias_variable([value_outputs], name='fc3/bias')
        self.weights.append(W_fc3)
        self.weights.append(b_fc3)
        h_fc3 = tf.add(tf.matmul(h_fc2, W_fc3), b_fc3, name='value_head')
        if not self.wdl:
            h_fc3 = tf.nn.tanh(h_fc3)
        else:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, b_fc3)


        return h_fc1, h_fc3
