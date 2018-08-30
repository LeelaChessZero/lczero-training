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

import math
import numpy as np
import os
import tensorflow as tf
import time
import unittest
import bisect
from net import Net


def weight_variable(name, shape):
    """Xavier initialization"""
    stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(shape, stddev=stddev)
    weights = tf.get_variable(name, initializer=initial)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
    return weights

# Bias weights for layers not followed by BatchNorm
# We do not regularlize biases, so they are not
# added to the regularlizer collection
def bias_variable(name, shape):
    initial = tf.constant(0.0, shape=shape)
    bias = tf.get_variable(name, initializer=initial)
    return bias


def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')

# Restore session from checkpoint. It silently ignore mis-matches
# between the checkpoint and the graph. Specifically
# 1. values in the checkpoint for which there is no corresponding variable.
# 2. variables in the graph for which there is no specified value in the
#    checkpoint.
# 3. values where the checkpoint shape differs from the variable shape.
# (variables without a value in the checkpoint are left at their default
# initialized value)
def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted(
        [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
         if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)

# Class holding statistics
class Stats:
    def __init__(self):
        self.s = {}
    def add(self, stat_dict):
        for (k,v) in stat_dict.items():
            if k not in self.s:
                self.s[k] = []
            self.s[k].append(v)
    def n(self, name):
        return len(self.s[name] or [])
    def mean(self, name):
        return np.mean(self.s[name] or [0])
    def stddev_mean(self, name):
        # standard deviation in the sample mean.
        return math.sqrt(
            np.var(self.s[name] or [0]) / max(0.0001, (len(self.s[name]) - 1)))
    def str(self):
        return ', '.join(
            ["{}={:g}".format(k, np.mean(v or [0])) for k,v in self.s.items()])
    def clear(self):
        self.s = {}
    def summaries(self, tags):
        return [tf.Summary.Value(
            tag=k, simple_value=self.mean(v)) for k,v in tags.items()]

# Simple timer
class Timer:
    def __init__(self):
        self.last = time.time()
    def elapsed(self):
        # Return time since last call to 'elapsed()'
        t = time.time()
        e = t - self.last
        self.last = t
        return e

class TFProcess:
    def __init__(self, cfg):
        self.cfg = cfg
        self.net = Net()
        self.root_dir = os.path.join(self.cfg['training']['path'], self.cfg['name'])

        # Network structure
        self.RESIDUAL_FILTERS = self.cfg['model']['filters']
        self.RESIDUAL_BLOCKS = self.cfg['model']['residual_blocks']

        # Set number of GPUs for training
        self.gpus = self.cfg['training']['gpus']

        # For exporting
        self.weights = []

        # Output weight file with averaged weights
        self.swa_enabled = self.cfg['training']['swa']

        # Take an exponentially weighted moving average over this
        # many networks. Under the SWA assumptions, this will reduce
        # the distance to the optimal value by a factor of 1/sqrt(n)
        self.swa_max_n = self.cfg['training']['swa_cycle']

        # Net sampling rate (e.g 2 == every 2nd network).
        self.swa_c = self.cfg['training']['swa_cycle']

        # Recalculate SWA weight batchnorm means and variances
        self.swa_recalc_bn = True

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.session = tf.Session(config=config)

        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = self.cfg['training']['lr1'][0]
        self.learning_rate = tf.placeholder(tf.float32)
    
    def learning_rate_(self, i, c, a1, a2):
        return (1 - self.time(i, c)) * a1 + self.time(i, c) * a2

    def time(self, i, c):
        return 1 / c * (((i - 1) % c) + 1)

    def init(self, logbase='leelalogs'):
        self.batch_size = self.cfg['training']['vram_batch_size']
        self.macrobatch = self.cfg['training']['batch_size'] // self.cfg['training']['vram_batch_size']
        self.logbase = logbase
        # Input batch placeholders
        self.planes = tf.placeholder(tf.string, name='in_planes')
        self.probs = tf.placeholder(tf.string, name='in_probs')
        self.winner = tf.placeholder(tf.string, name='in_winner')

        # Mini-batches come as raw packed strings. Decode
        # into tensors to feed into network.
        planes = tf.decode_raw(self.planes, tf.float32)
        probs = tf.decode_raw(self.probs, tf.float32)
        winner = tf.decode_raw(self.winner, tf.float32)

        planes = tf.reshape(planes, (self.batch_size, 112, 8*8))
        probs = tf.reshape(probs, (self.batch_size, 1858))
        winner = tf.reshape(winner, (self.batch_size, 1))

        self.init_net(planes, probs, winner)

    def init_net(self, planes, probs, winner):
        gpus_num = len(self.gpus)
        self.y_ = probs   # (tf.float32, [None, 362])
        self.sx = tf.split(planes, gpus_num)
        self.sy_ = tf.split(probs, gpus_num)
        self.sz_ = tf.split(winner, gpus_num)
        self.batch_norm_count = 0
        self.reuse_var = None

        # You need to change the learning rate here if you are training
        # from a self-play training set, for example start with 0.005 instead.
        opt = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True)

        # Construct net here.
        tower_grads = []
        tower_loss = []
        tower_policy_loss = []
        tower_mse_loss = []
        tower_reg_term = []
        tower_y_conv = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i, gpu in enumerate(self.gpus):
                with tf.device("/gpu:%d" % gpu):
                    with tf.name_scope("tower_%d" % i):
                        loss, policy_loss, mse_loss, reg_term, y_conv = self.tower_loss(
                            self.sx[i], self.sy_[i], self.sz_[i])

                        # Reset batchnorm key to 0.
                        self.reset_batchnorm_key()

                        tf.get_variable_scope().reuse_variables()
                        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                            grads = opt.compute_gradients(loss)

                        tower_grads.append(grads)
                        tower_loss.append(loss)
                        tower_policy_loss.append(policy_loss)
                        tower_mse_loss.append(mse_loss)
                        tower_reg_term.append(reg_term)
                        tower_y_conv.append(y_conv)

        # Average gradients from different GPUs
        self.loss = tf.reduce_mean(tower_loss)
        self.policy_loss = tf.reduce_mean(tower_policy_loss)
        self.mse_loss = tf.reduce_mean(tower_mse_loss)
        self.reg_term = tf.reduce_mean(tower_reg_term)
        self.y_conv = tf.concat(tower_y_conv, axis=0)
        self.mean_grads = self.average_gradients(tower_grads)

        # Do swa after we contruct the net
        if self.swa_enabled is True:
            # Count of networks accumulated into SWA
            self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
            # Count of networks to skip
            self.swa_skip = tf.Variable(self.swa_c, name='swa_skip',
                trainable=False)
            # Build the SWA variables and accumulators
            accum=[]
            load=[]
            n = self.swa_count
            for w in self.weights:
                name = w.name.split(':')[0]
                var = tf.Variable(
                    tf.zeros(shape=w.shape), name='swa/'+name, trainable=False)
                accum.append(
                    tf.assign(var, var * (n / (n + 1.)) + w * (1. / (n + 1.))))
                load.append(tf.assign(w, var))
            with tf.control_dependencies(accum):
                self.swa_accum_op = tf.assign_add(n, 1.)
            self.swa_load_op = tf.group(*load)

        # Accumulate gradients
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        total_grad=[]
        grad_ops=[]
        clear_var=[]
        self.grad_op_real = self.mean_grads
        for (g, v) in self.grad_op_real:
            if g is None:
                total_grad.append((g,v))
            name = v.name.split(':')[0]
            gsum = tf.get_variable(name='gsum/'+name,
                                   shape=g.shape,
                                   trainable=False,
                                   initializer=tf.zeros_initializer)
            total_grad.append((gsum, v))
            grad_ops.append(tf.assign_add(gsum, g))
            clear_var.append(gsum)
        # Op to compute gradients and add to running total in 'gsum/'
        self.grad_op = tf.group(*grad_ops)

        # Op to apply accmulated gradients
        self.train_op = opt.apply_gradients(total_grad)

        zero_ops = []
        for g in clear_var:
            zero_ops.append(
                tf.assign(g, tf.zeros(shape=g.shape, dtype=g.dtype)))
        # Op to clear accumulated gradients
        self.clear_op = tf.group(*zero_ops)

        # Op to increment global step counter
        self.step_op = tf.assign_add(self.global_step, 1)

        correct_prediction = \
            tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        # Summary part
        self.test_writer = tf.summary.FileWriter(
            os.path.join(self.root_dir,
                         self.logbase + "/" + self.cfg['name'] + "-test"), self.session.graph)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(self.root_dir,
                         self.logbase + "/" + self.cfg['name'] + "-train"), self.session.graph)

        # Build checkpoint saver
        self.saver = tf.train.Saver()

        # Initialize all variables
        self.session.run(tf.global_variables_initializer())

    def average_gradients(self, tower_grads):
        # Average gradients from different GPUs
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, dim=0)
                grads.append(expanded_g)

            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, reduction_indices=0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def tower_loss(self, x, y_, z_):
        y_conv, z_conv = self.construct_net(x)
        # Calculate loss on policy head
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                    logits=y_conv)
        policy_loss = tf.reduce_mean(cross_entropy)

        # Loss on value head
        mse_loss = \
            tf.reduce_mean(tf.squared_difference(z_, z_conv))

        # Regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        reg_variables = tf.get_collection(tf.GraphKeys.WEIGHTS)
        reg_term = \
            tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        # For training from a (smaller) dataset of strong players, you will
        # want to reduce the factor in front of self.mse_loss here.
        pol_loss_w = self.cfg['training']['policy_loss_weight']
        val_loss_w = self.cfg['training']['value_loss_weight']
        loss = pol_loss_w * policy_loss + val_loss_w * mse_loss + reg_term

        return loss, policy_loss, mse_loss, reg_term, y_conv

    def assign(self, var, values):
        try:
            self.session.run(tf.assign(var, values))
        except:
            print("Failed to assign {}: var shape {}, values shape {}".format(
                var.name, var.shape, values.shape))
            raise

    def replace_weights(self, new_weights):
        for e, weights in enumerate(self.weights):
            if isinstance(weights, str):
                weights = tf.get_default_graph().get_tensor_by_name(weights)
            if weights.name.endswith('/batch_normalization/beta:0'):
                # Batch norm beta is written as bias before the batch
                # normalization in the weight file for backwards
                # compatibility reasons.
                bias = tf.constant(new_weights[e], shape=weights.shape)
                # Weight file order: bias, means, variances
                var = tf.constant(new_weights[e + 2], shape=weights.shape)
                new_beta = tf.divide(bias, tf.sqrt(var + tf.constant(1e-5)))
                self.assign(weights, new_beta)
            elif weights.shape.ndims == 4:
                # Rescale rule50 related weights as clients do not normalize the input.
                if e == 0:
                    num_inputs = 112
                    # 50 move rule is the 110th input, or 109 starting from 0.
                    rule50_input = 109
                    for i in range(len(new_weights[e])):
                        if (i%(num_inputs*9))//9 == rule50_input:
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
                self.assign(weights, tf.transpose(new_weight, [2, 3, 1, 0]))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.assign(weights, tf.transpose(new_weight, [1, 0]))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[e], shape=weights.shape)
                self.assign(weights, new_weight)

    def restore(self, filename):
        print("Restoring from {0}".format(filename))
        optimistic_restore(self.session, filename)

    def measure_loss(self, batch, training=False):
        # Measure loss over one batch. If training is true, also
        # accumulate the gradient and increment the global step.
        ops = [self.policy_loss, self.mse_loss, self.reg_term, self.accuracy]
        if training:
            ops += [self.grad_op, self.step_op],
        r = self.session.run(ops, feed_dict={self.training: training,
                           self.learning_rate: self.lr,
                           self.planes: batch[0],
                           self.probs: batch[1],
                           self.winner: batch[2]})
        # Google's paper scales mse by 1/4 to a [0,1] range, so we do the same here
        return {'policy': r[0], 'mse': r[1]/4., 'reg': r[2],
                'accuracy': r[3] * 100, 'total': r[0]+r[1]+r[2] }

    def process(self, train_data, test_data, num_evals):
        total_steps = self.cfg['training']['total_steps']
        info_steps = self.cfg['training']['info_steps']
        test_steps = self.cfg['training']['test_steps']
        lr1 = self.cfg['training']['lr1']
        lr2 = self.cfg['training']['lr2']
        lr_boundaries = self.cfg['training']['lr_boundaries']
        checkpoint_steps = self.cfg['training'].get('checkpoint_steps', total_steps)
        stats = Stats()
        timer = Timer()
        for i in range(total_steps):
            batch = next(train_data)
            stats.add({'lr': self.lr})
            # Measure losses and compute gradients for this batch.
            losses = self.measure_loss(batch, training=True)
            # fetch the current global step.
            steps = tf.train.global_step(self.session, self.global_step)
            mod_steps = steps % total_steps
            a1 = lr1[bisect.bisect_right(lr_boundaries, mod_steps)]
            a2 = lr2[bisect.bisect_right(lr_boundaries, mod_steps)]
            self.lr = self.learning_rate_(mod_steps + 1, self.swa_max_n, a1, a2)
            stats.add(losses)
            if steps % self.macrobatch == (self.macrobatch-1):
                # Apply the accumulated gradients to the weights.
                self.session.run([self.train_op], feed_dict={self.learning_rate: self.lr})
                # Clear the accumulated gradient.
                self.session.run([self.clear_op], feed_dict={self.learning_rate: self.lr})

            if steps % info_steps == 0:
                speed = info_steps * self.batch_size / timer.elapsed()
                print("step {}, policy={:g} mse={:g} reg={:g} total={:g} lr={:g} ({:g} pos/s)".format(
                    steps, stats.mean('policy'), stats.mean('mse'), stats.mean('reg'),
                    stats.mean('total'), self.lr, speed))
                summaries = stats.summaries({'Policy Loss': 'policy',
                    'MSE Loss': 'mse', 'LR': 'lr', 'Reg term': 'reg'})
                self.train_writer.add_summary(
                    tf.Summary(value=summaries), steps)
                stats.clear()

            if steps % test_steps == 0:
                test_stats = Stats()
                for _ in range(num_evals):
                    test_batch = next(test_data)
                    losses = self.measure_loss(test_batch, training=False)
                    test_stats.add(losses)
                summaries = test_stats.summaries({'Policy Loss': 'policy',
                                                  'MSE Loss': 'mse',
                                                  'Accuracy': 'accuracy'})
                self.test_writer.add_summary(tf.Summary(value=summaries), steps)
                print("step {}, policy={:g} training accuracy={:g}%, mse={:g}".\
                    format(steps, test_stats.mean('policy'),
                        test_stats.mean('accuracy'),
                        test_stats.mean('mse')))

            path = os.path.join(self.root_dir, self.cfg['name'])
            leela_path = path + "-" + str(steps)
            if steps % checkpoint_steps == 0:
                self.net.pb.training_params.training_steps = steps
                self.save_leelaz_weights(leela_path)
                print("Weights saved in file: {}".format(leela_path))
                save_path = self.saver.save(self.session, path,
                                            global_step=steps)
                print("Model saved in file: {}".format(save_path))

            if self.swa_enabled:
                self.save_swa_network(steps, path, leela_path, train_data)


    def save_leelaz_weights(self, filename):
        all_weights = []
        for e, weights in enumerate(self.weights):
            # Newline unless last line (single bias)
            work_weights = None
            if weights.name.endswith('/batch_normalization/beta:0'):
                # Batch norm beta needs to be converted to biases before
                # the batch norm for backwards compatibility reasons
                var_key = weights.name.replace('beta', 'moving_variance')
                var = tf.get_default_graph().get_tensor_by_name(var_key)
                work_weights = tf.multiply(weights, tf.sqrt(var + tf.constant(1e-5)))
            elif weights.shape.ndims == 4:
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
            nparray = work_weights.eval(session=self.session)
            # Rescale rule50 related weights as clients do not normalize the input.
            if e == 0:
                num_inputs = 112
                # 50 move rule is the 110th input, or 109 starting from 0.
                rule50_input = 109
                wt_flt = []
                for i, weight in enumerate(np.ravel(nparray)):
                    if (i%(num_inputs*9))//9 == rule50_input:
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

    def reset_batchnorm_key(self):
        self.batch_norm_count = 0
        self.reuse_var = True

    def add_weights(self, variable):
        if self.reuse_var is None:
            self.weights.append(variable)

    def batch_norm(self, net):
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        scope = self.get_batchnorm_key()
        with tf.variable_scope(scope):
            net = tf.layers.batch_normalization(
                    net,
                    epsilon=1e-5, axis=1, fused=True,
                    center=True, scale=False,
                    training=self.training,
                    reuse=self.reuse_var)

        for v in ['beta', 'moving_mean', 'moving_variance' ]:
            name = scope + '/batch_normalization/' + v + ':0'
            var = tf.get_default_graph().get_tensor_by_name(name)
            self.add_weights(var)

        return net

    def conv_block(self, inputs, filter_size, input_channels, output_channels, name):
        W_conv = weight_variable(
            name,
            [filter_size, filter_size, input_channels, output_channels])

        self.add_weights(W_conv)

        net = inputs
        net = conv2d(net, W_conv)
        net = self.batch_norm(net)
        net = tf.nn.relu(net)
        return net

    def residual_block(self, inputs, channels, name):
        net = inputs
        orig = tf.identity(net)

        # First convnet weights
        W_conv_1 = weight_variable(name + "_conv_1", [3, 3, channels, channels])
        self.add_weights(W_conv_1)

        net = conv2d(net, W_conv_1)
        net = self.batch_norm(net)
        net = tf.nn.relu(net)

        # Second convnet weights
        W_conv_2 = weight_variable(name + "_conv_2", [3, 3, channels, channels])
        self.add_weights(W_conv_2)

        net = conv2d(net, W_conv_2)
        net = self.batch_norm(net)
        net = tf.add(net, orig)
        net = tf.nn.relu(net)

        return net

    def construct_net(self, planes):
        # NCHW format
        # batch, 112 input channels, 8 x 8
        x_planes = tf.reshape(planes, [-1, 112, 8, 8])

        # Input convolution
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=112,
                               output_channels=self.RESIDUAL_FILTERS,
                               name="first_conv")
        # Residual tower
        for i in range(0, self.RESIDUAL_BLOCKS):
            block_name = "res_" + str(i)
            flow = self.residual_block(flow, self.RESIDUAL_FILTERS,
                                       name=block_name)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=32,
                                   name="policy_head")
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 32 * 8 * 8])
        W_fc1 = weight_variable("w_fc_1", [32 * 8 * 8, 1858])
        b_fc1 = bias_variable("b_fc_1", [1858])
        self.add_weights(W_fc1)
        self.add_weights(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1)

        # Value head
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=32,
                                   name="value_head")
        h_conv_val_flat = tf.reshape(conv_val, [-1, 32 * 8 * 8])
        W_fc2 = weight_variable("w_fc_2", [32 * 8 * 8, 256])
        b_fc2 = bias_variable("b_fc_2", [256])
        self.add_weights(W_fc2)
        self.add_weights(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable("w_fc_3", [256, 1])
        b_fc3 = bias_variable("b_fc_3", [1])
        self.add_weights(W_fc3)
        self.add_weights(b_fc3)
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))

        return h_fc1, h_fc3

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
            self.save_op = tf.group(*save_ops)
            self.restore_op = tf.group(*rest_ops)
        self.session.run(self.save_op)

    def snap_restore(self):
        # Restore variables in the current graph from the snapshot.
        self.session.run(self.restore_op)

    def save_swa_network(self, steps, path, leela_path, data):
        # Sample 1 in self.swa_c of the networks. Compute in this way so
        # that it's safe to change the value of self.swa_c
        rem = self.session.run(tf.assign_add(self.swa_skip, -1))
        if rem > 1:
            return
        self.swa_skip.load(self.swa_c, self.session)

        # Add the current weight vars to the running average.
        num = self.session.run(self.swa_accum_op)
        num = min(num, self.swa_max_n)
        self.swa_count.load(float(num), self.session)

        swa_path = path + "-swa-" + str(int(num)) + "-" + str(steps)

        # save the current network.
        self.snap_save()
        # Copy the swa weights into the current network.
        self.session.run(self.swa_load_op)
        if self.swa_recalc_bn:
            print("Refining SWA batch normalization")
            for _ in range(200):
                batch = next(data)
                self.session.run(
                    [self.loss, self.update_ops],
                    feed_dict={self.training: True,
                               self.learning_rate: self.lr,
                               self.planes: batch[0], self.probs: batch[1],
                               self.winner: batch[2]})

        self.save_leelaz_weights(swa_path)
        # restore the saved network.
        self.snap_restore()

        print("Wrote averaged network to {}".format(swa_path))

# Unit tests for TFProcess.
def gen_block(size, f_in, f_out):
    return [ [1.1] * size * size * f_in * f_out, # conv
             [-.1] * f_out,  # bias weights
             [-.2] * f_out,  # batch norm mean
             [-.3] * f_out ] # batch norm var

class TFProcessTest(unittest.TestCase):
    def test_can_replace_weights(self):
        tfprocess = TFProcess()
        tfprocess.init(batch_size=1)
        # use known data to test replace_weights() works.
        data = gen_block(3, 18, tfprocess.RESIDUAL_FILTERS) # input conv
        for _ in range(tfprocess.RESIDUAL_BLOCKS):
            data.extend(gen_block(3,
                tfprocess.RESIDUAL_FILTERS, tfprocess.RESIDUAL_FILTERS))
            data.extend(gen_block(3,
                tfprocess.RESIDUAL_FILTERS, tfprocess.RESIDUAL_FILTERS))
        # policy
        data.extend(gen_block(1, tfprocess.RESIDUAL_FILTERS, 2))
        data.append([0.4] * 2*19*19 * (19*19+1))
        data.append([0.5] * (19*19+1))
        # value
        data.extend(gen_block(1, tfprocess.RESIDUAL_FILTERS, 1))
        data.append([0.6] * 19*19 * 256)
        data.append([0.7] * 256)
        data.append([0.8] * 256)
        data.append([0.9] * 1)
        tfprocess.replace_weights(data)

if __name__ == '__main__':
    unittest.main()
