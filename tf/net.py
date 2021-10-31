#!/usr/bin/env python3

import argparse
import gzip
import os
import numpy as np
import proto.net_pb2 as pb

LC0_MAJOR = 0
LC0_MINOR = 21
LC0_MINOR_WITH_INPUT_TYPE_3 = 25
LC0_MINOR_WITH_INPUT_TYPE_4 = 26
LC0_MINOR_WITH_INPUT_TYPE_5 = 27
LC0_PATCH = 0
WEIGHTS_MAGIC = 0x1c0


def nested_getattr(obj, attr):
    attributes = attr.split(".")
    for a in attributes:
        obj = getattr(obj, a)
    return obj


class Net:
    def __init__(self,
                 net=pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT,
                 input=pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE,
                 value=pb.NetworkFormat.VALUE_CLASSICAL,
                 policy=pb.NetworkFormat.POLICY_CLASSICAL,
                 moves_left=pb.NetworkFormat.MOVES_LEFT_V1):

        if net == pb.NetworkFormat.NETWORK_SE:
            net = pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT
        if net == pb.NetworkFormat.NETWORK_CLASSICAL:
            net = pb.NetworkFormat.NETWORK_CLASSICAL_WITH_HEADFORMAT

        self.pb = pb.Net()
        self.pb.magic = WEIGHTS_MAGIC
        self.pb.min_version.major = LC0_MAJOR
        self.pb.min_version.minor = LC0_MINOR
        self.pb.min_version.patch = LC0_PATCH
        self.pb.format.weights_encoding = pb.Format.LINEAR16

        self.weights = []

        self.set_networkformat(net)
        self.pb.format.network_format.input = input
        self.set_policyformat(policy)
        self.set_valueformat(value)
        self.set_movesleftformat(moves_left)

    def set_networkformat(self, net):
        self.pb.format.network_format.network = net

    def set_policyformat(self, policy):
        self.pb.format.network_format.policy = policy

    def set_valueformat(self, value):
        self.pb.format.network_format.value = value

        # OutputFormat is for search to know which kind of value the net returns.
        if value == pb.NetworkFormat.VALUE_WDL:
            self.pb.format.network_format.output = pb.NetworkFormat.OUTPUT_WDL
        else:
            self.pb.format.network_format.output = pb.NetworkFormat.OUTPUT_CLASSICAL

    def set_movesleftformat(self, moves_left):
        self.pb.format.network_format.moves_left = moves_left

    def set_input(self, input_format):
        self.pb.format.network_format.input = input_format
        if input_format == pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2 or input_format == pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON:
            self.pb.min_version.minor = LC0_MINOR_WITH_INPUT_TYPE_5
        elif input_format >= pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
            self.pb.min_version.minor = LC0_MINOR_WITH_INPUT_TYPE_4
        # Input type 2 was available before 3, but it was buggy, so also limit it to same version as 3.
        elif input_format != pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE:
            self.pb.min_version.minor = LC0_MINOR_WITH_INPUT_TYPE_3

    def fill_layer(self, layer, params):
        """Normalize and populate 16bit layer in protobuf"""
        params = params.flatten().astype(np.float32)
        layer.min_val = 0 if len(params) == 1 else float(np.min(params))
        layer.max_val = 1 if len(params) == 1 and np.max(
            params) == 0 else float(np.max(params))
        if layer.max_val == layer.min_val:
            # Avoid division by zero if max == min.
            params = (params - layer.min_val)
        else:
            params = (params - layer.min_val) / (layer.max_val - layer.min_val)
        params *= 0xffff
        params = np.round(params)
        layer.params = params.astype(np.uint16).tobytes()

    def denorm_layer(self, layer):
        """Denormalize a layer from protobuf"""
        params = np.frombuffer(layer.params, np.uint16).astype(np.float32)
        params /= 0xffff
        return params * (layer.max_val - layer.min_val) + layer.min_val

    def save_proto(self, filename):
        """Save weights gzipped protobuf file"""
        if len(filename.split('.')) == 1:
            filename += ".pb.gz"

        with gzip.open(filename, 'wb') as f:
            data = self.pb.SerializeToString()
            f.write(data)

        size = os.path.getsize(filename) / 1024**2
        print("Weights saved as '{}' {}M".format(filename, round(size, 2)))

    def tf_name_to_pb_name(self, name):
        """Given Tensorflow variable name returns the protobuf name and index
        of residual block if weight belong in a residual block."""
        def convblock_to_bp(w):
            w = w.split(':')[0]
            d = {
                'kernel': 'weights',
                'gamma': 'bn_gammas',
                'beta': 'bn_betas',
                'moving_mean': 'bn_means',
                'moving_variance': 'bn_stddivs',
                'bias': 'biases'
            }

            return d[w]

        def se_to_bp(l, w):
            if l == 'dense1':
                n = 1
            elif l == 'dense2':
                n = 2
            else:
                raise ValueError('Unable to decode SE-weight {}/{}'.format(
                    l, w))
            w = w.split(':')[0]
            d = {'kernel': 'w', 'bias': 'b'}

            return d[w] + str(n)

        def value_to_bp(l, w):
            if l == 'dense1':
                n = 1
            elif l == 'dense2':
                n = 2
            else:
                raise ValueError('Unable to decode value weight {}/{}'.format(
                    l, w))
            w = w.split(':')[0]
            d = {'kernel': 'ip{}_val_w', 'bias': 'ip{}_val_b'}

            return d[w].format(n)

        def policy_to_bp(w):
            w = w.split(':')[0]
            d = {'kernel': 'ip_pol_w', 'bias': 'ip_pol_b'}

            return d[w]

        def moves_left_to_bp(l, w):
            if l == 'dense1':
                n = 1
            elif l == 'dense2':
                n = 2
            else:
                raise ValueError(
                    'Unable to decode moves_left weight {}/{}'.format(l, w))
            w = w.split(':')[0]
            d = {'kernel': 'ip{}_mov_w', 'bias': 'ip{}_mov_b'}

            return d[w].format(n)

        layers = name.split('/')
        base_layer = layers[0]
        weights_name = layers[-1]
        pb_name = None
        block = None

        if base_layer == 'input':
            pb_name = 'input.' + convblock_to_bp(weights_name)
        elif base_layer == 'policy1':
            pb_name = 'policy1.' + convblock_to_bp(weights_name)
        elif base_layer == 'policy':
            if 'dense' in layers[1]:
                pb_name = policy_to_bp(weights_name)
            else:
                pb_name = 'policy.' + convblock_to_bp(weights_name)
        elif base_layer == 'value':
            if 'dense' in layers[1]:
                pb_name = value_to_bp(layers[1], weights_name)
            else:
                pb_name = 'value.' + convblock_to_bp(weights_name)
        elif base_layer == 'moves_left':
            if 'dense' in layers[1]:
                pb_name = moves_left_to_bp(layers[1], weights_name)
            else:
                pb_name = 'moves_left.' + convblock_to_bp(weights_name)
        elif base_layer.startswith('residual'):
            block = int(base_layer.split('_')[1]) - 1  # 1 indexed
            if layers[1] == '1':
                pb_name = 'conv1.' + convblock_to_bp(weights_name)
            elif layers[1] == '2':
                pb_name = 'conv2.' + convblock_to_bp(weights_name)
            elif layers[1] == 'se':
                pb_name = 'se.' + se_to_bp(layers[-2], weights_name)

        return (pb_name, block)

    def get_weights(self, names):
        # `names` is a list of Tensorflow tensor names to get from the protobuf.
        # Returns list of [Tensor name, Tensor weights].
        tensors = {}

        for tf_name in names:
            name = tf_name
            if 'stddev' in name:
                # Get variance instead of stddev.
                name = name.replace('stddev', 'variance')
            if 'renorm' in name:
                # Renorm variables are not populated.
                continue

            pb_name, block = self.tf_name_to_pb_name(name)

            if pb_name is None:
                raise ValueError(
                    "Don't know where to store weight in protobuf: {}".format(
                        name))

            if block == None:
                pb_weights = self.pb.weights
            else:
                pb_weights = self.pb.weights.residual[block]

            w = self.denorm_layer(nested_getattr(pb_weights, pb_name))

            # Only variance is stored in the protobuf.
            if 'stddev' in tf_name:
                w = np.sqrt(w + 1e-5)

            tensors[tf_name] = w
        return tensors

    def filters(self):
        layer = self.pb.weights.input.bn_means
        params = np.frombuffer(layer.params, np.uint16).astype(np.float32)
        return len(params)

    def blocks(self):
        return len(self.pb.weights.residual)

    def print_stats(self):
        print("Blocks: {}".format(self.blocks()))
        print("Filters: {}".format(self.filters()))
        print_pb_stats(self.pb)
        print()

    def parse_proto(self, filename):
        with gzip.open(filename, 'rb') as f:
            self.pb = self.pb.FromString(f.read())
        # Populate policyFormat and valueFormat fields in old protobufs
        # without these fields.
        if self.pb.format.network_format.network == pb.NetworkFormat.NETWORK_SE:
            self.set_networkformat(pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT)
            self.set_valueformat(pb.NetworkFormat.VALUE_CLASSICAL)
            self.set_policyformat(pb.NetworkFormat.POLICY_CLASSICAL)
            self.set_movesleftformat(pb.NetworkFormat.MOVES_LEFT_NONE)
        elif self.pb.format.network_format.network == pb.NetworkFormat.NETWORK_CLASSICAL:
            self.set_networkformat(
                pb.NetworkFormat.NETWORK_CLASSICAL_WITH_HEADFORMAT)
            self.set_valueformat(pb.NetworkFormat.VALUE_CLASSICAL)
            self.set_policyformat(pb.NetworkFormat.POLICY_CLASSICAL)
            self.set_movesleftformat(pb.NetworkFormat.MOVES_LEFT_NONE)

    def fill_net(self, all_weights):
        # all_weights is array of [name of weight, numpy array of weights].
        self.pb.format.weights_encoding = pb.Format.LINEAR16

        has_renorm = any('renorm' in w[0] for w in all_weights)
        weight_names = [w[0] for w in all_weights]

        del self.pb.weights.residual[:]

        for name, weights in all_weights:
            layers = name.split('/')
            weights_name = layers[-1]
            if weights.ndim == 4:
                # Convolution weights need a transpose
                #
                # TF
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela
                # [output, input, filter_size, filter_size]
                weights = np.transpose(weights, axes=[3, 2, 0, 1])
            elif weights.ndim == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                weights = np.transpose(weights, axes=[1, 0])

            if 'renorm' in name:
                # Batch renorm has extra weights, but we don't know what to do with them.
                continue
            if has_renorm:
                if 'variance:' in weights_name:
                    # Renorm has variance, but it is not the primary source of truth.
                    continue
                # Renorm has moving stddev not variance, undo the transform to make it compatible.
                if 'stddev:' in weights_name:
                    weights = np.square(weights) - 1e-5
                    name = name.replace('stddev', 'variance')

            if name == 'input/conv2d/kernel:0' and self.pb.format.network_format.input < pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
                # 50 move rule is the 110th input, or 109 starting from 0.
                weights[:, 109, :, :] /= 99

            pb_name, block = self.tf_name_to_pb_name(name)

            if pb_name is None:
                raise ValueError(
                    "Don't know where to store weight in protobuf: {}".format(
                        name))

            if block == None:
                pb_weights = self.pb.weights
            else:
                assert block >= 0
                while block >= len(self.pb.weights.residual):
                    self.pb.weights.residual.add()
                pb_weights = self.pb.weights.residual[block]

            self.fill_layer(nested_getattr(pb_weights, pb_name), weights)

            if pb_name.endswith('bn_betas'):
                # Check if we need to add constant one gammas.
                gamma_name = name.replace('beta', 'gamma')
                if gamma_name in weight_names:
                    continue
                gamma = np.ones(weights.shape)
                pb_gamma = pb_name.replace('bn_betas', 'bn_gammas')
                self.fill_layer(nested_getattr(pb_weights, pb_gamma), gamma)


def print_pb_stats(obj, parent=None):
    for descriptor in obj.DESCRIPTOR.fields:
        value = getattr(obj, descriptor.name)
        if descriptor.name == "weights":
            return
        if descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                map(print_pb_stats, value)
            else:
                print_pb_stats(value, obj)
        elif descriptor.type == descriptor.TYPE_ENUM:
            enum_name = descriptor.enum_type.values[value].name
            print("%s: %s" % (descriptor.full_name, enum_name))
        else:
            print("%s: %s" % (descriptor.full_name, value))


def main(argv):
    net = Net()
    net.parse_proto(argv.input)
    net.print_stats()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Print net stats')
    argparser.add_argument('-i',
                           '--input',
                           type=str,
                           help='input network weight file')
    main(argparser.parse_args())
