#!/usr/bin/env python3

import argparse
import gzip
import os
import numpy as np
import proto.net_pb2 as pb

LC0_MAJOR = 0
LC0_MINOR = 21
LC0_PATCH = 0
WEIGHTS_MAGIC = 0x1c0


class Net:
    def __init__(self,
                 net=pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT,
                 input=pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE,
                 value=pb.NetworkFormat.VALUE_CLASSICAL,
                 policy=pb.NetworkFormat.POLICY_CLASSICAL):

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
        self.set_policyformat(value)
        self.set_valueformat(value)

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

    def get_weight_amounts(self):
        value_weights = 8
        policy_weights = 6
        head_weights = value_weights + policy_weights
        if self.pb.format.network_format.network == pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT:
            # Batch norm gammas in head convolutions.
            head_weights += 2
        if self.pb.format.network_format.network == pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT:
            return {"input": 5, "residual": 14, "head": head_weights}
        else:
            return {"input": 4, "residual": 8, "head": head_weights}

    def fill_layer(self, layer, weights):
        """Normalize and populate 16bit layer in protobuf"""
        params = np.array(weights.pop(), dtype=np.float32)
        layer.min_val = 0 if len(params) == 1 else float(np.min(params))
        layer.max_val = 1 if len(params) == 1 and np.max(params) == 0 else float(np.max(params))
        if layer.max_val == layer.min_val:
            # Avoid division by zero if max == min.
            params = (params - layer.min_val)
        else:
            params = (params - layer.min_val) / (layer.max_val - layer.min_val)
        params *= 0xffff
        params = np.round(params)
        layer.params = params.astype(np.uint16).tobytes()

    def fill_conv_block(self, convblock, weights, gammas):
        """Normalize and populate 16bit convblock in protobuf"""
        if gammas:
            self.fill_layer(convblock.bn_stddivs, weights)
            self.fill_layer(convblock.bn_means, weights)
            self.fill_layer(convblock.bn_betas, weights)
            self.fill_layer(convblock.bn_gammas, weights)
            self.fill_layer(convblock.weights, weights)
        else:
            self.fill_layer(convblock.bn_stddivs, weights)
            self.fill_layer(convblock.bn_means, weights)
            self.fill_layer(convblock.biases, weights)
            self.fill_layer(convblock.weights, weights)

    def fill_plain_conv(self, convblock, weights):
        """Normalize and populate 16bit convblock in protobuf"""
        self.fill_layer(convblock.biases, weights)
        self.fill_layer(convblock.weights, weights)

    def fill_se_unit(self, se_unit, weights):
        self.fill_layer(se_unit.b2, weights)
        self.fill_layer(se_unit.w2, weights)
        self.fill_layer(se_unit.b1, weights)
        self.fill_layer(se_unit.w1, weights)

    def denorm_layer(self, layer, weights):
        """Denormalize a layer from protobuf"""
        params = np.frombuffer(layer.params, np.uint16).astype(np.float32)
        params /= 0xffff
        weights.insert(0, params * (layer.max_val - layer.min_val) + layer.min_val)

    def denorm_conv_block(self, convblock, weights):
        """Denormalize a convblock from protobuf"""
        se = self.pb.format.network_format.network == pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT

        if se:
            self.denorm_layer(convblock.bn_stddivs, weights)
            self.denorm_layer(convblock.bn_means, weights)
            self.denorm_layer(convblock.bn_betas, weights)
            self.denorm_layer(convblock.bn_gammas, weights)
            self.denorm_layer(convblock.weights, weights)
        else:
            self.denorm_layer(convblock.bn_stddivs, weights)
            self.denorm_layer(convblock.bn_means, weights)
            self.denorm_layer(convblock.biases, weights)
            self.denorm_layer(convblock.weights, weights)

    def denorm_plain_conv(self, convblock, weights):
        """Denormalize a plain convolution from protobuf"""
        self.denorm_layer(convblock.biases, weights)
        self.denorm_layer(convblock.weights, weights)

    def denorm_se_unit(self, convblock, weights):
        """Denormalize SE-unit from protobuf"""
        se = self.pb.format.network_format.network == pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT

        assert se

        self.denorm_layer(convblock.b2, weights)
        self.denorm_layer(convblock.w2, weights)
        self.denorm_layer(convblock.b1, weights)
        self.denorm_layer(convblock.w1, weights)

    def save_txt(self, filename):
        """Save weights as txt file"""
        weights = self.get_weights()

        if len(filename.split('.')) == 1:
            filename += ".txt.gz"

        # Legacy .txt files are version 2, SE is version 3.

        version = 2
        if self.pb.format.network_format.network == pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT:
            version = 3

        if self.pb.format.network_format.policy == pb.NetworkFormat.POLICY_CONVOLUTION:
            version = 4

        with gzip.open(filename, 'wb') as f:
            f.write("{}\n".format(version).encode('utf-8'))
            for row in weights:
                f.write((" ".join(map(str, row.tolist())) + "\n").encode('utf-8'))

        size = os.path.getsize(filename) / 1024**2
        print("saved as '{}' {}M".format(filename, round(size, 2)))

    def save_proto(self, filename):
        """Save weights gzipped protobuf file"""
        if len(filename.split('.')) == 1:
            filename += ".pb.gz"

        with gzip.open(filename, 'wb') as f:
            data = self.pb.SerializeToString()
            f.write(data)

        size = os.path.getsize(filename) / 1024**2
        print("saved as '{}' {}M".format(filename, round(size, 2)))

    def get_weights(self):
        """Returns the weights as floats per layer"""
        se = self.pb.format.network_format.network == pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT
        if self.weights == []:
            self.denorm_layer(self.pb.weights.ip2_val_b, self.weights)
            self.denorm_layer(self.pb.weights.ip2_val_w, self.weights)
            self.denorm_layer(self.pb.weights.ip1_val_b, self.weights)
            self.denorm_layer(self.pb.weights.ip1_val_w, self.weights)
            self.denorm_conv_block(self.pb.weights.value, self.weights)

            if self.pb.format.network_format.policy == pb.NetworkFormat.POLICY_CONVOLUTION:
                self.denorm_plain_conv(self.pb.weights.policy, self.weights)
                self.denorm_conv_block(self.pb.weights.policy1, self.weights)
            else:
                self.denorm_layer(self.pb.weights.ip_pol_b, self.weights)
                self.denorm_layer(self.pb.weights.ip_pol_w, self.weights)
                self.denorm_conv_block(self.pb.weights.policy, self.weights)

            for res in reversed(self.pb.weights.residual):
                if se:
                    self.denorm_se_unit(res.se, self.weights)
                self.denorm_conv_block(res.conv2, self.weights)
                self.denorm_conv_block(res.conv1, self.weights)

            self.denorm_conv_block(self.pb.weights.input, self.weights)

        return self.weights

    def filters(self):
        w = self.get_weights()
        return len(w[1])

    def blocks(self):
        w = self.get_weights()

        ws = self.get_weight_amounts()
        blocks = len(w) - (ws['input'] + ws['head'])

        if blocks % ws['residual'] != 0:
            raise ValueError("Inconsistent number of weights in the file")

        return blocks // ws['residual']

    def parse_proto(self, filename):
        with gzip.open(filename, 'rb') as f:
            self.pb = self.pb.FromString(f.read())

    def parse_txt(self, filename):
        weights = []

        with open(filename, 'r') as f:
            try:
                version = int(f.readline()[0])
            except:
                raise ValueError('Unable to read version.')
            for e, line in enumerate(f):
                weights.append(list(map(float, line.split(' '))))

        if version == 3:
            self.set_networkformat(pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT)

        if version == 4:
            self.set_networkformat(pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT)
            self.set_policyformat(pb.NetworkFormat.POLICY_CONVOLUTION)

        self.fill_net(weights)

    def fill_net(self, weights):
        self.weights = []
        # Batchnorm gammas in ConvBlock?
        se = self.pb.format.network_format.network == pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT
        gammas = se

        ws = self.get_weight_amounts()

        blocks = len(weights) - (ws['input'] + ws['head'])

        if blocks % ws['residual'] != 0:
            raise ValueError("Inconsistent number of weights in the file")
        blocks //= ws['residual']

        self.pb.format.weights_encoding = pb.Format.LINEAR16
        self.fill_layer(self.pb.weights.ip2_val_b, weights)
        self.fill_layer(self.pb.weights.ip2_val_w, weights)
        self.fill_layer(self.pb.weights.ip1_val_b, weights)
        self.fill_layer(self.pb.weights.ip1_val_w, weights)
        self.fill_conv_block(self.pb.weights.value, weights, gammas)

        if self.pb.format.network_format.policy == pb.NetworkFormat.POLICY_CONVOLUTION:
            self.fill_plain_conv(self.pb.weights.policy, weights)
            self.fill_conv_block(self.pb.weights.policy1, weights, gammas)
        else:
            self.fill_layer(self.pb.weights.ip_pol_b, weights)
            self.fill_layer(self.pb.weights.ip_pol_w, weights)
            self.fill_conv_block(self.pb.weights.policy, weights, gammas)

        del self.pb.weights.residual[:]
        tower = []
        for i in range(blocks):
            tower.append(self.pb.weights.residual.add())

        for res in reversed(tower):
            if se:
                self.fill_se_unit(res.se, weights)
            self.fill_conv_block(res.conv2, weights, gammas)
            self.fill_conv_block(res.conv1, weights, gammas)

        self.fill_conv_block(self.pb.weights.input, weights, gammas)


def main(argv):
    net = Net(net=pb.NetworkFormat.NETWORK_CLASSICAL)

    if argv.input.endswith(".txt"):
        print('Found .txt network')
        net.parse_txt(argv.input)
        print("Blocks: {}".format(net.blocks()))
        print("Filters: {}".format(net.filters()))
        if argv.output == None:
            argv.output = argv.input.replace('.txt', '.pb.gz')
            assert argv.output.endswith('.pb.gz')
            print('Writing output to: {}'.format(argv.output))
        net.save_proto(argv.output)
    elif argv.input.endswith(".pb.gz"):
        print('Found .pb.gz network')
        net.parse_proto(argv.input)
        print("Blocks: {}".format(net.blocks()))
        print("Filters: {}".format(net.filters()))
        if argv.output == None:
            argv.output = argv.input.replace('.pb.gz', '.txt.gz')
            print('Writing output to: {}'.format(argv.output))
            assert argv.output.endswith('.txt.gz')
        net.save_txt(argv.output)
    else:
        print('Unable to detect the network format. '
              'Filename should end in ".txt" or ".pb.gz"')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Convert network textfile to proto.')
    argparser.add_argument('-i', '--input', type=str,
                           help='input network weight text file')
    argparser.add_argument('-o', '--output', type=str,
                           help='output filepath without extension')
    main(argparser.parse_args())
