#!/usr/bin/env python3
import argparse
import tensorflow as tf
import os
import yaml
import tfprocess
from net import Net

argparser = argparse.ArgumentParser(description='Convert net to model.')
argparser.add_argument('net', type=str,
    help='Net file to be converted to a model checkpoint.')
argparser.add_argument('--start', type=int, default=0,
    help='Offset to set global_step to.')
argparser.add_argument('--cfg', type=argparse.FileType('r'),
    help='yaml configuration with training parameters')
args = argparser.parse_args()
cfg = yaml.safe_load(args.cfg.read())
print(yaml.dump(cfg, default_flow_style=False))
START_FROM = args.start
net = Net()
net.parse_proto(args.net)

filters, blocks = net.filters(), net.blocks()
if cfg['model']['filters'] != filters:
    raise ValueError("Number of filters in YAML doesn't match the network")
if cfg['model']['residual_blocks'] != blocks:
    raise ValueError("Number of blocks in YAML doesn't match the network")
weights = net.get_weights()

x = [
    tf.placeholder(tf.float32, [None, 112, 8*8]),
    tf.placeholder(tf.float32, [None, 1858]),
    tf.placeholder(tf.float32, [None, 1])
    ]

tfp = tfprocess.TFProcess(cfg)
tfp.init_net(x)
tfp.replace_weights(weights)
update_global_step = tfp.global_step.assign(START_FROM)
tfp.session.run(update_global_step)

root_dir = os.path.join(cfg['training']['path'], cfg['name'])
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
path = os.path.join(root_dir, cfg['name'])
save_path = tfp.saver.save(tfp.session, path, global_step=START_FROM)
print("Wrote model to {}".format(root_dir))
