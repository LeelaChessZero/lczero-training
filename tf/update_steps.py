#!/usr/bin/env python3
import argparse
import os
import yaml
import sys
import tensorflow as tf
from tfprocess import TFProcess

START_FROM = 0

def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    x = [
        tf.placeholder(tf.float32, [None, 112, 8*8]),
        tf.placeholder(tf.float32, [None, 1858]),
        tf.placeholder(tf.float32, [None, 3]),
        tf.placeholder(tf.float32, [None, 3]),
    ]

    tfprocess = TFProcess(cfg)
    tfprocess.init_net(x)

    if os.path.exists(os.path.join(root_dir, 'checkpoint')):
        cp = tf.train.latest_checkpoint(root_dir)
        tfprocess.restore(cp)

    START_FROM = cmd.start

    update_global_step = tfp.global_step.assign(START_FROM)
    tfp.session.run(update_global_step)
    save_path = tfp.saver.save(tfp.session, root_dir, global_step=START_FROM)

    tfprocess.session.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Convert current checkpoint to new step count.')
    argparser.add_argument('--cfg', type=argparse.FileType('r'),
        help='yaml configuration with training parameters')
    argparser.add_argument('--start', type=int, default=0,
        help='Offset to set global_step to.')

    main(argparser.parse_args())
