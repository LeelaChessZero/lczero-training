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
        reader = tf.train.NewCheckpointReader(cp)
        saved_shapes = reader.get_variable_to_shape_map()
        new_names = sorted(
            [var.name.split(':')[0] for var in tf.global_variables()
             if var.name.split(':')[0] not in saved_shapes])
        for saved_var_name in new_names:
            print("New name {} will use default value".format(saved_var_name))
        var_names = sorted(
            [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
             if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        restore_names = []
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_default_graph().get_tensor_by_name(var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
                restore_names.append(saved_var_name)
            else:
                print("Dropping {} due to shape change".format(saved_var_name))
        legacy_names = sorted(
            [name for name in saved_shapes.keys()
             if name not in restore_names])
        for saved_var_name in legacy_names:
            print("Dropping {} as no longer used".format(saved_var_name))
        opt_saver = tf.train.Saver(restore_vars)
        opt_saver.restore(tfprocess.session, cp)
    else:
        print("No checkpoint to upgrade!")
        exit(1)

    steps = tf.train.global_step(tfprocess.session, tfprocess.global_step)
    path = os.path.join(root_dir, cfg['name'])
    save_path = tfprocess.saver.save(tfprocess.session, path, global_step=steps)
    tfprocess.session.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Convert current checkpoint to new training script or incompatible training parameters.')
    argparser.add_argument('--cfg', type=argparse.FileType('r'),
        help='yaml configuration with training parameters')

    main(argparser.parse_args())
