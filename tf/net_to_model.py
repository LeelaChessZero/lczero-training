#!/usr/bin/env python3
import argparse
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

tfp = tfprocess.TFProcess(cfg)
tfp.init_net_v2()
tfp.replace_weights_v2(args.net)
tfp.global_step.assign(START_FROM)

root_dir = os.path.join(cfg['training']['path'], cfg['name'])
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
tfp.manager.save(checkpoint_number=START_FROM)
print("Wrote model to {}".format(tfp.manager.latest_checkpoint))
