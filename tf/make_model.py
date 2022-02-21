#!/usr/bin/env python3
import argparse
import os
import yaml
import tfprocess

argparser = argparse.ArgumentParser(description='Convert net to model.')
argparser.add_argument('--start',
                       type=int,
                       default=0,
                       help='Offset to set global_step to.')
argparser.add_argument('--cfg',
                       type=argparse.FileType('r'),
                       help='yaml configuration with training parameters')
args = argparser.parse_args()
cfg = yaml.safe_load(args.cfg.read())
print(yaml.dump(cfg, default_flow_style=False))
START_FROM = args.start

tfp = tfprocess.TFProcess(cfg)
tfp.init_net()
tfp.global_step.assign(START_FROM)

root_dir = os.path.join(cfg['training']['path'], cfg['name'])
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
tfp.manager.save(checkpoint_number=START_FROM)
print("Wrote model to {}".format(tfp.manager.latest_checkpoint))
path = os.path.join(tfp.root_dir, tfp.cfg['name'])
leela_path = path + "-" + str(START_FROM)
swa_path = path + "-swa-" + str(START_FROM)
tfp.net.pb.training_params.training_steps = START_FROM
tfp.save_leelaz_weights(leela_path)
if tfp.swa_enabled:
    tfp.save_swa_weights(swa_path)
