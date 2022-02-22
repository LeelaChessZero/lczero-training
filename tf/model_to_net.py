#!/usr/bin/env python3
import argparse
import os
import yaml
import tfprocess

argparser = argparse.ArgumentParser(description='Convert model to net.')
argparser.add_argument('--cfg',
                       type=argparse.FileType('r'),
                       help='yaml configuration with training parameters')
args = argparser.parse_args()
cfg = yaml.safe_load(args.cfg.read())
print(yaml.dump(cfg, default_flow_style=False))

tfp = tfprocess.TFProcess(cfg)
tfp.init_net()

tfp.restore()

root_dir = os.path.join(cfg['training']['path'], cfg['name'])
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
path = os.path.join(tfp.root_dir, tfp.cfg['name'])
steps = tfp.global_step.read_value().numpy()
leela_path = path + "-" + str(steps)
swa_path = path + "-swa-" + str(steps)
tfp.net.pb.training_params.training_steps = steps
tfp.save_leelaz_weights(leela_path)
if tfp.swa_enabled:
    tfp.save_swa_weights(swa_path)

