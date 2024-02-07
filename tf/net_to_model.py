#!/usr/bin/env python3
import argparse
import os
import yaml
import tfprocess

def convert(include_attn_wts_output=True, rescale_rule50=True):
    argparser = argparse.ArgumentParser(description='Convert net to model.')
    argparser.add_argument('net',
                        type=str,
                        help='Net file to be converted to a model checkpoint.')
    argparser.add_argument('--start',
                        type=int,
                        default=0,
                        help='Offset to set global_step to.')
    argparser.add_argument('--cfg',
                        type=argparse.FileType('r'),
                        help='yaml configuration with training parameters')
    argparser.add_argument('-e',
                        '--ignore-errors',
                        action='store_true',
                        help='Ignore missing and wrong sized values.')
    args = argparser.parse_args()
    cfg = yaml.safe_load(args.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))
    START_FROM = args.start

    tfp = tfprocess.TFProcess(cfg)
    tfp.init_net(include_attn_wts_output)
    tfp.replace_weights(args.net, args.ignore_errors, rescale_rule50)
    tfp.global_step.assign(START_FROM)

    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    tfp.manager.save(checkpoint_number=START_FROM)
    print("Wrote model to {}".format(tfp.manager.latest_checkpoint))
    return args, root_dir, tfp

if __name__ == "__main__":
    convert()
