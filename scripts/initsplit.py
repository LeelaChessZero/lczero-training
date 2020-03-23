#!/usr/bin/env python

import glob
import os
import argparse


def get_sorted_chunk_ids(dirs):
    ids = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "training.*.gz")):
            ids.append(int(os.path.basename(f).split('.')[-2]))
    ids.sort(reverse=True)
    return ids


def main(argv):
    a = get_sorted_chunk_ids([argv.input])
    n = min(argv.wsize, len(a))
    for i in sorted(a[:n]):
        if i % 100 >= 90:
            os.link(os.path.join(argv.input, "training.{}.gz".format(i)),
                    os.path.join(argv.output, "test/training.{}.gz".format(i)))
        else:
            os.link(
                os.path.join(argv.input, "training.{}.gz".format(i)),
                os.path.join(argv.output, "train/training.{}.gz".format(i)))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Link input to test/train subdirectories of output in 10:90 ratio.')
    argparser.add_argument('-i', '--input', type=str, help='input directory')
    argparser.add_argument(
        '-w',
        '--wsize',
        type=int,
        help=
        'window size - should be padded a bit to ensure both sides of split exceed fraction of target'
    )
    argparser.add_argument('-o', '--output', type=str, help='output directory')

    main(argparser.parse_args())
