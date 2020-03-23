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
    b = get_sorted_chunk_ids(argv.dirs)
    n = min(argv.wsize, len(a))
    diff = set(a[:n]) - set(b)
    for i in sorted(diff):
        print('training.{}.gz'.format(i))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Print diffset of input dir and output dirs.')
    argparser.add_argument('-i', '--input', type=str, help='input directory')
    argparser.add_argument('-w', '--wsize', type=int, help='window size')
    argparser.add_argument('dirs', nargs='+', help='output directories')

    main(argparser.parse_args())
