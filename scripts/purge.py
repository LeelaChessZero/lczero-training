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
    for i in a[n:]:
        os.remove(os.path.join(argv.input, "training.{}.gz".format(i)))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Delete from input not in window.')
    argparser.add_argument('-i', '--input', type=str, help='input directory')
    argparser.add_argument('-w', '--wsize', type=int, help='window size')

    main(argparser.parse_args())
