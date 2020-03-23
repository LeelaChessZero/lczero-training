#!/usr/bin/env python

import glob
import os
import argparse


def get_sorted_chunk_ids(dirs):
    ids = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "training.*.gz")):
            ids.append(int(os.path.basename(f).split('.')[-2]))
    ids.sort()
    return ids


def main(argv):
    a = get_sorted_chunk_ids([argv.input])
    for i in a:
        os.utime(os.path.join(argv.input, "training.{}.gz".format(i)), None)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Change modification time on training files to match their numeric order.')
    argparser.add_argument('-i', '--input', type=str, help='input directory')

    main(argparser.parse_args())
