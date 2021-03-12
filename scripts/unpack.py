#!/usr/bin/env python3

import os
import argparse
import gzip
import bz2
import numpy as np
import pickle
import struct
from pack import RECORD_SIZE


def unpack(filepath):
    front, back = os.path.basename(filepath).split('-')
    back = back.split('.')[0]
    first = int(front)
    last = int(back)
    num_chunks = last - first + 1

    buf = bz2.BZ2File(filepath, 'rb').read()
    size = struct.unpack('I', buf[-4:])[0]
    plylist = np.frombuffer(buf[-4 - size:-4], dtype=np.int16)
    data = np.frombuffer(buf[:-4 - size],
                         dtype=np.int8).reshape(-1, RECORD_SIZE)
    assert (num_chunks == len(plylist))

    begin = 0
    for i, plies in enumerate(plylist):
        end = begin + plies
        filename = os.path.join(argv.output,
                                "training.{}.gz".format(i + first))

        with gzip.open(filename, 'wb') as f:
            for row in data[begin:end]:
                f.write(row)

        begin = end

    print("Written {} chunks".format(num_chunks))


def main():
    if not os.path.exists(argv.output):
        os.makedirs(argv.output)
        print("Created directory '{}'".format(argv.output))

    unpack(argv.input)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Unpack *-*.bz2 file into gz chunks.')
    argparser.add_argument('-i', '--input', type=str, help='input file')
    argparser.add_argument('-o', '--output', type=str, help='output directory')
    argv = argparser.parse_args()

    main()
