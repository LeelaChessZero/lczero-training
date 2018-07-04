#!/usr/bin/env python3

import os
import argparse
import gzip
import bz2
import numpy as np
from pack import RECORD_SIZE


def unpack(filepath):
    front, back = os.path.basename(filepath).split('-')
    back = back.split('.')[0]
    first = int(front)
    last = int(back)
    num_chunks = last - first + 1

    buf = bz2.BZ2File(filepath, 'rb').read()
    data = np.frombuffer(buf, dtype=np.int8).reshape(-1, RECORD_SIZE)
    records_per_chunk = data.shape[0] // (num_chunks)

    for i in range(first, last + 1):
        filename = os.path.join(argv.output, "training.{}.gz".format(i))
        with gzip.open(filename, 'wb') as f:
            begin = (i - first) * records_per_chunk
            for row in data[begin:begin+records_per_chunk]:
                f.write(row)

    # append remaining records to last chunk
    with gzip.open(filename, 'ab') as f:
        for row in data[begin+records_per_chunk:]:
            f.write(row)
        print("Appending {}".format(data[begin+records_per_chunk:].shape))

    print("Written {} chunks with {} records per chunk".format(num_chunks, records_per_chunk))


def main():
    if not os.path.exists(argv.output):
        os.makedirs(argv.output)
        print("Created directory '{}'".format(argv.output))

    unpack(argv.input)
                

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Unpack *-*.bz2 file into gz chunks.')
    argparser.add_argument('-i', '--input', type=str,
            help='input file')
    argparser.add_argument('-o', '--output', type=str,
            help='output directory')
    argv = argparser.parse_args()

    main()
