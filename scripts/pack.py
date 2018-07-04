#!/usr/bin/env python3

import glob
import os
import argparse
import gzip
import bz2
import struct
import numpy as np
from multiprocessing import Pool


RECORD_SIZE = 8276


def get_uncompressed_size(filename):
    with open(filename, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def get_sorted_chunk_ids(dirs):
    ids = []
    sizes = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "training.*.gz")):
            ids.append(int(os.path.basename(f).split('.')[-2]))
            sizes.append(get_uncompressed_size(f))
    I = np.argsort(ids)
    ids = np.array(ids)[I]
    sizes = np.array(sizes)[I]
    return ids, sizes


def pack(ids, sizes):
    total = np.sum(sizes)
    data = np.zeros(total, dtype=np.int8)

    begin = 0
    for i, tid in enumerate(ids):
        filename = os.path.join(argv.input, 'training.{}.gz'.format(tid))
        end = begin + sizes[i]
        with gzip.open(filename, 'rb') as f:
            f.readinto(data[begin:end])
        if argv.remove:
            os.remove(filename)
        begin = end

    data = data.reshape(RECORD_SIZE, -1)
    filename = os.path.join(argv.output, '{}-{}.bz2'.format(ids[0], ids[-1]))
    with bz2.open(filename, 'xb') as f:
        for row in data:
            f.write(row)
        plylist = (sizes // RECORD_SIZE).astype(np.int16)
        size = struct.pack('I', len(plylist)*2)
        f.write(plylist.tobytes())
        f.write(size)

    print("Written '{}' {}x{}".format(filename, data.shape[0], data.shape[1]))


def main():
    if not os.path.exists(argv.output):
        os.makedirs(argv.output)
        print("Created directory '{}'".format(argv.output))

    ids, sizes = get_sorted_chunk_ids([argv.input])
    n = len(ids) // argv.number
    m = argv.number
    print("Processing {} ids, {} - {} ({}x{})".format(len(ids), ids[0], ids[-1], n, m))
    packs = [(ids[i*m:i*m+m], sizes[i*m:i*m+m]) for i in range(n)]

    with Pool() as pool:
        pool.starmap(pack, packs)
                

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Repack training.*.gz files in batches of bz2 format.')
    argparser.add_argument('-i', '--input', type=str,
            help='input directory')
    argparser.add_argument('-o', '--output', type=str,
            help='output directory')
    argparser.add_argument('-r', '--remove', action='store_true',
            help='remove input files while processing')
    argparser.add_argument('-n', '--number', type=int, default=1000,
            help='number of games to repack per bz2 package')
    argv = argparser.parse_args()

    main()
