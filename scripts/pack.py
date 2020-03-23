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
    for d in dirs:
        for f in glob.glob(os.path.join(d, "training.*.gz")):
            ids.append(int(os.path.basename(f).split('.')[-2]))
    ids.sort()
    return ids


def pack(ids):
    plies = []
    fout_name = os.path.join(argv.output, '{}-{}.bz2'.format(ids[0], ids[-1]))
    with bz2.open(fout_name, 'xb') as fout:
        for tid in ids:
            fin_name = os.path.join(argv.input, 'training.{}.gz'.format(tid))
            plies.append(get_uncompressed_size(fin_name) // RECORD_SIZE)
            with gzip.open(fin_name, 'rb') as fin:
                fout.write(fin.read())
            if argv.remove:
                os.remove(fin_name)

        plylist = np.array(plies, dtype=np.int16)
        size = struct.pack('I', len(plylist) * 2)
        fout.write(plylist.tobytes())
        fout.write(size)

    print("Written '{}' {} records".format(fout_name, np.sum(plies)))


def main():
    if not os.path.exists(argv.output):
        os.makedirs(argv.output)
        print("Created directory '{}'".format(argv.output))

    ids = get_sorted_chunk_ids([argv.input])
    n = len(ids) // argv.number
    m = argv.number
    print("Processing {} ids, {} - {} ({}x{})".format(len(ids), ids[0],
                                                      ids[-1], n, m))
    packs = [ids[i * m:i * m + m] for i in range(n)]

    # add remaining ids to last pack
    packs[-1] += ids[n * m + m:]

    with Pool() as pool:
        pool.map(pack, packs)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Repack training.*.gz files in batches of bz2 format.')
    argparser.add_argument('-i', '--input', type=str, help='input directory')
    argparser.add_argument('-o', '--output', type=str, help='output directory')
    argparser.add_argument('-r',
                           '--remove',
                           action='store_true',
                           help='remove input files while processing')
    argparser.add_argument('-n',
                           '--number',
                           type=int,
                           default=1000,
                           help='number of games to repack per bz2 package')
    argv = argparser.parse_args()

    main()
