#!/usr/bin/env python3

import glob
import os
import argparse
import gzip
import bz2
from multiprocessing import Pool


def get_sorted_chunk_ids(dirs):
    ids = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "training.*.gz")):
            ids.append(int(os.path.basename(f).split('.')[-2]))
    ids.sort(reverse=True)
    return ids


def pack(ids):
    fout_name = os.path.join(argv.output, '{}-{}.bz2'.format(ids[-1], ids[0]))

    with bz2.open(fout_name, 'xb') as fout:
        for tid in reversed(ids):
            fin_name = os.path.join(argv.input, 'training.{}.gz'.format(tid))
            with gzip.open(fin_name, 'rb') as fin:
                fout.write(fin.read())

    print("Written '{}'".format(fout_name))


def main():
    if not os.path.exists(argv.output):
        os.makedirs(argv.output)
        print("Created directory '{}'".format(argv.output))

    ids = get_sorted_chunk_ids([argv.input])
    print("Processing {} ids, {} - {}".format(len(ids), ids[-1], ids[0]))
    n = len(ids) // argv.number
    packs = [ids[i*argv.number:i*argv.number+argv.number] for i in range(n)]
    pool = Pool()
    pool.map(pack, packs)
                

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Repack training.*.gz files in batches of bz2 format.')
    argparser.add_argument('-i', '--input', type=str,
            help='input directory')
    argparser.add_argument('-n', '--number', type=int,
            help='number of games to repack per bz2 package')
    argparser.add_argument('-o', '--output', type=str,
            help='output directory')
    argv = argparser.parse_args()

    main()
