#!/usr/bin/python3
import gzip
import sys
import glob
import os
import random
from multiprocessing import Pool
import tqdm

merge_files = 100
processes = 8
shuffle = True
record_length = 8292


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def positions(chunk):
    pos = []
    i = 0
    while record_length * i < len(chunk):
        pos.append(chunk[record_length * i:record_length * (i + 1)])
        i += 1
    return pos


def shuffle(files):
    data = []
    for filename in files:
        with gzip.open(filename, 'rb') as f:
            data.extend(positions(f.read()))
    if shuffle:
        random.shuffle(data)
    for d in data:
        if d[0] != 0x04:
            print(files)
            raise ValueError('Wrong training data format, not V4')
    new_file = list(os.path.splitext(files[0]))
    new_file[0] += '_shuffled'
    new_file = ''.join(new_file)
    new_file_temp = new_file + '.temp'
    with gzip.open(new_file_temp, 'wb', compresslevel=9) as f:
        for d in data:
            f.write(d)
    # For interrupt safety, make sure not to write partial chunks.
    os.rename(new_file_temp, new_file)
    for filename in files:
        os.remove(filename)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Expected one argument, got {}'.format(len(sys.argv) - 1))
        exit(1)
    s = sys.argv[1]
    files = glob.glob(os.path.join(sys.argv[1], '*.gz'))
    files = [f for f in files if '_shuffled' not in f]
    print('Found {} files'.format(len(files)))
    if len(files) == 0:
        exit(1)
    files = split(files, len(files) // merge_files)
    pool = Pool(processes)

    for _ in tqdm.tqdm(pool.imap_unordered(shuffle, files), total=len(files)):
        pass
