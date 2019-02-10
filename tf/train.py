#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
import yaml
import sys
import glob
import gzip
import random
import multiprocessing as mp
import tensorflow as tf
from tfprocess import TFProcess
from chunkparser import ChunkParser

SKIP = 32


def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")


def get_latest_chunks(path, num_chunks):
    chunks = []
    for d in glob.glob(path):
        chunks += get_chunks(d)

    if len(chunks) < num_chunks:
        print("Not enough chunks {}".format(len(chunks)))
        sys.exit(1)

    print("sorting {} chunks...".format(len(chunks)), end='')
    chunks.sort(key=os.path.getmtime, reverse=True)
    print("[done]")
    chunks = chunks[:num_chunks]
    print("{} - {}".format(os.path.basename(chunks[-1]), os.path.basename(chunks[0])))
    random.shuffle(chunks)
    return chunks


class FileDataSrc:
    """
        data source yielding chunkdata from chunk files.
    """
    def __init__(self, chunks):
        self.chunks = []
        self.done = chunks
    def next(self):
        if not self.chunks:
            self.chunks, self.done = self.done, self.chunks
            random.shuffle(self.chunks)
        if not self.chunks:
            return None
        while len(self.chunks):
            filename = self.chunks.pop()
            try:
                with gzip.open(filename, 'rb') as chunk_file:
                    self.done.append(filename)
                    return chunk_file.read()
            except:
                print("failed to parse {}".format(filename))


def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    num_chunks = cfg['dataset']['num_chunks']
    train_ratio = cfg['dataset']['train_ratio']
    num_train = int(num_chunks*train_ratio)
    num_test = num_chunks - num_train
    if 'input_test' in cfg['dataset']:
        train_chunks = get_latest_chunks(cfg['dataset']['input_train'], num_train)
        test_chunks = get_latest_chunks(cfg['dataset']['input_test'], num_test)
    else:
        chunks = get_latest_chunks(cfg['dataset']['input'], num_chunks)
        train_chunks = chunks[:num_train]
        test_chunks = chunks[num_train:]

    shuffle_size = cfg['training']['shuffle_size']
    total_batch_size = cfg['training']['batch_size']
    batch_splits = cfg['training'].get('num_batch_splits', 1)
    if total_batch_size % batch_splits != 0:
        raise ValueError('num_batch_splits must divide batch_size evenly')
    split_batch_size = total_batch_size // batch_splits
    # Load data with split batch size, which will be combined to the total batch size in tfprocess.
    ChunkParser.BATCH_SIZE = split_batch_size

    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    train_parser = ChunkParser(FileDataSrc(train_chunks),
            shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        train_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
    dataset = dataset.map(ChunkParser.parse_function)
    dataset = dataset.prefetch(4)
    train_iterator = dataset.make_one_shot_iterator()

    shuffle_size = int(shuffle_size*(1.0-train_ratio))
    test_parser = ChunkParser(FileDataSrc(test_chunks),
            shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        test_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
    dataset = dataset.map(ChunkParser.parse_function)
    dataset = dataset.prefetch(4)
    test_iterator = dataset.make_one_shot_iterator()

    tfprocess = TFProcess(cfg)
    tfprocess.init(dataset, train_iterator, test_iterator)

    if os.path.exists(os.path.join(root_dir, 'checkpoint')):
        cp = tf.train.latest_checkpoint(root_dir)
        tfprocess.restore(cp)

    # If number of test positions is not given
    # sweeps through all test chunks statistically
    # Assumes average of 10 samples per test game.
    # For simplicity, testing can use the split batch size instead of total batch size.
    # This does not affect results, because test results are simple averages that are independent of batch size.
    num_evals = cfg['training'].get('num_test_positions', num_test * 10)
    num_evals = max(1, num_evals // ChunkParser.BATCH_SIZE)
    print("Using {} evaluation batches".format(num_evals))

    tfprocess.process_loop(total_batch_size, num_evals, batch_splits=batch_splits)

    if cmd.output is not None:
        tfprocess.save_leelaz_weights(cmd.output)

    tfprocess.session.close()
    train_parser.shutdown()
    test_parser.shutdown()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Tensorflow pipeline for training Leela Chess.')
    argparser.add_argument('--cfg', type=argparse.FileType('r'),
        help='yaml configuration with training parameters')
    argparser.add_argument('--output', type=str,
        help='file to store weights in')

    mp.set_start_method('spawn')
    main(argparser.parse_args())
    mp.freeze_support()
