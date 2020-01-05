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

def get_all_chunks(path):
    chunks = []
    for d in glob.glob(path):
        chunks += get_chunks(d)
    return chunks

def get_latest_chunks(path, num_chunks, allow_less):
    chunks = get_all_chunks(path)
    if len(chunks) < num_chunks:
        if allow_less:
            print("sorting {} chunks...".format(len(chunks)), end='')
            chunks.sort(key=os.path.getmtime, reverse=True)
            print("[done]")
            print("{} - {}".format(os.path.basename(chunks[-1]), os.path.basename(chunks[0])))
            random.shuffle(chunks)
            return chunks
        else:
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

def extract_inputs_outputs(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 7432 are easy, policy extraction.
    policy = tf.io.decode_raw(tf.strings.substr(raw, 4, 7432), tf.float32)
    # Next are 104 bit packed chess boards, they have to be expanded.
    bit_planes = tf.expand_dims(tf.reshape(tf.io.decode_raw(tf.strings.substr(raw, 7436, 832), tf.uint8), [-1, 104, 8]), -1)
    bit_planes = tf.bitwise.bitwise_and(tf.tile(bit_planes, [1, 1, 1, 8]), [128, 64, 32, 16, 8, 4, 2, 1])
    bit_planes = tf.minimum(1., tf.cast(bit_planes, tf.float32))
    # Next 5 planes are 1 or 0 to indicate 8x8 of 1 or 0.
    unit_planes = tf.expand_dims(tf.expand_dims(tf.io.decode_raw(tf.strings.substr(raw, 8268, 5), tf.uint8), -1), -1)
    unit_planes = tf.cast(tf.tile(unit_planes, [1, 1, 8, 8]), tf.float32)
    # rule50 count plane.
    rule50_plane = tf.expand_dims(tf.expand_dims(tf.io.decode_raw(tf.strings.substr(raw, 8273, 1), tf.uint8), -1), -1)
    rule50_plane = tf.cast(tf.tile(rule50_plane, [1, 1, 8, 8]), tf.float32)
    rule50_plane = tf.divide(rule50_plane, 99.)
    # zero plane and one plane
    zero_plane = tf.zeros_like(rule50_plane)
    one_plane = tf.ones_like(rule50_plane)
    inputs = tf.reshape(tf.concat([bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1), [-1, 112, 64])

    # winner is stored in one signed byte and needs to be converted to one hot.
    winner = tf.cast(tf.io.decode_raw(tf.strings.substr(raw, 8275, 1), tf.int8), tf.float32)
    winner = tf.tile(winner, [1,3])
    z = tf.cast(tf.equal(winner, [1., 0., -1.]), tf.float32)

    # Outcome distribution needs to be calculated from q and d.
    best_q = tf.io.decode_raw(tf.strings.substr(raw, 8280, 4), tf.float32)
    best_d = tf.io.decode_raw(tf.strings.substr(raw, 8288, 4), tf.float32)
    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)

    q = tf.concat([best_q_w, best_d, best_q_l], 1)

    return (inputs, policy, z, q)

def sample(x):
    return tf.math.equal(tf.random.uniform([], 0, SKIP-1, dtype=tf.int32), 0)

def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    num_chunks = cfg['dataset']['num_chunks']
    allow_less = cfg['dataset'].get('allow_less_chunks', False)
    train_ratio = cfg['dataset']['train_ratio']
    experimental_parser = cfg['dataset'].get('experimental_v4_only_dataset', False)
    num_train = int(num_chunks*train_ratio)
    num_test = num_chunks - num_train
    if 'input_test' in cfg['dataset']:
        train_chunks = get_latest_chunks(cfg['dataset']['input_train'], num_train, allow_less)
        test_chunks = get_latest_chunks(cfg['dataset']['input_test'], num_test, allow_less)
    else:
        chunks = get_latest_chunks(cfg['dataset']['input'], num_chunks, allow_less)
        if allow_less:
            num_train = int(len(chunks)*train_ratio)
            num_test = len(chunks) - num_train
        train_chunks = chunks[:num_train]
        test_chunks = chunks[num_train:]

    shuffle_size = cfg['training']['shuffle_size']
    total_batch_size = cfg['training']['batch_size']
    batch_splits = cfg['training'].get('num_batch_splits', 1)
    train_workers = cfg['dataset'].get('train_workers', None)
    test_workers = cfg['dataset'].get('test_workers', None)
    if total_batch_size % batch_splits != 0:
        raise ValueError('num_batch_splits must divide batch_size evenly')
    split_batch_size = total_batch_size // batch_splits
    # Load data with split batch size, which will be combined to the total batch size in tfprocess.
    ChunkParser.BATCH_SIZE = split_batch_size

    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    tfprocess = TFProcess(cfg)

    if experimental_parser:
        train_dataset = tf.data.Dataset.from_tensor_slices(train_chunks).shuffle(len(train_chunks)).repeat()\
                         .interleave(lambda x: tf.data.FixedLengthRecordDataset(x, 8292, compression_type='GZIP', num_parallel_reads=1).filter(sample), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                         .shuffle(shuffle_size)\
                         .batch(split_batch_size).map(extract_inputs_outputs).prefetch(4)
    else:
        train_parser = ChunkParser(FileDataSrc(train_chunks),
                shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE,
                workers=train_workers)
        train_dataset = tf.data.Dataset.from_generator(
            train_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
        train_dataset = train_dataset.map(ChunkParser.parse_function)
        train_dataset = train_dataset.prefetch(4)

    shuffle_size = int(shuffle_size*(1.0-train_ratio))
    if experimental_parser:
        test_dataset = tf.data.Dataset.from_tensor_slices(test_chunks).shuffle(len(test_chunks)).repeat()\
                         .interleave(lambda x: tf.data.FixedLengthRecordDataset(x, 8292, compression_type='GZIP', num_parallel_reads=1).filter(sample), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                         .shuffle(shuffle_size)\
                         .batch(split_batch_size).map(extract_inputs_outputs).prefetch(4)
    else:
        test_parser = ChunkParser(FileDataSrc(test_chunks),
                shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE,
                workers=test_workers)
        test_dataset = tf.data.Dataset.from_generator(
            test_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
        test_dataset = test_dataset.map(ChunkParser.parse_function)
        test_dataset = test_dataset.prefetch(4)

    validation_dataset = None
    if 'input_validation' in cfg['dataset']:
        valid_chunks = get_all_chunks(cfg['dataset']['input_validation'])
        validation_dataset = tf.data.FixedLengthRecordDataset(valid_chunks, 8292, compression_type='GZIP', num_parallel_reads=1)\
                               .batch(split_batch_size, drop_remainder=True).map(extract_inputs_outputs).prefetch(4)

    tfprocess.init_v2(train_dataset, test_dataset, validation_dataset)

    tfprocess.restore_v2()

    # If number of test positions is not given
    # sweeps through all test chunks statistically
    # Assumes average of 10 samples per test game.
    # For simplicity, testing can use the split batch size instead of total batch size.
    # This does not affect results, because test results are simple averages that are independent of batch size.
    num_evals = cfg['training'].get('num_test_positions', len(test_chunks) * 10)
    num_evals = max(1, num_evals // ChunkParser.BATCH_SIZE)
    print("Using {} evaluation batches".format(num_evals))

    tfprocess.process_loop_v2(total_batch_size, num_evals, batch_splits=batch_splits)

    if cmd.output is not None:
        if cfg['training'].get('swa_output', False):
            tfprocess.save_swa_weights_v2(cmd.output)
        else:
            tfprocess.save_leelaz_weights_v2(cmd.output)

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
