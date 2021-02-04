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
SKIP_MULTIPLE = 1024


def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")


def get_all_chunks(path):
    if isinstance(path, list):
        print("getting chunks for", path)
        chunks = []
        for i in path:
            chunks += get_all_chunks(i)
        return chunks
    chunks = []
    for d in glob.glob(path):
        chunks += get_chunks(d)
    print("got", len(chunks), "chunks for", path)
    return chunks


def get_latest_chunks(path, num_chunks, allow_less, sort_key_fn):
    chunks = get_all_chunks(path)
    if len(chunks) < num_chunks:
        if allow_less:
            print("sorting {} chunks...".format(len(chunks)), end='', flush=True)
            chunks.sort(key=sort_key_fn, reverse=True)
            print("[done]")
            print("{} - {}".format(os.path.basename(chunks[-1]),
                                   os.path.basename(chunks[0])))
            random.shuffle(chunks)
            return chunks
        else:
            print("Not enough chunks {}".format(len(chunks)))
            sys.exit(1)

    print("sorting {} chunks...".format(len(chunks)), end='', flush=True)
    chunks.sort(key=sort_key_fn, reverse=True)
    print("[done]")
    chunks = chunks[:num_chunks]
    print("{} - {}".format(os.path.basename(chunks[-1]),
                           os.path.basename(chunks[0])))
    random.shuffle(chunks)
    return chunks


def identity_function(name):
    return name


def game_number_for_name(name):
    num_str = os.path.basename(name).upper().strip("ABCDEFGHIJKLMNOPQRSTUVWXYZ_-.")
    return int(num_str)


def extract_policy_bits(raw):
    # Next 7432 are easy, policy extraction.
    policy = tf.io.decode_raw(tf.strings.substr(raw, 8, 7432), tf.float32)
    # Next are 104 bit packed chess boards, they have to be expanded.
    bit_planes = tf.expand_dims(
        tf.reshape(
            tf.io.decode_raw(tf.strings.substr(raw, 7440, 832), tf.uint8),
            [-1, 104, 8]), -1)
    bit_planes = tf.bitwise.bitwise_and(tf.tile(bit_planes, [1, 1, 1, 8]),
                                        [128, 64, 32, 16, 8, 4, 2, 1])
    bit_planes = tf.minimum(1., tf.cast(bit_planes, tf.float32))
    return policy, bit_planes


def extract_byte_planes(raw):
    # 5 bytes in input are expanded and tiled
    unit_planes = tf.expand_dims(
        tf.expand_dims(
            tf.io.decode_raw(tf.strings.substr(raw, 8272, 5), tf.uint8), -1),
        -1)
    unit_planes = tf.tile(unit_planes, [1, 1, 8, 8])
    return unit_planes


def extract_rule50_zero_one(raw):
    # rule50 count plane.
    rule50_plane = tf.expand_dims(
        tf.expand_dims(
            tf.io.decode_raw(tf.strings.substr(raw, 8277, 1), tf.uint8), -1),
        -1)
    rule50_plane = tf.cast(tf.tile(rule50_plane, [1, 1, 8, 8]), tf.float32)
    rule50_plane = tf.divide(rule50_plane, 99.)
    # zero plane and one plane
    zero_plane = tf.zeros_like(rule50_plane)
    one_plane = tf.ones_like(rule50_plane)
    return rule50_plane, zero_plane, one_plane


def extract_rule50_100_zero_one(raw):
    # rule50 count plane.
    rule50_plane = tf.expand_dims(
        tf.expand_dims(
            tf.io.decode_raw(tf.strings.substr(raw, 8277, 1), tf.uint8), -1),
        -1)
    rule50_plane = tf.cast(tf.tile(rule50_plane, [1, 1, 8, 8]), tf.float32)
    rule50_plane = tf.divide(rule50_plane, 100.)
    # zero plane and one plane
    zero_plane = tf.zeros_like(rule50_plane)
    one_plane = tf.ones_like(rule50_plane)
    return rule50_plane, zero_plane, one_plane


def extract_invariance(raw):
    # invariance plane.
    invariance_plane = tf.expand_dims(
        tf.expand_dims(
            tf.io.decode_raw(tf.strings.substr(raw, 8278, 1), tf.uint8), -1),
        -1)
    return tf.cast(tf.tile(invariance_plane, [1, 1, 8, 8]), tf.float32)


def extract_outputs(raw):
    # winner is stored in one signed byte and needs to be converted to one hot.
    winner = tf.cast(
        tf.io.decode_raw(tf.strings.substr(raw, 8279, 1), tf.int8), tf.float32)
    winner = tf.tile(winner, [1, 3])
    z = tf.cast(tf.equal(winner, [1., 0., -1.]), tf.float32)

    # Outcome distribution needs to be calculated from q and d.
    best_q = tf.io.decode_raw(tf.strings.substr(raw, 8284, 4), tf.float32)
    best_d = tf.io.decode_raw(tf.strings.substr(raw, 8292, 4), tf.float32)
    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)

    q = tf.concat([best_q_w, best_d, best_q_l], 1)

    ply_count = tf.io.decode_raw(tf.strings.substr(raw, 8304, 4), tf.float32)
    return z, q, ply_count


def extract_inputs_outputs_if1(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    #input_format = tf.reshape(
    #    tf.io.decode_raw(tf.strings.substr(raw, 4, 4), tf.int32),
    #    [-1, 1, 1, 1])
    # tf.debugging.assert_equal(input_format, tf.ones_like(input_format))

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 are castling + stm, all of which simply copy the byte value to all squares.
    unit_planes = tf.cast(extract_byte_planes(raw), tf.float32)

    rule50_plane, zero_plane, one_plane = extract_rule50_zero_one(raw)

    inputs = tf.reshape(
        tf.concat(
            [bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1),
        [-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return (inputs, policy, z, q, ply_count)


def extract_unit_planes_with_bitsplat(raw):
    unit_planes = extract_byte_planes(raw)
    bitsplat_unit_planes = tf.bitwise.bitwise_and(
        unit_planes, [1, 2, 4, 8, 16, 32, 64, 128])
    bitsplat_unit_planes = tf.minimum(
        1., tf.cast(bitsplat_unit_planes, tf.float32))
    unit_planes = tf.cast(unit_planes, tf.float32)
    return unit_planes, bitsplat_unit_planes


def make_frc_castling(bitsplat_unit_planes, zero_plane):
    queenside = tf.concat([
        bitsplat_unit_planes[:, :1, :1], zero_plane[:, :, :6],
        bitsplat_unit_planes[:, 2:3, :1]
    ], 2)
    kingside = tf.concat([
        bitsplat_unit_planes[:, 1:2, :1], zero_plane[:, :, :6],
        bitsplat_unit_planes[:, 3:4, :1]
    ], 2)
    return queenside, kingside


def extract_inputs_outputs_if2(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    #input_format = tf.reshape(
    #    tf.io.decode_raw(tf.strings.substr(raw, 4, 4), tf.int32),
    #    [-1, 1, 1, 1])
    # tf.debugging.assert_equal(input_format, tf.multiply(tf.ones_like(input_format), 2))

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 inputs are 4 frc castling and 1 stm.
    # In order to do frc we need to make bit unpacked versions.  Note little endian for these fields so the bitwise_and array is reversed.
    # Although we only need bit unpacked for first 4 of 5 planes, its simpler just to create them all.
    unit_planes, bitsplat_unit_planes = extract_unit_planes_with_bitsplat(raw)

    rule50_plane, zero_plane, one_plane = extract_rule50_zero_one(raw)

    # For FRC the old unit planes must be replaced with 0 and 2 merged, 1 and 3 merged, two zero planes and then original 4.
    queenside, kingside = make_frc_castling(bitsplat_unit_planes, zero_plane)
    unit_planes = tf.concat(
        [queenside, kingside, zero_plane, zero_plane, unit_planes[:, 4:]], 1)

    inputs = tf.reshape(
        tf.concat(
            [bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1),
        [-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return (inputs, policy, z, q, ply_count)


def make_canonical_unit_planes(bitsplat_unit_planes, zero_plane):
    # For canonical the old unit planes must be replaced with 0 and 2 merged, 1 and 3 merged, two zero planes and then en-passant.
    queenside, kingside = make_frc_castling(bitsplat_unit_planes, zero_plane)
    enpassant = tf.concat(
        [zero_plane[:, :, :7], bitsplat_unit_planes[:, 4:, :1]], 2)
    unit_planes = tf.concat(
        [queenside, kingside, zero_plane, zero_plane, enpassant], 1)
    return unit_planes


def extract_inputs_outputs_if3(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    #input_format = tf.reshape(
    #    tf.io.decode_raw(tf.strings.substr(raw, 4, 4), tf.int32),
    #    [-1, 1, 1, 1])
    # tf.debugging.assert_equal(input_format, tf.multiply(tf.ones_like(input_format), 3))

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 inputs are 4 castling and 1 enpassant.
    # In order to do the frc castling and if3 enpassant plane we need to make bit unpacked versions.  Note little endian for these fields so the bitwise_and array is reversed.
    unit_planes, bitsplat_unit_planes = extract_unit_planes_with_bitsplat(raw)

    rule50_plane, zero_plane, one_plane = extract_rule50_zero_one(raw)

    unit_planes = make_canonical_unit_planes(bitsplat_unit_planes, zero_plane)

    inputs = tf.reshape(
        tf.concat(
            [bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1),
        [-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return (inputs, policy, z, q, ply_count)


def extract_inputs_outputs_if4(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    #input_format = tf.reshape(
    #    tf.io.decode_raw(tf.strings.substr(raw, 4, 4), tf.int32),
    #    [-1, 1, 1, 1])
    # tf.debugging.assert_equal(input_format, tf.multiply(tf.ones_like(input_format), 3))

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 inputs are 4 castling and 1 enpassant.
    # In order to do the frc castling and if3 enpassant plane we need to make bit unpacked versions.  Note little endian for these fields so the bitwise_and array is reversed.
    unit_planes, bitsplat_unit_planes = extract_unit_planes_with_bitsplat(raw)

    rule50_plane, zero_plane, one_plane = extract_rule50_100_zero_one(raw)

    unit_planes = make_canonical_unit_planes(bitsplat_unit_planes, zero_plane)

    inputs = tf.reshape(
        tf.concat(
            [bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1),
        [-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return (inputs, policy, z, q, ply_count)


def make_armageddon_stm(invariance_plane):
    # invariance_plane contains values of 128 or higher if its black side to move, 127 or lower otherwise.
    # Convert this to 0,1 by subtracting off 127 and then clipping.
    return tf.clip_by_value(invariance_plane - 127., 0., 1.)


def extract_inputs_outputs_if132(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    #input_format = tf.reshape(
    #    tf.io.decode_raw(tf.strings.substr(raw, 4, 4), tf.int32),
    #    [-1, 1, 1, 1])
    # tf.debugging.assert_equal(input_format, tf.multiply(tf.ones_like(input_format), 3))

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 inputs are 4 castling and 1 enpassant.
    # In order to do the frc castling and if3 enpassant plane we need to make bit unpacked versions.  Note little endian for these fields so the bitwise_and array is reversed.
    unit_planes, bitsplat_unit_planes = extract_unit_planes_with_bitsplat(raw)

    rule50_plane, zero_plane, one_plane = extract_rule50_100_zero_one(raw)

    unit_planes = make_canonical_unit_planes(bitsplat_unit_planes, zero_plane)

    armageddon_stm = make_armageddon_stm(extract_invariance(raw))

    inputs = tf.reshape(
        tf.concat(
            [bit_planes, unit_planes, rule50_plane, armageddon_stm, one_plane],
            1), [-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return (inputs, policy, z, q, ply_count)


def select_extractor(mode):
    if mode == 1:
        return extract_inputs_outputs_if1
    if mode == 2:
        return extract_inputs_outputs_if2
    if mode == 3:
        return extract_inputs_outputs_if3
    if mode == 4 or mode == 5:
        return extract_inputs_outputs_if4
    if mode == 132 or mode == 133:
        return extract_inputs_outputs_if132
    assert (false)


def semi_sample(x):
    return tf.slice(tf.random.shuffle(x), [0], [SKIP_MULTIPLE])


def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    num_chunks = cfg['dataset']['num_chunks']
    allow_less = cfg['dataset'].get('allow_less_chunks', False)
    train_ratio = cfg['dataset']['train_ratio']
    experimental_parser = cfg['dataset'].get('experimental_v5_only_dataset',
                                             False)
    num_train = int(num_chunks * train_ratio)
    num_test = num_chunks - num_train
    sort_type = cfg['dataset'].get('sort_type', 'mtime')
    if sort_type == 'mtime':
        sort_key_fn = os.path.getmtime
    elif sort_type == 'number':
        sort_key_fn = game_number_for_name
    elif sort_type == 'name':
        sort_key_fn = identity_function
    else:
        raise ValueError('Unknown dataset sort_type: {}'.format(sort_type))
    if 'input_test' in cfg['dataset']:
        train_chunks = get_latest_chunks(cfg['dataset']['input_train'],
                                         num_train, allow_less, sort_key_fn)
        test_chunks = get_latest_chunks(cfg['dataset']['input_test'], num_test,
                                        allow_less, sort_key_fn)
    else:
        chunks = get_latest_chunks(cfg['dataset']['input'], num_chunks,
                                   allow_less, sort_key_fn)
        if allow_less:
            num_train = int(len(chunks) * train_ratio)
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

    value_focus_min = cfg['training'].get('value_focus_min', 1)
    value_focus_slope = cfg['training'].get('value_focus_slope', 0)

    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    tfprocess = TFProcess(cfg)
    experimental_reads = max(2, mp.cpu_count() - 2) // 2
    extractor = select_extractor(tfprocess.INPUT_MODE)

    if experimental_parser and (value_focus_min != 1 or value_focus_slope != 0):
        raise ValueError('Experimental parser does not support non-default value \
                          focus parameters.')

    def read(x):
        return tf.data.FixedLengthRecordDataset(
            x,
            8308,
            compression_type='GZIP',
            num_parallel_reads=experimental_reads)

    if experimental_parser:
        train_dataset = tf.data.Dataset.from_tensor_slices(train_chunks).shuffle(len(train_chunks)).repeat().batch(256)\
                         .interleave(read, num_parallel_calls=2)\
                         .batch(SKIP_MULTIPLE*SKIP).map(semi_sample).unbatch()\
                         .shuffle(shuffle_size)\
                         .batch(split_batch_size).map(extractor).prefetch(4)
    else:
        train_parser = ChunkParser(train_chunks,
                                   tfprocess.INPUT_MODE,
                                   shuffle_size=shuffle_size,
                                   sample=SKIP,
                                   batch_size=ChunkParser.BATCH_SIZE,
                                   value_focus_min=value_focus_min,
                                   value_focus_slope=value_focus_slope,
                                   workers=train_workers)
        train_dataset = tf.data.Dataset.from_generator(
            train_parser.parse,
            output_types=(tf.string, tf.string, tf.string, tf.string,
                          tf.string))
        train_dataset = train_dataset.map(ChunkParser.parse_function)
        train_dataset = train_dataset.prefetch(4)

    shuffle_size = int(shuffle_size * (1.0 - train_ratio))
    if experimental_parser:
        test_dataset = tf.data.Dataset.from_tensor_slices(test_chunks).shuffle(len(test_chunks)).repeat().batch(256)\
                         .interleave(read, num_parallel_calls=2)\
                         .batch(SKIP_MULTIPLE*SKIP).map(semi_sample).unbatch()\
                         .shuffle(shuffle_size)\
                         .batch(split_batch_size).map(extractor).prefetch(4)
    else:
        # no value focus for test_parser
        test_parser = ChunkParser(test_chunks,
                                  tfprocess.INPUT_MODE,
                                  shuffle_size=shuffle_size,
                                  sample=SKIP,
                                  batch_size=ChunkParser.BATCH_SIZE,
                                  workers=test_workers)
        test_dataset = tf.data.Dataset.from_generator(
            test_parser.parse,
            output_types=(tf.string, tf.string, tf.string, tf.string,
                          tf.string))
        test_dataset = test_dataset.map(ChunkParser.parse_function)
        test_dataset = test_dataset.prefetch(4)

    validation_dataset = None
    if 'input_validation' in cfg['dataset']:
        valid_chunks = get_all_chunks(cfg['dataset']['input_validation'])
        validation_dataset = tf.data.FixedLengthRecordDataset(valid_chunks, 8308, compression_type='GZIP', num_parallel_reads=experimental_reads)\
                               .batch(split_batch_size, drop_remainder=True).map(extractor).prefetch(4)

    tfprocess.init_v2(train_dataset, test_dataset, validation_dataset)

    tfprocess.restore_v2()

    # If number of test positions is not given
    # sweeps through all test chunks statistically
    # Assumes average of 10 samples per test game.
    # For simplicity, testing can use the split batch size instead of total batch size.
    # This does not affect results, because test results are simple averages that are independent of batch size.
    num_evals = cfg['training'].get('num_test_positions',
                                    len(test_chunks) * 10)
    num_evals = max(1, num_evals // ChunkParser.BATCH_SIZE)
    print("Using {} evaluation batches".format(num_evals))

    tfprocess.process_loop_v2(total_batch_size,
                              num_evals,
                              batch_splits=batch_splits)

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
    argparser.add_argument('--cfg',
                           type=argparse.FileType('r'),
                           help='yaml configuration with training parameters')
    argparser.add_argument('--output',
                           type=str,
                           help='file to store weights in')

    #mp.set_start_method('spawn')
    main(argparser.parse_args())
    mp.freeze_support()
