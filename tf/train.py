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
from chunkparser import ChunkParser


SKIP = 32

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
            print("sorting {} chunks...".format(len(chunks)),
                  end="",
                  flush=True)
            if True:
                print("sorting disabled")
            else:
                chunks.sort(key=sort_key_fn, reverse=True)
            print("[done]")
            print("{} - {}".format(os.path.basename(chunks[-1]),
                                   os.path.basename(chunks[0])))
            print("shuffling chunks", end="")
            if False:
                print("shuffling disabled")
            else:
                random.shuffle(chunks)
            print("[done]")
            return chunks
        else:
            print("Not enough chunks {}".format(len(chunks)))
            sys.exit(1)

    print("sorting {} chunks...".format(len(chunks)), end="", flush=True)
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
    num_str = os.path.basename(name).upper().strip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ_-.")
    return int(num_str)


def get_input_mode(cfg):
    import proto.net_pb2 as pb
    input_mode = cfg["model"].get("input_type", "classic")

    if input_mode == "classic":
        return pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE
    elif input_mode == "frc_castling":
        return pb.NetworkFormat.INPUT_112_WITH_CASTLING_PLANE
    elif input_mode == "canonical":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION
    elif input_mode == "canonical_100":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES
    elif input_mode == "canonical_armageddon":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON
    elif input_mode == "canonical_v2":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2
    elif input_mode == "canonical_v2_armageddon":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON
    else:
        raise ValueError("Unknown input mode format: {}".format(input_mode))


def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    num_chunks = cfg["dataset"]["num_chunks"]
    allow_less = cfg["dataset"].get("allow_less_chunks", False)
    train_ratio = cfg["dataset"]["train_ratio"]
    num_train = int(num_chunks * train_ratio)
    num_test = num_chunks - num_train
    sort_type = cfg["dataset"].get("sort_type", "mtime")
    if sort_type == "mtime":
        sort_key_fn = os.path.getmtime
    elif sort_type == "number":
        sort_key_fn = game_number_for_name
    elif sort_type == "name":
        sort_key_fn = identity_function
    else:
        raise ValueError("Unknown dataset sort_type: {}".format(sort_type))
    if "input_test" in cfg["dataset"]:
        train_chunks = get_latest_chunks(cfg["dataset"]["input_train"],
                                         num_train, allow_less, sort_key_fn)
        test_chunks = get_latest_chunks(cfg["dataset"]["input_test"], num_test,
                                        allow_less, sort_key_fn)
    else:
        chunks = get_latest_chunks(cfg["dataset"]["input"], num_chunks,
                                   allow_less, sort_key_fn)
        if allow_less:
            num_train = int(len(chunks) * train_ratio)
            num_test = len(chunks) - num_train
        train_chunks = chunks[:num_train]
        test_chunks = chunks[num_train:]

    shuffle_size = cfg["training"]["shuffle_size"]
    total_batch_size = cfg["training"]["batch_size"]
    batch_splits = cfg["training"].get("num_batch_splits", 1)
    train_workers = cfg["dataset"].get("train_workers", None)
    test_workers = cfg["dataset"].get("test_workers", None)
    if total_batch_size % batch_splits != 0:
        raise ValueError("num_batch_splits must divide batch_size evenly")
    split_batch_size = total_batch_size // batch_splits

    diff_focus_min = cfg["training"].get("diff_focus_min", 1)
    diff_focus_slope = cfg["training"].get("diff_focus_slope", 0)
    diff_focus_q_weight = cfg["training"].get("diff_focus_q_weight", 6.0)
    diff_focus_pol_scale = cfg["training"].get("diff_focus_pol_scale", 3.5)

    root_dir = os.path.join(cfg["training"]["path"], cfg["name"])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    train_parser = ChunkParser(train_chunks,
                               get_input_mode(cfg),
                               shuffle_size=shuffle_size,
                               sample=SKIP,
                               batch_size=split_batch_size,
                               diff_focus_min=diff_focus_min,
                               diff_focus_slope=diff_focus_slope,
                               diff_focus_q_weight=diff_focus_q_weight,
                               diff_focus_pol_scale=diff_focus_pol_scale,
                               workers=train_workers)
    test_shuffle_size = int(shuffle_size * (1.0 - train_ratio))
    # no diff focus for test_parser
    test_parser = ChunkParser(test_chunks,
                              get_input_mode(cfg),
                              shuffle_size=test_shuffle_size,
                              sample=SKIP,
                              batch_size=split_batch_size,
                              workers=test_workers)
    
    
    if "input_validation" in cfg["dataset"]:
        valid_chunks = get_all_chunks(cfg["dataset"]["input_validation"])
        validation_parser = ChunkParser(valid_chunks,
                                        get_input_mode(cfg),
                                        sample=1,
                                        batch_size=split_batch_size,
                                        workers=0)

    import tensorflow as tf
    from chunkparsefunc import parse_function
    from tfprocess import TFProcess

    print("Creating TFProcess")
    tfprocess = TFProcess(cfg)
    print("Done")
    output_types = 8 * (tf.string,)

    print("Initializing datasets")
    train_dataset = tf.data.Dataset.from_generator(
        train_parser.parse,
        output_types=output_types)
    train_dataset = train_dataset.map(parse_function)
    test_dataset = tf.data.Dataset.from_generator(
        test_parser.parse,
        output_types=output_types)
    test_dataset = test_dataset.map(parse_function)

    validation_dataset = None
    if "input_validation" in cfg["dataset"]:
        validation_dataset = tf.data.Dataset.from_generator(
            validation_parser.sequential,
            output_types=output_types)
        validation_dataset = validation_dataset.map(parse_function)

    if tfprocess.strategy is None:  # Mirrored strategy appends prefetch itself with a value depending on number of replicas
        train_dataset = train_dataset.prefetch(4)
        test_dataset = test_dataset.prefetch(4)
        if validation_dataset is not None:
            validation_dataset = validation_dataset.prefetch(4)
    else:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_dataset = train_dataset.with_options(options)
        test_dataset = test_dataset.with_options(options)
        if validation_dataset is not None:
            validation_dataset = validation_dataset.with_options(options)
    print("Done")

    print("Initializing TFProcess")
    tfprocess.init(train_dataset, test_dataset,
                   validation_dataset)  # None, None, None

    tfprocess.restore()
    print("Done")

    # If number of test positions is not given
    # sweeps through all test chunks statistically
    # Assumes average of 10 samples per test game.
    # For simplicity, testing can use the split batch size instead of total batch size.
    # This does not affect results, because test results are simple averages that are independent of batch size.
    num_evals = cfg["training"].get("num_test_positions",
                                    len(test_chunks) * 10)
    num_evals = max(1, num_evals // split_batch_size)
    print("Using {} evaluation batches".format(num_evals))
    tfprocess.total_batch_size = total_batch_size
    tfprocess.process_loop(total_batch_size,
                           num_evals,
                           batch_splits=batch_splits)

    if cmd.output is not None:
        if cfg["training"].get("swa_output", False):
            tfprocess.save_swa_weights(cmd.output)
        else:
            tfprocess.save_leelaz_weights(cmd.output)

    train_parser.shutdown()
    test_parser.shutdown()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Tensorflow pipeline for training Leela Chess.")
    argparser.add_argument("--cfg",
                           type=argparse.FileType("r"),
                           help="yaml configuration with training parameters")
    argparser.add_argument("--output",
                           type=str,
                           help="file to store weights in")

    # mp.set_start_method("spawn")
    main(argparser.parse_args())
    mp.freeze_support()