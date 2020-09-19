#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2018 Folkert Huizinga
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Chess is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Chess is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

import itertools
import multiprocessing as mp
import numpy as np
import random
import shufflebuffer as sb
import struct
import tensorflow as tf
import unittest
import gzip
from select import select

V5_VERSION = struct.pack('i', 5)
CLASSICAL_INPUT = struct.pack('i', 1)
V4_VERSION = struct.pack('i', 4)
V3_VERSION = struct.pack('i', 3)
V5_STRUCT_STRING = '4si7432s832sBBBBBBBbfffffff'
V4_STRUCT_STRING = '4s7432s832sBBBBBBBbffff'
V3_STRUCT_STRING = '4s7432s832sBBBBBBBb'


def reverse_expand_bits(plane):
    return np.unpackbits(np.array([plane], dtype=np.uint8))[::-1].astype(
        np.float32).tobytes()


# Interface for a chunk data source.
class ChunkDataSrc:
    def __init__(self, items):
        self.items = items

    def next(self):
        if not self.items:
            return None
        return self.items.pop()


def chunk_reader(chunk_filenames, chunk_filename_queue):
    """
    Reads chunk filenames from a list and writes them in shuffled
    order to output_pipes.
    """
    chunks = []
    done = chunk_filenames

    while True:
        if not chunks:
            chunks, done = done, chunks
            random.shuffle(chunks)
        if not chunks:
            print("chunk_reader didn't find any chunks.")
            return None
        while len(chunks):
            filename = chunks.pop()
            done.append(filename)
            chunk_filename_queue.put(filename)
    print("chunk_reader exiting.")
    return None


class ChunkParser:
    # static batch size
    BATCH_SIZE = 8

    def __init__(self,
                 chunks,
                 expected_input_format,
                 shuffle_size=1,
                 sample=1,
                 buffer_size=1,
                 batch_size=256,
                 workers=None):
        """
        Read data and yield batches of raw tensors.

        'chunks' list of chunk filenames.
        'shuffle_size' is the size of the shuffle buffer.
        'sample' is the rate to down-sample.
        'workers' is the number of child workers to use.

        The data is represented in a number of formats through this dataflow
        pipeline. In order, they are:

        chunk: The name of a file containing chunkdata

        chunkdata: type Bytes. Multiple records of v5 format where each record
        consists of (state, policy, result, q)

        raw: A byte string holding raw tensors contenated together. This is
        used to pass data from the workers to the parent. Exists because
        TensorFlow doesn't have a fast way to unpack bit vectors. 7950 bytes
        long.
        """

        self.expected_input_format = expected_input_format

        # Build 2 flat float32 planes with values 0,1
        self.flat_planes = []
        for i in range(2):
            self.flat_planes.append(
                (np.zeros(64, dtype=np.float32) + i).tobytes())

        # set the down-sampling rate
        self.sample = sample
        # set the mini-batch size
        self.batch_size = batch_size
        # set number of elements in the shuffle buffer.
        self.shuffle_size = shuffle_size
        # Start worker processes, leave 2 for TensorFlow
        if workers is None:
            workers = max(1, mp.cpu_count() - 2)

        print("Using {} worker processes.".format(workers))

        # Start the child workers running
        self.readers = []
        self.writers = []
        self.processes = []
        self.chunk_filename_queue = mp.Queue(maxsize=4096)
        for _ in range(workers):
            read, write = mp.Pipe(duplex=False)
            p = mp.Process(target=self.task,
                           args=(self.chunk_filename_queue, write))
            p.daemon = True
            self.processes.append(p)
            p.start()
            self.readers.append(read)
            self.writers.append(write)

        self.chunk_process = mp.Process(target=chunk_reader,
                                        args=(chunks,
                                              self.chunk_filename_queue))
        self.chunk_process.daemon = True
        self.chunk_process.start()

        self.init_structs()

    def shutdown(self):
        """
        Terminates all the workers
        """
        for i in range(len(self.readers)):
            self.processes[i].terminate()
            self.processes[i].join()
            self.readers[i].close()
            self.writers[i].close()
        self.chunk_process.terminate()
        self.chunk_process.join()

    def init_structs(self):
        """
        struct.Struct doesn't pickle, so it needs to be separately
        constructed in workers.
        """
        self.v5_struct = struct.Struct(V5_STRUCT_STRING)
        self.v4_struct = struct.Struct(V4_STRUCT_STRING)
        self.v3_struct = struct.Struct(V3_STRUCT_STRING)

    @staticmethod
    def parse_function(planes, probs, winner, q, plies_left):
        """
        Convert unpacked record batches to tensors for tensorflow training
        """
        planes = tf.io.decode_raw(planes, tf.float32)
        probs = tf.io.decode_raw(probs, tf.float32)
        winner = tf.io.decode_raw(winner, tf.float32)
        q = tf.io.decode_raw(q, tf.float32)
        plies_left = tf.io.decode_raw(plies_left, tf.float32)

        planes = tf.reshape(planes, (ChunkParser.BATCH_SIZE, 112, 8 * 8))
        probs = tf.reshape(probs, (ChunkParser.BATCH_SIZE, 1858))
        winner = tf.reshape(winner, (ChunkParser.BATCH_SIZE, 3))
        q = tf.reshape(q, (ChunkParser.BATCH_SIZE, 3))
        plies_left = tf.reshape(plies_left, (ChunkParser.BATCH_SIZE, ))

        return (planes, probs, winner, q, plies_left)

    def convert_v5_to_tuple(self, content):
        """
        Unpack a v5 binary record to 5-tuple (state, policy pi, result, q, m)

        v5 struct format is (8308 bytes total)
            int32 version (4 bytes)
            int32 input_format (4 bytes)
            1858 float32 probabilities (7432 bytes)
            104 (13*8) packed bit planes of 8 bytes each (832 bytes)
            uint8 castling us_ooo (1 byte)
            uint8 castling us_oo (1 byte)
            uint8 castling them_ooo (1 byte)
            uint8 castling them_oo (1 byte)
            uint8 side_to_move (1 byte)
            uint8 rule50_count (1 byte)
            uint8 dep_ply_count (1 byte) (unused)
            int8 result (1 byte)
            float32 root_q (4 bytes)
            float32 best_q (4 bytes)
            float32 root_d (4 bytes)
            float32 best_d (4 bytes)
            float32 root_m (4 bytes)
            float32 best_m (4 bytes)
            float32 plies_left (4 bytes)
        """
        (ver, input_format, probs, planes, us_ooo, us_oo, them_ooo, them_oo,
         stm, rule50_count, dep_ply_count, winner, root_q, best_q, root_d,
         best_d, root_m, best_m, plies_left) = self.v5_struct.unpack(content)
        # v3/4 data sometimes has a useful value in dep_ply_count, so copy that over if the new ply_count is not populated.
        if plies_left == 0:
            plies_left = dep_ply_count
        plies_left = struct.pack('f', plies_left)

        assert input_format == self.expected_input_format

        # Unpack bit planes and cast to 32 bit float
        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(
            np.float32)
        rule50_divisor = 99.0
        if input_format > 3:
            rule50_divisor = 100.0
        rule50_plane = struct.pack('f', rule50_count / rule50_divisor) * 64

        if input_format == 1:
            middle_planes = self.flat_planes[us_ooo] + \
                            self.flat_planes[us_oo] + \
                            self.flat_planes[them_ooo] + \
                            self.flat_planes[them_oo] + \
                            self.flat_planes[stm]
        elif input_format == 2:
            # Each inner array has to be reversed as these fields are in opposite endian to the planes data.
            them_ooo_bytes = reverse_expand_bits(them_ooo)
            us_ooo_bytes = reverse_expand_bits(us_ooo)
            them_oo_bytes = reverse_expand_bits(them_oo)
            us_oo_bytes = reverse_expand_bits(us_oo)
            middle_planes = us_ooo_bytes + (6*8*4) * b'\x00' + them_ooo_bytes + \
                            us_oo_bytes + (6*8*4) * b'\x00' + them_oo_bytes + \
                            self.flat_planes[0] + \
                            self.flat_planes[0] + \
                            self.flat_planes[stm]
        elif input_format == 3 or input_format == 4 or input_format == 132 or input_format == 5 or input_format == 133:
            # Each inner array has to be reversed as these fields are in opposite endian to the planes data.
            them_ooo_bytes = reverse_expand_bits(them_ooo)
            us_ooo_bytes = reverse_expand_bits(us_ooo)
            them_oo_bytes = reverse_expand_bits(them_oo)
            us_oo_bytes = reverse_expand_bits(us_oo)
            enpassant_bytes = reverse_expand_bits(stm)
            middle_planes = us_ooo_bytes + (6*8*4) * b'\x00' + them_ooo_bytes + \
                            us_oo_bytes + (6*8*4) * b'\x00' + them_oo_bytes + \
                            self.flat_planes[0] + \
                            self.flat_planes[0] + \
                            (7*8*4) * b'\x00' + enpassant_bytes

        # Concatenate all byteplanes. Make the last plane all 1's so the NN can
        # detect edges of the board more easily
        aux_plus_6_plane = self.flat_planes[0]
        if (input_format == 132 or input_format == 133) and dep_ply_count >= 128:
            aux_plus_6_plane = self.flat_planes[1]
        planes = planes.tobytes() + \
                 middle_planes + \
                 rule50_plane + \
                 aux_plus_6_plane + \
                 self.flat_planes[1]

        assert len(planes) == ((8 * 13 * 1 + 8 * 1 * 1) * 8 * 8 * 4)
        winner = float(winner)
        assert winner == 1.0 or winner == -1.0 or winner == 0.0
        winner = struct.pack('fff', winner == 1.0, winner == 0.0,
                             winner == -1.0)

        best_q_w = 0.5 * (1.0 - best_d + best_q)
        best_q_l = 0.5 * (1.0 - best_d - best_q)
        assert -1.0 <= best_q <= 1.0 and 0.0 <= best_d <= 1.0
        best_q = struct.pack('fff', best_q_w, best_d, best_q_l)

        return (planes, probs, winner, best_q, plies_left)

    def sample_record(self, chunkdata):
        """
        Randomly sample through the v3/4/5 chunk data and select records in v5 format
        """
        version = chunkdata[0:4]
        if version == V5_VERSION:
            record_size = self.v5_struct.size
        elif version == V4_VERSION:
            record_size = self.v4_struct.size
        elif version == V3_VERSION:
            record_size = self.v3_struct.size
        else:
            return

        for i in range(0, len(chunkdata), record_size):
            if self.sample > 1:
                # Downsample, using only 1/Nth of the items.
                if random.randint(0, self.sample - 1) != 0:
                    continue  # Skip this record.
            record = chunkdata[i:i + record_size]
            if version == V3_VERSION:
                # add 16 bytes of fake root_q, best_q, root_d, best_d to match V4 format
                record += 16 * b'\x00'
            if version == V3_VERSION or version == V4_VERSION:
                # add 12 bytes of fake root_m, best_m, plies_left to match V5 format
                record += 12 * b'\x00'
                # insert 4 bytes of classical input format tag to match v5 format
                record = record[:4] + CLASSICAL_INPUT + record[4:]
            yield record

    def task(self, chunk_filename_queue, writer):
        """
        Run in fork'ed process, read data from chunkdatasrc, parsing, shuffling and
        sending v5 data through pipe back to main process.
        """
        self.init_structs()
        while True:
            filename = chunk_filename_queue.get()
            try:
                with gzip.open(filename, 'rb') as chunk_file:
                    version = chunk_file.read(4)
                    chunk_file.seek(0)
                    if version == V5_VERSION:
                        record_size = self.v5_struct.size
                    elif version == V4_VERSION:
                        record_size = self.v4_struct.size
                    elif version == V3_VERSION:
                        record_size = self.v3_struct.size
                    else:
                        print('Unknown version {} in file {}'.format(
                            version, filename))
                        continue
                    while True:
                        chunkdata = chunk_file.read(256 * record_size)
                        if len(chunkdata) == 0:
                            break
                        for item in self.sample_record(chunkdata):
                            writer.send_bytes(item)

            except:
                print("failed to parse {}".format(filename))
                continue

    def v5_gen(self):
        """
        Read v5 records from child workers, shuffle, and yield
        records.
        """
        sbuff = sb.ShuffleBuffer(self.v5_struct.size, self.shuffle_size)
        while len(self.readers):
            #for r in mp.connection.wait(self.readers):
            for r in self.readers:
                try:
                    s = r.recv_bytes()
                    s = sbuff.insert_or_replace(s)
                    if s is None:
                        continue  # shuffle buffer not yet full
                    yield s
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
        # drain the shuffle buffer.
        while True:
            s = sbuff.extract()
            if s is None:
                return
            yield s

    def tuple_gen(self, gen):
        """
        Take a generator producing v5 records and convert them to tuples.
        applying a random symmetry on the way.
        """
        for r in gen:
            yield self.convert_v5_to_tuple(r)

    def batch_gen(self, gen):
        """
        Pack multiple records into a single batch
        """
        # Get N records. We flatten the returned generator to
        # a list because we need to reuse it.
        while True:
            s = list(itertools.islice(gen, self.batch_size))
            if not len(s):
                return
            yield (b''.join([x[0] for x in s]), b''.join([x[1] for x in s]),
                   b''.join([x[2] for x in s]), b''.join([x[3] for x in s]),
                   b''.join([x[4] for x in s]))

    def parse(self):
        """
        Read data from child workers and yield batches of unpacked records
        """
        gen = self.v5_gen()  # read from workers
        gen = self.tuple_gen(gen)  # convert v5->tuple
        gen = self.batch_gen(gen)  # assemble into batches
        for b in gen:
            yield b


# Tests to check that records parse correctly
class ChunkParserTest(unittest.TestCase):
    def setUp(self):
        self.v4_struct = struct.Struct(V4_STRUCT_STRING)

    def generate_fake_pos(self):
        """
        Generate a random game position.
        Result is ([[64] * 104], [1]*5, [1858], [1], [1])
        """
        # 0. 104 binary planes of length 64
        planes = [
            np.random.randint(2, size=64).tolist() for plane in range(104)
        ]

        # 1. generate the other integer data
        integer = np.zeros(7, dtype=np.int32)
        for i in range(5):
            integer[i] = np.random.randint(2)
        integer[5] = np.random.randint(100)

        # 2. 1858 probs
        probs = np.random.randint(9, size=1858, dtype=np.int32)

        # 3. And a winner: 1, 0, -1
        winner = np.random.randint(3) - 1

        # 4. evaluation after search
        best_q = np.random.uniform(-1, 1)
        best_d = np.random.uniform(0, 1 - np.abs(best_q))
        return (planes, integer, probs, winner, best_q, best_d)

    def v4_record(self, planes, i, probs, winner, best_q, best_d):
        pl = []
        for plane in planes:
            pl.append(np.packbits(plane))
        pl = np.array(pl).flatten().tobytes()
        pi = probs.tobytes()
        root_q, root_d = 0.0, 0.0
        return self.v4_struct.pack(V4_VERSION, pi, pl, i[0], i[1], i[2], i[3],
                                   i[4], i[5], i[6], winner, root_q, best_q,
                                   root_d, best_d)

    def test_structsize(self):
        """
        Test struct size
        """
        self.assertEqual(self.v4_struct.size, 8292)

    def test_parsing(self):
        """
        Test game position decoding pipeline.
        """
        truth = self.generate_fake_pos()
        batch_size = 4
        records = []
        for i in range(batch_size):
            record = b''
            for j in range(2):
                record += self.v4_record(*truth)
            records.append(record)

        parser = ChunkParser(ChunkDataSrc(records),
                             shuffle_size=1,
                             workers=1,
                             batch_size=batch_size)
        batchgen = parser.parse()
        data = next(batchgen)

        batch = (np.reshape(np.frombuffer(data[0], dtype=np.float32),
                            (batch_size, 112, 64)),
                 np.reshape(np.frombuffer(data[1], dtype=np.int32),
                            (batch_size, 1858)),
                 np.reshape(np.frombuffer(data[2], dtype=np.float32),
                            (batch_size, 3)),
                 np.reshape(np.frombuffer(data[3], dtype=np.float32),
                            (batch_size, 3)))

        fltplanes = truth[1].astype(np.float32)
        fltplanes[5] /= 99
        for i in range(batch_size):
            data = (batch[0][i][:104],
                    np.array([batch[0][i][j][0] for j in range(104, 111)]),
                    batch[1][i], batch[2][i], batch[3][i])
            self.assertTrue((data[0] == truth[0]).all())
            self.assertTrue((data[1] == fltplanes).all())
            self.assertTrue((data[2] == truth[2]).all())
            scalar_win = data[3][0] - data[3][-1]
            self.assertTrue(np.abs(scalar_win - truth[3]) < 1e-6)
            scalar_q = data[4][0] - data[4][-1]
            self.assertTrue(np.abs(scalar_q - truth[4]) < 1e-6)

        parser.shutdown()

    def test_tensorflow_parsing(self):
        """
        Test game position decoding pipeline including tensorflow.
        """
        truth = self.generate_fake_pos()
        batch_size = 4
        ChunkParser.BATCH_SIZE = batch_size
        records = []
        for i in range(batch_size):
            record = b''
            for j in range(2):
                record += self.v4_record(*truth)
            records.append(record)

        parser = ChunkParser(ChunkDataSrc(records),
                             shuffle_size=1,
                             workers=1,
                             batch_size=batch_size)
        batchgen = parser.parse()
        data = next(batchgen)

        planes = np.frombuffer(data[0],
                               dtype=np.float32,
                               count=112 * 8 * 8 * batch_size)
        planes = planes.reshape(batch_size, 112, 8 * 8)
        probs = np.frombuffer(data[1],
                              dtype=np.float32,
                              count=1858 * batch_size)
        probs = probs.reshape(batch_size, 1858)
        winner = np.frombuffer(data[2], dtype=np.float32, count=3 * batch_size)
        winner = winner.reshape(batch_size, 3)
        best_q = np.frombuffer(data[3], dtype=np.float32, count=3 * batch_size)
        best_q = best_q.reshape(batch_size, 3)

        # Pass it through tensorflow
        with tf.compat.v1.Session() as sess:
            graph = ChunkParser.parse_function(data[0], data[1], data[2],
                                               data[3])
            tf_planes, tf_probs, tf_winner, tf_q = sess.run(graph)

            for i in range(batch_size):
                self.assertTrue((probs[i] == tf_probs[i]).all())
                self.assertTrue((planes[i] == tf_planes[i]).all())
                self.assertTrue((winner[i] == tf_winner[i]).all())
                self.assertTrue((best_q[i] == tf_q[i]).all())

        parser.shutdown()


if __name__ == '__main__':
    unittest.main()
