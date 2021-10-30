#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2021 Leela Chess Authors
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
import tensorflow as tf
from chunkparser import ChunkParser

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
