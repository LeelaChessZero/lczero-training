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

SKIP_MULTIPLE = 1024


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
