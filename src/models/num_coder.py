import tensorflow as tf
from src.models import util
import numpy as np
import matplotlib.pyplot as plt
import math


def values_to_digits(values,  # (nums,)
                     decimal_lengths=2,  # (nums,)
                     base=10,
                     ):
    integer_lengths = tf.floor(tf.maximum(tf.log(tf.maximum(values, 0.1))/tf.log(float(base)), 0)) + 1
    r0 = values/(float(base) ** integer_lengths)
    decimal_lengths = (decimal_lengths + 1) * tf.cast(decimal_lengths > 0, dtype=tf.int32)
    integer_lengths = tf.cast(integer_lengths, dtype=tf.int32)

    def body_decimal(_i, _j, _r, _d):
        new_r = base * _r  # shift left
        cur_d = tf.cast(tf.floor(new_r), tf.int32)  # leftmost digit
        # IF reached integer length, but NO decimal has been appended
        cur_d = tf.where(tf.logical_and(tf.equal(_i, integer_lengths), tf.equal(_j, 0)),
                         11 * tf.ones_like(_r, dtype=tf.int32),  # append DECIMAL_POINT
                         cur_d)  # append next digit
        new_r = tf.where(tf.logical_and(tf.equal(_i, integer_lengths), tf.equal(_j, 0)),
                         _r,  # pause, just extract DECIMAL_POINT
                         new_r - tf.floor(new_r))
        # IF just finished
        cur_d = tf.where(tf.logical_and(tf.equal(_i, integer_lengths), tf.equal(_j, decimal_lengths)),
                         12 * tf.ones_like(_r, dtype=tf.int32),  # append EOS
                         cur_d)
        # IF overshot
        cur_d = tf.where(tf.logical_and(tf.equal(_i, integer_lengths), _j > decimal_lengths),
                         13 * tf.ones_like(_r, dtype=tf.int32),  # append PAD
                         cur_d)
        # IF finished with integer part
        new_i = tf.where(tf.equal(_i, integer_lengths),
                         _i,
                         _i+1)
        # IF ALREADY finished with integer part
        new_j = tf.where(tf.logical_and(tf.equal(_i, integer_lengths), tf.equal(new_i, integer_lengths)),
                         _j + 1,
                         _j)
        cur_d = tf.expand_dims(cur_d, -1)  # (nums, 1)
        new_d = tf.concat([_d, cur_d], axis=-1)  # (nums, digits)
        return new_i, new_j, new_r, new_d

    i0 = tf.zeros_like(integer_lengths, dtype=tf.int32)
    j0 = tf.zeros_like(integer_lengths, dtype=tf.int32)
    d0 = tf.expand_dims(10 * tf.ones_like(values, dtype=tf.int32), -1)
    _, _, _, digits = tf.while_loop(lambda i, j, r, d: tf.logical_or(
            tf.reduce_any(tf.not_equal(i, integer_lengths)),
            tf.reduce_any(j < decimal_lengths + 2)),
                                          body_decimal,
                                          loop_vars=[i0, j0, r0, d0],
                                          shape_invariants=[i0.get_shape(),
                                                            j0.get_shape(),
                                                            r0.get_shape(),
                                                            tf.TensorShape([d0.get_shape()[0], None]),
                                                            ])
    # digits = tf.Print(digits, [values, digits, max_decimal_length], summarize=100)
    return digits


class NumCoder(object):

    def __init__(self, option=3, hidden_size=50, dtype=tf.float32, name=None):
        self.name = name if name is not None else self.__class__.__name__
        self.hidden_size = hidden_size
        self.option = option
        self.base = 10

        # initialise representation variables
        self.reuse_represent = False
        _ = self.represent(tf.constant([1.0], dtype=dtype))


    def represent(self, values,  # (nums, )
                  decimal_lengths=5,  # (nums, )
                  dtype=tf.float32):
        reuse = self.reuse_represent
        with tf.variable_scope(self.name, reuse=reuse):
            if self.option == 0:  # always zero
                codes = tf.zeros([tf.shape(values)[0], self.hidden_size])
            elif self.option == 1:  # value
                w = tf.get_variable('tw_'+self.name, [1, self.hidden_size], dtype=dtype)
                codes = tf.expand_dims(values, -1)
                codes = tf.einsum('cm,mh->ch', codes, w)
            elif self.option == 2:  # encoded seq of digits
                self.w_emb_digits = tf.get_variable("w_emb_digits", [self.base + 4, self.hidden_size], dtype=dtype)
                cell = util.create_rnn_cell('lstm', self.hidden_size, num_layers=1, reuse=reuse)
                #
                digits = values_to_digits(values, decimal_lengths, self.base)
                n_digits = tf.reduce_sum(tf.cast(tf.not_equal(digits, 13), dtype=tf.int32), -1)
                codes = tf.nn.embedding_lookup(self.w_emb_digits, digits)  # (num, n_digits, hidden)
                codes = util.get_h(tf.nn.dynamic_rnn(cell, codes, dtype=dtype, sequence_length=n_digits)[1])

        self.reuse_represent = True  # will reuse next time
        return codes  # (nums, hidden)
