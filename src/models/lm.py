import tensorflow as tf
from src.models import util
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import math
from sklearn import mixture
from src.models.num_coder import values_to_digits

IDX_PAD = 0
IDX_UNK_WORD = 1
RNG = np.random.RandomState(121455)


def shift_left_pad_right(x, pad=0, lag=1):
    y = tf.concat([x[:, lag:], 0 * x[:, -1:] + pad], axis=-1)
    return y


def infs_like(x):
    return -tf.log(tf.zeros_like(x))


def norm_log_pdf(x, loc, scale):
    return -0.5 * tf.pow((x-loc)/scale, 2.0) - tf.log(np.sqrt(2.0 * np.pi) * scale)


def norm_pdf(x, loc, scale):
    return tf.exp(norm_log_pdf(x, loc, scale))


def norm_cdf(x, loc, scale):
    return 0.5 + 0.5 * tf.erf((x-loc)/(scale*np.sqrt(2.0)))


def get_log_delta_cdf(values, rounds, locs, scales, dtype):
    delta_x = tf.pow(10.0, -tf.cast(rounds, dtype=dtype))  # B
    x_up = tf.expand_dims(values + 0.5 * delta_x, -1)
    x_down = tf.expand_dims(values - 0.5 * delta_x, -1)
    delta_x = tf.expand_dims(delta_x, -1)
    cdf_up = norm_cdf(x_up, locs, scales)
    cdf_down = norm_cdf(x_down, locs, scales)
    delta_cdf = cdf_up - cdf_down
    log_pdf = norm_log_pdf(0.5*x_up+0.5*x_down, locs, scales)
    # in extreme cases, approximate with flat pdf  # NOTE: indispensable for numerical stability
    log_delta_cdf = tf.where(delta_cdf > 0.0,
                             tf.log(delta_cdf),
                             log_pdf + tf.log(delta_x))  # flat_height * Dx
    return log_delta_cdf  # Bc


def get_decimal_pattern_and_mask(decimal_count, dtype):
    bbb = tf.shape(decimal_count)[0]
    decimal_maxlen = tf.reduce_max(decimal_count) + 3
    decimal_pattern = tf.sparse_to_dense(tf.concat([tf.expand_dims(tf.range(bbb), -1),
                                                    tf.expand_dims(decimal_count, -1)], -1),
                                         (bbb, decimal_maxlen),
                                         1 * tf.ones_like(decimal_count, dtype=tf.int32))
    decimal_mask = 1.0 - tf.cast(tf.cumsum(decimal_pattern, -1, exclusive=True) >= 1, dtype=dtype)
    decimal_pattern = tf.concat([2 * tf.ones_like(decimal_pattern[:, 0:1]),
                                 decimal_pattern], axis=-1)
    decimal_mask = tf.concat([1 * tf.ones_like(decimal_mask[:, 0:1]),
                              decimal_mask], axis=-1)
    return decimal_pattern, decimal_mask


def str_to_decimal_count(x):
    decimals = 0
    parts = x.split('.')
    if len(parts) > 1:
        decimals = len(parts[1])
    return decimals


class LM(object):

    def __init__(self, placeholders,
                 hidden_size,
                 partition_sizes=None,
                 iv_nums=None,
                 all_nums=None,
                 adjust_counts=None,
                 pretrained=None,
                 kernel_locs=None,
                 kernel_scales=None,
                 is_training=False,
                 numcoders=None,
                 output_mode=None,  # ['categorical', 'rnn', 'continuous']
                 use_loc_embs=True,
                 use_hierarchical=True,
                 dtype=tf.float32):

        dropout = 0.1
        num_layers = 1
        cell_type = 'lstm'  # gru/lstm
        use_encoder = False
        share_enc_dec_rnn = True
        balance_classes = False

        print([(a, b) for a, b in sorted(locals().items(), key=lambda x:x[0]) if a not in {'self', 'pretrained', 'iv_nums', 'all_nums', 'adjust_counts'}])

        # DEFINE INPUTS
        self._dec_tokens = placeholders['tokens']
        self._dec_numbers = placeholders['numbers']
        self._batch_floats_value = placeholders['batch_floats_value']
        self._batch_floats_ids = placeholders['batch_floats_ids']
        self._batch_floats_round = placeholders['batch_floats_round']
        if use_encoder:
            self._enc_tokens = placeholders['enc_tokens']
            self._enc_numbers = placeholders['enc_numbers']

        vocab_size = sum(partition_sizes)
        n_vocab_words = partition_sizes[0]
        n_vocab_nums = partition_sizes[1]

        if pretrained is not None:
            print('Wll use pretrained')
            pretrained = {k: v.astype(dtype='float32') for k, v in pretrained.items()}

        adjust_counts = np.maximum(np.asarray(adjust_counts, dtype='float32'), 1.0)  # avoid zeros (for stability)

        # PROCESS NUMBERS
        self.numcoders = numcoders
        iv_nums = sorted(iv_nums, key=lambda x: float(x))
        iv_floats = [float(x) for x in iv_nums]
        iv_rounds = [str_to_decimal_count(x) for x in iv_nums]
        all_nums = sorted(all_nums, key=lambda x: float(x))
        all_floats = [float(x) for x in all_nums]
        all_rounds = [str_to_decimal_count(x) for x in all_nums]
        self.all_floats = all_floats
        #print(iv_nums)
        # DEFINE EXACT NUMBERS
        exact_locs = [v for v in iv_floats] + [np.mean(all_floats)]  # last element is UNK_NUM
        exact_scales = [0.0 for _ in iv_floats] + [np.std(all_floats)]
        exact_rounds = [x for x in iv_rounds] + [0]
        # DEFINE KERNELS
        n_kernels = len(kernel_locs)
        print(n_kernels)
        assert(len(exact_locs) == n_vocab_nums)
        assert(len(kernel_locs) == len(kernel_scales))
        self.exact_locs = np.asarray(exact_locs).astype('float32')
        self.exact_rounds = np.asarray(exact_rounds).astype('int32')
        self.kernel_locs = np.asarray(kernel_locs).astype('float32')
        self.kernel_scales = np.asarray(kernel_scales).astype('float32')
        self.kernel_locs = tf.get_variable('kernel_locs', initializer=self.kernel_locs, dtype=dtype)
        self.kernel_scales = tf.get_variable('kernel_scales', initializer=self.kernel_scales, dtype=dtype)
        self.kernel_locs = tf.stop_gradient(self.kernel_locs)
        self.kernel_scales = tf.stop_gradient(self.kernel_scales)

        keep_prob = (1.0 - dropout) if is_training else 1.0
        batch_size, step_size = tf.unstack(tf.shape(self._dec_tokens))

        # DEFINE MASKS
        self._dec_mask_is_token = tf.cast(tf.not_equal(self._dec_tokens, IDX_PAD), dtype=dtype)
        self._dec_mask_has_value = tf.cast(self._dec_tokens >= n_vocab_words, dtype=dtype)
        self._dec_mask_is_word = self._dec_mask_is_token * (1.0 - self._dec_mask_has_value)

        # DEFINE TARGETS
        # target is shifted output (pad with zeros to maintain shape)
        self.target_tokens = shift_left_pad_right(self._dec_tokens)
        self.target_numbers = shift_left_pad_right(self._dec_numbers)
        self.target_values = shift_left_pad_right(tf.nn.embedding_lookup(self._batch_floats_value, self._dec_numbers))
        self.target_round = shift_left_pad_right(tf.nn.embedding_lookup(self._batch_floats_round, self._dec_numbers))
        self.target_mask_is_token = shift_left_pad_right(self._dec_mask_is_token)
        self.target_mask_has_value = shift_left_pad_right(self._dec_mask_has_value)
        self.target_mask_is_word = shift_left_pad_right(self._dec_mask_is_word)
        self.target_mask_is_unk_num = tf.cast(tf.equal(self.target_tokens, vocab_size-1), dtype=dtype)
        self.target_mask_is_iv_num = self.target_mask_has_value * (1.0-self.target_mask_is_unk_num)

        # count tokens and numbers
        self.n_tokens = tf.reduce_sum(self.target_mask_is_token)
        self.n_numbers = tf.reduce_sum(self.target_mask_has_value)
        self.n_words = tf.reduce_sum(self.target_mask_is_word)

        with tf.variable_scope("input"):
            if pretrained is None:
                self.input_embs_words = tf.get_variable("embs_words", [n_vocab_words, hidden_size], dtype=dtype)
                self.input_embs_nums = tf.get_variable("embs_nums", [n_vocab_nums, hidden_size], dtype=dtype)
            else:
                self.input_embs_words = tf.get_variable("embs_words", initializer=pretrained['words'].T, dtype=dtype)
                self.input_embs_nums = tf.get_variable("embs_nums", initializer=pretrained['nums'].T, dtype=dtype)
            self.embs_tokens = tf.concat([self.input_embs_words, self.input_embs_nums], axis=0)
            self.w_gate = tf.get_variable('w_gate', [2*hidden_size, hidden_size], dtype=dtype)
            self.b_gate = tf.get_variable('b_gate', initializer=tf.zeros([hidden_size]), dtype=dtype)

        # Calls to num coder (get value embeddings)
        self.in_embs_batch_values = numcoders[0].represent(self._batch_floats_value, self._batch_floats_round)  # ch
        self.out_embs_batch_values = numcoders[1].represent(self._batch_floats_value, self._batch_floats_round)  # ch
        exact_loc_out_embs = numcoders[1].represent(self.exact_locs, self.exact_rounds)  # ch
        exact_loc_out_embs = tf.concat([exact_loc_out_embs[:-1, :], tf.zeros_like(exact_loc_out_embs[0:1, :])], 0)
        kernel_loc_out_embs = numcoders[1].represent(self.kernel_locs)  # ch

        # to test numcoders
        self.reps_in = numcoders[0].represent(self.exact_locs, self.exact_rounds)  # ch
        self.reps_out = numcoders[1].represent(self.exact_locs, self.exact_rounds)  # ch

        with tf.variable_scope("output"):
            self.w_soft_class = tf.get_variable("w_soft_class", [hidden_size, 2], dtype=dtype)
            if pretrained is None:
                self.w_soft_words = tf.get_variable("w_soft_words", [hidden_size, n_vocab_words], dtype=dtype)
            else:
                self.w_soft_words = tf.get_variable("w_soft_words", initializer=pretrained['words'], dtype=dtype)
            self.b_soft_class = tf.get_variable("b_soft_class", initializer=tf.zeros([2]), dtype=dtype)
            self.b_soft_words = tf.get_variable('b_soft_words', initializer=tf.zeros([n_vocab_words]), dtype=dtype)
            self.w_soft_mode = tf.get_variable("w_soft_mode", [hidden_size, 3], dtype=dtype)
            self.b_soft_mode = tf.get_variable("b_soft_mode", initializer=tf.zeros([3]), dtype=dtype)
            self.w_soft_nums = tf.get_variable("w_soft_nums", [hidden_size, n_vocab_nums], dtype=dtype)
            self.b_soft_nums = tf.get_variable('b_soft_nums', initializer=tf.zeros([n_vocab_nums]), dtype=dtype)
            self.w_soft_kernels = tf.get_variable("w_soft_kernels", [hidden_size, n_kernels], dtype=dtype)
            self.b_soft_kernels = tf.get_variable('b_soft_kernels', initializer=tf.zeros([n_kernels]), dtype=dtype)

        if use_encoder:
            # DEFINE MASKS
            enc_mask_is_token = tf.cast(tf.not_equal(self._enc_tokens, IDX_PAD), dtype=dtype)
            enc_mask_has_value = tf.cast(self._enc_tokens >= n_vocab_words, dtype=dtype)
            enc_mask_is_word = enc_mask_is_token * (1.0 - enc_mask_has_value)
            # GET ENCODER EMBEDDINGS
            enc_embs_tokens = tf.nn.embedding_lookup(self.embs_tokens, self._enc_tokens)  # bth
            enc_embs_values = tf.nn.embedding_lookup(self.in_embs_batch_values, self._enc_numbers)  # bth
            gate = tf.sigmoid(tf.einsum('btd,dh->bth', tf.concat([enc_embs_tokens, enc_embs_values], -1), self.w_gate) + self.b_gate)
            gate *= tf.expand_dims(self._enc_mask_has_value, -1)
            enc_embs = (1.0-gate) * enc_embs_tokens + gate * enc_embs_values
            enc_embs = tf.nn.dropout(enc_embs, keep_prob)
            # ENCODER RNN
            with tf.variable_scope('encoder' if not share_enc_dec_rnn else 'decoder'):
                enc_cell = util.create_rnn_cell(cell_type, hidden_size, num_layers)
                enc_outputs, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embs,
                                                           initial_state=enc_cell.zero_state(batch_size, dtype=dtype),
                                                           sequence_length=tf.reduce_sum(enc_mask_is_token, axis=-1))
                enc_outputs = tf.nn.dropout(enc_outputs, keep_prob)

        # GET DECODER EMBEDDINGS
        dec_embs_tokens = tf.nn.embedding_lookup(self.embs_tokens, self._dec_tokens)  # bth
        dec_embs_values = tf.nn.embedding_lookup(self.in_embs_batch_values, self._dec_numbers)  # bth
        gate = tf.sigmoid(tf.einsum('btd,dh->bth', tf.concat([dec_embs_tokens, dec_embs_values], -1), self.w_gate) + self.b_gate)
        # if g==1 use num embedding else g = 0 use token embedding
        gate *= tf.expand_dims(self._dec_mask_has_value, -1)  # clamp gate to 0 for non-numerics
        dec_embs = (1.0-gate) * dec_embs_tokens + gate * dec_embs_values
        dec_embs = tf.nn.dropout(dec_embs, keep_prob)

        # (for dev) gates for IV
        _embs_token = tf.nn.embedding_lookup(self.embs_tokens, n_vocab_words + tf.range(n_vocab_nums, dtype=tf.int32))  # ch
        _embs_value = numcoders[0].represent(self.exact_locs, self.exact_rounds)  # ch
        self.g = tf.sigmoid(tf.einsum('cd,dh->ch', tf.concat([_embs_token, _embs_value], -1), self.w_gate) + self.b_gate)  # use token when g==0, use num when g==1

        # DECODER RNN
        dec_cell = util.create_rnn_cell(cell_type, hidden_size, num_layers, reuse=use_encoder and share_enc_dec_rnn)
        with tf.variable_scope('decoder', reuse=use_encoder and share_enc_dec_rnn):
            self._dec_init_state = enc_state if use_encoder else dec_cell.zero_state(batch_size, dtype=dtype)
            dec_outputs, dec_state = tf.nn.dynamic_rnn(dec_cell, dec_embs,
                                                       initial_state=self._dec_init_state,
                                                       sequence_length=tf.reduce_sum(self._dec_mask_is_token, axis=-1))
            dec_outputs = tf.nn.dropout(dec_outputs, keep_prob)
        # PARTITIONS
        partitions = tf.cast(tf.reshape(self.target_mask_has_value, [-1]), dtype=tf.int32)
        stitch_indices = tf.dynamic_partition(tf.range(batch_size * step_size), partitions, 2)
        part_dec_outputs = tf.dynamic_partition(tf.reshape(dec_outputs, [-1, hidden_size]),
                                                partitions, 2)  # [Ah, Bh]
        part_target_tokens = tf.dynamic_partition(tf.reshape(self.target_tokens, [-1]),
                                                  partitions, 2)  # [A, B]
        part_target_round = tf.dynamic_partition(tf.reshape(self.target_round, [-1]),
                                                 partitions, 2)  # [A, B]
        part_target_values = tf.dynamic_partition(tf.reshape(self.target_values, [-1]),
                                                  partitions, 2)  # [A, B]
        part_target_mask_is_token = tf.dynamic_partition(tf.reshape(self.target_mask_is_token, [-1]),
                                                         partitions, 2)  # [A, B]
        part_target_numbers = tf.dynamic_partition(tf.reshape(self.target_numbers, [-1]),
                                                  partitions, 2)  # [A, B]
        self.target_mask_is_unk_num = tf.dynamic_partition(tf.reshape(self.target_mask_is_unk_num, [-1]),
                                                  partitions, 2)[1]  # [A, B]
        #part_dec_outputs[1] = tf.Print(part_dec_outputs[1], [tf.shape(part_target_tokens[1])[0]], summarize=1000)
        # part_dec_outputs[1] = tf.Print(part_dec_outputs[1], [part_target_tokens[1], part_target_values[1]], summarize=1000)

        if not use_hierarchical:
            part_dec_outputs = tf.reshape(dec_outputs, [-1, hidden_size])
            part_dec_outputs = [part_dec_outputs, part_dec_outputs]
            part_target_tokens = tf.reshape(self.target_tokens, [-1])
            part_target_tokens = [part_target_tokens, part_target_tokens]
            part_target_mask_is_token = tf.reshape(self.target_mask_is_token, [-1])
            part_target_mask_is_token = [part_target_mask_is_token, part_target_mask_is_token]
            #part_target_values = tf.reshape(self.target_values, [-1])
            #part_target_values = [part_target_values, part_target_values]
            #part_target_round = tf.reshape(self.target_round, [-1])
            #part_target_round = [part_target_round, part_target_round]

        # calculate logits
        logits_a_class = tf.einsum('ah,hc->ac', part_dec_outputs[0], self.w_soft_class) + self.b_soft_class
        logits_a_words = tf.einsum('ah,hv->av', part_dec_outputs[0], self.w_soft_words) + self.b_soft_words
        logits_b_class = tf.einsum('bh,hc->bc', part_dec_outputs[1], self.w_soft_class) + self.b_soft_class
        logits_b_mode = tf.einsum('bh,hi->bi', part_dec_outputs[1], self.w_soft_mode) + self.b_soft_mode

        # set unused mode logits to -inf
        if 'categorical' not in output_mode:
            logits_b_mode += tf.log(tf.one_hot(indices=[0], depth=3, on_value=0.0, off_value=1.0))
        elif 'rnn' not in output_mode:
            logits_b_mode += tf.log(tf.one_hot(indices=[1], depth=3, on_value=0.0, off_value=1.0))
        elif 'continuous' not in output_mode:
            logits_b_mode += tf.log(tf.one_hot(indices=[2], depth=3, on_value=0.0, off_value=1.0))

        self.logits_exact = sum([
            tf.einsum('bh,hv->bv', part_dec_outputs[1], self.w_soft_nums),
            tf.einsum('bh,vh->bv', part_dec_outputs[1], exact_loc_out_embs) if use_loc_embs else 0.0,
            self.b_soft_nums
        ])  # Bc

        self.logits_kernels = sum([
            tf.einsum('bh,hv->bv', part_dec_outputs[1], self.w_soft_kernels),
            # tf.einsum('bh,vh->bv', part_dec_outputs[1], kernel_loc_out_embs) if use_loc_embs else 0.0,
            self.b_soft_kernels
        ])  # Bc

        # calculate losses
        losses_a_words = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_a_words,
                                                                        labels=tf.minimum(part_target_tokens[0],
                                                                                          n_vocab_words - 1))
        losses_a_class = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_a_class,
                                                                        labels=tf.zeros_like(logits_a_class[:, 0], dtype=tf.int32))
        losses_b_class = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_b_class,
                                                                        labels=tf.ones_like(logits_b_class[:, 0], dtype=tf.int32))
        log_prob_mode = tf.nn.log_softmax(logits_b_mode)  # B3

        # MODE: CATEGORICAL
        if not ('categorical' in output_mode and len(output_mode) == 1):
            # if mode != categorical, set logit for UNK_NUM to -inf (there is no class for unknown numbers)
            self.logits_exact = tf.concat([self.logits_exact[:, :-1], -infs_like(self.logits_exact[:, -1:])], -1)
        log_lik_categorical = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_exact,
                                                                              labels=tf.maximum(part_target_tokens[1] - n_vocab_words, 0))

        if not use_hierarchical:
            losses_a_class, losses_b_class = 0.0, 0.0
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.concat([logits_a_words, self.logits_exact], -1),
                                                                  labels=part_target_tokens[1])
            mmm = tf.reshape(self.target_mask_has_value * self.target_mask_is_token, [-1])
            log_lik_categorical = -xent * mmm
            mmm = tf.reshape((1.0-self.target_mask_has_value) * self.target_mask_is_token, [-1])
            losses_a_words = xent * mmm

        # MODE: RNN
        self.w_digits = tf.get_variable("w_digits", [14, hidden_size], dtype=dtype)  # 0-9, 'SOS', '.', 'EOS', 'PAD'
        self.w_soft_digits = tf.get_variable("w_soft_digits", [hidden_size, 14], dtype=dtype)
        self.b_soft_digits = tf.get_variable('b_soft_digits', initializer=tf.zeros([14]), dtype=dtype)
        self.prob_digits = tf.ones([1, 1, 14], dtype=dtype)
        self.target_digits = tf.zeros([1, 1], dtype=tf.int32)
        if 'rnn' in output_mode:
            batch_floats_digits = values_to_digits(self._batch_floats_value, self._batch_floats_round)  # (nums, lens) nl
            part_dec_nums = tf.dynamic_partition(tf.reshape(shift_left_pad_right(self._dec_numbers), [-1]),
                                                 partitions, 2)  # B
            digits = tf.nn.embedding_lookup(batch_floats_digits, part_dec_nums[1])  # Bl
            mask_digits = tf.cast(digits < 13, dtype=dtype)  # Bl
            enc_digits = tf.nn.embedding_lookup(self.w_digits, digits)  # Blh
            target_digits = shift_left_pad_right(digits, pad=13)
            target_mask_digits = shift_left_pad_right(mask_digits)
            with tf.variable_scope('numrnn'):
                num_cell = util.create_rnn_cell(cell_type, hidden_size, num_layers)
                initial_state = tf.contrib.rnn.LSTMStateTuple(h=part_dec_outputs[1],  # Bh
                                                              c=part_dec_outputs[1] * 0.0)
                h_digits, _ = tf.nn.dynamic_rnn(num_cell, enc_digits,
                                                initial_state=initial_state,
                                                sequence_length=tf.reduce_sum(mask_digits, axis=-1))
                logits_digits = tf.einsum('blh,hc->blc', h_digits, self.w_soft_digits) + self.b_soft_digits
                xent_digits = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_digits,
                                                                             labels=target_digits)  # Bl
                log_lik_rnn = -tf.reduce_sum(xent_digits * target_mask_digits, -1)  # B
                #xent_nums = tf.dynamic_stitch(stitch_indices, [part_dec_outputs[0][:, 0] * 0.0, xent_nums])
                #xent_nums = tf.reshape(tf.expand_dims(xent_nums, -1), [batch_size, step_size])
                self.prob_digits = tf.nn.softmax(logits_digits)
                self.target_digits = target_digits

        # MODE: CONTINUOUS
        self.w_pattern = tf.get_variable("w_pattern", [3, hidden_size], dtype=dtype)
        self.w_soft_pattern = tf.get_variable("w_soft_pattern", [hidden_size, 3], dtype=dtype)
        self.b_soft_pattern = tf.get_variable('b_soft_pattern', initializer=tf.zeros([3]), dtype=dtype)
        if 'continuous' in output_mode:
            decimal_pattern, decimal_mask = get_decimal_pattern_and_mask(part_target_round[1], dtype)
            # decimal_mask = tf.Print(decimal_mask, [part_target_values[1], part_target_round[1], decimal_pattern, decimal_mask], summarize=1000)
            enc_decimal_pattern = tf.nn.embedding_lookup(self.w_pattern, decimal_pattern)  # Blh
            target_decimal_pattern = shift_left_pad_right(decimal_pattern)
            target_decimal_mask = shift_left_pad_right(decimal_mask)
            with tf.variable_scope('pattern'):
                num_cell = util.create_rnn_cell(cell_type, hidden_size, num_layers)
                initial_state = tf.contrib.rnn.LSTMStateTuple(h=part_dec_outputs[1],  # Bh
                                                              c=part_dec_outputs[1] * 0.0)
                h_pattern, _ = tf.nn.dynamic_rnn(num_cell, enc_decimal_pattern,
                                                 initial_state=initial_state,
                                                 sequence_length=tf.reduce_sum(decimal_mask, axis=-1))
                decimal_pattern_logits = tf.einsum('blh,hc->blc', h_pattern, self.w_soft_pattern) + self.b_soft_pattern
                xent_decimal_pattern = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decimal_pattern_logits,
                                                                                      labels=target_decimal_pattern)  # Bl
                log_prob_decimal_pattern = -tf.reduce_sum(xent_decimal_pattern * target_decimal_mask, -1)  # B

            log_delta_cdf = get_log_delta_cdf(part_target_values[1], part_target_round[1],
                                              self.kernel_locs, self.kernel_scales,
                                              dtype=dtype)  # Bc
            log_lik_continuous = tf.reduce_logsumexp(sum([
                tf.expand_dims(log_prob_decimal_pattern, -1),
                tf.nn.log_softmax(self.logits_kernels),
                log_delta_cdf,
            ]), -1)  # B
            #log_lik_continuous = tf.Print(log_lik_continuous, [tf.reduce_max(tf.exp(log_delta_cdf))], summarize=1000)
            #log_lik_continuous = tf.Print(log_lik_continuous, [tf.reduce_mean(tf.cast(tf.equal(delta_cdf, 0.0), dtype=dtype))], summarize=1000)

        def combine_num_losses(do_adjustment=False):
            unk_bin_adjustment = 0.0
            if do_adjustment:
                unk_bin_adjustment = tf.nn.embedding_lookup(np.log(1.0 / adjust_counts), part_target_tokens[1])
            # gather losses for each output mode
            total_mode_logits = []
            if 'categorical' in output_mode:
                total_mode_logits.append(log_prob_mode[:, 0] + log_lik_categorical + unk_bin_adjustment)
            if 'rnn' in output_mode:
                total_mode_logits.append(log_prob_mode[:, 1] + log_lik_rnn)
            if 'continuous' in output_mode:
                total_mode_logits.append(log_prob_mode[:, 2] + log_lik_continuous)
            # combine losses for output modes
            if len(total_mode_logits) == 1:
                return -total_mode_logits[0]
            else:
                return -tf.reduce_logsumexp(tf.concat([tf.expand_dims(x, -1) for x in total_mode_logits], -1), -1)

        losses_nums = combine_num_losses(False)
        losses_nums_adjusted = combine_num_losses('categorical' in output_mode and len(output_mode) == 1)

        # aggregate total batch loss
        weight_at_words, weight_at_nums = 0.5, 0.5
        if not balance_classes:
            weight_at_words = self.n_words/self.n_tokens
            weight_at_nums = self.n_numbers/self.n_tokens

        adj_words = tf.nn.embedding_lookup(np.log(1.0 / adjust_counts), part_target_tokens[0])
        nll_words = (losses_a_class + losses_a_words) * part_target_mask_is_token[0]
        nll_nums = (losses_b_class + losses_nums) * part_target_mask_is_token[1]
        nll_adj_words = (losses_a_class + losses_a_words - adj_words) * part_target_mask_is_token[0]
        nll_adj_nums = (losses_b_class + losses_nums_adjusted) * part_target_mask_is_token[1]

        total_losses = sum([
            tf.reduce_sum(nll_words * weight_at_words / self.n_words),
            tf.reduce_sum(nll_nums * weight_at_nums / self.n_numbers),
        ])

        #total_losses = tf.Print(total_losses, [log_lik_categorical, losses_nums], summarize=1000)
        self.cost_total = total_losses

        # TESTING NEGATIVE LOG LIKELIHOOD
        # for perplexity
        self.nll_total_words = tf.reduce_sum(nll_words)
        self.nll_total_numbers = tf.reduce_sum(nll_nums)
        self.nll_total_tokens = self.nll_total_words + self.nll_total_numbers

        # for adjusted perplexity
        self.nll_adj_total_words = tf.reduce_sum(nll_adj_words)
        self.nll_adj_total_numbers = tf.reduce_sum(nll_adj_nums)
        self.nll_adj_total_tokens = self.nll_adj_total_words + self.nll_adj_total_numbers

        # TESTING MODES
        logits_exact = self.logits_exact
        if not use_hierarchical:
            part_dec_outputs = tf.dynamic_partition(tf.reshape(dec_outputs, [-1, hidden_size]),
                                                    partitions, 2)  # [Ah, Bh]
            logits_b_mode = tf.dynamic_partition(tf.reshape(logits_b_mode, [-1, 3]),
                                                 partitions, 2)[1]  # [A3, B3]
            logits_exact = tf.dynamic_partition(tf.reshape(logits_exact, [-1, n_vocab_nums]),
                                                partitions, 2)[1]  # [Av, Bv]

        self.prob_mode = tf.nn.softmax(logits_b_mode)
        self.pred_mode = util.make_hard(self.prob_mode)
        #self.prob_mode = tf.Print(self.prob_mode, [self.prob_mode], summarize=1000)
        #if False:
        #    pred_mode = tf.Print(pred_mode, [tf.concat([
        #        tf.expand_dims(tf.cast(tf.argmax(pred_mode, -1), dtype=dtype), -1),
        #        tf.expand_dims(part_target_values[1], -1),
        #    ], -1)], summarize=1000)
        self.prob_is_unk_num = 1.0 - self.prob_mode[:, 0]
        if 'categorical' in output_mode and len(output_mode) == 1:
            self.prob_is_unk_num = tf.nn.softmax(logits_exact)[:, -1]

        # TESTING PDF
        #  get candidates
        e1 = iv_floats
        r2count = defaultdict(int)
        for r in all_rounds:
        #for r in [all_rounds[iii] for iii in np.unique(all_nums, return_index=True)[1]]:
            r2count[r] += 1
        r2count = sorted(r2count.items(), key=lambda x: x[0])
        print('count analysis')
        print(r2count)

        float2rounds = defaultdict(set)
        for x in all_nums:
            float2rounds[float(x)].add(str_to_decimal_count(x))
        some_nums = [all_nums[iii] for iii in np.linspace(0, len(all_nums)-1, 100, endpoint=True, dtype='int32')]
        some_nums = sorted(list(set(some_nums)), key=lambda x: float(x))
        e2, r2 = [], []
        for x in some_nums:
            x = float(x)
            for _r in float2rounds[x]:
                e2.append(float(x))
                r2.append(_r)

        #print(list(zip(some_nums, e2, r2)))
        n_candidates = len(e1) + len(e2)
        self.cand_values = tf.constant(e1 + e2)
        self.cand_rounds = tf.constant(iv_rounds + r2)

        num_size = tf.shape(part_dec_outputs[1])[0]
        # MODE (predict): categorical
        cand_logits_cat = tf.log(tf.zeros([num_size, n_candidates], dtype=dtype))
        if 'categorical' in output_mode and not is_training:
            cand_logits_cat = tf.nn.log_softmax(logits_exact)
            cand_logits_cat = tf.concat([cand_logits_cat[:, :-1],
                                         tf.log(tf.zeros([num_size, len(e2)], dtype=dtype))], axis=-1)
            #print(cand_logits_cat)
        # MODE (predict): RNN
        cand_logits_rnn = tf.log(0.0)
        if 'rnn' in output_mode and not is_training:
            with tf.variable_scope('numrnn', reuse=True):
                digits = values_to_digits(self.cand_values, self.cand_rounds)  # (nums, lens) nl
                digits = tf.tile(tf.expand_dims(digits, 0), [num_size, 1, 1])  # Bnl
                digits = tf.reshape(digits, [num_size * n_candidates, -1])  # Bl
                mask_digits = tf.cast(digits < 13, dtype=dtype)  # Bl
                enc_digits = tf.nn.embedding_lookup(self.w_digits, digits)  # Blh
                target_digits = shift_left_pad_right(digits)
                target_mask_digits = shift_left_pad_right(mask_digits)
                initial_state = tf.reshape(tf.tile(tf.expand_dims(part_dec_outputs[1], 1), [1, n_candidates, 1]), [-1, hidden_size])  # Bh
                initial_state = tf.contrib.rnn.LSTMStateTuple(c=0.0 * initial_state, h=initial_state)
                num_outputs, _ = tf.nn.dynamic_rnn(num_cell, enc_digits,
                                                   initial_state=initial_state,
                                                   sequence_length=tf.reduce_sum(mask_digits, axis=-1))
                rnn_num_logits = tf.einsum('blh,hc->blc', num_outputs, self.w_soft_digits) + self.b_soft_digits
                cand_logits_rnn = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rnn_num_logits,
                                                                                  labels=target_digits)  # Bl
                cand_logits_rnn = tf.reduce_sum(cand_logits_rnn * target_mask_digits, -1)  # B
                cand_logits_rnn = tf.reshape(tf.expand_dims(cand_logits_rnn, -1), [num_size, n_candidates])

        # MODE (predict): continuous
        cand_logits_cont = tf.log(0.0)
        if 'continuous' in output_mode and not is_training:
            # calculate p(r)
            n_r = tf.reduce_max(self.cand_rounds)+1
            rrr = tf.range(n_r)
            decimal_count = tf.tile(tf.expand_dims(rrr, 0), [num_size, 1])  # Br
            decimal_count = tf.reshape(decimal_count, [num_size * n_r])  # B'
            decimal_pattern, decimal_mask = get_decimal_pattern_and_mask(decimal_count, dtype)
            enc_decimal_pattern = tf.nn.embedding_lookup(self.w_pattern, decimal_pattern)  # B'lh
            target_decimal_pattern = shift_left_pad_right(decimal_pattern)
            target_decimal_mask = shift_left_pad_right(decimal_mask)
            with tf.variable_scope('pattern', reuse=True):
                num_cell = util.create_rnn_cell(cell_type, hidden_size, num_layers)
                initial_state = tf.reshape(tf.tile(tf.expand_dims(part_dec_outputs[1], 1), [1, n_r, 1]), [-1, hidden_size])  # B'h
                initial_state = tf.contrib.rnn.LSTMStateTuple(c=0.0 * initial_state, h=initial_state)
                h_pattern, _ = tf.nn.dynamic_rnn(num_cell, enc_decimal_pattern,
                                                 initial_state=initial_state,
                                                 sequence_length=tf.reduce_sum(decimal_mask, axis=-1))
                decimal_pattern_logits = tf.einsum('blh,hc->blc', h_pattern, self.w_soft_pattern) + self.b_soft_pattern
                xent_decimal_pattern = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decimal_pattern_logits,
                                                                                      labels=target_decimal_pattern)  # Bl
                log_prob_decimal_pattern = -tf.reduce_sum(xent_decimal_pattern * target_decimal_mask, -1)  # B'
                log_prob_decimal_pattern = tf.reshape(tf.expand_dims(log_prob_decimal_pattern, -1), [-1, n_r])  # Br
            log_prob_decimal_pattern = tf.transpose(tf.nn.embedding_lookup(tf.transpose(log_prob_decimal_pattern), self.cand_rounds))  # Bn
            #log_prob_decimal_pattern = tf.Print(log_prob_decimal_pattern, [100*tf.nn.softmax(log_prob_decimal_pattern)], summarize=1000)
            # calculate p(v|r)
            log_delta_cdf = get_log_delta_cdf(self.cand_values, self.cand_rounds,
                                              self.kernel_locs, self.kernel_scales,
                                              dtype=dtype)  # nc
            cand_logits_cont = tf.reduce_logsumexp(sum([
                tf.expand_dims(log_prob_decimal_pattern, -1),  # Bn1
                tf.expand_dims(tf.nn.log_softmax(self.logits_kernels), -2),  # B1c
                tf.expand_dims(log_delta_cdf, 0),  # 1nc
            ]), -1)  # Bn

        # combine probs from all modes
        self.probs_cands = sum([
            tf.expand_dims(self.prob_mode[:, 0], -1) * tf.exp(cand_logits_cat),  # Bn or B1
            tf.expand_dims(self.prob_mode[:, 1], -1) * tf.exp(cand_logits_rnn),
            tf.expand_dims(self.prob_mode[:, 2], -1) * tf.exp(cand_logits_cont),
        ])  # Bn
        self.pred_values = tf.einsum('be,e->b', util.make_hard(self.probs_cands), self.cand_values)  # B
        self.target_values = part_target_values[1]
        self.target_round = part_target_round[1]

        self.prob_kernel = tf.nn.softmax(self.logits_kernels)

