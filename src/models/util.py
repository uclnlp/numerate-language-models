import numpy as np
import tensorflow as tf


def print_variable_report():
    print('-----Variable report-----')
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        print('%10d\t%s\t' % (variable_parametes, variable.name), shape)
        total_parameters += variable_parametes
    print('Total params:', total_parameters)


def build_optimiser(loss, optim, l2=0.0, clip=None, clip_op=tf.clip_by_value):
    #optim = tf.train.AdamOptimizer(learning_rate=0.001)
    if l2 != 0.0:
        loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2
    gradients = optim.compute_gradients(loss)
    capped_gradients = gradients
    if clip is not None:
        if clip_op == tf.clip_by_value:
            capped_gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var) for grad, var in gradients]
        elif clip_op == tf.clip_by_norm:
            capped_gradients = [(tf.clip_by_norm(grad, clip), var) for grad, var in gradients]
    min_op = optim.apply_gradients(capped_gradients)
    #min_op = optim.minimize(loss)
    grad_norm = tf.add_n([tf.nn.l2_loss(g) for g, v in gradients if g is not None])
    return min_op, loss, grad_norm

def get_h(state):
    #print(state)
    try:
        h = state.h
        #h = state[0].h  # TODO support multicell
    except AttributeError:
        h = state
    return h


def apply_mask(w, mask, default=0.0):
    #assert(len(w.get_shape()) == len(mask.get_shape()))  # ASSUMES shapes are the same
    masked = w * mask
    if default != 0.0:
        if default == -np.inf:
            default = -tf.reduce_max(w)-2000.0
        masked += (1.0-mask) * default
    return masked


def mask_for_lengths(lengths,
                     batch_size=None, max_length=None,
                     dim2=None, mask_right=True,
                     value=-1000.0):
    """
    Creates a [batch_size x max_length] mask.
    :param lengths: int32 1-dim tensor of batch_size lengths
    :param batch_size: int32 0-dim tensor or python int
    :param max_length: int32 0-dim tensor or python int
    :param mask_right: if True, everything before "lengths" becomes zero and the
        rest "value", else vice versa
    :param value: value for the mask
    :return: [batch_size x max_length] mask of zeros and "value"s
    """
    if max_length is None:
        max_length = tf.reduce_max(lengths)
    if batch_size is None:
        batch_size = tf.shape(lengths)[0]

    if dim2 != None:
        # [batch_size x max_length x dim2]
        mask = tf.reshape(tf.tile(tf.range(0, max_length), [batch_size * dim2]), tf.pack([batch_size, max_length, dim2]))
        if mask_right:
            mask = tf.greater_equal(mask, tf.expand_dims(tf.expand_dims(lengths, 1), 1))
        else:
            mask = tf.less(mask, tf.expand_dims(tf.expand_dims(lengths, 1), 1))
    else:
        # [batch_size x max_length]
        mask = tf.reshape(tf.tile(tf.range(0, max_length), [batch_size]), tf.pack([batch_size, -1]))
        if mask_right:
            mask = tf.greater_equal(mask, tf.expand_dims(lengths, 1))
        else:
            mask = tf.less(mask, tf.expand_dims(lengths, 1))
    # otherwise we return a boolean mask
    if value != 0:
        mask = tf.cast(mask, tf.float32) * value
    return mask



def get_mask(x, value=0.0):
    """
    Creates a [batch_size x max_length] mask.
    :param value: value for the mask
    :return: [batch_size x max_length] mask of zeros and "value"s
    """
    mask = tf.greater(x, 0)
    mask = tf.cast(mask, tf.float32)
    # otherwise we return a boolean mask
    if value != 0.0:
        mask *= value
    return mask


def between(x, low, high, dtype=tf.float32):
    a = tf.cast(tf.greater_equal(x, low), dtype=dtype)
    b = tf.cast(tf.less_equal(x, high), dtype=dtype)
    return a*b


def embed(embeddings, tokens,
          numbers=None, number_dims=0,
          keep_prob=1.0, dtype=tf.float32):
    emb_tokens = tf.nn.embedding_lookup(embeddings, tokens)
    #emb_tokens = tf.gather(embeddings, tokens)  # embed sentence tokens
    if keep_prob < 1:
        emb_tokens = tf.nn.dropout(emb_tokens, keep_prob)
    embedded = emb_tokens
    if number_dims > 0:
        assert(numbers is not None)
        emb_numbers = tf.expand_dims(numbers, -1)
        embedded = tf.concat([emb_numbers, emb_tokens], -1)
    return embedded


def create_rnn_cell(cell_type, hidden_size, num_layers=1, activation=tf.nn.tanh, keep_prob=1.0, reuse=False):
    # Create the internal multi-layer cell for our RNN.
    state_is_tuple = False
    if cell_type == 'lstm':
        state_is_tuple = True
        def single_cell():
            #return tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=state_is_tuple, reuse=reuse)
            return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=state_is_tuple, reuse=reuse, activation=activation)
    elif cell_type == 'gru':
        def single_cell():
            return tf.contrib.rnn.GRUCell(hidden_size, reuse=reuse)
    elif cell_type == 'simple':
        def single_cell():
            return tf.contrib.rnn.BasicRNNCell(hidden_size, reuse=reuse)
    else:
        raise Exception('Cell type not supported')
    # Dropout for cell
    dropped_single_cell = single_cell
    if keep_prob < 1:
        def dropped_single_cell():
            return tf.contrib.rnn.DropoutWrapper(single_cell(), output_keep_prob=keep_prob)
    if num_layers == 1:
        cell = dropped_single_cell()
    else: # Stack cells
        cell = tf.contrib.rnn.MultiRNNCell([dropped_single_cell() for _ in range(num_layers)],
                                           state_is_tuple=state_is_tuple)
    return cell


def get_attention_fn(attended, a_size, att_v, att_w2, att_w=None, mask=None, normalised=True, dtype=tf.float32):
    # Encode attention_inputs for attention
    if att_w is None:
        batch_size, _, _ = tf.unstack(tf.shape(attended))
        hidden_size = attended._shape_as_list()[-1]
        part1 = tf.reshape(attended, [batch_size, -1, 1, hidden_size])
        part1 = tf.contrib.layers.conv2d(part1, num_outputs=a_size, kernel_size=(1, 1))
        part1 = tf.squeeze(part1, 2)
        part1 = tf.transpose(part1, perm=[0, 2, 1])  # move time to last dim
    else:
        part1 = tf.einsum('bth,ha->bat', attended, att_w)
    if mask is not None:
        _mask = tf.expand_dims(mask, -1)
        _mask = tf.transpose(_mask, [0, 2, 1])
        part1 = tf.multiply(part1, _mask)

    def attention_fn(guide=None):
        att_activ = part1
        if guide is not None:
            guide = tf.einsum('bh,ha->ba', guide, att_w2)
            guide = tf.expand_dims(guide, -1)
            att_activ = part1 + guide
        att_activ = tf.nn.tanh(att_activ)
        att_logits = tf.einsum('bat,a->bt', att_activ, att_v)
        if normalised:
            att_probs = tf.nn.softmax(att_logits)
        else:
            att_probs = tf.nn.sigm(att_logits)
        att = tf.multiply(tf.expand_dims(att_probs, -1), attended)
        att = tf.reduce_sum(att, axis=1)
        return att, att_probs
    return attention_fn


def get_attention_fn2(a_size, att_v, att_w2, att_w=None, normalised=True, dtype=tf.float32):
    def attention_fn(attended, guide=None, mask=None):
        # Encode attention_inputs for attention
        if att_w is None:
            batch_size, _, _ = tf.unstack(tf.shape(attended))
            hidden_size = attended._shape_as_list()[-1]
            part1 = tf.reshape(attended, [batch_size, -1, 1, hidden_size])
            part1 = tf.contrib.layers.conv2d(part1, num_outputs=a_size, kernel_size=(1, 1))
            part1 = tf.squeeze(part1, 2)
            part1 = tf.transpose(part1, perm=[0, 2, 1])  # move time to last dim
        else:
            part1 = tf.einsum('bth,ha->bat', attended, att_w)
        if mask is not None:
            _mask = tf.expand_dims(mask, -1)
            _mask = tf.transpose(_mask, [0, 2, 1])
            part1 = tf.multiply(part1, _mask)
        att_activ = part1
        if guide is not None:
            guide = tf.einsum('bh,ha->ba', guide, att_w2)
            guide = tf.expand_dims(guide, -1)
            att_activ = part1 + guide
        att_activ = tf.nn.tanh(att_activ)
        att_logits = tf.einsum('bat,a->bt', att_activ, att_v)
        if normalised:
            att_probs = tf.nn.softmax(att_logits)
        else:
            att_probs = tf.nn.sigm(att_logits)
        att = tf.multiply(tf.expand_dims(att_probs, -1), attended)
        att = tf.reduce_sum(att, axis=1)
        return att, att_probs
    return attention_fn



def get_decoder_fn(embeddings, softmax_w, softmax_b, make_hard=True, dtype=tf.float32):
    def decoder_fn(state):
        # softmax output
        logits = tf.einsum('bh,hv->bv', state, softmax_w) + softmax_b
        vocab_prob = tf.nn.softmax(logits)
        if make_hard:
            # input is embedding of argmax
            next_input = tf.gather(embeddings, tf.argmax(vocab_prob, -1))
        else:
            # input is expectation over embeddings
            next_input = tf.einsum('bv,vh->bh', vocab_prob, embeddings)
        return next_input
    return decoder_fn


def get_pred_numeric_from_vocab(pred_token_probs,  # (batch, time, vocab)
                                vocab_numbers,  # (vocab, )
                                hard=False
                                ):
    batch_size, max_time, _ = tf.unstack(tf.shape(pred_token_probs))
    # Expectation{ mu(number|word) * p(word) }
    # re-normalise
    pred_num_probs = pred_token_probs/tf.reduce_sum(pred_token_probs, axis=-1, keep_dims=True)

    if hard:
        # make hard and predict with argmax value
        pred_num_probs = tf.where(tf.equal(pred_num_probs, tf.reduce_max(pred_num_probs, axis=-1, keep_dims=True)),
                                  tf.ones_like(pred_num_probs),
                                  tf.zeros_like(pred_num_probs))
        pred_numbers = tf.einsum('btv,v->bt', pred_num_probs, vocab_numbers)
        # sample  # TODO
        # predict with median  # FIXME
        #pred_numbers = get_median(vocab_numbers, pred_num_probs, 0.5)  # (batch, time)
        # predict with expected value
    else:
        pred_numbers = tf.einsum('btv,v->bt', pred_num_probs, vocab_numbers)
    return pred_numbers


def get_median(sorted_numbers, sorted_weights, alpha=0.5, dtype=tf.float32):
    total_weight = tf.reduce_sum(sorted_weights, axis=-1, keep_dims=True)
    upper_mask = tf.greater_equal(tf.cumsum(sorted_weights, axis=-1), total_weight*alpha)
    lower_mask = tf.greater_equal(tf.cumsum(sorted_weights, axis=-1, reverse=True), total_weight*alpha)
    median_mask = tf.cast(upper_mask, dtype=dtype) * tf.cast(lower_mask, dtype=dtype)
    median_mask *= sorted_weights  # FIXME ???
    median_mask = median_mask/tf.maximum(tf.reduce_sum(median_mask, axis=-1, keep_dims=True), 0.0000000001)
    median = tf.reduce_sum(median_mask * sorted_numbers, axis=-1)
    return median


def get_median2(sorted_numbers, sorted_weights, alpha=0.5, dtype=tf.float32):
    is_num = tf.cast(tf.greater(sorted_weights, 0.0), dtype=dtype)
    upward = tf.cumsum(sorted_weights, axis=-1, exclusive=True)
    downward = tf.cumsum(sorted_weights, axis=-1, exclusive=True, reverse=True)
    c = tf.cumsum(is_num, axis=-1, exclusive=True, reverse=True)
    loss = (upward+downward-c*sorted_numbers)
    loss = apply_mask(loss, is_num, -np.inf)
    loss = tf.nn.softmax(-loss * sorted_weights)
    median = tf.reduce_sum(loss * sorted_numbers, axis=-1)
    return median


def get_huber_loss_fn(delta=1.37, pseudo=False):
    # Huber loss
    # abs(error)<delta -> square loss, for small errors
    # abs(error)>delta -> absolute loss, for larger values
    def huber_loss(x, y):
        errors = x-y
        losses = tf.where(tf.less_equal(tf.abs(errors), delta),
                          0.5 * tf.square(errors),
                          delta * tf.abs(errors) - 0.5 * delta ** 2,
                          )
        return losses

    def pseudo_huber_loss(x, y):
        # A smooth approximation to Huber loss
        errors = x-y
        losses = (delta ** 2) * (tf.sqrt(1.0 + tf.square(errors/delta)) - 1.0)
        return losses
    if pseudo:
        return pseudo_huber_loss
    else:
        return huber_loss


def get_biweight_loss_fn(gamma=4.6851, normalise=True):  # gamma = 2.5 (from another source)
    # Tukey's biweight (bisquare) loss
    # gamma = 4.6851 => approx. 95% asymptotic efficiency (for unit variance!)
    # error=0.0 => loss=0.0
    # abs(error)>>gamma => loss->gamma, rapid convergence to max(loss)=gamma for errors larger than gamma
    def biweight_loss(x, y):
        residuals = x-y
        # mean, variance = tf.nn.moments(residuals, axes=[0])
        median, mad = get_robust_statistics(residuals)
        variance = tf.square(1.4826*mad)  # stdev = 1.4826*MAD is a consistent estimator of the standard deviation
        residuals = tf.nn.batch_normalization(residuals, median, variance, None, None, variance_epsilon=1E-9)
        ones = tf.ones_like(residuals)
        losses = tf.where(tf.less_equal(tf.abs(residuals), gamma),
                          gamma ** 2 / 6.0 * (1.0 - tf.pow(1.0 - tf.pow(residuals / gamma, 2 * ones), 3 * ones)),
                          gamma ** 2 / 6.0 * ones)
        if normalise:
            losses /= gamma
        return losses
    return biweight_loss


def get_robust_statistics(x):
    # median and median absolute deviation (MAD)
    def _get_median(_x):
        # FIXME this is just an approximation
        n_elements = tf.unstack(tf.shape(x))[0]
        n_halfway = tf.cast(tf.floor(tf.cast(n_elements, dtype=tf.float32)/2.0), dtype=tf.int32)
        topk_values, _ = tf.nn.top_k(_x, k=n_halfway, sorted=True)
        _median = tf.slice(topk_values, [n_halfway-1], [1])
        return _median
    median = _get_median(x)
    mad = _get_median(tf.abs(x-median))
    return median, mad


def make_hard(probs, is_binary=False, dtype=tf.float32):
    if is_binary:
        probs = tf.greater_equal(probs, 0.5)
        probs = tf.cast(probs, dtype=dtype)
    else:
        probs = tf.cast(tf.equal(probs, tf.reduce_max(probs, axis=-1, keep_dims=True)), dtype=dtype)
        probs /= tf.reduce_sum(probs, axis=-1, keep_dims=True)  # in case of ties
    return probs


def get_log_of_gaussians(x,  # (batch, time, vocab)
                         mu,  # (vocab,)
                         sigma  # (vocab, )
                         ):
    mu = tf.expand_dims(tf.expand_dims(mu, 0), 0)  # (1,1,vocab)
    sigma = tf.expand_dims(tf.expand_dims(sigma, 0), 0)  # (1,1,vocab)
    kernel_values = -0.5 * tf.square((x - mu)/sigma) - tf.log(np.sqrt(2.0 * np.pi) * sigma) # (batch, time, vocab)
    return kernel_values

def get_cumulative_attributes(x,  # (batch, time)
                              mu,  # (vocab,)
                              dtype=tf.float32
                              ):
    x = tf.expand_dims(x, -1)  # (batch, time, 1)
    mu = tf.expand_dims(tf.expand_dims(mu, 0), 0)  # (1, 1, vocab)
    attributes = tf.less_equal(mu, x)
    return tf.cast(attributes, dtype=dtype)   # (batch, time, vocab)

def get_cumulative_attributes2(x,  # (batch, time)
                              mu,  # (vocab,)
                              dtype=tf.float32
                              ):
    x = tf.expand_dims(x, -1)  # (batch, time, 1)
    mu = tf.expand_dims(tf.expand_dims(mu, 0), 0)  # (1, 1, vocab)
    dist = tf.abs(mu-x)
    attributes = tf.equal(dist, tf.reduce_min(dist,axis=-1,keep_dims=True))
    return tf.cast(attributes, dtype=dtype)   # (batch, time, vocab)


def cdf_to_histogram(cdf,  #  (batch,time,codes)
                      bin_upper,  #  (codes,)  in ascending order
                      dtype=tf.float32
                    ):
    min_value = tf.reduce_min(bin_upper)-1.0
    max_value = tf.reduce_max(bin_upper)+1.0
    batch_size, step_size, code_size = tf.unstack(tf.shape(cdf))
    _prob_upper = tf.concat([cdf, tf.ones([batch_size, step_size, 1])], axis=-1)
    _prob_lower = tf.concat([tf.zeros([batch_size, step_size, 1]), cdf], axis=-1)
    histogram = _prob_upper - _prob_lower  # (batch, time, codes+1)
    _bin_upper = tf.concat([bin_upper, tf.fill([1, ], max_value)], axis=-1)
    _bin_lower = tf.concat([tf.fill([1, ], min_value), bin_upper], axis=-1)
    bin_center = (_bin_upper+_bin_lower)/2.0  # (codes+1)
    return histogram, bin_center


def cdf_to_prediction(cdf,  #  (batch,time,codes)
                      bin_upper,  #  (codes,)  in ascending order
                      ):
    histogram, bin_center = cdf_to_histogram(cdf, bin_upper)
    histogram = make_hard(histogram)  # to predict with mode
    prediction = tf.reduce_sum(histogram * bin_center, axis=-1)
    '''
    mmm = make_hard(-tf.abs(cdf - 0.5))
    numeric_prediction = tf.expand_dims(tf.expand_dims(bin_upper, 0), 0) * mmm
    prediction = tf.reduce_sum(numeric_prediction, axis=-1) / tf.maximum(1.0, tf.reduce_sum(mmm, axis=-1))
    '''
    return prediction

def get_num_prob(x,  #(batch, time)
             cdf, bin_upper, dtype=tf.float32):
    histogram, bin_center = cdf_to_histogram(cdf, bin_upper)
    x = tf.expand_dims(x, -1)
    bin_center = tf.expand_dims(tf.expand_dims(bin_center, 0), 0)
    #
    max_dist = (tf.reduce_max(tf.abs(x-bin_center))+2000.0)
    #
    dist_left = x-bin_center
    dist_left = tf.where(tf.greater_equal(dist_left, 0.0),
                         dist_left,
                         tf.ones_like(dist_left) * max_dist)
    dist_left = tf.where(tf.equal(dist_left, 0.0),
                         tf.ones_like(dist_left),
                         dist_left)
    dist_left *= make_hard(-dist_left)
    #
    dist_right = bin_center-x
    dist_right = tf.where(tf.greater_equal(dist_right, 0.0),
                          dist_right,
                          tf.ones_like(dist_right) * max_dist)
    dist_right = tf.where(tf.equal(dist_right, 0.0),
                          tf.ones_like(dist_right),
                          dist_right)
    dist_right *= make_hard(-dist_right)
    #
    dist_weight = dist_left+dist_right  # (b,t,c+1)
    dist_weight /= tf.reduce_sum(dist_weight, axis=-1, keep_dims=True)
    prob = tf.reduce_sum(histogram * dist_weight, axis=-1)
    return prob

