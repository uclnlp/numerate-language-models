import tensorflow as tf
import numpy as np
from collections import defaultdict
from pprint import pprint
from src.models.lm import LM
from src.models.config import get_model_config
from src.models.num_coder import NumCoder
from src.models import util
import json
from src.number_cloze import reader, ploty
from src.preproc import vocabulary
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import sklearn
from sklearn import mixture, cluster
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sys, os
from time import time, strftime, gmtime
import random
from src.number_cloze import evalui
import argparse

RNG = np.random.RandomState(1214)
tf.set_random_seed(1234)
np.random.seed(5678)
random.seed(4321)

placeholders = [
    tf.placeholder(tf.int32, [None, None], name="tokens"),
    tf.placeholder(tf.int32, [None, None], name="numbers"),
    tf.placeholder(tf.int32, [None, None], name="enc_tokens"),
    tf.placeholder(tf.int32, [None, None], name="enc_numbers"),
    tf.placeholder(tf.float32, [None], name="batch_floats_value"),
    tf.placeholder(tf.int32, [None], name="batch_floats_ids"),
    tf.placeholder(tf.int32, [None], name="batch_floats_round"),
    ]
placeholders = {p.op.name: p for p in placeholders}
print(placeholders)
to_feed_dict = reader.build_to_feed_dict(placeholders)


def load_vocab(load_dir, hidden):
    input_file = os.path.join(load_dir, 'vocab.json')
    # load vocab
    with open(input_file, 'r') as fin:
        vocab = json.load(fin)
    # load pretrained matrices
    pretrained = {}
    for name in ['class', 'words', 'nums']:
        input_file = os.path.join(load_dir, 'glove_{name}_{hidden}.npy'.format(name=name, hidden=hidden))
        pretrained[name] = np.load(input_file)
    vocab['pretrained'] = pretrained
    assert(len(vocab['word2index']) == sum(vocab['partition_sizes']) == len(vocab['adjusted_counts']))
    return vocab


def get_kernels(all_nums, is_dummy=False):
    all_floats = [float(x) for x in all_nums]
    xs = [x for x in all_floats]
    X = np.array(xs)[:, None]
    locs_kernel = [np.mean(all_floats)]
    scales_kernel = [np.std(all_floats)]
    for k in range(1, 8):
        kkk = 2**k
        if not is_dummy:
            qs = np.linspace(0.0, 1.0, kkk, endpoint=True)
            ms = np.percentile(xs, q=100.0 * qs)[:, None]
            de = mixture.GaussianMixture(kkk,
                                         n_init=1,
                                         means_init=ms,
                                         covariance_type='spherical',
                                         random_state=42).fit(X)
            a = np.argsort(de.means_[:, 0])
            locs_kernel.extend(de.means_[:, 0][a].tolist())
            scales_kernel.extend(np.sqrt(de.covariances_[a]).tolist())
        else:
            locs_kernel.extend([0.0] * kkk)
            scales_kernel.extend([0.0] * kkk)
    a = np.argsort(locs_kernel)
    locs = np.array(locs_kernel)[a].tolist()
    scales = np.array(scales_kernel)[a].tolist()
    return locs, scales


def build_models(vocab,
                 which=None,
                 log=None):
    config = get_model_config(which)
    if log:
        print(config)
        with open(log, 'w', encoding='utf-8') as log:
            json.dump(config, log)
    # define kernels
    kernel_locs, kernel_scales = get_kernels(vocab['all_nums'], 'continuous' not in config['output_mode'])
    if 'continuous' in config['output_mode']:
        print('kernels:')
        print(sorted(list(zip(kernel_locs, kernel_scales)), key=lambda x: x[0]))
    # create model
    ms = []
    for idx, name_scope in enumerate(['Train', 'Test']):
        with tf.name_scope(name_scope):
            with tf.variable_scope('Model', reuse=(idx != 0)):
                numcoder_in = NumCoder(option=config['numcoders'][0],
                                       hidden_size=config['emb_dim'],
                                       name='NumCoderIn')
                numcoder_out = numcoder_in
                if len(config['numcoders']) > 1:
                    numcoder_out = NumCoder(option=config['numcoders'][1],
                                            hidden_size=config['emb_dim'],
                                            name='NumCoderOut')
                numcoders = [numcoder_in, numcoder_out]
                m = LM(placeholders,
                       hidden_size=config['emb_dim'],
                       partition_sizes=vocab['partition_sizes'],
                       iv_nums=vocab['iv_nums'],
                       all_nums=vocab['all_nums'],
                       adjust_counts=vocab['adjusted_counts'],  # ASSUMES numbers appear at the end of vocab
                       pretrained=vocab['pretrained'],
                       kernel_locs=kernel_locs,
                       kernel_scales=kernel_scales,
                       is_training=(idx == 0),
                       numcoders=numcoders,
                       output_mode=config['output_mode'],
                       use_loc_embs=config['use_loc_embs'],
                       use_hierarchical=config['use_hierarchical'],
                       )
                ms.append(m)
    model = ms[0]
    model_test = ms[1]
    return model, model_test


def train_model(model,
                get_batch_gen, n_instances=None,
                min_op=None, grad_norm=None,
                batch_loss=None, epoch=0,
                sess=None, verbose=True, log=None):
    is_training = min_op is not None and grad_norm is not None
    # initialise metrics
    progress = evalui.Progress(time(), n_instances)
    total_loss_aggr = evalui.Aggregator()
    if is_training:
        total_grad_aggr = evalui.Aggregator()
    eval_nll = {
        'tokens': evalui.Aggregator(),  # negative log-likelihood,
        'words': evalui.Aggregator(),
        'numbers': evalui.Aggregator(),
    }
    eval_nll_adj = {
        'tokens': evalui.Aggregator(),
        'words': evalui.Aggregator(),
        'numbers': evalui.Aggregator(),
    }
    for j, batch in enumerate(get_batch_gen()):
        #for k,v in batch.items(): print(k, v.shape)
        #print([w for w in np.sum(batch['numbers'], axis=-1) if w != 0.0])
        #print([w for w in np.sum(batch['is_numbers'], axis=-1) if w != 0.0])
        #print([w for w in np.sum(batch['numbers'], axis=-1)/np.maximum(np.sum(batch['is_numbers'], axis=-1), 1.0) if w != 0.0])
        #print(to_feed_dict(batch))
        fetch = {'tokens_mask': model.target_mask_is_token,  # (batch, time)
                 'batch_loss': batch_loss,
                 # counts
                 'n_tokens': model.n_tokens,
                 'n_words': model.n_words,
                 'n_numbers': model.n_numbers,
                 # PP
                 'nll_total_tokens': model.nll_total_tokens,
                 'nll_total_words': model.nll_total_words,
                 'nll_total_numbers': model.nll_total_numbers,
                 # APP
                 'nll_adj_total_tokens': model.nll_adj_total_tokens,
                 'nll_adj_total_words': model.nll_adj_total_words,
                 'nll_adj_total_numbers': model.nll_adj_total_numbers,
                 }

        if is_training:
            fetch['min_op'] = min_op
            fetch['grad_norm'] = grad_norm

        fetch = sess.run(fetch, feed_dict=to_feed_dict(batch))

        if is_training:
            total_grad_aggr.aggregate(fetch['grad_norm'], 1.0)
            if not np.isfinite(fetch['batch_loss']):
                print()
                print(fetch['batch_loss'])
                raise(Exception('Encountered NaN/INF cost'))
        else:
            if not np.isfinite(fetch['nll_total_tokens']).sum():
                print('...Encountered NaN/INF cost...')
        total_loss_aggr.aggregate(fetch['batch_loss'], 1.0)
        # update metrics
        for c, e in eval_nll.items():
            e.aggregate(fetch['nll_total_%s' % c], fetch['n_%s' % c])
        for c, e in eval_nll_adj.items():
            e.aggregate(fetch['nll_adj_total_%s' % c], fetch['n_%s' % c])
        # Monitor execution speed
        progress.update(np.sum(fetch['tokens_mask']), fetch['tokens_mask'].shape[0])
        epoch_rate, epoch_remaining = progress.result(time())
        print('\r', end='')  # clear line
        line = ['Epoch %d -' % epoch,
                'TRAIN' if is_training else 'DEV',
                'speed=%.2f tokens/sec' % epoch_rate,
                'remaining=%.2f secs' % epoch_remaining]
        print(' '.join(line), end='')
        sys.stdout.flush()  # force print progress summary
    # Monitor metrics
    cost = total_loss_aggr.result()
    grad_norm = total_grad_aggr.result() if is_training else None
    line = [
        'Epoch %d -' % epoch,
        'TRAIN' if is_training else 'DEV',
        # progress
        'secs=%.2f' % (time()-progress.start_time),
        'grad_norm=%.8f' % grad_norm if is_training else '',
        'cost=%.8f' % cost,
        # APP
        'app_tokens=%.3f' % np.exp(eval_nll_adj['tokens'].result()),
        'app_words=%.3f' % np.exp(eval_nll_adj['words'].result()),
        'app_nums=%.3f' % np.exp(eval_nll_adj['numbers'].result()),
        # PP
        'pp_tokens=%.3f' % np.exp(eval_nll['tokens'].result()),
        'pp_words=%.3f' % np.exp(eval_nll['words'].result()),
        'pp_nums=%.3f' % np.exp(eval_nll['numbers'].result()),
       ]
    if verbose:
        print('\r', end='')  # clear line
        print(' '.join(line))
    if log:
        log = open(log, 'a+', encoding='utf-8')
        log.write('\t'.join(line))
        log.write('\n')
        log.close()
    return cost, grad_norm


def test_model(model,
               get_batch_gen, n_instances=None,
               sess=None, verbose=True, log=None,
               baseline=None,
               #baseline='mean',
               #baseline='median',
               ):
    mean_float = np.mean(model.all_floats)
    median_float = np.median(model.all_floats)
    # initialise metrics
    progress = evalui.Progress(time(), n_instances)
    eval_nll = {
        'tokens': evalui.Aggregator(),  # negative log-likelihood,
        'words': evalui.Aggregator(),
        'numbers': evalui.Aggregator(),
    }
    eval_nll_adj = {
        'tokens': evalui.Aggregator(),
        'words': evalui.Aggregator(),
        'numbers': evalui.Aggregator(),
    }
    numeric_eval_regression = evalui.Numeric()
    for j, batch in enumerate(get_batch_gen()):
        fetch = {'tokens_mask': model.target_mask_is_token,  # (batch, time)
                 #'numeric_reg': model.numeric_reg,
                 # counts
                 'n_tokens': model.n_tokens,
                 'n_words': model.n_words,
                 'n_numbers': model.n_numbers,
                 # PP
                 'nll_total_tokens': model.nll_total_tokens,
                 'nll_total_words': model.nll_total_words,
                 'nll_total_numbers': model.nll_total_numbers,
                 # APP
                 'nll_adj_total_tokens': model.nll_adj_total_tokens,
                 'nll_adj_total_words': model.nll_adj_total_words,
                 'nll_adj_total_numbers': model.nll_adj_total_numbers,
                 #
                 'target_is_oov_num': model.target_mask_is_unk_num,  # B
                 'target_num': model.target_values,  # B
                 'pred_num': model.pred_values,  # B
                 'target_round': model.target_round,  # B
                 }
        fetch = sess.run(fetch, feed_dict=to_feed_dict(batch))
        if not np.isfinite(fetch['nll_total_tokens']).sum():
            print('...Encountered NaN/INF cost...')
        if baseline == 'mean':
            fetch['pred_num'] = 0.0 * fetch['pred_num'] + mean_float
        elif baseline == 'median':
            fetch['pred_num'] = 0.0 * fetch['pred_num'] + median_float
        # update metrics
        for c, e in eval_nll.items():
            e.aggregate(fetch['nll_total_%s' % c], fetch['n_%s' % c])
        for c, e in eval_nll_adj.items():
            e.aggregate(fetch['nll_adj_total_%s' % c], fetch['n_%s' % c])
        #num_reg_all.aggregate(fetch['numeric_reg'])
        numeric_eval_regression.aggregate(preds=fetch['pred_num'].tolist(),
                                          golds=fetch['target_num'].tolist(),
                                          is_iv=(1 - fetch['target_is_oov_num']).tolist())
        # Monitor execution speed
        progress.update(np.sum(fetch['tokens_mask']), fetch['tokens_mask'].shape[0])
        epoch_rate, epoch_remaining = progress.result(time())
        print('\r', end='')  # clear line
        line = ['TEST',
                'speed=%.2f tokens/sec' % epoch_rate,
                'remaining=%.2f secs' % epoch_remaining]
        print(' '.join(line), end='')
        sys.stdout.flush()  # force print progress summary
    # Monitor metrics
    line = [
        'TEST',
        'secs=%.2f' % (time()-progress.start_time),
        # APP
        'app_tokens=%.3f' % np.exp(eval_nll_adj['tokens'].result()),
        'app_words=%.3f' % np.exp(eval_nll_adj['words'].result()),
        'app_nums=%.3f' % np.exp(eval_nll_adj['numbers'].result()),
        # PP
        'pp_tokens=%.3f' % np.exp(eval_nll['tokens'].result()),
        'pp_words=%.3f' % np.exp(eval_nll['words'].result()),
        'pp_nums=%.3f' % np.exp(eval_nll['numbers'].result()),
        #
        #'num_reg=%.2f' % num_reg_all.result(),
        # pred (mode)
        *['%s=%.3f' % (k, v) for k, v in numeric_eval_regression.result().items()],
       ]
    if verbose:
        print('\r', end='')  # clear line
        print(' '.join(line))
    if log:
        log = open(log, 'a+', encoding='utf-8')
        log.write('\t'.join(line))
        log.write('\n')
        log.close()


def dev_model(model, get_dev_batch_gen, n_instances=None,
              sess=None, online=False):
    do_benford = True
    do_regression = True
    do_modes = True
    # initialise metrics
    progress = evalui.Progress(time(), n_instances)
    if do_benford:
        all_prob_digits, all_target_digits = [], []
    if do_regression:
        numeric_eval_regression = evalui.Numeric()
    if do_modes:
        numeric_eval_modes = evalui.Modes()
    for j, batch in enumerate(get_dev_batch_gen()):
        fetch = {
            'tokens_mask': model.target_mask_is_token,  # (batch, time)
            'target_is_oov_num': model.target_mask_is_unk_num,  # B
            'target_num': model.target_values,  # B
            'target_round': model.target_round,  # B
            'candidate_values': model.cand_values,
            'candidate_rounds': model.cand_rounds,
        }
        if do_modes:
            #fetch['pred_mode'] = model.pred_mode  # B3
            fetch['pred_mode'] = model.prob_mode  # B3
        if do_regression:
            fetch['probs_cands'] = model.probs_cands
            fetch['pred_num'] = model.pred_values  # B
            fetch['pred_mode'] = model.prob_mode  # B3
        if do_benford:
            fetch['prob_digits'] = model.prob_digits
            fetch['target_digits'] = model.target_digits
        fetch = sess.run(fetch, feed_dict=to_feed_dict(batch))

        if do_benford:
            nnn = fetch['target_digits'].shape[0]
            all_target_digits.extend([fetch['target_digits'][iii, :].tolist() for iii in range(nnn)])  # l
            all_prob_digits.extend([fetch['prob_digits'][iii, :, :] for iii in range(nnn)])  # lc
        if do_modes:
            numeric_eval_modes.aggregate(fetch['pred_mode'].tolist(),
                                         fetch['target_num'].tolist(),
                                         fetch['target_round'].tolist())
        if do_regression:
            numeric_eval_regression.aggregate(preds=fetch['pred_num'].tolist(),
                                              golds=fetch['target_num'].tolist(),
                                              is_iv=(1 - fetch['target_is_oov_num']).tolist())

        # Monitor execution speed
        progress.update(np.sum(fetch['tokens_mask']), fetch['tokens_mask'].shape[0])
        epoch_rate, epoch_remaining = progress.result(time())
        print('\r', end='')  # clear line
        line = ['DEV',
                'speed=%.2f tokens/sec' % epoch_rate,
                'remaining=%.2f secs' % epoch_remaining]
        print(' '.join(line), end='')
        sys.stdout.flush()  # force print progress summary


        markers = ['o', '^', 'd', '*', 'o', 'o', 'o', 'o', 'o']
        if online:
            targets = fetch['target_num']
            preds = fetch['pred_num']
            modes = fetch['pred_mode']
            x = fetch['candidate_values']  # n
            a = np.argsort(x)
            xs = x[a]
            rs = fetch['candidate_rounds'][a]
            ys = fetch['probs_cands'][:, a]  # Bn
            #ys = np.log(ys)
            for iii in range(ys.shape[0]):
                _y = ys[iii, :]
                for cur_r in sorted(np.unique(rs).tolist()):
                    x = []
                    y = []
                    for jjj, _ in enumerate(xs):
                        if rs[jjj] == cur_r:
                            x.append(xs[jjj])
                            y.append(_y[jjj])
                    # plt.plot(x, y, 'o', x, y, '-', label='r=%d' % cur_r)
                    #plt.plot(x, y, '-', label='r=%d' % cur_r, marker=markers[cur_r], markerfacecolor='None')
                    plt.semilogy(x, y, '-', label='r=%d' % cur_r, marker=markers[cur_r], markerfacecolor='None')
                print(modes[iii])
                plt.title('target=%.2f pred=%.2f' % (targets[iii], preds[iii]))
                plt.axis([np.min(xs), 200, 0, max(_y)])
                #plt.axis([np.min(xs), 2020, min(_y), max(_y)])
                plt.legend()
                plt.show()

                #plt.setp(baseline, 'color', 'r', 'linewidth', 2)

    print('\r', end='')  # clear line
    print('DEV\tsecs=%.2f' % (time()-progress.start_time))
    # MOST CATEGORICAL
    if do_modes:
        numeric_eval_modes.print_result()
    # BENFORD
    if do_benford:
        nnn = max(len(x) for x in all_target_digits)
        digit_probs_in_pos = [np.zeros([10]) for _ in range(nnn)]
        digit_freqs_in_pos = [np.zeros([10]) for _ in range(nnn)]
        logits_in_pos = np.zeros([nnn])
        counts_in_pos = np.zeros([nnn])
        for probs, digits in zip(all_prob_digits, all_target_digits):  # loop over numbers
            # exclude special chars and renormalise
            probs = probs[:, :10]
            probs = probs/np.sum(probs, axis=-1, keepdims=True)
            significant_pos = None
            for pos, d in enumerate(digits):  # loop over digits
                if d >= 13:
                    # stop after EOS
                    break
                if 0 <= d <= 9:
                    if significant_pos is None:
                        if 0 < d <= 9:
                            # first non-zero is first most significant digit
                            significant_pos = 0
                    if significant_pos is not None:
                        # this is a significant digit
                        ps = probs[pos, :]
                        if significant_pos == 0:
                            # leftmost significant is non-zero => renormalise
                            ps[0] = 0.0
                            ps = ps/np.sum(ps)
                        logits_in_pos[significant_pos] += -np.log(ps[d])
                        counts_in_pos[significant_pos] += 1.0
                        digit_freqs_in_pos[significant_pos][d] += 1
                        digit_probs_in_pos[significant_pos] += ps
                        significant_pos += 1
        ppl_in_pos = np.exp(logits_in_pos/np.maximum(counts_in_pos, 1.0))
        ppl_total = np.exp(np.sum(logits_in_pos)/np.sum(counts_in_pos))
        print('PP (per significant digit):', ppl_total)
        print('Breakdown per significant position:')
        print(list(enumerate(ppl_in_pos.tolist())))
        digit_freqs_in_pos = [b/np.sum(b) for b in digit_freqs_in_pos]
        digit_probs_in_pos = [b/np.sum(b) for b in digit_probs_in_pos]
        benford = [
            [0.00000, 0.30103, 0.17609, 0.12494, 0.09691, 0.07918, 0.06695, 0.05799, 0.05115, 0.04576],
            [0.11968, 0.11389, 0.10882, 0.10433, 0.10031, 0.09668, 0.09337, 0.09035, 0.08757, 0.08500],
            [0.10178, 0.10138, 0.10097, 0.10057, 0.10018, 0.09979, 0.09940, 0.09902, 0.09864, 0.09827],
            [0.10018, 0.10014, 0.10010, 0.10006, 0.10002, 0.09998, 0.09994, 0.09990, 0.09986, 0.09982],
            [0.10002, 0.10001, 0.10001, 0.10001, 0.10000, 0.10000, 0.09999, 0.09999, 0.09999, 0.09998],
        ]
        for pos, (freqs, probs) in enumerate(zip(digit_freqs_in_pos, digit_probs_in_pos)):
            print('Significant Position:', pos)
            ben = benford[pos] if pos < len(benford) else [0.0] * 10
            if np.isnan(np.sum(freqs)):
                continue
            print('d\tModel\tData\tBenford')
            for d in range(10):
                print('%d\t%.2f\t%.2f\t%.2f' % (d, probs[d]*100.0, freqs[d]*100.0, ben[d]*100.0))
    # REGRESSION ACTUAL VS PREDS
    if do_regression:
        #print(sorted(zip(numeric_eval_regression.golds, numeric_eval_regression.preds), key=lambda x: (abs(x[0]-x[1]), x[0])))
        ploty.plot_preds(numeric_eval_regression.preds, numeric_eval_regression.golds, 'pred', 'target')


def inspect_model(model, sess=None):

    fetch = {
        'w_emb_digits_in': model.numcoders[0].w_emb_digits,
        'w_emb_digits_out': model.numcoders[1].w_emb_digits,
        'w_digits': model.w_digits,
        'w_soft_digits': model.w_soft_digits,
        'w_soft_nums': model.w_soft_nums,  # hc
        'w_soft_kernels': model.w_soft_kernels,  # hc
        'kernel_locs': model.kernel_locs,
        'kernel_scales': model.kernel_scales,
        'input_embs_nums': model.input_embs_nums,
        'gate': model.g,
        'reps_out': model.reps_out,
        'reps_in': model.reps_in,
    }
    fetch = sess.run(fetch)

    if False:
        # INPUT GATE
        w = fetch['gate'][:-1, :]
        exact_labels = [('%.'+str(r)+'f') % x for x, r in zip(model.exact_locs.tolist(), model.exact_rounds.tolist())][:-1]
        plt.imshow(w, vmin=0.0, vmax=1.0, cmap='seismic')
        plt.yticks(np.arange(len(exact_labels)), exact_labels)
        plt.colorbar()
        plt.title('input gates')
        plt.show()
        #pprint(list(zip(exact_labels, np.mean(fetch['g'][:-1, :], -1).tolist())))

    if True:
        # DIGIT EMBEDDINGS
        digit_labels = [str(x) for x in range(10)] + ['.', 'EOS']
        w = fetch['w_emb_digits_in'][[0,1,2,3,4,5,6,7,8,9,11,12], :]
        #ploty.plot_sims(w, np.arange(w.shape[0]), digit_labels, 'input digit embeddings')
        w = fetch['w_emb_digits_out'][[0,1,2,3,4,5,6,7,8,9,11,12], :]
        ploty.plot_sims(w, np.arange(w.shape[0]), digit_labels, 'output digit embeddings')
        w = fetch['w_digits'][[0,1,2,3,4,5,6,7,8,9,11,12], :]
        ploty.plot_sims(w, np.arange(w.shape[0]), digit_labels, 'RNN input digit embeddings')
        w = fetch['w_soft_digits'].T[[0,1,2,3,4,5,6,7,8,9,11,12], :]
        ploty.plot_sims(w, np.arange(w.shape[0]), digit_labels, 'RNN output digit embeddings')

    if True:
        # OUTPUT EMBEDDINGS
        w = fetch['w_soft_nums'].T[:-1, :]
        exact_labels = [('%.'+str(r)+'f') % x for x, r in zip(model.exact_locs.tolist(), model.exact_rounds.tolist())][:-1]
        ploty.plot_pca(w, exact_labels, 'output numerals (exact) - PCA')
        ploty.plot_tsne(w, exact_labels, 'output numerals (exact)')
        ploty.plot_sims(w, np.arange(len(exact_labels)), [x if i % 3 == 0 else '' for i,x in enumerate(exact_labels)], 'output numerals (exact)')
        w = fetch['reps_out'][:-1, :]
        ploty.plot_pca(w, exact_labels, 'output numerals (RNN) - PCA')
        ploty.plot_tsne(w, exact_labels, 'output numerals (RNN)')
        ploty.plot_sims(w, np.arange(len(exact_labels)), exact_labels, 'output numerals (RNN)')
        # KERNEL EMBEDDINGS
        w = fetch['w_soft_kernels'].T
        kernel_labels = ['(%.2f, %.2f)' % (m, s) for m, s in zip(fetch['kernel_locs'].tolist(), fetch['kernel_scales'].tolist())]
        ploty.plot_pca(w, kernel_labels, 'kernels - PCA')
        ploty.plot_tsne(w, kernel_labels, 'kernels')
        ploty.plot_sims(w, np.arange(len(kernel_labels)), kernel_labels, 'output kernels')

    if True:
        # INPUT EMBEDDINGS
        exact_labels = [('%.'+str(r)+'f') % x for x, r in zip(model.exact_locs.tolist(), model.exact_rounds.tolist())][:-1]
        w = fetch['input_embs_nums'][:-1, :]
        exact_labels = [('%.'+str(r)+'f') % x for x, r in zip(model.exact_locs.tolist(), model.exact_rounds.tolist())][:-1]
        ploty.plot_tsne(w, exact_labels, 'input numerals (exact)')
        ploty.plot_sims(w, np.arange(len(exact_labels)), exact_labels, 'input numerals (exact)')
        w = fetch['reps_in'][:-1, :]
        ploty.plot_pca(w, exact_labels, 'input numerals (RNN) - PCA')
        ploty.plot_tsne(w, exact_labels, 'input numerals (RNN)')
        ploty.plot_sims(w, np.arange(len(exact_labels)), exact_labels, 'input numerals (RNN)')


def load_model_params(load_model_dir, sess, saver):
    if os.path.isdir(load_model_dir):
        # load_model_dir = tf.train.latest_checkpoint(load_model_dir) # if file not specified, use latest checkpoint
        load_model_dir = os.path.join(load_model_dir, 'best')  # if file not specified, use best checkpoint
    print('Loading variables from: ', load_model_dir)
    try:
        saver.restore(sess, load_model_dir)
    except tf.errors.NotFoundError:
        # only load numcoder
        for n in ['NumCoderIn', 'NumCoderOut']:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Model/%s' % n)
            var_dict = {var.op.name.replace(n, 'NumCoder'): var for var in var_list}
            _saver = tf.train.Saver(var_dict)
            _saver.restore(sess, load_model_dir)


def main():
    # --data clinical --train 500 --config a1
    # --data clinical --load a2_2018_02_14_21_02_40_clinical
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', help='dataset', type=str, choices=['clinical', 'arxmliv'], default="clinical")
    parser.add_argument('--train', help='max training epochs', type=int, default=0)
    parser.add_argument('--no-test', help='skip testing', dest='do_test', action='store_false', default=True)
    parser.add_argument('--no-inspect', help='skip inspecting', dest='do_inspect', action='store_false', default=True)
    parser.add_argument('--batch', help='batch size', type=int, default=128)
    parser.add_argument('--load', help='load model', type=str, default='')
    parser.add_argument('--config', help='config', type=str, default='')
    args = parser.parse_args()
    print(args)
    do_inspect = args.do_inspect
    do_test = args.do_test
    batch_size = args.batch
    max_epochs = args.train
    dataset_name = args.data
    root_dir = './experiments/%s' % dataset_name
    load_model_dir = ''
    if not args.load and not args.config:
        print('Error: stored model or model config needed')
        return -1
    if args.load:
        load_model_dir = args.load
        which = args.load.split('_')[0]
    if args.config:
        which = args.config
    ####
    timestamp = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    print('Root dir:', root_dir)
    print('Timestamp:', timestamp)
    print('Model config:', which)
    # Model load/save dirs
    if load_model_dir:
        load_model_dir = os.path.join(root_dir, 'checkpoints', load_model_dir)
    save_model_dir = os.path.join(root_dir,
                                  'checkpoints',
                                  '{config}_{timestamp}_{dataset}'.format(config=which,
                                                                          timestamp=timestamp,
                                                                          dataset=dataset_name))
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    print('Will load model from:', load_model_dir)
    print('Will save model to:  ', save_model_dir)

    print('Loading dataset...')
    get_instance_gen = reader.load_data(path_to_data=os.path.join(root_dir, 'data/{fold}_bucketed.strings_to_floats'),
                                        )#max_lines={'test': 10})
                                        #max_lines={'train': 100})
                                        #max_lines={'train': 50, 'test': 100, 'dev':50})
    n_instances = {}
    for fold, _ in get_instance_gen.items():
        print('--', fold, '--')
        n_instances[fold] = reader.analyse_dataset(get_instance_gen[fold])

    print('Loading vocabulary...')
    vocab = load_vocab(load_dir=os.path.join(root_dir, 'vocab'), hidden=get_model_config(which)['emb_dim'])

    counts = np.zeros(len(vocab['word2index']), dtype=np.float32)
    n_unks = 0
    n_tokens = 0
    for instance in get_instance_gen['train']():
        for tok in instance['tokens']:
            n_tokens += 1
            if tok == '<UNK>' or tok.startswith('<UNK_NUM'):
                n_unks += 1
            counts[vocab['word2index'][tok]] += 1
    print('#unseen in train:', (counts == 0).sum())
    print('#unks:', n_unks)
    print('#tokens:', n_tokens)
    print(float(n_unks)/n_tokens*100.0)

    print('Setting up batcher...')
    def prepare_batch(_batch):
        return reader.prepare_batch(_batch,
                                    lambda w: vocab['word2index'].get(w,
                                                                      vocab['word2index']['<UNK_NUM>']
                                                                      if vocabulary.to_float(w) is not None else
                                                                      vocab['word2index']['<UNK>']
                                                                      ))
    get_batch_gen = {
        'train': reader.get_batch_gen_builder(get_instance_gen['train'], batch_size=batch_size, prepare=prepare_batch),
        'dev': reader.get_batch_gen_builder(get_instance_gen['dev'], batch_size=batch_size, prepare=prepare_batch),
        'test': reader.get_batch_gen_builder(get_instance_gen['dev'], batch_size=50, prepare=prepare_batch),
    }

    print('Building model...')
    model, model_test = build_models(which=which,
                                     vocab=vocab,
                                     log=os.path.join(save_model_dir, 'config.txt'))
    saver = tf.train.Saver(max_to_keep=1000)
    util.print_variable_report()

    min_op, loss = None, None
    if max_epochs > 0:
        print('Building optimiser...')
        print('===============')
        min_op, loss, grad_norm = util.build_optimiser(model.cost_total,
                                                       tf.train.AdamOptimizer(learning_rate=0.001),
                                                       l2=0.0)  # TODO regularisation
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True  # Do not take up all the GPU memory, all the time
    with tf.Session(config=sess_config) as sess:
        print('Initialising...')
        print('===============')
        sess.run(tf.global_variables_initializer())
        if load_model_dir:
            load_model_params(load_model_dir, sess, saver)
        max_patience = 5
        if max_epochs > 0:
            print('Training...')
            print('============')
            converged = False
            prev_train_cost = np.inf
            best_dev_cost = np.inf
            bad_epochs = 0
            for epoch in range(1, max_epochs + 1):
                # TRAIN STEP
                train_cost, cur_grad_norm = train_model(model,
                                                        get_batch_gen['train'], n_instances=n_instances['train'],
                                                        min_op=min_op, grad_norm=grad_norm,
                                                        batch_loss=loss, epoch=epoch,
                                                        sess=sess,
                                                        log=os.path.join(save_model_dir, 'log_train.txt'))
                if cur_grad_norm < 1e-8 or abs(prev_train_cost - train_cost) < 1e-8:
                    converged = True
                # SAVE MODEL
                print('Saving model to:', save_model_dir, end='')
                sys.stdout.flush()
                saver.save(sess, os.path.join(save_model_dir, 'model'), global_step=epoch)
                # DEV STEP
                dev_cost, _ = train_model(model_test,
                                          get_batch_gen['dev'], n_instances=n_instances['dev'],
                                          batch_loss=loss, epoch=epoch, sess=sess,
                                          verbose=False,
                                          log=os.path.join(save_model_dir, 'log_dev.txt'))
                if dev_cost < best_dev_cost:
                    # new best model found
                    best_dev_cost = dev_cost
                    bad_epochs = 0
                    print(' Saving best model to:', save_model_dir, end='')
                    sys.stdout.flush()
                    saver.save(sess, os.path.join(save_model_dir, 'best'))
                else:
                    bad_epochs += 1
                    print(' BAD EPOCHS:', bad_epochs, 'dev cost:', dev_cost, 'best dev cost:', best_dev_cost)
                if bad_epochs >= max_patience:
                    print()
                    print(' Early stopping!')
                    break
                elif converged:
                    print()
                    print(' Converged!')
                    break
            print()
            load_model_params(os.path.join(save_model_dir, 'best'), sess, saver)

        if do_test:
            print('Testing...')
            print('============')
            test_model(model_test, get_batch_gen['test'],
                       n_instances=n_instances['test'],
                       sess=sess, log=os.path.join(save_model_dir, 'log_test.txt'))

        print('Inspecting...')
        print('==============')
        if do_inspect:
            inspect_model(model_test, sess=sess)
        print('Inspecting on DEV...')
        print('==============')
        #dev_model(model_test, get_batch_gen['dev'], n_instances=n_instances['dev'], sess=sess)
        dev_model(model_test, get_batch_gen['dev'], n_instances=n_instances['dev'], sess=sess, online=True)


if __name__ == "__main__":
    main()
    print('Done!')
