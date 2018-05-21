import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import matplotlib.pyplot as plt


def plot_pca(W,  # (n_samples, n_features)
             vocab,
             title=None
             ):
    X_tsne = PCA(n_components=2).fit_transform(W)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    for row_id, target_word in enumerate(vocab):
        x = X_tsne[row_id, 0]
        y = X_tsne[row_id, 1]
        plt.annotate(str(target_word), (x, y))
    if title:
        plt.title(title)
    plt.show()


def plot_tsne(W,  # (n_samples, n_features)
              vocab,
              title=None
              ):
    X_tsne = TSNE(n_components=2).fit_transform(W)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    for row_id, target_word in enumerate(vocab):
        x = X_tsne[row_id, 0]
        y = X_tsne[row_id, 1]
        plt.annotate(str(target_word), (x, y))
    if title:
        plt.title(title)
    plt.show()


def plot_sims(W,  # (n_samples, n_features)
              points,
              labels,
              title=None
              ):
    im = plt.imshow(cos_sim(W), vmin=-1.0, vmax=1.0, cmap='seismic')#'hot')
    im.axes.xaxis.tick_top()
    plt.colorbar()
    plt.xticks(points, labels, rotation='vertical', verticalalignment='bottom')
    plt.yticks(points, labels)
    if title:
        plt.xlabel(title)
    plt.show()


def plot_preds(pred_nums_all, target_nums_all, xlabel, ylabel):
    amin = min(np.min(target_nums_all), np.min(pred_nums_all))
    amax = max(np.max(target_nums_all), np.max(pred_nums_all))
    plt.scatter(pred_nums_all, target_nums_all)
    plt.plot([amin, amax], [amin, amax], 'r--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
