from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import umap.plot
import umap


def get_pca(inp_data, inp_labels, mean=False, title=""):
    pca = PCA(n_components=2)
    if mean:
        components = pca.fit_transform(np.mean(inp_data[:, :, :], axis=2))
    else:
        components = pca.fit_transform(inp_data)
    for label in np.unique(inp_labels):
        idx = inp_labels == label
        plt.scatter(components[idx, 0], components[idx, 1], label=str(label))
    plt.title("PCA clusters from the embeddings of " + title)
    plt.legend()
    plt.show()


def get_tsne(inp_data, inp_labels, mean=False, title=""):
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    if mean:
        tsne_res = tsne.fit_transform(np.mean(inp_data[:, :, :], axis=2))
    else:
        tsne_res = tsne.fit_transform(inp_data)
    plt.title("TSNE clusters from the embeddings of " + title)
    for label in np.unique(inp_labels):
        idx = inp_labels == label
        plt.scatter(tsne_res[idx, 0], tsne_res[idx, 1], label=str(label))
    # plt.scatter(tsne_res[:, 0], tsne_res[:, 1], c=inp_labels)
    plt.legend()
    plt.show()


def get_umap(inp_data, inp_labels, mean=False, title=""):
    if mean:
        mapper = umap.UMAP().fit(np.mean(inp_data, axis=1))
    else:
        mapper = umap.UMAP().fit(inp_data, axis=1)
    umap.plot.points(mapper, labels=np.array(inp_labels))
    plt.title("UMAP clusters from the embeddings of " + title)
    plt.show()


def generate_plots(input, labels, mean=False, title=""):
    get_pca(input, labels, title=title, mean=mean)
    get_tsne(input, labels, title=title, mean=mean)
    get_umap(input, labels, title=title, mean=mean)
