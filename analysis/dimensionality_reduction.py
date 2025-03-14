import argparse
import json
import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd


class DimReduction:
    def __init__(self, pred_file, ref_file):
        with open(pred_file, "r") as f:
            self.predictions = json.load(f)

        with open(ref_file, "r") as f:
            self.reference = json.load(f)

        self.embedding_type = pred_file.split("_")[-1].split(".")[0]

        self.data = self.process_data()

        self.pred_tsne = None
        self.ref_tsne = None
        self.get_tsne_embeddings()

    def process_data(self):
        pred_df = pd.DataFrame(self.predictions)
        ref_df = pd.DataFrame(self.reference)

        merge_df = pd.merge(left=pred_df, right=ref_df, on="id", suffixes=("_pred", "_ref"))
        merge_df[f"{self.embedding_type}_pred"] = [np.array(i) for i in merge_df[f"{self.embedding_type}_pred"]]
        merge_df[f"{self.embedding_type}_ref"] = [np.array(i) for i in merge_df[f"{self.embedding_type}_ref"]]

        return merge_df

    def get_tsne_embeddings(self):
        tsne = TSNE(n_components=2, random_state=1, perplexity=50)
        word_vectors = tsne.fit_transform(np.stack(np.concatenate([self.data[f"{self.embedding_type}_pred"], self.data[f"{self.embedding_type}_ref"]])))

        pred_word_vectors = word_vectors[:len(self.predictions)]
        ref_word_vectors = word_vectors[len(self.predictions):]

        self.pred_tsne = pred_word_vectors
        self.ref_tsne = ref_word_vectors

    def plot_embeddings(self, save_path, num_samples=None):
        if num_samples:
            idx = np.random.choice(self.data.shape[0], size=num_samples, replace=False)
        else:
            idx = np.arange(self.data.shape[0])

        plt.figure(figsize=(12, 8))

        for i in idx:
            plt.text(self.pred_tsne[i, 0] + 0.1, self.pred_tsne[i, 1] + 0.1, self.data["word"].iloc[i], fontsize=9)
            plt.text(self.ref_tsne[i, 0] + 0.1, self.ref_tsne[i, 1] + 0.1, self.data["word"].iloc[i], fontsize=9)

            plt.plot(np.stack([self.pred_tsne[i, 0], self.ref_tsne[i, 0]]),
                     np.stack([self.pred_tsne[i, 1], self.ref_tsne[i, 1]]),
                     linestyle='--', color='black')

        plt.scatter(self.pred_tsne[idx, 0], self.pred_tsne[idx, 1], c="r")
        plt.scatter(self.ref_tsne[idx, 0], self.ref_tsne[idx, 1], c="b")

        plt.title("2D Visualization of Word Embeddings using t-SNE")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        # plt.show()

        plt.savefig(os.path.join(save_path, "tsne_plot.png"))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--pred_file", help="Revdict prediction file path")
    argparser.add_argument("--ref_file", help="Revdict reference file path")

    argparser.add_argument("--num_samples", help="Number of random words to plot. If None, all words are plotted.",
                           default=None, type=int)

    argparser.add_argument("--save_path", help="Directory to save plots")

    args = argparser.parse_args()

    dim_reduction = DimReduction(args.pred_file, args.ref_file)
    dim_reduction.plot_embeddings(args.save_path, args.num_samples)