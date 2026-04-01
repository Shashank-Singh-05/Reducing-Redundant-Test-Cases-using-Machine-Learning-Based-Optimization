"""
Redundancy Detection on Quora Question Pairs
Algorithms compared:
1. KMeans
2. DBSCAN
3. Agglomerative Clustering
"""

import re
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import time

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


# Load Dataset
def load_dataset(path="questions.csv", max_rows=7000):

    df = pd.read_csv(path, nrows=max_rows)

    df["Text"] = df["question1"].fillna("") + " " + df["question2"].fillna("")

    df = df.dropna(subset=["Text"]).reset_index(drop=True)

    print(f"Loaded {len(df)} rows")

    return df


# Text Cleaning
def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = [t for t in text.split() if t not in STOP_WORDS]

    return " ".join(tokens)


# TF-IDF
def generate_embeddings(df):

    vectorizer = TfidfVectorizer(max_features=5000)

    matrix = vectorizer.fit_transform(df["Text"])

    print("TF-IDF matrix shape:", matrix.shape)

    return matrix


# KMeans
def cluster_kmeans(matrix):

    print("\nRunning KMeans...")

    model = KMeans(
        n_clusters=100,       # FIX: was 500, too many for 7000 rows
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(matrix)

    return labels


# DBSCAN
def cluster_dbscan(matrix):

    print("\nRunning DBSCAN...")

    svd = TruncatedSVD(n_components=100, random_state=42)
    matrix_reduced = svd.fit_transform(matrix)

    model = DBSCAN(
        eps=0.3,              
        min_samples=5,
        metric="cosine"
    )

    labels = model.fit_predict(matrix_reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = np.sum(labels == -1)
    print(f"  DBSCAN found {n_clusters} clusters, {n_noise} noise points")

    return labels


# Agglomerative
def cluster_agglomerative(matrix):

    print("\nRunning Agglomerative Clustering...")

    # Reduce dimensionality first (required for ward linkage)
    svd = TruncatedSVD(n_components=100, random_state=42)
    matrix_reduced = svd.fit_transform(matrix)

    sample_size = min(7000, matrix_reduced.shape[0])
    idx = np.random.choice(matrix_reduced.shape[0], sample_size, replace=False)

    matrix_small = matrix_reduced[idx]

    model = AgglomerativeClustering(
        n_clusters=150,
        linkage="ward"        # ward requires euclidean distance (fine after SVD)
    )

    labels_small = model.fit_predict(matrix_small)

    # Map sampled labels back to full array; unsampled rows get -1
    labels = np.full(matrix.shape[0], -1)
    labels[idx] = labels_small

    return labels


# Remove Redundancy
def remove_redundancy(df, matrix, labels, threshold=0.80):

    df = df.copy()

    df["cluster"] = labels

    keep_indices = []

    for cluster_id in np.unique(labels):

        cluster_indices = df.index[df["cluster"] == cluster_id].tolist()

        if len(cluster_indices) == 1:
            keep_indices.append(cluster_indices[0])
            continue

        cluster_matrix = matrix[cluster_indices]
        if hasattr(cluster_matrix, "toarray"):
            cluster_matrix = cluster_matrix.toarray()

        sim_matrix = cosine_similarity(cluster_matrix)

        selected = []

        for i, idx in enumerate(cluster_indices):

            if not selected:
                selected.append(i)
                keep_indices.append(idx)
                continue

            sims = [sim_matrix[i][j] for j in selected]

            if max(sims) < threshold:
                selected.append(i)
                keep_indices.append(idx)

    reduced_df = df.loc[sorted(keep_indices)].reset_index(drop=True)

    return reduced_df


# Evaluation
def evaluate(original, reduced, algo, runtime):

    reduction = (original - reduced) / original * 100

    print("\n-----", algo, "RESULTS -----")

    print("Original rows    :", original)
    print("Reduced rows     :", reduced)
    print("Reduction rate   :", f"{reduction:.2f}%")
    print("Runtime          :", f"{runtime:.2f} sec")


# Visualization
def visualize_all(matrix, labels_k, labels_d, labels_a):

    print("\nGenerating cluster visualizations...")

    n = matrix.shape[0]
    sample_size = min(1000, n)                  
    idx = np.random.choice(n, sample_size, replace=False)

    matrix_sample = matrix[idx]
    if hasattr(matrix_sample, "toarray"):
        matrix_sample = matrix_sample.toarray()  

    lk = np.array(labels_k)[idx]
    ld = np.array(labels_d)[idx]
    la = np.array(labels_a)[idx]

    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(matrix_sample)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plots = [
        ("KMeans Clusters",        lk),
        ("DBSCAN Clusters",        ld),
        ("Agglomerative Clusters", la),
    ]

    for ax, (title, labels) in zip(axes, plots):

        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            cmap="tab20",
            s=12,
            alpha=0.8
        )

        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        legend = ax.legend(
            *scatter.legend_elements(num=6),
            title="Cluster ID",
            loc="best",
            fontsize=8
        )

        ax.add_artist(legend)

    plt.tight_layout()
    plt.savefig("cluster_visualization.png", dpi=150)
    plt.show()
    print("Saved cluster_visualization.png")


# Pipeline
def run_pipeline(df, matrix, algo):

    start = time.time()

    if algo == "kmeans":
        labels = cluster_kmeans(matrix)

    elif algo == "dbscan":
        labels = cluster_dbscan(matrix)

    elif algo == "agglomerative":
        labels = cluster_agglomerative(matrix)

    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    df_reduced = remove_redundancy(df, matrix, labels)

    runtime = time.time() - start

    evaluate(len(df), len(df_reduced), algo.upper(), runtime)

    return labels, df_reduced


# Main
def main():

    df = load_dataset()

    df["Text"] = df["Text"].astype(str).apply(preprocess_text)

    matrix = generate_embeddings(df)

    labels_k, reduced_k = run_pipeline(df, matrix, "kmeans")

    labels_d, reduced_d = run_pipeline(df, matrix, "dbscan")

    labels_a, reduced_a = run_pipeline(df, matrix, "agglomerative")

    visualize_all(matrix, labels_k, labels_d, labels_a)

    reduced_k.to_csv("optimized_kmeans.csv",        index=False)
    reduced_d.to_csv("optimized_dbscan.csv",         index=False)
    reduced_a.to_csv("optimized_agglomerative.csv",  index=False)

    print("\nAll outputs saved:")
    print("  optimized_kmeans.csv")
    print("  optimized_dbscan.csv")
    print("  optimized_agglomerative.csv")
    print("  cluster_visualization.png")


if __name__ == "__main__":
    main()