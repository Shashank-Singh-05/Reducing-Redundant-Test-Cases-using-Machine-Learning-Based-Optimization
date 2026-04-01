#  Redundancy Detection in Text Data Using Clustering Algorithms

A machine learning project that detects and removes redundant question pairs from the Quora Question Pairs dataset using three unsupervised clustering algorithms — **KMeans**, **DBSCAN**, and **Agglomerative Clustering**.
This project detects and removes redundant textual data using unsupervised machine learning clustering algorithms. The system processes textual questions, groups similar ones into clusters, and removes duplicates based on similarity thresholds.

The goal is to improve data quality, reduce storage, and eliminate duplicate entries in large text datasets.
---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Visualization](#visualization)
- [Known Fixes Applied](#known-fixes-applied)
- [Future Work](#future-work)

---

## Overview

Large textual datasets often contain duplicate or near-identical entries that degrade data quality, slow retrieval, and inflate storage costs. This project builds an **unsupervised pipeline** to automatically detect and remove such redundancies using clustering and cosine similarity.

The pipeline:
1. Loads and preprocesses the Quora Question Pairs dataset
2. Converts text to numerical vectors using TF-IDF
3. Clusters documents using three different algorithms
4. Removes redundant entries within each cluster using cosine similarity thresholding
5. Evaluates each algorithm on reduction rate and runtime
6. Visualizes cluster structure using 2D SVD projection

---

## Dataset

**Source:** [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs/data)

| Property | Value |
|---|---|
| File | `questions.csv` |
| Rows used | 7,000 |
| Columns used | `question1`, `question2` |
| Task | Detect semantically duplicate question pairs |

Each row contains two questions that may or may not be semantic duplicates. The two questions are concatenated into a single text field for clustering.

**Example:**
```
question1: "What is AI?"
question2: "What is Artificial Intelligence?"
→ Treated as redundant (high cosine similarity)
```

---

## Usage

Run the full pipeline with:

```bash
python code.py
```

This will:
- Load and preprocess 7,000 rows from `questions.csv`
- Run all three clustering algorithms sequentially
- Print evaluation results for each algorithm to the console
- Save three deduplicated CSV files to the project directory
- Display and save a cluster visualization plot as `cluster_visualization.png`

### Expected Console Output

```
Loaded 7000 rows
TF-IDF matrix shape: (7000, 5000)

Running KMeans...
----- KMEANS RESULTS -----
Original rows : 7000
Reduced rows  : 5199
Reduction rate: 25.73%
Runtime       : 6.26 sec

Running DBSCAN...
  DBSCAN found 42 clusters, 1203 noise points
----- DBSCAN RESULTS -----
Original rows : 7000
Reduced rows  : 4604
Reduction rate: 34.23%
Runtime       : 3.96 sec

Running Agglomerative Clustering...
----- AGGLOMERATIVE RESULTS -----
Original rows : 7000
Reduced rows  : 5287
Reduction rate: 24.47%
Runtime       : 2.59 sec
```

---

## Methodology

### 1. Text Preprocessing

Raw question text is cleaned before vectorization:

| Step | Operation | Example |
|---|---|---|
| Combine | Merge Q1 + Q2 into one field | `"What is AI? What is Artificial Intelligence?"` |
| Lowercase | Convert all text | `"what is ai? what is artificial intelligence?"` |
| Remove punctuation | Regex strip | `"what is ai what is artificial intelligence"` |
| Remove stopwords | NLTK stopword list | `"ai artificial intelligence"` |

### 2. Feature Extraction — TF-IDF

Text is converted to a sparse numerical matrix using **TF-IDF (Term Frequency–Inverse Document Frequency)**:

```
TF-IDF(t, d) = TF(t, d) × log(N / df(t))
```

- **Output shape:** `(7000, 5000)` — 7,000 documents × 5,000 vocabulary features
- Common words across all documents receive lower weights
- Rare but informative words are boosted

### 3. Clustering Algorithms

#### KMeans
- Partitions data into **K = 100** clusters
- Assigns each document to the nearest centroid
- Iterates until centroids stabilise
- `n_init = 10`, `random_state = 42`

#### DBSCAN
- Density-based — groups points in dense regions
- Automatically identifies noise points (labelled `-1`)
- **Pre-processing:** TruncatedSVD to 100 dimensions applied first (required for cosine metric on large sparse matrices)
- `eps = 0.3`, `min_samples = 5`, `metric = cosine`

#### Agglomerative Clustering
- Hierarchical bottom-up merging using **ward linkage**
- **Pre-processing:** TruncatedSVD to 100 dimensions (required — ward linkage is O(n²) on raw features)
- Merges closest clusters until `n_clusters = 150` remain

### 4. Redundancy Removal

Within each cluster, documents are deduplicated using a **greedy cosine similarity** filter:

```
threshold = 0.65
```

For each cluster:
1. Compute pairwise cosine similarity matrix
2. Start with an empty "keep" set
3. For each document, add it to the keep set only if its similarity with every already-kept document is **below** the threshold
4. Documents above the threshold are dropped as redundant

Cosine similarity range: `0` (completely different) → `1` (identical)

---

## Results

All results at `threshold = 0.65` on 7,000 rows:

| Algorithm | Original | Reduced | Removed | Reduction | Runtime |
|---|---|---|---|---|---|
| KMeans | 7,000 | 5,199 | 1,801 | 25.73% | 6.26 sec |
| DBSCAN | 7,000 | 4,604 | 2,396 | **34.23%** | 3.96 sec |
| Agglomerative | 7,000 | 5,287 | 1,713 | 24.47% | **2.59 sec** |

**Key observations:**
- **DBSCAN** achieved the highest redundancy removal (34.23%) by leveraging density-based grouping and noise detection
- **Agglomerative** was the fastest (2.59 sec) and most memory-efficient after SVD pre-reduction
- **KMeans** offered the most stable and predictable cluster structure across runs

---

## Visualization

Clusters are visualized by projecting TF-IDF vectors to 2D using **TruncatedSVD** and rendering a scatter plot for each algorithm:

```python
# 1,000 randomly sampled points
# SVD reduces 5000 dims → 2 dims
# Each point coloured by cluster ID (tab20 colormap)
```

The output image `cluster_visualization.png` contains three side-by-side scatter plots — one per algorithm. Insert this into the presentation slide 13 after running the script.

---

## Known Fixes Applied

The original codebase had several bugs that were corrected in `code2_fixed.py`:

| # | Bug | Fix |
|---|---|---|
| 1 | `n_clusters=500` — too many clusters for 7,000 rows | Reduced to `n_clusters=100` |
| 2 | DBSCAN ran on raw 5,000-dim sparse matrix (extremely slow) | Added `TruncatedSVD(100)` before DBSCAN |
| 3 | Agglomerative `sample_size=7000` hardcoded — crashes if rows < 7,000 after `dropna` | Changed to `min(7000, actual_row_count)` |
| 4 | Sparse matrix sliced without `.toarray()` in `remove_redundancy` | Added `.toarray()` conversion before `cosine_similarity` |
| 5 | `visualize_all` mutated caller's label arrays | Renamed inner variables to `lk`, `ld`, `la` |
| 6 | No `else` clause in `run_pipeline` for unknown algo names | Added `raise ValueError` for unknown algorithms |

---

## Future Work

- **Semantic embeddings:** Replace TF-IDF with BERT or `sentence-transformers` for contextual similarity rather than keyword overlap
- **Larger datasets:** Scale to the full Quora dataset (400K+ pairs) and benchmark performance
- **Supervised evaluation:** Use the provided `is_duplicate` labels to compute precision/recall against ground truth
- **Hyperparameter tuning:** Grid search over `eps`, `min_samples`, and `threshold` for optimal redundancy removal
- **Streaming pipeline:** Process data in batches to support real-time deduplication in production systems

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | TF-IDF, clustering algorithms, cosine similarity, SVD |
| `nltk` | Stopword removal |
| `matplotlib` | Cluster visualization |
| `re` | Text cleaning with regex |

---

## Author
Shashank Singh
