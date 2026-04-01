*Redundancy Detection in Text Data Using Clustering Algorithms*

This project detects and removes **redundant textual data** using unsupervised machine learning clustering algorithms. The system processes textual questions, groups similar ones into clusters, and removes duplicates based on similarity thresholds.

The goal is to improve **data quality, reduce storage, and eliminate duplicate entries** in large text datasets.

**Project Overview**

Large datasets such as question forums or review platforms often contain **duplicate or highly similar entries**. This project applies **Natural Language Processing (NLP)** and **clustering algorithms** to detect such redundancy automatically.

The pipeline performs:

- Text preprocessing
- TF-IDF vectorization
- Clustering using multiple algorithms
- Cosine similarity comparison
- Redundant question removal
- Cluster visualization

**Dataset**

The project uses a subset of the **Quora Question Dataset**.

Dataset properties:

| **Property**          | **Value**                                 |
| --------------------- | ----------------------------------------- |
| Rows used             | 7000                                      |
| Features after TF-IDF | 5000                                      |
| Input columns         | Question1, Question2                      |
| Output                | Reduced dataset without redundant entries |

Example dataset entry:

| **Question1** | **Question2**                    |
| ------------- | -------------------------------- |
| What is AI?   | What is Artificial Intelligence? |

These questions may represent **semantic duplicates**.

**Libraries Used**

The project uses the following Python libraries:

| **Library**  | **Purpose**                   |
| ------------ | ----------------------------- |
| pandas       | Data loading and manipulation |
| numpy        | Numerical operations          |
| scikit-learn | Machine learning algorithms   |
| matplotlib   | Visualization                 |
| nltk         | Text preprocessing            |
| re           | Text cleaning                 |

Key sklearn modules used:

- TfidfVectorizer
- KMeans
- DBSCAN
- AgglomerativeClustering
- cosine_similarity
- TruncatedSVD

**System Workflow**

The overall system pipeline is:

Dataset

↓

Text Cleaning

↓

TF-IDF Vectorization

↓

Clustering Algorithms

↓

Cosine Similarity Calculation

↓

Redundancy Detection

↓

Visualization & Results

**Algorithms Implemented**

**1\. KMeans Clustering**

KMeans is a **partition-based clustering algorithm** that divides data into K clusters based on distance to cluster centroids.

Features:

- Fast and scalable
- Works well with large datasets
- Requires predefined cluster number

Example applications:

- Document clustering
- Customer segmentation

**2\. DBSCAN**

DBSCAN is a **density-based clustering algorithm**.

Clusters are formed based on dense regions of data points.

Features:

- Detects clusters of arbitrary shape
- Identifies noise points
- Does not require predefined cluster number

Applications:

- Anomaly detection
- Geospatial clustering

**3\. Agglomerative Clustering**

Agglomerative clustering is a **hierarchical clustering algorithm** that builds clusters in a bottom-up approach.

Process:

- Start with each point as its own cluster
- Merge closest clusters
- Continue until cluster criteria is met

Features:

- Hierarchical cluster structure
- Good for relationship analysis
- Computationally intensive for large datasets

**Similarity Measurement**

Redundancy detection uses **Cosine Similarity**.

Cosine similarity measures similarity between TF-IDF vectors.

Range:

0 → completely different

1 → identical

A **threshold value** determines redundancy.

Example:

| **Threshold** | **Meaning**         |
| ------------- | ------------------- |
| 0.85          | Strict similarity   |
| 0.70          | Moderate similarity |
| 0.50          | Loose similarity    |

Lower threshold values detect more redundancy.

**Experimental Results**

Results for **Threshold = 0.50**

| **Algorithm** | **Original Rows** | **Reduced Rows** | **Reduction Rate** | **Runtime** |
| ------------- | ----------------- | ---------------- | ------------------ | ----------- |
| KMeans        | 7000              | 5199             | 25.73%             | 6.26 sec    |
| DBSCAN        | 7000              | 4604             | 34.23%             | 3.96 sec    |
| Agglomerative | 7000              | 5287             | 24.47%             | 2.59 sec    |

Observations:

- **DBSCAN achieved the highest redundancy removal**
- **Agglomerative clustering had the fastest runtime**
- KMeans produced consistent cluster grouping.

**Visualization**

Clusters were visualized using **Truncated SVD dimensionality reduction**.

High-dimensional TF-IDF vectors were projected into **2D space** to visualize cluster formation.

Each point represents a question, and colors represent cluster assignments.

Example visualization:

cluster_visualization.png

**Output Files**

The program generates the following files:

| **File**                    | **Description**                                |
| --------------------------- | ---------------------------------------------- |
| optimized_kmeans.csv        | Reduced dataset using KMeans                   |
| optimized_dbscan.csv        | Reduced dataset using DBSCAN                   |
| optimized_agglomerative.csv | Reduced dataset using Agglomerative clustering |
| cluster_visualization.png   | Cluster visualization plot                     |

**How to Run**

**1\. Install Dependencies**

pip install pandas numpy scikit-learn matplotlib nltk

**2\. Run the Script**

python code.py

**3\. Output**

The script will:

- Perform clustering
- Remove redundant questions
- Generate CSV files
- Produce cluster visualization

**Key Findings**

- Clustering algorithms can effectively detect redundant textual entries.
- DBSCAN performs best for redundancy detection.
- TF-IDF combined with cosine similarity provides strong textual similarity measurement.
- Visualization helps interpret cluster formation.

**Future Improvements**

Possible improvements for this project:

- Use **BERT embeddings** for semantic similarity
- Apply **larger datasets**
- Use **advanced similarity metrics**
- Implement **real-time duplicate detection systems**

**Author**

Shashank Singh
