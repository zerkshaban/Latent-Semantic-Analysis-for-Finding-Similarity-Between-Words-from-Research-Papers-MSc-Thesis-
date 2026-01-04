# Latent Semantic Analysis for Cross-Domain Research Paper Similarity

## Project Overview

This repository hosts the source code and implementation for the MSc Advanced Computer Science dissertation titled **"Latent Semantic Analysis for Finding Similarity Between Words from Research Papers on the Same Topic from Different Domains"**. The project addresses a critical challenge in Information Retrieval (IR) and Natural Language Processing (NLP): identifying semantic connections between academic texts that share conceptual ground but originate from distinct disciplines. Standard keyword-based search methods often fail in these scenarios due to vocabulary mismatch and semantic ambiguity. This solution leverages **Latent Semantic Analysis (LSA)** to uncover hidden conceptual structures within unstructured text data, enabling a more nuanced and accurate similarity assessment.

## Methodology & Implementation

The core pipeline is built using **Python** and relies on a rigorous mathematical framework to process text data. The workflow consists of several critical stages:

1. **Data Preprocessing:** Raw text from research papers undergoes extensive cleaning to ensure data quality. This includes tokenization to break down text streams, normalization to standard formats, and the removal of stop-words and noise using the **NLTK (Natural Language Toolkit)** library.
2. **Feature Extraction (TF-IDF):** To convert textual data into a machine-readable numerical format, the system employs **Term Frequency-Inverse Document Frequency (TF-IDF)**. This statistical measure evaluates the importance of words within a document relative to the entire corpus, filtering out common terms while highlighting domain-specific vocabulary.
3. **Dimensionality Reduction (SVD):** The high-dimensional TF-IDF matrix is processed using **Singular Value Decomposition (SVD)**. This step is the heart of LSA, as it reduces the dimensionality of the dataset to capture "latent" semantic relationships between terms and documents, effectively filtering out noise and synonymy issues.
4. **Similarity Calculation:** Finally, **Cosine Similarity** measures are applied to the reduced vector space to quantify the degree of similarity between documents. This metric provides a robust score indicating how closely related two research papers are, regardless of their specific terminology.

## Technologies

The project is implemented using a robust stack of Python libraries essential for data science and NLP:

* **Scikit-learn:** For implementing TF-IDF vectorization and SVD algorithms.
* **NLTK:** For advanced text processing and linguistic data handling.
* **NumPy & Pandas:** For efficient numerical computation and data manipulation.

## Academic Context

This research was conducted at the **University of Hertfordshire** under the supervision of Harpreet Singh and Dr. Na Helian. The work received a **Distinction**, recognized for its technical complexity and its successful application of advanced mathematical models to solve real-world semantic ambiguity problems in academic literature.
