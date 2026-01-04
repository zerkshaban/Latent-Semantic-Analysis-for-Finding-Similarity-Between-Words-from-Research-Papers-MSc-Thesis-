"""
Latent Semantic Analysis (LSA) for Finding Similarity Between Words from Research Papers

This script performs Latent Semantic Analysis on scientific and social science documents to:
1. Preprocess and clean text documents
2. Apply LSA using Truncated SVD for dimensionality reduction
3. Extract top words based on vector space analysis
4. Calculate cosine similarity between top words
5. Visualize semantic relationships using network graphs
6. Generate word clouds for topic visualization

Created on Sun Jul 18 12:58:24 2021
@author: systems
"""

import pandas as pd
import networkx as nx
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of documents to process for each category
NUMBER_OF_SCIENCE_FILES = 20
NUMBER_OF_SOCIAL_SCIENCE_FILES = 20

# Number of top words to extract for similarity analysis
TOP_N_WORDS = 5

# Number of top words per topic for word cloud generation
TOP_WORDS_PER_TOPIC = 30

# ============================================================================
# CUSTOM CLASSES
# ============================================================================

class CustomDictionary(dict):
    """
    Custom dictionary class that extends Python's dict with an add method.
    
    This class provides a convenient way to add key-value pairs and is used
    for storing word vectors in the LSA analysis.
    """
    
    def __init__(self):
        """Initialize an empty dictionary."""
        self = dict()
    
    def add(self, key, value):
        """
        Add a key-value pair to the dictionary.
        
        Args:
            key: The key to store
            value: The value to associate with the key
        """
        self[key] = value


# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize text processing tools
stemmer = SnowballStemmer("english")
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = stopwords.words('english')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_documents(df, text_column='documents'):
    """
    Preprocess text documents through multiple cleaning and normalization steps.
    
    Processing steps:
    1. Remove special characters (keep only alphabets and #)
    2. Remove words with less than 3 characters
    3. Convert to lowercase
    4. Tokenize, remove stop words
    5. Apply stemming
    6. Apply lemmatization
    7. Rejoin tokens into cleaned text
    
    Args:
        df: DataFrame containing the documents
        text_column: Name of the column containing text to preprocess
        
    Returns:
        Series of preprocessed document strings
    """
    # Step 1: Remove special characters (keep only alphabets and #)
    df['clean_documents'] = df[text_column].str.replace("[^a-zA-Z#]", " ", regex=True)
    
    # Step 2: Remove words with less than 3 characters
    df['clean_documents'] = df['clean_documents'].fillna("").apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 2])
    )
    
    # Step 3: Convert to lowercase
    df['clean_documents'] = df['clean_documents'].fillna("").apply(lambda x: x.lower())
    
    # Step 4: Tokenize and remove stop words
    tokenized_doc = df['clean_documents'].fillna("").apply(lambda x: x.split())
    tokenized_doc = tokenized_doc.apply(
        lambda x: [item for item in x if item not in stop_words]
    )
    
    # Step 5: Apply stemming (reduce words to their root form)
    tokenized_doc = tokenized_doc.apply(lambda x: [stemmer.stem(y) for y in x])
    
    # Step 6: Apply lemmatization (convert words to their base/dictionary form)
    tokenized_doc = tokenized_doc.apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    
    # Step 7: Detokenize (rejoin tokens into text strings)
    detokenized_doc = [
        ' '.join(tokenized_doc[i]) 
        for i in range(len(df))
    ]
    
    return pd.Series(detokenized_doc)


def load_documents(file_prefix, num_files, skip_first_n_words=4):
    """
    Load and process text files, skipping the first N words from each file.
    
    Args:
        file_prefix: Prefix of the file names (e.g., 'science' or 'social-science')
        num_files: Number of files to load
        skip_first_n_words: Number of words to skip from the beginning of each file
        
    Returns:
        List of processed document strings
    """
    document_list = []
    
    for i in range(num_files):
        file_name = f'{file_prefix}-{str(i+1)}.txt'
        
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Split text into words and skip the first N words
            words = text.split()
            processed_words = words[skip_first_n_words:]  # Skip first N words
            document_detail = ' '.join(processed_words)
            document_list.append(document_detail)
            
        except FileNotFoundError:
            print(f"Warning: File '{file_name}' not found. Skipping...")
            document_list.append("")  # Add empty string to maintain list length
    
    return document_list


def perform_lsa_and_extract_top_words(df, document_names, num_components, top_n=5):
    """
    Perform Latent Semantic Analysis (LSA) and extract top words.
    
    This function:
    1. Creates a bag-of-words representation using CountVectorizer
    2. Applies Truncated SVD for dimensionality reduction (LSA)
    3. Calculates the magnitude (L2 norm) of each word vector across all components
    4. Extracts the top N words based on vector magnitude
    
    Args:
        df: DataFrame with preprocessed documents
        document_names: List of names/labels for the documents
        num_components: Number of components for SVD (typically equals number of documents)
        top_n: Number of top words to extract
        
    Returns:
        tuple: (top_words_dataframe, encoding_matrix, vocabulary)
            - top_words_dataframe: DataFrame with top words and their component values
            - encoding_matrix: Matrix of word encodings in the latent space
            - vocabulary: List of all words in the vocabulary
    """
    # Create bag-of-words representation
    vectorizer = CountVectorizer(stop_words='english')
    bag_of_words = vectorizer.fit_transform(df['clean_documents'])
    vocabulary = vectorizer.get_feature_names_out()
    
    # Perform Truncated SVD (LSA)
    # n_components: number of dimensions in the latent space
    # algorithm='randomized': faster algorithm for large matrices
    # n_iter: number of iterations for randomized algorithm
    # random_state: seed for reproducibility
    svd_model = TruncatedSVD(
        n_components=num_components,
        algorithm='randomized',
        n_iter=100,
        random_state=122
    )
    lsa_transformed = svd_model.fit_transform(bag_of_words)
    
    # Create encoding matrix: words (rows) x components (columns)
    # This matrix represents each word as a vector in the latent semantic space
    encoding_matrix = pd.DataFrame(
        svd_model.components_.T,  # Transpose to get words as rows
        index=vocabulary,
        columns=document_names
    )
    
    # Calculate the magnitude (L2 norm) of each word vector
    # This measures the "importance" of each word across all semantic dimensions
    vector_magnitudes = []
    for word in encoding_matrix.index:
        # Calculate sum of squares across all components for this word
        sum_of_squares = sum(
            np.square(encoding_matrix.loc[word, doc_name])
            for doc_name in document_names
        )
        # Take square root to get L2 norm (magnitude)
        magnitude = np.sqrt(sum_of_squares)
        vector_magnitudes.append(magnitude)
    
    # Create DataFrame with word magnitudes and sort by magnitude
    magnitude_df = pd.DataFrame({
        'word': vocabulary,
        'magnitude': vector_magnitudes
    })
    sorted_magnitude_df = magnitude_df.sort_values('magnitude', ascending=False)
    
    # Extract top N words
    top_words_df = sorted_magnitude_df.head(n=top_n)
    top_words_list = top_words_df['word'].tolist()
    
    # Extract component values for top words
    top_words_vectors = CustomDictionary()
    for word in top_words_list:
        # Get the vector representation of this word across all components
        word_vector = [
            encoding_matrix.loc[word, doc_name]
            for doc_name in document_names
        ]
        top_words_vectors.add(word, word_vector)
    
    # Create DataFrame with top words and their component values
    top_words_dataframe = pd.DataFrame(top_words_vectors).T
    
    # Set column names as PC1, PC2, PC3, etc. (Principal Components)
    pc_column_names = [f'PC{i+1}' for i in range(num_components)]
    top_words_dataframe.columns = pc_column_names
    
    return top_words_dataframe, encoding_matrix, vocabulary, svd_model


def calculate_cosine_similarity_matrix(top_words_df):
    """
    Calculate cosine similarity between all pairs of top words.
    
    Cosine similarity measures the cosine of the angle between two vectors,
    indicating how similar two words are in the semantic space.
    Values range from -1 to 1, where 1 indicates identical direction.
    
    Args:
        top_words_df: DataFrame with words as index and component values as columns
        
    Returns:
        List of dictionaries containing word pairs and their similarity scores
    """
    word_list = top_words_df.index.tolist()
    similarity_combinations = []
    
    # Calculate similarity for each pair of words
    for word1, word2 in combinations(word_list, 2):
        vector1 = top_words_df.loc[word1].values
        vector2 = top_words_df.loc[word2].values
        
        # Calculate cosine similarity (returns array of shape (1,1))
        similarity_score = cosine_similarity([vector1], [vector2])[0][0]
        
        # Store the word pair and similarity score
        similarity_combinations.append({
            'word_pair': [word1, word2],
            'similarity': similarity_score
        })
    
    return similarity_combinations


def create_semantic_graph(word_list, similarity_combinations, graph_title="Semantic Graph"):
    """
    Create and visualize a semantic graph showing relationships between words.
    
    Nodes represent words, edges represent similarity relationships.
    Edge thickness and labels indicate the strength of similarity.
    
    Args:
        word_list: List of words to include as nodes
        similarity_combinations: List of dictionaries with word pairs and similarities
        graph_title: Title for the graph
    """
    # Create a multigraph (allows multiple edges between nodes)
    graph = nx.MultiGraph()
    
    # Add nodes (words)
    for word in word_list:
        graph.add_node(word, name=word)
    
    # Add edges with weights (similarity scores)
    for combination in similarity_combinations:
        word1, word2 = combination['word_pair']
        similarity = combination['similarity']
        graph.add_edge(word1, word2, weight=similarity)
    
    # Set up graph layout (circular arrangement)
    pos = nx.circular_layout(graph)
    
    # Scale weights for visualization (thicker edges = higher similarity)
    edge_weights = [comb['similarity'] * 5 for comb in similarity_combinations]
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph, pos,
        edge_color='black',
        width=edge_weights,
        linewidths=1,
        node_size=1500,
        node_color='lightblue',
        alpha=0.9,
        with_labels=True,
        font_size=10,
        font_weight='bold'
    )
    
    # Add edge labels (similarity scores)
    edge_labels = {}
    for combination in similarity_combinations:
        word1, word2 = combination['word_pair']
        similarity = combination['similarity']
        edge_labels[(word1, word2)] = f"{similarity:.2f}"
    
    nx.draw_networkx_edge_labels(
        graph, pos,
        edge_labels=edge_labels,
        font_color='red',
        font_size=8
    )
    
    plt.title(graph_title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def generate_wordclouds(svd_model, vocabulary, num_topics, file_prefix, top_words_per_topic=30):
    """
    Generate word clouds for each topic/component from the LSA model.
    
    Args:
        svd_model: Fitted TruncatedSVD model
        vocabulary: List of words in the vocabulary
        num_topics: Number of topics/components to generate word clouds for
        file_prefix: Prefix for output file names
        top_words_per_topic: Number of top words to include per topic
    """
    topics_list = []
    
    # Extract top words for each component/topic
    for component in svd_model.components_:
        # Zip words with their component values and sort by value (descending)
        word_scores = list(zip(vocabulary, component))
        sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_words_per_topic]
        
        # Create a string of words for word cloud generation
        topic_string = ' '.join([word for word, score in sorted_words])
        topics_list.append(topic_string)
    
    # Try to load custom mask image, otherwise use default shape
    try:
        custom_mask = np.array(Image.open('cloud.png'))
        wordcloud = WordCloud(
            background_color='white',
            stopwords=stop_words,
            mask=custom_mask,
            contour_width=3,
            contour_color='black'
        )
    except FileNotFoundError:
        print("Note: 'cloud.png' not found, using default wordcloud shape.")
        wordcloud = WordCloud(
            background_color='white',
            stopwords=stop_words,
            width=800,
            height=400
        )
    
    # Generate word cloud for each topic
    for i, topic_string in enumerate(topics_list, start=1):
        wordcloud.generate(topic_string)
        output_filename = f'{file_prefix}-topic-{i}.png'
        wordcloud.to_file(output_filename)
        print(f"Generated word cloud: {output_filename}")


# ============================================================================
# MAIN PROCESSING: SCIENCE DOCUMENTS
# ============================================================================

print("=" * 70)
print("PROCESSING SCIENCE DOCUMENTS")
print("=" * 70)

# --- PART 1: Load and preprocess science documents ---
print("\nStep 1: Loading science documents...")
science_documents = load_documents('science', NUMBER_OF_SCIENCE_FILES)
df_science = pd.DataFrame(science_documents, columns=['documents'])

print("Step 2: Preprocessing science documents...")
df_science['clean_documents'] = preprocess_documents(df_science)
science_document_names = [f"Science Documents PC {i+1}" for i in range(NUMBER_OF_SCIENCE_FILES)]

# --- PART 2: Perform LSA and extract top words ---
print("Step 3: Performing LSA on science documents...")
top_words_science_df, science_encoding_matrix, science_vocab, science_svd_model = perform_lsa_and_extract_top_words(
    df_science,
    science_document_names,
    NUMBER_OF_SCIENCE_FILES,
    top_n=TOP_N_WORDS
)

print(f"Top {TOP_N_WORDS} science words extracted: {list(top_words_science_df.index)}")

# --- PART 3: Calculate cosine similarity and create graph ---
print("Step 4: Calculating cosine similarity between top science words...")
science_similarities = calculate_cosine_similarity_matrix(top_words_science_df)

print("Step 5: Creating semantic graph for science words...")
create_semantic_graph(
    list(top_words_science_df.index),
    science_similarities,
    graph_title="Semantic Similarity Graph - Science Documents"
)


# ============================================================================
# MAIN PROCESSING: SOCIAL SCIENCE DOCUMENTS
# ============================================================================

print("\n" + "=" * 70)
print("PROCESSING SOCIAL SCIENCE DOCUMENTS")
print("=" * 70)

# --- PART 4: Load and preprocess social science documents ---
print("\nStep 1: Loading social science documents...")
social_science_documents = load_documents('social-science', NUMBER_OF_SOCIAL_SCIENCE_FILES)
df_social_science = pd.DataFrame(social_science_documents, columns=['documents'])

print("Step 2: Preprocessing social science documents...")
df_social_science['clean_documents'] = preprocess_documents(df_social_science)
social_science_document_names = [
    f"Social Science Documents PC {i+1}" 
    for i in range(NUMBER_OF_SOCIAL_SCIENCE_FILES)
]

# --- PART 5: Perform LSA and extract top words ---
print("Step 3: Performing LSA on social science documents...")
top_words_social_science_df, social_science_encoding_matrix, social_science_vocab, social_science_svd_model = perform_lsa_and_extract_top_words(
    df_social_science,
    social_science_document_names,
    NUMBER_OF_SOCIAL_SCIENCE_FILES,
    top_n=TOP_N_WORDS
)

print(f"Top {TOP_N_WORDS} social science words extracted: {list(top_words_social_science_df.index)}")

# --- PART 6: Calculate cosine similarity and create graph ---
print("Step 4: Calculating cosine similarity between top social science words...")
social_science_similarities = calculate_cosine_similarity_matrix(top_words_social_science_df)

print("Step 5: Creating semantic graph for social science words...")
create_semantic_graph(
    list(top_words_social_science_df.index),
    social_science_similarities,
    graph_title="Semantic Similarity Graph - Social Science Documents"
)


# ============================================================================
# WORD CLOUD GENERATION
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING WORD CLOUDS")
print("=" * 70)

print("\nGenerating word clouds for science topics...")
generate_wordclouds(
    science_svd_model,
    science_vocab,
    NUMBER_OF_SCIENCE_FILES,
    'science',
    top_words_per_topic=TOP_WORDS_PER_TOPIC
)

print("\nGenerating word clouds for social science topics...")
generate_wordclouds(
    social_science_svd_model,
    social_science_vocab,
    NUMBER_OF_SOCIAL_SCIENCE_FILES,
    'social-science',
    top_words_per_topic=TOP_WORDS_PER_TOPIC
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
